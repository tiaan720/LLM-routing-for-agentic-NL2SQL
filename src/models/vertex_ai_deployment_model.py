import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import vertexai
from google.cloud import aiplatform
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

try:
    from langchain_core.messages.tool import ToolCall
except ImportError:
    # Fallback if ToolCall is not available in this version
    from pydantic import BaseModel

    class ToolCall(BaseModel):
        name: str
        args: dict
        id: str
        type: str = "tool_call"


from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from src.models.base import ModelRegistry


class CustomVertexAIEndpoint(BaseChatModel):
    """Custom LLM wrapper for deployed Vertex AI endpoints.

    This wrapper works around the issue where LangChain's VertexAIModelGarden
    doesn't properly support dedicated endpoints that require custom domains.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying model's documentation or API.

    Example:
        .. code-block:: python

            model = CustomVertexAIEndpoint(
                project="my-project-id",
                endpoint_id="1234567890",
                location="us-central1"
            )
            result = model.invoke("Hello, how are you?")
    """

    project: str = Field(
        description="The GCP project ID where the endpoint is deployed"
    )
    endpoint_id: str = Field(description="The Vertex AI endpoint ID")
    location: str = Field(
        default="us-central1",
        description="The GCP region where the endpoint is deployed",
    )
    model: Optional[str] = Field(
        default=None,
        description="The model name deployed at the endpoint (informational only)",
    )

    # Model configuration
    max_tokens: Optional[int] = Field(
        default=256, description="Maximum number of tokens to generate (default: 256)"
    )
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="Stop sequences for generation"
    )
    hourly_cost: Optional[float] = Field(
        default=4.6028,
        description="Hourly cost in USD for the endpoint (default: $4.6028)",
    )

    # Private fields
    endpoint: Any = Field(
        default=None, exclude=True, description="The Vertex AI endpoint instance"
    )
    executor: ThreadPoolExecutor = Field(
        default=None, exclude=True, description="Thread pool for async operations"
    )
    bound_tools: List[Any] = Field(
        default=[], exclude=True, description="Tools bound to this model"
    )
    model_name: str = Field(
        default=None, exclude=True, description="Model name for identification"
    )

    def __init__(
        self,
        project: str,
        endpoint_id: str,
        location: str = "us-central1",
        model: Optional[str] = None,
        hourly_cost: Optional[float] = 4.6028,
        max_tokens: Optional[int] = 256,
        **kwargs,
    ):
        super().__init__(
            project=project,
            endpoint_id=endpoint_id,
            location=location,
            model=model,
            hourly_cost=hourly_cost,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Initialize Vertex AI and get the endpoint
        vertexai.init(project=project, location=location)
        self.endpoint = aiplatform.Endpoint(endpoint_id)
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Set model_name for identification
        self.model_name = model or f"vertex-ai-endpoint-{endpoint_id}"

    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from the generated text."""
        tool_calls = []

        # Get available tool names safely
        def get_tool_names():
            tool_names = []
            if self.bound_tools:
                for tool in self.bound_tools:
                    try:
                        if isinstance(tool, dict):
                            # Handle dict-based tool definitions
                            if "name" in tool:
                                tool_names.append(tool["name"])
                        elif hasattr(tool, "name"):
                            tool_names.append(tool.name)
                        elif hasattr(tool, "__name__"):
                            tool_names.append(tool.__name__)
                        elif hasattr(tool, "_name"):
                            tool_names.append(tool._name)
                    except Exception:
                        # Skip tools that can't be processed
                        continue
            return tool_names

        available_tools = get_tool_names()

        # Look for function call patterns in the response
        # Pattern for function calls like: postgres_execute_sql({"query": "SELECT ..."})
        function_pattern = r"(\w+)\s*\(\s*(\{.*?\})\s*\)"
        matches = re.findall(function_pattern, text, re.DOTALL)

        for i, (function_name, args_str) in enumerate(matches):
            if function_name in available_tools:
                try:
                    # Parse the arguments
                    args = json.loads(args_str)
                    tool_call = ToolCall(
                        name=function_name,
                        args=args,
                        id=f"call_{i}_{function_name}",
                        type="tool_call",
                    )
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    # If JSON parsing fails, skip this call
                    continue

        # Alternative pattern: Action: tool_name \n Action Input: {"param": "value"}
        action_pattern = r"Action:\s*(\w+)\s*\n\s*Action Input:\s*(\{.*?\})"
        action_matches = re.findall(action_pattern, text, re.DOTALL | re.MULTILINE)

        for i, (function_name, args_str) in enumerate(action_matches):
            if function_name in available_tools:
                try:
                    args = json.loads(args_str)
                    tool_call = ToolCall(
                        name=function_name,
                        args=args,
                        id=f"action_call_{len(tool_calls) + i}_{function_name}",
                        type="tool_call",
                    )
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue

        return tool_calls

    def _extract_tool_call_content(self, text: str) -> str:
        """Extract the content that should be shown, excluding tool call details."""
        # Remove Action/Action Input patterns
        text = re.sub(
            r"Action:\s*\w+\s*\n\s*Action Input:\s*\{.*?\}",
            "",
            text,
            flags=re.DOTALL | re.MULTILINE,
        )

        # Remove function call patterns
        text = re.sub(r"\w+\s*\(\s*\{.*?\}\s*\)", "", text, flags=re.DOTALL)

        # Keep only the reasoning/thought parts
        lines = text.split("\n")
        content_lines = []
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("Observation:")
                and not line.startswith("Final Answer:")
            ):
                if "Thought:" in line:
                    content_lines.append(line.replace("Thought:", "").strip())
                elif not any(
                    keyword in line.lower()
                    for keyword in ["action", "observation", "tool"]
                ):
                    content_lines.append(line)

        return "\n".join(content_lines).strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate responses for the given messages."""
        start_time = time.time()

        # Convert messages to prompt format
        prompt_parts = []
        for message in messages:
            if hasattr(message, "type"):
                role = message.type.capitalize()
            else:
                role = message.__class__.__name__.replace("Message", "")
            prompt_parts.append(f"{role}: {message.content}")

        # Add tool information to the prompt if tools are bound
        if self.bound_tools:
            try:
                tool_descriptions = []
                for tool in self.bound_tools:
                    try:
                        if isinstance(tool, dict):
                            if "name" in tool and "description" in tool:
                                tool_descriptions.append(
                                    f"- {tool['name']}: {tool['description']}"
                                )
                        elif hasattr(tool, "name") and hasattr(tool, "description"):
                            tool_descriptions.append(
                                f"- {tool.name}: {tool.description}"
                            )
                    except Exception as e:
                        print(f"Warning: Failed to process tool: {e}")
                        continue

                if tool_descriptions:
                    prompt_parts.append(f"\nYou have access to these tools:")
                    prompt_parts.extend(tool_descriptions)
                    prompt_parts.append(
                        '\nWhen you need to use a tool, call it using the format: tool_name({"param": "value"})'
                    )
                    prompt_parts.append(
                        "Only make ONE tool call at a time. After a tool call, wait for the result before proceeding."
                    )
            except Exception as e:
                print(f"Warning: Failed to process bound tools: {e}")
                # Continue without tool descriptions

        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        # Merge stop sequences
        combined_stop = []
        if self.stop_sequences:
            combined_stop.extend(self.stop_sequences)
        if stop:
            combined_stop.extend(stop)

        # Format the instance for the endpoint (specific to this Qwen model deployment)
        instance = {"prompt": prompt}

        # Add generation parameters within the instance
        if self.max_tokens:
            instance["max_tokens"] = self.max_tokens
        else:
            instance["max_tokens"] = 512

        if self.temperature is not None:
            instance["temperature"] = self.temperature
        else:
            instance["temperature"] = 0.1

        # Add other commonly used parameters for better generation
        instance["top_p"] = 0.95
        instance["top_k"] = 20

        instances = [instance]

        try:

            def _make_prediction():
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return self.endpoint.predict(instances=instances)

            response = _make_prediction()

            # Extract the generated text from the response
            if response.predictions and len(response.predictions) > 0:
                generated_text = response.predictions[0]

                # Apply stop sequences if any
                if combined_stop:
                    for stop_seq in combined_stop:
                        if stop_seq in generated_text:
                            generated_text = generated_text[
                                : generated_text.index(stop_seq)
                            ]
                            break

                end_time = time.time()
                response_time_seconds = end_time - start_time

                # Check if this contains tool calls
                tool_calls = []
                try:
                    tool_calls = (
                        self._parse_tool_calls(generated_text)
                        if self.bound_tools
                        else []
                    )
                except Exception as e:
                    # If tool parsing fails, log the error but continue without tool calls
                    print(f"Warning: Tool parsing failed: {e}")
                    tool_calls = []

                if tool_calls:
                    # If tool calls are found, extract just the reasoning content
                    try:
                        content = self._extract_tool_call_content(generated_text)
                    except Exception as e:
                        print(f"Warning: Content extraction failed: {e}")
                        content = generated_text
                else:
                    # If no tool calls, use the full generated text
                    content = generated_text

                ai_message = AIMessage(
                    content=content,
                    tool_calls=tool_calls,
                    response_metadata={
                        "model_name": self.model_name,
                        "response_time_seconds": response_time_seconds,
                        "hourly_cost": self.hourly_cost,
                        "endpoint_id": self.endpoint_id,
                        "is_vertex_model_garden": True,
                    },
                )

                # Set additional_kwargs for function_call compatibility if there's exactly one tool call
                try:
                    if len(tool_calls) == 1:
                        ai_message.additional_kwargs = {
                            "function_call": {
                                "name": tool_calls[0].name,
                                "arguments": json.dumps(tool_calls[0].args),
                            }
                        }
                except Exception as e:
                    print(f"Warning: Failed to set additional_kwargs: {e}")
                    # Continue without additional_kwargs

                generation = ChatGeneration(message=ai_message)
                return ChatResult(generations=[generation])
            else:
                end_time = time.time()
                response_time_seconds = end_time - start_time

                ai_message = AIMessage(
                    content="",
                    response_metadata={
                        "model_name": self.model_name,
                        "response_time_seconds": response_time_seconds,
                        "hourly_cost": self.hourly_cost,
                        "endpoint_id": self.endpoint_id,
                        "is_vertex_model_garden": True,
                    },
                )
                generation = ChatGeneration(message=ai_message)
                return ChatResult(generations=[generation])

        except Exception as e:
            # Handle errors gracefully
            end_time = time.time()
            response_time_seconds = end_time - start_time

            ai_message = AIMessage(
                content=f"Error: {str(e)}",
                response_metadata={
                    "model_name": self.model_name,
                    "response_time_seconds": response_time_seconds,
                    "hourly_cost": self.hourly_cost,
                    "endpoint_id": self.endpoint_id,
                    "is_vertex_model_garden": True,
                },
            )
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom_vertex_ai_endpoint"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes to make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model or f"vertex-ai-endpoint-{self.endpoint_id}",
            "project": self.project,
            "location": self.location,
            "endpoint_id": self.endpoint_id,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable:
        """Bind tools to the model.

        Creates a new instance with tools that will be used to generate
        tool calling prompts for models that don't natively support tool calling.
        """
        # Create a new instance with the same parameters but with tools bound
        new_instance = self.__class__(
            project=self.project,
            endpoint_id=self.endpoint_id,
            location=self.location,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_sequences=self.stop_sequences,
            hourly_cost=self.hourly_cost,
        )

        # Store the tools for prompt generation
        new_instance.bound_tools = list(tools)
        # Ensure model_name is preserved
        new_instance.model_name = self.model_name

        return new_instance


@ModelRegistry.register("vertex_model_garden")
def create_vertex_ai_model_garden(**model_kwargs) -> Runnable:
    """Create a Vertex AI Model Garden LLM instance.

    This function creates a custom wrapper that properly handles dedicated endpoints.
    """
    # Extract the parameters for our custom class
    project = model_kwargs.get("project")
    endpoint_id = model_kwargs.get("endpoint_id")
    location = model_kwargs.get("location", "us-central1")
    model = model_kwargs.get("model")
    hourly_cost = model_kwargs.get("hourly_cost", 4.6028)
    max_tokens = model_kwargs.get("max_tokens", 256)

    return CustomVertexAIEndpoint(
        project=project,
        endpoint_id=endpoint_id,
        location=location,
        model=model,
        hourly_cost=hourly_cost,
        max_tokens=max_tokens,
    )


from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    model_kwargs = {
        "project": "970641581678",
        "endpoint_id": "4549617487527804928",
        "location": "us-central1",
        "model": "qwen/qwen3-32B",
        "max_tokens": 256,
    }
    model = create_vertex_ai_model_garden(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)

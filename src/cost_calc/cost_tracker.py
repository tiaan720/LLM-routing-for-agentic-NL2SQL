import json
import logging
from difflib import get_close_matches
from typing import Any, Dict, Generator, List, Optional, Tuple

from litellm import model_cost

from src.utils.logger import logger


def _fuzzy_match_model_key(
    target_key: str, available_keys: List[str], prefix_filter: Optional[str] = None
) -> Optional[str]:
    """
    Find the best fuzzy match for a model key among available keys.
    Uses difflib to find close matches with a similarity threshold.
    Optionally restricts candidates to those starting with prefix_filter.
    """
    # First try exact match
    if target_key in available_keys:
        return target_key

    candidates = available_keys
    if prefix_filter:
        candidates = [k for k in available_keys if k.startswith(prefix_filter)]
        if not candidates:
            candidates = available_keys  # fallback if no candidates with prefix

    # Try with lower threshold for vertex models (0.6) since they often have version suffixes
    if prefix_filter and "vertex_ai/" in prefix_filter:
        matches = get_close_matches(target_key, candidates, n=1, cutoff=0.6)
        if matches:
            logger.info(
                f"Fuzzy matched '{target_key}' to '{matches[0]}' (lower threshold)"
            )
            return matches[0]

    # Try fuzzy matching with high similarity threshold (0.8)
    matches = get_close_matches(target_key, candidates, n=1, cutoff=0.8)
    if matches:
        logger.info(f"Fuzzy matched '{target_key}' to '{matches[0]}'")
        return matches[0]

    return None


def get_model_costs(
    model_name: str, provider: Optional[str] = None
) -> Optional[Tuple[float, float]]:
    """Get input and output costs per token for a specific model."""
    if not model_name:
        logger.warning("No model name provided for cost lookup")
        return None

    # Ollama models are local and have zero cost
    if model_name.startswith("ollama/"):
        logger.info(f"Ollama model detected: {model_name} - returning zero cost")
        return (0.0, 0.0)

    # Try direct lookup first (for non-vertex models like OpenAI)
    if model_name in model_cost:
        model_info = model_cost[model_name]
        return (
            model_info.get("input_cost_per_token"),
            model_info.get("output_cost_per_token"),
        )

    available_keys = list(model_cost.keys())

    # Try fuzzy matching for direct lookup
    # For vertex models, restrict candidates to vertex_ai/ keys only
    is_vertex_provider = provider and provider.startswith("vertex")
    prefix_filter = "vertex_ai/" if is_vertex_provider else None

    fuzzy_match = _fuzzy_match_model_key(
        model_name, available_keys, prefix_filter=prefix_filter
    )
    if fuzzy_match:
        model_info = model_cost[fuzzy_match]
        return (
            model_info.get("input_cost_per_token"),
            model_info.get("output_cost_per_token"),
        )

    # For vertex models, try common patterns with fuzzy matching restricted to vertex_ai/ prefix
    possible_lookup_keys = [
        f"vertex_ai/{model_name}",  # Direct vertex_ai prefix
        f"vertex_ai/meta/{model_name}",  # Meta/Llama models
        f"vertex_ai/mistral-{model_name}",  # Mistral models (if not already prefixed)
    ]

    for lookup_key in possible_lookup_keys:
        # Try exact match first
        if lookup_key in model_cost:
            model_info = model_cost[lookup_key]
            return (
                model_info.get("input_cost_per_token"),
                model_info.get("output_cost_per_token"),
            )

        # Restrict fuzzy matching to vertex_ai/ prefix if present
        prefix = None
        if lookup_key.startswith("vertex_ai/"):
            prefix = "vertex_ai/"
        fuzzy_match = _fuzzy_match_model_key(
            lookup_key, available_keys, prefix_filter=prefix
        )
        if fuzzy_match:
            model_info = model_cost[fuzzy_match]
            return (
                model_info.get("input_cost_per_token"),
                model_info.get("output_cost_per_token"),
            )

    logger.warning(
        f"No cost information found for model: {model_name} (tried lookup patterns with fuzzy matching)"
    )
    return None


def calculate_token_costs(
    usage_metadata: dict,
    model_name: Optional[str] = None,
) -> dict:
    """Calculate costs for input and output tokens using official model costs when available."""
    if not usage_metadata or not isinstance(usage_metadata, dict):
        logger.warning("No usage metadata provided or usage_metadata is not a dict")
        return {"input_cost": None, "output_cost": None}

    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)

    if not model_name:
        logger.warning("No model name provided for cost calculation")
        return {"input_cost": None, "output_cost": None}

    costs = get_model_costs(model_name)
    if not costs:
        return {"input_cost": None, "output_cost": None}

    input_cost, output_cost = costs
    return {
        "input_cost": input_tokens * input_cost if input_cost is not None else None,
        "output_cost": output_tokens * output_cost if output_cost is not None else None,
    }


def calculate_time_based_costs(
    response_metadata: dict,
) -> dict:
    """Calculate costs for Vertex AI Model Garden deployments based on time usage."""
    if not response_metadata or not isinstance(response_metadata, dict):
        logger.warning("No response metadata provided for time-based cost calculation")
        return {"input_cost": None, "output_cost": None}

    # Check if this is a vertex model garden deployment
    if not response_metadata.get("is_vertex_model_garden", False):
        return {"input_cost": None, "output_cost": None}

    response_time_seconds = response_metadata.get("response_time_seconds", 0)
    hourly_cost = response_metadata.get("hourly_cost", 4.6028)  # Default cost per hour

    if response_time_seconds <= 0:
        logger.warning("Invalid response time for cost calculation")
        return {"input_cost": None, "output_cost": None}

    # Convert seconds to hours and calculate cost
    response_time_hours = response_time_seconds / 3600
    total_cost = response_time_hours * hourly_cost

    logger.info(
        f"Time-based cost calculation: {response_time_seconds:.3f}s "
        f"({response_time_hours:.6f}h) × ${hourly_cost:.4f}/h = ${total_cost:.8f}"
    )

    # For time-based models, we return the total cost as "input_cost" and 0 for "output_cost"
    # to maintain compatibility with the existing cost tracking structure
    return {
        "input_cost": total_cost,
        "output_cost": 0.0,
    }


def extract_model_from_tool_calls(step: Dict) -> Optional[str]:
    """Extract model name from tool calls in the response, prioritizing the selected model over supervisor model."""
    # Iterate through all keys in the step to find messages
    for step_name, step_data in step.items():
        if isinstance(step_data, dict) and "messages" in step_data:
            for message in step_data["messages"]:
                # First check for tool_calls args (this contains the selected model)
                if hasattr(message, "tool_calls"):
                    for tool_call in message.tool_calls:
                        if (
                            isinstance(tool_call, dict)
                            and "args" in tool_call
                            and isinstance(tool_call["args"], dict)
                        ):
                            # Check for both 'model_name' and 'model' keys
                            if "model_name" in tool_call["args"]:
                                return tool_call["args"]["model_name"]
                            if "model" in tool_call["args"]:
                                return tool_call["args"]["model"]

                # If no model found in tool_calls, fallback to response_metadata
                if hasattr(message, "response_metadata"):
                    # Check for both 'model_name' and 'model' keys
                    model_name = message.response_metadata.get("model_name")
                    if model_name:
                        return model_name
                    model = message.response_metadata.get("model")
                    if model:
                        return model
    return None


def process_agent_stream(
    agent_stream: Generator, model_name: Optional[str] = None
) -> Tuple[List[Dict], Optional[float], Optional[float], Optional[str]]:
    """Process the agent stream and track costs."""
    responses = []
    total_input_cost = 0
    total_output_cost = 0
    selected_model = None
    has_valid_costs = False  # Track if we found any valid costs

    for step in agent_stream:
        responses.append(step)
        # Extract model name if not already found
        selected_model = extract_model_from_tool_calls(step) or model_name
        # Fallback: try to fetch model_name from agent_stream's model attribute
        if not selected_model:
            model_obj = getattr(agent_stream, "model", None)
            if model_obj and hasattr(model_obj, "model_name"):
                selected_model = model_obj.model_name

        # Process any step that contains messages with usage metadata or response metadata
        for step_name, step_data in step.items():
            if isinstance(step_data, dict) and "messages" in step_data:
                for message in step_data["messages"]:
                    costs = {"input_cost": None, "output_cost": None}

                    # Check if this is a vertex model garden deployment (time-based cost)
                    if hasattr(
                        message, "response_metadata"
                    ) and message.response_metadata.get(
                        "is_vertex_model_garden", False
                    ):
                        costs = calculate_time_based_costs(message.response_metadata)
                    # Otherwise use token-based cost calculation
                    elif hasattr(message, "usage_metadata"):
                        costs = calculate_token_costs(
                            message.usage_metadata, model_name=selected_model
                        )

                    # Only process if we got valid costs
                    if (
                        costs["input_cost"] is not None
                        and costs["output_cost"] is not None
                    ):
                        has_valid_costs = True
                        total_input_cost += costs["input_cost"]
                        total_output_cost += costs["output_cost"]
                        logger.info(
                            f"Input Costs: ${costs['input_cost']:.8f}, "
                            f"Output Cost: ${costs['output_cost']:.8f}"
                        )
                    elif hasattr(message, "usage_metadata"):
                        # Only warn if there was usage_metadata but we couldn't calculate costs
                        logger.warning(
                            f"Unable to calculate costs for {selected_model}"
                        )

        logger.debug(f"Step: {step}")

    # Log final cost summary
    logger.info("\nFinal Cost Summary:")
    if has_valid_costs:
        logger.info(f"Total Input Cost: ${total_input_cost:.8f}")
        logger.info(f"Total Output Cost: ${total_output_cost:.8f}")
        logger.info(
            f"Total Combined Cost: ${(total_input_cost + total_output_cost):.8f}"
        )
    else:
        logger.info("Cost information not available for this model")

    if selected_model:
        logger.info(f"Selected Model: {selected_model}")

    return (
        responses,
        total_input_cost if has_valid_costs else None,
        total_output_cost if has_valid_costs else None,
        selected_model,
    )

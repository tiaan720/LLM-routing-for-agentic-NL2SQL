import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.cost_calc.cost_tracker import get_model_costs
from src.models.base import create_model

MODEL_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "llm_model_config.json"
)
MODEL_DESCRIPTION_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "llm_model_descriptions.json"
)

PROMPT_TEMPLATE = """
You are an expert LLM evaluator and technical writer. Your task is to assess the model '{model_name}' (provider: {provider}) and generate a structured JSON object with the following fields:

- description: Provide a concise, unbiased summary of the model, highlighting its unique capabilities and strengths. Avoid marketing language.
- intelligence_rating: Rate the model's reasoning and problem-solving ability as High, Medium, or Low, based on technical benchmarks and real-world performance.
- speed_rating: Rate the model's response speed (inference latency and throughput) as High, Medium, or Low, considering typical deployment scenarios.
- cost_rating: Rate the model's cost-effectiveness as Affordable, Moderate, or Expensive. Affordable = low cost per token, Expensive = high cost per token.
- type: Specify the model's primary use case (e.g., chat-completion, reasoning, code generation, multi-modal, etc.).
- strengths: List 3-5 specific strengths, such as advanced reasoning, multi-turn chat, factual accuracy, code generation, etc. Use short phrases.
- use_cases: List 3-5 ideal use cases for this model, such as customer support, tutoring, code review, etc. Use short phrases.
- notes: Add any other relevant technical details, limitations, or deployment considerations. Be concise and factual.

{cost_info_section}

Respond ONLY with the JSON object, with no extra commentary. Format lists as arrays of strings. Be precise and avoid speculation.
"""


# Define the structured output schema using Pydantic
class ModelDescription(BaseModel):
    description: str = Field(
        ...,
        description="Concise, unbiased summary of the model, focusing on unique capabilities and technical strengths. Avoid marketing language.",
    )
    intelligence_rating: str = Field(
        ...,
        description="Model's reasoning and problem-solving ability: High, Medium, or Low. Base on benchmarks and real-world performance.",
    )
    speed_rating: str = Field(
        ...,
        description="Model's response speed (latency/throughput): High, Medium, or Low. Consider typical deployment scenarios.",
    )
    cost_rating: str = Field(
        ...,
        description="Model's cost-effectiveness: Affordable (low cost), Moderate, or Expensive (high cost). Consider API pricing and compute requirements.",
    )
    type: str = Field(
        ...,
        description="Primary use case or modality (e.g., chat-completion, reasoning, code generation, multi-modal, etc.).",
    )
    strengths: list = Field(
        ...,
        description="3-5 specific strengths (e.g., advanced reasoning, multi-turn chat, factual accuracy, code generation, etc.). Short phrases.",
    )
    use_cases: list = Field(
        ...,
        description="3-5 ideal use cases (e.g., customer support, tutoring, code review, etc.). Short phrases.",
    )
    notes: str = Field(
        ...,
        description="Other relevant technical details, limitations, or deployment considerations. Be concise and factual.",
    )


def get_cost_info_for_prompt(model_name: str, provider: str = None) -> str:
    """Get cost information for a model and format it for inclusion in the prompt."""
    costs = get_model_costs(model_name, provider)

    if not costs:
        return "COST INFORMATION: No cost data available for this model. Rate cost based on model size and typical cloud deployment patterns."

    input_cost, output_cost = costs

    if input_cost == 0.0 and output_cost == 0.0:
        return "COST INFORMATION: This is a local/self-hosted model with zero API costs. Rate as Affordable."

    # Convert to costs per 1M tokens for easier interpretation
    input_cost_per_million = input_cost * 1_000_000 if input_cost else 0
    output_cost_per_million = output_cost * 1_000_000 if output_cost else 0

    cost_info = f"""COST INFORMATION:
- Input cost: ${input_cost_per_million:.4f} per 1M tokens
- Output cost: ${output_cost_per_million:.4f} per 1M tokens
- Total cost for typical 1M input + 1M output tokens: ${input_cost_per_million + output_cost_per_million:.4f}

Use this information to accurately rate the cost_rating field:
- Affordable: < $5 per 1M total tokens
- Moderate: $5-20 per 1M total tokens  
- Expensive: > $20 per 1M total tokens"""

    return cost_info


def generate_model_description_llm(
    model_name: str, provider: str, model_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    # Get cost information for the model
    cost_info_section = get_cost_info_for_prompt(model_name, provider)

    prompt = PROMPT_TEMPLATE.format(
        model_name=model_name, provider=provider, cost_info_section=cost_info_section
    )
    parser = PydanticOutputParser(pydantic_object=ModelDescription)
    full_prompt = f"{prompt}\n{parser.get_format_instructions()}\nModel kwargs:\n{json.dumps(model_kwargs, indent=2)}"

    openai_model_kwargs = {
        "model": "gpt-5",
        "temperature": 0.0,
        "max_tokens": None,
        "max_retries": 5,
        "stop": None,
        "temperature": 1,
    }
    try:
        llm = create_model("openai", openai_model_kwargs)
        response = llm.invoke([("system", full_prompt)])
        parsed = parser.parse(
            response.content if hasattr(response, "content") else str(response)
        )
        return parsed.model_dump()
    except Exception as e:
        return {
            "description": f"{model_name} ({provider}) - Error generating description: {e}",
            "intelligence_rating": "Unknown",
            "speed_rating": "Unknown",
            "cost_rating": "Unknown",
            "type": "Unknown",
            "strengths": [],
            "use_cases": [],
            "notes": str(e),
        }


def main():
    # Clear the model descriptions file at the start
    with open(MODEL_DESCRIPTION_PATH, "w") as f:
        json.dump({}, f)

    with open(MODEL_CONFIG_PATH, "r") as f:
        config = json.load(f)
    models = config.get("models", {})
    descriptions = {}
    total_models = sum(len(model_dict) for model_dict in models.values())
    with tqdm(
        total=total_models, desc="Generating model descriptions", unit="model"
    ) as pbar:
        for provider, model_dict in models.items():
            for model_key, model_info in model_dict.items():
                model_name = (
                    model_info.get("model_name") or model_info.get("model") or model_key
                )
                desc = generate_model_description_llm(model_name, provider, model_info)
                descriptions[model_name] = desc
                with open(MODEL_DESCRIPTION_PATH, "w") as f:
                    json.dump(descriptions, f, indent=2)
                pbar.update(1)
    print(f"Model descriptions written to {MODEL_DESCRIPTION_PATH}")


if __name__ == "__main__":
    main()

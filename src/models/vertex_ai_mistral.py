from langchain_core.runnables import Runnable

# Import the base module to extend the models list
from langchain_google_vertexai.model_garden_maas import _base
from langchain_google_vertexai.model_garden_maas.mistral import VertexModelGardenMistral

from src.models.base import ModelRegistry


def extend_mistral_models(additional_models: list[str]) -> None:
    """
    Extend the list of supported Mistral models.

    Args:
        additional_models: List of additional model names to support
    """
    for model in additional_models:
        if model not in _base._MISTRAL_MODELS:
            _base._MISTRAL_MODELS.append(model)


@ModelRegistry.register("vertex_mistral")
def create_vertex_ai_mistral_model(**model_kwargs) -> Runnable:
    # Extend supported models if needed
    additional_models = [
        "mistral-large-2411",
        "mistral-small-2503",
        "mistral-small-2503@001",
        # Add any other custom models you want to support here
    ]
    extend_mistral_models(additional_models)

    return VertexModelGardenMistral(streaming=True, **model_kwargs)  # Enable streaming


from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    # Example: Add support for additional models if needed
    # extend_mistral_models(["custom-mistral-model@001", "another-model"])

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    model_kwargs = {
        "model": "mistral-small-2503",
        "temperature": 0.0,
        "project": "research-su-llm-routing",
        "location": "us-central1",
    }

    model = create_vertex_ai_mistral_model(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)

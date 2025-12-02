from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden_maas import _base
from langchain_google_vertexai.model_garden_maas.llama import VertexModelGardenLlama

from src.models.base import ModelRegistry


def extend_llama_models(additional_models: list[str]) -> None:
    """
    Extend the list of supported Llama models.

    Args:
        additional_models: List of additional model names to support
    """
    for model in additional_models:
        if model not in _base._LLAMA_MODELS:
            _base._LLAMA_MODELS.append(model)


@ModelRegistry.register("vertex_meta")
def create_vertex_ai_meta_model(**model_kwargs) -> Runnable:
    # Extend supported models if needed
    additional_models = [
        "meta/llama-4-maverick-17b-128e-instruct-maas",
        "meta/llama-4-scout-17b-16e-instruct-maas",
        "meta/llama-3.3-70b-instruct-maas",
        # Add any other custom models you want to support here
    ]
    extend_llama_models(additional_models)

    return VertexModelGardenLlama(streaming=True, **model_kwargs)  # Enable streaming


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
        "model": "meta/llama-4-scout-17b-16e-instruct-maas",
        "temperature": 0.0,
        "project": "research-su-llm-routing",
        "location": "us-east5",
        "append_tools_to_system_message": True,
    }

    # model_kwargs = {
    #     "model": "meta/llama-3.3-70b-instruct-maas",
    #     "temperature": 0.0,
    #     "project": "research-su-llm-routing",
    #     "location": "us-central1",
    #     "append_tools_to_system_message": True,
    # }
    model = create_vertex_ai_meta_model(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)

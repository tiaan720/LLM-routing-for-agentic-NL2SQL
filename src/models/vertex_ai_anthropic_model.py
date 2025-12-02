from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from src.models.base import ModelRegistry


@ModelRegistry.register("vertex_anthropic")
def create_vertex_ai_anthropic_model(**model_kwargs) -> Runnable:
    return ChatAnthropicVertex(streaming=True, **model_kwargs)  # Enable streaming


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
        "model": "claude-3-7-sonnet@20250219",
        "temperature": 0.0,
        "project": "research-su-llm-routing",
        "location": "us-east5",
    }
    model = create_vertex_ai_anthropic_model(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)

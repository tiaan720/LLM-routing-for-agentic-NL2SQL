from langchain_core.runnables import Runnable
from langchain_google_vertexai import ChatVertexAI

from src.models.base import ModelRegistry


@ModelRegistry.register("vertex_ai")
def create_vertex_ai_model(**model_kwargs) -> Runnable:
    return ChatVertexAI(streaming=True, **model_kwargs)  # Enable streaming


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
        "model_name": "gemini-2.5-pro",
        "temperature": 0.0,
        "max_tokens": None,
        "max_retries": 5,
        "stop": None,
    }
    model = create_vertex_ai_model(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)


# example of where this tests comes from: https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/

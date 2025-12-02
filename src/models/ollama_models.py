from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from src.models.base import ModelRegistry


@ModelRegistry.register("ollama")
def create_ollama_model(**model_kwargs) -> Runnable:
    return ChatOllama(**model_kwargs)


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

    model_kwargs = {"model": "gemma3n:latest", "temperature": 0.0}
    model = create_ollama_model(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)

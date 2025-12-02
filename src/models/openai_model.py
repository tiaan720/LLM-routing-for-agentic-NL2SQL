from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.models.base import ModelRegistry


@ModelRegistry.register("openai")
def create_openai_model(**model_kwargs) -> Runnable:
    # Include token usage in streaming responses
    model_kwargs["model_kwargs"] = model_kwargs.get("model_kwargs", {})
    model_kwargs["model_kwargs"]["stream_options"] = {"include_usage": True}
    return ChatOpenAI(streaming=True, **model_kwargs)


if __name__ == "__main__":

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    model_kwargs = {
        "model": "gpt-4.1",
        "temperature": 0.0,
        "max_tokens": None,
        "max_retries": 5,
        "stop": None,
    }

    model = create_openai_model(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)

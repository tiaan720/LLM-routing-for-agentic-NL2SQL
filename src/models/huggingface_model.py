import os

import dotenv
from langchain_core.runnables import Runnable
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)

from src.models.base import ModelRegistry

dotenv.load_dotenv()


@ModelRegistry.register("huggingface")
def create_huggingface_model(**model_kwargs) -> Runnable:
    return ChatHuggingFace(streaming=True, **model_kwargs)


if __name__ == "__main__":

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    # Option 1: Using HuggingFaceEndpoint (Inference API - Remote)
    # llm = HuggingFaceEndpoint(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    #     task="text-generation",
    #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    #     max_new_tokens=512,
    #     do_sample=False,
    #     repetition_penalty=1.03,
    # )

    # Option 2: Using HuggingFacePipeline (Local Model)
    llm = HuggingFacePipeline.from_model_id(
        model_id="agentica-org/DeepScaleR-1.5B-Preview",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.03,
        },
    )

    model_kwargs = {"llm": llm}

    model = create_huggingface_model(**model_kwargs)
    response = model.invoke(messages)
    print(response.content)

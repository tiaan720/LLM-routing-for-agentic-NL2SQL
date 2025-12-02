# from langchain_core.runnables import Runnable
# from langchain_litellm import ChatLiteLLM


# def create_litellm_model(**model_kwargs) -> Runnable:
#     return ChatLiteLLM(streaming=True, **model_kwargs)  # Enable streaming


# from dotenv import load_dotenv

# load_dotenv()

# # Lite llm for langchain seems to have all other integrations but not vertex ai as an option.
# # openai_api_key: str | None = None,
# # azure_api_key: str | None = None,
# # anthropic_api_key: str | None = None,
# # replicate_api_key: str | None = None,
# # cohere_api_key: str | None = None,
# # openrouter_api_key: str | None = None,

# if __name__ == "__main__":

#     messages = [
#         (
#             "system",
#             "You are a helpful assistant that translates English to French. Translate the user sentence.",
#         ),
#         ("human", "I love programming."),
#     ]

#     model_kwargs = {
#         "model": "vertex_ai/claude-3-7-sonnet@20250219",
#         "temperature": 0.0,
#     }

#     model = create_litellm_model(**model_kwargs)
#     response = model.invoke(messages)
#     print(response.content)

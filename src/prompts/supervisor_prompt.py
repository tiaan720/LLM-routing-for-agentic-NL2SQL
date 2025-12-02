import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


def get_supervisor_system_prompt_text():
    """
    Returns a system prompt for the supervisor agent, including model descriptions from config.
    """
    # Load model descriptions
    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "llm_model_descriptions.json"
    )
    with open(config_path, "r") as f:
        descriptions = json.load(f)

    # Build model descriptions text
    model_descs = []
    for model, info in descriptions.items():
        desc = f"Model: {model}\nDescription: {info.get('description', '')}\nStrengths: {', '.join(info.get('strengths', []))}\nUse cases: {', '.join(info.get('use_cases', []))}\nNotes: {info.get('notes', '')}\n"
        model_descs.append(desc)
    models_text = "\n".join(model_descs)

    prompt = f"""You are an AI supervisor that routes SQL queries to the most appropriate model.\n\n
    Available models and their descriptions:\n{models_text}\n\n
    For each user query:\n
    1. Carefully analyze the user question and its complexity, type, and requirements.\n
    2. Review the descriptions, strengths, and use cases of all available models.\n
    3. Select the single best model that is most likely to answer the question effectively, based on the model descriptions and the query analysis.\n
    4. Use sql_agent_as_tool with the chosen model configuration.\n
    5. Only if the selected model fails to answer or errors, try the next best model. Do NOT loop through all models' answers or compare them. Decide on the best model first, and only pick another if the first fails.\n
    6. Return the first successful result to the user, and indicate which model was used.\n\n
    After your selection the sql agent tool will return an answer to you and you will interpret it, relate it to user question and then fully return the answer from the sql agent tool back to the user. 
    Dont mention the model used in final response. Just return the answer you recieved from the sql agent tool that suucesfully answered the question.
    Only recall the sql agent tool response again if the with a different LLM model if the previous model in the response failed to answer the question or returned an error. If you see the same or similar output answer from a different model with the sql agent tool, then it means that could be the answer and then no need to get stuck in a loop where you keep trying different models. The entire goal is to pick a model for an sql agent to delegate to and then that agent will return back the answer to you as the superviosr and then its your jobs to see if the anwer is valid and then if not deligate to a different model. Only try another model if the previous one failed to answer or returned an error. 
    """
    return prompt


def get_supervisor_system_prompt():
    """
    Returns a ChatPromptTemplate for the supervisor agent.
    """
    prompt_text = get_supervisor_system_prompt_text()
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "{messages}"),
        ]
    )

import csv
from io import StringIO
from typing import Literal, Optional, Tuple

import dotenv
import google.auth
import pandas as pd
from google.auth.transport.requests import Request

from src.agents.sql_agent import create_sql_agent
from src.models.base import create_model

# Define models to compare
models = {
    "gpt-4o": create_model(model="openai", model_kwargs={"model": "gpt-4o"}),
    "gpt-3.5-turbo": create_model(
        model="openai", model_kwargs={"model": "gpt-3.5-turbo"}
    ),
}

# Create judge model with proper configuration
judge_model = create_model(
    model="openai", model_kwargs={"model": "gpt-4o", "temperature": 0.0}
)


def capture_response(agent, query):
    response_text = ""
    try:
        input = {"messages": [("human", query)]}
        config = {"configurable": {"thread_id": "1", "recursion_limit": 15}}

        for chunk in agent.stream(input, config, stream_mode="values"):
            response = chunk["messages"][-1]
            response_text += str(response)

        return True, response_text
    except Exception as e:
        return (
            False,
            f"{response_text}\n[ERROR: {str(e)}]",
        )  # Append error to any partial response


def judge_responses(response_a, response_b):
    judge_prompt = f"""Compare these two model responses and select the better one. 
    Response 1: {response_a}
    Response 2: {response_b}
    
    Reply with just the number (1 or 2) of the better response."""

    winner_response = judge_model.invoke(judge_prompt)
    winner = "1" if "1" in winner_response else "2"

    reason_prompt = f"Why did you choose response {winner}? Provide a brief explanation. Here is Response 1: {response_a} and Response 2: {response_b}. The winner is the respons that won and you should write in short why a response won over the other when judgining the two responses. "

    # reason_response = ""
    # for chunk in judge_model.stream(reason_prompt):
    #     reason_response += str(chunk)
    reason_response = judge_model.invoke(reason_prompt)

    return winner, reason_response


model = create_model(model="openai", model_kwargs={"model": "gpt-4o"})
user_creds, project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/bigquery"]
)

if not user_creds.valid:
    user_creds.refresh(Request())
user_token = user_creds.token

agent_kwargs = {
    "user_token": user_token,
    "project_id": project,
    "dataset_id": "pagila",
}


query = (
    "Make a varplot of the top three customers who rented the most dvd's in store 2."
)
results = []

model_names = list(models.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model_a_name = model_names[i]
        model_b_name = model_names[j]

        agent_a = create_sql_agent(model=models[model_a_name], **agent_kwargs)
        agent_b = create_sql_agent(model=models[model_b_name], **agent_kwargs)

        success_a, response_a = capture_response(agent_a, query)
        success_b, response_b = capture_response(agent_b, query)

        # Only judge if both models succeeded
        if success_a and success_b:
            winner, reason = judge_responses(response_a, response_b)
            winner_name = model_a_name if winner == "1" else model_b_name
        else:
            winner_name = model_b_name if not success_a else model_a_name
            reason = "Winner selected by default due to other model failure"

        results.append(
            {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "winner": winner_name,
                "conversation_a": response_a,
                "conversation_b": response_b,
                "reason": reason,
            }
        )


df = pd.DataFrame(results)
df.to_csv("data/model_comparison_results.csv", index=False)

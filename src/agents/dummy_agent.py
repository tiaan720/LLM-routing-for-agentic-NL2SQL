import sys
from pathlib import Path

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from src.cost_calc.cost_tracker import process_agent_stream
from src.models.base import create_model
from src.tools.dummy_agent_tools import create_dummy_agent_tools

dotenv.load_dotenv()


def create_dummy_agent(
    model: Runnable,
    input_query: str,
    tools: list = None,
    **kwargs,
) -> CompiledGraph:
    """
    Create a dummy agent with the given model.

    Args:
        model: The language model to use
        input_query: The user's input query for RAG retrieval
        **kwargs: Additional configuration parameters

    Returns:
        CompiledGraph: The configured SQL agent
    """

    agent_tools = []
    agent_tools.extend(
        create_dummy_agent_tools(
            requested_tools=tools,
            **kwargs,
        )
    )

    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful mathematical assistant. Your task is simple:

                1. When a user asks you to add two numbers, use the plus_calculator tool EXACTLY ONCE
                2. Call the tool with the two numbers from the user's question
                3. Return the result to the user
                4. Do NOT call the tool multiple times
                5. Do NOT overthink the problem

                For the question "What is 1 + 1?":
                - Call plus_calculator(1, 1) 
                - Report the result
                - STOP

                Be direct and concise. One tool call, one answer, done.""",
            ),
            ("human", "{messages}"),
        ]
    )
    agent = create_react_agent(
        model=model, tools=agent_tools, prompt=system_prompt, store=InMemoryStore()
    )

    return agent


if __name__ == "__main__":

    # model = create_model(
    #     model="vertex_ai",
    #     model_kwargs={
    #         "model_name": "gemini-2.5-pro",
    #         "temperature": 0.0,
    #         "max_retries": 0,
    #         "project": "research-su-llm-routing",
    #         "location": "us-east4",
    #     },
    # )

    model = create_model(
        model="vertex_mistral",
        model_kwargs={
            "model": "mistral-small-2503",
            "temperature": 0.0,
            "project": "research-su-llm-routing",
            "location": "us-central1",
            "append_tools_to_system_message": True,
        },
    )

    agent_kwargs = {
        "tools": [
            "plus_calculator",
        ],
    }

    query = "What is 1 + 1?"

    agent = create_dummy_agent(model=model, input_query=query, **agent_kwargs)

    input = {"messages": [("user", query)]}

    config = {"configurable": {"thread_id": "1"}}

    responses, input_cost, output_cost, selected_model = process_agent_stream(
        agent.stream(input, config, stream_mode="updates")
    )

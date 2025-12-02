import json
from pathlib import Path
from typing import Annotated, Any

import dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from pydantic import BaseModel, Field

from src.agents.sql_agent import create_sql_agent
from src.cost_calc.cost_tracker import process_agent_stream
from src.cost_calc.shared_store import get_shared_store
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model

dotenv.load_dotenv()


# Dynamically build the enum from llm_model_config.json
def get_model_enum():
    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "llm_model_config.json"
    )
    with open(config_path, "r") as f:
        configs = json.load(f)["models"]
    model_keys = []
    for models in configs.values():
        model_keys.extend(list(models.keys()))
    return model_keys


class ModelSelectionSchema(BaseModel):
    model_name: str = Field(
        description="The name of the model to use",
        enum=get_model_enum(),
    )


def create_supervisor_tools(
    db_client, input_query: str, requested_tools: list = None, **kwargs
):
    """Creates tools with proper state management, using db_client and config like sql_agent_tools"""

    client = db_client.client
    db_type = db_client.db_type

    @tool(args_schema=ModelSelectionSchema)
    def sql_agent_as_tool(
        model_name: str,
        store: Annotated[Any, InjectedStore()] = None,
    ) -> str:
        """
        You should always use this tool!!!

        Use this tool to select the appropriate model for SQL queries.

        Args:
            model_name: Name of the model to use
        Returns:
            The execution of the sql agent and its results.
        """

        # Clear any previous SQL queries from the shared store
        shared_store = get_shared_store()
        shared_store.put(("sql_execution",), "queries", [])

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "llm_model_config.json"
        )
        with open(config_path, "r") as f:
            configs = json.load(f)["models"]

        # Find model config by matching model_name prefix
        base_name = model_name.split("_")[0] if "_" in model_name else model_name
        found = False
        for provider, models in configs.items():
            for key, val in models.items():
                if key.lower().startswith(base_name.lower()):
                    model_provider = provider
                    model_kwargs = val
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"No config found for model: {model_name}")

        llm_model = create_model(model=model_provider, model_kwargs=model_kwargs)

        agent_kwargs = {
            "database_type": db_type,
            "host": getattr(db_client, "host", None),
            "port": getattr(db_client, "port", None),
            "database": getattr(db_client, "database", None),
            "username": getattr(db_client, "username", None),
            "password": getattr(db_client, "password", None),
            "user_token": getattr(db_client, "user_token", None),
            "project_id": getattr(db_client, "project_id", None),
            "dataset_id": getattr(db_client, "dataset_id", None),
            "tools": [
                "postgres_execute_sql",
                "postgres_list_schema",
            ],
        }
        agent_kwargs = {k: v for k, v in agent_kwargs.items() if v is not None}

        agent = create_sql_agent(
            model=llm_model, input_query=input_query, **agent_kwargs
        )

        input = {"messages": [("user", input_query)]}
        config = {"configurable": {"thread_id": "1"}}

        # Use store-based cost tracking for nested agent
        responses, input_cost, output_cost, selected_model, cost_tracker = (
            process_agent_stream_with_store(
                agent.stream(input, config, stream_mode="updates"),
                store=get_shared_store(),
                agent_name=f"sql_agent_{model_name}",
                model_name=llm_model.model_name,
            )
        )

        # Get the final AI response from the last response
        final_response = "No response generated from the SQL agent"
        if responses:
            last_response = responses[-1]
            if "agent" in last_response and "messages" in last_response["agent"]:
                for message in reversed(last_response["agent"]["messages"]):
                    if (
                        hasattr(message, "type")
                        and message.type == "ai"
                        and hasattr(message, "content")
                        and message.content
                        and message.content.strip()
                    ):
                        final_response = message.content
                        break

        # Get SQL queries from shared store
        executed_queries = []
        shared_store = get_shared_store()
        stored_queries_obj = shared_store.get(("sql_execution",), "queries")
        if stored_queries_obj is not None:
            executed_queries = stored_queries_obj.value

        # Build response with SQL queries appended
        if executed_queries:
            sql_section = "\n\nSQL Queries Executed:\n" + "\n".join(
                f"Query {i}: {query}"
                for i, query in enumerate(executed_queries, 1)
                if query.strip()
            )
            return final_response + sql_section

        return final_response

    all_tools = {
        name: func
        for name, func in locals().items()
        if hasattr(func, "name") and hasattr(func, "invoke")
    }

    tools = []
    if requested_tools is None:
        tools.extend(all_tools.values())
    else:
        for tool_name in requested_tools:
            if tool_name in all_tools:
                tools.append(all_tools[tool_name])

    return tools

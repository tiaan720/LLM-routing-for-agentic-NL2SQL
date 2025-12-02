from typing import Optional

import dotenv
import google.auth
import mlflow.langchain
from google.auth.transport.requests import Request
from langchain_core.runnables import Runnable
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from src.agents.system_state_modifier import create_state_modifier
from src.cost_calc.cost_tracker import process_agent_stream
from src.cost_calc.shared_store import get_shared_store
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model
from src.models.ollama_models import create_ollama_model
from src.prompts import get_system_prompt_for_db_type
from src.retrievers import example_retriever
from src.schema import DatabaseConfig
from src.tools.sql_agent_tools import create_sql_agent_tools
from src.utils.db_clients import create_database_client

dotenv.load_dotenv()


def create_sql_agent(
    model: Runnable,
    input_query: str,
    database_type: str = None,
    user_token: str = None,
    project_id: str = None,
    dataset_id: str = None,
    host: str = None,
    port: int = None,
    database: str = None,
    username: str = None,
    password: str = None,
    tools: list = None,
    **kwargs,
) -> CompiledGraph:
    """
    Create a SQL agent with the given model and database configuration.

    Args:
        model: The language model to use
        input_query: The user's input query for RAG retrieval
        database_type: "bigquery" or "postgres"
        user_token: BigQuery user token (for BigQuery)
        project_id: BigQuery project ID (for BigQuery)
        dataset_id: BigQuery dataset ID (for BigQuery)
        host: PostgreSQL host (for PostgreSQL)
        port: PostgreSQL port (for PostgreSQL)
        database: PostgreSQL database name (for PostgreSQL)
        username: PostgreSQL username (for PostgreSQL)
        password: PostgreSQL password (for PostgreSQL)
        tools: Optional list of tool names to use instead of default ones
        **kwargs: Additional configuration parameters

    Returns:
        CompiledGraph: The configured SQL agent
    """
    db_config = DatabaseConfig(**locals())

    db_client = create_database_client(**db_config.to_dict(), **kwargs)

    agent_tools = []
    agent_tools.extend(
        create_sql_agent_tools(
            db_client,
            dataset_id=db_config.dataset_id,
            project_id=db_config.project_id,
            database=db_config.database,
            requested_tools=tools,
            **kwargs,
        )
    )

    system_prompt = get_system_prompt_for_db_type(
        db_type=db_client.db_type,
        project_id=db_config.project_id,
        dataset_id=db_config.dataset_id,
        database=db_config.database,
        **kwargs,
    )

    state_modifier = create_state_modifier(
        system_prompt=system_prompt, retriever=example_retriever
    )

    agent = create_react_agent(
        model=model,
        tools=agent_tools,
        prompt=state_modifier,
        store=get_shared_store(),
    )

    # Attach the database client to the agent for cleanup purposes
    agent.db_client = db_client

    return agent


if __name__ == "__main__":

    # model = create_model(model="openai", model_kwargs={"model": "gpt-4o"})
    # model = create_model(model="vertex_ai", model_kwargs={"model": "gemini-1.5-pro"})

    # model = create_model(
    #     model="vertex_anthropic",
    #     model_kwargs={
    #         "model": "claude-opus-4@20250514",
    #         "temperature": 0.0,
    #         "project": "research-su-llm-routing",
    #         "location": "us-east5",
    #     },
    # )

    # model = create_model(
    #     model="ollama",
    #     model_kwargs={"model": "qwen3:14b", "temperature": 0.0},
    # )

    model = create_model(
        model="vertex_meta",
        model_kwargs={
            "model": "meta/llama-3.3-70b-instruct-maas",
            "temperature": 0.0,
            "project": "research-su-llm-routing",
            "location": "us-central1",
            "append_tools_to_system_message": True,
        },
    )

    # model = create_model(
    #     model="vertex_model_garden",
    #     model_kwargs={
    #         "project": "970641581678",
    #         "endpoint_id": "4549617487527804928",
    #         "location": "us-central1",
    #         "model": "qwen/qwen3-32B",
    #         "hourly_cost": 4.6028,
    #     },
    # )

    user_creds, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )

    if not user_creds.valid:
        user_creds.refresh(Request())
    user_token = user_creds.token

    # BigQuery database configuration as kwargs
    # agent_kwargs = {
    #     "database_type": "bigquery",
    #     "user_token": user_token,
    #     "project_id": project,
    #     "dataset_id": "pagila",
    #     "tools": [
    #         "bigquery_execute_sql_with_validation",
    #         "generate_plot_from_data",
    #         "big_query_list_schema",
    #     ],
    # }

    # PostgreSQL database configuration example (uncomment to use)
    agent_kwargs = {
        "database_type": "postgres",
        "host": "localhost",
        "port": 5432,
        "database": "academic",
        "username": "postgres",
        "password": "postgres",
        "tools": [
            "postgres_execute_sql",
            "postgres_list_schema",
        ],
    }

    query = 'Which authors have written publications in both the domain "Machine Learning" and the domain "Data Science"?'

    agent = create_sql_agent(model=model, input_query=query, **agent_kwargs)

    input = {"messages": [("user", query)]}

    config = {"configurable": {"thread_id": "1"}}

    responses, input_cost, output_cost, selected_model, cost_tracker = (
        process_agent_stream_with_store(
            agent.stream(input, config, stream_mode="updates"),
            store=get_shared_store(),
            agent_name="sql_agent",
            model_name=model.model_name,
        )
    )

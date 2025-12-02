import dotenv
from langchain_core.runnables import Runnable
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from src.cost_calc.cost_tracker import process_agent_stream
from src.cost_calc.shared_store import get_shared_store
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model
from src.prompts.supervisor_prompt import get_supervisor_system_prompt
from src.schema import DatabaseConfig
from src.tools.supervisor_tool import create_supervisor_tools
from src.utils.db_clients import create_database_client

dotenv.load_dotenv()


def create_supervisor_agent(
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
    **kwargs,
) -> CompiledGraph:
    """Create a supervisor agent that manages model routing for SQL queries"""

    db_config = DatabaseConfig(**locals())

    db_client = create_database_client(**db_config.to_dict(), **kwargs)

    agent_tools = []
    agent_tools.extend(
        create_supervisor_tools(
            db_client=db_client,
            input_query=input_query,
            **kwargs,
        )
    )

    system_prompt = get_supervisor_system_prompt()

    agent = create_react_agent(
        model=model,
        tools=agent_tools,
        prompt=system_prompt,
        store=get_shared_store(),
    )

    return agent


if __name__ == "__main__":

    model = create_model(
        model="vertex_ai",
        model_kwargs={
            "model_name": "gemini-2.5-pro",
            "temperature": 0.0,
            "max_tokens": None,
            "max_retries": 5,
            "stop": None,
        },
    )

    # Uncomment this block for BigQuery
    # user_creds, project = google.auth.default(
    #     scopes=["https://www.googleapis.com/auth/bigquery"]
    # )
    # if not user_creds.valid:
    #     user_creds.refresh(Request())
    # user_token = user_creds.token
    # agent_kwargs = {
    #     "database_type": "bigquery",
    #     "user_token": user_token,
    #     "project_id": project,
    #     "dataset_id": "pagila",
    # }

    # Uncomment this block for PostgreSQL
    agent_kwargs = {
        "database_type": "postgres",
        "host": "localhost",
        "port": 5432,
        "database": "academic",
        "username": "postgres",
        "password": "postgres",
        "tools": [
            "sql_agent_as_tool",
        ],
    }

    queries = [
        "Show me a simple count of all customers",
    ]

    for query in queries:
        print(f"\nExecuting query: {query}")
        agent = create_supervisor_agent(model=model, input_query=query, **agent_kwargs)
        input_data = {"messages": [("human", query)]}
        config = {"configurable": {"thread_id": "1"}}
        responses, input_cost, output_cost, selected_model, cost_tracker = (
            process_agent_stream_with_store(
                agent.stream(input_data, config, stream_mode="updates"),
                store=get_shared_store(),
                agent_name="supervisor_agent",
                model_name=model.model_name,
            )
        )

import json
from pathlib import Path

import dotenv

from src.agents.sql_agent import create_sql_agent
from src.models.base import create_model
from src.routers.matrix_router import ModelRouter

dotenv.load_dotenv()


def get_routed_model(query: str, model_filename: str) -> tuple[str, str]:
    """
    Get the appropriate model configuration based on the query using a specific trained router model.

    Args:
        query: The user's query
        model_filename: Specific model file to load from models/ directory
    Returns:
        tuple[str, str]: (model_type, model_name)
    """

    router = ModelRouter()
    if not router.load_specific_model(model_filename):
        raise ValueError(f"Could not load model: {model_filename}")
    model_name = router.route_query(query)
    # Remove location from model_name (everything after first underscore)
    if "_" in model_name:
        base_name = model_name.split("_")[0]
    else:
        base_name = model_name
    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "llm_model_config.json"
    )
    with open(config_path, "r") as f:
        configs = json.load(f)["models"]
    for provider, models in configs.items():
        for key, val in models.items():
            if key.lower().startswith(base_name.lower()):
                # Normalize model name key
                model_name_val = val.get("model_name") or val.get("model")
                val = dict(val)  # copy to avoid mutating original
                val["model_name"] = model_name_val
                return provider, val
    raise ValueError(f"No config found for model: {model_name}")


if __name__ == "__main__":

    query = "How many conferences where held?"

    # Load router model and route query
    model_filename = "router_model.pkl"
    router = ModelRouter()
    if not router.load_specific_model(model_filename):
        raise ValueError(f"Could not load model: {model_filename}")

    model_type, model_kwargs = get_routed_model(query, model_filename)
    routed_model = create_model(model=model_type, model_kwargs=model_kwargs)

    print(f"Routing query to model: {model_kwargs}")

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

    agent = create_sql_agent(model=routed_model, input_query=query, **agent_kwargs)

    input_data = {"messages": [("user", query)]}
    config = {"configurable": {"thread_id": "1"}}

    for chunk in agent.stream(input_data, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

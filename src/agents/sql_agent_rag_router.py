import json
import re
from pathlib import Path
from typing import Dict, Tuple

import dotenv

from src.agents.sql_agent import create_sql_agent
from src.models.base import create_model
from src.routers.rag_router import create_rag_router

dotenv.load_dotenv()


def _normalize(s: str) -> str:
    """Normalize model identifiers for reliable matching.

    Steps:
    - Lowercase
    - Replace '/' with '_' (CSV uses underscores where config uses slashes)
    - Remove region suffixes like '@us-central1', '@europe-west1', etc.
    - Collapse duplicate separators
    - Strip surrounding whitespace
    """
    if not s:
        return s
    s = s.lower().strip()
    s = s.replace("/", "_")
    # Split on '@' to remove location; keep first two segments if version present
    parts = s.split("@")
    if len(parts) >= 2:
        # keep model and optional version/date (first 2 parts), drop location that follows
        s = "@".join(parts[:2])
    else:
        s = parts[0]
    # Convert any remaining '@' (between model and version) to underscore for CSV parity
    s = s.replace("@", "_")
    # Remove any duplicate underscores
    s = re.sub(r"_+", "_", s)
    return s


def _build_config_index(configs: Dict) -> Dict[str, Tuple[str, Dict]]:
    """Build an index of normalized names -> (provider, config dict)."""
    index: Dict[str, Tuple[str, Dict]] = {}
    for provider, models in configs.items():
        for key, val in models.items():
            # Base raw names we might want to normalize
            candidates = {key}
            raw_model = val.get("model") or val.get("model_name")
            if raw_model:
                candidates.add(raw_model)
            # Add variant stripping location suffix fully (everything after first '@')
            if raw_model and "@" in raw_model:
                candidates.add(raw_model.split("@")[0])
            for cand in list(candidates):
                norm = _normalize(cand)
                if norm not in index:  # first wins to avoid accidental overwrites
                    # store a shallow copy so we can safely augment later
                    copied = dict(val)
                    copied["model_name"] = val.get("model_name") or val.get("model")
                    index[norm] = (provider, copied)
    return index


def get_routed_model(query: str) -> tuple[str, dict]:
    """Return (provider, model_config) for the model chosen by the RAG router.

    Uses strict normalized equality (NOT prefix) to avoid mapping e.g. 'meta_llama-4...' to 'meta/llama-3.3-70b'.
    Raises a clear error listing available normalized names if no match is found.
    """
    vector_store_path = "data/chatbot_arena_inmemory_vectorstore.pkl"
    routed_name = create_rag_router(query, vector_store_path)
    norm_routed = _normalize(routed_name)

    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "llm_model_config.json"
    )
    with open(config_path, "r") as f:
        configs = json.load(f)["models"]

    index = _build_config_index(configs)

    if norm_routed in index:
        provider, cfg = index[norm_routed]
        return provider, cfg

    # Fallback: attempt a secondary heuristic only if a unique close match exists
    close_matches = [k for k in index.keys() if norm_routed in k or k in norm_routed]
    if len(close_matches) == 1:
        provider, cfg = index[close_matches[0]]
        return provider, cfg

    raise ValueError(
        "No exact model config match for routed model '"
        f"{routed_name}' (normalized '{norm_routed}'). Available normalized names: "
        f"{sorted(index.keys())}"
    )


if __name__ == "__main__":
    query = "How many conferences where held?"

    model_type, model_kwargs = get_routed_model(query)
    routed_model = create_model(model=model_type, model_kwargs=model_kwargs)

    print(f"Routing query to model: {model_kwargs}")

    # PostgreSQL database configuration as kwargs
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

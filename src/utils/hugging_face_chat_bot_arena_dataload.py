import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

load_dotenv()

RAW_OUTPUT_FILE = "data/chatbot_arena_filtered.csv"
PROCESSED_OUTPUT_FILE = "data/chatbot_arena_processed.csv"


def extract_query(conversation) -> str:
    """Extract user query from conversation string."""
    if conversation is None:
        return None

    # Convert to string if it's a pandas Series or numpy array
    if isinstance(conversation, pd.Series):
        conversation = str(conversation.iloc[0]) if not conversation.empty else None
    elif isinstance(conversation, np.ndarray):
        conversation = str(conversation[0]) if conversation.size > 0 else None
    else:
        conversation = str(conversation)

    try:
        # Find the user message part
        user_start = conversation.find("'role': 'user'")
        if user_start == -1:
            return None

        # Find content associated with user role
        content_start = conversation.rfind("'content': '", 0, user_start)
        if content_start == -1:
            return None

        content_start += len("'content': '")
        content_end = conversation.find("', 'role'", content_start)

        if content_end == -1:
            return None

        content = conversation[content_start:content_end].strip()
        return content if content else None

    except Exception as e:
        print(f"Error extracting query: {str(e)}")
        return None


def is_target_model(model_name: str) -> bool:
    """Check if model name contains 'gpt' or 'gemini'."""
    return any(name in model_name.lower() for name in ["gpt", "gemini"])


def extract_and_process_data(
    models_to_include: set = None, save_raw: bool = True
) -> pd.DataFrame:
    """Extract, filter, and process chatbot arena conversations."""

    login(token=os.getenv("HF_TOKEN"))

    print("Loading dataset from HuggingFace...")
    ds = pd.read_parquet(
        "hf://datasets/lmsys/chatbot_arena_conversations/data/train-00000-of-00001-cced8514c7ed782a.parquet"
    )

    # Filter by models
    mask = ds["model_a"].apply(is_target_model) & ds["model_b"].apply(is_target_model)
    filtered_ds = ds[mask][
        ["model_a", "model_b", "winner", "conversation_a", "conversation_b"]
    ]

    if save_raw:
        filtered_ds.to_csv(RAW_OUTPUT_FILE, index=False)
        print(f"Saved raw filtered data to {RAW_OUTPUT_FILE}")

    # Process conversations with error handling
    print("Processing conversations...")
    processed_rows = []

    for _, row in tqdm(filtered_ds.iterrows(), total=len(filtered_ds)):
        try:
            query = extract_query(row["conversation_a"])
            if query:  # Only keep rows where we can extract a valid query
                processed_rows.append(
                    {
                        "model_a": row["model_a"],
                        "model_b": row["model_b"],
                        "winner": row["winner"],
                        "query": query,
                    }
                )
        except Exception as e:
            print(f"Error processing row: {str(e)}")
            continue

    processed_df = pd.DataFrame(processed_rows)

    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_OUTPUT_FILE), exist_ok=True)
    processed_df.to_csv(PROCESSED_OUTPUT_FILE, index=False)
    print(f"Saved processed data to {PROCESSED_OUTPUT_FILE}")

    print(f"Original rows: {len(filtered_ds)}")
    print(f"Processed rows: {len(processed_df)}")
    print(f"Rows dropped: {len(filtered_ds) - len(processed_df)}")

    return processed_df


if __name__ == "__main__":
    df = extract_and_process_data()

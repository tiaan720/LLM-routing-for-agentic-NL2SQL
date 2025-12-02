import argparse
import json
import time
from typing import Dict, List

import dotenv
import mlflow
from experiments.common.evaluation import run_evaluation
from experiments.common.plot_utils import (
    create_simple_gauge_figure,
    log_gauge_charts_and_metrics,
)
from experiments.common.retry_utils import retry_with_backoff

from src.agents.sql_agent import create_sql_agent
from src.cost_calc.cost_tracker import process_agent_stream
from src.models.base import create_model
from src.utils.logger import logger

dotenv.load_dotenv()


def process_single_query(
    question: str,
    base_kwargs: Dict,
    model_type: str,
    model_config: Dict,
    dataset_name: str = None,
) -> Dict:
    """Process a single query using a specific model and SQL agent"""
    start_time = time.time()
    input_cost = 0
    output_cost = 0
    model_name = "failed"
    agent = None

    def stream_and_process():
        nonlocal agent, input_cost, output_cost, model_name
        model_name_local = model_config.get("model_name") or model_config.get("model")
        model = create_model(model=model_type, model_kwargs=model_config)
        agent = create_sql_agent(
            model=model,
            input_query=question,
            **base_kwargs,
        )
        stream_generator = agent.stream(
            {"messages": [("human", question)]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="updates",
        )
        return process_agent_stream(
            stream_generator,
            model_name=model.model_name,
        )

    try:
        result = retry_with_backoff(stream_and_process, max_retries=3, initial_delay=1)
        execution_time = time.time() - start_time

        if result:
            responses, input_cost, output_cost, model_name = (
                result  # Update model_name here
            )
            tool_call_messages = []
            final_response = None
            for response in responses:
                if (
                    "agent" in response
                    and "messages" in response["agent"]
                    and response["agent"]["messages"]
                ):
                    for message in response["agent"]["messages"]:
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                if isinstance(tool_call, dict):
                                    tool_call_messages.append(str(tool_call))
                        if hasattr(message, "content") and message.content:
                            final_response = message.content
            if tool_call_messages:
                final_response = f"{final_response}\n\nTool Calls:\n" + "\n".join(
                    tool_call_messages
                )
            if final_response is None:
                raise ValueError("No response generated from the model")
        else:
            raise ValueError("No result from agent stream")
    except Exception as e:
        execution_time = time.time() - start_time
        final_response = f"Error executing query: {str(e)}"
        logger.error(
            f"Error processing query for dataset '{dataset_name}': {question[:100]}..., error: {str(e)}"
        )
    finally:
        if agent and hasattr(agent, "db_client"):
            try:
                agent.db_client.close()
                logger.debug("Closed database connection")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")

    # Ensure all numeric fields have valid values (not None or NaN)
    safe_query_cost = 0.0
    if input_cost is not None and output_cost is not None:
        try:
            safe_query_cost = float(input_cost + output_cost)
        except (TypeError, ValueError):
            safe_query_cost = 0.0

    safe_execution_time = 0.0
    if execution_time is not None:
        try:
            safe_execution_time = float(execution_time)
        except (TypeError, ValueError):
            safe_execution_time = 0.0

    return {
        "actual_answer": final_response,
        "query_cost": safe_query_cost,
        "execution_time": safe_execution_time,
        "model_used": model_name if model_name else "unknown",
        "dataset": dataset_name if dataset_name else "unknown",
    }


def run_all_model_evaluations(
    selected_models: List[str] = None,
    selected_datasets: List[str] = None,
    prefix: str = None,
):
    with open("configs/llm_model_config.json", "r") as f:
        model_config = json.load(f)

    with open("experiments/test_sets/comprehensive_qa.json", "r") as f:
        all_qa_data = json.load(f)

    # Filter out entries with missing required fields
    def is_valid_qa_item(item):
        # Check all required fields exist and are non-empty strings
        if not all(
            item.get(key) and isinstance(item.get(key), str) and item.get(key).strip()
            for key in ["question", "answer", "dataset"]
        ):
            return False

        # Exclude problematic answer values
        answer = item.get("answer")
        if answer in [None, "success", "No results", ""]:
            return False

        # If specific datasets are selected, check if this item matches
        if selected_datasets is not None and len(selected_datasets) > 0:
            if item.get("dataset") not in selected_datasets:
                return False

        return True

    filtered_qa_data = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "dataset": item["dataset"],
        }
        for item in all_qa_data
        if is_valid_qa_item(item)
    ]

    logger.info(f"Loaded {len(filtered_qa_data)} valid QA pairs after filtering")

    # Log breakdown by dataset
    dataset_counts = {}
    for item in filtered_qa_data:
        ds = item["dataset"]
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    logger.info(f"QA pairs by dataset: {dataset_counts}")

    # Loop through each model type and its configurations
    for model_type, models in model_config["models"].items():
        for model_key, model_kwargs in models.items():
            # If selected_models is provided, skip models not in the list
            if selected_models and model_key not in selected_models:
                continue
            try:
                clean_model_name = (
                    model_key.replace("@", "_").replace("/", "_").replace(":", "_")
                )
                experiment_name = f"dataset_gen_{clean_model_name}"
                if prefix:
                    experiment_name = f"{prefix}{experiment_name}"

                logger.info(f"Running evaluation for model: {model_key}")

                def process_query_with_model(question: str, base_kwargs: Dict) -> Dict:
                    dataset_name = None
                    for qa_pair in filtered_qa_data:
                        if qa_pair["question"] == question:
                            dataset_name = qa_pair["dataset"]
                            break

                    agent_kwargs = {
                        "database_type": "postgres",
                        "host": "localhost",
                        "port": 5432,
                        "database": dataset_name,
                        "username": "postgres",
                        "password": "postgres",
                        "tools": [
                            "postgres_execute_sql",
                            "postgres_list_schema",
                        ],
                    }

                    return process_single_query(
                        question, agent_kwargs, model_type, model_kwargs, dataset_name
                    )

                eval_result = run_evaluation(
                    experiment_name=experiment_name,
                    qa_pairs=filtered_qa_data,
                    process_query_fn=process_query_with_model,
                    csv_suffix=clean_model_name,
                    log_artifacts_fn=log_gauge_charts_and_metrics,
                )

                logger.info(
                    f"Evaluation completed for {model_key}. Check MLflow UI for detailed results."
                )

            except Exception as e:
                logger.error(f"Failed to evaluate model {model_key}: {str(e)}")
                # Continue to next model on error
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluations.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="List of model names to run (as in config). If not provided, runs all models.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="List of dataset names to use. If not provided, uses all datasets except those filtered out.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Optional prefix to add to experiment names in MLflow (e.g., 'val_' for validation experiments).",
    )
    args = parser.parse_args()
    run_all_model_evaluations(
        selected_models=args.models, selected_datasets=args.datasets, prefix=args.prefix
    )

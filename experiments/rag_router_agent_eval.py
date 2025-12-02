import json
import time
from typing import Dict

import dotenv
from experiments.common.evaluation import run_evaluation
from experiments.common.plot_utils import (
    create_simple_gauge_figure,
    log_gauge_charts_and_metrics,
)
from experiments.common.retry_utils import retry_with_backoff

from src.agents.sql_agent import create_sql_agent
from src.agents.sql_agent_rag_router import get_routed_model
from src.cost_calc.shared_store import get_shared_store
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model
from src.utils.logger import logger

dotenv.load_dotenv()


def process_single_query(question: str, base_kwargs: Dict) -> Dict:
    """Process a single query using the RAG router and SQL agent, with retry and tool call extraction"""
    start_time = time.time()
    input_cost = 0
    output_cost = 0
    model_config = "failed"
    routed_model_name = "failed"
    agent = None

    def stream_and_process():
        nonlocal agent, input_cost, output_cost, model_config, routed_model_name
        model_type, model_config = get_routed_model(query=question)
        # Extract just the model name from the config
        routed_model_name = model_config.get("model_name") or model_config.get(
            "model", "unknown"
        )
        logger.info(f"Routed question to model: {model_config}")
        routed_model = create_model(model=model_type, model_kwargs=model_config)
        # Use the agent_kwargs passed from the main function
        agent_kwargs = base_kwargs
        agent = create_sql_agent(
            model=routed_model,
            input_query=question,
            **agent_kwargs,
        )
        stream_generator = agent.stream(
            {"messages": [("human", question)]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="updates",
        )
        return process_agent_stream_with_store(
            stream_generator,
            store=get_shared_store(),
            agent_name="rag_router_sql_agent",
            model_name=routed_model_name,
        )

    try:
        result = retry_with_backoff(stream_and_process, max_retries=3, initial_delay=1)
        execution_time = time.time() - start_time

        if result:
            responses, input_cost, output_cost, _, cost_tracker = result
            input_cost = float(input_cost) if input_cost is not None else 0
            output_cost = float(output_cost) if output_cost is not None else 0
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
        # Ensure routed_model_name is set even on error
        if isinstance(model_config, dict):
            routed_model_name = model_config.get("model_name") or model_config.get(
                "model", "error"
            )
        else:
            routed_model_name = "error"
        logger.error(f"Error processing query: {question}, error: {str(e)}")
    finally:
        if agent and hasattr(agent, "db_client"):
            try:
                agent.db_client.close()
                logger.debug("Closed database connection")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
    return {
        "actual_answer": final_response,
        "query_cost": float(input_cost + output_cost),
        "execution_time": execution_time,
        "routed_model": routed_model_name,
    }


def is_valid_qa_item(item, target_dataset="atis"):
    return (
        all(
            item.get(key) and item.get(key).strip()
            for key in ["question", "answer", "dataset"]
        )
        and item.get("dataset") == target_dataset
        and item.get("answer") not in [None, "success", "No results"]
    )


def main(datasets=None, experiment_name_suffix=""):
    """Main evaluation pipeline with configurable datasets"""
    if datasets is None:
        datasets = ["atis"]

    with open("experiments/test_sets/comprehensive_qa.json", "r") as f:
        all_qa_data = json.load(f)

    # Filter for multiple datasets
    filtered_qa_data = []
    for dataset in datasets:
        dataset_qa = [
            {
                "question": item["question"],
                "answer": item["answer"],
                "dataset": item["dataset"],
            }
            for item in all_qa_data
            if is_valid_qa_item(item, target_dataset=dataset)
        ]
        filtered_qa_data.extend(dataset_qa)

    logger.info(
        f"Loaded {len(filtered_qa_data)} valid QA pairs after filtering from datasets: {datasets}"
    )

    # Create experiment name based on datasets
    if len(datasets) == 1:
        exp_name = f"rag_router_evaluation_{datasets[0]}"
    else:
        exp_name = f"rag_router_evaluation_multi"

    if experiment_name_suffix:
        exp_name += f"_{experiment_name_suffix}"

    def process_query_with_database(question: str, base_kwargs: Dict) -> Dict:
        # Find the dataset name for this specific question
        dataset_name = None
        for qa_pair in filtered_qa_data:
            if qa_pair["question"] == question:
                dataset_name = qa_pair["dataset"]
                break

        if dataset_name is None:
            dataset_name = "atis"  # fallback default

        # Create agent_kwargs with the correct database name
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

        return process_single_query(question, agent_kwargs)

    results = run_evaluation(
        experiment_name=exp_name,
        qa_pairs=filtered_qa_data,
        process_query_fn=process_query_with_database,
        log_artifacts_fn=log_gauge_charts_and_metrics,
    )
    return results


if __name__ == "__main__":
    main()

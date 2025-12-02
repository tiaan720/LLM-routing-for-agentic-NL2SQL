import json
import time
from typing import Dict, Literal

import dotenv
from experiments.common.evaluation import run_evaluation
from experiments.common.plot_utils import (
    create_simple_gauge_figure,
    log_gauge_charts_and_metrics,
)
from experiments.common.retry_utils import retry_with_backoff

from src.agents.supervisor_agent import create_supervisor_agent
from src.cost_calc.shared_store import (
    finalize_current_session,
    get_shared_store,
    reset_shared_store,
)
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model

dotenv.load_dotenv()


def process_single_query(
    question: str,
    base_kwargs: dict,
    model_name: str = "gpt-5",
    model_provider: Literal["openai"] = "openai",
):
    """Process a single query using the Supervisor agent"""
    start_time = time.time()
    input_cost = 0
    output_cost = 0
    selected_model = "failed"

    def stream_and_process():
        nonlocal input_cost, output_cost, selected_model
        agent_config = {
            "model": create_model(
                model_provider,
                {
                    "model_name": model_name,
                    "temperature": 1,
                    "max_tokens": None,
                    "max_retries": 5,
                    "stop": None,
                },
            ),
            "model_name": model_name,
        }
        agent_kwargs = {
            "database_type": "postgres",
            "host": "localhost",
            "port": 5432,
            "database": base_kwargs.get("database", "academic"),
            "username": "postgres",
            "password": "postgres",
            "tools": [
                "sql_agent_as_tool",
            ],
        }
        agent = create_supervisor_agent(
            model=agent_config["model"],
            input_query=question,
            **agent_kwargs,
        )
        result = process_agent_stream_with_store(
            agent.stream(
                {"messages": [("human", question)]},
                {"configurable": {"thread_id": "1"}},
                stream_mode="updates",
            ),
            store=get_shared_store(),
            agent_name="supervisor_agent",
            model_name=agent_config["model"].model_name,
        )
        return result

    try:
        result = retry_with_backoff(stream_and_process, max_retries=3, initial_delay=1)
        execution_time = time.time() - start_time

        if result:
            responses, input_cost, output_cost, selected_model, cost_tracker = result
            input_cost = input_cost or 0
            output_cost = output_cost or 0

            # Get total costs from store for tracking
            total_input_cost, total_output_cost, total_cost = (
                cost_tracker.get_total_costs()
            )

            # Finalize session (logs comprehensive summary internally)
            cost_tracker.finalize_session()

            # Extract tool calls and final response for supervisor agent - simplified
            tool_call_messages = []
            tool_responses = []
            final_response = None
            selected_model_from_tool = "failed"

            for response in responses:
                if (
                    "agent" in response
                    and "messages" in response["agent"]
                    and response["agent"]["messages"]
                ):
                    for message in response["agent"]["messages"]:
                        # Extract tool calls to get selected model
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                tool_call_messages.append(
                                    f"{{'name': '{tool_call['name']}', 'args': {tool_call['args']}, 'id': '{tool_call['id']}', 'type': '{tool_call['type']}'}}"
                                )
                                # Extract selected model from tool call
                                if (
                                    tool_call["name"] == "sql_agent_as_tool"
                                    and "model_name" in tool_call["args"]
                                ):
                                    selected_model_from_tool = tool_call["args"][
                                        "model_name"
                                    ]

                        # Extract tool responses (contains SQL queries and results)
                        if (
                            hasattr(message, "type")
                            and message.type == "tool"
                            and hasattr(message, "content")
                        ):
                            if message.content and message.content.strip():
                                tool_responses.append(message.content)

                        # Extract final AI response
                        if (
                            hasattr(message, "type")
                            and message.type == "ai"
                            and hasattr(message, "content")
                        ):
                            if message.content and message.content.strip():
                                final_response = message.content

            # Override selected_model with the model selected by supervisor
            selected_model = selected_model_from_tool

            # Build complete response with all components for SQL evaluation
            response_parts = []

            # Start with final AI response if it exists and doesn't have tool calls
            if final_response and not (
                hasattr(final_response, "tool_calls")
                and getattr(final_response, "tool_calls")
            ):
                response_parts.append(final_response)

            # Add tool calls section
            if tool_call_messages:
                response_parts.append("\nTool Calls:")
                response_parts.extend(tool_call_messages)

            # Add tool responses section (this is crucial for SQL evaluation)
            if tool_responses:
                response_parts.append("\nTool Responses:")
                response_parts.extend(tool_responses)

            # If we only have tool responses but no final response, use the last tool response as the main answer
            if not final_response and tool_responses:
                response_parts.insert(0, tool_responses[-1])

            final_response = (
                "\n".join(response_parts)
                if response_parts
                else "No response generated from the model"
            )
        else:
            final_response = "No result from agent stream"

    except Exception as e:
        execution_time = time.time() - start_time
        final_response = f"Error executing query: {str(e)}"
        model_name = "failed"

    return {
        "actual_answer": final_response,
        "query_cost": float(input_cost + output_cost),
        "execution_time": execution_time,
        "supervisor_model": model_name,
        "selected_model": selected_model,
    }


def is_valid_qa_item(item, target_dataset="academic"):
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
        datasets = ["academic"]

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

    print(
        f"Loaded {len(filtered_qa_data)} valid QA pairs after filtering from datasets: {datasets}"
    )

    # Create experiment name based on datasets
    if len(datasets) == 1:
        exp_name = f"supervisor_agent_evaluation_{datasets[0]}"
    else:
        exp_name = f"supervisor_agent_evaluation_multi"

    if experiment_name_suffix:
        exp_name += f"_{experiment_name_suffix}"

    def process_query_with_database(question: str, base_kwargs: Dict) -> Dict:
        # Reset store for each query to ensure clean cost tracking
        reset_shared_store()

        # Find the dataset name for this specific question
        dataset_name = None
        for qa_pair in filtered_qa_data:
            if qa_pair["question"] == question:
                dataset_name = qa_pair["dataset"]
                break

        if dataset_name is None:
            dataset_name = "academic"  # fallback default for supervisor

        # Create base_kwargs with the correct database name
        query_kwargs = {"database": dataset_name}

        result = process_single_query(question, query_kwargs)

        # Finalize session after processing
        finalize_current_session()

        return result

    results = run_evaluation(
        experiment_name=exp_name,
        qa_pairs=filtered_qa_data,
        process_query_fn=process_query_with_database,
        log_artifacts_fn=log_gauge_charts_and_metrics,
    )
    return results


if __name__ == "__main__":
    main()

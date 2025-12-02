import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

import dotenv
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Suppress verbose logging
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
os.environ["TQDM_DISABLE"] = "1"

dotenv.load_dotenv()

# Create output directory for plots
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def clean_model_name(model_name):
    """Clean model name by removing location suffixes and other unwanted parts"""
    if not model_name:
        return model_name

    # Remove common location patterns
    location_patterns = [
        r"_us-east\d+$",
        r"_us-west\d+$",
        r"_us-central\d+$",
        r"_europe-west\d+$",
        r"_europe-central\d+$",
        r"_asia-\w+\d*$",
        r"@us-east\d+$",
        r"@us-west\d+$",
        r"@us-central\d+$",
        r"@europe-west\d+$",
        r"@europe-central\d+$",
        r"@asia-\w+\d*$",
    ]

    cleaned_name = model_name
    for pattern in location_patterns:
        cleaned_name = re.sub(pattern, "", cleaned_name)

    # Clean up specific patterns
    cleaned_name = cleaned_name.replace(
        "meta_llama-4-maverick-17b-128e-instruct-maas", "llama-4-maverick-17b"
    )
    cleaned_name = cleaned_name.replace("claude-3-5-haiku_20241022", "claude-3.5-haiku")
    cleaned_name = cleaned_name.replace(
        "claude-3-7-sonnet_20250219", "claude-3.7-sonnet"
    )
    cleaned_name = cleaned_name.replace("claude-opus-4_20250514", "claude-opus-4")
    cleaned_name = cleaned_name.replace("claude-sonnet-4_20250514", "claude-sonnet-4")
    cleaned_name = cleaned_name.replace("mistral-small-2503_001", "mistral-small-2503")

    return cleaned_name


def get_router_validation_data(experiment_name):
    """Get validation data for a specific experiment from MLflow"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    router_data = {
        "total_cost": 0,
        "total_time": 0,
        "num_queries": 0,
        "sql_accuracy": 0,
        "answer_correctness": 0,
        "faithfulness": 0,
        "model_usage": {},
        "raw_runs": [],
    }

    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name], filter_string="tags.val = 'val'"
        )

        if runs.empty:
            print(f"No runs with tag 'val:val' found in experiment '{experiment_name}'")
            return router_data

        client = mlflow.tracking.MlflowClient()
        for _, run in runs.iterrows():
            router_data["raw_runs"].append(run.to_dict())

            # Get metrics from the run itself
            router_data["sql_accuracy"] += run.get("metrics.sql_accuracy/v1/mean", 0)
            router_data["answer_correctness"] += run.get(
                "metrics.answer_correctness/v1/mean", 0
            )
            router_data["faithfulness"] += run.get("metrics.faithfulness/v1/mean", 0)

            # Try to load detailed evaluation results - EXACT COPY FROM model_analysis_plots.py
            try:
                artifacts = client.list_artifacts(run["run_id"])

                for artifact in artifacts:
                    if artifact.path == "eval_results_table.json":
                        local_path = client.download_artifacts(
                            run["run_id"], "eval_results_table.json"
                        )
                        with open(local_path, "r") as f:
                            eval_data = json.load(f)

                        if "data" in eval_data and eval_data["data"]:
                            costs = [
                                float(row[1])
                                for row in eval_data["data"]
                                if len(row) >= 3 and row[1] is not None
                            ]
                            times = [
                                float(row[2])
                                for row in eval_data["data"]
                                if len(row) >= 3 and row[2] is not None
                            ]

                            # Extract model usage - check for supervisor router specifically
                            models = []
                            if "columns" in eval_data and eval_data["columns"]:
                                # For supervisor router, look for selected_model column
                                if "supervisor" in experiment_name.lower():
                                    try:
                                        selected_model_idx = eval_data["columns"].index(
                                            "selected_model"
                                        )
                                        models = [
                                            clean_model_name(
                                                str(row[selected_model_idx])
                                            )
                                            for row in eval_data["data"]
                                            if len(row) > selected_model_idx
                                            and row[selected_model_idx] is not None
                                        ]
                                    except (ValueError, IndexError):
                                        # Fallback to column 3 if selected_model not found
                                        models = [
                                            clean_model_name(str(row[3]))
                                            for row in eval_data["data"]
                                            if len(row) >= 4 and row[3] is not None
                                        ]
                                else:
                                    # For other routers, use column 3
                                    models = [
                                        clean_model_name(str(row[3]))
                                        for row in eval_data["data"]
                                        if len(row) >= 4 and row[3] is not None
                                    ]
                            else:
                                # Fallback to column 3 if no column info
                                models = [
                                    clean_model_name(str(row[3]))
                                    for row in eval_data["data"]
                                    if len(row) >= 4 and row[3] is not None
                                ]

                            if costs:
                                router_data["total_cost"] += sum(costs)
                                router_data["total_time"] += sum(times)
                                router_data["num_queries"] += len(costs)

                                # Count model usage
                                for model in models:
                                    router_data["model_usage"][model] = (
                                        router_data["model_usage"].get(model, 0) + 1
                                    )
                        break

                if router_data["num_queries"] == 0:
                    # Fallback if no detailed data found
                    router_data["num_queries"] += 1
                    router_data["total_cost"] += run.get("metrics.total_cost", 0)
                    router_data["total_time"] += run.get("metrics.total_time", 0)

            except Exception as e:
                print(f"Could not process artifacts for run {run['run_id']}: {e}")
                # Fallback to top-level metrics if artifact processing fails
                router_data["num_queries"] += 1
                router_data["total_cost"] += run.get("metrics.total_cost", 0)
                router_data["total_time"] += run.get("metrics.total_time", 0)

    except Exception as e:
        print(f"Error processing experiment {experiment_name}: {e}")

    # Calculate averages
    if router_data["num_queries"] > 0:
        # Metrics from runs are already averaged, so if we have multiple runs, we average them
        num_runs = len(router_data["raw_runs"])
        if num_runs > 0:
            router_data["sql_accuracy"] /= num_runs
            router_data["answer_correctness"] /= num_runs
            router_data["faithfulness"] /= num_runs

        router_data["avg_cost_per_query"] = (
            router_data["total_cost"] / router_data["num_queries"]
        )
        router_data["avg_time_per_query"] = (
            router_data["total_time"] / router_data["num_queries"]
        )
    else:
        router_data["avg_cost_per_query"] = 0
        router_data["avg_time_per_query"] = 0

    return router_data


def get_model_usage_from_router_logs(experiment_name):
    """This function is deprecated and its logic has been moved into get_router_validation_data."""
    warnings.warn(
        "get_model_usage_from_router_logs is deprecated and will be removed in a future version. "
        "Model usage is now calculated within get_router_validation_data.",
        DeprecationWarning,
    )
    return {}


def print_model_distribution(model_usage, experiment_name):
    """Print the distribution of models used by the experiment"""
    if not model_usage:
        print(f"\n{experiment_name} - Model Usage Distribution:")
        print("No model usage data found in MLflow artifacts or tags")
        return

    total_selections = sum(model_usage.values())
    print(f"\n{experiment_name} - Model Usage Distribution:")
    print(f"Total queries processed: {total_selections}")
    print("-" * 50)

    # Sort by usage count
    sorted_usage = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)

    for model, count in sorted_usage:
        percentage = (count / total_selections) * 100 if total_selections > 0 else 0
        print(f"{model:<30} {count:>6} ({percentage:>5.1f}%)")


def create_router_radar_chart(router_data, experiment_name):
    """Create a radar chart for experiment performance metrics - DISABLED"""
    # Radar chart functionality removed as requested
    pass


def calculate_router_efficiency_score(router_data):
    """Calculate a composite efficiency score for the router"""
    if router_data["num_queries"] == 0:
        return 0

    # Weighted score combining accuracy, cost efficiency, and speed
    accuracy_score = (
        router_data["sql_accuracy"] * 0.4
        + router_data["answer_correctness"] * 0.4
        + router_data["faithfulness"] * 0.2
    )

    # Cost efficiency (lower cost is better, normalize to 0-1 scale)
    max_reasonable_cost = 0.01  # $0.01 per query as reasonable max
    cost_efficiency = max(
        0, 1 - (router_data["avg_cost_per_query"] / max_reasonable_cost)
    )

    # Speed efficiency (lower time is better, normalize to 0-1 scale)
    max_reasonable_time = 10  # 10 seconds as reasonable max
    speed_efficiency = max(
        0, 1 - (router_data["avg_time_per_query"] / max_reasonable_time)
    )

    # Combined score
    efficiency_score = (
        accuracy_score * 0.6 + cost_efficiency * 0.2 + speed_efficiency * 0.2
    )

    return efficiency_score


def analyze_router_decision_quality(experiment_name):
    """Analyze the quality of router decisions by comparing with optimal choices"""
    print(f"\n{experiment_name} - Decision Quality Analysis:")
    print("=" * 50)

    # Get validation data to analyze decision patterns
    router_data = get_router_validation_data(experiment_name)

    if router_data["num_queries"] == 0:
        print("No validation data available for decision quality analysis")
        return {"efficiency_score": 0, "cost_per_accuracy": 0, "time_per_accuracy": 0}

    # Calculate efficiency score
    efficiency_score = calculate_router_efficiency_score(router_data)
    print(f"Overall Efficiency Score: {efficiency_score:.3f}")

    # Analyze cost vs performance trade-offs
    cost_per_accuracy = router_data["avg_cost_per_query"] / max(
        router_data["sql_accuracy"], 0.001
    )
    print(f"Cost per Accuracy Point: ${cost_per_accuracy:.6f}")

    # Analyze speed vs performance trade-offs
    time_per_accuracy = router_data["avg_time_per_query"] / max(
        router_data["sql_accuracy"], 0.001
    )
    print(f"Time per Accuracy Point: {time_per_accuracy:.2f} seconds")

    return {
        "efficiency_score": efficiency_score,
        "cost_per_accuracy": cost_per_accuracy,
        "time_per_accuracy": time_per_accuracy,
    }


def create_router_comparison_metrics(
    router_data, experiment_name, baseline_metrics=None
):
    """Create comparison metrics against baseline (e.g., best individual model)"""
    if router_data["num_queries"] == 0:
        print(f"No data available for {experiment_name} comparison")
        return

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "SQL Accuracy",
                "Answer Correctness",
                "Faithfulness",
                "Avg Cost per Query ($)",
                "Avg Time per Query (s)",
                "Total Queries Processed",
            ],
            experiment_name: [
                f"{router_data['sql_accuracy']:.3f}",
                f"{router_data['answer_correctness']:.3f}",
                f"{router_data['faithfulness']:.3f}",
                f"${router_data['avg_cost_per_query']:.6f}",
                f"{router_data['avg_time_per_query']:.2f}",
                f"{router_data['num_queries']}",
            ],
        }
    )

    if baseline_metrics:
        metrics_df["Baseline"] = [
            f"{baseline_metrics.get('sql_accuracy', 0):.3f}",
            f"{baseline_metrics.get('answer_correctness', 0):.3f}",
            f"{baseline_metrics.get('faithfulness', 0):.3f}",
            f"${baseline_metrics.get('avg_cost_per_query', 0):.6f}",
            f"{baseline_metrics.get('avg_time_per_query', 0):.2f}",
            f"{baseline_metrics.get('num_queries', 0)}",
        ]

        # Calculate improvement
        improvements = []
        for i, metric in enumerate(
            ["sql_accuracy", "answer_correctness", "faithfulness"]
        ):
            if metric in baseline_metrics and baseline_metrics[metric] > 0:
                improvement = (
                    (router_data[metric] - baseline_metrics[metric])
                    / baseline_metrics[metric]
                ) * 100
                improvements.append(f"{improvement:+.1f}%")
            else:
                improvements.append("N/A")

        # For cost and time, lower is better
        for metric in ["avg_cost_per_query", "avg_time_per_query"]:
            if metric in baseline_metrics and baseline_metrics[metric] > 0:
                improvement = (
                    (baseline_metrics[metric] - router_data[metric])
                    / baseline_metrics[metric]
                ) * 100
                improvements.append(f"{improvement:+.1f}%")
            else:
                improvements.append("N/A")

        improvements.append("N/A")  # For total queries
        metrics_df["Improvement"] = improvements

    print(f"\n{experiment_name} - Performance Summary:")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    return metrics_df


def validate_experiment_performance(experiment_name, baseline_metrics=None):
    """Complete validation analysis for an experiment"""
    print(f"\n{'='*60}")
    print(f"VALIDATION ANALYSIS FOR {experiment_name.upper()}")
    print(f"{'='*60}")

    # Get router validation data
    router_data = get_router_validation_data(experiment_name)

    if router_data["num_queries"] == 0:
        print(f"No validation data found for {experiment_name}")
        print("Ensure MLflow runs are tagged with 'val:val'")
        return None

    # Get model usage from logs/artifacts
    model_usage = get_model_usage_from_router_logs(experiment_name)
    if not model_usage and router_data["model_usage"]:
        model_usage = router_data["model_usage"]

    # Print model distribution
    print_model_distribution(model_usage, experiment_name)

    # Create radar chart
    create_router_radar_chart(router_data, experiment_name)

    # Analyze decision quality
    decision_analysis = analyze_router_decision_quality(experiment_name)

    # Create comparison metrics
    metrics_df = create_router_comparison_metrics(
        router_data, experiment_name, baseline_metrics
    )

    # Additional validation insights
    print(f"\n{experiment_name} - Additional Validation Insights:")
    print("-" * 50)

    # Model diversity
    if model_usage:
        unique_models = len(model_usage)
        print(f"Model Diversity: Used {unique_models} different models")

        # Calculate entropy of model selection
        total_selections = sum(model_usage.values())
        if total_selections > 0:
            import math

            entropy = -sum(
                (count / total_selections) * math.log2(count / total_selections)
                for count in model_usage.values()
                if count > 0
            )
            max_entropy = math.log2(unique_models) if unique_models > 0 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            print(
                f"Selection Entropy: {entropy:.3f} (normalized: {normalized_entropy:.3f})"
            )

    # Performance consistency
    if len(router_data["raw_runs"]) > 1:
        accuracies = []
        for run in router_data["raw_runs"]:
            if "metrics.sql_accuracy" in run and pd.notna(run["metrics.sql_accuracy"]):
                accuracies.append(run["metrics.sql_accuracy"])

        if len(accuracies) > 1:
            std_dev = pd.Series(accuracies).std()
            print(f"Accuracy Consistency: σ = {std_dev:.4f} (lower is more consistent)")

    return {
        "router_data": router_data,
        "model_usage": model_usage,
        "decision_analysis": decision_analysis,
        "metrics_df": metrics_df,
    }


# Main function that accepts experiment name
def validate_experiment(experiment_name, baseline_metrics=None):
    """Validate performance of any experiment by name"""
    return validate_experiment_performance(experiment_name, baseline_metrics)


if __name__ == "__main__":
    # Example usage
    print("Experiment Validation Analysis")
    print("=" * 30)

    # You can run individual experiment validations by name
    # validate_experiment("matrix_router_eval")
    # validate_experiment("rag_router_eval")
    # validate_experiment("supervisor_router_eval")

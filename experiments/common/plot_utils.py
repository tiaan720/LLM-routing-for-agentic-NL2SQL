import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logger import logger


def create_simple_gauge_figure(eval_result, experiment_name: str):
    """
    Create a simple gauge chart for the three key evaluation metrics.

    Args:
        eval_result: MLflow evaluation result object
        experiment_name: Name of the experiment

    Returns:
        Plotly figure object or None if no metrics found
    """
    target_metrics = [
        "sql_accuracy/v1/mean",
        "answer_correctness/v1/mean",
        "faithfulness/v1/mean",
    ]
    metrics = {}

    if hasattr(eval_result, "metrics") and eval_result.metrics:
        for metric_name, metric_value in eval_result.metrics.items():
            if metric_name in target_metrics and isinstance(metric_value, (int, float)):
                # Clean up metric names for display
                display_name = metric_name.split("/")[0].replace("_", " ").title()
                metrics[display_name] = round(metric_value, 3)

    if not metrics:
        logger.warning("No target metrics found for gauge chart")
        return None

    # Create simple 1x3 subplot layout
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        specs=[[{"type": "indicator"}] * len(metrics)],
        subplot_titles=list(metrics.keys()),
    )

    # Add gauge for each metric
    for i, (metric_name, value) in enumerate(metrics.items()):
        color = "green" if value >= 3 else "orange" if value >= 2 else "red"

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                gauge={
                    "axis": {"range": [0, 5]},
                    "bar": {"color": color},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                },
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        height=400, title_text=f"Key Metrics - {experiment_name}", title_x=0.5
    )

    return fig


def log_gauge_charts_and_metrics(
    eval_result, experiment_name: str, results: list, results_df
):
    """
    Create and log simple gauge chart for key metrics to MLflow.

    Args:
        eval_result: MLflow evaluation result object
        experiment_name: Name of the experiment
        results: List of result dictionaries
        results_df: DataFrame containing the results
    """
    import mlflow

    try:
        gauge_fig = create_simple_gauge_figure(eval_result, experiment_name)
        if gauge_fig:
            mlflow.log_figure(gauge_fig, "key_metrics_gauge.html")
            logger.info("Key metrics gauge chart logged to MLflow")
        if len(results) > 0:
            if "query_cost" in results_df.columns:
                mlflow.log_metric("avg_query_cost", results_df["query_cost"].mean())
                mlflow.log_metric("max_query_cost", results_df["query_cost"].max())

            if "execution_time" in results_df.columns:
                mlflow.log_metric(
                    "avg_execution_time", results_df["execution_time"].mean()
                )
                mlflow.log_metric(
                    "max_execution_time", results_df["execution_time"].max()
                )
            successful_queries = len(
                [r for r in results if not r["actual_answer"].startswith("Error")]
            )
            success_rate = (successful_queries / len(results)) * 100
            mlflow.log_metric("success_rate", success_rate)

    except Exception as e:
        logger.error(f"Failed to create gauge charts: {str(e)}")

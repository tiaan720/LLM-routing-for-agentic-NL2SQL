"""
Model Analysis Plotting Module

Reusable functions for analyzing and plotting model evaluation results from MLflow.
Supports different tag filters to analyze different sets of models.
"""

import base64
import json
import logging
import os
import re
import time
import warnings

import dotenv
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    """Clean model name by removing location suffixes"""
    if not model_name:
        return model_name

    # Remove location patterns in one pass
    location_pattern = (
        r"[@_](us-east|us-west|us-central|europe-west|europe-central|asia-\w*)\d*$"
    )
    cleaned_name = re.sub(location_pattern, "", model_name)

    # Remove common prefixes
    cleaned_name = cleaned_name.replace("datasetgen_", "").replace("val_", "")

    return cleaned_name


def get_models_data(tag_filter="tags.arena = 'arena'", dataset_name="arena"):
    """Get detailed data for models with specific tag filter"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    models_data = []
    experiments = mlflow.search_experiments()

    for exp in experiments:
        try:
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=tag_filter,
                order_by=["start_time DESC"],
            )

            if not runs.empty:
                for _, run in runs.iterrows():
                    raw_model_name = (
                        exp.name.replace("dataset_gen_", "")
                        if "dataset_gen_" in exp.name
                        else run.get("tags.mlflow.runName", run["run_id"])
                    )
                    model_name = clean_model_name(raw_model_name)

                    # Get metrics
                    metrics_data = {
                        "model_name": model_name,
                        "raw_model_name": raw_model_name,
                        "experiment_name": exp.name,
                        "run_id": run["run_id"],
                        "status": run["status"],
                        "sql_accuracy": run.get("metrics.sql_accuracy/v1/mean", 0),
                        "answer_correctness": run.get(
                            "metrics.answer_correctness/v1/mean", 0
                        ),
                        "faithfulness": run.get("metrics.faithfulness/v1/mean", 0),
                        "total_cost": 0,
                        "avg_cost_per_query": 0,
                        "total_time": 0,
                        "avg_time_per_query": 0,
                        "num_queries": 0,
                        "cost_data": [],
                        "time_data": [],
                    }

                    # Try to load detailed evaluation results
                    try:
                        client = mlflow.tracking.MlflowClient()
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

                                    if costs:
                                        metrics_data.update(
                                            {
                                                "total_cost": sum(costs),
                                                "avg_cost_per_query": sum(costs)
                                                / len(costs),
                                                "total_time": (
                                                    sum(times) if times else 0
                                                ),
                                                "avg_time_per_query": (
                                                    sum(times) / len(times)
                                                    if times
                                                    else 0
                                                ),
                                                "num_queries": len(eval_data["data"]),
                                                "cost_data": costs,
                                                "time_data": times,
                                            }
                                        )
                                break

                    except Exception:
                        pass  # Keep default values

                    models_data.append(metrics_data)

        except Exception:
            pass  # Skip problematic experiments

    return pd.DataFrame(models_data)


def find_pareto_frontier(costs, performance):
    """Find the Pareto frontier points - models that are not dominated by any other model"""
    if len(costs) != len(performance) or len(costs) < 2:
        return [], []

    points = list(zip(costs, performance))
    pareto_points = []
    pareto_indices = []

    for i, (cost_i, perf_i) in enumerate(points):
        is_pareto = True
        for j, (cost_j, perf_j) in enumerate(points):
            if i != j and (
                (cost_j < cost_i and perf_j >= perf_i)
                or (cost_j <= cost_i and perf_j > perf_i)
            ):
                is_pareto = False
                break

        if is_pareto:
            pareto_points.append((cost_i, perf_i))
            pareto_indices.append(i)

    # Sort by cost
    sorted_data = sorted(zip(pareto_points, pareto_indices), key=lambda x: x[0][0])
    pareto_points = [point for point, idx in sorted_data]
    pareto_indices = [idx for point, idx in sorted_data]

    print(
        f"Found {len(pareto_points)} Pareto optimal points out of {len(points)} total"
    )
    return pareto_points, pareto_indices


def create_bar_chart(df, column, title, color, format_func=None):
    """Create a standardized bar chart"""
    if format_func is None:
        format_func = lambda x: f"{x:.3f}"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["model_name"],
            y=df[column],
            text=[format_func(val) for val in df[column]],
            textposition="auto",
            marker_color=color,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=column.replace("_", " ").title(),
        height=500,
        width=1200,
        template="plotly_white",
        xaxis={"tickangle": 45},
        font=dict(size=14),
        title_font=dict(size=18),
    )
    filename = (
        title.replace(" ", "_")
        .replace("/", "_")
        .replace(":", "")
        .replace("-", "_")
        .lower()
        + ".pdf"
    )
    fig.write_image(
        os.path.join(output_dir, filename), format="pdf", width=1200, height=500
    )
    fig.show()


def create_combined_bar_chart(df, title_suffix=""):
    """Create a combined bar chart showing all metrics for each model"""
    if df.empty:
        return

    # Normalize metrics for comparison
    metrics = [
        "sql_accuracy",
        "answer_correctness",
        "faithfulness",
        "avg_cost_per_query",
        "avg_time_per_query",
    ]
    normalized_df = df.copy()
    fixed_scale_metrics = ["sql_accuracy", "answer_correctness", "faithfulness"]

    for metric in metrics:
        if metric in df.columns and df[metric].notna().any():
            if metric in fixed_scale_metrics:
                # Normalize from a 1-5 scale to a 0-1 scale
                normalized_df[metric] = (df[metric] - 1) / 4.0
            else:
                # Standard min-max scaling for other metrics
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    normalized_df[metric] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_df[metric] = (
                        0.5  # If all values are the same, set to 0.5
                    )
            # Fill any NaN values with 0
            normalized_df[metric] = normalized_df[metric].fillna(0.0)

    fig = go.Figure()

    # Add traces for each metric
    colors = ["lightgreen", "orange", "purple", "lightcoral", "lightblue"]
    labels = [
        "SQL Accuracy",
        "Answer Correctness",
        "Faithfulness",
        "Avg Cost (norm)",
        "Avg Time (norm)",
    ]

    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        fig.add_trace(
            go.Bar(
                x=normalized_df["model_name"],
                y=normalized_df[metric],
                name=label,
                marker_color=color,
                offsetgroup=i,
            )
        )

    fig.update_layout(
        title=f"Combined Metrics Comparison{title_suffix}",
        xaxis_title="Model",
        yaxis_title="Normalized Value (0-1)",
        barmode="group",
        height=800,  # Increased height for better visibility
        width=1400,  # Set width for better proportions
        template="plotly_white",
        xaxis={"tickangle": 45},
        legend_title="Metrics",
        margin=dict(t=100, b=100, l=80, r=80),  # Add margins for better spacing
        font=dict(size=14),
        title_font=dict(size=18),
    )
    filename = (
        f"combined_metrics_comparison{title_suffix}".replace(" ", "_")
        .replace("/", "_")
        .replace(":", "")
        .replace("-", "_")
        .lower()
        + ".pdf"
    )
    fig.write_image(
        os.path.join(output_dir, filename), format="pdf", width=1400, height=800
    )
    fig.show()


def create_radar_chart(df, title_suffix=""):
    """Create separate radar charts for each model in subplots"""
    if df.empty or len(df) < 1:
        return

    # Sort models by combined score for radar chart
    df = df.copy()
    df["radar_score"] = (
        df["sql_accuracy"] + df["answer_correctness"] + df["faithfulness"]
    ) / 3
    df = df.sort_values("radar_score", ascending=False).reset_index(drop=True)

    # Metrics to include in radar
    metrics = ["sql_accuracy", "answer_correctness", "faithfulness"]
    cost_metrics = ["avg_cost_per_query", "avg_time_per_query"]

    # Normalize all metrics to 0-1 scale
    normalized_df = df.copy()
    all_metrics = metrics + cost_metrics
    fixed_scale_metrics = ["sql_accuracy", "answer_correctness", "faithfulness"]

    for metric in all_metrics:
        if metric in df.columns and df[metric].notna().any():
            if metric in fixed_scale_metrics:
                # Normalize from a 1-5 scale to a 0-1 scale
                normalized_df[metric] = (df[metric] - 1) / 4.0
            else:
                # Standard min-max scaling for other metrics
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    normalized_df[metric] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_df[metric] = (
                        0.5  # If all values are the same, set to 0.5
                    )
            # Fill any NaN values with 0
            normalized_df[metric] = normalized_df[metric].fillna(0.0)

    # For cost and time, invert since lower is better
    for metric in cost_metrics:
        normalized_df[metric] = 1 - normalized_df[metric]
        # Fill any NaN values with 0 after inversion
        normalized_df[metric] = normalized_df[metric].fillna(0.0)

    # Radar categories
    categories = [
        "SQL Accuracy",
        "Answer Correctness",
        "Faithfulness",
        "Cost Efficiency",
        "Speed",
    ]
    categories_closed = categories + [categories[0]]

    # Determine subplot layout - max 3 columns, flexible rows
    n_models = len(normalized_df)
    cols = 3
    rows = (n_models + cols - 1) // cols  # Ceiling division

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=normalized_df["model_name"].tolist(),
        specs=[[{"type": "polar"} for _ in range(cols)] for _ in range(rows)],
        horizontal_spacing=0.15,  # Increase horizontal spacing
        vertical_spacing=0.1,  # Increased vertical spacing for better row separation
    )

    for i, (_, row) in enumerate(normalized_df.iterrows()):
        row_idx = i // cols + 1
        col_idx = i % cols + 1

        # Get original values for hover display
        orig_row = df.iloc[i]

        values = [
            row["sql_accuracy"],
            row["answer_correctness"],
            row["faithfulness"],
            row["avg_cost_per_query"],  # inverted
            row["avg_time_per_query"],  # inverted
        ]
        values += values[:1]  # Close the loop

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill="toself",
                name=row["model_name"],
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "SQL Accuracy: %{customdata[0]:.3f}<br>"
                + "Answer Correctness: %{customdata[1]:.3f}<br>"
                + "Faithfulness: %{customdata[2]:.3f}<br>"
                + "Avg Cost: $%{customdata[3]:.6f}<br>"
                + "Avg Time: %{customdata[4]:.2f}s<extra></extra>",
                customdata=[
                    [
                        orig_row["sql_accuracy"],
                        orig_row["answer_correctness"],
                        orig_row["faithfulness"],
                        orig_row["avg_cost_per_query"],
                        orig_row["avg_time_per_query"],
                    ]
                ],
            ),
            row=row_idx,
            col=col_idx,
        )

    fig.update_layout(
        title="",  # Removed main title to save space
        height=400 * rows,  # Increased height per row
        width=1600,  # Increased width for more horizontal space
        template="plotly_white",
        showlegend=False,
        margin=dict(
            t=130,
            b=50,
            l=120,  # Increased left margin for slight padding
            r=150,  # Increased right margin from 80 to 150 for more whitespace on the right
        ),  # Increased top margin for more space between title and charts
        font=dict(size=16),
        title_font=dict(size=24),
    )

    # Update polar axes with better spacing
    for i in range(1, rows * cols + 1):
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12),
                tickangle=0,
                gridcolor="darkgray",
                linecolor="darkgray",
            ),
            angularaxis=dict(
                tickfont=dict(size=16), gridcolor="darkgray", linecolor="darkgray"
            ),
            row=(i - 1) // cols + 1,
            col=(i - 1) % cols + 1,
        )

    # Move subplot titles up and increase font size
    for annotation in fig.layout.annotations:
        annotation.font.size = 18
    fig.update_annotations(yshift=30)

    filename = (
        f"model_comparison_radar_charts{title_suffix}".replace(" ", "_")
        .replace("/", "_")
        .replace(":", "")
        .replace("-", "_")
        .lower()
        + ".html"
    )
    html_path = os.path.join(output_dir, filename)
    fig.write_html(html_path)
    fig.show()

    # Use Selenium to increase font sizes and export to PDF
    pdf_filename = filename.replace(".html", ".pdf")
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Set up headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Load the HTML file
        driver.get(f"file://{os.path.abspath(html_path)}")

        # Wait for the plot to load and render
        driver.implicitly_wait(10)

        # Give extra time for Plotly to render
        time.sleep(3)

        # Inject CSS to increase font sizes
        css_injection = """
        var style = document.createElement('style');
        style.innerHTML = `
            /* General text sizing */
            .plotly-graph-div text {
                font-size: 22px !important;
            }
            /* Subplot titles */
            .subplot-title {
                font-size: 26px !important;
            }
            /* Angular axis labels (e.g., 'SQL Accuracy') */
            .angularaxistick text {
                font-size: 20px !important;
            }
            /* Radial axis labels (e.g., 0.2, 0.4, 0.6) */
            .xtick text {
                font-size: 13px !important;
            }
        `;
        document.head.appendChild(style);
        """

        driver.execute_script(css_injection)

        # Wait a bit for CSS to apply
        time.sleep(1)

        # Get the main plot container to determine the full size
        plot_container = driver.find_element(By.CSS_SELECTOR, ".plotly-graph-div")
        width_px = plot_container.size["width"]
        height_px = plot_container.size["height"]

        # Convert pixels to inches for PDF printing (assuming 96 DPI)
        # Add a small buffer to avoid clipping
        width_in = (width_px / 96) + 0.1
        height_in = (height_px / 96) + 0.1

        # Print to PDF
        pdf_result = driver.execute_cdp_cmd(
            "Page.printToPDF",
            {
                "paperWidth": width_in,
                "paperHeight": height_in,
                "printBackground": True,
                "marginTop": 0,
                "marginBottom": 0,
                "marginLeft": 0,
                "marginRight": 0,
            },
        )

        # Decode and save PDF
        pdf_data = base64.b64decode(pdf_result["data"])
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)

        print(f"Radar chart saved as PDF with increased fonts: {pdf_path}")

    finally:
        driver.quit()


def create_individual_charts(df, title_suffix=""):
    """Create charts for each metric"""
    if df.empty:
        print("No models found!")
        return

    df_filtered = df[df["num_queries"] > 0].copy()
    if df_filtered.empty:
        print("No models with evaluation data found!")
        return

    # Create combined bar chart first
    create_combined_bar_chart(df_filtered, title_suffix)

    # Create radar chart for multi-dimensional comparison
    create_radar_chart(df_filtered, title_suffix)

    # Create bar charts for main metrics
    create_bar_chart(
        df_filtered,
        "total_cost",
        f"Total Cost per Model{title_suffix}",
        "lightcoral",
        lambda x: f"${x:.4f}",
    )
    create_bar_chart(
        df_filtered,
        "avg_time_per_query",
        f"Average Query Speed per Model{title_suffix}",
        "lightblue",
        lambda x: f"{x:.2f}s",
    )
    create_bar_chart(
        df_filtered,
        "sql_accuracy",
        f"SQL Accuracy by Model{title_suffix}",
        "lightgreen",
    )
    create_bar_chart(
        df_filtered,
        "answer_correctness",
        f"Answer Correctness by Model{title_suffix}",
        "orange",
    )
    create_bar_chart(
        df_filtered, "faithfulness", f"Faithfulness by Model{title_suffix}", "purple"
    )

    # Cost vs Speed scatter plot
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=df_filtered["avg_cost_per_query"],
            y=df_filtered["avg_time_per_query"],
            mode="markers+text",
            text=df_filtered["model_name"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=15,
                color=df_filtered["sql_accuracy"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="SQL Accuracy"),
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>Avg Cost: $%{x:.6f}<br>Avg Time: %{y:.2f}s<br>SQL Accuracy: %{marker.color:.3f}<extra></extra>",
        )
    )

    fig_scatter.update_layout(
        title=f"Cost vs Speed Trade-off{title_suffix}",
        xaxis_title="Average Cost per Query ($)",
        yaxis_title="Average Time per Query (seconds)",
        height=650,
        width=950,
        template="plotly_white",
        font=dict(size=14),
        title_font=dict(size=18),
    )
    filename = (
        f"cost_vs_speed_trade_off{title_suffix}".replace(" ", "_")
        .replace("/", "_")
        .replace(":", "")
        .replace("-", "_")
        .lower()
        + ".pdf"
    )
    fig_scatter.write_image(
        os.path.join(output_dir, filename), format="pdf", width=950, height=650
    )
    fig_scatter.show()

    # Pareto analysis - using weights based on experiment_dataset_gen_arena.py
    # Weights: query_cost: 0.20, execution_time: 0.10, sql_accuracy: 0.10,
    #          answer_correctness: 0.40, faithfulness: 0.10, arena_hard_score: 0.0 (excluded for validation)
    # Total quality metrics weight: 0.60, Cost/Time weight: 0.30, Arena: 0.0

    # Normalize cost and time (lower is better, so invert)
    cost_norm = 1 - (df_filtered["total_cost"] - df_filtered["total_cost"].min()) / (
        df_filtered["total_cost"].max() - df_filtered["total_cost"].min() + 1e-10
    )
    time_norm = 1 - (
        df_filtered["avg_time_per_query"] - df_filtered["avg_time_per_query"].min()
    ) / (
        df_filtered["avg_time_per_query"].max()
        - df_filtered["avg_time_per_query"].min()
        + 1e-10
    )

    # Calculate combined performance with arena_hard_score excluded (weight = 0)
    df_filtered["combined_performance"] = (
        df_filtered["sql_accuracy"] * 0.10  # sql_accuracy weight
        + df_filtered["answer_correctness"] * 0.40  # answer_correctness weight
        + df_filtered["faithfulness"] * 0.10  # faithfulness weight
        + cost_norm * 0.20  # query_cost weight
        + time_norm * 0.10  # execution_time weight
        # arena_hard_score: 0.10 weight excluded (not available for validation)
    )

    pareto_points, pareto_indices = find_pareto_frontier(
        df_filtered["total_cost"].tolist(), df_filtered["combined_performance"].tolist()
    )

    # Create Pareto plot
    fig_pareto = go.Figure()

    # Calculate text positions to avoid overlaps on log scale
    # Stagger text positions alternating between top and bottom for closely clustered models
    df_filtered_sorted = df_filtered.sort_values("total_cost").reset_index(drop=True)
    text_positions_all = []
    for i in range(len(df_filtered_sorted)):
        # Alternate positions for models close together on log scale
        if i % 2 == 0:
            text_positions_all.append("top center")
        else:
            text_positions_all.append("bottom center")

    # Map back to original dataframe order
    text_positions_mapped = [
        text_positions_all[list(df_filtered_sorted.index).index(i)]
        for i in df_filtered.index
    ]

    # Hide labels for Pareto optimal models in the "All Models" trace
    # They'll get their own labels in the "Pareto Optimal" trace
    text_labels = []
    for i in range(len(df_filtered)):
        if i in pareto_indices:
            text_labels.append("")  # Empty string for Pareto optimal models
        else:
            text_labels.append(df_filtered.iloc[i]["model_name"])

    # All models - show text labels only for non-Pareto models
    fig_pareto.add_trace(
        go.Scatter(
            x=df_filtered["total_cost"],
            y=df_filtered["combined_performance"],
            mode="markers+text",
            text=text_labels,  # Use filtered labels (empty for Pareto models)
            textposition=text_positions_mapped,
            textfont=dict(size=18, color="darkblue"),  # Increased from 9 to 18
            marker=dict(
                size=15,  # Increased from 12 to 15
                color="lightblue",
                opacity=0.7,
                line=dict(width=2, color="darkblue"),
            ),
            hovertemplate="<b>%{customdata}</b><br>Cost: $%{x:.4f}<br>Performance: %{y:.3f}<extra></extra>",
            customdata=df_filtered[
                "model_name"
            ],  # Use customdata for hover to show all names
            name="All Models",
        )
    )

    # Pareto frontier
    if len(pareto_points) >= 2:
        pareto_costs, pareto_perfs = zip(*pareto_points)

        fig_pareto.add_trace(
            go.Scatter(
                x=pareto_costs,
                y=pareto_perfs,
                mode="lines+markers",
                line=dict(color="red", width=4),  # Increased from 3 to 4
                marker=dict(size=12, color="red"),  # Increased from 8 to 12
                name="Pareto Frontier",
            )
        )

        # Highlight optimal models with clearer labels
        pareto_df = df_filtered.iloc[pareto_indices].copy()

        # Calculate smart text positions for Pareto optimal models
        pareto_text_positions = []
        for i in range(len(pareto_df)):
            # Alternate between different positions to reduce overlap
            positions = [
                "top right",
                "bottom right",
                "top left",
                "bottom left",
                "top center",
                "bottom center",
            ]
            pareto_text_positions.append(positions[i % len(positions)])

        fig_pareto.add_trace(
            go.Scatter(
                x=pareto_df["total_cost"],
                y=pareto_df["combined_performance"],
                mode="markers+text",
                text=pareto_df["model_name"],
                textposition=pareto_text_positions,
                textfont=dict(
                    size=20, color="darkred", family="Arial Black"
                ),  # Increased from 10 to 20
                marker=dict(
                    size=22,  # Increased from 16 to 22
                    color="#FFD700",
                    symbol="star",
                    line=dict(width=3, color="darkred"),  # Increased from 2 to 3
                ),
                name="Pareto Optimal",
                hovertemplate="<b>%{text}</b><br>Cost: $%{x:.4f}<br>Performance: %{y:.3f}<extra></extra>",
            )
        )

        print(f"Pareto optimal models: {', '.join(pareto_df['model_name'].tolist())}")

    # Calculate better x-axis range for log scale with more spacing
    min_cost = df_filtered["total_cost"].min()
    max_cost = df_filtered["total_cost"].max()

    # For proper logarithmic spacing, we need actual cost values
    log_min = np.log10(max(min_cost, 0.0001))  # Avoid log(0)
    log_max = np.log10(max_cost)

    # Generate proper logarithmic tick values with less aggressive spacing
    # We want ticks like: 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, etc.
    tick_values = []

    # Start from 0.01 and work up with standard 1-2-5 pattern
    multipliers = [1, 2, 5]
    magnitudes = [0.01, 0.1, 1, 10, 100]

    for magnitude in magnitudes:
        for multiplier in multipliers:
            tick = magnitude * multiplier
            if (
                tick >= min_cost * 0.3 and tick <= max_cost * 3
            ):  # Include ticks in relevant range
                tick_values.append(tick)

    # Remove duplicates and sort
    tick_values = sorted(list(set(tick_values)))

    # Set x-axis range with minimal padding
    left_padding = 0.8  # Multiplier to start slightly before min
    right_padding = 1.5  # Multiplier to end slightly after max

    fig_pareto.update_layout(
        title=dict(
            text=f"Pareto Analysis: Cost vs Performance{title_suffix}",
            font=dict(size=28),
        ),
        xaxis=dict(
            title=dict(text="Total Cost ($, log scale)", font=dict(size=24)),
            type="log",  # Make x-axis logarithmic
            range=[
                np.log10(min_cost * left_padding),
                np.log10(max_cost * right_padding),
            ],
            tickvals=tick_values,  # Use our custom tick values
            ticktext=[f"{v:.3g}" for v in tick_values],  # Format nicely
            tickformat="",  # Let ticktext handle formatting
            showgrid=True,
            gridcolor="lightgray",
            minor=dict(ticklen=4, tickcolor="lightgray", showgrid=True, griddash="dot"),
            tickfont=dict(size=20),  # Increased tick font size
        ),
        yaxis=dict(
            title=dict(text="Combined Performance Score", font=dict(size=24)),
            tickfont=dict(size=20),  # Increased tick font size
        ),
        height=550,  # Further reduced height for compact PDF
        width=1800,  # Adjusted width to maintain good proportions
        template="plotly_white",
        showlegend=True,
        legend=dict(font=dict(size=18)),  # Increased legend font size
        font=dict(size=18),  # Increased from 12 to 18
        margin=dict(l=100, r=120, t=80, b=80),  # Increased margins for larger fonts
    )
    filename = (
        f"pareto_analysis_cost_vs_performance{title_suffix}".replace(" ", "_")
        .replace("/", "_")
        .replace(":", "")
        .replace("-", "_")
        .lower()
        + ".pdf"
    )
    # Export to PDF with compact dimensions
    fig_pareto.write_image(
        os.path.join(output_dir, filename),
        format="pdf",
        width=1800,  # Match layout width
        height=550,  # Match layout height - more compact vertically
        scale=1,  # Ensure 1:1 scale to avoid extra whitespace
    )
    fig_pareto.show()


def create_summary_table(df):
    """Create a summary table with key metrics"""
    if df.empty:
        return None

    summary_df = df[
        [
            "model_name",
            "total_cost",
            "avg_time_per_query",
            "sql_accuracy",
            "answer_correctness",
            "faithfulness",
            "num_queries",
        ]
    ].copy()

    summary_df = summary_df.round(4)
    summary_df.columns = [
        "Model",
        "Total Cost ($)",
        "Avg Query Time (s)",
        "SQL Accuracy",
        "Answer Correctness",
        "Faithfulness",
        "Queries Tested",
    ]

    return summary_df


def create_cost_performance_table(df):
    """Create a cost-performance summary table with weighted average score"""
    if df.empty:
        return None

    # Calculate weighted average score (metrics are on 1-5 scale)
    # Higher weight for answer_correctness as requested
    df_copy = df.copy()
    df_copy["weighted_avg_score"] = (
        df_copy["sql_accuracy"] * 0.25  # 25% weight
        + df_copy["answer_correctness"] * 0.5  # 50% weight (higher weight)
        + df_copy["faithfulness"] * 0.25  # 25% weight
    )

    cost_perf_df = df_copy[
        [
            "model_name",
            "total_cost",
            "avg_time_per_query",
            "weighted_avg_score",
            "sql_accuracy",
            "answer_correctness",
            "faithfulness",
        ]
    ].copy()

    cost_perf_df = cost_perf_df.round(4)
    cost_perf_df.columns = [
        "Model",
        "Total Cost ($)",
        "Avg Query Time (s)",
        "Weighted Avg Score",
        "SQL Accuracy",
        "Answer Correctness",
        "Faithfulness",
    ]

    # Sort by total cost (highest to lowest) for better cost comparison
    cost_perf_df = cost_perf_df.sort_values("Total Cost ($)", ascending=False)

    return cost_perf_df


def analyze_models(
    tag_filter="tags.arena = 'arena'", dataset_name="arena", title_suffix=""
):
    """Complete analysis workflow for models with specific tag filter"""
    models_df = get_models_data(tag_filter, dataset_name)

    if not models_df.empty:
        print(f"Found {len(models_df)} {dataset_name} models")

        # Create charts
        create_individual_charts(models_df, title_suffix)

        # Display summary table
        summary_table = create_summary_table(models_df)
        if summary_table is not None:
            print(f"\n=== {dataset_name.upper()} MODELS SUMMARY ===")
            print(summary_table.to_string(index=False))

        # Display cost-performance table
        cost_perf_table = create_cost_performance_table(models_df)
        if cost_perf_table is not None:
            print(f"\n=== {dataset_name.upper()} COST-PERFORMANCE SUMMARY ===")
            print(
                "Note: Weighted Avg Score = SQL Accuracy (25%) + Answer Correctness (50%) + Faithfulness (25%)"
            )
            print(cost_perf_table.to_string(index=False))

        # Show top performers
        if len(models_df) > 1:
            print(f"\n=== TOP PERFORMERS ({dataset_name.upper()}) ===")
            best_accuracy = models_df.loc[models_df["sql_accuracy"].idxmax()]
            print(
                f"Best SQL Accuracy: {best_accuracy['model_name']} ({best_accuracy['sql_accuracy']:.3f})"
            )

            lowest_cost = models_df.loc[models_df["total_cost"].idxmin()]
            print(
                f"Lowest Cost: {lowest_cost['model_name']} (${lowest_cost['total_cost']:.4f})"
            )

            fastest = models_df.loc[models_df["avg_time_per_query"].idxmin()]
            print(
                f"Fastest: {fastest['model_name']} ({fastest['avg_time_per_query']:.2f}s)"
            )
    else:
        print(f"No {dataset_name} models found. Check your MLflow tags.")

    return models_df


# Convenience functions
def analyze_arena_models():
    """Analyze arena-tagged models"""
    return analyze_models("tags.arena = 'arena'", "arena", "")


def analyze_validation_models():
    """Analyze validation-tagged models"""
    return analyze_models("tags.val = 'val'", "validation", " - Validation Set")


def analyze_models_by_tag(tag_key, tag_value, dataset_name=None, title_suffix=""):
    """Analyze models by custom tag key and value"""
    if dataset_name is None:
        dataset_name = tag_value
    tag_filter = f"tags.{tag_key} = '{tag_value}'"
    return analyze_models(tag_filter, dataset_name, title_suffix)

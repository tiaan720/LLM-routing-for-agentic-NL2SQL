import concurrent.futures
import json
import logging
import os
import time
import warnings
from typing import Callable, Dict, List

import dotenv
import google.auth
import mlflow
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from mlflow.metrics.genai import EvaluationExample, make_genai_metric

from src.utils.logger import logger

dotenv.load_dotenv()

# Suppress numpy warnings about NaN in mean/std calculations
# These occur when LLM judges return empty results, which we handle with our cleanup code
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

sql_accuracy = make_genai_metric(
    name="sql_accuracy",
    definition="Evaluates the quality and correctness of SQL query syntax, structure, and best practices",
    grading_prompt=(
        "Score from 1-5 based on SQL query quality, where:"
        "\n1: Syntax errors, completely incorrect SQL structure"
        "\n2: Basic syntax correct but poor structure, inefficient queries"
        "\n3: Correct syntax and structure, some optimization opportunities"
        "\n4: Well-written SQL with good practices, minor improvements possible"
        "\n5: Excellent SQL with optimal structure, proper joins, indexing considerations"
        "\nConsider: syntax correctness, query efficiency, proper use of joins, appropriate aggregations, and SQL best practices."
    ),
    examples=[
        EvaluationExample(
            input="Question: How many customers are there? SQL: SELECT COUNT(*) FROM customer;",
            output="SELECT COUNT(*) FROM customer;",
            score=5,
            justification="Perfect SQL: correct syntax, efficient query for counting records",
        ),
        EvaluationExample(
            input="Question: Which store has the most DVDs? SQL: SELECT store_id FROM inventory GROUP BY store_id ORDER BY COUNT(*) DESC LIMIT 1;",
            output="SELECT store_id FROM inventory GROUP BY store_id ORDER BY COUNT(*) DESC LIMIT 1;",
            score=4,
            justification="Good SQL structure with proper aggregation and ordering, could include store name with JOIN",
        ),
        EvaluationExample(
            input="Question: List all customers? SQL: SELECT * FROM customer WHERE 1=1;",
            output="SELECT * FROM customer WHERE 1=1;",
            score=2,
            justification="Basic syntax correct but inefficient with unnecessary WHERE clause, SELECT * is not optimal",
        ),
        EvaluationExample(
            input="Question: Get customer names? SQL: SELECT c.first_name, c.last_name FROM customer c WHERE c.active = 1;",
            output="SELECT c.first_name, c.last_name FROM customer c WHERE c.active = 1;",
            score=5,
            justification="Excellent SQL: proper column selection, table alias, meaningful WHERE condition",
        ),
        EvaluationExample(
            input="Question: Find movies? SQL: SELECT film.title FROM film, film_category WHERE film.film_id = film_category.film_id;",
            output="SELECT film.title FROM film, film_category WHERE film.film_id = film_category.film_id;",
            score=3,
            justification="Correct syntax but uses old-style JOIN syntax, modern INNER JOIN would be better",
        ),
    ],
    model="openai:/gpt-4o",
    parameters={"temperature": 0.0},
)


def run_evaluation(
    experiment_name: str,
    qa_pairs: List[Dict],
    process_query_fn: Callable,
    max_workers: int = 3,
    csv_suffix: str = None,
    log_artifacts_fn: Callable = None,
) -> mlflow.models.EvaluationResult:
    """Generic evaluation runner for all agent types"""

    logger.info(f"Starting evaluation with {len(qa_pairs)} test cases")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)
    mlflow.langchain.autolog()
    mlflow.openai.autolog()

    user_creds, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    if not user_creds.valid:
        user_creds.refresh(Request())

    base_kwargs = {
        "user_token": user_creds.token,
        "project_id": project,
        "dataset_id": "pagila",
    }

    with mlflow.start_run(run_name=f"{experiment_name}_eval") as run:
        results = []
        failed_cases = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qa = {
                executor.submit(
                    lambda q: {
                        "question": q["question"],
                        "expected_answer": q["answer"],
                        **process_query_fn(q["question"], base_kwargs),
                    },
                    qa_pair,
                ): qa_pair
                for qa_pair in qa_pairs
            }

            for future in concurrent.futures.as_completed(future_to_qa):
                qa_pair = future_to_qa[future]
                try:
                    result = future.result()
                    # Ensure result has all required fields with valid types
                    if "actual_answer" not in result:
                        result["actual_answer"] = "Error: No answer generated"
                    if "query_cost" not in result:
                        result["query_cost"] = 0.0
                    if "execution_time" not in result:
                        result["execution_time"] = 0.0
                    if "model_used" not in result:
                        result["model_used"] = "unknown"
                    if "dataset" not in result:
                        result["dataset"] = "unknown"

                    results.append(result)
                    logger.info(f"Processed question: {qa_pair['question'][:80]}...")
                except Exception as e:
                    error_msg = str(e)
                    failed_cases.append(
                        {"question": qa_pair["question"], "error": error_msg}
                    )
                    logger.error(
                        f"Failed to process question: {qa_pair['question'][:80]}..., error: {error_msg}"
                    )

        if failed_cases:
            logger.warning(f"Failed cases: {json.dumps(failed_cases, indent=2)}")

        results_df = pd.DataFrame(results)

        # Log initial DataFrame info
        logger.info(f"Initial results DataFrame shape: {results_df.shape}")
        logger.info(f"DataFrame columns: {results_df.columns.tolist()}")

        # Fill missing values for text columns
        results_df["actual_answer"] = results_df["actual_answer"].fillna(
            "LLM did not provide answer"
        )
        results_df["actual_answer"] = results_df["actual_answer"].replace(
            "", "LLM did not provide answer"
        )

        results_df["expected_answer"] = results_df["expected_answer"].fillna(
            "No expected answer provided"
        )
        results_df["expected_answer"] = results_df["expected_answer"].replace(
            "", "No expected answer provided"
        )

        # Filter out questions that resulted in errors
        # Identify rows where actual_answer contains error messages
        error_patterns = [
            "Error executing query",
            "Error:",
            "failed",
            "exception",
        ]

        error_mask = (
            results_df["actual_answer"]
            .str.lower()
            .str.contains("|".join(error_patterns), case=False, na=False)
        )

        if error_mask.any():
            error_count = error_mask.sum()
            logger.warning(f"Filtering out {error_count} questions with errors")

            # Log the error questions at debug level (less noisy)
            if logger.isEnabledFor(logging.DEBUG):
                error_questions = results_df[error_mask][
                    ["question", "actual_answer", "dataset"]
                ]
                logger.debug(f"Error questions:\n{error_questions.to_string()}")

            # Remove error questions from evaluation
            results_df = results_df[~error_mask].copy()
            logger.info(f"Remaining valid questions after filtering: {len(results_df)}")

        # Ensure numeric columns have valid values
        numeric_columns = ["query_cost", "execution_time"]
        for col in numeric_columns:
            if col in results_df.columns:
                # Convert to numeric, replacing any errors with 0
                results_df[col] = pd.to_numeric(
                    results_df[col], errors="coerce"
                ).fillna(0.0)
                # Ensure no infinite values
                results_df[col] = results_df[col].replace(
                    [float("inf"), float("-inf")], 0.0
                )

        # Final check: ensure we have data to evaluate
        if len(results_df) == 0:
            logger.error("No valid results to evaluate after filtering!")
            raise ValueError(
                "All questions failed or returned errors. Cannot proceed with evaluation."
            )

        logger.info(f"Final DataFrame shape for evaluation: {results_df.shape}")
        logger.info(f"DataFrame dtypes:\n{results_df.dtypes}")

        # Clean ALL NaN, inf, and -inf values from the entire DataFrame before MLflow evaluation
        # This prevents "Float cannot represent non numeric value: nan" errors
        for col in results_df.columns:
            if results_df[col].dtype in ["float64", "float32", "int64", "int32"]:
                # Replace NaN and infinite values with 0
                results_df[col] = results_df[col].replace(
                    [float("inf"), float("-inf")], 0.0
                )
                results_df[col] = results_df[col].fillna(0.0)
            elif results_df[col].dtype == "object":
                # For object columns, replace None/NaN with empty string or appropriate default
                results_df[col] = results_df[col].fillna("")

        logger.info(f"DataFrame cleaned of NaN/inf values")

        # mlflow.log_artifact(results_df)

        logger.info(
            f"Successfully processed {len(results_df)} out of {len(qa_pairs)} test cases for evaluation"
        )

        eval_result = mlflow.evaluate(
            model=None,
            data=results_df,
            predictions="actual_answer",
            targets="expected_answer",
            extra_metrics=[
                sql_accuracy,
                mlflow.metrics.genai.answer_correctness(
                    model="openai:/gpt-4o",
                    examples=[
                        EvaluationExample(
                            input="How many customers are there?",
                            output="599 customers",
                            score=5,
                            justification="Exact match with expected answer (599)",
                        ),
                        EvaluationExample(
                            input="Which store has the most dvds in stock?",
                            output="Store 2",
                            score=3,
                            justification="Correct store but missing quantity information",
                        ),
                        EvaluationExample(
                            input="What is the most popular film category?",
                            output="Sports category with 74 films",
                            score=5,
                            justification="Correct category and exact count provided",
                        ),
                        EvaluationExample(
                            input="How many actors are in the database?",
                            output="There are approximately 200 actors",
                            score=4,
                            justification="Correct general magnitude but not exact count (actual: 200)",
                        ),
                        EvaluationExample(
                            input="What is the average rental rate?",
                            output="The rental cost varies by film",
                            score=2,
                            justification="Vague answer, missing specific average value",
                        ),
                        EvaluationExample(
                            input="Which customer has rented the most films?",
                            output="Customer ID 148 Eleanor Hunt",
                            score=5,
                            justification="Correct customer ID and name provided",
                        ),
                    ],
                ),
                mlflow.metrics.genai.faithfulness(model="openai:/gpt-4o"),
            ],
            evaluators="default",
            evaluator_config={
                "col_mapping": {
                    "inputs": "question",
                    "context": "question",
                    "query_cost": "query_cost",
                    "execution_time": "execution_time",
                }
            },
        )

        # Clean up NaN values in eval_result.metrics to prevent MLflow logging errors
        # This must happen BEFORE MLflow auto-logs the results
        if eval_result and hasattr(eval_result, "metrics"):
            cleaned_count = 0
            cleaned_metrics = {}
            for key, value in eval_result.metrics.items():
                if pd.isna(value) or (
                    isinstance(value, float)
                    and (value == float("inf") or value == float("-inf"))
                ):
                    cleaned_metrics[key] = 0.0
                    cleaned_count += 1
                else:
                    cleaned_metrics[key] = value

            # IMPORTANT: Update the metrics dict in-place to affect auto-logging
            eval_result.metrics.clear()
            eval_result.metrics.update(cleaned_metrics)

            # Also manually log the cleaned metrics to override any auto-logged NaN values
            for key, value in cleaned_metrics.items():
                try:
                    mlflow.log_metric(key, value)
                except Exception as e:
                    logger.warning(f"Failed to log metric {key}: {e}")

            if cleaned_count > 0:
                logger.warning(
                    f"Replaced {cleaned_count} invalid metric values (NaN/Inf) with 0.0"
                )
            else:
                logger.info("All metrics are valid - no cleanup needed")

        # Also clean the tables DataFrame if it exists (this is what causes the MLflow UI error)
        if eval_result and hasattr(eval_result, "tables") and eval_result.tables:
            for table_name, table_df in eval_result.tables.items():
                if isinstance(table_df, pd.DataFrame):
                    # Replace NaN/Inf in all numeric columns
                    for col in table_df.columns:
                        if pd.api.types.is_numeric_dtype(table_df[col]):
                            table_df[col] = table_df[col].replace(
                                [float("inf"), float("-inf")], 0.0
                            )
                            table_df[col] = table_df[col].fillna(0.0)
                    logger.debug(f"Cleaned NaN/Inf values in table: {table_name}")

        # Call optional artifacts logging function if provided (inside the MLflow run context)
        if log_artifacts_fn:
            try:
                log_artifacts_fn(eval_result, experiment_name, results, results_df)
            except Exception as e:
                logger.error(f"Failed to log custom artifacts: {str(e)}")

    return eval_result

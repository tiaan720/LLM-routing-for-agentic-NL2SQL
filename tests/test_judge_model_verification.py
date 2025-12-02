"""
Verification test for MLflow Judge Models (SQL Accuracy, Answer Correctness, Faithfulness).

This test validates that the judge models used in MLflow evaluation are working correctly by:
- Testing with mock SQL queries and answers with known quality levels
- Verifying judge model scores match expected score ranges
- Running MULTIPLE ITERATIONS to test consistency
- Comparing expected scores vs actual scores across all runs
- Generating easy-to-read comparison table with all metrics

The test runs each evaluation 3 times (configurable) and:
1. Checks if scores fall within expected ranges
2. Calculates consistency (standard deviation across runs)
3. Creates comprehensive CSV showing:
   - Expected score ranges for each test case
   - Actual scores from each run (Run 1, Run 2, Run 3)
   - Averages and standard deviations
   - All 3 metrics: SQL Accuracy, Answer Correctness, Faithfulness

Results saved to MLflow for thesis verification.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import dotenv
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.metrics.genai import EvaluationExample, make_genai_metric

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Replicate the exact SQL accuracy metric from experiments/common/evaluation.py
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


class MockTestCaseGenerator:
    """Generate mock test cases with known quality levels for judge model verification"""

    @staticmethod
    def generate_test_cases() -> List[Dict]:
        """
        Generate test cases spanning different quality levels.
        Each case has expected score range for validation.
        """
        return [
            # EXCELLENT SQL (Score 5)
            {
                "question": "How many active customers do we have?",
                "actual_answer": "584 active customers",
                "expected_answer": "584",
                "sql_query": "SELECT COUNT(*) FROM customers WHERE active = 1;",
                "expected_sql_score": 5.0,
                "expected_answer_score": 5.0,
                "expected_faithfulness_score": 4.0,
                "category": "excellent",
                "description": "Perfect SQL with exact answer match",
            },
            {
                "question": "What is the total revenue from orders?",
                "actual_answer": "The total revenue from completed orders is $125,480.50",
                "expected_answer": "$125,480.50",
                "sql_query": "SELECT SUM(amount) FROM orders WHERE status = 'completed';",
                "expected_sql_score": 5.0,
                "expected_answer_score": 5.0,
                "expected_faithfulness_score": 3.0,
                "category": "excellent",
                "description": "Excellent SQL with precise answer",
            },
            # GOOD SQL (Score 4)
            {
                "question": "List top 5 customers by spending",
                "actual_answer": "Alice, Bob, Carol, David, Eve",
                "expected_answer": "Alice, Bob, Carol, David, Eve",
                "sql_query": "SELECT customer_name, SUM(amount) as total FROM orders GROUP BY customer_id, customer_name ORDER BY total DESC LIMIT 5;",
                "expected_sql_score": 4.0,
                "expected_answer_score": 5.0,
                "expected_faithfulness_score": 5.0,
                "category": "good",
                "description": "Good SQL with exact answer match",
            },
            {
                "question": "Find products in electronics category",
                "actual_answer": "23 products",
                "expected_answer": "23 products",
                "sql_query": "SELECT p.* FROM products p INNER JOIN categories c ON p.category_id = c.id WHERE c.name = 'Electronics';",
                "expected_sql_score": 4.0,
                "expected_answer_score": 5.0,
                "expected_faithfulness_score": 3.0,
                "category": "good",
                "description": "Good JOIN query with exact answer",
            },
            # ACCEPTABLE SQL (Score 3)
            {
                "question": "Get all customers",
                "actual_answer": "There are 599 customers in the database",
                "expected_answer": "599 customers",
                "sql_query": "SELECT * FROM customers;",
                "expected_sql_score": 3.0,
                "expected_answer_score": 4.0,
                "expected_faithfulness_score": 2.0,
                "category": "acceptable",
                "description": "SELECT * is not optimal, but answer is accurate",
            },
            {
                "question": "Show recent customer orders",
                "actual_answer": "Found orders from the last month",
                "expected_answer": "Customer orders with names, dates and amounts",
                "sql_query": "SELECT customers.name, orders.order_date, orders.amount FROM customers, orders WHERE customers.id = orders.customer_id AND orders.order_date > '2024-10-01';",
                "expected_sql_score": 3.0,
                "expected_answer_score": 3.0,
                "expected_faithfulness_score": 3.0,
                "category": "acceptable",
                "description": "Old-style JOIN syntax (not modern), vague but accurate answer",
            },
            # POOR SQL (Score 2)
            {
                "question": "Count orders per customer",
                "actual_answer": "Multiple orders found. Additionally, we found that 85% of customers are premium members and eligible for discounts.",
                "expected_answer": "Customer order counts with names",
                "sql_query": "SELECT * FROM orders WHERE 1=1 GROUP BY customer_id;",
                "expected_sql_score": 2.0,
                "expected_answer_score": 2.0,
                "expected_faithfulness_score": 2.0,
                "category": "poor",
                "description": "Poor SQL with unnecessary WHERE, answer adds hallucinated premium member information",
            },
            {
                "question": "Average product price",
                "actual_answer": "Prices vary. The most expensive item costs $2,500 and is a limited edition product.",
                "expected_answer": "$45.99",
                "sql_query": "SELECT price FROM products;",
                "expected_sql_score": 2.0,
                "expected_answer_score": 2.0,
                "expected_faithfulness_score": 2.0,
                "category": "poor",
                "description": "Missing aggregation, answer adds hallucinated information about expensive items",
            },
            # VERY POOR SQL (Score 1)
            {
                "question": "Total sales by month",
                "actual_answer": "Error processing query. However, based on our records, January had the highest sales at $50,000.",
                "expected_answer": "Monthly sales breakdown",
                "sql_query": "SELECT SUM(amount) FROM orders WHERE date >",
                "expected_sql_score": 1.0,
                "expected_answer_score": 1.0,
                "expected_faithfulness_score": 1.0,
                "category": "very_poor",
                "description": "Syntax error with hallucinated sales data",
            },
            {
                "question": "List all products",
                "actual_answer": "Query failed. Our catalog includes over 1000 products across 20 categories, with new items added weekly.",
                "expected_answer": "All product names and prices",
                "sql_query": "SELCT * FRM products;",
                "expected_sql_score": 1.0,
                "expected_answer_score": 1.0,
                "expected_faithfulness_score": 1.0,
                "category": "very_poor",
                "description": "Syntax errors with made-up catalog information",
            },
        ]


class JudgeModelVerifier:
    """Verify MLflow judge models are scoring correctly"""

    def __init__(self, num_iterations: int = 3):
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment("judge_model_verification")
        self.run = None
        self.original_stdout = None
        self.num_iterations = num_iterations
        self.all_scores = []  # Store scores from all iterations

    def start_run(self):
        """Start MLflow run"""
        self.original_stdout = sys.stdout

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            self.run = mlflow.start_run(run_name="judge_model_scoring_verification")
        finally:
            if sys.platform == "win32" and sys.stdout != self.original_stdout:
                sys.stdout.close()
            sys.stdout = self.original_stdout

    def evaluate_test_cases(self, test_cases: List[Dict]) -> pd.DataFrame:
        """Run MLflow evaluation on test cases multiple times to test consistency"""
        logger.info(f"Evaluating {len(test_cases)} test cases with judge models...")
        logger.info(f"Running {self.num_iterations} iterations to test consistency...")

        all_iterations_scores = []

        for iteration in range(self.num_iterations):
            logger.info(f"\n  Iteration {iteration + 1}/{self.num_iterations}...")

            # Prepare data in MLflow format
            eval_data = {
                "question": [tc["question"] for tc in test_cases],
                "actual_answer": [tc["actual_answer"] for tc in test_cases],
                "expected_answer": [tc["expected_answer"] for tc in test_cases],
            }
            eval_df = pd.DataFrame(eval_data)

            # Create inputs column for SQL accuracy (combines question + SQL)
            inputs = []
            for tc in test_cases:
                inputs.append(f"Question: {tc['question']} SQL: {tc['sql_query']}")
            eval_df["inputs"] = inputs

            # Run evaluation
            result = mlflow.evaluate(
                model=None,
                data=eval_df,
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
                                input="What is the average rental rate?",
                                output="The rental cost varies by film",
                                score=2,
                                justification="Vague answer, missing specific average value",
                            ),
                        ],
                    ),
                    mlflow.metrics.genai.faithfulness(model="openai:/gpt-4o"),
                ],
                evaluators="default",
                evaluator_config={
                    "col_mapping": {
                        "inputs": "inputs",
                        "context": "question",
                    }
                },
            )

            # Extract scores from result
            scores_df = result.tables["eval_results_table"].copy()
            scores_df["iteration"] = iteration + 1
            all_iterations_scores.append(scores_df)

        # Combine all iterations
        combined_scores = pd.concat(all_iterations_scores, ignore_index=True)
        self.all_scores = all_iterations_scores

        return combined_scores

    def analyze_judge_scores(
        self, scores_df: pd.DataFrame, test_cases: List[Dict]
    ) -> Dict:
        """Analyze if judge model scores match expected ranges and calculate consistency"""
        logger.info("Analyzing judge model scoring accuracy and consistency...")

        # Get column names
        sql_col = "sql_accuracy/v1/score"
        answer_col = "answer_correctness/v1/score"
        faith_col = "faithfulness/v1/score"

        # Calculate consistency statistics across iterations
        consistency_stats = {}

        if self.num_iterations > 1:
            for test_idx, tc in enumerate(test_cases):
                # Get all scores for this test case across iterations
                test_scores = scores_df[scores_df.index % len(test_cases) == test_idx]

                consistency_stats[test_idx] = {
                    "question": tc["question"],
                    "category": tc["category"],
                    "sql_scores": test_scores[sql_col].tolist(),
                    "answer_scores": test_scores[answer_col].tolist(),
                    "faithfulness_scores": test_scores[faith_col].tolist(),
                    "sql_mean": float(test_scores[sql_col].mean()),
                    "answer_mean": float(test_scores[answer_col].mean()),
                    "faithfulness_mean": float(test_scores[faith_col].mean()),
                    "sql_std": float(test_scores[sql_col].std()),
                    "answer_std": float(test_scores[answer_col].std()),
                    "faithfulness_std": float(test_scores[faith_col].std()),
                    "sql_range": float(
                        test_scores[sql_col].max() - test_scores[sql_col].min()
                    ),
                    "answer_range": float(
                        test_scores[answer_col].max() - test_scores[answer_col].min()
                    ),
                    "faithfulness_range": float(
                        test_scores[faith_col].max() - test_scores[faith_col].min()
                    ),
                }

        # Use first iteration for range checking (or average across iterations)
        first_iter_mask = scores_df["iteration"] == 1
        analysis_df = scores_df[first_iter_mask].copy().reset_index(drop=True)

        # Add expected scores
        for i, tc in enumerate(test_cases):
            analysis_df.loc[i, "expected_sql_score"] = tc.get("expected_sql_score", 3.0)
            analysis_df.loc[i, "expected_answer_score"] = tc.get(
                "expected_answer_score", 3.0
            )
            analysis_df.loc[i, "expected_faithfulness_score"] = tc.get(
                "expected_faithfulness_score", 3.0
            )
            analysis_df.loc[i, "category"] = tc["category"]
            analysis_df.loc[i, "description"] = tc["description"]

        # Calculate difference from expected (how close judge got to expected score)
        analysis_df["sql_diff"] = abs(
            analysis_df[sql_col] - analysis_df["expected_sql_score"]
        )
        analysis_df["answer_diff"] = abs(
            analysis_df[answer_col] - analysis_df["expected_answer_score"]
        )
        analysis_df["faithfulness_diff"] = abs(
            analysis_df[faith_col] - analysis_df["expected_faithfulness_score"]
        )

        # Check if scores are close to expected (within 1.0 point tolerance)
        analysis_df["sql_close"] = analysis_df["sql_diff"] <= 1.0
        analysis_df["answer_close"] = analysis_df["answer_diff"] <= 1.0
        analysis_df["faithfulness_close"] = analysis_df["faithfulness_diff"] <= 1.0

        # Calculate accuracy by category with better interpretability
        category_stats = {}
        # Define expected order from worst to best
        category_order = [
            "very_poor",
            "poor",
            "acceptable",
            "good",
            "excellent",
            "partial",
        ]

        for category in category_order:
            if category not in analysis_df["category"].values:
                continue

            cat_data = analysis_df[analysis_df["category"] == category]
            category_stats[category] = {
                "count": len(cat_data),
                "sql_close_pct": (cat_data["sql_close"].sum() / len(cat_data)) * 100,
                "answer_close_pct": (cat_data["answer_close"].sum() / len(cat_data))
                * 100,
                "avg_sql_score": float(cat_data[sql_col].mean()),
                "avg_answer_score": float(cat_data[answer_col].mean()),
                "avg_faithfulness": (
                    float(cat_data[faith_col].mean()) if faith_col in cat_data else 0
                ),
                "expected_sql_score": float(cat_data["expected_sql_score"].iloc[0]),
                "expected_answer_score": float(
                    cat_data["expected_answer_score"].iloc[0]
                ),
                "avg_sql_diff": float(cat_data["sql_diff"].mean()),
                "avg_answer_diff": float(cat_data["answer_diff"].mean()),
            }

        # Check if scores follow expected quality ordering (better quality → higher scores)
        quality_ordering_valid = True
        quality_order_sql = []
        quality_order_answer = []

        for cat in ["very_poor", "poor", "acceptable", "good", "excellent"]:
            if cat in category_stats:
                quality_order_sql.append(category_stats[cat]["avg_sql_score"])
                quality_order_answer.append(category_stats[cat]["avg_answer_score"])

        # Check if generally increasing (allowing some tolerance)
        sql_ordering_issues = sum(
            1
            for i in range(len(quality_order_sql) - 1)
            if quality_order_sql[i] > quality_order_sql[i + 1] + 0.3
        )
        answer_ordering_issues = sum(
            1
            for i in range(len(quality_order_answer) - 1)
            if quality_order_answer[i] > quality_order_answer[i + 1] + 0.3
        )

        # Overall statistics
        overall_stats = {
            "total_cases": len(analysis_df),
            "num_iterations": self.num_iterations,
            "sql_close_to_expected": int(analysis_df["sql_close"].sum()),
            "sql_close_to_expected_pct": (
                analysis_df["sql_close"].sum() / len(analysis_df)
            )
            * 100,
            "answer_close_to_expected": int(analysis_df["answer_close"].sum()),
            "answer_close_to_expected_pct": (
                analysis_df["answer_close"].sum() / len(analysis_df)
            )
            * 100,
            "avg_sql_score": float(analysis_df[sql_col].mean()),
            "avg_answer_score": float(analysis_df[answer_col].mean()),
            "avg_faithfulness": (
                float(analysis_df[faith_col].mean()) if faith_col in analysis_df else 0
            ),
            "avg_sql_diff": float(analysis_df["sql_diff"].mean()),
            "avg_answer_diff": float(analysis_df["answer_diff"].mean()),
            "quality_ordering_valid": {
                "sql_ordering_issues": sql_ordering_issues,
                "answer_ordering_issues": answer_ordering_issues,
                "sql_follows_expected_order": sql_ordering_issues == 0,
                "answer_follows_expected_order": answer_ordering_issues == 0,
            },
        }

        # Add consistency metrics if multiple iterations
        if self.num_iterations > 1 and consistency_stats:
            all_sql_stds = [s["sql_std"] for s in consistency_stats.values()]
            all_answer_stds = [s["answer_std"] for s in consistency_stats.values()]
            all_faith_stds = [s["faithfulness_std"] for s in consistency_stats.values()]

            overall_stats["consistency"] = {
                "avg_sql_std": float(np.mean(all_sql_stds)),
                "max_sql_std": float(np.max(all_sql_stds)),
                "avg_answer_std": float(np.mean(all_answer_stds)),
                "max_answer_std": float(np.max(all_answer_stds)),
                "avg_faithfulness_std": float(np.mean(all_faith_stds)),
                "max_faithfulness_std": float(np.max(all_faith_stds)),
                "sql_highly_consistent": float(np.mean(all_sql_stds)) < 0.3,
                "answer_highly_consistent": float(np.mean(all_answer_stds)) < 0.3,
            }

        return {
            "overall": overall_stats,
            "by_category": category_stats,
            "consistency_by_test": consistency_stats,
            "scores_df": analysis_df,
            "all_scores_df": scores_df,  # Include all iterations
        }

    def _format_summary_text(self, analysis: Dict, test_cases: List[Dict]) -> str:
        """Generate formatted summary text for both logging and artifact"""
        lines = []
        lines.append("=" * 60)
        lines.append("JUDGE MODEL VERIFICATION SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        lines.append("✓ VALIDATION RESULTS:")
        lines.append(f"  Total test cases: {len(test_cases)}")
        lines.append(f"  Iterations per case: {self.num_iterations}")
        sql_pass = (
            "✓ PASS"
            if analysis["overall"]["sql_close_to_expected_pct"] >= 65
            else "✗ FAIL"
        )
        ans_pass = (
            "✓ PASS"
            if analysis["overall"]["answer_close_to_expected_pct"] >= 65
            else "✗ FAIL"
        )
        lines.append(
            f"  SQL Accuracy close to expected: {analysis['overall']['sql_close_to_expected_pct']:.1f}% ({sql_pass})"
        )
        lines.append(
            f"  Answer Correctness close to expected: {analysis['overall']['answer_close_to_expected_pct']:.1f}% ({ans_pass})"
        )
        lines.append(f"  Average difference from expected:")
        lines.append(f"    SQL: {analysis['overall']['avg_sql_diff']:.2f} points")
        lines.append(f"    Answer: {analysis['overall']['avg_answer_diff']:.2f} points")

        # Consistency metrics
        if "consistency" in analysis["overall"]:
            lines.append("")
            lines.append(
                f"✓ CONSISTENCY CHECK (across {self.num_iterations} iterations):"
            )
            consistency = analysis["overall"]["consistency"]
            sql_cons = (
                "✓ Highly consistent"
                if consistency["sql_highly_consistent"]
                else "⚠ Some variation"
            )
            ans_cons = (
                "✓ Highly consistent"
                if consistency["answer_highly_consistent"]
                else "⚠ Some variation"
            )
            lines.append(
                f"  SQL Accuracy variation: {consistency['avg_sql_std']:.3f} avg, {consistency['max_sql_std']:.3f} max ({sql_cons})"
            )
            lines.append(
                f"  Answer Correctness variation: {consistency['avg_answer_std']:.3f} avg, {consistency['max_answer_std']:.3f} max ({ans_cons})"
            )

        # Simple score comparison by category
        lines.append("")
        lines.append("✓ SCORE COMPARISON BY QUALITY LEVEL:")
        lines.append(
            f"  {'Category':<15} {'Expected SQL':<13} {'Actual SQL':<13} {'Diff':<8} {'Expected Ans':<13} {'Actual Ans':<13} {'Diff':<8} {'Close'}"
        )
        lines.append(f"  {'-'*105}")

        for category in [
            "very_poor",
            "poor",
            "acceptable",
            "good",
            "excellent",
            "partial",
        ]:
            if category in analysis["by_category"]:
                stats = analysis["by_category"][category]
                sql_close = "✓" if stats["sql_close_pct"] >= 50 else "✗"
                ans_close = "✓" if stats["answer_close_pct"] >= 50 else "✗"

                lines.append(
                    f"  {category.upper():<15} "
                    f"{stats['expected_sql_score']:<13.1f} "
                    f"{stats['avg_sql_score']:<13.2f} "
                    f"{stats['avg_sql_diff']:<8.2f} "
                    f"{stats['expected_answer_score']:<13.1f} "
                    f"{stats['avg_answer_score']:<13.2f} "
                    f"{stats['avg_answer_diff']:<8.2f} "
                    f"{sql_close} {ans_close}"
                )

        lines.append("")
        lines.append("✓ See 'judge_scores_comparison.csv' for detailed score breakdown")
        lines.append("  - Shows expected score ranges vs actual scores for each run")
        lines.append(
            "  - Includes all 3 metrics: SQL Accuracy, Answer Correctness, Faithfulness"
        )
        lines.append("  - Easy to compare consistency across runs")

        lines.append("")
        lines.append("=" * 60)
        lines.append("✓ Judge model verification complete")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_results(self, analysis: Dict, test_cases: List[Dict]):
        """Save verification results to MLflow"""
        if not self.run:
            return

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            # Log parameters
            mlflow.log_param("judge_model", "gpt-4o")
            mlflow.log_param("num_test_cases", len(test_cases))
            mlflow.log_param("num_categories", len(analysis["by_category"]))
            mlflow.log_param("num_iterations", self.num_iterations)

            # Log overall metrics
            mlflow.log_metric(
                "sql_close_to_expected_pct",
                analysis["overall"]["sql_close_to_expected_pct"],
            )
            mlflow.log_metric(
                "answer_close_to_expected_pct",
                analysis["overall"]["answer_close_to_expected_pct"],
            )
            mlflow.log_metric("avg_sql_diff", analysis["overall"]["avg_sql_diff"])
            mlflow.log_metric("avg_answer_diff", analysis["overall"]["avg_answer_diff"])
            mlflow.log_metric("avg_sql_score", analysis["overall"]["avg_sql_score"])
            mlflow.log_metric(
                "avg_answer_score", analysis["overall"]["avg_answer_score"]
            )
            mlflow.log_metric(
                "avg_faithfulness", analysis["overall"]["avg_faithfulness"]
            )

            # Log consistency metrics if available
            if "consistency" in analysis["overall"]:
                consistency = analysis["overall"]["consistency"]
                mlflow.log_metric("avg_sql_consistency_std", consistency["avg_sql_std"])
                mlflow.log_metric("max_sql_consistency_std", consistency["max_sql_std"])
                mlflow.log_metric(
                    "avg_answer_consistency_std", consistency["avg_answer_std"]
                )
                mlflow.log_metric(
                    "max_answer_consistency_std", consistency["max_answer_std"]
                )

            # Log category metrics
            for category, stats in analysis["by_category"].items():
                mlflow.log_metric(f"{category}_sql_close_pct", stats["sql_close_pct"])
                mlflow.log_metric(
                    f"{category}_answer_close_pct", stats["answer_close_pct"]
                )
                mlflow.log_metric(f"{category}_avg_sql_diff", stats["avg_sql_diff"])
                mlflow.log_metric(
                    f"{category}_avg_answer_diff", stats["avg_answer_diff"]
                )

            # Save artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create simplified comparison table
                comparison_rows = []

                for idx, tc in enumerate(test_cases):
                    row = {
                        "test_case": idx + 1,
                        "question": tc["question"],
                        "expected_answer": tc["expected_answer"],
                        "actual_answer": tc["actual_answer"],
                    }

                    # Get actual scores from all iterations
                    if analysis["consistency_by_test"]:
                        test_data = analysis["consistency_by_test"][idx]

                        # SQL ACCURACY - all together
                        row["sql_expected"] = tc.get("expected_sql_score", 3.0)
                        for run_num in range(self.num_iterations):
                            row[f"sql_actual_run{run_num + 1}"] = test_data[
                                "sql_scores"
                            ][run_num]
                        row["sql_average"] = test_data["sql_mean"]
                        row["sql_std_dev"] = test_data["sql_std"]
                        row["sql_difference"] = abs(
                            test_data["sql_mean"] - row["sql_expected"]
                        )

                        # ANSWER CORRECTNESS - all together
                        row["answer_expected"] = tc.get("expected_answer_score", 3.0)
                        for run_num in range(self.num_iterations):
                            row[f"answer_actual_run{run_num + 1}"] = test_data[
                                "answer_scores"
                            ][run_num]
                        row["answer_average"] = test_data["answer_mean"]
                        row["answer_std_dev"] = test_data["answer_std"]
                        row["answer_difference"] = abs(
                            test_data["answer_mean"] - row["answer_expected"]
                        )

                        # FAITHFULNESS - all together
                        row["faithfulness_expected"] = tc.get(
                            "expected_faithfulness_score", 3.0
                        )
                        for run_num in range(self.num_iterations):
                            row[f"faithfulness_actual_run{run_num + 1}"] = test_data[
                                "faithfulness_scores"
                            ][run_num]
                        row["faithfulness_average"] = test_data["faithfulness_mean"]
                        row["faithfulness_std_dev"] = test_data["faithfulness_std"]
                        row["faithfulness_difference"] = abs(
                            test_data["faithfulness_mean"]
                            - row["faithfulness_expected"]
                        )

                    comparison_rows.append(row)

                comparison_df = pd.DataFrame(comparison_rows)
                comparison_path = Path(tmpdir) / "judge_scores_comparison.csv"
                comparison_df.to_csv(comparison_path, index=False)
                mlflow.log_artifact(comparison_path)

                # Save human-readable summary as text file
                summary_text_path = Path(tmpdir) / "judge_verification_summary.txt"
                with open(summary_text_path, "w", encoding="utf-8") as f:
                    f.write(self._format_summary_text(analysis, test_cases))
                mlflow.log_artifact(summary_text_path)

            run_id = self.run.info.run_id
            logger.info(f"\nJudge model verification saved to MLflow run: {run_id}")
            logger.info(f"View at: {mlflow.get_tracking_uri()}/#/experiments")
        finally:
            if sys.platform == "win32" and sys.stdout != self.original_stdout:
                sys.stdout.close()
            sys.stdout = self.original_stdout

    def end_run(self):
        """End MLflow run"""
        if self.run:
            try:
                if sys.platform == "win32":
                    sys.stdout = open(os.devnull, "w", encoding="utf-8")
                mlflow.end_run()
            finally:
                if sys.platform == "win32" and sys.stdout != self.original_stdout:
                    sys.stdout.close()
                sys.stdout = self.original_stdout


@pytest.mark.integration
def test_judge_model_scoring():
    """
    Verification test for MLflow judge models with consistency checking.

    This test verifies that the judge models (SQL Accuracy, Answer Correctness, Faithfulness)
    are scoring correctly and consistently by:
    1. Creating mock test cases with known quality levels and expected score ranges
    2. Running MLflow evaluation MULTIPLE TIMES (3 iterations)
    3. Comparing actual scores against expected ranges
    4. Measuring consistency (variation across runs)
    5. Generating comparison CSV showing expected vs actual scores for all runs

    The CSV makes it easy to see:
    - If judge scores are close to expected scores
    - If judge scores are consistent across multiple runs
    - All 3 metrics side-by-side for easy comparison
    """
    logger.info("\n" + "=" * 60)
    logger.info("JUDGE MODEL VERIFICATION TEST")
    logger.info("Testing: SQL Accuracy, Answer Correctness, Faithfulness")
    logger.info("With Multiple Iterations for Consistency Validation")
    logger.info("=" * 60)

    # Generate test cases
    generator = MockTestCaseGenerator()
    test_cases = generator.generate_test_cases()
    logger.info(f"Generated {len(test_cases)} test cases across quality categories")

    # Initialize verifier with 3 iterations
    verifier = JudgeModelVerifier(num_iterations=3)
    verifier.start_run()

    # Evaluate with judge models (multiple iterations)
    scores_df = verifier.evaluate_test_cases(test_cases)

    # Analyze results
    analysis = verifier.analyze_judge_scores(scores_df, test_cases)

    # Save to MLflow
    verifier.save_results(analysis, test_cases)
    verifier.end_run()

    # Print summary using the same formatter
    logger.info("\n" + verifier._format_summary_text(analysis, test_cases))

    # Assertions
    assert analysis["overall"]["sql_close_to_expected_pct"] >= 65, (
        f"SQL Accuracy judge model not close to expected scores: "
        f"{analysis['overall']['sql_close_to_expected_pct']:.1f}% within 1 point (expected >=65%)"
    )

    assert analysis["overall"]["answer_close_to_expected_pct"] >= 65, (
        f"Answer Correctness judge model not close to expected scores: "
        f"{analysis['overall']['answer_close_to_expected_pct']:.1f}% within 1 point (expected >=65%)"
    )

    # Consistency assertion - scores shouldn't vary too wildly
    if "consistency" in analysis["overall"]:
        consistency = analysis["overall"]["consistency"]
        assert (
            consistency["max_sql_std"] < 1.5
        ), f"SQL Accuracy judge model too inconsistent (max std dev: {consistency['max_sql_std']:.3f}, expected <1.5)"
        assert (
            consistency["max_answer_std"] < 1.5
        ), f"Answer Correctness judge model too inconsistent (max std dev: {consistency['max_answer_std']:.3f}, expected <1.5)"


if __name__ == "__main__":
    test_judge_model_scoring()

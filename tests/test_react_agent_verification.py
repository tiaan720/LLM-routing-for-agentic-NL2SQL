#!/usr/bin/env python3
"""
Verification test for ReAct SQL Agent behavior and consistency.

This test validates the ReAct agent's behavior by running multiple iterations
of the same query and analyzing:
- Distribution of tool calls
- Variability in answers
- Execution time consistency
- MLflow trace patterns

Results are stored as artifacts for thesis verification.
"""

import json
import logging
import os
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import dotenv
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.agents.sql_agent import create_sql_agent
from src.cost_calc.shared_store import get_shared_store
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model
from src.utils.db_clients import DatabaseClient
from src.utils.logger import logger

dotenv.load_dotenv()

# Set up logging for test
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)


class MockPostgresDatabase:
    """Mock PostgreSQL database with simple test data using SQLite"""

    def __init__(self):
        # Create in-memory SQLite database
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        # Don't use Row factory - we need plain tuples for psycopg2 compatibility
        # self.conn.row_factory = sqlite3.Row
        self._setup_schema()
        self._insert_test_data()

    def _setup_schema(self):
        """Create simple test schema"""
        cursor = self.conn.cursor()

        # Create customers table
        cursor.execute(
            """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                city TEXT,
                total_purchases INTEGER DEFAULT 0
            )
        """
        )

        # Create orders table
        cursor.execute(
            """
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date TEXT,
                amount REAL,
                status TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """
        )

        self.conn.commit()

    def _insert_test_data(self):
        """Insert deterministic test data"""
        cursor = self.conn.cursor()

        # Insert customers
        customers = [
            (1, "Alice Johnson", "alice@example.com", "Boston", 5),
            (2, "Bob Smith", "bob@example.com", "New York", 3),
            (3, "Carol White", "carol@example.com", "Boston", 8),
            (4, "David Brown", "david@example.com", "Chicago", 2),
            (5, "Eve Davis", "eve@example.com", "Boston", 12),
        ]
        cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

        # Insert orders
        orders = [
            (1, 1, "2024-01-15", 150.00, "completed"),
            (2, 1, "2024-02-20", 200.00, "completed"),
            (3, 2, "2024-01-10", 75.00, "completed"),
            (4, 3, "2024-03-05", 300.00, "completed"),
            (5, 3, "2024-03-10", 450.00, "pending"),
            (6, 5, "2024-02-28", 600.00, "completed"),
        ]
        cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

        self.conn.commit()

    def create_mock_cursor(self):
        """Create a cursor wrapper that matches psycopg2 interface"""

        class MockCursor:
            """Wrapper around SQLite cursor to match psycopg2 interface"""

            def __init__(self, sqlite_conn):
                self._cursor = sqlite_conn.cursor()
                self.rowcount = 0

            def execute(self, query, params=None):
                """Execute query with psycopg2-compatible interface"""
                try:
                    if params:
                        result = self._cursor.execute(query, params)
                    else:
                        result = self._cursor.execute(query)
                    self.rowcount = self._cursor.rowcount
                    return result
                except Exception as e:
                    logger.error(f"Error executing query: {query}, error: {e}")
                    raise

            def fetchall(self):
                """Fetch all results as tuples (psycopg2-compatible)"""
                results = self._cursor.fetchall()
                # SQLite returns tuples by default when row_factory is not set
                return results

            @property
            def description(self):
                """Return column descriptions (psycopg2-compatible)"""
                return self._cursor.description

            def close(self):
                """Close the cursor"""
                self._cursor.close()

        return MockCursor(self.conn)

    def create_mock_client(self):
        """Create a mock DatabaseClient that uses this SQLite database"""
        mock_client = MagicMock()
        mock_client.db_type = "postgres"
        mock_client.cursor = self.create_mock_cursor
        mock_client.commit = self.conn.commit
        mock_client.rollback = self.conn.rollback
        mock_client.close = self.conn.close

        # Wrap the client in DatabaseClient
        db_client = DatabaseClient(
            client=mock_client,
            db_type="postgres",
            host="localhost",
            port=5432,
            database="test_db",
            username="test",
            password="test",
        )

        # Override cursor method to use our SQLite connection
        db_client.client.cursor = self.create_mock_cursor

        return db_client

    def close(self):
        """Close database connection"""
        try:
            self.conn.close()
        except Exception:
            pass


class ReactAgentVerificationTest:
    """Verification test suite for ReAct SQL Agent"""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        num_iterations: int = 20,
        test_query: str = "How many customers are in Boston?",
    ):
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.test_query = test_query
        self.results = []
        self.experiment_name = "react_agent_verification"

    def run_single_iteration(
        self, iteration_num: int, mock_db_client: DatabaseClient
    ) -> Dict:
        """Run single agent iteration and collect metrics"""
        logger.info(f"Running iteration {iteration_num + 1}/{self.num_iterations}")

        # Create model with temperature 0 for consistency
        model = create_model(
            model="vertex_ai",
            model_kwargs={
                "model": self.model_name,
                "temperature": 0.0,
                "project": os.getenv("GCP_PROJECT", "research-su-llm-routing"),
                "location": "us-central1",
            },
        )

        # Track execution
        start_time = time.time()
        store = get_shared_store()

        try:
            # Patch create_database_client to return our mock
            with patch(
                "src.agents.sql_agent.create_database_client",
                return_value=mock_db_client,
            ):
                agent = create_sql_agent(
                    model=model,
                    input_query=self.test_query,
                    database_type="postgres",
                    host="localhost",
                    port=5432,
                    database="test_db",
                    username="test",
                    password="test",
                    tools=["postgres_execute_sql", "postgres_list_schema"],
                )

                stream_generator = agent.stream(
                    {"messages": [("human", self.test_query)]},
                    {"configurable": {"thread_id": f"verify_{iteration_num}"}},
                    stream_mode="updates",
                )

                responses, input_cost, output_cost, selected_model, cost_tracker = (
                    process_agent_stream_with_store(
                        stream_generator,
                        store=store,
                        agent_name="sql_agent",
                        model_name=model.model_name,
                    )
                )

            execution_time = time.time() - start_time

            # Extract metrics from responses
            tool_calls = []
            final_answer = None

            for response in responses:
                if (
                    "agent" in response
                    and "messages" in response["agent"]
                    and response["agent"]["messages"]
                ):
                    for message in response["agent"]["messages"]:
                        # Count tool calls
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                if isinstance(tool_call, dict):
                                    tool_calls.append(tool_call.get("name", "unknown"))

                        # Extract final answer
                        if hasattr(message, "content") and message.content:
                            final_answer = message.content

            return {
                "iteration": iteration_num,
                "execution_time": execution_time,
                "tool_call_count": len(tool_calls),
                "tool_calls": tool_calls,
                "final_answer": final_answer or "No answer generated",
                "input_cost": input_cost or 0.0,
                "output_cost": output_cost or 0.0,
                "total_cost": (input_cost or 0.0) + (output_cost or 0.0),
                "success": final_answer is not None,
            }

        except Exception as e:
            logger.error(f"Iteration {iteration_num} failed: {str(e)}")
            return {
                "iteration": iteration_num,
                "execution_time": time.time() - start_time,
                "tool_call_count": 0,
                "tool_calls": [],
                "final_answer": f"Error: {str(e)}",
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "success": False,
            }

    def run_verification(self, mock_db_client: DatabaseClient) -> pd.DataFrame:
        """Run all iterations and collect results"""
        logger.info(
            f"Starting ReAct agent verification with {self.num_iterations} iterations"
        )

        for i in range(self.num_iterations):
            result = self.run_single_iteration(i, mock_db_client)
            self.results.append(result)

        return pd.DataFrame(self.results)

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Analyze collected results and compute statistics"""
        successful_runs = results_df[results_df["success"] == True]

        if len(successful_runs) == 0:
            logger.error("No successful runs to analyze!")
            return {}

        # Tool call statistics
        tool_call_counts = successful_runs["tool_call_count"].values
        tool_call_stats = {
            "mean": float(np.mean(tool_call_counts)),
            "std": float(np.std(tool_call_counts)),
            "min": int(np.min(tool_call_counts)),
            "max": int(np.max(tool_call_counts)),
            "median": float(np.median(tool_call_counts)),
        }

        # Execution time statistics
        exec_times = successful_runs["execution_time"].values
        exec_time_stats = {
            "mean": float(np.mean(exec_times)),
            "std": float(np.std(exec_times)),
            "min": float(np.min(exec_times)),
            "max": float(np.max(exec_times)),
            "median": float(np.median(exec_times)),
        }

        # Cost statistics
        costs = successful_runs["total_cost"].values
        cost_stats = {
            "mean": float(np.mean(costs)),
            "std": float(np.std(costs)),
            "min": float(np.min(costs)),
            "max": float(np.max(costs)),
            "total": float(np.sum(costs)),
        }

        # Answer variability
        unique_answers = successful_runs["final_answer"].nunique()
        answer_distribution = successful_runs["final_answer"].value_counts().to_dict()

        # Tool call distribution
        all_tools = []
        for tools in successful_runs["tool_calls"]:
            all_tools.extend(tools)
        tool_distribution = dict(Counter(all_tools))

        return {
            "total_iterations": len(results_df),
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / len(results_df),
            "tool_call_stats": tool_call_stats,
            "exec_time_stats": exec_time_stats,
            "cost_stats": cost_stats,
            "unique_answers": unique_answers,
            "answer_distribution": answer_distribution,
            "tool_distribution": tool_distribution,
        }

    def save_results_to_mlflow(self, results_df: pd.DataFrame, analysis: Dict):
        """Save results and artifacts to MLflow"""
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment(self.experiment_name)

        # Suppress MLflow's emoji output that causes Windows encoding issues
        import sys

        original_stdout = sys.stdout

        try:
            # Redirect stdout temporarily to avoid emoji encoding errors on Windows
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            with mlflow.start_run(run_name=f"verification_{self.model_name}") as run:
                # Log parameters
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("num_iterations", self.num_iterations)
                mlflow.log_param("test_query", self.test_query)

                # Log metrics
                mlflow.log_metric("success_rate", analysis["success_rate"])
                mlflow.log_metric(
                    "mean_tool_calls", analysis["tool_call_stats"]["mean"]
                )
                mlflow.log_metric("std_tool_calls", analysis["tool_call_stats"]["std"])
                mlflow.log_metric("mean_exec_time", analysis["exec_time_stats"]["mean"])
                mlflow.log_metric("std_exec_time", analysis["exec_time_stats"]["std"])
                mlflow.log_metric("mean_cost", analysis["cost_stats"]["mean"])
                mlflow.log_metric("total_cost", analysis["cost_stats"]["total"])
                mlflow.log_metric("unique_answers", analysis["unique_answers"])
                mlflow.log_metric(
                    "answer_consistency",
                    1.0 - (analysis["unique_answers"] / analysis["successful_runs"]),
                )

                # Save artifacts
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save results DataFrame
                    results_path = Path(tmpdir) / "verification_results.csv"
                    results_df.to_csv(results_path, index=False)
                    mlflow.log_artifact(results_path)

                    # Save analysis JSON
                    analysis_path = Path(tmpdir) / "analysis_summary.json"
                    with open(analysis_path, "w") as f:
                        json.dump(analysis, f, indent=2)
                    mlflow.log_artifact(analysis_path)

                run_id = run.info.run_id
        finally:
            # Restore stdout
            if sys.platform == "win32":
                sys.stdout.close()
            sys.stdout = original_stdout

        logger.info(f"Results saved to MLflow run: {run_id}")
        logger.info(f"View at: {mlflow.get_tracking_uri()}/#/experiments")


@pytest.mark.integration
def test_react_agent_consistency():
    """
    Integration test to verify ReAct SQL agent behavior consistency.

    This test verifies:
    1. Agent can execute queries consistently
    2. Tool call patterns are stable (low variance)
    3. Answers are consistent across runs
    4. Execution times are reasonable
    5. No unexpected divergence in behavior
    """
    # Setup mock database
    db = MockPostgresDatabase()
    mock_db_client = db.create_mock_client()

    try:
        # Initialize and run verification
        verification = ReactAgentVerificationTest(
            model_name="gemini-2.5-flash",
            num_iterations=20,
            test_query="How many customers are in Boston?",
        )

        # Run verification
        results_df = verification.run_verification(mock_db_client)

        # Analyze results
        analysis = verification.analyze_results(results_df)

        # Save to MLflow
        verification.save_results_to_mlflow(results_df, analysis)

        # Assertions for verification
        assert analysis["success_rate"] >= 0.8, (
            f"Success rate too low: {analysis['success_rate']:.2%}. "
            "Expected at least 80% successful runs."
        )

        assert analysis["tool_call_stats"]["std"] < 2.0, (
            f"Tool call variance too high: {analysis['tool_call_stats']['std']:.2f}. "
            "ReAct agent should have consistent tool usage patterns."
        )

        assert analysis["unique_answers"] <= 3, (
            f"Too many unique answers: {analysis['unique_answers']}. "
            "Expected mostly consistent answers for deterministic query."
        )

        # Check that mean tool calls is reasonable (between 1-8 for this query)
        assert 1 <= analysis["tool_call_stats"]["mean"] <= 8, (
            f"Mean tool calls {analysis['tool_call_stats']['mean']:.1f} outside expected range. "
            "Should be between 1-8 for simple query."
        )

        logger.info("✓ All verification assertions passed")

    finally:
        db.close()


if __name__ == "__main__":
    # Run test directly
    test_react_agent_consistency()

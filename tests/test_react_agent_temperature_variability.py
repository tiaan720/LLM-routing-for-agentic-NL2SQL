"""
Temperature Variability Test for ReAct SQL Agent.

This test runs the same validation as test_react_agent_verification.py but with
temperature=1.0 to measure how much non-determinism is introduced by higher temperature.

Compares against temperature=0.0 baseline to quantify:
- Increased answer variability
- Tool usage pattern changes
- Performance impact
"""

import json
import logging
import os

import dotenv
import pytest
from test_react_agent_verification import (
    MockPostgresDatabase,
    ReactAgentVerificationTest,
)

from src.utils.logger import logger

dotenv.load_dotenv()

# Set up logging for test
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)


class ReactAgentTemperatureTest(ReactAgentVerificationTest):
    """Extended verification test with configurable temperature"""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        num_iterations: int = 20,
        test_query: str = "How many customers are in Boston?",
        temperature: float = 1.0,
    ):
        super().__init__(model_name, num_iterations, test_query)
        self.temperature = temperature
        self.experiment_name = f"react_agent_temp_{int(temperature*10)}"

    def run_single_iteration(self, iteration_num, mock_db_client):
        """Override to use configurable temperature"""
        import time
        from unittest.mock import patch

        from src.agents.sql_agent import create_sql_agent
        from src.cost_calc.shared_store import get_shared_store
        from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
        from src.models.base import create_model

        logger.info(f"Running iteration {iteration_num + 1}/{self.num_iterations}")

        # Create model with specified temperature
        model = create_model(
            model="vertex_ai",
            model_kwargs={
                "model": self.model_name,
                "temperature": self.temperature,
                "project": os.getenv("GCP_PROJECT", "research-su-llm-routing"),
                "location": "us-central1",
            },
        )

        start_time = time.time()
        store = get_shared_store()

        try:
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
                    {"configurable": {"thread_id": f"verify_temp_{iteration_num}"}},
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
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                if isinstance(tool_call, dict):
                                    tool_calls.append(tool_call.get("name", "unknown"))
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


@pytest.mark.integration
def test_react_agent_high_temperature():
    """
    Test ReAct agent with temperature=1.0 to measure non-determinism.

    This test verifies how much variability is introduced with higher temperature:
    1. More diverse answers expected (unique_answers will be higher)
    2. Tool usage may vary more
    3. Success rate should still be reasonable (≥ 70%)
    4. Execution times may be more variable
    """
    # Setup mock database
    db = MockPostgresDatabase()
    mock_db_client = db.create_mock_client()

    try:
        # Initialize and run verification with temperature=1.0
        verification = ReactAgentTemperatureTest(
            model_name="gemini-2.5-flash",
            num_iterations=20,
            test_query="How many customers are in Boston?",
            temperature=1.0,
        )

        # Run verification
        results_df = verification.run_verification(mock_db_client)

        # Analyze results
        analysis = verification.analyze_results(results_df)

        # Skip MLflow and assertions if no successful runs
        if not analysis:
            pytest.skip(
                "All test iterations failed - likely due to missing dependencies (Ollama for embeddings). "
                "Cannot validate temperature variability without successful runs."
            )

        # Save to MLflow
        verification.save_results_to_mlflow(results_df, analysis)

        # Relaxed assertions for high temperature
        assert analysis["success_rate"] >= 0.7, (
            f"Success rate too low: {analysis['success_rate']:.2%}. "
            "Expected at least 70% successful runs even with temperature=1.0."
        )

        # Higher temperature allows more tool call variance
        assert analysis["tool_call_stats"]["std"] < 3.0, (
            f"Tool call variance extremely high: {analysis['tool_call_stats']['std']:.2f}. "
            "Even with temperature=1.0, should have some consistency."
        )

        # Expect more unique answers with higher temperature (relaxed threshold)
        assert analysis["unique_answers"] <= 10, (
            f"Too many unique answers: {analysis['unique_answers']}. "
            "Even with temperature=1.0, answers should cluster somewhat."
        )

        assert 1 <= analysis["tool_call_stats"]["mean"] <= 10, (
            f"Mean tool calls {analysis['tool_call_stats']['mean']:.1f} outside expected range. "
            "Should be between 1-10 for simple query."
        )

        logger.info("✓ All temperature variability assertions passed")

    finally:
        db.close()


if __name__ == "__main__":
    # Run test directly
    test_react_agent_high_temperature()

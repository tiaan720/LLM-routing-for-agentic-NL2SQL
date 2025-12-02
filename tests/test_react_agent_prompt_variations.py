#!/usr/bin/env python3
"""
Prompt Variation Test for ReAct SQL Agent.

This test validates that different prompt instructions actually affect the agent's behavior.
Tests various prompt styles to ensure prompts are being followed:
- Double-check prompt: Agent should run extra verification queries
- Single-value prompt: Agent should return only a number
- Afrikaans prompt: Agent should reason and answer in Afrikaans
- Verbose explanation prompt: Agent should provide detailed reasoning

Results stored in MLflow for thesis verification of prompt effectiveness.
"""

import json
import logging
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import dotenv
import mlflow
import numpy as np
import pandas as pd
import pytest
from test_react_agent_verification import (
    MockPostgresDatabase,
    ReactAgentVerificationTest,
)

from src.agents.sql_agent import create_sql_agent
from src.cost_calc.shared_store import get_shared_store
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model
from src.utils.logger import logger

dotenv.load_dotenv()

# Set up logging for test
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)


# Define test prompts
TEST_PROMPTS = {
    "double_check": """You are working with PostgreSQL database 'test_db'. You are a react agent and will following the react style loop of reasoning using tool to execute and then reflecting on the tools output and you will continue this loop untill can answer the users questions. When writing PostgreSQL queries, refer to tables directly by their names (e.g., table_name) in the public schema.

CRITICAL INSTRUCTION: You MUST always double-check your results by running additional verification queries. Never trust a single query result. Always run at least 2-3 different queries to verify the answer from different angles.

Follow these steps when handling user queries:
1. First, use the postgres_list_schema tool to understand the database structure if needed
2. Form your SQL query carefully using PostgreSQL syntax
3. Use the postgres_execute_sql tool to run the query
4. IMPORTANT: Run additional verification queries to double-check your results
5. Only after verifying with multiple queries, provide your final answer
6. Explain the results to the user in a clear and concise way

Always execute your queries using the provided tools. Don't just write SQL without executing it.

CRITICAL: You MUST use postgres_execute_sql tool for every SQL query. Never show SQL code without executing it first. You cannot get answers without successfully using the tool to query the database.

PostgreSQL-specific notes:
- Use table_name directly (no schema prefix needed for public schema)
- PostgreSQL supports advanced features like window functions, CTEs, etc.
- Be aware of PostgreSQL-specific data types and functions""",
    "single_value": """You are working with PostgreSQL database 'test_db'. You are a react agent and will following the react style loop of reasoning using tool to execute and then reflecting on the tools output and you will continue this loop untill can answer the users questions. When writing PostgreSQL queries, refer to tables directly by their names (e.g., table_name) in the public schema.

CRITICAL INSTRUCTION: You MUST answer with ONLY a single value (e.g., just "3" or "5"). Do NOT provide any explanation, reasoning, or additional text. Just the number or value, nothing else.

Follow these steps when handling user queries:
1. First, use the postgres_list_schema tool to understand the database structure if needed
2. Form your SQL query carefully using PostgreSQL syntax
3. Use the postgres_execute_sql tool to run the query
4. Return ONLY the final value - no explanations, no reasoning, just the answer

Always execute your queries using the provided tools. Don't just write SQL without executing it.

CRITICAL: You MUST use postgres_execute_sql tool for every SQL query. Never show SQL code without executing it first. You cannot get answers without successfully using the tool to query the database.

PostgreSQL-specific notes:
- Use table_name directly (no schema prefix needed for public schema)
- PostgreSQL supports advanced features like window functions, CTEs, etc.
- Be aware of PostgreSQL-specific data types and functions""",
    "afrikaans": """Jy werk met PostgreSQL databasis 'test_db'. Jy is 'n react agent en sal die react-styl lus volg van redenering deur gereedskap te gebruik om uit te voer en dan te reflekteer op die gereedskap se uitset en jy sal hierdie lus voortgaan totdat jy die gebruiker se vrae kan beantwoord. Wanneer jy PostgreSQL navrae skryf, verwys na tabelle direk deur hul name (bv. table_name) in die publieke skema.

KRITIEKE INSTRUKSIE: Jy MOET alles in Afrikaans doen - jou redenering, jou verduidelikings, en jou finale antwoord. Dink in Afrikaans, redeneer in Afrikaans, en kommunikeer in Afrikaans.

Volg hierdie stappe wanneer jy gebruikersvrae hanteer:
1. Eerstens, gebruik die postgres_list_schema gereedskap om die databasis struktuur te verstaan indien nodig
2. Formuleer jou SQL navraag versigtig met PostgreSQL sintaks
3. Gebruik die postgres_execute_sql gereedskap om die navraag uit te voer
4. Verduidelik die resultate aan die gebruiker in Afrikaans op 'n duidelike en beknopte manier

Voer altyd jou navrae uit deur die verskafde gereedskap te gebruik. Moenie net SQL skryf sonder om dit uit te voer nie.

KRITIEK: Jy MOET postgres_execute_sql gereedskap gebruik vir elke SQL navraag. Moenie SQL kode wys sonder om dit eerste uit te voer nie. Jy kan nie antwoorde kry sonder om die gereedskap suksesvol te gebruik om die databasis te ondervra nie.

PostgreSQL-spesifieke notas:
- Gebruik table_name direk (geen skema voorvoegsel nodig vir publieke skema nie)
- PostgreSQL ondersteun gevorderde funksies soos venster funksies, CTEs, ens.
- Wees bewus van PostgreSQL-spesifieke data tipes en funksies""",
    "verbose_explanation": """You are working with PostgreSQL database 'test_db'. You are a react agent and will following the react style loop of reasoning using tool to execute and then reflecting on the tools output and you will continue this loop untill can answer the users questions. When writing PostgreSQL queries, refer to tables directly by their names (e.g., table_name) in the public schema.

CRITICAL INSTRUCTION: You MUST provide extremely detailed, verbose explanations. For every step, explain:
- Why you're taking that action
- What you expect to find
- How the query works
- What the results mean
- How this relates to the user's question
Be thorough and educational in your explanations.

Follow these steps when handling user queries:
1. First, use the postgres_list_schema tool to understand the database structure if needed - EXPLAIN why you need this
2. Form your SQL query carefully using PostgreSQL syntax - EXPLAIN your query design choices
3. Use the postgres_execute_sql tool to run the query - EXPLAIN what you expect
4. EXPLAIN the results in detail, breaking down each aspect
5. Provide a comprehensive final answer with full context

Always execute your queries using the provided tools. Don't just write SQL without executing it.

CRITICAL: You MUST use postgres_execute_sql tool for every SQL query. Never show SQL code without executing it first. You cannot get answers without successfully using the tool to query the database.

PostgreSQL-specific notes:
- Use table_name directly (no schema prefix needed for public schema)
- PostgreSQL supports advanced features like window functions, CTEs, etc.
- Be aware of PostgreSQL-specific data types and functions""",
}


class ReactAgentPromptVariationTest(ReactAgentVerificationTest):
    """Test suite for validating different prompt variations"""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        num_iterations: int = 10,
        test_query: str = "How many customers are in Boston?",
        prompt_name: str = "baseline",
        custom_prompt: str = None,
    ):
        super().__init__(model_name, num_iterations, test_query)
        self.prompt_name = prompt_name
        self.custom_prompt = custom_prompt
        self.experiment_name = f"react_agent_prompt_{prompt_name}"

    def run_single_iteration(self, iteration_num, mock_db_client):
        """Override to use custom prompt"""
        from langchain_core.prompts import ChatPromptTemplate

        logger.info(
            f"Running iteration {iteration_num + 1}/{self.num_iterations} with prompt: {self.prompt_name}"
        )

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

        start_time = time.time()
        store = get_shared_store()

        try:
            with patch(
                "src.agents.sql_agent.create_database_client",
                return_value=mock_db_client,
            ):
                # Patch get_system_prompt_for_db_type to return our custom prompt
                def custom_prompt_func(**kwargs):
                    return self.custom_prompt

                with patch(
                    "src.agents.sql_agent.get_system_prompt_for_db_type",
                    side_effect=custom_prompt_func,
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
                        {
                            "configurable": {
                                "thread_id": f"prompt_{self.prompt_name}_{iteration_num}"
                            }
                        },
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
                "answer_length": len(final_answer) if final_answer else 0,
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
                "answer_length": 0,
            }

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Analyze results with prompt-specific metrics"""
        analysis = super().analyze_results(results_df)

        # Add prompt-specific analysis
        successful_runs = results_df[results_df["success"] == True]

        if len(successful_runs) > 0:
            # Answer length statistics (useful for verbose/single-value prompts)
            answer_lengths = successful_runs["answer_length"].values
            analysis["answer_length_stats"] = {
                "mean": float(np.mean(answer_lengths)),
                "std": float(np.std(answer_lengths)),
                "min": int(np.min(answer_lengths)),
                "max": int(np.max(answer_lengths)),
                "median": float(np.median(answer_lengths)),
            }

            # Check for Afrikaans content (simple heuristic)
            afrikaans_indicators = ["Daar", "die", "kliënte", "is", "van", "in"]
            afrikaans_count = sum(
                1
                for answer in successful_runs["final_answer"]
                if any(word in answer for word in afrikaans_indicators)
            )
            analysis["afrikaans_ratio"] = afrikaans_count / len(successful_runs)

            # Check for single-value answers (answer length < 10 chars)
            single_value_count = sum(1 for length in answer_lengths if length < 10)
            analysis["single_value_ratio"] = single_value_count / len(successful_runs)

        return analysis

    def save_results_to_mlflow(self, results_df: pd.DataFrame, analysis: Dict):
        """Save prompt variation results to MLflow"""
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment(self.experiment_name)

        import sys

        original_stdout = sys.stdout

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            with mlflow.start_run(
                run_name=f"prompt_{self.prompt_name}_{self.model_name}"
            ) as run:
                # Log parameters
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("num_iterations", self.num_iterations)
                mlflow.log_param("test_query", self.test_query)
                mlflow.log_param("prompt_name", self.prompt_name)

                # Log standard metrics
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

                # Log prompt-specific metrics
                if "answer_length_stats" in analysis:
                    mlflow.log_metric(
                        "mean_answer_length", analysis["answer_length_stats"]["mean"]
                    )
                    mlflow.log_metric(
                        "std_answer_length", analysis["answer_length_stats"]["std"]
                    )
                if "afrikaans_ratio" in analysis:
                    mlflow.log_metric("afrikaans_ratio", analysis["afrikaans_ratio"])
                if "single_value_ratio" in analysis:
                    mlflow.log_metric(
                        "single_value_ratio", analysis["single_value_ratio"]
                    )

                # Save artifacts
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save results DataFrame
                    results_path = (
                        Path(tmpdir) / f"prompt_{self.prompt_name}_results.csv"
                    )
                    results_df.to_csv(results_path, index=False)
                    mlflow.log_artifact(results_path)

                    # Save analysis JSON
                    analysis_path = (
                        Path(tmpdir) / f"prompt_{self.prompt_name}_analysis.json"
                    )
                    with open(analysis_path, "w") as f:
                        json.dump(analysis, f, indent=2)
                    mlflow.log_artifact(analysis_path)

                    # Save the prompt used
                    prompt_path = Path(tmpdir) / f"prompt_{self.prompt_name}_text.txt"
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write(self.custom_prompt)
                    mlflow.log_artifact(prompt_path)

                run_id = run.info.run_id
        finally:
            if sys.platform == "win32":
                sys.stdout.close()
            sys.stdout = original_stdout

        logger.info(f"Prompt variation results saved to MLflow run: {run_id}")


@pytest.mark.integration
def test_prompt_double_check():
    """Test that double-check prompt increases tool calls"""
    db = MockPostgresDatabase()
    mock_db_client = db.create_mock_client()

    try:
        verification = ReactAgentPromptVariationTest(
            model_name="gemini-2.5-flash",
            num_iterations=10,
            test_query="How many customers are in Boston?",
            prompt_name="double_check",
            custom_prompt=TEST_PROMPTS["double_check"],
        )

        results_df = verification.run_verification(mock_db_client)
        analysis = verification.analyze_results(results_df)
        verification.save_results_to_mlflow(results_df, analysis)

        # Assertions: Should have more tool calls due to double-checking
        assert (
            analysis["success_rate"] >= 0.7
        ), f"Success rate too low: {analysis['success_rate']:.2%}"
        assert (
            analysis["tool_call_stats"]["mean"] >= 2.0
        ), f"Expected more tool calls for double-check prompt, got {analysis['tool_call_stats']['mean']:.2f}"

        logger.info(
            f"✓ Double-check prompt test passed (mean tool calls: {analysis['tool_call_stats']['mean']:.2f})"
        )

    finally:
        db.close()


@pytest.mark.integration
def test_prompt_single_value():
    """Test that single-value prompt produces short answers"""
    db = MockPostgresDatabase()
    mock_db_client = db.create_mock_client()

    try:
        verification = ReactAgentPromptVariationTest(
            model_name="gemini-2.5-flash",
            num_iterations=10,
            test_query="How many customers are in Boston?",
            prompt_name="single_value",
            custom_prompt=TEST_PROMPTS["single_value"],
        )

        results_df = verification.run_verification(mock_db_client)
        analysis = verification.analyze_results(results_df)
        verification.save_results_to_mlflow(results_df, analysis)

        # Assertions: Should have very short answers
        assert (
            analysis["success_rate"] >= 0.7
        ), f"Success rate too low: {analysis['success_rate']:.2%}"
        assert (
            analysis["answer_length_stats"]["mean"] < 50
        ), f"Expected short answers for single-value prompt, got mean length {analysis['answer_length_stats']['mean']:.1f}"

        logger.info(
            f"✓ Single-value prompt test passed (mean answer length: {analysis['answer_length_stats']['mean']:.1f})"
        )

    finally:
        db.close()


@pytest.mark.integration
def test_prompt_afrikaans():
    """Test that Afrikaans prompt produces Afrikaans responses"""
    db = MockPostgresDatabase()
    mock_db_client = db.create_mock_client()

    try:
        verification = ReactAgentPromptVariationTest(
            model_name="gemini-2.5-flash",
            num_iterations=10,
            test_query="How many customers are in Boston?",
            prompt_name="afrikaans",
            custom_prompt=TEST_PROMPTS["afrikaans"],
        )

        results_df = verification.run_verification(mock_db_client)
        analysis = verification.analyze_results(results_df)
        verification.save_results_to_mlflow(results_df, analysis)

        # Assertions: Should have Afrikaans content
        assert (
            analysis["success_rate"] >= 0.7
        ), f"Success rate too low: {analysis['success_rate']:.2%}"
        assert (
            analysis["afrikaans_ratio"] >= 0.5
        ), f"Expected Afrikaans responses, got ratio {analysis['afrikaans_ratio']:.2%}"

        logger.info(
            f"✓ Afrikaans prompt test passed (Afrikaans ratio: {analysis['afrikaans_ratio']:.2%})"
        )

    finally:
        db.close()


@pytest.mark.integration
def test_prompt_verbose_explanation():
    """Test that verbose prompt produces longer, detailed answers"""
    db = MockPostgresDatabase()
    mock_db_client = db.create_mock_client()

    try:
        verification = ReactAgentPromptVariationTest(
            model_name="gemini-2.5-flash",
            num_iterations=10,
            test_query="How many customers are in Boston?",
            prompt_name="verbose_explanation",
            custom_prompt=TEST_PROMPTS["verbose_explanation"],
        )

        results_df = verification.run_verification(mock_db_client)
        analysis = verification.analyze_results(results_df)
        verification.save_results_to_mlflow(results_df, analysis)

        # Assertions: Should have longer, detailed answers
        assert (
            analysis["success_rate"] >= 0.7
        ), f"Success rate too low: {analysis['success_rate']:.2%}"
        assert (
            analysis["answer_length_stats"]["mean"] > 100
        ), f"Expected verbose answers, got mean length {analysis['answer_length_stats']['mean']:.1f}"

        logger.info(
            f"✓ Verbose prompt test passed (mean answer length: {analysis['answer_length_stats']['mean']:.1f})"
        )

    finally:
        db.close()


@pytest.mark.integration
def test_all_prompt_variations():
    """Run all prompt variations in sequence for comparison"""
    db = MockPostgresDatabase()
    mock_db_client = db.create_mock_client()

    try:
        results_summary = {}

        for prompt_name, prompt_text in TEST_PROMPTS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing prompt variation: {prompt_name}")
            logger.info(f"{'='*60}")

            verification = ReactAgentPromptVariationTest(
                model_name="gemini-2.5-flash",
                num_iterations=10,
                test_query="How many customers are in Boston?",
                prompt_name=prompt_name,
                custom_prompt=prompt_text,
            )

            results_df = verification.run_verification(mock_db_client)
            analysis = verification.analyze_results(results_df)
            verification.save_results_to_mlflow(results_df, analysis)

            # Store summary for comparison
            results_summary[prompt_name] = {
                "success_rate": analysis["success_rate"],
                "mean_tool_calls": analysis["tool_call_stats"]["mean"],
                "mean_answer_length": analysis.get("answer_length_stats", {}).get(
                    "mean", 0
                ),
                "afrikaans_ratio": analysis.get("afrikaans_ratio", 0),
                "single_value_ratio": analysis.get("single_value_ratio", 0),
            }

        # Log comparison summary
        logger.info(f"\n{'='*60}")
        logger.info("PROMPT VARIATION COMPARISON SUMMARY")
        logger.info(f"{'='*60}")
        for prompt_name, metrics in results_summary.items():
            logger.info(f"\n{prompt_name}:")
            logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
            logger.info(f"  Mean Tool Calls: {metrics['mean_tool_calls']:.2f}")
            logger.info(f"  Mean Answer Length: {metrics['mean_answer_length']:.1f}")
            if metrics["afrikaans_ratio"] > 0:
                logger.info(f"  Afrikaans Ratio: {metrics['afrikaans_ratio']:.2%}")
            if metrics["single_value_ratio"] > 0:
                logger.info(
                    f"  Single Value Ratio: {metrics['single_value_ratio']:.2%}"
                )

        logger.info(f"\n{'='*60}")
        logger.info("✓ All prompt variation tests completed")

    finally:
        db.close()


if __name__ == "__main__":
    # Run all prompt variation tests
    test_all_prompt_variations()

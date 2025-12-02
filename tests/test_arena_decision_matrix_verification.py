#!/usr/bin/env python3
"""
Arena Decision Matrix Verification Test.

This test validates the arena scoring and winner selection logic by:
- Testing with mock model data
- Verifying winner selection is consistent
- Analyzing impact of different weight configurations
- Ensuring edge cases are handled correctly

Results output to terminal for quick verification and saved to MLflow.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import dotenv
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.utils.logger import logger

dotenv.load_dotenv()

# Set up logging for test
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)


@pytest.fixture(autouse=True)
def cleanup_mlflow_run():
    """Ensure MLflow run is ended after each test"""
    yield
    # Cleanup after test
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


class MockArenaData:
    """Generate mock evaluation data for testing arena logic"""

    def __init__(self, num_questions: int = 10, num_models: int = 5):
        self.num_questions = num_questions
        self.num_models = num_models
        self.model_names = [f"model_{chr(65+i)}" for i in range(num_models)]
        self.questions = [f"question_{i+1}" for i in range(num_questions)]

    def generate_balanced_data(self) -> Dict[str, pd.DataFrame]:
        """Generate balanced mock data where all models perform similarly"""
        model_data = {}

        for model_name in self.model_names:
            rows = []
            for question in self.questions:
                rows.append(
                    {
                        "question": question,
                        "query_cost": np.random.uniform(0.001, 0.005),
                        "execution_time": np.random.uniform(1.0, 3.0),
                        "sql_accuracy/v1/score": np.random.uniform(3.0, 5.0),
                        "answer_correctness/v1/score": np.random.uniform(3.0, 5.0),
                        "faithfulness/v1/score": np.random.uniform(3.0, 5.0),
                    }
                )
            model_data[model_name] = pd.DataFrame(rows)

        return model_data

    def generate_dominant_model_data(
        self, dominant_model: str = "model_A"
    ) -> Dict[str, pd.DataFrame]:
        """Generate data where one model clearly dominates"""
        model_data = {}

        for model_name in self.model_names:
            rows = []
            for question in self.questions:
                if model_name == dominant_model:
                    # Dominant model has best scores
                    rows.append(
                        {
                            "question": question,
                            "query_cost": 0.001,  # Lowest cost
                            "execution_time": 0.5,  # Fastest
                            "sql_accuracy/v1/score": 5.0,  # Perfect accuracy
                            "answer_correctness/v1/score": 5.0,  # Perfect correctness
                            "faithfulness/v1/score": 5.0,  # Perfect faithfulness
                        }
                    )
                else:
                    # Other models have worse scores
                    rows.append(
                        {
                            "question": question,
                            "query_cost": np.random.uniform(0.003, 0.01),
                            "execution_time": np.random.uniform(2.0, 5.0),
                            "sql_accuracy/v1/score": np.random.uniform(2.0, 3.5),
                            "answer_correctness/v1/score": np.random.uniform(2.0, 3.5),
                            "faithfulness/v1/score": np.random.uniform(2.0, 3.5),
                        }
                    )
            model_data[model_name] = pd.DataFrame(rows)

        return model_data

    def generate_specialized_models_data(self) -> Dict[str, pd.DataFrame]:
        """Generate data where models excel in different areas"""
        model_data = {}

        specializations = {
            "model_A": "cost",  # Cheapest
            "model_B": "speed",  # Fastest
            "model_C": "accuracy",  # Most accurate
            "model_D": "correctness",  # Best answers
            "model_E": "balanced",  # Balanced across all
        }

        for model_name in self.model_names:
            rows = []
            specialty = specializations.get(model_name, "balanced")

            for question in self.questions:
                if specialty == "cost":
                    row = {
                        "query_cost": 0.0005,  # Extremely cheap
                        "execution_time": 3.0,
                        "sql_accuracy/v1/score": 3.0,
                        "answer_correctness/v1/score": 3.0,
                        "faithfulness/v1/score": 3.0,
                    }
                elif specialty == "speed":
                    row = {
                        "query_cost": 0.004,
                        "execution_time": 0.3,  # Extremely fast
                        "sql_accuracy/v1/score": 3.0,
                        "answer_correctness/v1/score": 3.0,
                        "faithfulness/v1/score": 3.0,
                    }
                elif specialty == "accuracy":
                    row = {
                        "query_cost": 0.005,
                        "execution_time": 3.0,
                        "sql_accuracy/v1/score": 5.0,  # Perfect SQL
                        "answer_correctness/v1/score": 3.0,
                        "faithfulness/v1/score": 3.0,
                    }
                elif specialty == "correctness":
                    row = {
                        "query_cost": 0.005,
                        "execution_time": 3.0,
                        "sql_accuracy/v1/score": 3.0,
                        "answer_correctness/v1/score": 5.0,  # Perfect answers
                        "faithfulness/v1/score": 5.0,  # Perfect faithfulness
                    }
                else:  # balanced
                    row = {
                        "query_cost": 0.003,
                        "execution_time": 2.0,
                        "sql_accuracy/v1/score": 4.0,
                        "answer_correctness/v1/score": 4.0,
                        "faithfulness/v1/score": 4.0,
                    }

                row["question"] = question
                rows.append(row)

            model_data[model_name] = pd.DataFrame(rows)

        return model_data


class ArenaDecisionMatrixVerifier:
    """Verifies arena decision matrix logic with different weight configurations"""

    def __init__(self, weights: Dict[str, float] = None):
        # Default weights from the arena code
        self.weights = weights or {
            "query_cost": 0.20,
            "execution_time": 0.10,
            "sql_accuracy": 0.10,
            "answer_correctness": 0.40,
            "faithfulness": 0.10,
            "arena_hard_score": 0.10,
        }
        self.arena_scores = {}  # Mock arena hard scores

    def set_arena_scores(self, scores: Dict[str, float]):
        """Set mock arena hard scores for models"""
        self.arena_scores = scores

    def get_arena_hard_score(self, model_name: str) -> float:
        """Get arena hard score for model"""
        return self.arena_scores.get(model_name, 50.0)  # Default to 50.0

    def _normalize_series(self, s: pd.Series, higher_is_better: bool) -> pd.Series:
        """Normalize series to 0-1 range"""
        s_clean = s.astype(float)
        min_v = s_clean.min()
        max_v = s_clean.max()
        if pd.isna(min_v) or pd.isna(max_v):
            return pd.Series([0.5] * len(s), index=s.index)
        if max_v - min_v == 0:
            return pd.Series([0.5] * len(s), index=s.index)
        if higher_is_better:
            return (s_clean - min_v) / (max_v - min_v)
        else:
            return (max_v - s_clean) / (max_v - min_v)

    def score_question(
        self, question: str, model_data: Dict[str, pd.DataFrame]
    ) -> Tuple[str, Dict[str, float]]:
        """Score all models for a single question and return winner"""
        rows = {}
        for model_name, df in model_data.items():
            try:
                row = df[df["question"] == question].iloc[0]
            except Exception:
                continue
            rows[model_name] = row

        if len(rows) < 2:
            return None, {}

        # Filter out models with answer correctness score <= 1.0
        ANSWER_COL = "answer_correctness/v1/score"
        filtered_rows = {}
        for m, r in rows.items():
            try:
                val = pd.to_numeric(r.get(ANSWER_COL), errors="coerce")
            except Exception:
                val = None
            if pd.isna(val) or val > 1.0:
                filtered_rows[m] = r
        rows = filtered_rows

        if len(rows) < 2:
            return None, {}

        # Collect metric series
        def collect(col_name: str) -> pd.Series:
            data = {m: rows[m].get(col_name) for m in rows}
            return pd.Series(data, dtype=float)

        ser_cost = collect("query_cost")
        ser_time = collect("execution_time")
        ser_sql = collect("sql_accuracy/v1/score")
        ser_ans = collect("answer_correctness/v1/score")
        ser_faith = collect("faithfulness/v1/score")
        ser_arena = pd.Series({m: self.get_arena_hard_score(m) for m in rows})

        # Normalize metrics
        arena_min = min(self.arena_scores.values()) if self.arena_scores else 0.0
        arena_max = max(self.arena_scores.values()) if self.arena_scores else 100.0

        def norm_arena(val: float) -> float:
            if arena_max - arena_min == 0:
                return 0.5
            return (val - arena_min) / (arena_max - arena_min)

        norm = {
            "query_cost": self._normalize_series(ser_cost, higher_is_better=False),
            "execution_time": self._normalize_series(ser_time, higher_is_better=False),
            "sql_accuracy": self._normalize_series(ser_sql, higher_is_better=True),
            "answer_correctness": self._normalize_series(
                ser_ans, higher_is_better=True
            ),
            "faithfulness": self._normalize_series(ser_faith, higher_is_better=True),
            "arena_hard_score": ser_arena.map(norm_arena),
        }

        # Calculate composite scores
        composite = pd.Series({m: 0.0 for m in rows})
        for key, weight in self.weights.items():
            composite += norm[key] * weight

        # Get winner
        ranking = composite.sort_values(ascending=False)
        winner = ranking.index[0] if len(ranking) > 0 else None

        return winner, composite.to_dict()

    def verify_dataset(
        self, model_data: Dict[str, pd.DataFrame], dataset_name: str
    ) -> Dict:
        """Verify entire dataset and return statistics"""
        questions = list(next(iter(model_data.values()))["question"])
        winner_counts = {}
        all_scores = []

        for question in questions:
            winner, scores = self.score_question(question, model_data)
            if winner:
                winner_counts[winner] = winner_counts.get(winner, 0) + 1
                all_scores.append(scores)

        return {
            "dataset_name": dataset_name,
            "total_questions": len(questions),
            "winner_distribution": winner_counts,
            "all_scores": all_scores,
        }

    def compare_weight_configurations(
        self, model_data: Dict[str, pd.DataFrame], weight_configs: List[Dict[str, Dict]]
    ) -> pd.DataFrame:
        """Compare results across different weight configurations"""
        results = []

        for config in weight_configs:
            self.weights = config["weights"]
            stats = self.verify_dataset(model_data, config["name"])

            results.append(
                {
                    "config_name": config["name"],
                    "winner_distribution": stats["winner_distribution"],
                    "winner": (
                        max(stats["winner_distribution"].items(), key=lambda x: x[1])[0]
                        if stats["winner_distribution"]
                        else None
                    ),
                    "win_count": (
                        max(stats["winner_distribution"].values())
                        if stats["winner_distribution"]
                        else 0
                    ),
                    "total_questions": stats["total_questions"],
                }
            )

        return pd.DataFrame(results)

    def calculate_model_stats(
        self, model_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Calculate average statistics for each model"""
        model_stats = {}

        for model_name, df in model_data.items():
            stats = {
                "num_questions": len(df),
            }

            # Calculate averages for key metrics
            metrics = [
                "query_cost",
                "execution_time",
                "sql_accuracy/v1/score",
                "answer_correctness/v1/score",
                "faithfulness/v1/score",
            ]

            for metric in metrics:
                if metric in df.columns:
                    values = pd.to_numeric(df[metric], errors="coerce")
                    stats[f"avg_{metric.replace('/', '_')}"] = round(values.mean(), 4)
                    stats[f"min_{metric.replace('/', '_')}"] = round(values.min(), 4)
                    stats[f"max_{metric.replace('/', '_')}"] = round(values.max(), 4)

            model_stats[model_name] = stats

        return model_stats


class MLflowVerificationTracker:
    """Tracks all verification tests in a single MLflow run"""

    def __init__(self):
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment("arena_decision_matrix_verification")
        self.run = None
        self.original_stdout = None
        self.all_results = {}

    def start_run(self):
        """Start MLflow run"""
        self.original_stdout = sys.stdout

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            self.run = mlflow.start_run(run_name="arena_verification_complete")
        finally:
            if sys.platform == "win32" and sys.stdout != self.original_stdout:
                sys.stdout.close()
            sys.stdout = self.original_stdout

    def log_test_result(
        self,
        test_name: str,
        stats: Dict,
        weight_config: Dict[str, float] = None,
        model_stats: Dict[str, Dict] = None,
    ):
        """Log individual test results to the current run"""
        if not self.run:
            return

        # Store results for later artifact generation
        self.all_results[test_name] = {
            "stats": stats,
            "weights": weight_config,
            "model_stats": model_stats,
        }

        # Log metrics with test name prefix
        prefix = test_name.replace(" ", "_").lower()

        mlflow.log_param(f"{prefix}_total_questions", stats["total_questions"])

        if weight_config:
            for metric, weight in weight_config.items():
                mlflow.log_param(f"{prefix}_weight_{metric}", weight)

        winner_counts = stats["winner_distribution"]
        if winner_counts:
            max_wins = max(winner_counts.values())
            win_percentage = max_wins / stats["total_questions"]

            mlflow.log_metric(f"{prefix}_max_wins", max_wins)
            mlflow.log_metric(f"{prefix}_win_percentage", win_percentage)
            mlflow.log_metric(f"{prefix}_unique_winners", len(winner_counts))

            # Log individual model win counts
            for model, count in winner_counts.items():
                mlflow.log_metric(f"{prefix}_wins_{model}", count)

    def end_run(self):
        """End MLflow run and save all artifacts"""
        if not self.run:
            return

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            # Build comprehensive overview
            overview = {
                "experiment_name": "arena_decision_matrix_verification",
                "run_date": pd.Timestamp.now().isoformat(),
                "total_tests": len(self.all_results),
                "tests": {},
                "weight_impact_analysis": {},
            }

            # Track weight configurations and their impacts
            weight_configs_seen = {}

            # Add each test result
            for test_name, result in self.all_results.items():
                stats = result["stats"]
                weights = result["weights"]
                model_stats = result.get("model_stats", {})
                winner_dist = stats["winner_distribution"]

                test_summary = {
                    "total_questions": stats["total_questions"],
                    "unique_winners": len(winner_dist),
                    "winner_distribution": winner_dist,
                }

                # Add winner info
                if winner_dist:
                    winner_model = max(winner_dist.items(), key=lambda x: x[1])[0]
                    winner_count = winner_dist[winner_model]
                    test_summary["primary_winner"] = winner_model
                    test_summary["winner_count"] = winner_count
                    test_summary["win_percentage"] = round(
                        winner_count / stats["total_questions"] * 100, 1
                    )

                # Add weights if present
                if weights:
                    test_summary["weights"] = weights

                    # Track weight impact
                    weight_key = json.dumps(weights, sort_keys=True)
                    if weight_key not in weight_configs_seen:
                        weight_configs_seen[weight_key] = {
                            "weights": weights,
                            "tests": [],
                            "winner_distributions": [],
                        }
                    weight_configs_seen[weight_key]["tests"].append(test_name)
                    weight_configs_seen[weight_key]["winner_distributions"].append(
                        {
                            "test": test_name,
                            "distribution": winner_dist,
                            "winner": winner_model if winner_dist else None,
                        }
                    )

                # Add model input statistics
                if model_stats:
                    test_summary["model_input_stats"] = model_stats

                overview["tests"][test_name] = test_summary

            # Add weight impact analysis
            if weight_configs_seen:
                overview["weight_impact_analysis"] = {
                    "description": "Shows how different weight configurations affect winner selection",
                    "total_configurations": len(weight_configs_seen),
                    "configurations": [],
                }

                for idx, (weight_key, config_data) in enumerate(
                    weight_configs_seen.items(), 1
                ):
                    config_summary = {
                        "config_id": idx,
                        "weights": config_data["weights"],
                        "tests_using_config": config_data["tests"],
                        "winner_distributions": config_data["winner_distributions"],
                    }

                    # Calculate overall winner across all tests with this config
                    all_wins = {}
                    for dist_data in config_data["winner_distributions"]:
                        for model, count in dist_data["distribution"].items():
                            all_wins[model] = all_wins.get(model, 0) + count

                    if all_wins:
                        overall_winner = max(all_wins.items(), key=lambda x: x[1])
                        config_summary["overall_winner"] = overall_winner[0]
                        config_summary["overall_winner_count"] = overall_winner[1]
                        config_summary["total_questions_all_tests"] = sum(
                            all_wins.values()
                        )

                    overview["weight_impact_analysis"]["configurations"].append(
                        config_summary
                    )

            # Save single comprehensive artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                overview_path = Path(tmpdir) / "arena_verification_overview.json"
                with open(overview_path, "w") as f:
                    json.dump(overview, f, indent=2)
                mlflow.log_artifact(overview_path)

            run_id = self.run.info.run_id
            mlflow.end_run()
            self.run = None  # Clear the run reference

            logger.info(f"\n{'='*60}")
            logger.info(f"All verification results saved to MLflow run: {run_id}")
            logger.info(f"View at: {mlflow.get_tracking_uri()}/#/experiments")
            logger.info(f"Total tests completed: {len(self.all_results)}")
            if weight_configs_seen:
                logger.info(f"Weight configurations tested: {len(weight_configs_seen)}")
            logger.info(f"{'='*60}")
        finally:
            if sys.platform == "win32" and sys.stdout != self.original_stdout:
                sys.stdout.close()
            sys.stdout = self.original_stdout


# Global tracker instance
_mlflow_tracker = None


def get_mlflow_tracker():
    """Get or create global MLflow tracker"""
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowVerificationTracker()
    return _mlflow_tracker


@pytest.mark.integration
def test_arena_balanced_data():
    """Test that balanced data produces roughly even winner distribution"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Balanced Data - All Models Perform Similarly")
    logger.info("=" * 60)

    tracker = get_mlflow_tracker()
    if tracker.run is None:
        tracker.start_run()

    mock_data = MockArenaData(num_questions=20, num_models=5)
    model_data = mock_data.generate_balanced_data()

    verifier = ArenaDecisionMatrixVerifier()
    verifier.set_arena_scores({model: 50.0 for model in mock_data.model_names})

    stats = verifier.verify_dataset(model_data, "balanced")
    model_stats = verifier.calculate_model_stats(model_data)

    logger.info(f"Total questions: {stats['total_questions']}")
    logger.info(f"Winner distribution: {stats['winner_distribution']}")

    # With balanced data, no single model should dominate (max wins < 50%)
    max_wins = max(stats["winner_distribution"].values())
    win_percentage = max_wins / stats["total_questions"]

    logger.info(f"Max wins: {max_wins} ({win_percentage:.1%})")

    # Save to MLflow
    tracker.log_test_result("balanced_data_test", stats, model_stats=model_stats)

    assert (
        win_percentage < 0.6
    ), f"With balanced data, no model should win more than 60%, got {win_percentage:.1%}"
    logger.info("✓ Balanced data test passed")


@pytest.mark.integration
def test_arena_dominant_model():
    """Test that a clearly dominant model wins most questions"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Dominant Model - Model A Should Win Most")
    logger.info("=" * 60)

    tracker = get_mlflow_tracker()
    if tracker.run is None:
        tracker.start_run()

    mock_data = MockArenaData(num_questions=20, num_models=5)
    model_data = mock_data.generate_dominant_model_data(dominant_model="model_A")

    verifier = ArenaDecisionMatrixVerifier()
    verifier.set_arena_scores({model: 50.0 for model in mock_data.model_names})

    stats = verifier.verify_dataset(model_data, "dominant_model_A")
    model_stats = verifier.calculate_model_stats(model_data)

    logger.info(f"Total questions: {stats['total_questions']}")
    logger.info(f"Winner distribution: {stats['winner_distribution']}")

    # Dominant model should win majority
    model_a_wins = stats["winner_distribution"].get("model_A", 0)
    win_percentage = model_a_wins / stats["total_questions"]

    logger.info(f"Model A wins: {model_a_wins} ({win_percentage:.1%})")

    # Save to MLflow
    tracker.log_test_result("dominant_model_test", stats, model_stats=model_stats)

    assert (
        win_percentage >= 0.7
    ), f"Dominant model should win at least 70%, got {win_percentage:.1%}"
    logger.info("✓ Dominant model test passed")


@pytest.mark.integration
def test_arena_specialized_models():
    """Test specialized models with different weight configurations"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Specialized Models - Different Weights Impact")
    logger.info("=" * 60)

    tracker = get_mlflow_tracker()
    if tracker.run is None:
        tracker.start_run()

    mock_data = MockArenaData(num_questions=20, num_models=5)
    model_data = mock_data.generate_specialized_models_data()

    # Define weight configurations that favor different specializations
    weight_configs = [
        {
            "name": "Default (Correctness-Heavy)",
            "weights": {
                "query_cost": 0.20,
                "execution_time": 0.10,
                "sql_accuracy": 0.10,
                "answer_correctness": 0.40,
                "faithfulness": 0.10,
                "arena_hard_score": 0.10,
            },
        },
        {
            "name": "Cost-Optimized",
            "weights": {
                "query_cost": 0.50,
                "execution_time": 0.20,
                "sql_accuracy": 0.10,
                "answer_correctness": 0.10,
                "faithfulness": 0.05,
                "arena_hard_score": 0.05,
            },
        },
        {
            "name": "Speed-Optimized",
            "weights": {
                "query_cost": 0.10,
                "execution_time": 0.50,
                "sql_accuracy": 0.10,
                "answer_correctness": 0.20,
                "faithfulness": 0.05,
                "arena_hard_score": 0.05,
            },
        },
        {
            "name": "Quality-Focused",
            "weights": {
                "query_cost": 0.05,
                "execution_time": 0.05,
                "sql_accuracy": 0.20,
                "answer_correctness": 0.40,
                "faithfulness": 0.20,
                "arena_hard_score": 0.10,
            },
        },
    ]

    verifier = ArenaDecisionMatrixVerifier()
    verifier.set_arena_scores({model: 50.0 for model in mock_data.model_names})

    comparison_df = verifier.compare_weight_configurations(model_data, weight_configs)
    model_stats = verifier.calculate_model_stats(model_data)

    logger.info("\nWeight Configuration Comparison:")
    for _, row in comparison_df.iterrows():
        logger.info(f"\n{row['config_name']}:")
        logger.info(
            f"  Winner: {row['winner']} ({row['win_count']}/{row['total_questions']} wins)"
        )
        logger.info(f"  Distribution: {row['winner_distribution']}")

    # Verify different weights produce different winners
    unique_winners = comparison_df["winner"].nunique()
    logger.info(f"\nUnique winners across configurations: {unique_winners}")

    # Save each configuration to MLflow
    for config in weight_configs:
        verifier.weights = config["weights"]
        stats = verifier.verify_dataset(model_data, config["name"])
        tracker.log_test_result(
            f"specialized_models_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
            stats,
            weight_config=config["weights"],
            model_stats=model_stats,
        )

    assert (
        unique_winners >= 2
    ), "Different weight configurations should produce different winners"
    logger.info("✓ Specialized models test passed")


@pytest.mark.integration
def test_arena_answer_correctness_filter():
    """Test that models with answer_correctness <= 1.0 are filtered out"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Answer Correctness Filter - Models with score ≤ 1.0 excluded")
    logger.info("=" * 60)

    tracker = get_mlflow_tracker()
    if tracker.run is None:
        tracker.start_run()

    mock_data = MockArenaData(num_questions=10, num_models=3)
    model_data = mock_data.generate_balanced_data()

    # Manually set one model to have low answer correctness scores
    model_data["model_A"]["answer_correctness/v1/score"] = 1.0  # Should be filtered
    model_data["model_B"]["answer_correctness/v1/score"] = 3.0  # Should pass
    model_data["model_C"]["answer_correctness/v1/score"] = 4.0  # Should pass

    verifier = ArenaDecisionMatrixVerifier()
    verifier.set_arena_scores({model: 50.0 for model in mock_data.model_names})

    stats = verifier.verify_dataset(model_data, "filtered")
    model_stats = verifier.calculate_model_stats(model_data)

    logger.info(f"Winner distribution: {stats['winner_distribution']}")

    # Model A should never win (filtered out)
    model_a_wins = stats["winner_distribution"].get("model_A", 0)

    logger.info(f"Model A wins (should be 0): {model_a_wins}")

    # Save to MLflow
    tracker.log_test_result(
        "answer_correctness_filter_test", stats, model_stats=model_stats
    )

    assert (
        model_a_wins == 0
    ), "Model with answer_correctness = 1.0 should not win any questions"
    logger.info("✓ Answer correctness filter test passed")


@pytest.mark.integration
def test_all_arena_verifications():
    """Run all verification tests and output summary"""
    logger.info("\n" + "=" * 60)
    logger.info("ARENA DECISION MATRIX VERIFICATION SUMMARY")
    logger.info("=" * 60)

    tracker = get_mlflow_tracker()

    # Only start a new run if one doesn't exist
    if tracker.run is None:
        tracker.start_run()

    # Run all tests
    test_arena_balanced_data()
    test_arena_dominant_model()
    test_arena_specialized_models()
    test_arena_answer_correctness_filter()

    # End the MLflow run and save all artifacts
    tracker.end_run()

    logger.info("\n" + "=" * 60)
    logger.info("✓ All arena verification tests passed")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run all verification tests
    test_all_arena_verifications()

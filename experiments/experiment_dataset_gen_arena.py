import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import dotenv
import mlflow
import pandas as pd

from src.utils.logger import logger

dotenv.load_dotenv()


class ModelPerQuestionEvaluator:
    """Simplified evaluator: per question compute a composite score for every model,
    pick winner and runner-up (top2), output one row per question mimicking arena CSV schema.
    """

    def __init__(self, max_workers: int = 4):
        self.weights = {
            "query_cost": 0.20,
            "execution_time": 0.10,
            "sql_accuracy": 0.10,
            "answer_correctness": 0.40,
            "faithfulness": 0.10,
            "arena_hard_score": 0.10,
        }
        self.max_workers = max_workers
        self.arena_hard_scores = self._load_arena_hard_scores()

    # ---------------- Arena hard scores -----------------
    def _clean_model_name(self, model_name: str) -> str:
        parts = model_name.split("_")
        if len(parts) > 1 and parts[-1].count("-") >= 1:
            return "_".join(parts[:-1])
        return model_name

    def _load_arena_hard_scores(self) -> Dict[str, float]:
        path = os.path.join("data", "arena_hard_results.csv")
        if not os.path.exists(path):
            logger.warning("arena_hard_results.csv not found; scores default to 0")
            return {}
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error(f"Failed loading arena hard results: {e}")
            return {}
        scores = {}
        for _, row in df.iterrows():
            name = self._clean_model_name(str(row["Model"]).strip()).lower()
            val = float(row["Score"]) if pd.notna(row["Score"]) else 0.0
            scores[name] = val
        return scores

    def get_arena_hard_score(self, model_name: str) -> float:
        if not model_name:
            return 0.0
        cleaned = self._clean_model_name(model_name).lower()
        if cleaned in self.arena_hard_scores:
            return self.arena_hard_scores[cleaned]
        # fuzzy contains both ways
        cands = [k for k in self.arena_hard_scores if k in cleaned or cleaned in k]
        if cands:
            best = sorted(cands, key=len, reverse=True)[0]
            return self.arena_hard_scores[best]
        return 0.0

    # ---------------- Data loading -----------------
    def get_arena_experiments(self) -> List[Dict]:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        experiments = []
        try:
            for exp in mlflow.search_experiments():
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="tags.arena = 'arena'",
                    max_results=1000,
                )
                if not runs.empty:
                    experiments.append(
                        {
                            "experiment_id": exp.experiment_id,
                            "experiment_name": exp.name,
                            "runs": runs,
                        }
                    )
        except Exception as e:
            logger.warning(f"Fallback experiment scan due to: {e}")
            for exp in mlflow.search_experiments():
                if "dataset_gen" in exp.name:
                    runs = mlflow.search_runs(
                        experiment_ids=[exp.experiment_id], max_results=1000
                    )
                    if not runs.empty:
                        experiments.append(
                            {
                                "experiment_id": exp.experiment_id,
                                "experiment_name": exp.name,
                                "runs": runs,
                            }
                        )
        return experiments

    def load_eval_results(self, run_id: str) -> Optional[pd.DataFrame]:
        try:
            client = mlflow.tracking.MlflowClient()
            for art in client.list_artifacts(run_id):
                if art.path == "eval_results_table.json":
                    p = client.download_artifacts(run_id, art.path)
                    with open(p, "r") as f:
                        data = json.load(f)
                    df = pd.DataFrame(data["data"], columns=data["columns"])
                    # Cast numeric
                    for col in [
                        "query_cost",
                        "execution_time",
                        "sql_accuracy/v1/score",
                        "answer_correctness/v1/score",
                        "faithfulness/v1/score",
                    ]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
            return None
        except Exception as e:
            logger.error(f"Error loading eval for run {run_id}: {e}")
            return None

    def load_all_models(self, experiments: List[Dict]) -> Dict[str, pd.DataFrame]:
        tasks: List[Tuple[str, str]] = []  # (model_name, run_id)
        for exp in experiments:
            for _, run in exp["runs"].iterrows():
                run_id = run["run_id"]
                model_name = run.get("tags.mlflow.runName", run_id)
                if "dataset_gen_" in exp["experiment_name"]:
                    model_name = exp["experiment_name"].replace("dataset_gen_", "")
                model_name = self._clean_model_name(model_name)
                tasks.append((model_name, run_id))

        model_data: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_map = {
                ex.submit(self.load_eval_results, run_id): (model_name, run_id)
                for model_name, run_id in tasks
            }
            for fut in as_completed(fut_map):
                model_name, run_id = fut_map[fut]
                try:
                    df = fut.result()
                    if df is not None and not df.empty:
                        df["__arena_model_name"] = model_name
                        model_data[model_name] = df
                except Exception as e:
                    logger.error(f"Failed model {model_name} run {run_id}: {e}")
        logger.info(f"Loaded {len(model_data)} models")
        return model_data

    # ---------------- Scoring -----------------
    def _normalize_series(self, s: pd.Series, higher_is_better: bool) -> pd.Series:
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

    def build_per_question_results(
        self, model_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not model_data:
            return pd.DataFrame(), pd.DataFrame()
        # Intersect questions answered by all models
        question_sets = [
            set(df["question"].tolist())
            for df in model_data.values()
            if "question" in df.columns
        ]
        common_questions = set.intersection(*question_sets) if question_sets else set()
        logger.info(f"Common questions across all models: {len(common_questions)}")
        if not common_questions:
            return pd.DataFrame(), pd.DataFrame()

        top2_results_rows = []
        pairwise_results_rows = []
        # Pre-compute arena hard score normalization across models
        arena_scores = {m: self.get_arena_hard_score(m) for m in model_data}
        arena_min = min(arena_scores.values()) if arena_scores else 0.0
        arena_max = max(arena_scores.values()) if arena_scores else 0.0

        def norm_arena(val: float) -> float:
            if arena_max - arena_min == 0:
                return 0.5
            return (val - arena_min) / (arena_max - arena_min)

        # For each question, gather metric values per model then normalize per metric
        for q in common_questions:
            rows = {}
            for model_name, df in model_data.items():
                try:
                    row = df[df["question"] == q].iloc[0]
                except Exception:
                    continue
                rows[model_name] = row
            if len(rows) < 2:
                continue  # need at least 2 models for a comparison style output

            # Exclude models whose answer correctness score is exactly 1 (lowest 1/5)
            # We do this before normalization so they don't influence min/max scaling.
            ANSWER_COL = "answer_correctness/v1/score"
            filtered_rows = {}
            for m, r in rows.items():
                try:
                    val = pd.to_numeric(r.get(ANSWER_COL), errors="coerce")
                except Exception:
                    val = None
                # Keep model only if score is NaN (unknown) or > 1.0
                if pd.isna(val) or val > 1.0:
                    filtered_rows[m] = r
            rows = filtered_rows
            if len(rows) < 2:
                continue  # after filtering we still require 2 models

            # Build per-metric series
            def collect(col_name: str) -> pd.Series:
                data = {m: rows[m].get(col_name) for m in rows}
                return pd.Series(data, dtype=float)

            # Column mapping from original artifact
            sql_col = "sql_accuracy/v1/score"
            ans_col = "answer_correctness/v1/score"
            faith_col = "faithfulness/v1/score"

            ser_cost = collect("query_cost")
            ser_time = collect("execution_time")
            ser_sql = collect(sql_col)
            ser_ans = collect(ans_col)
            ser_faith = collect(faith_col)
            ser_arena = pd.Series({m: arena_scores[m] for m in rows})

            norm = {
                "query_cost": self._normalize_series(ser_cost, higher_is_better=False),
                "execution_time": self._normalize_series(
                    ser_time, higher_is_better=False
                ),
                "sql_accuracy": self._normalize_series(ser_sql, higher_is_better=True),
                "answer_correctness": self._normalize_series(
                    ser_ans, higher_is_better=True
                ),
                "faithfulness": self._normalize_series(
                    ser_faith, higher_is_better=True
                ),
                "arena_hard_score": ser_arena.map(norm_arena),
            }

            composite = pd.Series({m: 0.0 for m in rows})
            for key, weight in self.weights.items():
                composite += norm[key] * weight

            # Generate top2 results
            ranking = composite.sort_values(ascending=False)
            winner = ranking.index[0]
            second = ranking.index[1] if len(ranking) > 1 else winner
            top2_results_rows.append(
                {
                    "model_a": winner,
                    "model_b": second,
                    "winner": winner,
                    "query": q,
                }
            )

            # Generate pairwise results
            model_names = list(composite.index)
            for model_a, model_b in itertools.combinations(model_names, 2):
                score_a = composite[model_a]
                score_b = composite[model_b]
                winner_pairwise = model_a if score_a >= score_b else model_b
                pairwise_results_rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "winner": winner_pairwise,
                        "query": q,
                    }
                )

        return pd.DataFrame(top2_results_rows), pd.DataFrame(pairwise_results_rows)

    def run(self) -> None:
        logger.info("Starting per-question evaluation...")
        experiments = self.get_arena_experiments()
        if not experiments:
            logger.error("No experiments found with arena tag")
            return

        model_data = self.load_all_models(experiments)
        top2_df, pairwise_df = self.build_per_question_results(model_data)

        # Save top2 results
        if not top2_df.empty:
            out_path_top2 = "data/model_arena_results.csv"
            os.makedirs(os.path.dirname(out_path_top2), exist_ok=True)
            top2_df.to_csv(out_path_top2, index=False)
            logger.info(
                f"Saved per-question top2 results to {out_path_top2} (rows={len(top2_df)})"
            )
        else:
            logger.warning("No top2 results generated.")

        # Save pairwise results
        if not pairwise_df.empty:
            out_path_pairwise = "data/model_arena_results_matrix_router.csv"
            os.makedirs(os.path.dirname(out_path_pairwise), exist_ok=True)
            pairwise_df.to_csv(out_path_pairwise, index=False)
            logger.info(
                f"Saved pairwise model comparison results to {out_path_pairwise} (rows={len(pairwise_df)})"
            )
        else:
            logger.warning("No pairwise results generated.")


def main(max_workers: int = 4):
    evaluator = ModelPerQuestionEvaluator(max_workers=max_workers)
    evaluator.run()


if __name__ == "__main__":
    main()

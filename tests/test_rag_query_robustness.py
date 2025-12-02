"""
Verification test for RAG retrieval robustness with query variations.

Tests how different phrasings of the same question affect retrieval results
using a single embedding model (Gemini). Verifies if semantically similar
queries return consistent results.
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import dotenv
import mlflow
import pandas as pd
import pytest
from langchain.embeddings.base import init_embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MockSQLExampleGenerator:
    """Generate mock SQL query examples for RAG testing"""

    @staticmethod
    def generate_examples(num_examples: int = 20) -> List[Dict]:
        """Generate mock SQL query examples with various characteristics"""
        examples = [
            # Simple SELECT queries
            {
                "question": "Get all customers",
                "sql": "SELECT * FROM customers",
                "complexity": "simple",
                "category": "select",
                "description": "Basic SELECT query to retrieve all customer records",
            },
            {
                "question": "Find users by email",
                "sql": "SELECT * FROM users WHERE email = 'user@example.com'",
                "complexity": "simple",
                "category": "select_filter",
                "description": "Simple filtering query with WHERE clause",
            },
            {
                "question": "Count total orders",
                "sql": "SELECT COUNT(*) FROM orders",
                "complexity": "simple",
                "category": "aggregate",
                "description": "Basic COUNT aggregation query",
            },
            # JOIN queries
            {
                "question": "Get customer orders with details",
                "sql": "SELECT c.name, o.order_date, o.total FROM customers c JOIN orders o ON c.id = o.customer_id",
                "complexity": "medium",
                "category": "join",
                "description": "INNER JOIN between customers and orders tables",
            },
            {
                "question": "List products with category names",
                "sql": "SELECT p.name, c.category_name FROM products p LEFT JOIN categories c ON p.category_id = c.id",
                "complexity": "medium",
                "category": "join",
                "description": "LEFT JOIN to include products without categories",
            },
            # GROUP BY queries
            {
                "question": "Total sales by customer",
                "sql": "SELECT customer_id, SUM(total) as total_sales FROM orders GROUP BY customer_id",
                "complexity": "medium",
                "category": "group_by",
                "description": "Aggregation with GROUP BY clause",
            },
            {
                "question": "Average order value per month",
                "sql": "SELECT DATE_TRUNC('month', order_date) as month, AVG(total) FROM orders GROUP BY month ORDER BY month",
                "complexity": "medium",
                "category": "group_by",
                "description": "Time-based aggregation with date functions",
            },
            # Complex queries
            {
                "question": "Top 5 customers by spending",
                "sql": "SELECT c.name, SUM(o.total) as spending FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY spending DESC LIMIT 5",
                "complexity": "complex",
                "category": "top_n",
                "description": "Multi-table join with aggregation and ranking",
            },
            {
                "question": "Products never ordered",
                "sql": "SELECT p.* FROM products p WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = p.id)",
                "complexity": "complex",
                "category": "subquery",
                "description": "Subquery with NOT EXISTS clause",
            },
            {
                "question": "Customer lifetime value",
                "sql": "SELECT c.id, c.name, COALESCE(SUM(o.total), 0) as ltv FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name",
                "complexity": "complex",
                "category": "aggregate_join",
                "description": "Customer analysis with NULL handling",
            },
            # Window functions
            {
                "question": "Running total of sales",
                "sql": "SELECT order_date, total, SUM(total) OVER (ORDER BY order_date) as running_total FROM orders",
                "complexity": "complex",
                "category": "window",
                "description": "Window function for cumulative sum",
            },
            {
                "question": "Rank customers by order frequency",
                "sql": "SELECT customer_id, COUNT(*) as order_count, RANK() OVER (ORDER BY COUNT(*) DESC) as rank FROM orders GROUP BY customer_id",
                "complexity": "complex",
                "category": "window",
                "description": "Ranking with window function and aggregation",
            },
            # Date/Time queries
            {
                "question": "Orders from last 30 days",
                "sql": "SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'",
                "complexity": "simple",
                "category": "date_filter",
                "description": "Date range filtering with interval",
            },
            {
                "question": "Monthly revenue trends",
                "sql": "SELECT DATE_TRUNC('month', order_date) as month, SUM(total) as revenue FROM orders GROUP BY month ORDER BY month",
                "complexity": "medium",
                "category": "time_series",
                "description": "Time series aggregation for trend analysis",
            },
            # String operations
            {
                "question": "Search customers by name pattern",
                "sql": "SELECT * FROM customers WHERE name ILIKE '%smith%'",
                "complexity": "simple",
                "category": "text_search",
                "description": "Case-insensitive pattern matching",
            },
            {
                "question": "Concatenate customer full names",
                "sql": "SELECT id, first_name || ' ' || last_name as full_name FROM customers",
                "complexity": "simple",
                "category": "string_ops",
                "description": "String concatenation operation",
            },
            # CASE statements
            {
                "question": "Categorize orders by size",
                "sql": "SELECT id, total, CASE WHEN total > 1000 THEN 'Large' WHEN total > 100 THEN 'Medium' ELSE 'Small' END as size FROM orders",
                "complexity": "medium",
                "category": "conditional",
                "description": "Conditional logic with CASE statement",
            },
            {
                "question": "Customer status based on orders",
                "sql": "SELECT c.id, c.name, CASE WHEN COUNT(o.id) > 10 THEN 'VIP' WHEN COUNT(o.id) > 5 THEN 'Regular' ELSE 'New' END as status FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name",
                "complexity": "complex",
                "category": "conditional_aggregate",
                "description": "Customer segmentation with conditional aggregation",
            },
            # DISTINCT queries
            {
                "question": "List unique product categories",
                "sql": "SELECT DISTINCT category_id FROM products WHERE category_id IS NOT NULL",
                "complexity": "simple",
                "category": "distinct",
                "description": "Get unique values with NULL filtering",
            },
            {
                "question": "Count distinct customers per month",
                "sql": "SELECT DATE_TRUNC('month', order_date) as month, COUNT(DISTINCT customer_id) as unique_customers FROM orders GROUP BY month",
                "complexity": "medium",
                "category": "distinct_aggregate",
                "description": "Count unique values within time periods",
            },
        ]

        return examples[:num_examples]


class QueryVariationGenerator:
    """Generate different phrasings of the same question"""

    @staticmethod
    def get_query_variations() -> Dict[str, List[str]]:
        """Return dictionary of query intents with multiple phrasings"""
        return {
            "get_all_customers": [
                "Get all customers",
                "Show me every customer in the database",
                "List all customer records",
                "Retrieve complete customer information",
                "Give me all the customers we have",
                "Display the entire customer list",
            ],
            "calculate_total_sales": [
                "Calculate total sales by customer",
                "Sum up all sales for each customer",
                "What's the total revenue per customer?",
                "Show me how much each customer has spent",
                "Aggregate sales amount grouped by customer",
                "Customer spending totals",
            ],
            "top_performing_products": [
                "Find top performing products",
                "Which products sell the best?",
                "Show me the highest selling items",
                "List the most popular products",
                "What are our best-selling products?",
                "Products ranked by sales performance",
            ],
            "monthly_revenue": [
                "Show monthly revenue trends",
                "What's our revenue by month?",
                "Calculate monthly income",
                "Display revenue for each month",
                "Monthly sales totals over time",
                "Revenue broken down by month",
            ],
            "inactive_customers": [
                "Find customers who haven't ordered",
                "List customers with no purchases",
                "Which customers have never bought anything?",
                "Show me customers without any orders",
                "Customers who haven't made a purchase",
                "Identify non-purchasing customers",
            ],
        }


class RAGQueryRobustnessVerifier:
    """Test RAG retrieval consistency across query variations"""

    def __init__(
        self,
        embedding_model: str = "gemini-embedding-001",
        embedding_provider: str = "google_vertexai",
    ):
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding = None
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> InMemoryVectorStore:
        """Create an in-memory vector store with Gemini embeddings"""
        try:
            self.embedding = init_embeddings(
                model=self.embedding_model, provider=self.embedding_provider
            )
            self.vector_store = InMemoryVectorStore.from_documents(
                documents=documents, embedding=self.embedding
            )
            logger.info(f"Created vector store with {self.embedding_model}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def retrieve_similar(self, query: str, k: int = 5) -> Tuple[List[Document], float]:
        """Retrieve similar documents and measure time"""
        start_time = time.time()
        try:
            results = self.vector_store.similarity_search(query, k=k)
            retrieval_time = time.time() - start_time
            return results, retrieval_time
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [], time.time() - start_time

    def compare_retrieval_consistency(
        self, query_variations: Dict[str, List[str]], top_k: int = 5
    ) -> pd.DataFrame:
        """Compare retrieval results across query variations"""
        results = []

        for intent, variations in query_variations.items():
            logger.info(f"\nTesting intent: {intent}")
            logger.info(f"  Testing {len(variations)} variations...")

            intent_results = []

            for idx, query in enumerate(variations, 1):
                # Retrieve similar documents
                retrieved_docs, retrieval_time = self.retrieve_similar(query, k=top_k)

                # Extract retrieved questions in order
                retrieved_questions = [
                    doc.metadata.get("question", "N/A") for doc in retrieved_docs
                ]

                # Extract categories and complexities
                categories = [
                    doc.metadata.get("category", "unknown") for doc in retrieved_docs
                ]
                complexities = [
                    doc.metadata.get("complexity", "unknown") for doc in retrieved_docs
                ]

                result = {
                    "intent": intent,
                    "variation_number": idx,
                    "query": query,
                    "retrieval_time_ms": round(retrieval_time * 1000, 2),
                    "num_results": len(retrieved_docs),
                    "top_5_questions": retrieved_questions,
                    "top_1_question": (
                        retrieved_questions[0] if retrieved_questions else None
                    ),
                    "top_3_questions": (
                        retrieved_questions[:3]
                        if len(retrieved_questions) >= 3
                        else retrieved_questions
                    ),
                    "categories": categories,
                    "complexities": complexities,
                }

                intent_results.append(result)
                results.append(result)

                logger.info(
                    f"    Variation {idx}: {query[:50]}... -> Top result: {result['top_1_question']}"
                )

            # Analyze consistency for this intent
            top_1_results = [r["top_1_question"] for r in intent_results]
            top_1_unique = len(set(top_1_results))

            all_top_5 = [tuple(r["top_5_questions"]) for r in intent_results]
            exact_matches = len([t for t in all_top_5 if t == all_top_5[0]])

            logger.info(f"  Consistency analysis:")
            logger.info(f"    Top-1 unique results: {top_1_unique}/{len(variations)}")
            logger.info(f"    Exact top-5 matches: {exact_matches}/{len(variations)}")

        return pd.DataFrame(results)

    def calculate_consistency_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate consistency metrics across query variations"""
        metrics = {}

        for intent in results_df["intent"].unique():
            intent_data = results_df[results_df["intent"] == intent]

            # Top-1 consistency
            top_1_results = intent_data["top_1_question"].tolist()
            top_1_unique = len(set(top_1_results))
            top_1_consistency = (
                1.0 - (top_1_unique - 1) / len(top_1_results)
                if len(top_1_results) > 1
                else 1.0
            )

            # Top-5 exact match rate
            all_top_5 = [tuple(row) for row in intent_data["top_5_questions"]]
            exact_matches = sum(1 for t in all_top_5 if t == all_top_5[0])
            exact_match_rate = exact_matches / len(all_top_5) if all_top_5 else 0

            # Average retrieval time
            avg_time = intent_data["retrieval_time_ms"].mean()

            metrics[intent] = {
                "num_variations": len(intent_data),
                "top_1_consistency_score": round(top_1_consistency, 3),
                "top_1_unique_results": top_1_unique,
                "exact_top_5_match_rate": round(exact_match_rate, 3),
                "exact_matches": exact_matches,
                "avg_retrieval_time_ms": round(avg_time, 2),
            }

        return metrics


class MLflowQueryRobustnessTracker:
    """Track query robustness results in MLflow"""

    def __init__(self):
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment("rag_query_robustness_verification")
        self.run = None
        self.original_stdout = None

    def start_run(self):
        """Start MLflow run"""
        self.original_stdout = sys.stdout

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            self.run = mlflow.start_run(run_name="query_variation_robustness")
        finally:
            if sys.platform == "win32" and sys.stdout != self.original_stdout:
                sys.stdout.close()
            sys.stdout = self.original_stdout

    def log_results(
        self, results_df: pd.DataFrame, consistency_metrics: Dict, embedding_model: str
    ):
        """Log results to MLflow"""
        if not self.run:
            return

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            # Log parameters
            mlflow.log_param("embedding_model", embedding_model)
            mlflow.log_param("num_intents", results_df["intent"].nunique())
            mlflow.log_param("total_queries", len(results_df))
            mlflow.log_param("top_k", 5)

            # Log consistency metrics
            for intent, metrics in consistency_metrics.items():
                mlflow.log_metric(
                    f"{intent}_top1_consistency", metrics["top_1_consistency_score"]
                )
                mlflow.log_metric(
                    f"{intent}_exact_match_rate", metrics["exact_top_5_match_rate"]
                )
                mlflow.log_metric(
                    f"{intent}_avg_time_ms", metrics["avg_retrieval_time_ms"]
                )

            # Log overall metrics
            avg_top1_consistency = sum(
                m["top_1_consistency_score"] for m in consistency_metrics.values()
            ) / len(consistency_metrics)
            avg_exact_match = sum(
                m["exact_top_5_match_rate"] for m in consistency_metrics.values()
            ) / len(consistency_metrics)

            mlflow.log_metric(
                "overall_top1_consistency", round(avg_top1_consistency, 3)
            )
            mlflow.log_metric("overall_exact_match_rate", round(avg_exact_match, 3))

            # Save artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save full results CSV
                results_path = Path(tmpdir) / "query_robustness_results.csv"
                results_df.to_csv(results_path, index=False)
                mlflow.log_artifact(results_path)

                # Save consistency metrics
                metrics_path = Path(tmpdir) / "consistency_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(consistency_metrics, f, indent=2)
                mlflow.log_artifact(metrics_path)

                # Create overview JSON
                overview = {
                    "experiment_name": "rag_query_robustness_verification",
                    "run_date": pd.Timestamp.now().isoformat(),
                    "embedding_model": embedding_model,
                    "num_intents_tested": results_df["intent"].nunique(),
                    "total_query_variations": len(results_df),
                    "overall_metrics": {
                        "avg_top1_consistency": round(avg_top1_consistency, 3),
                        "avg_exact_match_rate": round(avg_exact_match, 3),
                    },
                    "per_intent_metrics": consistency_metrics,
                }

                overview_path = Path(tmpdir) / "query_robustness_overview.json"
                with open(overview_path, "w") as f:
                    json.dump(overview, f, indent=2)
                mlflow.log_artifact(overview_path)

            run_id = self.run.info.run_id
            logger.info(f"\n{'='*60}")
            logger.info(f"Query robustness results saved to MLflow run: {run_id}")
            logger.info(f"View at: {mlflow.get_tracking_uri()}/#/experiments")
            logger.info(f"{'='*60}")
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
def test_rag_query_robustness():
    """Test RAG retrieval consistency with query variations"""
    logger.info("\n" + "=" * 60)
    logger.info("RAG QUERY ROBUSTNESS VERIFICATION")
    logger.info("Testing: Gemini Embedding Model")
    logger.info("=" * 60)

    # Generate mock SQL examples
    generator = MockSQLExampleGenerator()
    examples = generator.generate_examples(num_examples=20)
    logger.info(f"Generated {len(examples)} SQL examples")

    # Convert to documents
    documents = [
        Document(
            page_content=f"Question: {ex['question']}\nSQL: {ex['sql']}\nDescription: {ex['description']}",
            metadata={
                "question": ex["question"],
                "sql": ex["sql"],
                "complexity": ex["complexity"],
                "category": ex["category"],
            },
        )
        for ex in examples
    ]

    # Get query variations
    variation_generator = QueryVariationGenerator()
    query_variations = variation_generator.get_query_variations()

    total_variations = sum(len(v) for v in query_variations.values())
    logger.info(
        f"Testing {len(query_variations)} intents with {total_variations} total query variations"
    )

    # Create verifier and run tests
    verifier = RAGQueryRobustnessVerifier(
        embedding_model="gemini-embedding-001", embedding_provider="google_vertexai"
    )
    verifier.create_vector_store(documents)

    results_df = verifier.compare_retrieval_consistency(query_variations, top_k=5)
    consistency_metrics = verifier.calculate_consistency_metrics(results_df)

    # Track in MLflow
    tracker = MLflowQueryRobustnessTracker()
    tracker.start_run()
    tracker.log_results(results_df, consistency_metrics, "gemini-embedding-001")
    tracker.end_run()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CONSISTENCY METRICS BY INTENT")
    logger.info("=" * 60)

    for intent, metrics in consistency_metrics.items():
        logger.info(f"\n{intent}:")
        logger.info(f"  Variations tested: {metrics['num_variations']}")
        logger.info(
            f"  Top-1 consistency: {metrics['top_1_consistency_score']:.1%} ({metrics['top_1_unique_results']} unique results)"
        )
        logger.info(
            f"  Exact top-5 match: {metrics['exact_top_5_match_rate']:.1%} ({metrics['exact_matches']}/{metrics['num_variations']})"
        )
        logger.info(f"  Avg retrieval time: {metrics['avg_retrieval_time_ms']:.2f}ms")

    # Overall statistics
    avg_consistency = sum(
        m["top_1_consistency_score"] for m in consistency_metrics.values()
    ) / len(consistency_metrics)
    avg_match_rate = sum(
        m["exact_top_5_match_rate"] for m in consistency_metrics.values()
    ) / len(consistency_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("OVERALL STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Average top-1 consistency: {avg_consistency:.1%}")
    logger.info(f"Average exact top-5 match rate: {avg_match_rate:.1%}")
    logger.info("=" * 60)
    logger.info("✓ RAG query robustness test passed")
    logger.info("=" * 60)

    # Assertions
    assert len(results_df) > 0, "Should have retrieval results"
    assert len(consistency_metrics) > 0, "Should have consistency metrics"
    assert avg_consistency > 0, "Should have positive consistency score"


if __name__ == "__main__":
    test_rag_query_robustness()

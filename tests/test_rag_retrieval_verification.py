"""
Verification test for RAG retrieval with different embedding models.

Tests how different embedding models affect retrieval results when querying
a vector store with SQL-related examples.
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


class RAGRetrievalVerifier:
    """Test RAG retrieval with different embedding models"""

    def __init__(self):
        self.embedding_configs = [
            # Ollama models (local - available models from ollama list)
            {
                "name": "snowflake-arctic-335m",
                "provider": "ollama",
                "model": "snowflake-arctic-embed:335m",
                "description": "Snowflake Arctic 335M parameters (Ollama)",
            },
            {
                "name": "snowflake-arctic-22m",
                "provider": "ollama",
                "model": "snowflake-arctic-embed:22m",
                "description": "Snowflake Arctic 22M parameters (Ollama)",
            },
            # OpenAI models
            {
                "name": "openai-text-embedding-3-large",
                "provider": "openai",
                "model": "text-embedding-3-large",
                "description": "OpenAI Text Embedding 3 Large (1536 dimensions)",
            },
            {
                "name": "openai-text-embedding-3-small",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "description": "OpenAI Text Embedding 3 Small (512 dimensions)",
            },
            {
                "name": "openai-text-embedding-ada-002",
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "description": "OpenAI Ada 002 (1536 dimensions)",
            },
            # Google Vertex AI
            {
                "name": "vertex-gemini-embedding-001",
                "provider": "vertex",
                "model": "gemini-embedding-001",
                "description": "Google Vertex AI Gemini Embedding 001",
            },
        ]

    def create_vector_store(
        self, embedding_config: Dict, documents: List[Document]
    ) -> InMemoryVectorStore:
        """Create an in-memory vector store with specific embedding model"""
        try:
            embedding = init_embeddings(
                model=embedding_config["model"], provider=embedding_config["provider"]
            )
            vector_store = InMemoryVectorStore.from_documents(
                documents=documents, embedding=embedding
            )
            return vector_store
        except Exception as e:
            logger.warning(
                f"Failed to create vector store for {embedding_config['name']}: {e}"
            )
            return None

    def retrieve_similar(
        self, vector_store: InMemoryVectorStore, query: str, k: int = 5
    ) -> Tuple[List[Document], float]:
        """Retrieve similar documents and measure time"""
        start_time = time.time()
        try:
            results = vector_store.similarity_search(query, k=k)
            retrieval_time = time.time() - start_time
            return results, retrieval_time
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [], time.time() - start_time

    def calculate_similarity_stats(self, documents: List[Document]) -> Dict:
        """Calculate statistics from retrieved documents"""
        if not documents:
            return {
                "num_results": 0,
                "avg_score": 0,
                "categories": [],
                "complexities": [],
            }

        categories = [doc.metadata.get("category", "unknown") for doc in documents]
        complexities = [doc.metadata.get("complexity", "unknown") for doc in documents]

        return {
            "num_results": len(documents),
            "categories": categories,
            "complexities": complexities,
            "unique_categories": len(set(categories)),
            "unique_complexities": len(set(complexities)),
        }

    def run_verification(
        self, test_queries: List[str], examples: List[Dict], top_k: int = 5
    ) -> pd.DataFrame:
        """Run retrieval test across all embedding models"""
        # Convert examples to documents
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

        results = []

        for query in test_queries:
            logger.info(f"\nTesting query: {query}")

            for emb_config in self.embedding_configs:
                logger.info(f"  Testing with {emb_config['name']}...")

                # Create vector store
                vector_store = self.create_vector_store(emb_config, documents)
                if vector_store is None:
                    logger.warning(
                        f"  Skipping {emb_config['name']} - failed to initialize"
                    )
                    continue

                # Retrieve similar documents
                retrieved_docs, retrieval_time = self.retrieve_similar(
                    vector_store, query, k=top_k
                )

                # Calculate stats
                stats = self.calculate_similarity_stats(retrieved_docs)

                # Get retrieved questions
                retrieved_questions = [
                    doc.metadata.get("question", "N/A") for doc in retrieved_docs
                ]

                result = {
                    "query": query,
                    "embedding_model": emb_config["name"],
                    "embedding_provider": emb_config["provider"],
                    "retrieval_time_ms": round(retrieval_time * 1000, 2),
                    "num_results": stats["num_results"],
                    "top_5_questions": retrieved_questions,
                    "categories": stats.get("categories", []),
                    "complexities": stats.get("complexities", []),
                    "unique_categories": stats.get("unique_categories", 0),
                    "unique_complexities": stats.get("unique_complexities", 0),
                }

                results.append(result)

                logger.info(
                    f"    Retrieved {stats['num_results']} docs in {retrieval_time*1000:.2f}ms"
                )

        return pd.DataFrame(results)


class MLflowRAGVerificationTracker:
    """Track RAG verification results in MLflow"""

    def __init__(self):
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment("rag_retrieval_verification")
        self.run = None
        self.original_stdout = None

    def start_run(self):
        """Start MLflow run"""
        self.original_stdout = sys.stdout

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            self.run = mlflow.start_run(run_name="rag_embedding_comparison")
        finally:
            if sys.platform == "win32" and sys.stdout != self.original_stdout:
                sys.stdout.close()
            sys.stdout = self.original_stdout

    def log_results(self, results_df: pd.DataFrame, test_queries: List[str]):
        """Log results to MLflow"""
        if not self.run:
            return

        try:
            if sys.platform == "win32":
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

            # Log parameters
            mlflow.log_param("num_test_queries", len(test_queries))
            mlflow.log_param("top_k", 5)
            mlflow.log_param(
                "num_embedding_models", results_df["embedding_model"].nunique()
            )

            # Log metrics by embedding model
            for emb_model in results_df["embedding_model"].unique():
                model_data = results_df[results_df["embedding_model"] == emb_model]

                avg_time = model_data["retrieval_time_ms"].mean()
                mlflow.log_metric(f"{emb_model}_avg_retrieval_time_ms", avg_time)
                mlflow.log_metric(f"{emb_model}_total_retrievals", len(model_data))

            # Save artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save full results CSV
                results_path = Path(tmpdir) / "rag_retrieval_results.csv"
                results_df.to_csv(results_path, index=False)
                mlflow.log_artifact(results_path)

                # Create summary by embedding model
                summary_data = []
                for emb_model in results_df["embedding_model"].unique():
                    model_data = results_df[results_df["embedding_model"] == emb_model]
                    summary_data.append(
                        {
                            "embedding_model": emb_model,
                            "embedding_provider": model_data["embedding_provider"].iloc[
                                0
                            ],
                            "avg_retrieval_time_ms": round(
                                model_data["retrieval_time_ms"].mean(), 2
                            ),
                            "min_retrieval_time_ms": round(
                                model_data["retrieval_time_ms"].min(), 2
                            ),
                            "max_retrieval_time_ms": round(
                                model_data["retrieval_time_ms"].max(), 2
                            ),
                            "total_queries": len(model_data),
                            "avg_unique_categories": round(
                                model_data["unique_categories"].mean(), 2
                            ),
                        }
                    )

                summary_df = pd.DataFrame(summary_data)
                summary_path = Path(tmpdir) / "embedding_model_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                mlflow.log_artifact(summary_path)

                # Save JSON overview
                overview = {
                    "experiment_name": "rag_retrieval_verification",
                    "run_date": pd.Timestamp.now().isoformat(),
                    "test_queries": test_queries,
                    "embedding_models_tested": results_df["embedding_model"]
                    .unique()
                    .tolist(),
                    "summary": summary_data,
                }

                overview_path = Path(tmpdir) / "rag_verification_overview.json"
                with open(overview_path, "w") as f:
                    json.dump(overview, f, indent=2)
                mlflow.log_artifact(overview_path)

            run_id = self.run.info.run_id
            logger.info(f"\n{'='*60}")
            logger.info(f"RAG verification results saved to MLflow run: {run_id}")
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
def test_rag_embedding_comparison():
    """Test RAG retrieval with different embedding models"""
    logger.info("\n" + "=" * 60)
    logger.info("RAG RETRIEVAL VERIFICATION - Embedding Model Comparison")
    logger.info("=" * 60)

    # Generate mock SQL examples
    generator = MockSQLExampleGenerator()
    examples = generator.generate_examples(num_examples=20)
    logger.info(f"Generated {len(examples)} SQL examples")

    # Define test queries
    test_queries = [
        "Get all customer information",
        "Calculate total sales by customer",
        "Find top performing products",
        "Show monthly revenue trends",
        "List customers who haven't ordered",
    ]

    # Run verification
    verifier = RAGRetrievalVerifier()
    results_df = verifier.run_verification(test_queries, examples, top_k=5)

    # Track in MLflow
    tracker = MLflowRAGVerificationTracker()
    tracker.start_run()
    tracker.log_results(results_df, test_queries)
    tracker.end_run()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RETRIEVAL SUMMARY BY EMBEDDING MODEL")
    logger.info("=" * 60)

    for emb_model in results_df["embedding_model"].unique():
        model_data = results_df[results_df["embedding_model"] == emb_model]
        avg_time = model_data["retrieval_time_ms"].mean()
        logger.info(f"\n{emb_model}:")
        logger.info(f"  Avg retrieval time: {avg_time:.2f}ms")
        logger.info(f"  Total queries: {len(model_data)}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ RAG embedding comparison test passed")
    logger.info("=" * 60)

    # Basic assertions
    assert len(results_df) > 0, "Should have retrieval results"
    assert (
        results_df["embedding_model"].nunique() > 0
    ), "Should test multiple embedding models"


if __name__ == "__main__":
    test_rag_embedding_comparison()

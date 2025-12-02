import json
import logging
import os
from typing import Callable, List, Optional

import dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from src.utils.logger import configure_logging

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class LLMExamplesDataLoader:
    """Data loader for LLM examples using in-memory vector store with dialect filtering."""

    def __init__(self, embedding_model_name: str = "snowflake-arctic-embed:22m"):
        self.embedding_model = OllamaEmbeddings(
            model=embedding_model_name, mirostat_eta=0.8
        )
        self.vector_store = InMemoryVectorStore(embedding=self.embedding_model)

    def load_examples_from_file(self, file_path: str) -> None:
        """Load examples from JSON file into vector store."""
        with open(file_path, "r", encoding="utf-8") as f:
            examples_data = json.load(f)

        documents = []
        allowed_datasets = {"pagila"}

        for i, example in enumerate(examples_data):
            if (
                example.get("input")
                and example.get("query")
                and example.get("dataset") in allowed_datasets
            ):
                doc = Document(
                    id=str(i),
                    page_content=example["input"],
                    metadata={"query": example["query"], "dialect": example["dialect"]},
                )
                documents.append(doc)

        # Add documents one by one to avoid batch size limit with gemini-embedding-001
        for doc in documents:
            self.vector_store.add_documents([doc])

        logger.info(
            f"Loaded {len(documents)} examples from the following datasets: {', '.join(allowed_datasets)}"
        )

    def save_vector_store(self, path: str) -> None:
        """Save vector store to file for persistence."""
        self.vector_store.dump(path)
        logger.info(f"Vector store saved to {path}")

    def load_vector_store(self, path: str) -> None:
        """Load vector store from file."""
        if os.path.exists(path):
            self.vector_store = InMemoryVectorStore.load(path, self.embedding_model)
            logger.info(f"Vector store loaded from {path}")
        else:
            logger.warning(f"Vector store file not found: {path}")

    def _create_dialect_filter(self, dialect: str) -> Callable[[Document], bool]:
        """Create filter function for SQL dialect."""
        return lambda doc: doc.metadata.get("dialect", "").lower() == dialect.lower()

    def search(
        self, query: str, k: int = 5, dialect: Optional[str] = None
    ) -> List[Document]:
        """Search for similar examples with optional dialect filtering."""
        kwargs = {"k": k}
        if dialect:
            kwargs["filter"] = self._create_dialect_filter(dialect)
        return self.vector_store.similarity_search(query, **kwargs)


def create_llm_examples_retriever(
    examples_file_path: str, save_path: Optional[str] = None
) -> LLMExamplesDataLoader:
    """Create and initialize LLM examples data loader with optional persistence."""
    loader = LLMExamplesDataLoader()

    if save_path and os.path.exists(save_path):
        loader.load_vector_store(save_path)
    else:
        loader.load_examples_from_file(examples_file_path)
        if save_path:
            loader.save_vector_store(save_path)

    return loader


if __name__ == "__main__":
    examples_file = "data/llm_examples/examples.json"
    vector_store_path = "data/llm_examples/examples_vector_store.pkl"

    # Create loader with persistence
    loader = create_llm_examples_retriever(examples_file, vector_store_path)

    # Test search without filter
    results = loader.search("How many flights are there?", k=2)
    print("All dialects:")
    for doc in results:
        print(f"- {doc.page_content} ({doc.metadata})")

    # Test search with Postgres filter
    postgres_results = loader.search(
        "How many flights are there?", k=2, dialect="Dialect Postgres SQL"
    )
    print("\nPostgres only:")
    for doc in postgres_results:
        print(f"- {doc.page_content} ({doc.metadata})")

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import dotenv
import pandas as pd
from langchain.embeddings.base import init_embeddings
from langchain.schema import Document
from langchain_core.vectorstores import InMemoryVectorStore
from tqdm import tqdm

from src.utils.logger import logger

dotenv.load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)


class ChatbotArenaDataLoader:
    """Data loader for chatbot arena data with in-memory vector store."""

    def __init__(
        self,
        embedding_model: str,
        embedding_provider: Optional[str],
        **embedding_kwargs,
    ):
        """
        Initialize the data loader with flexible embedding configuration.

        Args:
            embedding_model: Name of the embedding model
            embedding_provider: Provider for the embedding model (e.g., "vertex", "openai", "ollama")
            **embedding_kwargs: Additional arguments for embedding initialization
        """
        self.embedding_model = init_embeddings(
            model=embedding_model,
            provider=embedding_provider,
            **embedding_kwargs,
        )
        self.vector_store: Optional[InMemoryVectorStore] = None

    def load_csv_data(self, csv_path: str) -> List[Document]:
        """Load CSV data and convert to Document objects."""
        df = pd.read_csv(csv_path)

        required_columns = ["model_a", "model_b", "winner", "query"]
        if missing := [col for col in required_columns if col not in df.columns]:
            raise ValueError(f"Missing columns: {missing}")

        df = df.dropna(subset=required_columns)
        df = df[df["query"].str.strip() != ""]

        logger.info(f"Loaded {len(df)} valid records from {csv_path}")

        # Convert to documents
        documents = []
        for idx, row in df.iterrows():
            metadata = {
                "model_a": str(row["model_a"]).strip(),
                "model_b": str(row["model_b"]).strip(),
                "winner": str(row["winner"]).strip(),
                "source": csv_path,
                "row_id": idx,
            }
            documents.append(
                Document(page_content=str(row["query"]).strip(), metadata=metadata)
            )

        return documents

    def setup_vector_store(
        self, vector_store_path: Optional[str] = None, recreate: bool = False
    ):
        """Setup vector store, optionally loading from file."""
        if recreate or not vector_store_path or not os.path.exists(vector_store_path):
            logger.info("Creating new vector store")
            self.vector_store = InMemoryVectorStore(embedding=self.embedding_model)
        else:
            logger.info(f"Loading vector store from {vector_store_path}")
            self.vector_store = InMemoryVectorStore.load(
                vector_store_path, self.embedding_model
            )

        return self.vector_store

    def add_documents(self, documents: List[Document], max_workers: int = 10):
        """Add documents to vector store using parallel processing."""
        total = len(documents)
        logger.info(f"Processing {total} documents with {max_workers} parallel workers")

        def process_single_document(doc_with_index):
            """Process a single document and return it with its embedding."""
            index, doc = doc_with_index
            try:
                embedding = self.embedding_model.embed_query(doc.page_content)
                return index, doc, embedding, None
            except Exception as e:
                return index, doc, None, str(e)

        successful_docs = []
        failed_docs = []

        # Create progress bar
        pbar = tqdm(total=total, desc="Processing documents", unit="docs")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(process_single_document, (i, doc)): i
                for i, doc in enumerate(documents)
            }
            for future in as_completed(future_to_doc):
                try:
                    index, doc, embedding, error = future.result()
                    if error:
                        failed_docs.append((index, error))
                        logger.warning(f"Failed to process document {index}: {error}")
                    else:
                        successful_docs.append((doc, embedding))

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(
                        {"success": len(successful_docs), "failed": len(failed_docs)}
                    )

                except Exception as e:
                    logger.error(f"Unexpected error processing document: {e}")
                    pbar.update(1)

        pbar.close()

        if successful_docs:
            logger.info(f"Adding {len(successful_docs)} embeddings to vector store")
            docs_to_add = [doc for doc, _ in successful_docs]
            embeddings = [emb for _, emb in successful_docs]
            self.vector_store.add_documents(docs_to_add, embeddings=embeddings)

        if failed_docs:
            logger.warning(f"{len(failed_docs)} documents failed to process")

        return len(successful_docs), len(failed_docs)

    def save(self, save_path: str):
        """Save vector store to file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.dump(save_path)
        logger.info(f"Vector store saved to {save_path}")


def load_data(
    loader: ChatbotArenaDataLoader,
    csv_file: str = "data/model_arena_results.csv",
    save_path: str = "data/chatbot_arena_inmemory_vectorstore.pkl",
    max_workers: int = 15,
    recreate: bool = True,
):
    """Load chatbot arena data into vector store."""

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    loader.setup_vector_store(save_path, recreate=recreate)

    if recreate:
        documents = loader.load_csv_data(csv_file)
        successful, failed = loader.add_documents(documents, max_workers)
        loader.save(save_path)
        logger.info(
            f"Completed: {successful} successful, {failed} failed out of {len(documents)} total documents"
        )
    else:
        logger.info("Vector store loaded from existing file")


if __name__ == "__main__":
    # Initialize the loader once
    loader = ChatbotArenaDataLoader(
        embedding_model="snowflake-arctic-embed:335m",
        embedding_provider="ollama",
    )

    # Process the standard arena results
    logger.info("Processing standard model arena results...")
    load_data(
        loader=loader,
        csv_file="data/model_arena_results.csv",
        save_path="data/chatbot_arena_inmemory_vectorstore.pkl",
        recreate=True,
    )

    # Process the matrix router arena results
    logger.info("\nProcessing matrix router model arena results...")
    load_data(
        loader=loader,
        csv_file="data/model_arena_results_matrix_router.csv",
        save_path="data/chatbot_arena_inmemory_vectorstore_matrix_router.pkl",
        recreate=True,
    )

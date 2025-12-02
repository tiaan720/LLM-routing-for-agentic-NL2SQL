from typing import Callable, Dict, Optional

import dotenv
from langchain.embeddings.base import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from src.retrievers.InMemoryVectorStore import create_inmemory_vector_store_retriever

dotenv.load_dotenv()

retriever_callables: Dict[str, Callable] = {
    "inmemory_vector_store": create_inmemory_vector_store_retriever,
}


def create_retriever(
    retriever_type: str,
    embedding_model: str,
    embedding_provider: Optional[str] = None,
    **kwargs,
) -> Runnable:
    """
    Create a retriever with the specified embedding model and provider.

    Args:
        retriever_type: Type of retriever to create
        embedding_model: Name of the embedding model (e.g., "text-embedding-3-small")
        embedding_provider: Optional provider name (e.g., "openai", "vertex", etc.)
        **kwargs: Additional arguments specific to the retriever type

    Returns:
        Configured retriever instance
    """
    # Create embedding model using init_embeddings
    if embedding_provider:
        embedding = init_embeddings(
            model=embedding_model,
            provider=embedding_provider,
            **kwargs.pop("embedding_kwargs", {}),
        )
    else:
        # Try to parse provider from model string (e.g., "openai:text-embedding-3-small")
        embedding = init_embeddings(
            model=embedding_model, **kwargs.pop("embedding_kwargs", {})
        )

    # Create retriever with the embedding model
    return retriever_callables[retriever_type](embedding=embedding, **kwargs)


if __name__ == "__main__":
    messages = "Who is Tiaan?"

    # Example using Ollama embeddings with in-memory vector store
    retriever = create_retriever(
        retriever_type="inmemory_vector_store",
        embedding_model="snowflake-arctic-embed:22m",
        embedding_provider="ollama",
        vector_store_path="data/llm_examples/examples_vector_store.pkl",
    )
    response = retriever.invoke(messages)
    print("InMemory response:", response)

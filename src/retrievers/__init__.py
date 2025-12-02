from langchain.embeddings.base import init_embeddings

from src.retrievers.cached_retriever import CachedRetriever
from src.retrievers.InMemoryVectorStore import create_inmemory_vector_store_retriever

# Create embedding for the base retriever
embedding = init_embeddings(
    model="snowflake-arctic-embed:22m", provider="ollama", mirostat_eta=0.8
)

base_retriever = create_inmemory_vector_store_retriever(
    embedding=embedding, vector_store_path="data/llm_examples/examples_vector_store.pkl"
)

example_retriever = CachedRetriever(base_retriever, cache_size=100)

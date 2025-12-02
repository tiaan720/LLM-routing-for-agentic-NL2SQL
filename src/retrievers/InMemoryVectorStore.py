import dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

dotenv.load_dotenv()


def create_inmemory_vector_store_retriever(
    embedding: Embeddings,
    vector_store_path: str = None,
):
    """
    Connect to an in memory vector store.

    Args:
        embedding (Embeddings): Pre-configured embedding model instance.
        vector_store_path (str): Path to a saved vector store .pkl file. If None, creates empty store.
    """
    if vector_store_path:
        vector_store = InMemoryVectorStore.load(vector_store_path, embedding)
    else:
        vector_store = InMemoryVectorStore(embedding)

    retriever = vector_store.as_retriever()

    return retriever


if __name__ == "__main__":
    from langchain.embeddings.base import init_embeddings

    messages = "Who is Tiaan?"

    # Create embedding using init_embeddings with Ollama
    embedding = init_embeddings(
        model="snowflake-arctic-embed:22m", provider="ollama", mirostat_eta=0.8
    )

    vector_store_path = "data/llm_examples/examples_vector_store.pkl"
    retriever = create_inmemory_vector_store_retriever(
        embedding=embedding, vector_store_path=vector_store_path
    )
    response = retriever.invoke(messages)

    print(response)

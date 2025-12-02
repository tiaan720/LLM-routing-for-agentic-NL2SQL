import os

from src.retrievers.base import create_retriever


def create_rag_router(
    query: str, vector_store_path: str = "data/chatbot_arena_inmemory_vectorstore.pkl"
):
    """
    Create a RAG router using Ollama embeddings and in-memory vector store.

    Args:
        query (str): The query to route
        vector_store_path (str): Path to the vector store file
    """
    retriever = create_retriever(
        retriever_type="inmemory_vector_store",
        embedding_model="snowflake-arctic-embed:335m",
        embedding_provider="ollama",
        vector_store_path=vector_store_path,
    )
    response = retriever.invoke(query)

    # Get the winning model name
    winner_label = response[0].metadata.get("winner")
    winning_model = (
        response[0].metadata.get("model_a")
        if winner_label == "model_a"
        else response[0].metadata.get("model_b")
    )
    return winning_model


if __name__ == "__main__":
    query = "Statement of work of cloud infrastructure provider"
    vector_store_path = "data/chatbot_arena_inmemory_vectorstore.pkl"

    result = create_rag_router(query, vector_store_path)
    print(result)

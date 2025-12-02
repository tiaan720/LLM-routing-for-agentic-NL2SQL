from langchain_core.prompts import ChatPromptTemplate

from .postgres_prompts import (
    get_postgres_system_prompt,
    get_postgres_system_prompt_text,
)


def get_system_prompt_for_db_type(
    db_type: str,
    project_id: str = None,
    dataset_id: str = None,
    database: str = None,
    **kwargs,
) -> str:
    """
    Get the appropriate system prompt based on database type.

    Args:
        db_type: Database type ("bigquery" or "postgres")
        project_id: BigQuery project ID (for BigQuery)
        dataset_id: BigQuery dataset ID (for BigQuery)
        database: PostgreSQL database name (for PostgreSQL)
        **kwargs: Additional configuration parameters

    Returns:
        str: System prompt text for the database type

    Raises:
        ValueError: If database type is not supported
    """
    if db_type == "postgres":
        return get_postgres_system_prompt_text(database)
    else:
        # Default generic prompt for unsupported database types
        return """You are working with a SQL database.
        Follow these steps when handling user queries:
        1. First, use the list_schema tool to understand the database structure if needed
        2. Form your SQL query carefully
        3. Use the execute_sql_with_validation tool to run the query
        4. If appropriate, use the generate_plot_from_data tool to visualize the results
        5. Explain the results to the user in a clear and concise way
        
        Always execute your queries using the provided tools. Don't just write SQL without executing it."""

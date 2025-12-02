"""PostgreSQL-specific prompt templates."""

from langchain_core.prompts import ChatPromptTemplate


def get_postgres_system_prompt_text(database: str) -> str:
    """
    Get the system prompt text for PostgreSQL SQL agent.

    Args:
        database: PostgreSQL database name

    Returns:
        str: System prompt text for PostgreSQL
    """

    db_specific_instructions = f"""You are working with PostgreSQL database '{database}'. You are a react agent and will following the react style loop of reasoning using tool to execute and then reflecting on the tools output and you will continue this loop untill can answer the users questions. When writing PostgreSQL queries, refer to tables directly by their names (e.g., table_name) in the public schema."""

    return f"""{db_specific_instructions}
    Follow these steps when handling user queries:
    1. First, use the postgres_list_schema tool to understand the database structure if needed
    2. Form your SQL query carefully using PostgreSQL syntax
    3. Use the postgres_execute_sql tool to run the query
    4. If appropriate, use the generate_plot_from_data tool to visualize the results (given the tool is available)
    5. Explain the results to the user in a clear and concise way
    
    Always execute your queries using the provided tools. Don't just write SQL without executing it.
    
    CRITICAL: You MUST use postgres_execute_sql tool for every SQL query. Never show SQL code without executing it first. You cannot get answers without successfully using the tool to query the database.
    
    PostgreSQL-specific notes:
    - Use table_name directly (no schema prefix needed for public schema)
    - PostgreSQL supports advanced features like window functions, CTEs, etc.
    - Be aware of PostgreSQL-specific data types and functions"""


def get_postgres_system_prompt(
    database: str, retrieved_examples_text: str
) -> ChatPromptTemplate:
    """
    Get the system prompt for PostgreSQL SQL agent.

    Args:
        database: PostgreSQL database name
        retrieved_examples_text: Retrieved examples from RAG

    Returns:
        ChatPromptTemplate: Configured system prompt for PostgreSQL
    """

    # Escape curly braces in retrieved examples to prevent template variable issues
    escaped_examples = retrieved_examples_text.replace("{", "{{").replace("}", "}}")

    base_prompt = get_postgres_system_prompt_text(database)

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""{base_prompt}
                
                Here are some example queries for reference:
                {escaped_examples}
                """,
            ),
            ("human", "{messages}"),
        ]
    )

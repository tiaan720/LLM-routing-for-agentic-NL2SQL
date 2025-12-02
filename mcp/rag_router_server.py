import logging
import sys
from pathlib import Path
from typing import Optional

# CRITICAL: Import fastmcp BEFORE adding project root to path
# This avoids the local 'mcp' directory shadowing the installed 'mcp' package
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# NOW we can safely add project root for our own imports
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.agents.sql_agent import create_sql_agent
from src.agents.sql_agent_rag_router import get_routed_model
from src.cost_calc.shared_store import get_shared_store
from src.cost_calc.store_cost_tracker import process_agent_stream_with_store
from src.models.base import create_model
from src.schema import DatabaseConfig
from src.utils.db_clients import create_database_client

mcp_server = FastMCP("SQL Agent RAG Router")


class RAGRouterQuerySchema(BaseModel):
    """Schema for RAG router SQL query execution."""

    query: str = Field(
        description="The SQL question to answer. The RAG router will automatically select the best model for this query."
    )
    database: Optional[str] = Field(
        default="academic",
        description="PostgreSQL database name to query against",
    )
    host: Optional[str] = Field(
        default="localhost", description="PostgreSQL host address"
    )
    port: Optional[int] = Field(default=5432, description="PostgreSQL port number")
    username: Optional[str] = Field(
        default="postgres", description="PostgreSQL username"
    )
    password: Optional[str] = Field(
        default="postgres", description="PostgreSQL password"
    )


@mcp_server.tool()
async def execute_sql_query_with_rag_router(
    query: str,
    database: str = "academic",
    host: str = "localhost",
    port: int = 5432,
    username: str = "postgres",
    password: str = "postgres",
) -> str:
    """
    Execute a SQL query using RAG router for automatic model selection.

    This tool uses retrieval-augmented generation to select the optimal language model
    for your SQL query based on similar queries in the knowledge base. The selected
    model then generates and executes the SQL query against the PostgreSQL database.

    Args:
        query: Natural language question to convert to SQL and execute
        database: PostgreSQL database name (default: academic)
        host: PostgreSQL server host (default: localhost)
        port: PostgreSQL server port (default: 5432)
        username: Database username (default: postgres)
        password: Database password (default: postgres)

    Returns:
        The answer to the query, including any SQL results and explanations
    """
    try:
        logging.info(f"RAG Router: Processing query: {query}")

        # Step 1: Use RAG router to select the best model for this query
        model_type, model_kwargs = get_routed_model(query)
        selected_model_name = model_kwargs.get("model_name", "unknown")
        logging.info(f"RAG Router: Selected model: {selected_model_name}")

        # Step 2: Create the selected model
        model = create_model(model=model_type, model_kwargs=model_kwargs)

        # Step 3: Set up database configuration
        agent_kwargs = {
            "database_type": "postgres",
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "tools": [
                "postgres_execute_sql",
                "postgres_list_schema",
            ],
        }

        # Step 4: Create SQL agent with the selected model
        agent = create_sql_agent(model=model, input_query=query, **agent_kwargs)

        # Step 5: Execute the query through the agent
        input_data = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": "1"}}

        # Process the agent stream
        responses, input_cost, output_cost, selected_model, cost_tracker = (
            process_agent_stream_with_store(
                agent.stream(input_data, config, stream_mode="updates"),
                store=get_shared_store(),
                agent_name="rag_router_sql_agent",
                model_name=model.model_name,
            )
        )

        # Step 6: Extract the final response
        final_response = "No response generated"
        if responses:
            last_response = responses[-1]
            if "agent" in last_response and "messages" in last_response["agent"]:
                for message in reversed(last_response["agent"]["messages"]):
                    if (
                        hasattr(message, "type")
                        and message.type == "ai"
                        and hasattr(message, "content")
                        and message.content
                        and message.content.strip()
                    ):
                        final_response = message.content
                        break

        # Step 7: Get executed SQL queries from shared store
        executed_queries = []
        shared_store = get_shared_store()
        stored_queries_obj = shared_store.get(("sql_execution",), "queries")
        if stored_queries_obj is not None:
            executed_queries = stored_queries_obj.value

        # Step 8: Format response with metadata
        result = f"[Model Selected: {selected_model_name}]\n\n{final_response}"

        if executed_queries:
            sql_section = "\n\nSQL Queries Executed:\n" + "\n".join(
                f"  {i}. {q}" for i, q in enumerate(executed_queries, 1) if q.strip()
            )
            result += sql_section

        # Add cost information if available
        if input_cost or output_cost:
            cost_section = f"\n\nCost: Input=${input_cost:.6f}, Output=${output_cost:.6f}, Total=${input_cost + output_cost:.6f}"
            result += cost_section

        logging.info("RAG Router: Query execution completed successfully")
        return result

    except Exception as e:
        error_msg = f"Error executing query with RAG router: {str(e)}"
        logging.error(error_msg)
        return error_msg


if __name__ == "__main__":
    # Run the local MCP server
    logging.info("Starting SQL Agent RAG Router MCP Server...")
    logging.info("Available tool: execute_sql_query_with_rag_router")
    mcp_server.run(transport="streamable-http")

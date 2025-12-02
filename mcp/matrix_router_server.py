import logging

from mcp.server.fastmcp import FastMCP

from src.agents.sql_agent_matrix_router import get_routed_model
from src.models.base import create_model
from src.schema import DatabaseConfig
from src.tools.sql_agent_tools import create_sql_agent_tools
from src.utils.db_clients import create_database_client

try:
    from .tool_converter import convert_tools_to_mcp
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from tool_converter import convert_tools_to_mcp

mcp_server = FastMCP("SQL Agent Matrix Router Tools")


def initialize_matrix_router_sql_agent_tools():
    """Initialize SQL agent with Matrix router and convert its tools to MCP tools."""

    # Default query for initialization - will be overridden when tool is called
    default_query = "Show me a simple count of all customers"

    # Model filename for the trained router
    model_filename = "router_model.pkl"

    # Get routed model based on default query
    model_type, model_kwargs = get_routed_model(default_query, model_filename)
    model = create_model(model=model_type, model_kwargs=model_kwargs)

    logging.info(
        f"Initialized Matrix router with model: {model_kwargs.get('model_name')}"
    )

    agent_kwargs = {
        "database_type": "postgres",
        "host": "localhost",
        "port": 5432,
        "database": "academic",
        "username": "postgres",
        "password": "postgres",
        "tools": [
            "postgres_execute_sql",
            "postgres_list_schema",
        ],
    }

    # Create database config and client
    db_config = DatabaseConfig(
        database_type=agent_kwargs["database_type"],
        host=agent_kwargs["host"],
        port=agent_kwargs["port"],
        database=agent_kwargs["database"],
        username=agent_kwargs["username"],
        password=agent_kwargs["password"],
    )

    db_client = create_database_client(**db_config.to_dict())

    tools = create_sql_agent_tools(
        db_client,
        dataset_id=db_config.dataset_id,
        project_id=db_config.project_id,
        database=db_config.database,
        requested_tools=agent_kwargs["tools"],
    )

    mcp_tools = convert_tools_to_mcp(tools, mcp_server)

    logging.info(
        f"Initialized {len(mcp_tools)} MCP tools from Matrix router SQL agent configuration"
    )
    return mcp_tools


matrix_router_sql_agent_tools = initialize_matrix_router_sql_agent_tools()


if __name__ == "__main__":
    # Run the local MCP server
    logging.info("Starting SQL Agent Matrix Router MCP Server...")
    logging.info(
        f"Available tools: {[tool.__name__ for tool in matrix_router_sql_agent_tools]}"
    )
    mcp_server.run(transport="streamable-http")

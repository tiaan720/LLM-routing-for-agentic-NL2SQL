import logging

from mcp.server.fastmcp import FastMCP

from src.agents.sql_agent import create_sql_agent
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

mcp_server = FastMCP("SQL Agent Tools")


def initialize_sql_agent_tools():
    """Initialize SQL agent and convert its tools to MCP tools."""

    model = create_model(
        model="vertex_ai",
        model_kwargs={
            "model": "gemini-2.5-flash",
            "temperature": 0.0,
            "project": "research-su-llm-routing",
            "location": "us-central1",
            "append_tools_to_system_message": True,
        },
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

    logging.info(f"Initialized {len(mcp_tools)} MCP tools from SQL agent configuration")
    return mcp_tools


sql_agent_tools = initialize_sql_agent_tools()


if __name__ == "__main__":
    # Run the local MCP server
    logging.info("Starting SQL Agent MCP Server...")
    logging.info("Server available at http://localhost:8000/mcp")
    logging.info(f"Available tools: {[tool.__name__ for tool in sql_agent_tools]}")
    # mcp_server.run(transport="sse")
    mcp_server.run(transport="streamable-http")

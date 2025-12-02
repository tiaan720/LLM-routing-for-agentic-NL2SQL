import logging

from mcp.server.fastmcp import FastMCP

from src.models.base import create_model
from src.schema import DatabaseConfig
from src.tools.supervisor_tool import create_supervisor_tools
from src.utils.db_clients import create_database_client

try:
    from .tool_converter import convert_tools_to_mcp
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from tool_converter import convert_tools_to_mcp

mcp_server = FastMCP("Supervisor Agent Tools")


def initialize_supervisor_agent_tools():
    """Initialize Supervisor agent and convert its tools to MCP tools."""

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
            "sql_agent_as_tool",
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

    # For supervisor tools, we need to provide a default input_query
    # This will be overridden by the actual query when the tool is called
    default_input_query = "Show me a simple count of all customers"

    tools = create_supervisor_tools(
        db_client,
        input_query=default_input_query,
        requested_tools=agent_kwargs["tools"],
    )

    mcp_tools = convert_tools_to_mcp(tools, mcp_server)

    logging.info(
        f"Initialized {len(mcp_tools)} MCP tools from Supervisor agent configuration"
    )
    return mcp_tools


supervisor_agent_tools = initialize_supervisor_agent_tools()


if __name__ == "__main__":
    logging.info("Starting Supervisor Agent MCP Server...")
    logging.info(
        f"Available tools: {[tool.__name__ for tool in supervisor_agent_tools]}"
    )
    mcp_server.run(transport="streamable-http")

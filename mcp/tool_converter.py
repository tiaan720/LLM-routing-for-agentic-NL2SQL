import inspect
import logging
from typing import Callable, List

from langchain_core.tools import BaseTool
from mcp.server.fastmcp import FastMCP

from src.cost_calc.shared_store import get_shared_store

# TODO: find way to remove the exec for a beter way.


def convert_langchain_tool_to_mcp(tool: BaseTool, mcp_server: FastMCP) -> Callable:
    """
    Convert a LangChain tool to an MCP tool.

    Args:
        tool: LangChain tool instance
        mcp_server: FastMCP server instance

    Returns:
        MCP tool function
    """

    # Get the tool's schema for parameters
    schema = tool.args_schema

    # Create the async wrapper function with proper parameter handling
    if schema and (hasattr(schema, "model_fields") or hasattr(schema, "__fields__")):
        # Extract field information for dynamic function creation - use model_fields (new) or __fields__ (deprecated fallback)
        fields = getattr(schema, "model_fields", None) or getattr(
            schema, "__fields__", {}
        )

        # Create parameter list excluding 'store' field
        param_names = [name for name in fields.keys() if name != "store"]

        # Create the function dynamically with proper parameters
        if param_names:
            # Create function with named parameters
            param_str = ", ".join(f"{name}=None" for name in param_names)

            # Build kwargs assignment lines with proper indentation
            kwargs_lines = []
            for name in param_names:
                kwargs_lines.append(
                    f"        if {name} is not None: kwargs['{name}'] = {name}"
                )
            kwargs_assignments = "\n".join(kwargs_lines)

            func_str = f'''async def mcp_tool_wrapper({param_str}):
                """
                Wrapper function that calls the original LangChain tool.
                """
                try:
                    # Build kwargs from parameters
                    kwargs = {{}}
            {kwargs_assignments}
                    
                    # Check if the tool function expects a 'store' parameter
                    sig = inspect.signature(tool.func)
                    if "store" in sig.parameters:
                        kwargs["store"] = get_shared_store()

                    # Call the original tool
                    result = tool.invoke(kwargs)
                    return result
                except Exception as e:
                    return f"Error executing {{tool.name}}: {{str(e)}}"
            '''
            # Create the function
            namespace = {
                "tool": tool,
                "inspect": inspect,
                "get_shared_store": get_shared_store,
                "str": str,
            }
            exec(func_str, namespace)
            mcp_tool_wrapper = namespace["mcp_tool_wrapper"]
        else:
            # Tool has no parameters (like postgres_list_schema)
            async def mcp_tool_wrapper():
                """
                Wrapper function that calls the original LangChain tool.
                """
                try:
                    # Check if the tool function expects a 'store' parameter
                    sig = inspect.signature(tool.func)
                    kwargs = {}
                    if "store" in sig.parameters:
                        kwargs["store"] = get_shared_store()

                    # Call the original tool
                    result = tool.invoke(kwargs)
                    return result
                except Exception as e:
                    return f"Error executing {tool.name}: {str(e)}"

    else:
        # Fallback for tools without schema
        async def mcp_tool_wrapper(**kwargs):
            """
            Wrapper function that calls the original LangChain tool.
            """
            try:
                # Check if the tool function expects a 'store' parameter
                sig = inspect.signature(tool.func)
                if "store" in sig.parameters:
                    kwargs["store"] = get_shared_store()

                # Call the original tool
                result = tool.invoke(kwargs)
                return result
            except Exception as e:
                return f"Error executing {tool.name}: {str(e)}"

    # Set the function name and docstring
    mcp_tool_wrapper.__name__ = tool.name
    mcp_tool_wrapper.__doc__ = tool.description

    # Register with MCP server
    return mcp_server.tool()(mcp_tool_wrapper)


def convert_tools_to_mcp(tools: List[BaseTool], mcp_server: FastMCP) -> List[Callable]:
    """
    Convert a list of LangChain tools to MCP tools.

    Args:
        tools: List of LangChain tool instances
        mcp_server: FastMCP server instance

    Returns:
        List of MCP tool functions
    """
    mcp_tools = []

    for tool in tools:
        mcp_tool = convert_langchain_tool_to_mcp(tool, mcp_server)
        mcp_tools.append(mcp_tool)
        logging.info(f"Converted tool: {tool.name}")

    return mcp_tools

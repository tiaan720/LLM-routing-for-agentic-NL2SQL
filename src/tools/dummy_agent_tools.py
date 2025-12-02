from typing import Optional

import dotenv
from langchain_core.tools import tool

dotenv.load_dotenv()


def create_dummy_agent_tools(
    requested_tools: Optional[list[str]] = None,
    **kwargs,
):

    @tool
    def plus_calculator(
        a: int,
        b: int,
    ) -> str:
        """
        Add two integer numbers together and return the sum.

        This tool performs simple addition of two integers.
        Use this tool when you need to calculate the sum of two numbers.

        Args:
            a: The first integer number to add
            b: The second integer number to add

        Returns:
            str: The sum of a + b as a string

        Example:
            plus_calculator(1, 1) returns "2"
            plus_calculator(5, 3) returns "8"
        """
        result = a + b
        return f"Result: {result}"

    # Get all tool functions from local scope
    all_tools = {
        name: func
        for name, func in locals().items()
        if hasattr(func, "name") and hasattr(func, "invoke")
    }

    tools = []
    if requested_tools is None:
        tools.extend(all_tools.values())
    else:
        for tool_name in requested_tools:
            if tool_name in all_tools:
                tools.append(all_tools[tool_name])

    return tools

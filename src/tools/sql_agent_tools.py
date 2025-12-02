import os
from datetime import datetime
from typing import Annotated, Any, Literal

import dotenv
import pandas as pd
import plotly.express as px
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedStore
from pydantic import BaseModel

from src.schema import ArtifactModel
from src.utils.db_clients import DatabaseClient
from src.utils.logger import logger

dotenv.load_dotenv()


def create_sql_agent_tools(
    db_client: DatabaseClient,
    dataset_id: str = None,
    project_id: str = None,
    database: str = None,
    requested_tools: list = None,
    **kwargs,
):
    """Creates tools with proper state management for both BigQuery and PostgreSQL

    Args:
        db_client: Database client instance
        dataset_id: BigQuery dataset ID (for BigQuery)
        project_id: BigQuery project ID (for BigQuery)
        database: PostgreSQL database name (for PostgreSQL)
        requested_tools: Optional list of tool names to create. If None, creates all available tools.
        **kwargs: Additional configuration parameters
    """

    client = db_client.client
    db_type = db_client.db_type

    @tool
    def postgres_execute_sql(query: str, store: Annotated[Any, InjectedStore()]) -> str:
        """
        Execute PostgreSQL SQL query.

        Args:
            query: SQL string (SELECT, INSERT, UPDATE, DELETE)

        Example:
            "SELECT * FROM customers WHERE city = 'Boston'"

        Returns:
            Query results or affected rows count
        """
        try:
            cursor = client.cursor()
            cursor.execute(query)

            # Store the executed query in the shared store for supervisor access
            from src.cost_calc.shared_store import get_shared_store

            shared_store = get_shared_store()
            try:
                # Get existing queries from shared store
                existing_queries_obj = shared_store.get(("sql_execution",), "queries")
                if existing_queries_obj is not None:
                    existing_queries = existing_queries_obj.value
                else:
                    existing_queries = []
                existing_queries.append(query)
                shared_store.put(("sql_execution",), "queries", existing_queries)
            except Exception as e:
                pass  # Continue even if storing fails

            # For SELECT queries, fetch results
            if query.strip().upper().startswith("SELECT"):
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                # Convert to list of dictionaries
                result_list = [dict(zip(columns, row)) for row in results]

                result_str = str(result_list)

                store.put(("values",), "last_query_result", result_list)

                if len(result_str) > 500:
                    summary = {
                        "row_count": len(result_list),
                        "preview": result_list[:3],
                        "columns": columns,
                    }
                    return f"Query results stored. Found {summary['row_count']} rows. Preview: {summary['preview']}"
                else:
                    return result_str
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE)
                client.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"

        except Exception as e:
            client.rollback()
            raise ToolException(f"Query execution failed: {str(e)}")
        finally:
            cursor.close()

    @tool
    def postgres_list_schema() -> str:
        """
        List PostgreSQL database schema (tables and columns).
        Use this FIRST to understand database structure.

        Returns:
            Formatted string with tables, columns, and data types
        """
        try:
            cursor = client.cursor()

            # Get all tables in public schema
            cursor.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """
            )

            tables = cursor.fetchall()
            schema_info = []
            print(f"Database name: {database}")

            for (table_name,) in tables:
                schema_info.append(f"\nTable: {table_name}")

                # Get column information for each table
                cursor.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                """,
                    (table_name,),
                )

                columns = cursor.fetchall()
                schema_info.append("Columns:")
                for column_name, data_type, is_nullable in columns:
                    nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                    schema_info.append(f"  - {column_name}: {data_type} ({nullable})")

            cursor.close()
            return "\n".join(schema_info)

        except Exception as e:
            raise ToolException(f"Error listing schema: {str(e)}")

    class PlotParameters(BaseModel):
        x: str
        y: str | None = None
        color: str | None = None
        title: str | None = None
        labels: dict | None = None
        template: str = "plotly"

        model_config = {
            "extra": "allow",
            "json_schema_extra": {
                "properties": {
                    "x": {"description": "Column name for x-axis", "type": "string"},
                    "y": {"description": "Column name for y-axis", "type": "string"},
                    "color": {"description": "Column name for color", "type": "string"},
                    "title": {"description": "Plot title", "type": "string"},
                    "labels": {"description": "Custom axis labels", "type": "object"},
                    "template": {
                        "description": "Plotly template to use",
                        "type": "string",
                    },
                }
            },
        }

    class PlotRequest(BaseModel):
        plot_type: Literal[
            "scatter",
            "line",
            "bar",
            "histogram",
            "box",
            "violin",
            "pie",
            "sunburst",
            "treemap",
            "funnel",
            "density_contour",
            "density_heatmap",
            "ecdf",
            "strip",
            "area",
            "scatter_3d",
            "line_3d",
            "scatter_matrix",
            "parallel_coordinates",
            "parallel_categories",
        ]
        parameters: PlotParameters

    @tool(response_format="content_and_artifact")
    def generate_plot_from_data(
        request: PlotRequest, store: Annotated[Any, InjectedStore()]
    ) -> str:
        """
        Create Plotly visualization from last SQL query results.

        Args:
            request: PlotRequest with plot_type and parameters

        Example:
            PlotRequest(plot_type="bar", parameters={"x": "category", "y": "count"})

        Returns:
            Plot description and Plotly artifact
        """
        result_list = store.get(("values",), "last_query_result").value
        df = pd.DataFrame(result_list)

        # Convert parameters to dictionary and unpack
        plot_params = request.parameters.model_dump(exclude_none=True)

        # Validate plot parameters
        if not isinstance(plot_params, dict):
            raise ToolException(
                "Invalid plot parameters: parameters should be a dictionary"
            )

        # Get the appropriate plotting function
        plot_func = getattr(px, request.plot_type, None)
        if plot_func is None:
            raise ToolException(f"Unsupported plot type: {request.plot_type}")

        try:
            fig = plot_func(df, **plot_params)
        except Exception as e:
            raise ToolException(f"Error generating plot: {str(e)}")

        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plots_dir}/{request.plot_type}_{timestamp}.png"
        fig.write_image(filename)

        content = f"Generated a {request.plot_type} plot"

        data = fig.to_json()
        artifact = ArtifactModel(data=data, type="plotly")

        return content, artifact

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

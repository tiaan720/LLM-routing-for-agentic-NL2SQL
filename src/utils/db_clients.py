"""Database client factory for SQL agents."""

import os
from typing import Any, Dict, Optional, Union

import dotenv
import psycopg2
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

dotenv.load_dotenv()


class DatabaseClient:
    """Base class for database clients."""

    def __init__(self, client: Any, db_type: str, **kwargs):
        self.client = client
        self.db_type = db_type
        # Store all extra kwargs as attributes (for connection params)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def close(self):
        """Close the database connection if applicable."""
        if self.db_type == "postgres" and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass  # Ignore errors when closing
        # BigQuery client doesn't need explicit closing

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()


def create_bigquery_client(user_token: str, project_id: str) -> DatabaseClient:
    """
    Create a BigQuery client.

    Args:
        user_token: BigQuery user token
        project_id: BigQuery project ID

    Returns:
        DatabaseClient: Configured BigQuery client wrapper
    """
    if not user_token:
        raise ValueError("BigQuery user token is required")
    if not project_id:
        raise ValueError("BigQuery project ID is required")

    user_credentials = Credentials(
        token=user_token,
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )

    client = bigquery.Client(project=project_id, credentials=user_credentials)
    return DatabaseClient(
        client, "bigquery", user_token=user_token, project_id=project_id
    )


def create_postgres_client(
    host: str, port: int, database: str, username: str, password: str
) -> DatabaseClient:
    """
    Create a PostgreSQL client.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: PostgreSQL database name
        username: PostgreSQL username
        password: PostgreSQL password

    Returns:
        DatabaseClient: Configured PostgreSQL client wrapper
    """
    if not all([host, port, database, username, password]):
        raise ValueError("All PostgreSQL connection parameters are required")

    try:
        client = psycopg2.connect(
            host=host, port=port, database=database, user=username, password=password
        )
        return DatabaseClient(
            client,
            "postgres",
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
        )
    except psycopg2.Error as e:
        raise ValueError(f"Failed to connect to PostgreSQL: {e}")


def create_database_client(**kwargs) -> DatabaseClient:
    """
    Factory function to create database clients based on configuration.

    Args:
        **kwargs: Database configuration parameters

    Returns:
        DatabaseClient: Configured database client

    Raises:
        ValueError: If configuration is invalid or incomplete
    """
    database_type = kwargs.get("database_type") or kwargs.get("type")

    if database_type == "bigquery":
        return create_bigquery_client(
            user_token=kwargs.get("user_token"),
            project_id=kwargs.get("project_id"),
        )
    elif database_type == "postgres":
        return create_postgres_client(
            host=kwargs.get("host"),
            port=kwargs.get("port", 5432),
            database=kwargs.get("database"),
            username=kwargs.get("username"),
            password=kwargs.get("password"),
        )
    else:
        raise ValueError(f"Unsupported database type: {database_type}")


def validate_db_config(db_config: Dict[str, Any]) -> bool:
    """
    Validate database configuration.

    Args:
        db_config: Database configuration dictionary

    Returns:
        bool: True if configuration is valid
    """
    db_type = db_config.get("type")

    if db_type == "bigquery":
        required_fields = ["user_token", "project_id", "dataset_id"]
        return all(db_config.get(field) for field in required_fields)
    elif db_type == "postgres":
        required_fields = ["host", "database", "username", "password"]
        return all(db_config.get(field) for field in required_fields)

    return False

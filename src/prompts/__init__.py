"""Prompt templates for SQL agents."""

from .postgres_prompts import (
    get_postgres_system_prompt,
    get_postgres_system_prompt_text,
)
from .prompt_utils import get_system_prompt_for_db_type

__all__ = [
    "get_postgres_system_prompt",
    "get_postgres_system_prompt_text",
    "get_system_prompt_for_db_type",
]

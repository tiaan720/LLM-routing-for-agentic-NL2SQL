from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ArtifactModel(BaseModel):
    data: str | Dict[str, Any] = Field(
        description="Artifact data in string or dictionary format"
    )
    type: str = Field(description="The artifact type")


class DatabaseConfig(BaseModel):
    """Database configuration parameters."""

    database_type: Optional[str] = None
    user_token: Optional[str] = None
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

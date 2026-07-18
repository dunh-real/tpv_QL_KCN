import re
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class IngestRequest(BaseModel):
    """Request body for submitting a document to the ingestion queue."""

    file_url: HttpUrl = Field(
        ..., description="Presigned URL of the markdown file on MinIO"
    )
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename (e.g. 'report.md')",
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if re.search(r'[/\\<>:"|?*\x00-\x1f]', v):
            raise ValueError("Filename contains invalid characters")
        return v.strip()


class IngestQueueResponse(BaseModel):
    """Response returned when a document is queued for ingestion."""

    status: str = Field(default="queued")
    message: str = Field(
        default="File has been queued for background ingestion."
    )
    task_id: str


class IngestStatusResponse(BaseModel):
    """Response returned when checking the status of an ingestion task."""

    task_id: str
    status: str
    progress: Optional[int] = Field(
        default=None, description="Processing progress 0-100%"
    )
    step: Optional[str] = Field(
        default=None, description="Current processing step"
    )
    result: Optional[dict] = Field(
        default=None,
        description="Ingest result with parent_chunks and children_chunks count",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the task failed"
    )

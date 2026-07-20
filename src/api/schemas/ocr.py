import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, HttpUrl


class OCRRequest(BaseModel):
    """Request body for submitting a document to the OCR queue."""

    file_url: HttpUrl = Field(..., description="Presigned URL of the source file")
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename (e.g. 'report.pdf')",
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if re.search(r'[/\\<>:"|?*\x00-\x1f]', v):
            raise ValueError("Filename contains invalid characters")
        return v.strip()


class OCRQueueResponse(BaseModel):
    """Response returned when a document is queued for OCR."""

    status: str = Field(default="queued")
    message: str = Field(default="File has been queued for background processing.")
    task_id: str


class OCRStatusResponse(BaseModel):
    """Response returned when checking the status of an OCR task."""

    task_id: str
    status: str
    progress: Optional[int] = Field(
        default=None, description="Processing progress 0-100%"
    )
    step: Optional[str] = Field(
        default=None, description="Current processing step"
    )
    result_file: Optional[str] = Field(
        default=None, description="Name of the result file in MinIO"
    )
    download_url: Optional[str] = Field(
        default=None, description="Presigned download URL (valid for 1 hour)"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the task failed"
    )

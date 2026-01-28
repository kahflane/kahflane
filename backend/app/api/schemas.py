"""
Common API response schemas for OpenAPI documentation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(examples=["Resource not found"])


class ValidationErrorDetail(BaseModel):
    """Validation error detail item."""
    loc: list
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Validation error response (422)."""
    detail: list[ValidationErrorDetail]

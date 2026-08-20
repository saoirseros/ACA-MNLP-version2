"""Pydantic response models for the /health endpoint."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

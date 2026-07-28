"""Pydantic response models for the opinion job API. Deliberately carries
only metadata (ids, counts, booleans, status strings, error text) -- never
document content."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    sale_deed_count: int
    has_revenue_record: bool
    has_ec: bool
    created_at: float
    finished_at: Optional[float] = None

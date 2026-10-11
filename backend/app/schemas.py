"""JSON shapes for the job API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StageProgress(BaseModel):
    name: str
    status: str
    elapsed_ms: float | None = None
    message: str | None = None
    provider: str | None = None


class JobState(BaseModel):
    id: str
    status: str
    image_hash: str
    cached: bool = False
    stages: list[StageProgress]
    error: str | None = None
    region_count: int | None = None
    source: str | None = None
    created_at: str


class JobView(JobState):
    artifacts: dict[str, str] = Field(default_factory=dict)


class ConvertResponse(BaseModel):
    job_id: str


class CacheMeta(BaseModel):
    image_hash: str
    region_count: int
    source: str
    stages: list[StageProgress]


class SampleInfo(BaseModel):
    id: str
    title: str
    description: str
    file: str
    hash: str


class SampleView(BaseModel):
    id: str
    title: str
    description: str
    hash: str
    image_url: str


class AskPart(BaseModel):
    name: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=400)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=400)
    subject: str = Field(default="", max_length=200)
    summary: str = Field(default="", max_length=800)
    parts: list[AskPart] = Field(default_factory=list, max_length=12)


class AskResponse(BaseModel):
    answer: str = ""
    source: str

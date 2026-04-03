"""
Shared data schema for the hiring automation pipeline.
Every module consumes and produces data conforming to these models.

Design decision: Pydantic v2 for validation + serialization.
I tried dataclasses first but needed the JSON schema export and 
field-level validation that Pydantic gives for free.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import Optional, List
from datetime import datetime
import uuid
import re


class Answer(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question_text: str = ""
    answer_text: str = ""
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @field_validator("answer_text", mode="before")
    @classmethod
    def clean_answer(cls, v):
        if v is None:
            return ""
        return str(v).strip()


class Profile(BaseModel):
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    resume_url: Optional[str] = None

    @field_validator("github_url", mode="before")
    @classmethod
    def validate_github(cls, v):
        if v and isinstance(v, str) and v.strip():
            v = v.strip()
            # Normalize github URLs
            if "github.com" not in v.lower() and not v.startswith("http"):
                v = f"https://github.com/{v}"
            return v
        return None

    @field_validator("linkedin_url", mode="before")
    @classmethod
    def validate_linkedin(cls, v):
        if v and isinstance(v, str) and v.strip():
            v = v.strip()
            if "linkedin.com" not in v.lower() and not v.startswith("http"):
                v = f"https://linkedin.com/in/{v}"
            return v
        return None


class CandidateMetadata(BaseModel):
    response_time_seconds: float = 0.0
    round: int = 1
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Candidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    email: str = ""
    role_applied_for: str = ""
    answers: List[Answer] = Field(default_factory=list)
    profile: Profile = Field(default_factory=Profile)
    metadata: CandidateMetadata = Field(default_factory=CandidateMetadata)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v):
        if v is None:
            return ""
        v = str(v).strip().lower()
        # Basic email format check - not using EmailStr because
        # we want to handle bad data gracefully, not crash
        if v and "@" not in v:
            return ""
        return v

    @field_validator("name", mode="before")
    @classmethod
    def clean_name(cls, v):
        if v is None:
            return ""
        # Remove extra whitespace, title case
        return " ".join(str(v).strip().split())

    def to_json(self) -> dict:
        return self.model_dump()


class ScoredCandidate(BaseModel):
    """Output from the scoring module"""
    candidate: Candidate
    total_score: float = 0.0
    dimension_scores: dict = Field(default_factory=dict)
    tier: str = "Review"  # Fast-Track / Standard / Review / Reject
    explanation: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    cheat_probability: float = 0.0


class ModuleOutput(BaseModel):
    """Standard output wrapper for every module"""
    module: str
    status: str = "success"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    data: dict = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

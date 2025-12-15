"""
Pydantic models for memory system.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WorkingMemoryEntry(BaseModel):
    """Entry in working memory (ToT reasoning step)."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    iteration: int = Field(..., ge=0, description="Iteration number")
    node_id: str = Field(..., description="ToT node ID")
    depth: int = Field(..., ge=0, description="Tree depth")
    thought: str = Field(..., description="Agent's thought")
    tool_used: str | None = Field(None, description="Tool name")
    tool_params: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    observation: str = Field(default="", description="Observation from tool")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness score")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")


class ProceduralPattern(BaseModel):
    """Successful pattern stored in procedural memory."""

    pattern_id: str = Field(..., description="Unique pattern ID")
    topic_category: str = Field(..., description="Topic category (e.g., 'sorting')")
    user_level: str = Field(..., description="User level (beginner/intermediate/advanced)")
    tools_sequence: list[str] = Field(..., description="Sequence of tools used")
    avg_iterations: float = Field(..., ge=0, description="Average iterations to goal")
    success_score: float = Field(..., ge=0.0, le=1.0, description="Success score")
    usage_count: int = Field(default=1, ge=1, description="How many times used")
    reasoning_pattern: str = Field(..., description="Description of reasoning pattern")
    created_at: datetime = Field(default_factory=datetime.now, description="Created timestamp")
    last_used: datetime = Field(default_factory=datetime.now, description="Last used timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "pattern_id": "pat_xyz789",
                "topic_category": "sorting",
                "user_level": "intermediate",
                "tools_sequence": ["adaptive_rag", "corrective_check", "web_search"],
                "avg_iterations": 3.2,
                "success_score": 0.92,
                "usage_count": 15,
                "reasoning_pattern": "Start with RAG to get theory, verify quality, then web search for examples",
            }
        }


class MemoryContext(BaseModel):
    """Context loaded from memory for new request."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    procedural_hints: str = Field(default="", description="Hints from procedural memory")
    patterns: list[dict[str, Any]] = Field(default_factory=list, description="Loaded patterns")

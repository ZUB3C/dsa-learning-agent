"""
Pydantic models for 4get API response parsing.
"""

from typing import Any

from pydantic import BaseModel, Field


class ThumbInfo(BaseModel):
    """Thumbnail information."""

    ratio: str | None = None
    url: str | None = None


class WebResult(BaseModel):
    """Individual web search result."""

    title: str
    description: str
    url: str
    date: int | None = None
    type: str
    thumb: ThumbInfo
    sublink: list[Any] = Field(default_factory=list)
    table: dict[str, Any] | list[Any] = Field(default_factory=dict)


class SpellingInfo(BaseModel):
    """Spelling correction information."""

    type: str
    using: str | None = None
    correction: str | None = None


class FourGetResponse(BaseModel):
    """4get API response model."""

    status: str
    spelling: SpellingInfo
    npt: str
    answer: list[Any] = Field(default_factory=list)
    web: list[WebResult] = Field(default_factory=list)
    image: list[Any] = Field(default_factory=list)
    video: list[Any] = Field(default_factory=list)
    news: list[Any] = Field(default_factory=list)
    related: list[Any] = Field(default_factory=list)

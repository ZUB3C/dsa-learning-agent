"""
Pydantic schemas for Content Guard Layer.
"""

from pydantic import BaseModel, Field


class ToxicityCheckResult(BaseModel):
    """Result from toxicity check."""

    doc_id: int = Field(..., description="Document ID")
    is_safe: bool = Field(..., description="Is document safe")
    toxicity_score: float = Field(..., ge=0.0, le=1.0, description="Toxicity score 0-1")
    issues: list[str] = Field(default_factory=list, description="List of issues found")


class ToxicityBatchResult(BaseModel):
    """Batch toxicity check result."""

    results: list[ToxicityCheckResult] = Field(..., description="Results for each document")
    avg_toxicity: float = Field(0.0, description="Average toxicity score")
    filtered_count: int = Field(0, description="Number of documents filtered")


class PolicyCheckResult(BaseModel):
    """Result from policy compliance check."""

    compliant: bool = Field(..., description="Is compliant with policies")
    violations: list[str] = Field(default_factory=list, description="Policy violations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ContentSanitizationResult(BaseModel):
    """Result from content sanitization."""

    original_length: int = Field(..., description="Original content length")
    sanitized_length: int = Field(..., description="Sanitized content length")
    removed_elements: list[str] = Field(default_factory=list, description="Removed elements")
    sanitized_content: str = Field(..., description="Sanitized content")


class QualityGateResult(BaseModel):
    """Result from quality gate check."""

    passed: bool = Field(..., description="Passed quality checks")
    length_ok: bool = Field(..., description="Length check passed")
    sentence_count_ok: bool = Field(..., description="Sentence count check passed")
    content_type_ok: bool = Field(..., description="Content type check passed")
    reason: str | None = Field(None, description="Failure reason")


class CleanDocument(BaseModel):
    """Document after Content Guard processing."""

    page_content: str = Field(..., description="Cleaned content")
    metadata: dict = Field(default_factory=dict, description="Document metadata")
    source: str = Field("", description="Source URL or identifier")

    # Content Guard metadata
    content_guarded: bool = Field(True, description="Processed by Content Guard")
    toxicity_score: float = Field(0.0, description="Toxicity score")
    policy_compliant: bool = Field(True, description="Policy compliant")
    sanitized: bool = Field(True, description="Content sanitized")
    quality_passed: bool = Field(True, description="Quality checks passed")


class ContentGuardReport(BaseModel):
    """Summary report from Content Guard."""

    total_documents: int = Field(..., description="Total documents checked")
    passed_documents: int = Field(..., description="Documents that passed")
    filtered_by_toxicity: int = Field(0, description="Filtered by toxicity")
    filtered_by_policy: int = Field(0, description="Filtered by policy")
    filtered_by_quality: int = Field(0, description="Filtered by quality")
    avg_toxicity_score: float = Field(0.0, description="Average toxicity score")
    processing_time_ms: float = Field(0.0, description="Processing time")

    @property
    def filter_rate(self) -> float:
        """Calculate filter rate."""
        if self.total_documents == 0:
            return 0.0
        filtered = self.filtered_by_toxicity + self.filtered_by_policy + self.filtered_by_quality
        return filtered / self.total_documents

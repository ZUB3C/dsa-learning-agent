from enum import Enum
from pydantic import BaseModel


class TaskType(str, Enum):
    MATERIALS = "materials"
    TEST = "test"
    VERIFICATION = "verification"
    SUPPORT = "support"


class ClassificationResult(BaseModel):
    task_type: TaskType
    include_support: bool
    topic: str | None = None
    question: str | None = None
    user_answer: str | None = None
    reasoning: str | None = None


class ResolveRequest(BaseModel):
    user_id: str
    message: str
    user_level: str | None = "intermediate"


class SupportBlock(BaseModel):
    message: str
    recommendations: list[str] = []


class ResolveResponse(BaseModel):
    status: str
    main_content: str
    task_type: TaskType
    support: SupportBlock | None = None
    agents_used: list[str]
    execution_time_ms: int

from typing import Any, Literal

from pydantic import BaseModel, Field


# Модуль 1: Проверка тестирований
class VerificationDetails(BaseModel):
    """Детали верификации без баллов"""

    verification_id: str = Field(description="ID верификации")
    primary_is_correct: bool = Field(description="Вердикт первичной проверки")
    secondary_is_correct: bool | None = Field(
        default=None, description="Вердикт вторичной проверки"
    )
    agree_with_primary: bool | None = Field(
        default=None, description="Согласие вторичной с первичной"
    )
    verification_notes: str | None = Field(default=None, description="Заметки верификации")


class TestVerificationRequest(BaseModel):
    """Запрос на верификацию теста"""

    test_id: str = Field(description="ID теста")
    user_answer: str = Field(description="Ответ пользователя")
    language: str = Field(default="ru", description="Язык (ru/en)")
    question: str = Field(description="Текст вопроса")
    expected_answer: str | None = Field(default=None, description="Эталонный ответ")
    secondary_check: bool = Field(default=True, description="Использовать вторичную проверку")


class TestVerificationResponse(BaseModel):
    is_correct: bool = Field(description="Правильность ответа")
    feedback: str = Field(description="Обратная связь")
    verification_details: VerificationDetails = Field(description="Детали проверки")


# Модуль 2: Первичная оценка
class AssessmentStartRequest(BaseModel):
    user_id: str = Field(description="ID пользователя")


class AssessmentQuestion(BaseModel):
    question_id: int
    question_text: str
    options: list[str] | None = None


class AssessmentStartResponse(BaseModel):
    test_questions: list[AssessmentQuestion]
    session_id: str


class AssessmentSubmitRequest(BaseModel):
    session_id: str
    answers: list[dict[str, Any]]


class AssessmentSubmitResponse(BaseModel):
    level: Literal["beginner", "intermediate", "advanced"]
    knowledge_areas: dict[str, float]
    recommendations: list[str]


# Модуль 3: Материалы
class GetMaterialsRequest(BaseModel):
    topic: str = Field(description="Тема")
    user_level: str = Field(description="Уровень пользователя")
    language: str = Field(default="ru")


class GetMaterialsResponse(BaseModel):
    content: str = Field(description="Адаптированный контент")
    sources: list[str] = Field(description="Источники")
    adapted_for_level: str


class AskQuestionRequest(BaseModel):
    question: str
    context_topic: str
    user_level: str
    language: str = "ru"


class AskQuestionResponse(BaseModel):
    answer: str
    related_concepts: list[str]


class AddCustomTopicRequest(BaseModel):
    topic_name: str
    user_id: str
    content: str


class AddCustomTopicResponse(BaseModel):
    topic_id: str
    status: str


class TopicInfo(BaseModel):
    topic_id: str
    topic_name: str
    user_id: str


class GetTopicsResponse(BaseModel):
    predefined_topics: list[str]
    custom_topics: list[TopicInfo]


class SearchMaterialsRequest(BaseModel):
    query: str
    filters: dict[str, Any] | None = None


class MaterialSearchResult(BaseModel):
    content: str
    metadata: dict[str, Any]


class SearchMaterialsResponse(BaseModel):
    results: list[MaterialSearchResult]
    relevance_scores: list[float]


class GenerateMaterialRequest(BaseModel):
    topic: str
    format: str
    length: str
    language: str = "ru"


class GenerateMaterialResponse(BaseModel):
    material: str
    format: str
    word_count: int
    model_used: str
    topic_id: str


# Модуль 4: Генерация тестов
class GenerateTestRequest(BaseModel):
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    question_count: int = Field(default=5, ge=1, le=20)
    language: str = "ru"


class TestQuestion(BaseModel):
    question_id: int
    question_text: str
    expected_answer: str
    key_points: list[str]


class GenerateTestResponse(BaseModel):
    test_id: str
    questions: list[TestQuestion]
    expected_duration: int


class GenerateTaskRequest(BaseModel):
    topic: str
    difficulty: str
    task_type: str
    language: str = "ru"


class TaskHint(BaseModel):
    hint_level: int
    hint_text: str


class Task(BaseModel):
    task_id: int | str
    description: str
    topic: str
    difficulty: str
    task_type: str
    expected_answer: str | None = None


class GenerateTaskResponse(BaseModel):
    task: Task
    solution_hints: list[TaskHint]
    model_used: str


class GetTestResponse(BaseModel):
    test: dict[str, Any]
    metadata: dict[str, Any]


class CompletedTestInfo(BaseModel):
    result_id: int
    test_id: str
    topic: str
    difficulty: str
    submitted_at: str


class GetCompletedTestsResponse(BaseModel):
    completed_tests: list[CompletedTestInfo]
    statistics: dict[str, int]


class SubmitTestRequest(BaseModel):
    test_id: str
    user_id: str
    answers: list[dict[str, Any]]


class SubmitTestResponse(BaseModel):
    verification_id: str
    status: str


# Модуль 5: LLM Router
class LLMRouterRequest(BaseModel):
    request_type: Literal["material", "task", "test", "question", "support"]
    content: str
    language: str = "ru"
    parameters: dict[str, Any] = Field(default_factory=dict)


class LLMRouterResponse(BaseModel):
    generated_content: str
    model_used: str
    metadata: dict[str, Any]


class ModelInfo(BaseModel):
    name: str
    language: str
    provider: str


class GetAvailableModelsResponse(BaseModel):
    models: list[ModelInfo]
    capabilities: dict[str, bool]


class RouteRequestRequest(BaseModel):
    request_type: str
    content: str
    context: dict[str, Any] | None = None
    language: str = "ru"


class RouteRequestResponse(BaseModel):
    selected_model: str
    reasoning: str
    confidence: float
    alternative_models: list[str]


# Модуль 6: Психологическая поддержка
class SupportRequest(BaseModel):
    message: str
    emotional_state: str = Field(
        description="Эмоциональное состояние (stressed, confused, motivated, etc.)"
    )
    language: str = "ru"


class SupportResponse(BaseModel):
    support_message: str
    recommendations: list[str]
    resources: list[dict[str, str]]


class GetSupportResourcesResponse(BaseModel):
    articles: list[dict[str, str]]
    exercises: list[dict[str, str]]
    tips: list[str]


class SubmitFeedbackRequest(BaseModel):
    session_id: str
    helpful: bool
    comments: str = ""


class SubmitFeedbackResponse(BaseModel):
    status: str


# Модуль 7: Verification History
class VerificationHistoryItem(BaseModel):
    verification_id: str
    test_id: str
    question: str
    is_correct: bool
    created_at: str


class GetVerificationHistoryResponse(BaseModel):
    tests: list[VerificationHistoryItem]
    accuracy_rate: float  # Процент правильных ответов
    total_tests: int


# Модуль 8: Assessment Results
class GetAssessmentResultsResponse(BaseModel):
    message: str | None = None
    user_id: str | None = None
    initial_level: str | None = None
    score: float | None = None
    knowledge_areas: dict[str, float] | None = None
    recommendations: list[str] | None = None
    completed_at: str | None = None


# Модуль 9: Health Check
class HealthCheckResponse(BaseModel):
    status: str
    time: str


# Модуль 10: Root
class RootResponse(BaseModel):
    message: str
    version: str
    docs: str


# ═══════════════════════════════════════════════════════════════
# V2 SCHEMAS - Materials Agent with Tree-of-Thoughts
# ═══════════════════════════════════════════════════════════════


class GenerateMaterialV2Request(BaseModel):
    """Request for v2 material generation with ToT."""

    topic: str = Field(..., description="Topic to generate materials for")
    user_level: Literal["beginner", "intermediate", "advanced"] = Field(
        ..., description="User knowledge level"
    )
    user_id: str = Field(..., description="User ID for memory context")

    # Optional parameters
    language: str = Field(default="ru", description="Language (ru/en)")
    max_iterations: int | None = Field(
        default=None, description="Max ToT iterations (override config)"
    )
    completeness_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Completeness threshold (override config)"
    )

    # Feature flags (optional overrides)
    enable_web_search: bool | None = Field(
        default=None, description="Enable web search (override config)"
    )
    enable_adaptive_rag: bool | None = Field(
        default=None, description="Enable adaptive RAG (override config)"
    )


class ToTNodeInfo(BaseModel):
    """Information about a ToT node."""

    node_id: str
    depth: int
    thought: str
    tool_used: str | None
    completeness_score: float
    promise_score: float
    status: str


class ToTSearchMetrics(BaseModel):
    """Metrics from ToT search."""

    total_iterations: int
    explored_nodes: int
    best_path_length: int
    final_completeness: float
    final_relevance: float
    final_quality: float

    # Tool usage
    tools_used: list[str]
    tool_call_counts: dict[str, int]

    # LLM usage
    gigachat2_max_calls: int
    gigachat3_calls: int
    estimated_cost_usd: float

    # Performance
    total_time_seconds: float

    # Memory
    memory_hints_used: bool
    procedural_patterns_found: int


class MaterialSource(BaseModel):
    """Source of material."""

    source_type: str  # "rag", "web", "memory"
    url: str | None = None
    title: str | None = None
    relevance_score: float


class GenerateMaterialV2Response(BaseModel):
    """Response for v2 material generation."""

    # Generation metadata
    generation_id: str
    success: bool

    # Generated content
    material: str
    word_count: int

    # Sources
    sources: list[MaterialSource]
    documents_collected: int

    # ToT metrics
    tot_metrics: ToTSearchMetrics

    # Best path (optional, for debugging)
    best_path: list[ToTNodeInfo] | None = None

    # Warnings/errors
    warnings: list[str] = []
    fallbacks_used: list[str] = []


class GetGenerationStatusRequest(BaseModel):
    """Request to get generation status."""

    generation_id: str


class GetGenerationStatusResponse(BaseModel):
    """Response with generation status."""

    generation_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: float  # 0-1
    current_iteration: int | None = None
    estimated_time_remaining: float | None = None

    # If completed
    result: GenerateMaterialV2Response | None = None


class HealthCheckV2Response(BaseModel):
    """Enhanced health check response."""

    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str

    # Component status
    components: dict[str, dict[str, Any]] = Field(description="Status of each component")

    # Metrics
    metrics: dict[str, float] = Field(description="System metrics")

    # Features
    features: dict[str, bool] = Field(description="Feature flags")


class SystemMetricsResponse(BaseModel):
    """System metrics response."""

    # Request metrics (last hour)
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float

    # LLM usage (last hour)
    gigachat2_max_calls: int
    gigachat3_calls: int
    total_cost_usd: float

    # Tool usage (last hour)
    tool_usage: dict[str, int]

    # Content Guard (last hour)
    documents_filtered: int
    filter_rate: float

    # Memory
    procedural_patterns_stored: int
    working_memory_sessions: int


class GetLogsRequest(BaseModel):
    """Request to get generation logs."""

    generation_id: str | None = None
    user_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = Field(default=50, ge=1, le=1000)


class GenerationLogEntry(BaseModel):
    """Generation log entry."""

    generation_id: str
    user_id: str
    topic: str
    user_level: str
    success: bool
    final_completeness: float
    total_time: float
    created_at: str


class GetLogsResponse(BaseModel):
    """Response with generation logs."""

    logs: list[GenerationLogEntry]
    total_count: int
    page: int
    page_size: int

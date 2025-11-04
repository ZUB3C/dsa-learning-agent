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

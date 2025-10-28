from typing import Any, Literal

from pydantic import BaseModel, Field


# Модуль 1: Проверка тестирований
class TestVerificationRequest(BaseModel):
    test_id: str = Field(description="ID теста")
    user_answer: str = Field(description="Ответ пользователя")
    language: str = Field(default="ru", description="Язык (ru/en)")
    question: str = Field(description="Текст вопроса")
    expected_answer: str | None = Field(default=None, description="Эталонный ответ")


class TestVerificationResponse(BaseModel):
    is_correct: bool = Field(description="Правильность ответа")
    score: float = Field(description="Оценка от 0 до 100")
    feedback: str = Field(description="Обратная связь")
    verification_details: dict[str, Any] = Field(description="Детали проверки")


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


# Модуль 6: Психологическая поддержка
class SupportRequest(BaseModel):
    user_id: str
    message: str
    emotional_state: str = Field(description="Эмоциональное состояние (stressed, confused, motivated, etc.)")
    language: str = "ru"


class SupportResponse(BaseModel):
    support_message: str
    recommendations: list[str]
    resources: list[dict[str, str]]

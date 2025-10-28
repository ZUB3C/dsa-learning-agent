import uuid
from typing import Any

from fastapi import APIRouter

from ..models.schemas import (
    AssessmentQuestion,
    AssessmentStartRequest,
    AssessmentStartResponse,
    AssessmentSubmitRequest,
    AssessmentSubmitResponse,
)

router = APIRouter(prefix="/api/v1/assessment", tags=["Assessment"])


# Статичные вопросы для первичной оценки
ASSESSMENT_QUESTIONS = [
    {
        "question_id": 1,
        "question_text": "Что такое временная сложность алгоритма?",
        "options": [
            "Время работы программы на конкретном компьютере",
            "Оценка количества операций в зависимости от размера входных данных",
            "Размер памяти, занимаемой программой",
            "Скорость работы процессора"
        ],
        "correct_answer": 1,
        "topic": "complexity"
    },
    {
        "question_id": 2,
        "question_text": "Какая структура данных использует принцип LIFO (Last In First Out)?",
        "options": ["Очередь", "Стек", "Список", "Дерево"],
        "correct_answer": 1,
        "topic": "data_structures"
    },
    {
        "question_id": 3,
        "question_text": "Какова временная сложность бинарного поиска?",
        "options": ["O(n)", "O(log n)", "O(n²)", "O(1)"],
        "correct_answer": 1,
        "topic": "algorithms"
    },
    {
        "question_id": 4,
        "question_text": "Что такое рекурсия?",
        "options": [
            "Цикл в программе",
            "Функция, которая вызывает сама себя",
            "Сортировка массива",
            "Поиск элемента в списке"
        ],
        "correct_answer": 1,
        "topic": "basics"
    },
    {
        "question_id": 5,
        "question_text": "Какая из этих сортировок имеет наилучшую среднюю временную сложность?",
        "options": ["Пузырьковая сортировка", "Быстрая сортировка", "Сортировка вставками", "Сортировка выбором"],
        "correct_answer": 1,
        "topic": "sorting"
    }
]


@router.post("/start")
async def start_assessment(request: AssessmentStartRequest) -> AssessmentStartResponse:
    """Начать первичное тестирование"""

    session_id = str(uuid.uuid4())

    questions = [
        AssessmentQuestion(
            question_id=q["question_id"],
            question_text=q["question_text"],
            options=q["options"]
        )
        for q in ASSESSMENT_QUESTIONS
    ]

    # TODO: Сохранить сессию в БД/кеше

    return AssessmentStartResponse(
        test_questions=questions,
        session_id=session_id
    )


@router.post("/submit")
async def submit_assessment(request: AssessmentSubmitRequest) -> AssessmentSubmitResponse:
    """Отправить результаты тестирования"""

    # TODO: Получить вопросы из БД по session_id

    # Подсчет правильных ответов
    correct_count = 0
    topic_scores: dict[str, list[int]] = {}

    for answer in request.answers:
        question_id = answer.get("question_id")
        user_answer = answer.get("answer")

        # Находим вопрос
        question = next((q for q in ASSESSMENT_QUESTIONS if q["question_id"] == question_id), None)
        if not question:
            continue

        is_correct = user_answer == question["correct_answer"]
        if is_correct:
            correct_count += 1

        topic = question["topic"]
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(1 if is_correct else 0)

    # Определение уровня
    percentage = (correct_count / len(ASSESSMENT_QUESTIONS)) * 100

    if percentage >= 80:
        level = "advanced"
    elif percentage >= 50:
        level = "intermediate"
    else:
        level = "beginner"

    # Подсчет по темам
    knowledge_areas = {
        topic: sum(scores) / len(scores) * 100
        for topic, scores in topic_scores.items()
    }

    # Рекомендации
    recommendations = []
    if level == "beginner":
        recommendations.extend(("Рекомендуется начать с основ: временная и пространственная сложность", "Изучите базовые структуры данных: массивы, списки, стеки, очереди"))
    elif level == "intermediate":
        recommendations.extend(("Углубите знания по сложным структурам данных: деревья, графы, хеш-таблицы", "Практикуйтесь в решении алгоритмических задач средней сложности"))
    else:
        recommendations.extend(("Переходите к продвинутым темам: динамическое программирование, жадные алгоритмы", "Решайте сложные задачи и участвуйте в соревнованиях по программированию"))

    return AssessmentSubmitResponse(
        level=level,
        knowledge_areas=knowledge_areas,
        recommendations=recommendations
    )


@router.get("/results/{user_id}")
async def get_assessment_results(user_id: str) -> dict[str, Any]:
    """Получить результаты начальной оценки"""
    # TODO: Получить из БД
    return {
        "initial_level": "intermediate",
        "completed_at": "2025-10-28T10:00:00Z"
    }

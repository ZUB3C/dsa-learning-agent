import json
import uuid

from fastapi import APIRouter

from ..core.database import get_db_connection, get_or_create_user
from ..models.schemas import (
    AssessmentQuestion,
    AssessmentStartRequest,
    AssessmentStartResponse,
    AssessmentSubmitRequest,
    AssessmentSubmitResponse,
    GetAssessmentResultsResponse,
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

    get_or_create_user(request.user_id)

    session_id = str(uuid.uuid4())

    questions = [
        AssessmentQuestion(
            question_id=q["question_id"],
            question_text=q["question_text"],
            options=q["options"]
        )
        for q in ASSESSMENT_QUESTIONS
    ]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO assessment_sessions (session_id, user_id, questions)
               VALUES (?, ?, ?)""",
            (session_id, request.user_id, json.dumps(ASSESSMENT_QUESTIONS))
        )

    return AssessmentStartResponse(
        test_questions=questions,
        session_id=session_id
    )


@router.post("/submit")
async def submit_assessment(request: AssessmentSubmitRequest) -> AssessmentSubmitResponse:
    """Отправить результаты тестирования"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, questions FROM assessment_sessions WHERE session_id = ?",
            (request.session_id,)
        )
        session = cursor.fetchone()

        if not session:
            questions = ASSESSMENT_QUESTIONS
            user_id = "unknown"
        else:
            user_id = session["user_id"]
            questions = json.loads(session["questions"])

    correct_count = 0
    topic_scores: dict[str, list[int]] = {}

    for answer in request.answers:
        question_id = answer.get("question_id")
        user_answer = answer.get("answer")

        question = next((q for q in questions if q["question_id"] == question_id), None)
        if not question:
            continue

        is_correct = user_answer == question["correct_answer"]
        if is_correct:
            correct_count += 1

        topic = question["topic"]
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(1 if is_correct else 0)

    percentage = (correct_count / len(questions)) * 100

    if percentage >= 80:
        level = "advanced"
    elif percentage >= 50:
        level = "intermediate"
    else:
        level = "beginner"

    knowledge_areas = {
        topic: sum(scores) / len(scores) * 100
        for topic, scores in topic_scores.items()
    }

    recommendations = []
    if level == "beginner":
        recommendations.extend([
            "Рекомендуется начать с основ: временная и пространственная сложность",
            "Изучите базовые структуры данных: массивы, списки, стеки, очереди"
        ])
    elif level == "intermediate":
        recommendations.extend([
            "Углубите знания по сложным структурам данных: деревья, графы, хеш-таблицы",
            "Практикуйтесь в решении алгоритмических задач средней сложности"
        ])
    else:
        recommendations.extend([
            "Переходите к продвинутым темам: динамическое программирование, жадные алгоритмы",
            "Решайте сложные задачи и участвуйте в соревнованиях по программированию"
        ])

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO assessments
               (user_id, session_id, level, score, knowledge_areas, recommendations)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                request.session_id,
                level,
                percentage,
                json.dumps(knowledge_areas),
                json.dumps(recommendations)
            )
        )

    return AssessmentSubmitResponse(
        level=level,
        knowledge_areas=knowledge_areas,
        recommendations=recommendations
    )


@router.get("/results/{user_id}")
async def get_assessment_results(user_id: str) -> GetAssessmentResultsResponse:
    """Получить результаты начальной оценки"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT level, score, knowledge_areas, recommendations, completed_at
               FROM assessments
               WHERE user_id = ?
               ORDER BY completed_at DESC
               LIMIT 1""",
            (user_id,)
        )
        result = cursor.fetchone()

        if not result:
            return GetAssessmentResultsResponse(
                message="No assessment found for this user",
                user_id=user_id
            )

        return GetAssessmentResultsResponse(
            user_id=user_id,
            initial_level=result["level"],
            score=result["score"],
            knowledge_areas=json.loads(result["knowledge_areas"]) if result["knowledge_areas"] else {},
            recommendations=json.loads(result["recommendations"]) if result["recommendations"] else [],
            completed_at=result["completed_at"]
        )

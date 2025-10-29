import json
import uuid

from fastapi import APIRouter

from ..core.database import Assessment, AssessmentSession, get_db_session, get_or_create_user
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
            "Скорость работы процессора",
        ],
        "correct_answer": 1,
        "topic": "complexity",
    },
    {
        "question_id": 2,
        "question_text": "Какая структура данных использует принцип LIFO (Last In First Out)?",
        "options": ["Очередь", "Стек", "Список", "Дерево"],
        "correct_answer": 1,
        "topic": "data_structures",
    },
    {
        "question_id": 3,
        "question_text": "Какова временная сложность бинарного поиска?",
        "options": ["O(n)", "O(log n)", "O(n²)", "O(1)"],
        "correct_answer": 1,
        "topic": "algorithms",
    },
    {
        "question_id": 4,
        "question_text": "Что такое рекурсия?",
        "options": [
            "Цикл в программе",
            "Функция, которая вызывает сама себя",
            "Сортировка массива",
            "Поиск элемента в списке",
        ],
        "correct_answer": 1,
        "topic": "basics",
    },
    {
        "question_id": 5,
        "question_text": "Какая из этих сортировок имеет наилучшую среднюю временную сложность?",
        "options": [
            "Пузырьковая сортировка",
            "Быстрая сортировка",
            "Сортировка вставками",
            "Сортировка выбором",
        ],
        "correct_answer": 1,
        "topic": "sorting",
    },
]


@router.post("/start")
async def start_assessment(request: AssessmentStartRequest) -> AssessmentStartResponse:
    """Начать первичное тестирование"""

    get_or_create_user(request.user_id)

    session_id = str(uuid.uuid4())

    questions = [
        AssessmentQuestion(
            question_id=q["question_id"], question_text=q["question_text"], options=q["options"]
        )
        for q in ASSESSMENT_QUESTIONS
    ]

    with get_db_session() as session:
        assessment_session = AssessmentSession(
            session_id=session_id,
            user_id=request.user_id,
            questions=json.dumps(ASSESSMENT_QUESTIONS),
        )
        session.add(assessment_session)

    return AssessmentStartResponse(test_questions=questions, session_id=session_id)


@router.post("/submit")
async def submit_assessment(request: AssessmentSubmitRequest) -> AssessmentSubmitResponse:
    """Отправить результаты тестирования"""

    with get_db_session() as session:
        assessment_session = (
            session.query(AssessmentSession)
            .filter(AssessmentSession.session_id == request.session_id)
            .first()
        )

        if not assessment_session:
            questions = ASSESSMENT_QUESTIONS
            user_id = "unknown"
        else:
            user_id = assessment_session.user_id
            questions = json.loads(assessment_session.questions)

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
        topic: sum(scores) / len(scores) * 100 for topic, scores in topic_scores.items()
    }

    recommendations = []
    if level == "beginner":
        recommendations.extend([
            "Рекомендуется начать с основ: временная и пространственная сложность",
            "Изучите базовые структуры данных: массивы, списки, стеки, очереди",
        ])
    elif level == "intermediate":
        recommendations.extend([
            "Углубите знания по сложным структурам данных: деревья, графы, хеш-таблицы",
            "Практикуйтесь в решении алгоритмических задач средней сложности",
        ])
    else:
        recommendations.extend([
            "Переходите к продвинутым темам: динамическое программирование, жадные алгоритмы",
            "Решайте сложные задачи и участвуйте в соревнованиях по программированию",
        ])

    with get_db_session() as session:
        assessment = Assessment(
            user_id=user_id,
            session_id=request.session_id,
            level=level,
            score=percentage,
            knowledge_areas=json.dumps(knowledge_areas),
            recommendations=json.dumps(recommendations),
        )
        session.add(assessment)

    return AssessmentSubmitResponse(
        level=level, knowledge_areas=knowledge_areas, recommendations=recommendations
    )


@router.get("/results/{user_id}")
async def get_assessment_results(user_id: str) -> GetAssessmentResultsResponse:
    """Получить результаты начальной оценки"""

    with get_db_session() as session:
        result = (
            session.query(Assessment)
            .filter(Assessment.user_id == user_id)
            .order_by(Assessment.completed_at.desc())
            .first()
        )

        if not result:
            return GetAssessmentResultsResponse(
                message="No assessment found for this user", user_id=user_id
            )

        return GetAssessmentResultsResponse(
            user_id=user_id,
            initial_level=result.level,
            score=result.score,
            knowledge_areas=json.loads(result.knowledge_areas) if result.knowledge_areas else {},
            recommendations=json.loads(result.recommendations) if result.recommendations else [],
            completed_at=result.completed_at.isoformat(),
        )

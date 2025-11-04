import json
import uuid
from typing import Any

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


# Полноценный тест из 15 вопросов для первичной оценки
ASSESSMENT_QUESTIONS = [
    # Блок 1: Основы сложности (3 вопроса)
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
        "difficulty": "easy",
    },
    {
        "question_id": 2,
        "question_text": "Какова временная сложность линейного поиска в массиве из n элементов?",
        "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
        "correct_answer": 2,
        "topic": "complexity",
        "difficulty": "easy",
    },
    {
        "question_id": 3,
        "question_text": "Какова временная сложность бинарного поиска в отсортированном массиве?",
        "options": ["O(n)", "O(log n)", "O(n²)", "O(1)"],
        "correct_answer": 1,
        "topic": "complexity",
        "difficulty": "medium",
    },
    # Блок 2: Базовые структуры данных (4 вопроса)
    {
        "question_id": 4,
        "question_text": "Какая структура данных использует принцип LIFO (Last In First Out)?",
        "options": ["Очередь", "Стек", "Список", "Дерево"],
        "correct_answer": 1,
        "topic": "data_structures",
        "difficulty": "easy",
    },
    {
        "question_id": 5,
        "question_text": "Какая структура данных использует принцип FIFO (First In First Out)?",
        "options": ["Стек", "Очередь", "Хеш-таблица", "Граф"],
        "correct_answer": 1,
        "topic": "data_structures",
        "difficulty": "easy",
    },
    {
        "question_id": 6,
        "question_text": "Какова временная сложность доступа к элементу массива по индексу?",
        "options": ["O(n)", "O(log n)", "O(1)", "O(n²)"],
        "correct_answer": 2,
        "topic": "data_structures",
        "difficulty": "easy",
    },
    {
        "question_id": 7,
        "question_text": "Какая структура данных наиболее эффективна для поиска элемента по ключу за O(1) в среднем?",  # noqa: E501
        "options": ["Массив", "Связный список", "Хеш-таблица", "Двоичное дерево поиска"],
        "correct_answer": 2,
        "topic": "data_structures",
        "difficulty": "medium",
    },
    # Блок 3: Рекурсия и алгоритмы (3 вопроса)
    {
        "question_id": 8,
        "question_text": "Что такое рекурсия?",
        "options": [
            "Цикл в программе",
            "Функция, которая вызывает сама себя",
            "Сортировка массива",
            "Поиск элемента в списке",
        ],
        "correct_answer": 1,
        "topic": "recursion",
        "difficulty": "easy",
    },
    {
        "question_id": 9,
        "question_text": "Что является базовым случаем (base case) в рекурсии?",
        "options": [
            "Первый вызов функции",
            "Последний вызов функции",
            "Условие, при котором рекурсия прекращается",
            "Условие, при котором рекурсия начинается",
        ],
        "correct_answer": 2,
        "topic": "recursion",
        "difficulty": "medium",
    },
    {
        "question_id": 10,
        "question_text": "Какой алгоритм использует принцип 'разделяй и властвуй'?",
        "options": [
            "Пузырьковая сортировка",
            "Быстрая сортировка (QuickSort)",
            "Сортировка вставками",
            "Линейный поиск",
        ],
        "correct_answer": 1,
        "topic": "algorithms",
        "difficulty": "medium",
    },
    # Блок 4: Сортировки (3 вопроса)
    {
        "question_id": 11,
        "question_text": "Какая из этих сортировок имеет наилучшую среднюю временную сложность?",
        "options": [
            "Пузырьковая сортировка",
            "Быстрая сортировка",
            "Сортировка вставками",
            "Сортировка выбором",
        ],
        "correct_answer": 1,
        "topic": "sorting",
        "difficulty": "medium",
    },
    {
        "question_id": 12,
        "question_text": "Какова временная сложность пузырьковой сортировки в худшем случае?",
        "options": ["O(n)", "O(n log n)", "O(n²)", "O(log n)"],
        "correct_answer": 2,
        "topic": "sorting",
        "difficulty": "easy",
    },
    {
        "question_id": 13,
        "question_text": "Какая сортировка является стабильной?",
        "options": [
            "Сортировка слиянием (Merge Sort)",
            "Быстрая сортировка (QuickSort)",
            "Пирамидальная сортировка (HeapSort)",
            "Все вышеперечисленные",
        ],
        "correct_answer": 0,
        "topic": "sorting",
        "difficulty": "hard",
    },
    # Блок 5: Деревья и графы (2 вопроса)
    {
        "question_id": 14,
        "question_text": "Что такое двоичное дерево поиска (BST)?",
        "options": [
            "Дерево, где у каждого узла максимум два потомка",
            "Дерево, где левое поддерево содержит меньшие значения, а правое - большие",
            "Дерево, где все листья на одном уровне",
            "Дерево с фиксированной высотой",
        ],
        "correct_answer": 1,
        "topic": "trees",
        "difficulty": "medium",
    },
    {
        "question_id": 15,
        "question_text": "Какой обход дерева посещает узлы в порядке: левое поддерево → корень → правое поддерево?",  # noqa: E501
        "options": [
            "Прямой обход (preorder)",
            "Центрированный обход (inorder)",
            "Обратный обход (postorder)",
            "Обход в ширину (level order)",
        ],
        "correct_answer": 1,
        "topic": "trees",
        "difficulty": "medium",
    },
]


@router.post("/start")
async def start_assessment(request: AssessmentStartRequest) -> AssessmentStartResponse:
    """Начать первичное тестирование."""
    get_or_create_user(request.user_id)

    session_id: str = str(uuid.uuid4())

    questions: list[AssessmentQuestion] = [
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
    """Отправить результаты тестирования."""
    with get_db_session() as session:
        assessment_session = (
            session.query(AssessmentSession)
            .filter(AssessmentSession.session_id == request.session_id)
            .first()
        )

        if not assessment_session:
            questions: list[dict[str, Any]] = ASSESSMENT_QUESTIONS
            user_id: str = "unknown"
        else:
            user_id = assessment_session.user_id
            questions = json.loads(assessment_session.questions)

    correct_count: int = 0

    for answer in request.answers:
        question_id: int = answer.get("question_id")
        user_answer: int = answer.get("answer")

        question: dict[str, Any] | None = next(
            (q for q in questions if q["question_id"] == question_id), None
        )
        if not question:
            continue

        is_correct: bool = user_answer == question["correct_answer"]
        if is_correct:
            correct_count += 1

        question["topic"]

    percentage: float = (correct_count / len(questions)) * 100

    # Более детальная градация уровней
    if percentage >= 85:
        level: str = "advanced"
    elif percentage >= 60:
        level = "intermediate"
    else:
        level = "beginner"

    knowledge_areas = {}

    recommendations: list[str] = []
    if level == "beginner":
        recommendations.extend([
            "Рекомендуется начать с основ: временная и пространственная сложность",
            "Изучите базовые структуры данных: массивы, списки, стеки, очереди",
            "Практикуйтесь в написании простых алгоритмов",
            "Освойте базовые концепции рекурсии",
        ])
    elif level == "intermediate":
        recommendations.extend([
            "Углубите знания по сложным структурам данных: деревья, графы, хеш-таблицы",
            "Изучите продвинутые алгоритмы сортировки и поиска",
            "Практикуйтесь в решении алгоритмических задач средней сложности",
            "Начните изучение динамического программирования",
        ])
    else:
        recommendations.extend([
            "Переходите к продвинутым темам: динамическое программирование, жадные алгоритмы",
            "Изучайте сложные структуры данных: B-деревья, префиксные деревья, системы непересекающихся множеств",  # noqa: E501
            "Решайте сложные задачи и участвуйте в соревнованиях по программированию",
            "Углубите понимание анализа алгоритмов и оптимизации",
        ])

    with get_db_session() as session:
        assessment = Assessment(
            user_id=user_id,
            session_id=request.session_id,
            level=level,
            knowledge_areas=json.dumps(knowledge_areas),
            recommendations=json.dumps(recommendations),
        )
        session.add(assessment)

    return AssessmentSubmitResponse(
        knowledge_areas=knowledge_areas, recommendations=recommendations
    )


@router.get("/results/{user_id}")
async def get_assessment_results(user_id: str) -> GetAssessmentResultsResponse:
    """Получить результаты начальной оценки."""
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
            knowledge_areas=json.loads(result.knowledge_areas) if result.knowledge_areas else {},
            recommendations=json.loads(result.recommendations) if result.recommendations else [],
            completed_at=result.completed_at.isoformat(),
        )

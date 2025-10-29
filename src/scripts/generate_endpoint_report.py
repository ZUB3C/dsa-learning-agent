"""
Скрипт для тестирования всех API эндпоинтов
Генерирует Markdown документацию с описанием, вводом и выводом каждого эндпоинта
"""

import asyncio
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.routers import assessment, health, llm_router, materials, support, tests, verification


def extract_docstring(func) -> str:
    """Извлечь докстринг из функции"""
    return inspect.getdoc(func) or "Описание отсутствует"


def format_json(data: Any) -> str:
    """Форматировать JSON для вывода"""
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except:
        return str(data)


async def test_health_endpoints():
    """Тестирование health endpoints"""
    results = []

    # GET /health/
    try:
        result = health.health_check()
        results.append({
            "endpoint": "GET /health/",
            "description": extract_docstring(health.health_check),
            "input": "Без параметров",
            "output": format_json(result.model_dump()),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "GET /health/",
            "description": extract_docstring(health.health_check),
            "input": "Без параметров",
            "output": f"Error: {e!s}",
            "status": "error",
        })

    return results


async def test_assessment_endpoints():
    """Тестирование assessment endpoints"""
    from src.models.schemas import AssessmentStartRequest, AssessmentSubmitRequest

    results = []

    # POST /api/v1/assessment/start
    try:
        request = AssessmentStartRequest(user_id="test_user_123")
        result = await assessment.start_assessment(request)
        session_id = result.session_id

        results.append({
            "endpoint": "POST /api/v1/assessment/start",
            "description": extract_docstring(assessment.start_assessment),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "test_questions_count": len(result.test_questions),
                "session_id": result.session_id,
                "first_question": result.test_questions[0].model_dump()
                if result.test_questions
                else None,
            }),
            "status": "success",
        })
    except Exception as e:
        session_id = "test_session_id"
        results.append({
            "endpoint": "POST /api/v1/assessment/start",
            "description": extract_docstring(assessment.start_assessment),
            "input": format_json({"user_id": "test_user_123"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/assessment/submit
    try:
        submit_request = AssessmentSubmitRequest(
            session_id=session_id,
            answers=[
                {"question_id": 1, "answer": 1},
                {"question_id": 2, "answer": 2},
                {"question_id": 3, "answer": 1},
            ],
        )
        result = await assessment.submit_assessment(submit_request)

        results.append({
            "endpoint": "POST /api/v1/assessment/submit",
            "description": extract_docstring(assessment.submit_assessment),
            "input": format_json(submit_request.model_dump()),
            "output": format_json(result.model_dump()),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/assessment/submit",
            "description": extract_docstring(assessment.submit_assessment),
            "input": format_json({"session_id": session_id, "answers": []}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # GET /api/v1/assessment/results/{user_id}
    try:
        result = await assessment.get_assessment_results("test_user_123")

        results.append({
            "endpoint": "GET /api/v1/assessment/results/{user_id}",
            "description": extract_docstring(assessment.get_assessment_results),
            "input": format_json({"user_id": "test_user_123"}),
            "output": format_json(result.model_dump()),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "GET /api/v1/assessment/results/{user_id}",
            "description": extract_docstring(assessment.get_assessment_results),
            "input": format_json({"user_id": "test_user_123"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    return results


async def test_materials_endpoints():
    """Тестирование materials endpoints"""
    from src.models.schemas import (
        AddCustomTopicRequest,
        AskQuestionRequest,
        GenerateMaterialRequest,
        GetMaterialsRequest,
        SearchMaterialsRequest,
    )

    results = []

    # POST /api/v1/materials/get-materials
    try:
        request = GetMaterialsRequest(
            topic="Сортировка пузырьком", user_level="beginner", language="ru"
        )
        result = await materials.get_materials(request)

        results.append({
            "endpoint": "POST /api/v1/materials/get-materials",
            "description": extract_docstring(materials.get_materials),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "content_preview": result.content[:200] + "..."
                if len(result.content) > 200
                else result.content,
                "sources": result.sources,
                "adapted_for_level": result.adapted_for_level,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/materials/get-materials",
            "description": extract_docstring(materials.get_materials),
            "input": format_json({"topic": "Сортировка пузырьком", "user_level": "beginner"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/materials/ask-question
    try:
        request = AskQuestionRequest(
            question="Какова временная сложность пузырьковой сортировки?",
            context_topic="Сортировки",
            user_level="beginner",
            language="ru",
        )
        result = await materials.ask_question(request)

        results.append({
            "endpoint": "POST /api/v1/materials/ask-question",
            "description": extract_docstring(materials.ask_question),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "answer_preview": result.answer[:200] + "..."
                if len(result.answer) > 200
                else result.answer,
                "related_concepts": result.related_concepts,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/materials/ask-question",
            "description": extract_docstring(materials.ask_question),
            "input": format_json({"question": "Тестовый вопрос", "context_topic": "Тест"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/materials/generate-material
    try:
        request = GenerateMaterialRequest(
            topic="Бинарный поиск", format="summary", length="short", language="ru"
        )
        result = await materials.generate_material(request)

        results.append({
            "endpoint": "POST /api/v1/materials/generate-material",
            "description": extract_docstring(materials.generate_material),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "material_preview": result.material[:200] + "..."
                if len(result.material) > 200
                else result.material,
                "format": result.format,
                "word_count": result.word_count,
                "model_used": result.model_used,
                "topic_id": result.topic_id,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/materials/generate-material",
            "description": extract_docstring(materials.generate_material),
            "input": format_json({"topic": "Бинарный поиск", "format": "summary"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/materials/add-custom-topic
    try:
        request = AddCustomTopicRequest(
            topic_name="Моя кастомная тема",
            user_id="test_user_123",
            content="Тестовое содержание темы",
        )
        result = await materials.add_custom_topic(request)

        results.append({
            "endpoint": "POST /api/v1/materials/add-custom-topic",
            "description": extract_docstring(materials.add_custom_topic),
            "input": format_json(request.model_dump()),
            "output": format_json(result.model_dump()),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/materials/add-custom-topic",
            "description": extract_docstring(materials.add_custom_topic),
            "input": format_json({"topic_name": "Тест", "user_id": "test_user"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # GET /api/v1/materials/topics
    try:
        result = await materials.get_topics()

        results.append({
            "endpoint": "GET /api/v1/materials/topics",
            "description": extract_docstring(materials.get_topics),
            "input": "Без параметров",
            "output": format_json({
                "predefined_topics_count": len(result.predefined_topics),
                "custom_topics_count": len(result.custom_topics),
                "predefined_topics_preview": result.predefined_topics[:3],
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "GET /api/v1/materials/topics",
            "description": extract_docstring(materials.get_topics),
            "input": "Без параметров",
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/materials/search
    try:
        request = SearchMaterialsRequest(
            query="алгоритм сортировки", filters={"level": "beginner"}
        )
        result = await materials.search_materials(request)

        results.append({
            "endpoint": "POST /api/v1/materials/search",
            "description": extract_docstring(materials.search_materials),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "results_count": len(result.results),
                "relevance_scores": result.relevance_scores[:5],
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/materials/search",
            "description": extract_docstring(materials.search_materials),
            "input": format_json({"query": "тест"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    return results


async def test_tests_endpoints():
    """Тестирование tests endpoints"""
    from src.models.schemas import GenerateTaskRequest, GenerateTestRequest, SubmitTestRequest

    results = []

    # POST /api/v1/tests/generate
    try:
        request = GenerateTestRequest(
            topic="Структуры данных", difficulty="easy", question_count=3, language="ru"
        )
        result = await tests.generate_test(request)
        test_id = result.test_id

        results.append({
            "endpoint": "POST /api/v1/tests/generate",
            "description": extract_docstring(tests.generate_test),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "test_id": result.test_id,
                "questions_count": len(result.questions),
                "expected_duration": result.expected_duration,
            }),
            "status": "success",
        })
    except Exception as e:
        test_id = "test_test_id"
        results.append({
            "endpoint": "POST /api/v1/tests/generate",
            "description": extract_docstring(tests.generate_test),
            "input": format_json({"topic": "Структуры данных", "difficulty": "easy"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/tests/generate-task
    try:
        request = GenerateTaskRequest(
            topic="Алгоритмы поиска", difficulty="medium", task_type="coding", language="ru"
        )
        result = await tests.generate_task(request)

        results.append({
            "endpoint": "POST /api/v1/tests/generate-task",
            "description": extract_docstring(tests.generate_task),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "task_id": result.task.task_id,
                "description_preview": result.task.description[:100] + "..."
                if len(result.task.description) > 100
                else result.task.description,
                "hints_count": len(result.solution_hints),
                "model_used": result.model_used,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/tests/generate-task",
            "description": extract_docstring(tests.generate_task),
            "input": format_json({"topic": "Алгоритмы", "difficulty": "medium"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/tests/submit-for-verification
    try:
        request = SubmitTestRequest(
            test_id=test_id,
            user_id="test_user_123",
            answers=[{"question_id": 1, "answer": "Test answer"}],
        )
        result = await tests.submit_test_for_verification(request)

        results.append({
            "endpoint": "POST /api/v1/tests/submit-for-verification",
            "description": extract_docstring(tests.submit_test_for_verification),
            "input": format_json(request.model_dump()),
            "output": format_json(result.model_dump()),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/tests/submit-for-verification",
            "description": extract_docstring(tests.submit_test_for_verification),
            "input": format_json({"test_id": "test", "user_id": "test"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # GET /api/v1/tests/{test_id}
    try:
        result = await tests.get_test(test_id)

        results.append({
            "endpoint": f"GET /api/v1/tests/{test_id}",
            "description": extract_docstring(tests.get_test),
            "input": format_json({"test_id": test_id}),
            "output": format_json({
                "test_id": result.test.get("test_id"),
                "topic": result.test.get("topic"),
                "difficulty": result.test.get("difficulty"),
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": f"GET /api/v1/tests/{test_id}",
            "description": extract_docstring(tests.get_test),
            "input": format_json({"test_id": test_id}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # GET /api/v1/tests/user/{user_id}/completed
    try:
        result = await tests.get_completed_tests("test_user_123")

        results.append({
            "endpoint": "GET /api/v1/tests/user/{user_id}/completed",
            "description": extract_docstring(tests.get_completed_tests),
            "input": format_json({"user_id": "test_user_123"}),
            "output": format_json({
                "completed_tests_count": len(result.completed_tests),
                "statistics": result.statistics,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "GET /api/v1/tests/user/{user_id}/completed",
            "description": extract_docstring(tests.get_completed_tests),
            "input": format_json({"user_id": "test_user_123"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    return results


async def test_verification_endpoints():
    """Тестирование verification endpoints"""
    from src.models.schemas import TestVerificationRequest

    results = []

    # POST /api/v1/verification/check-test
    try:
        request = TestVerificationRequest(
            test_id="test_test_id",
            user_answer="Пузырьковая сортировка имеет сложность O(n²)",
            language="ru",
            question="Какова временная сложность пузырьковой сортировки?",
            expected_answer="O(n²)",
        )
        result = await verification.check_test(request)

        results.append({
            "endpoint": "POST /api/v1/verification/check-test",
            "description": extract_docstring(verification.check_test),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "is_correct": result.is_correct,
                "score": result.score,
                "feedback_preview": result.feedback[:150] + "..."
                if len(result.feedback) > 150
                else result.feedback,
                "verification_details": result.verification_details,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/verification/check-test",
            "description": extract_docstring(verification.check_test),
            "input": format_json({"test_id": "test", "user_answer": "test"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # GET /api/v1/verification/history/{user_id}
    try:
        result = await verification.get_verification_history("test_user_123")

        results.append({
            "endpoint": "GET /api/v1/verification/history/{user_id}",
            "description": extract_docstring(verification.get_verification_history),
            "input": format_json({"user_id": "test_user_123"}),
            "output": format_json({
                "tests_count": len(result.tests),
                "average_score": result.average_score,
                "total_tests": result.total_tests,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "GET /api/v1/verification/history/{user_id}",
            "description": extract_docstring(verification.get_verification_history),
            "input": format_json({"user_id": "test_user_123"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    return results


async def test_llm_router_endpoints():
    """Тестирование LLM router endpoints"""
    from src.models.schemas import LLMRouterRequest, RouteRequestRequest

    results = []

    # POST /api/v1/llm-router/select-and-generate
    try:
        request = LLMRouterRequest(
            request_type="material",
            content="Объясни быструю сортировку",
            language="ru",
            parameters={},
        )
        result = await llm_router.select_and_generate(request)

        results.append({
            "endpoint": "POST /api/v1/llm-router/select-and-generate",
            "description": extract_docstring(llm_router.select_and_generate),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "content_preview": result.generated_content[:200] + "..."
                if len(result.generated_content) > 200
                else result.generated_content,
                "model_used": result.model_used,
                "metadata": result.metadata,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/llm-router/select-and-generate",
            "description": extract_docstring(llm_router.select_and_generate),
            "input": format_json({"request_type": "material", "content": "test"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # GET /api/v1/llm-router/available-models
    try:
        result = await llm_router.get_available_models()

        results.append({
            "endpoint": "GET /api/v1/llm-router/available-models",
            "description": extract_docstring(llm_router.get_available_models),
            "input": "Без параметров",
            "output": format_json({
                "models": [m.model_dump() for m in result.models],
                "capabilities": result.capabilities,
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "GET /api/v1/llm-router/available-models",
            "description": extract_docstring(llm_router.get_available_models),
            "input": "Без параметров",
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/llm-router/route-request
    try:
        request = RouteRequestRequest(
            request_type="test",
            content="Создай тест по алгоритмам",
            context={"difficulty": "medium"},
            language="ru",
        )
        result = await llm_router.route_request(request)

        results.append({
            "endpoint": "POST /api/v1/llm-router/route-request",
            "description": extract_docstring(llm_router.route_request),
            "input": format_json(request.model_dump()),
            "output": format_json(result.model_dump()),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/llm-router/route-request",
            "description": extract_docstring(llm_router.route_request),
            "input": format_json({"request_type": "test", "content": "test"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    return results


async def test_support_endpoints():
    """Тестирование support endpoints"""
    from src.models.schemas import SubmitFeedbackRequest, SupportRequest

    results = []

    # POST /api/v1/support/get-support
    try:
        request = SupportRequest(
            user_id="test_user_123",
            message="Я чувствую себя перегруженным изучением алгоритмов",
            emotional_state="stressed",
            language="ru",
        )
        result = await support.get_support(request)

        results.append({
            "endpoint": "POST /api/v1/support/get-support",
            "description": extract_docstring(support.get_support),
            "input": format_json(request.model_dump()),
            "output": format_json({
                "support_message_preview": result.support_message[:200] + "..."
                if len(result.support_message) > 200
                else result.support_message,
                "recommendations_count": len(result.recommendations),
                "resources_count": len(result.resources),
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/support/get-support",
            "description": extract_docstring(support.get_support),
            "input": format_json({"user_id": "test", "message": "test"}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # GET /api/v1/support/resources
    try:
        result = await support.get_support_resources()

        results.append({
            "endpoint": "GET /api/v1/support/resources",
            "description": extract_docstring(support.get_support_resources),
            "input": "Без параметров",
            "output": format_json({
                "articles_count": len(result.articles),
                "exercises_count": len(result.exercises),
                "tips_count": len(result.tips),
            }),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "GET /api/v1/support/resources",
            "description": extract_docstring(support.get_support_resources),
            "input": "Без параметров",
            "output": f"Error: {e!s}",
            "status": "error",
        })

    # POST /api/v1/support/feedback
    try:
        request = SubmitFeedbackRequest(
            session_id="test_session_id", helpful=True, comments="Очень помогло!"
        )
        result = await support.submit_feedback(request)

        results.append({
            "endpoint": "POST /api/v1/support/feedback",
            "description": extract_docstring(support.submit_feedback),
            "input": format_json(request.model_dump()),
            "output": format_json(result.model_dump()),
            "status": "success",
        })
    except Exception as e:
        results.append({
            "endpoint": "POST /api/v1/support/feedback",
            "description": extract_docstring(support.submit_feedback),
            "input": format_json({"session_id": "test", "helpful": True}),
            "output": f"Error: {e!s}",
            "status": "error",
        })

    return results


def generate_markdown(all_results: list) -> str:
    """Генерация Markdown документации"""

    markdown = f"""# API Endpoints Documentation

Автоматически сгенерированная документация для всех API эндпоинтов.

**Дата генерации:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""

    # Группируем результаты по модулям
    modules = {
        "Health": [],
        "Assessment": [],
        "Materials": [],
        "Tests": [],
        "Verification": [],
        "LLM Router": [],
        "Support": [],
    }

    for result in all_results:
        endpoint = result["endpoint"]
        if "/health" in endpoint:
            modules["Health"].append(result)
        elif "/assessment" in endpoint:
            modules["Assessment"].append(result)
        elif "/materials" in endpoint:
            modules["Materials"].append(result)
        elif "/tests" in endpoint:
            modules["Tests"].append(result)
        elif "/verification" in endpoint:
            modules["Verification"].append(result)
        elif "/llm-router" in endpoint:
            modules["LLM Router"].append(result)
        elif "/support" in endpoint:
            modules["Support"].append(result)

    # Генерируем содержание
    markdown += "## Содержание\n\n"
    for module_name in modules:
        if modules[module_name]:
            markdown += f"- [{module_name}](#{module_name.lower().replace(' ', '-')})\n"
    markdown += "\n---\n\n"

    # Генерируем документацию для каждого модуля
    for module_name, results in modules.items():
        if not results:
            continue

        markdown += f"## {module_name}\n\n"

        for result in results:
            status_emoji = "✅" if result["status"] == "success" else "❌"

            markdown += f"### {status_emoji} `{result['endpoint']}`\n\n"
            markdown += f"**Описание:** {result['description']}\n\n"

            markdown += "**Входные данные:**\n\n"
            markdown += "```"
            markdown += result["input"]
            markdown += "\n```\n\n"

            markdown += "**Выходные данные:**\n\n"
            markdown += "```"
            markdown += result["output"]
            markdown += "\n```\n\n"

            markdown += "---\n\n"

    # Добавляем статистику
    total = len(all_results)
    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = total - successful

    markdown += f"""## Статистика

- **Всего эндпоинтов:** {total}
- **Успешных тестов:** {successful}
- **Неудачных тестов:** {failed}
- **Процент успеха:** {(successful / total * 100):.1f}%

"""

    return markdown


async def main() -> None:
    """Основная функция"""
    print("🚀 Начало тестирования API эндпоинтов...")
    print()

    all_results = []

    # Тестируем каждый модуль
    modules = [
        ("Health", test_health_endpoints),
        ("Assessment", test_assessment_endpoints),
        ("Materials", test_materials_endpoints),
        ("Tests", test_tests_endpoints),
        ("Verification", test_verification_endpoints),
        ("LLM Router", test_llm_router_endpoints),
        ("Support", test_support_endpoints),
    ]

    for module_name, test_func in modules:
        print(f"📦 Тестирование модуля: {module_name}")
        try:
            results = await test_func()
            all_results.extend(results)
            print(f"   ✅ Протестировано эндпоинтов: {len(results)}")
        except Exception as e:
            print(f"   ❌ Ошибка при тестировании: {e!s}")
        print()

    # Генерируем Markdown
    print("📝 Генерация Markdown документации...")
    markdown_content = generate_markdown(all_results)

    # Сохраняем в файл
    output_file = Path(__file__).parent.parent.parent / "api_documentation.md"
    output_file.write_text(markdown_content, encoding="utf-8")

    print(f"✅ Документация сохранена: {output_file}")
    print()
    print(f"📊 Всего протестировано эндпоинтов: {len(all_results)}")
    successful = sum(1 for r in all_results if r["status"] == "success")
    print(f"✅ Успешных: {successful}")
    print(f"❌ С ошибками: {len(all_results) - successful}")


if __name__ == "__main__":
    asyncio.run(main())

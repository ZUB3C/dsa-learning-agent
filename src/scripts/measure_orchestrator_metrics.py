from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..config import settings
from ..core.database import SupportSession, get_db_session
from ..models.orchestrator_schemas import ResolveRequest


@dataclass
class MetricsResult:
    classification_accuracy: float
    success_rate: float
    partial_rate: float
    error_rate: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    support_success_rate: float
    total_requests: int
    correct_classifications: int


def load_test_cases() -> list[dict[str, Any]]:
    """Загружает тестовые примеры из JSON."""
    test_data_path = Path(__file__).parent / "test_orchestrator_data.json"
    with Path(test_data_path).open(encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


async def measure_classification_accuracy(
    test_cases: list[dict[str, Any]],
    base_url: str = "http://127.0.0.1:8001",
) -> tuple[int, int, list[float]]:
    """
    Измеряет точность классификации оркестратора.
    Возвращает: (количество правильных, всего, список времён ответа в мс).
    """
    correct = 0
    response_times: list[float] = []

    async with httpx.AsyncClient(timeout=settings.timeout_s) as client:
        for test_case in test_cases:
            request_payload = ResolveRequest(
                message=test_case["message"],
                user_id=test_case["user_id"],
                user_level=test_case["user_level"],
            ).model_dump()

            start = time.time()
            try:
                response = await client.post(
                    f"{base_url}/api/v1/orchestrator/resolve",
                    json=request_payload,
                )
                elapsed_ms = (time.time() - start) * 1000
                response_times.append(elapsed_ms)

                if response.status_code == 200:
                    data = response.json()
                    predicted_task_type = data.get("task_type")
                    expected_task_type = test_case["expected_task_type"]

                    if predicted_task_type == expected_task_type:
                        correct += 1
                        print(
                            f"✓ Test {test_case['id']}: {expected_task_type} "
                            f"({elapsed_ms:.1f}ms)"
                        )
                    else:
                        print(
                            f"✗ Test {test_case['id']}: expected "
                            f"{expected_task_type}, got {predicted_task_type}"
                        )
                else:
                    print(f"✗ Test {test_case['id']}: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ Test {test_case['id']}: {e}")

    return correct, len(test_cases), response_times


async def measure_response_statuses(
    test_cases: list[dict[str, Any]],
    base_url: str = "http://127.0.0.1:8001",
) -> dict[str, int]:
    """
    Измеряет распределение статусов ответа (success, partial, error).
    Возвращает словарь с подсчётом.
    """
    statuses: dict[str, int] = {"success": 0, "partial": 0, "error": 0}

    async with httpx.AsyncClient(timeout=settings.timeout_s) as client:
        for test_case in test_cases:
            request_payload = ResolveRequest(
                message=test_case["message"],
                user_id=test_case["user_id"],
                user_level=test_case["user_level"],
            ).model_dump()

            try:
                response = await client.post(
                    f"{base_url}/api/v1/orchestrator/resolve",
                    json=request_payload,
                )
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    if status in statuses:
                        statuses[status] += 1
            except Exception:
                statuses["error"] += 1

    return statuses


def measure_support_session_success() -> float:
    """
    Измеряет долю успешно сохранённых сессий поддержки в БД.
    Возвращает процент сессий с заполненным полем content и успешным статусом.
    """
    with get_db_session() as session:
        total_sessions = session.query(SupportSession).count()
        if total_sessions == 0:
            return 0.0

        successful_sessions = (
            session.query(SupportSession)
            .filter(SupportSession.user_message.isnot(None))
            .filter(SupportSession.response_content.isnot(None))
            .count()
        )

        return (successful_sessions / total_sessions) * 100


def calculate_percentile_95(values: list[float]) -> float:
    """Вычисляет 95-й перцентиль из списка значений."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * 0.95)
    return sorted_values[min(index, len(sorted_values) - 1)]


async def run_all_metrics(
    base_url: str = "http://127.0.0.1:8001",
) -> MetricsResult:
    """
    Запускает все измерения метрик и возвращает агрегированный результат.
    """
    print("=" * 80)
    print("Orchestrator Metrics Measurement")
    print("=" * 80)

    test_cases = load_test_cases()
    print(f"\nLoaded {len(test_cases)} test cases.\n")

    # Точность классификации
    print("1. Measuring classification accuracy...")
    correct, total, response_times = await measure_classification_accuracy(
        test_cases, base_url
    )
    accuracy = (correct / total) * 100 if total > 0 else 0

    # Статусы ответов
    print("\n2. Measuring response statuses...")
    statuses = await measure_response_statuses(test_cases, base_url)

    success_count = statuses.get("success", 0)
    partial_count = statuses.get("partial", 0)
    error_count = statuses.get("error", 0)

    success_rate = (success_count / total) * 100 if total > 0 else 0
    partial_rate = (partial_count / total) * 100 if total > 0 else 0
    error_rate = (error_count / total) * 100 if total > 0 else 0

    # Времена ответа
    avg_time = sum(response_times) / len(response_times) if response_times else 0
    p95_time = calculate_percentile_95(response_times)

    # Успех сессий поддержки
    print("\n3. Measuring support session success rate...")
    support_success = measure_support_session_success()

    return MetricsResult(
        classification_accuracy=accuracy,
        success_rate=success_rate,
        partial_rate=partial_rate,
        error_rate=error_rate,
        avg_response_time_ms=avg_time,
        p95_response_time_ms=p95_time,
        support_success_rate=support_success,
        total_requests=total,
        correct_classifications=correct,
    )


def print_metrics_report(result: MetricsResult) -> None:
    """Выводит красивый отчёт по метрикам."""
    print("\n" + "=" * 80)
    print("METRICS REPORT")
    print("=" * 80)

    print("\n📊 Classification Accuracy:")
    print(f"   Correct: {result.correct_classifications}/{result.total_requests}")
    print(f"   Accuracy: {result.classification_accuracy:.2f}%")

    print("\n📈 Response Status Distribution:")
    print(f"   Success: {result.success_rate:.2f}%")
    print(f"   Partial: {result.partial_rate:.2f}%")
    print(f"   Error: {result.error_rate:.2f}%")

    print("\n⏱️  Response Time Metrics:")
    print(f"   Average: {result.avg_response_time_ms:.2f} ms")
    print(f"   95th Percentile: {result.p95_response_time_ms:.2f} ms")

    print("\n🤝 Support Agent Metrics:")
    print(f"   Session Success Rate: {result.support_success_rate:.2f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import asyncio

    metrics = asyncio.run(run_all_metrics())
    print_metrics_report(metrics)

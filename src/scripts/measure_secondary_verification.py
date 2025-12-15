import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.agents.registry import load_agent

# =======================
# Models
# =======================


class Question(BaseModel):
    question_id: int
    difficulty: str
    question_text: str
    expected_answer: str
    user_answer: str
    key_points: list[str]
    is_correct: bool


class Topic(BaseModel):
    topic_id: str
    topic_name: str
    questions: list[Question]


class TestCollection(BaseModel):
    creation_date: str
    total_questions: int
    topics_count: int
    topics: list[Topic]


class PrimaryEvaluation(BaseModel):
    verdict: bool


class SecondaryEvaluation(BaseModel):
    verdict: bool
    agree_with_primary: bool
    feedback: str


class TestVerification(BaseModel):
    question_id: int
    topic: str
    difficulty: str
    question_text: str
    user_answer: str
    expected_answer: str
    ground_truth: bool
    primary_evaluation: PrimaryEvaluation
    secondary_evaluation: SecondaryEvaluation
    timestamp: str


class VerificationMetrics(BaseModel):
    total_verifications: int
    agreement_count: int
    disagreement_count: int
    agreement_rate: float
    primary_accuracy: float
    secondary_accuracy: float
    improvement_rate: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    false_positive_rate: float


class EffectivenessReport(BaseModel):
    report_date: str
    overall_metrics: VerificationMetrics
    verifications: list[TestVerification]


# =======================
# Verification logic
# =======================


async def verify_answer(
    question: Question,
) -> tuple[PrimaryEvaluation, SecondaryEvaluation]:
    """Проверяет ответ на вопрос с помощью первичной и вторичной оценки."""
    primary_agent = load_agent("verification")
    primary_raw = await primary_agent.ainvoke({
        "question": question.question_text,
        "expected_answer": question.expected_answer,
        "user_answer": question.user_answer,
    })

    try:
        primary_eval = PrimaryEvaluation(**json.loads(primary_raw))
    except Exception:
        primary_eval = PrimaryEvaluation(verdict=False)

    secondary_agent = load_agent("verification-secondary")
    secondary_raw = await secondary_agent.ainvoke({
        "primary_verdict": primary_eval.verdict,
        "question": question.question_text,
        "user_answer": question.user_answer,
        "expected_answer": question.expected_answer,
    })

    try:
        secondary_eval = SecondaryEvaluation(**json.loads(secondary_raw))
    except Exception:
        secondary_eval = SecondaryEvaluation(
            verdict=primary_eval.verdict,
            agree_with_primary=True,
            feedback="Ошибка парсинга ответа судьи",
        )

    return primary_eval, secondary_eval


async def process_verifications(
    test_collection: TestCollection,
) -> list[TestVerification]:
    """Обрабатывает все вопросы и создает список верификаций."""
    results: list[TestVerification] = []

    for topic in test_collection.topics:
        for question in topic.questions:
            primary, secondary = await verify_answer(question)
            results.append(
                TestVerification(
                    question_id=question.question_id,
                    topic=topic.topic_name,
                    difficulty=question.difficulty,
                    question_text=question.question_text,
                    user_answer=question.user_answer,
                    expected_answer=question.expected_answer,
                    ground_truth=question.is_correct,
                    primary_evaluation=primary,
                    secondary_evaluation=secondary,
                    timestamp=datetime.now().isoformat(),
                )
            )

    return results


# =======================
# Metrics
# =======================


def calculate_metrics(
    verifications: list[TestVerification],
) -> VerificationMetrics:
    """Рассчитывает метрики эффективности верификации."""
    total = len(verifications)
    agreement_count = sum(v.secondary_evaluation.agree_with_primary for v in verifications)
    disagreement_count = total - agreement_count

    primary_correct = sum(v.primary_evaluation.verdict == v.ground_truth for v in verifications)
    secondary_correct = sum(
        v.secondary_evaluation.verdict == v.ground_truth for v in verifications
    )

    tp = sum(v.ground_truth and v.secondary_evaluation.verdict for v in verifications)
    tn = sum(not v.ground_truth and not v.secondary_evaluation.verdict for v in verifications)
    fp = sum(not v.ground_truth and v.secondary_evaluation.verdict for v in verifications)
    fn = sum(v.ground_truth and not v.secondary_evaluation.verdict for v in verifications)

    return VerificationMetrics(
        total_verifications=total,
        agreement_count=agreement_count,
        disagreement_count=disagreement_count,
        agreement_rate=agreement_count / total * 100,
        primary_accuracy=primary_correct / total * 100,
        secondary_accuracy=secondary_correct / total * 100,
        improvement_rate=(secondary_correct - primary_correct) / total * 100,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        false_positive_rate=(fp / (fp + tn) * 100) if (fp + tn) else 0.0,
    )


# =======================
# Report
# =======================


def generate_markdown_report(report: EffectivenessReport) -> str:
    """Генерирует markdown отчет об эффективности."""
    m = report.overall_metrics
    md_lines: list[str] = []

    # ===== Заголовок =====
    md_lines.append("# Отчёт об эффективности вторичной проверки\n")
    md_lines.append(f"Дата: {report.report_date}\n")

    # ===== Общая статистика =====
    md_lines.append("## Общая статистика\n")
    md_lines.append(f"- **Всего проверок:** {m.total_verifications}")
    md_lines.append(f"- **Согласие проверок:** {m.agreement_count} ({m.agreement_rate:.1f}%)")
    md_lines.append(f"- **Расхождения:** {m.disagreement_count} ({100 - m.agreement_rate:.1f}%)\n")

    # ===== Метрики точности =====
    md_lines.append("### 🎯 Точность относительно эталона\n")
    md_lines.append(f"- **Точность первичной проверки:** {m.primary_accuracy:.1f}%")
    md_lines.append(f"- **Точность вторичной проверки (Judge):** {m.secondary_accuracy:.1f}%")
    md_lines.append(f"- **Улучшение от вторичной проверки:** **{m.improvement_rate:+.1f}%**\n")

    # ===== Confusion Matrix =====
    md_lines.append("### 🧮 Confusion Matrix (Judge)\n")
    md_lines.append(f"- **True Positive (TP):** {m.true_positive}")
    md_lines.append(f"- **True Negative (TN):** {m.true_negative}")
    md_lines.append(f"- **False Positive (FP):** {m.false_positive}")
    md_lines.append(f"- **False Negative (FN):** {m.false_negative}")
    md_lines.append(f"- **False Positive Rate:** {m.false_positive_rate:.1f}%\n")

    # ===== Выводы =====
    if m.improvement_rate >= 10:
        effectiveness = (
            "✅ **Высокая эффективность**: "
            "Вторичная проверка значительно повышает точность оценки."
        )
    elif m.improvement_rate >= 5:
        effectiveness = (
            "✅ **Умеренная эффективность**: Вторичная проверка улучшает точность оценки."
        )
    elif m.improvement_rate > 0:
        effectiveness = "⚠️ **Низкая эффективность**: Вторичная проверка дает небольшое улучшение."
    else:
        effectiveness = "❌ **Неэффективно**: Вторичная проверка не улучшает точность."

    md_lines.append("## Выводы об эффективности\n")
    md_lines.append(f"{effectiveness}\n")

    # ===== ТАБЛИЦА ПО ТЕСТ-КЕЙСАМ =====
    md_lines.append("## Подробные результаты по вопросам\n")
    md_lines.append(
        "| ID | Топик | Сложность | Эталон | Первичная | Вторичная | Согласие | Статус |"
    )
    md_lines.append(
        "|:--:|:------|:---------:|:------:|:---------:|:---------:|:--------:|:------:|"
    )

    for v in report.verifications:
        gt = "✓" if v.ground_truth else "✗"
        p = "✓" if v.primary_evaluation.verdict else "✗"
        j = "✓" if v.secondary_evaluation.verdict else "✗"
        agree = "✓" if v.secondary_evaluation.agree_with_primary else "✗"

        # Статус
        if v.secondary_evaluation.verdict == v.ground_truth:
            if v.primary_evaluation.verdict == v.ground_truth:
                status = "🟢"
            else:
                status = "🟡"
        elif v.primary_evaluation.verdict == v.ground_truth:
            status = "🔴"
        else:
            status = "⚫️"

        md_lines.append(
            f"| {v.question_id} | {v.topic} | {v.difficulty} | "
            f"{gt} | {p} | {j} | {agree} | {status} |"
        )

    # ===== Легенда =====
    md_lines.append("\n### Легенда\n")
    md_lines.append(
        "- **Эталон**: правильность ответа согласно тестовым данным "
        "(✓ = правильно, ✗ = неправильно)."
    )
    md_lines.append(
        "- **Первичная/Вторичная**: оценка нейросети (✓ = правильно, ✗ = неправильно)."
    )
    md_lines.append("- **Согласие**: совпадение оценок первичной и вторичной проверки.")
    md_lines.append("- **Статус**: результат вторичной проверки:")
    md_lines.append("  - 🟢 **Корректно**: Вторичная проверка подтвердила правильную оценку.")
    md_lines.append("  - 🟡 **Исправлено**: Вторичная проверка исправила ошибку первичной.")
    md_lines.append(
        "  - 🔴 **Ошибка**: Вторичная проверка не исправила ошибку первичной или создала новую."
    )
    md_lines.append("  - ⚫️ **Оба неверны**: Обе проверки дали неправильную оценку.\n")

    # ===== ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО КАЖДОМУ ВОПРОСУ =====
    md_lines.append("## 📝 Детальная информация по вопросам\n")

    for v in report.verifications:
        gt_emoji = "✅" if v.ground_truth else "❌"
        p_emoji = "✅" if v.primary_evaluation.verdict else "❌"
        j_emoji = "✅" if v.secondary_evaluation.verdict else "❌"

        md_lines.append(f"### Вопрос {v.question_id}: {v.topic} ({v.difficulty})\n")
        md_lines.append(f"**Вопрос:** {v.question_text}\n")
        md_lines.append(f"**Ответ пользователя:** {v.user_answer}\n")
        md_lines.append(f"**Ожидаемый ответ:** {v.expected_answer}\n")
        md_lines.append("**Результаты проверки:**")
        md_lines.append(
            f"- Эталон (Ground Truth): {gt_emoji} "
            f"{'Правильно' if v.ground_truth else 'Неправильно'}"
        )
        md_lines.append(
            f"- Первичная проверка: {p_emoji} "
            f"{'Правильно' if v.primary_evaluation.verdict else 'Неправильно'}"
        )
        md_lines.append(
            f"- Вторичная проверка (Judge): {j_emoji} "
            f"{'Правильно' if v.secondary_evaluation.verdict else 'Неправильно'}"
        )
        md_lines.append(
            f"- Согласие: {'✓ Да' if v.secondary_evaluation.agree_with_primary else '✗ Нет'}"
        )

        if v.secondary_evaluation.feedback:
            md_lines.append(f"\n**Комментарий судьи:** {v.secondary_evaluation.feedback}")

        md_lines.append("\n---\n")

    return "\n".join(md_lines)


# =======================
# Entrypoint
# =======================


def main(args: argparse.Namespace) -> None:
    """Основная функция скрипта."""
    data: dict[str, Any] = json.loads(Path(args.test_data).read_text(encoding="utf-8"))

    # Правильный парсинг структуры JSON
    test_collection_data = data.get("test_collection", data)

    # Проверяем, есть ли поле "tests" (старая структура) или "topics" (новая структура)
    topics_data = test_collection_data.get("topics") or test_collection_data.get("tests", [])

    topics: list[Topic] = [
        Topic(
            topic_id=test.get("topic_id", test.get("test_id", "")),
            topic_name=test["topic_name"],
            questions=[Question(**q) for q in test["questions"]],
        )
        for test in topics_data
    ]

    test_collection = TestCollection(
        creation_date=test_collection_data["creation_date"],
        total_questions=test_collection_data["total_questions"],
        topics_count=test_collection_data["topics_count"],
        topics=topics,
    )

    verifications = asyncio.run(process_verifications(test_collection))
    metrics = calculate_metrics(verifications)

    report = EffectivenessReport(
        report_date=datetime.now().isoformat(),
        overall_metrics=metrics,
        verifications=verifications,
    )

    output = generate_markdown_report(report)
    Path(args.output).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True, help="Путь к JSON файлу с тестовыми данными")
    parser.add_argument("--language", default="ru", help="Язык для отчета")
    parser.add_argument(
        "--output", default="effectiveness_report.md", help="Путь к выходному файлу"
    )
    main(parser.parse_args())

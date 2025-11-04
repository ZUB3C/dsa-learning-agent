import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.agents.registry import load_agent


class Question(BaseModel):
    question_id: int
    difficulty: str
    question_text: str
    expected_answer: str
    user_answer: str
    key_points: list[str]
    is_correct: bool  # Ground truth


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
    is_correct: bool
    feedback: str


class SecondaryEvaluation(BaseModel):
    agree_with_primary: bool
    is_correct: bool
    feedback: str
    verification_notes: str | None = None


class TestVerification(BaseModel):
    question_id: int
    topic: str
    difficulty: str
    ground_truth: bool
    primary_evaluation: PrimaryEvaluation
    secondary_evaluation: SecondaryEvaluation
    timestamp: str


class VerificationMetrics(BaseModel):
    total_verifications: int
    agreement_count: int
    disagreement_count: int
    agreement_rate: float
    # Метрики точности
    primary_accuracy: float
    secondary_accuracy: float
    improvement_rate: float


class EffectivenessReport(BaseModel):
    report_date: str
    overall_metrics: VerificationMetrics
    verifications: list[TestVerification]


async def verify_answer(
    question: Question, language: str = "ru"
) -> tuple[PrimaryEvaluation, SecondaryEvaluation]:
    """Проверяет ответ без передачи is_correct"""
    try:
        # Первичная проверка
        primary_agent = load_agent("verification", language=language)
        primary_result = await primary_agent.ainvoke({
            "question": question.question_text,
            "expected_answer": question.expected_answer,
            "user_answer": question.user_answer,
        })

        try:
            primary_eval_dict = json.loads(primary_result)
            primary_eval = PrimaryEvaluation(**primary_eval_dict)
        except (json.JSONDecodeError, ValueError):
            primary_eval = PrimaryEvaluation(is_correct=False, feedback="Ошибка парсинга")

        # Вторичная проверка
        secondary_agent = load_agent("verification-secondary", language=language)
        secondary_result = await secondary_agent.ainvoke({
            "primary_evaluation": json.dumps(primary_eval.model_dump(), ensure_ascii=False),
            "question": question.question_text,
            "user_answer": question.user_answer,
        })

        try:
            secondary_eval_dict = json.loads(secondary_result)
            secondary_eval = SecondaryEvaluation(**secondary_eval_dict)
        except (json.JSONDecodeError, ValueError):
            secondary_eval = SecondaryEvaluation(
                agree_with_primary=True,
                is_correct=primary_eval.is_correct,
                feedback=primary_eval.feedback,
                verification_notes="Ошибка парсинга",
            )

        return primary_eval, secondary_eval

    except Exception as e:
        print(f"Ошибка в вопросе {question.question_id}: {e}")
        return (
            PrimaryEvaluation(is_correct=False, feedback="Ошибка"),
            SecondaryEvaluation(
                agree_with_primary=False,
                is_correct=False,
                feedback="Ошибка",
                verification_notes=str(e),
            ),
        )


async def process_verifications(
    test_collection: TestCollection, language: str = "ru"
) -> list[TestVerification]:
    """Обрабатывает все вопросы"""
    verifications = []
    total = test_collection.total_questions
    processed = 0

    for topic in test_collection.topics:
        for question in topic.questions:
            processed += 1
            print(f"[{processed}/{total}] Вопрос {question.question_id}: {topic.topic_name}")

            primary_eval, secondary_eval = await verify_answer(question, language)

            verification = TestVerification(
                question_id=question.question_id,
                topic=topic.topic_name,
                difficulty=question.difficulty,
                ground_truth=question.is_correct,
                primary_evaluation=primary_eval,
                secondary_evaluation=secondary_eval,
                timestamp=datetime.now().isoformat(),
            )

            verifications.append(verification)

    return verifications


def calculate_metrics(verifications: list[TestVerification]) -> VerificationMetrics:
    """Вычисляет метрики без использования баллов"""
    if not verifications:
        return VerificationMetrics(
            total_verifications=0,
            agreement_count=0,
            disagreement_count=0,
            agreement_rate=0.0,
            primary_accuracy=0.0,
            secondary_accuracy=0.0,
            improvement_rate=0.0,
        )

    total = len(verifications)
    agreements = sum(1 for v in verifications if v.secondary_evaluation.agree_with_primary)
    disagreements = total - agreements

    # Точность проверок относительно ground truth
    primary_correct = sum(
        1 for v in verifications if v.primary_evaluation.is_correct == v.ground_truth
    )
    secondary_correct = sum(
        1 for v in verifications if v.secondary_evaluation.is_correct == v.ground_truth
    )

    primary_accuracy = (primary_correct / total) * 100 if total > 0 else 0
    secondary_accuracy = (secondary_correct / total) * 100 if total > 0 else 0
    improvement_rate = secondary_accuracy - primary_accuracy

    return VerificationMetrics(
        total_verifications=total,
        agreement_count=agreements,
        disagreement_count=disagreements,
        agreement_rate=(agreements / total * 100) if total > 0 else 0,
        primary_accuracy=primary_accuracy,
        secondary_accuracy=secondary_accuracy,
        improvement_rate=improvement_rate,
    )


def load_test_collection_from_file(filepath: str) -> TestCollection:
    """Загружает тестовую коллекцию"""
    with Path(filepath).open(encoding="utf-8") as f:
        data = json.load(f)

    topics = []
    for test in data.get("test_collection", {}).get("tests", []):
        questions = [Question(**q) for q in test.get("questions", [])]
        topics.append(
            Topic(
                topic_id=test.get("test_id", ""),
                topic_name=test.get("topic", ""),
                questions=questions,
            )
        )

    return TestCollection(
        creation_date=data.get("test_collection", {}).get("creation_date", ""),
        total_questions=data.get("test_collection", {}).get("total_questions", 0),
        topics_count=data.get("test_collection", {}).get("topics_count", 0),
        topics=topics,
    )


def generate_markdown_report(report: EffectivenessReport) -> str:
    """Генерирует Markdown отчет с таблицей результатов"""
    md_lines = [
        "# Отчет об эффективности вторичной верификации",
        f"\n**Дата:** {report.report_date}",
        "\n## Общая статистика\n",
        f"- **Всего проверок:** {report.overall_metrics.total_verifications}",
        f"- **Согласие проверок:** {report.overall_metrics.agreement_count} ({report.overall_metrics.agreement_rate:.1f}%)",
        f"- **Расхождения:** {report.overall_metrics.disagreement_count} ({100 - report.overall_metrics.agreement_rate:.1f}%)",
        "\n### 🎯 Точность относительно эталона\n",
        f"- **Точность первичной проверки:** {report.overall_metrics.primary_accuracy:.1f}%",
        f"- **Точность вторичной проверки:** {report.overall_metrics.secondary_accuracy:.1f}%",
        f"- **Улучшение от вторичной проверки:** {report.overall_metrics.improvement_rate:+.1f}%",
    ]

    # Оценка эффективности
    md_lines.append("\n## Выводы об эффективности\n")
    if report.overall_metrics.improvement_rate > 5:
        md_lines.append(
            "✅ **Высокая эффективность**: Вторичная проверка значительно улучшает точность (>5%)"
        )
    elif report.overall_metrics.improvement_rate > 0:
        md_lines.append(
            "⚠️ **Умеренная эффективность**: Вторичная проверка дает небольшое улучшение"
        )
    else:
        md_lines.append("❌ **Низкая эффективность**: Вторичная проверка не улучшает результаты")

    # Таблица с подробными результатами
    md_lines.append("\n## Подробные результаты по вопросам\n")
    md_lines.append(
        "| ID | Топик | Сложность | Эталон | Первичная | Вторичная | Согласие | Статус |"
    )
    md_lines.append(
        "|:--:|:------|:---------:|:------:|:---------:|:---------:|:--------:|:------:|"
    )

    for v in report.verifications:
        # Форматирование данных
        q_id = v.question_id
        topic = v.topic[:20] + "..." if len(v.topic) > 20 else v.topic
        difficulty = {"easy": "Легко", "medium": "Средне", "hard": "Сложно"}.get(
            v.difficulty, v.difficulty
        )

        # Эмодзи для булевых значений
        ground_truth_emoji = "✓" if v.ground_truth else "✗"
        primary_emoji = "✓" if v.primary_evaluation.is_correct else "✗"
        secondary_emoji = "✓" if v.secondary_evaluation.is_correct else "✗"
        agreement_emoji = "✓" if v.secondary_evaluation.agree_with_primary else "✗"

        # Определение статуса
        if v.secondary_evaluation.is_correct == v.ground_truth:
            if v.primary_evaluation.is_correct == v.ground_truth:
                status = "🟢"  # Обе правильно
            else:
                status = "🟡"  # Вторичная исправила ошибку
        elif v.primary_evaluation.is_correct == v.ground_truth:
            status = "🔴"  # Вторичная ухудшила
        else:
            status = "🔴"  # Обе неправильно

        md_lines.append(
            f"| {q_id} | {topic} | {difficulty} | {ground_truth_emoji} | "
            f"{primary_emoji} | {secondary_emoji} | {agreement_emoji} | {status} |"
        )

    # Легенда
    md_lines.append("\n### Легенда\n")
    md_lines.append("- **Эталон**: правильность ответа согласно тестовым данным")
    md_lines.append("- **Первичная/Вторичная**: оценка нейросети (✓ = правильно, ✗ = неправильно)")
    md_lines.append("- **Согласие**: согласна ли вторичная проверка с первичной")
    md_lines.append(
        "- **Статус**: 🟢 = вторичная корректна, 🟡 = вторичная исправила, 🔴 = ошибка"
    )

    return "\n".join(md_lines)


async def main(args: argparse.Namespace) -> None:
    print("🔍 Загрузка тестовых данных...")
    test_collection = load_test_collection_from_file(args.test_data)

    print(f"📊 Найдено {test_collection.total_questions} вопросов\n")
    print("⚙️ Начинаем верификацию...")

    try:
        verifications = await process_verifications(test_collection, args.language)
        print(f"\n✅ Обработано {len(verifications)} вопросов")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    # Генерируем отчет
    overall_metrics = calculate_metrics(verifications)
    report = EffectivenessReport(
        report_date=datetime.now().isoformat(),
        overall_metrics=overall_metrics,
        verifications=verifications,
    )

    markdown = generate_markdown_report(report)

    # Сохраняем результаты
    output_path = Path(args.output)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"\n📝 Отчет сохранен: {args.output}")

    # Выводим ключевую метрику
    print(f"\n🎯 Улучшение от вторичной проверки: {overall_metrics.improvement_rate:+.1f}%")
    print(f"📊 Точность первичной: {overall_metrics.primary_accuracy:.1f}%")
    print(f"📊 Точность вторичной: {overall_metrics.secondary_accuracy:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="test_data_updated.json")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--output", default="effectiveness_report.md")
    args = parser.parse_args()

    asyncio.run(main(args))

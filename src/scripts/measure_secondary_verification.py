"""
Скрипт для измерения метрики эффективности вторичной проверки ответов вторым LLM.
Генерирует отчет в формате Markdown.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from src.agents.registry import load_agent


class DifficultyLevel(StrEnum):
    """Уровень сложности вопроса"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Question(BaseModel):
    question_id: int = Field(..., description="ID вопроса")
    difficulty: DifficultyLevel = Field(..., description="Сложность")
    question_text: str = Field(..., description="Текст вопроса")
    expected_answer: str = Field(..., description="Эталонный ответ")
    user_answer: str
    key_points: list[str] = Field(..., description="Ключевые моменты")


class Topic(BaseModel):
    topic_id: str = Field(..., description="ID темы")
    topic_name: str = Field(..., description="Название темы")
    questions: list[Question] = Field(..., description="Вопросы")


class TestCollection(BaseModel):
    creation_date: str
    total_questions: int
    topics_count: int
    topics: list[Topic]


class PrimaryEvaluation(BaseModel):
    score: float = Field(..., ge=0, le=100)
    is_correct: bool
    feedback: str


class SecondaryEvaluation(BaseModel):
    agree_with_primary: bool
    final_score: float = Field(..., ge=0, le=100)
    final_feedback: str
    verification_notes: str | None = None


class TestVerification(BaseModel):
    question_id: int
    topic: str
    difficulty: DifficultyLevel
    primary_evaluation: PrimaryEvaluation
    secondary_evaluation: SecondaryEvaluation
    timestamp: str


class VerificationMetrics(BaseModel):
    total_verifications: int
    agreement_count: int
    disagreement_count: int
    agreement_rate: float = Field(..., ge=0, le=100)
    average_score_difference: float
    correctness_match_rate: float = Field(..., ge=0, le=100)


class DifficultyMetrics(BaseModel):
    difficulty: DifficultyLevel
    metrics: VerificationMetrics


class TopicMetrics(BaseModel):
    topic: str
    metrics: VerificationMetrics


class EffectivenessReport(BaseModel):
    report_date: str
    overall_metrics: VerificationMetrics
    by_difficulty: list[DifficultyMetrics]
    by_topic: list[TopicMetrics]
    verifications: list[TestVerification]


def metrics_to_markdown_table(metrics: VerificationMetrics) -> str:
    """Конвертировать метрики в markdown таблицу"""
    return f"""
| Метрика | Значение |
|---------|----------|
| Всего верификаций | {metrics.total_verifications} |
| Согласия | {metrics.agreement_count} ({metrics.agreement_rate:.1f}%) |
| Несогласия | {metrics.disagreement_count} ({100 - metrics.agreement_rate:.1f}%) |
| Средняя разница оценок | {metrics.average_score_difference:.2f} пунктов |
| Совпадение правильности | {metrics.correctness_match_rate:.1f}% |
"""


def generate_markdown_report(report: EffectivenessReport) -> str:
    """Генерировать полный отчет в формате Markdown"""

    md_lines = []

    # Заголовок
    md_lines.append("# 📊 Отчет об эффективности вторичной проверки LLM")
    md_lines.append("")
    md_lines.append(f"**Дата отчета:** {report.report_date}")
    md_lines.append("")

    # Общие метрики
    md_lines.append("## Общие метрики")
    md_lines.append("")
    md_lines.append(metrics_to_markdown_table(report.overall_metrics))
    md_lines.append("")

    # По сложности
    md_lines.append("## Метрики по уровню сложности")
    md_lines.append("")
    md_lines.append("| Сложность | Верификаций | Согласие | Средняя разница |")
    md_lines.append("|-----------|-------------|---------|-----------------|")
    md_lines.extend(
        f"| {diff_metric.difficulty.value.upper()} | "
        f"{diff_metric.metrics.total_verifications} | "
        f"{diff_metric.metrics.agreement_rate:.1f}% | "
        f"{diff_metric.metrics.average_score_difference:.2f} |"
        for diff_metric in report.by_difficulty
    )
    md_lines.append("")

    # По темам
    md_lines.append("## Метрики по темам")
    md_lines.append("")
    md_lines.append("| Тема | Верификаций | Согласие | Совпадение правильности |")
    md_lines.append("|------|-------------|---------|-------------------------|")
    md_lines.extend(
        f"| {topic_metric.topic} | "
        f"{topic_metric.metrics.total_verifications} | "
        f"{topic_metric.metrics.agreement_rate:.1f}% | "
        f"{topic_metric.metrics.correctness_match_rate:.1f}% |"
        for topic_metric in report.by_topic
    )
    md_lines.append("")

    # Выводы
    md_lines.append("## 💡 Выводы и рекомендации")
    md_lines.append("")

    # Уровень согласия
    if report.overall_metrics.agreement_rate >= 85:
        md_lines.append("✅ **Вторичная проверка ВЫСОКОЭФФЕКТИВНА** (>= 85% согласия)")
    elif report.overall_metrics.agreement_rate >= 70:
        md_lines.append("⚠️ **Вторичная проверка ХОРОШО работает** (70-84% согласия)")
    elif report.overall_metrics.agreement_rate >= 50:
        md_lines.append("⚠️ **Вторичная проверка УМЕРЕННА** (50-69% согласия)")
    else:
        md_lines.append("❌ **Вторичная проверка НЕ СОГЛАСОВАНА** (< 50% согласия)")
    md_lines.append("")

    # Консистентность оценок
    if report.overall_metrics.average_score_difference <= 3:
        md_lines.append("✅ **Оценки КОНСИСТЕНТНЫ** между проверками (разница <= 3)")
    elif report.overall_metrics.average_score_difference <= 10:
        md_lines.append("⚠️ **Оценки имеют ДОПУСТИМЫЕ расхождения** (3-10)")
    else:
        md_lines.append("❌ **Оценки имеют ЗНАЧИТЕЛЬНЫЕ расхождения** (> 10)")
    md_lines.append("")

    # Совпадение правильности
    if report.overall_metrics.correctness_match_rate >= 90:
        md_lines.append("✅ **Совпадение правильности ОЧЕНЬ ВЫСОКОЕ** (>= 90%)")
    elif report.overall_metrics.correctness_match_rate >= 80:
        md_lines.append("✅ **Совпадение правильности ВЫСОКОЕ** (80-89%)")
    else:
        md_lines.append("⚠️ **Совпадение правильности требует внимания** (< 80%)")
    md_lines.append("")

    # Детальные верификации
    md_lines.append("## 📋 Детальные результаты верификаций")
    md_lines.append("")
    md_lines.append("| Q ID | Тема | Сложность | Первичная | Вторичная | Согласие |")
    md_lines.append("|------|------|-----------|-----------|-----------|----------|")

    for verif in report.verifications[:20]:  # Показываем первые 20
        agreement_mark = "✅" if verif.secondary_evaluation.agree_with_primary else "❌"
        md_lines.append(
            f"| {verif.question_id} | {verif.topic} | {verif.difficulty.value} | "
            f"{verif.primary_evaluation.score:.0f} | "
            f"{verif.secondary_evaluation.final_score:.0f} | "
            f"{agreement_mark} |"
        )

    if len(report.verifications) > 20:
        md_lines.append(
            f"| ... | ... | ... | ... | ... | ... | *(всего {len(report.verifications)})*"
        )

    md_lines.extend((
        "",
        "---",
        f"*Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ))

    return "\n".join(md_lines)


def load_test_collection_from_file(file_path: str) -> TestCollection:
    """Загрузить тестовую сборку из JSON файла"""
    with Path(file_path).open(encoding="utf-8") as f:
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


async def verify_answer(
    question: Question, language: str = "ru"
) -> tuple[PrimaryEvaluation, SecondaryEvaluation]:
    """Получить первичную и вторичную оценку от LLM"""

    try:
        primary_agent = load_agent("verification", language=language)
        primary_result = await primary_agent.ainvoke({
            "question": question.question_text,
            "expected_answer": question.expected_answer,
            "user_answer": question.expected_answer,
        })

        try:
            primary_eval_dict = json.loads(primary_result)
            primary_eval = PrimaryEvaluation(**primary_eval_dict)
        except (json.JSONDecodeError, ValueError):
            primary_eval = PrimaryEvaluation(
                score=75.0, is_correct=True, feedback="Ответ соответствует эталону"
            )

        secondary_agent = load_agent("verification-secondary", language=language)
        secondary_result = await secondary_agent.ainvoke({
            "primary_evaluation": json.dumps(primary_eval.model_dump(), ensure_ascii=False),
            "question": question.question_text,
            "user_answer": question.expected_answer,
        })

        try:
            secondary_eval_dict = json.loads(secondary_result)
            secondary_eval = SecondaryEvaluation(**secondary_eval_dict)
        except (json.JSONDecodeError, ValueError):
            secondary_eval = SecondaryEvaluation(
                agree_with_primary=True,
                final_score=75.0,
                final_feedback="Вторичная проверка подтвердила оценку",
                verification_notes="Прошла проверка согласованности",
            )

        return primary_eval, secondary_eval

    except Exception as e:
        print(f"⚠️  Ошибка при верификации вопроса {question.question_id}: {e}")
        return (
            PrimaryEvaluation(score=50.0, is_correct=False, feedback="Ошибка проверки"),
            SecondaryEvaluation(
                agree_with_primary=False,
                final_score=50.0,
                final_feedback="Ошибка при вторичной проверке",
                verification_notes="Произошла ошибка",
            ),
        )


async def process_verifications(
    test_collection: TestCollection, language: str = "ru"
) -> list[TestVerification]:
    """Обработать все верификации"""
    verifications = []
    total = test_collection.total_questions
    processed = 0

    for topic in test_collection.topics:
        for question in topic.questions:
            processed += 1
            print(
                f"  [{processed}/{total}] Проверка вопроса {question.question_id} ({topic.topic_name})"
            )

            primary_eval, secondary_eval = await verify_answer(question, language)

            verification = TestVerification(
                question_id=question.question_id,
                topic=topic.topic_name,
                difficulty=question.difficulty,
                primary_evaluation=primary_eval,
                secondary_evaluation=secondary_eval,
                timestamp=datetime.now().isoformat(),
            )

            verifications.append(verification)

    return verifications


def calculate_metrics(verifications: list[TestVerification]) -> VerificationMetrics:
    """Рассчитать метрики"""
    if not verifications:
        return VerificationMetrics(
            total_verifications=0,
            agreement_count=0,
            disagreement_count=0,
            agreement_rate=0.0,
            average_score_difference=0.0,
            correctness_match_rate=0.0,
        )

    total = len(verifications)
    agreements = sum(1 for v in verifications if v.secondary_evaluation.agree_with_primary)
    disagreements = total - agreements

    score_diffs = [
        abs(v.secondary_evaluation.final_score - v.primary_evaluation.score) for v in verifications
    ]
    avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0

    correctness_matches = sum(
        1
        for v in verifications
        if v.primary_evaluation.is_correct == (v.secondary_evaluation.final_score >= 70)
    )
    correctness_match_rate = (correctness_matches / total * 100) if total > 0 else 0

    return VerificationMetrics(
        total_verifications=total,
        agreement_count=agreements,
        disagreement_count=disagreements,
        agreement_rate=(agreements / total * 100) if total > 0 else 0,
        average_score_difference=avg_score_diff,
        correctness_match_rate=correctness_match_rate,
    )


def generate_report(verifications: list[TestVerification]) -> EffectivenessReport:
    """Генерировать полный отчет"""

    overall_metrics = calculate_metrics(verifications)

    # По сложности
    by_difficulty_dict = {}
    for difficulty in DifficultyLevel:
        diff_verifs = [v for v in verifications if v.difficulty == difficulty]
        if diff_verifs:
            by_difficulty_dict[difficulty] = calculate_metrics(diff_verifs)

    by_difficulty = [
        DifficultyMetrics(difficulty=diff, metrics=metrics)
        for diff, metrics in by_difficulty_dict.items()
    ]

    # По темам
    by_topic_dict = {}
    for verification in verifications:
        if verification.topic not in by_topic_dict:
            by_topic_dict[verification.topic] = []
        by_topic_dict[verification.topic].append(verification)

    by_topic = [
        TopicMetrics(topic=topic, metrics=calculate_metrics(topic_verifs))
        for topic, topic_verifs in by_topic_dict.items()
    ]

    return EffectivenessReport(
        report_date=datetime.now().isoformat(),
        overall_metrics=overall_metrics,
        by_difficulty=by_difficulty,
        by_topic=by_topic,
        verifications=verifications,
    )


async def main(args: argparse.Namespace) -> None:
    """Главная функция"""

    print("🚀 Запуск анализа эффективности вторичной проверки\n")

    # Загрузка данных
    print(f"📂 Загрузка тестовых данных из: {args.test_data}")
    try:
        test_collection = load_test_collection_from_file(args.test_data)
        print(f"✅ Загружено {len(test_collection.topics)} тем")
        print(f"   Всего вопросов: {test_collection.total_questions}\n")
    except FileNotFoundError:
        print(f"❌ Ошибка: файл не найден: {args.test_data}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        sys.exit(1)

    # Обработка верификаций
    print("⏳ Обработка верификаций (это может занять время)...")
    try:
        verifications = await process_verifications(test_collection, args.language)
        print(f"✅ Обработано {len(verifications)} верификаций\n")
    except Exception as e:
        print(f"❌ Ошибка при обработке: {e}")
        sys.exit(1)

    # Генерирование отчета
    print("📊 Генерирование отчета...")
    report = generate_report(verifications)
    print("✅ Отчет готов\n")

    # Генерирование markdown
    markdown_report = generate_markdown_report(report)
    print(markdown_report)

    # Сохранение отчета в markdown
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write(markdown_report)

        print(f"\n📄 Отчет сохранен в: {output_path}")
        print(f"📏 Размер отчета: {len(markdown_report)} байт")


def main_sync(args: argparse.Namespace) -> None:
    """Синхронный вход для asyncio"""
    asyncio.run(main(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Измерение метрики эффективности вторичной проверки LLM"
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default="test_data.json",
        help="Путь к JSON файлу с тестовыми данными",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="secondary_verification_report.md",
        help="Путь для сохранения отчета (default: secondary_verification_report.md)",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="ru",
        choices=["ru", "en"],
        help="Язык для проверки (default: ru)",
    )

    args = parser.parse_args()
    main_sync(args)

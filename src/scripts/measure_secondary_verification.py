import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

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
    language: str = "ru",
) -> tuple[PrimaryEvaluation, SecondaryEvaluation]:
    primary_agent = load_agent("verification", language=language)
    primary_raw = await primary_agent.ainvoke({
        "question": question.question_text,
        "expected_answer": question.expected_answer,
        "user_answer": question.user_answer,
    })

    try:
        primary_eval = PrimaryEvaluation(**json.loads(primary_raw))
    except Exception:
        primary_eval = PrimaryEvaluation(verdict=False)

    secondary_agent = load_agent("verification-secondary", language=language)
    secondary_raw = await secondary_agent.ainvoke({
        "primary_verdict": primary_eval.verdict,
        "question": question.question_text,
        "user_answer": question.user_answer,
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
    language: str,
) -> list[TestVerification]:
    results: list[TestVerification] = []

    for topic in test_collection.topics:
        for question in topic.questions:
            primary, secondary = await verify_answer(question, language)

            results.append(
                TestVerification(
                    question_id=question.question_id,
                    topic=topic.topic_name,
                    difficulty=question.difficulty,
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
    m = report.overall_metrics

    return f"""# Отчёт об эффективности вторичной проверки

**Дата:** {report.report_date}

## 🎯 Accuracy
- Primary Accuracy: {m.primary_accuracy:.1f}%
- Judge Accuracy: {m.secondary_accuracy:.1f}%
- Improvement Rate: {m.improvement_rate:+.1f}%

## ⚖️ Ошибки судьи
- False Positive Rate: {m.false_positive_rate:.1f}%

## 🤝 Agreement
- Agreement Rate: {m.agreement_rate:.1f}%
- Agreements: {m.agreement_count}
- Disagreements: {m.disagreement_count}

## 🧮 Confusion Matrix (Judge)
- TP: {m.true_positive}
- TN: {m.true_negative}
- FP: {m.false_positive}
- FN: {m.false_negative}
"""


# =======================
# Entrypoint
# =======================


def main(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.test_data).read_text(encoding="utf-8"))

    topics = [
        Topic(
            topic_id=test["test_id"],
            topic_name=test["topic"],
            questions=[Question(**q) for q in test["questions"]],
        )
        for test in data["test_collection"]["tests"]
    ]

    test_collection = TestCollection(
        creation_date=data["test_collection"]["creation_date"],
        total_questions=data["test_collection"]["total_questions"],
        topics_count=data["test_collection"]["topics_count"],
        topics=topics,
    )

    verifications = asyncio.run(process_verifications(test_collection, args.language))

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
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--output", default="effectiveness_report.md")
    main(parser.parse_args())

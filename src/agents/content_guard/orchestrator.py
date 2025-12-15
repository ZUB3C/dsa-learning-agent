"""
Content Guard Orchestrator.
Code from Section 3.2 of architecture.
"""

import logging
import time

from src.agents.content_guard.content_sanitizer import ContentSanitizer
from src.agents.content_guard.policy_checker import PolicyChecker
from src.agents.content_guard.quality_gate import QualityGate
from src.agents.content_guard.toxicity_checker import ToxicityChecker
from src.config import get_settings
from src.models.content_guard_schemas import CleanDocument, ContentGuardReport
from src.tools.base_tool import Document

logger = logging.getLogger(__name__)


class ContentGuardOrchestrator:
    """
    Content Guard Orchestrator: проверка и очистка всех документов.
    Code from Section 3.2 of architecture.

    Stages:
    1. Toxicity Check (batch with GigaChat3)
    2. Policy Compliance Check (GigaChat3)
    3. Content Sanitization (rule-based)
    4. Quality Gate (rule-based)

    Output: CleanDocument[] with metadata
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.toxicity_checker = ToxicityChecker()
        self.policy_checker = PolicyChecker()
        self.content_sanitizer = ContentSanitizer()
        self.quality_gate = QualityGate()

    async def process(self, documents: list[Document]) -> list[CleanDocument]:
        """
        Process documents through Content Guard pipeline.

        Args:
            documents: List of raw documents from tools

        Returns:
            List of cleaned documents that passed all checks
        """

        if not documents:
            return []

        if not self.settings.content_guard.content_guard_enabled:
            logger.info("⚠️ Content Guard disabled, passing all documents")
            return self._convert_to_clean_documents(documents)

        start_time = time.time()

        logger.info(f"🛡️ Content Guard: processing {len(documents)} documents")

        # Track statistics
        filtered_by_toxicity = 0
        filtered_by_policy = 0
        filtered_by_quality = 0
        total_toxicity_scores = []

        cleaned_documents = []

        # ═══════════════════════════════════════════════════════════
        # STAGE 1: BATCH TOXICITY CHECK
        # ═══════════════════════════════════════════════════════════

        logger.info("🛡️ Stage 1: Toxicity Check")

        doc_contents = [doc.page_content for doc in documents]

        try:
            toxicity_result = await self.toxicity_checker.check_batch(doc_contents)

            logger.info(
                f"📊 Toxicity: avg={toxicity_result.avg_toxicity:.2f}, filtered={toxicity_result.filtered_count}"
            )

            # Filter by toxicity
            safe_documents = []
            for doc, tox_result in zip(documents, toxicity_result.results, strict=False):
                total_toxicity_scores.append(tox_result.toxicity_score)

                if tox_result.is_safe:
                    safe_documents.append(doc)
                else:
                    filtered_by_toxicity += 1
                    logger.debug(f"❌ Filtered by toxicity: {doc.page_content[:100]}...")

            documents = safe_documents

        except Exception as e:
            logger.exception(f"❌ Toxicity check failed: {e}, continuing without filtering")

        if not documents:
            logger.warning("⚠️ All documents filtered by toxicity")
            return []

        # ═══════════════════════════════════════════════════════════
        # STAGE 2: POLICY COMPLIANCE CHECK
        # ═══════════════════════════════════════════════════════════

        logger.info(f"🛡️ Stage 2: Policy Compliance Check ({len(documents)} docs)")

        compliant_documents = []

        for doc in documents:
            try:
                policy_result = await self.policy_checker.check_compliance(doc.page_content)

                if policy_result.compliant:
                    compliant_documents.append(doc)
                else:
                    filtered_by_policy += 1
                    logger.debug(f"❌ Filtered by policy: {policy_result.violations}")

            except Exception as e:
                logger.warning(f"⚠️ Policy check failed for doc: {e}, assuming compliant")
                compliant_documents.append(doc)

        documents = compliant_documents

        if not documents:
            logger.warning("⚠️ All documents filtered by policy")
            return []

        # ═══════════════════════════════════════════════════════════
        # STAGE 3: CONTENT SANITIZATION
        # ═══════════════════════════════════════════════════════════

        logger.info(f"🛡️ Stage 3: Content Sanitization ({len(documents)} docs)")

        sanitized_documents = []

        for doc in documents:
            source_type = "web" if "web" in doc.metadata.get("source", "") else "rag"

            sanitization_result = self.content_sanitizer.sanitize(
                doc.page_content, source_type=source_type
            )

            # Update document with sanitized content
            doc.page_content = sanitization_result.sanitized_content
            doc.metadata["sanitized"] = True
            doc.metadata["removed_elements"] = sanitization_result.removed_elements

            sanitized_documents.append(doc)

        documents = sanitized_documents

        # ═══════════════════════════════════════════════════════════
        # STAGE 4: QUALITY GATE
        # ═══════════════════════════════════════════════════════════

        logger.info(f"🛡️ Stage 4: Quality Gate ({len(documents)} docs)")

        quality_passed_documents = []

        for doc in documents:
            quality_result = self.quality_gate.check(doc.page_content)

            if quality_result.passed:
                quality_passed_documents.append(doc)
            else:
                filtered_by_quality += 1
                logger.debug(f"❌ Filtered by quality: {quality_result.reason}")

        documents = quality_passed_documents

        # ═══════════════════════════════════════════════════════════
        # FINAL: CONVERT TO CLEAN DOCUMENTS
        # ═══════════════════════════════════════════════════════════

        for doc in documents:
            clean_doc = CleanDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
                source=doc.source,
                content_guarded=True,
                toxicity_score=0.0,  # Already filtered
                policy_compliant=True,
                sanitized=True,
                quality_passed=True,
            )
            cleaned_documents.append(clean_doc)

        # ═══════════════════════════════════════════════════════════
        # REPORT
        # ═══════════════════════════════════════════════════════════

        processing_time = (time.time() - start_time) * 1000  # ms

        avg_toxicity = (
            sum(total_toxicity_scores) / len(total_toxicity_scores)
            if total_toxicity_scores
            else 0.0
        )

        report = ContentGuardReport(
            total_documents=len(doc_contents),
            passed_documents=len(cleaned_documents),
            filtered_by_toxicity=filtered_by_toxicity,
            filtered_by_policy=filtered_by_policy,
            filtered_by_quality=filtered_by_quality,
            avg_toxicity_score=avg_toxicity,
            processing_time_ms=processing_time,
        )

        logger.info("✅ Content Guard complete:")
        logger.info(f"   - Total: {report.total_documents}")
        logger.info(f"   - Passed: {report.passed_documents}")
        logger.info(
            f"   - Filtered: toxicity={filtered_by_toxicity}, policy={filtered_by_policy}, quality={filtered_by_quality}"
        )
        logger.info(f"   - Avg toxicity: {avg_toxicity:.3f}")
        logger.info(f"   - Time: {processing_time:.0f}ms")

        return cleaned_documents

    def _convert_to_clean_documents(self, documents: list[Document]) -> list[CleanDocument]:
        """Convert Documents to CleanDocuments without processing."""

        return [
            CleanDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
                source=doc.source,
                content_guarded=False,
            )
            for doc in documents
        ]

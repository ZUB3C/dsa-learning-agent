from __future__ import annotations

from .base_worker import BaseWorker, WorkerResult


class VerificationWorker(BaseWorker):
    endpoint = "/api/v1/verification/check-test"

    async def run(
        self, user_id: str, question: str | None, user_answer: str | None
    ) -> WorkerResult:
        payload = {
            "user_id": user_id,
            "question": question or "",
            "user_answer": user_answer or "",
        }
        return await self._post(payload)

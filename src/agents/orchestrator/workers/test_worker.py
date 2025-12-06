from __future__ import annotations

from .base_worker import BaseWorker, WorkerResult


class TestWorker(BaseWorker):
    endpoint = "/api/v1/tests/generate"

    async def run(self, user_id: str, topic: str | None, user_level: str | None) -> WorkerResult:
        payload = {
            "user_id": user_id,
            "topic": topic or "алгоритмы и структуры данных",
            "difficulty": user_level or "intermediate",
            "question_count": 5,
        }
        return await self._post(payload)

from __future__ import annotations

from .base_worker import BaseWorker, WorkerResult


class MaterialsWorker(BaseWorker):
    endpoint = "/api/v1/materials/get-materials"

    async def run(self, user_id: str, topic: str | None, user_level: str | None) -> WorkerResult:
        payload = {
            "user_id": user_id,
            "topic": topic or "алгоритмы и структуры данных",
            "user_level": user_level or "intermediate",
        }
        return await self._post(payload)

from __future__ import annotations

from .base_worker import BaseWorker, WorkerResult


class SupportWorker(BaseWorker):
    endpoint = "/api/v1/support/get-support"

    async def run(self, user_id: str, message: str, emotional_state: str = "stressed") -> WorkerResult:
        payload = {
            "user_id": user_id,
            "message": message,
            "emotional_state": emotional_state,
        }
        return await self._post(payload)

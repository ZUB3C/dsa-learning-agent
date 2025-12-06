from __future__ import annotations

from dataclasses import dataclass

import httpx

from ...config import settings


@dataclass
class WorkerResult:
    ok: bool
    content: str
    raw: dict | None = None


class BaseWorker:
    """Базовый HTTP-воркер, вызывающий существующие REST эндпоинты."""

    endpoint: str

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "http://127.0.0.1:8001"

    async def _post(self, payload: dict) -> WorkerResult:
        url = f"{self.base_url}{self.endpoint}"
        async with httpx.AsyncClient(timeout=settings.timeout_s) as client:
            resp = await client.post(url, json=payload)
            ok = 200 <= resp.status_code < 300
            try:
                data = resp.json()
            except Exception:
                data = None
            return WorkerResult(
                ok=ok,
                content=data.get("content") if isinstance(data, dict) else resp.text,
                raw=data if isinstance(data, dict) else None,
            )

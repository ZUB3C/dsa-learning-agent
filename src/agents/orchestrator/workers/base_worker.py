from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from ....config import settings


@dataclass
class WorkerResult:
    """
    Represents the result of a worker HTTP request.

    Attributes
    ----------
    ok: bool
        ``True`` if the HTTP status code indicates success (2xx), otherwise ``False``.
    content: str
        The main payload returned by the worker. If the response is JSON and contains a
        ``content`` field, that value is used; otherwise the raw text body is returned.
    raw: Optional[Dict[str, Any]]
        The full JSON payload when the response can be parsed as JSON, otherwise ``None``.
    """

    ok: bool
    content: str
    raw: dict[str, Any] | None = None


class BaseWorker(ABC):
    """
    Base class for all orchestrator workers.

    Sub‑classes must define the ``endpoint`` class attribute (including a leading slash,
    e.g. ``"/api/v1/materials/get-materials"``). The ``_post`` method handles the HTTP
    request and normalises the response into a :class:`WorkerResult`.
    """

    # Sub‑classes should override this.
    endpoint: str = ""

    def __init__(self, base_url: str | None = None) -> None:
        """
        Parameters
        ----------
        base_url: Optional[str]
            Base URL of the service. If omitted, defaults to ``http://127.0.0.1:8001``.
        """
        self.base_url = base_url or "http://127.0.0.1:8001"
        # Normalise the base URL to avoid a trailing slash that would produce
        # a double slash when concatenated with ``endpoint``.
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> WorkerResult:
        """
        Execute the worker's main task.

        Sub‑classes must override this method with their specific signature.

        Returns
        -------
        WorkerResult
            The result of the worker execution.
        """
        msg = "Subclasses must implement the run method"
        raise NotImplementedError(msg)

    async def _post(self, payload: dict[str, Any]) -> WorkerResult:
        """
        Send a POST request to the worker's endpoint.

        Parameters
        ----------
        payload: Dict[str, Any]
            JSON‑serialisable payload to send.

        Returns
        -------
        WorkerResult
            Normalised result of the request.
        """
        if not self.endpoint:
            raise ValueError(
                f"The worker class {self.__class__.__name__} does not define an 'endpoint'."
            )

        # Ensure the endpoint starts with a slash.
        endpoint = self.endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        url = f"{self.base_url}{endpoint}"

        # Use a sensible default timeout if the settings object does not provide one.
        timeout_seconds = getattr(settings, "timeout_s", 30)

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json=payload)

        ok = 200 <= response.status_code < 300
        try:
            data = response.json()
        except Exception:
            data = None

        return WorkerResult(
            ok=ok,
            content=data.get("content") if isinstance(data, dict) else response.text,
            raw=data if isinstance(data, dict) else None,
        )

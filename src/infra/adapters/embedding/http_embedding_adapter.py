from __future__ import annotations

import httpx
from python_shared.http import ResilientHttpClient
from shared.ports.embedding_client_port import EmbeddingClientPort


class HttpEmbeddingClientAdapter(EmbeddingClientPort):
    def __init__(self, base_url: str, profile: str = "api_service") -> None:
        self._base_url: str = base_url.rstrip("/")
        self._client: ResilientHttpClient = ResilientHttpClient(
            base_url=self._base_url,
            timeout=0.5,
            profile=profile,
        )

    async def embed(self, text: str, timeout_ms: int = 500) -> list[float] | None:
        timeout_s: float = timeout_ms / 1000.0
        try:
            response = await self._client.apost(
                "/embed",
                json={"text": text},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            return list(data["embedding"])
        except (httpx.TimeoutException, httpx.HTTPError, KeyError, Exception):
            return None

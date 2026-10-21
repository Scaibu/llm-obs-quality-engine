from __future__ import annotations

import os
import logging
from typing import Optional
from python_shared.http import ResilientHttpClient
from shared.ports.scorer_client_port import ScorerClientPort

logger = logging.getLogger("quality-engine.http-scorer-client")


class HttpScorerClientAdapter(ScorerClientPort):
    def __init__(self, profile: str = "api_service") -> None:
        self.coherence_url: str = os.environ.get("COHERENCE_SERVICE_URL", "http://localhost:8005")
        self.faithfulness_url: str = os.environ.get("FAITHFULNESS_SERVICE_URL", "http://localhost:8006")
        self.toxicity_url: str = os.environ.get("TOXICITY_SERVICE_URL", "http://localhost:8007")
        self.perplexity_url: str = os.environ.get("PERPLEXITY_SERVICE_URL", "http://localhost:8008")
        self._client: ResilientHttpClient = ResilientHttpClient(profile=profile, timeout=5.0)

    def get_coherence_score(
        self,
        trace_id: str,
        span_id: str,
        prompt_type: str | None,
        pii_detected: bool | None,
        prompt_embedding: list[float] | None,
        response_embedding: list[float] | None,
    ) -> float | None:
        if not prompt_type:
            return None
        url = f"{self.coherence_url}/v1/score/semantic-coherence"
        payload = {
            "trace_id": trace_id,
            "span_id": span_id,
            "prompt_type": prompt_type,
            "pii_detected": pii_detected if pii_detected is not None else False,
            "prompt_embedding": prompt_embedding,
            "response_embedding": response_embedding,
        }
        try:
            r = self._client.post(url, json=payload, headers={"x-trace-id": trace_id})
            if r.status_code == 200:
                data = r.json()
                prim = data.get("primary")
                if prim:
                    return prim.get("score")
        except Exception as e:
            logger.error("Error querying coherence scorer: %s", e)
        return None

    def get_faithfulness_score(
        self,
        trace_id: str,
        span_id: str,
        response_text: str | None,
        completion_tokens: int | None,
        rag_context: str | None,
        finish_reason: str | None,
    ) -> float | None:
        if response_text is None or completion_tokens is None:
            return None
        if rag_context is None or len(rag_context) < 50:
            return None

        url = f"{self.faithfulness_url}/v1/score/faithfulness"
        payload = {
            "trace_id": trace_id,
            "span_id": span_id,
            "response_text": response_text,
            "completion_tokens": completion_tokens,
            "rag_context": rag_context,
            "finish_reason": finish_reason,
        }
        try:
            r = self._client.post(url, json=payload, headers={"x-trace-id": trace_id})
            if r.status_code == 200:
                data = r.json()
                return data.get("score")
        except Exception as e:
            logger.error("Error querying faithfulness scorer: %s", e)
        return None

    def get_toxicity_score(
        self,
        trace_id: str,
        span_id: str,
        response_text: str | None,
    ) -> float | None:
        if response_text is None:
            return None
        url = f"{self.toxicity_url}/v1/score/toxicity"
        payload = {
            "trace_id": trace_id,
            "span_id": span_id,
            "response_text": response_text,
        }
        try:
            r = self._client.post(url, json=payload, headers={"x-trace-id": trace_id})
            if r.status_code == 200:
                return r.json().get("score")
        except Exception as e:
            logger.error("Error querying toxicity scorer: %s", e)
        return None

    def get_perplexity_value(
        self,
        trace_id: str,
        span_id: str,
        response_text: str | None,
        completion_tokens: int | None,
        prompt_type: str | None,
        token_logprobs: list[float] | None,
        finish_reason: str | None,
    ) -> float | None:
        perplexity, _ = self.get_perplexity_result(
            trace_id=trace_id,
            span_id=span_id,
            response_text=response_text,
            completion_tokens=completion_tokens,
            prompt_type=prompt_type,
            token_logprobs=token_logprobs,
            finish_reason=finish_reason,
        )
        return perplexity

    def get_perplexity_result(
        self,
        trace_id: str,
        span_id: str,
        response_text: str | None,
        completion_tokens: int | None,
        prompt_type: str | None,
        token_logprobs: list[float] | None,
        finish_reason: str | None,
    ) -> tuple[float | None, bool]:
        if response_text is None or completion_tokens is None or not prompt_type:
            return None, False
        url = f"{self.perplexity_url}/v1/score/perplexity"
        payload = {
            "trace_id": trace_id,
            "span_id": span_id,
            "response_text": response_text,
            "completion_tokens": completion_tokens,
            "prompt_type": prompt_type,
            "token_logprobs": token_logprobs,
            "finish_reason": finish_reason,
        }
        try:
            r = self._client.post(url, json=payload, headers={"x-trace-id": trace_id})
            if r.status_code == 200:
                data = r.json()
                perplexity = data.get("perplexity")
                high_flag: bool = bool(data.get("high_perplexity_flag", False))
                return perplexity, high_flag
        except Exception as e:
            logger.error("Error querying perplexity scorer: %s", e)
        return None, False

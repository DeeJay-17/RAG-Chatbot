from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Request
from sqlalchemy import insert

from app.db import engine
from app.models import llm_traces


def get_request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if not rid:
        rid = str(uuid.uuid4())
        request.state.request_id = rid
    return rid


def start_timer() -> float:
    """Return a high-precision timestamp for latency measurement."""
    return time.perf_counter()


def elapsed_ms(start_ts: float) -> int:
    """Compute elapsed time in milliseconds."""
    return int((time.perf_counter() - start_ts) * 1000)


def persist_trace(
    *,
    request_id: str,
    endpoint: str,
    http_method: str,
    status_code: Optional[int] = None,
    user_query: Optional[str] = None,
    route: Optional[str] = None,
    retrieved_docs_with_scores: Optional[Any] = None,
    llm_prompt: Optional[str] = None,
    llm_prompt_messages: Optional[List[Dict[str, Any]]] = None,
    llm_response: Optional[str] = None,
    embeddings_metadata: Optional[Any] = None,
    state: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    request_payload: Optional[Dict[str, Any]] = None,
    response_body: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> None:

    try:
        with engine.begin() as conn:
            conn.execute(
                insert(llm_traces).values(
                    request_id=request_id,
                    endpoint=endpoint,
                    http_method=http_method,
                    status_code=status_code,
                    user_query=user_query,
                    route=route,
                    retrieved_docs_with_scores=retrieved_docs_with_scores,
                    llm_prompt=llm_prompt,
                    llm_prompt_messages=llm_prompt_messages,
                    llm_response=llm_response,
                    embeddings_metadata=embeddings_metadata,
                    state=state,
                    events=events,
                    request_payload=request_payload,
                    response_body=response_body,
                    error_message=error_message,
                    duration_ms=duration_ms,
                )
            )
    except Exception:
        return


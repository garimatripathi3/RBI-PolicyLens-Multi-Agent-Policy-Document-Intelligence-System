"""
Langfuse observability wrapper.

Observability is optional. If Langfuse keys are absent, every call becomes a
no-op so the pipeline runs locally with zero setup. When configured, each
`trace_span` creates a Langfuse trace and updates it on exit.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

_client = None
if _settings.langfuse_public_key and _settings.langfuse_secret_key:
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=_settings.langfuse_public_key,
            secret_key=_settings.langfuse_secret_key,
            host=_settings.langfuse_host,
        )
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("Langfuse init failed (%s); tracing disabled.", exc)
        _client = None


@contextlib.contextmanager
def trace_span(name: str, **metadata: Any) -> Iterator[None]:
    """Record a Langfuse trace if configured, else no-op."""
    if _client is None:
        yield
        return

    trace = None
    try:
        trace = _client.trace(name=name, metadata=metadata)
        yield
    except Exception:
        if trace is not None:
            with contextlib.suppress(Exception):
                trace.update(level="ERROR")
        raise
    else:
        if trace is not None:
            with contextlib.suppress(Exception):
                trace.update(output={"status": "ok"})
    finally:
        with contextlib.suppress(Exception):
            _client.flush()

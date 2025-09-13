# backend/pvrt/web/sse.py
from __future__ import annotations
import asyncio
import logging
from typing import AsyncIterator, Optional
from starlette.responses import StreamingResponse

_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None

def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Record the main event loop so background threads can enqueue safely."""
    global _EVENT_LOOP
    _EVENT_LOOP = loop

class LogBroker:
    """Broadcast log lines to any number of SSE subscribers."""
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[str]] = []

    async def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        self._subscribers.append(q)
        # small welcome so clients know it's alive
        await q.put("INFO: Client subscribed to logs.")
        return q

    def publish(self, line: str) -> None:
        # Called from any thread
        if _EVENT_LOOP is None:
            # Best-effort fallback: drop if no loop set yet
            return
        def _put_all() -> None:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(line)
                except asyncio.QueueFull:
                    # drop oldest behavior: clear one and push
                    try:
                        q.get_nowait()
                        q.put_nowait(line)
                    except Exception:
                        pass
        # Ensure this runs in the main loop thread
        _EVENT_LOOP.call_soon_threadsafe(_put_all)

def sse_response(q: asyncio.Queue[str]) -> StreamingResponse:
    async def gen() -> AsyncIterator[bytes]:
        try:
            while True:
                line = await q.get()
                # Basic SSE "data:" format; one line per event
                yield f"data: {line}\n\n".encode("utf-8")
        except asyncio.CancelledError:
            return
    return StreamingResponse(gen(), media_type="text/event-stream")

class SSELogHandler(logging.Handler):
    """Logging handler that forwards records to the LogBroker (thread-safe)."""
    def __init__(self, broker: LogBroker):
        super().__init__()
        self.broker = broker

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        # Forward to broker; safe from any thread
        self.broker.publish(msg)

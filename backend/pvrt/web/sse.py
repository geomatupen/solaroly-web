# backend/pvrt/web/sse.py
"""
Server-Sent Events (SSE) utilities:
- LogBroker: tiny pub/sub with history buffer
- SSELogHandler: logging.Handler that publishes to the broker
- sse_response: wraps a Queue or async-iterator as a proper SSE stream
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import AsyncIterator, Deque, Iterable, Optional, Union

from starlette.responses import StreamingResponse


# ---------------------------------------------------------------------------
# Log broker: store a small history and broadcast new lines to subscribers
# ---------------------------------------------------------------------------

class LogBroker:
    """
    A tiny async-safe log broker.

    - Keep a ring buffer of recent lines (history).
    - publish(text): append and fan out to all subscribers.
    - subscribe(): return an asyncio.Queue[str] already primed with history.
    - subscribe_iter(): async generator that yields history then live lines.

    Call `set_loop(asyncio.get_running_loop())` once on startup so that
    `publish()` can push to subscribers from background threads safely.
    """

    def __init__(self, buffer_size: int = 300) -> None:
        self._buffer: Deque[str] = deque(maxlen=max(buffer_size, 0))
        self._subs: set[asyncio.Queue[str]] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --- lifecycle ----------------------------------------------------------

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the main event loop (call this in app startup)."""
        self._loop = loop

    # --- publishers ---------------------------------------------------------

    def publish(self, text: str) -> None:
        """
        Add a line to history and fan out to subscribers.

        Safe to call from non-async code and background threads
        (when set_loop(...) has been called).
        """
        line = text if text.endswith("\n") else f"{text}\n"
        self._buffer.append(line)

        # Fan out to all subscriber queues
        if self._loop and not self._loop.is_closed():
            for q in list(self._subs):
                try:
                    self._loop.call_soon_threadsafe(q.put_nowait, line)
                except (RuntimeError, asyncio.QueueFull):
                    # If the loop is shutting down or the queue is full, drop this subscriber.
                    self._subs.discard(q)

    # --- subscribers --------------------------------------------------------

    async def subscribe(self) -> asyncio.Queue[str]:
        """
        Return a Queue of log lines, seeded with recent history.
        Callers can await this and pass the returned queue to sse_response().
        """
        q: asyncio.Queue[str] = asyncio.Queue()
        # prime with history
        for item in list(self._buffer):
            await q.put(item)
        self._subs.add(q)
        return q

    async def subscribe_iter(self) -> AsyncIterator[str]:
        """
        Async generator: yields history first, then live messages.
        Useful if your /api/logs prefers an iterator over a Queue.
        """
        q: asyncio.Queue[str] = asyncio.Queue()
        self._subs.add(q)

        # Yield history
        for item in list(self._buffer):
            yield item

        try:
            while True:
                item = await q.get()
                yield item
        finally:
            self._subs.discard(q)


# ---------------------------------------------------------------------------
# Logging handler that forwards log records to the broker
# ---------------------------------------------------------------------------

class SSELogHandler(logging.Handler):
    """
    logging.Handler that forwards formatted log records to LogBroker.
    """

    def __init__(self, broker: LogBroker, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.broker = broker

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            # Formatting can fail for many reasons; fall back to raw message.
            msg = record.getMessage()
        self.broker.publish(msg)


# ---------------------------------------------------------------------------
# SSE response wrapper with anti-buffering + heartbeat
# ---------------------------------------------------------------------------

def sse_response(
    source: Union[asyncio.Queue[str], AsyncIterator[str]],
    heartbeat_secs: float = 10.0,
) -> StreamingResponse:
    """
    Turn a Queue or async-iterator of strings into a text/event-stream response.

    - Adds headers to disable proxy buffering (Nginx, etc.).
    - Sends a `:` comment heartbeat every `heartbeat_secs` to keep the pipe hot.
    - Formats each line as an SSE event: "data: <line>\\n\\n".
    """

    async def _gen_from_queue(q: asyncio.Queue[str]):
        try:
            while True:
                try:
                    line = await asyncio.wait_for(q.get(), timeout=heartbeat_secs)
                    yield f"data: {line.rstrip()}\n\n".encode("utf-8")
                except asyncio.TimeoutError:
                    # heartbeat
                    yield b": keep-alive\n\n"
        except asyncio.CancelledError:
            return

    async def _gen_from_iter(it: AsyncIterator[str]):
        try:
            while True:
                try:
                    # try to get the next line with a timeout for heartbeat cadence
                    line_task = asyncio.create_task(it.__anext__())
                    done, _ = await asyncio.wait({line_task}, timeout=heartbeat_secs)
                    if done:
                        line = line_task.result()
                        yield f"data: {line.rstrip()}\n\n".encode("utf-8")
                    else:
                        # heartbeat if no line in this interval
                        line_task.cancel()
                        yield b": keep-alive\n\n"
                except StopAsyncIteration:
                    return
        except asyncio.CancelledError:
            return

    if isinstance(source, asyncio.Queue):
        body = _gen_from_queue(source)
    else:
        body = _gen_from_iter(source)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # Nginx: disable response buffering
        "Connection": "keep-alive",
    }
    return StreamingResponse(body, media_type="text/event-stream", headers=headers)


# ---------------------------------------------------------------------------
# Compatibility shim (kept for API parity with older code)
# ---------------------------------------------------------------------------

def set_event_loop(_loop: asyncio.AbstractEventLoop) -> None:
    """
    Deprecated shim kept for backwards compatibility.
    Call broker.set_loop(loop) in app startup instead.
    """
    # Keep the function for backwards compatibility but emit a visible
    # deprecation warning so callers migrate to broker.set_loop(loop).
    logging.getLogger("pvrt").warning("set_event_loop is deprecated; call broker.set_loop(loop) instead.")
    return

"""Machine resources shared by jobs that run at the same time (D-120).

A queue of videos is processed as a pipeline: while one episode is transcribed on the GPU, the next one downloads,
an earlier one has its speakers detected on the CPU and two more are translated by the cloud AI. Each kind of work
holds a slot of the resource it really uses:

- ``net``: video downloads (two at a time by default),
- ``gpu``: speech recognition and the local translation model (one at a time: the design GPU has 4 GB),
- ``cpu``: voice separation, speaker detection, shot detection (one at a time: each already uses 4 threads),
- ``ai``:  cloud AI translation (two episodes at a time; bounded by the free providers' rate limits).

A gate hands its free slots to the waiting jobs in queue order (lower priority value first), never to whichever
thread happens to wake first, so the first episode of a playlist is always finished first. A thread must never wait
for a gate while holding another one; the pipeline releases each gate before it asks for the next, so the gates
cannot deadlock.
"""

from __future__ import annotations

import itertools
import threading
from contextlib import contextmanager
from typing import Callable, Iterator

from app.core.errors import JobCancelled

NET = "net"
GPU = "gpu"
CPU = "cpu"
AI = "ai"
DEFAULT_CAPACITY = {NET: 2, GPU: 1, CPU: 1, AI: 2}

_POLL_S = 0.2           # how often a waiting job looks at its cancel event


class ResourceGate:
    """A counting semaphore that serves waiters strictly by (priority, arrival)."""

    def __init__(self, name: str, capacity: int):
        self.name = name
        self._capacity = max(1, int(capacity))
        self._holders = 0
        self._waiting: list[tuple] = []          # sorted tickets: (priority, sequence)
        self._sequence = itertools.count()
        self._cond = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        with self._cond:
            return self._holders

    @property
    def waiting(self) -> int:
        with self._cond:
            return len(self._waiting)

    def set_capacity(self, capacity: int) -> None:
        with self._cond:
            self._capacity = max(1, int(capacity))
            self._cond.notify_all()

    def _granted(self, ticket: tuple) -> bool:
        free = self._capacity - self._holders
        return free > 0 and self._waiting.index(ticket) < free

    def acquire(self, priority=0, cancel: threading.Event | None = None,
                on_wait: Callable[[], None] | None = None) -> None:
        """Block until this caller owns a slot. Raises JobCancelled (and gives up its place) when `cancel` is set.
        `on_wait` is called once, outside the lock, when the caller has to wait."""
        with self._cond:
            ticket = (priority, next(self._sequence))
            self._waiting.append(ticket)
            self._waiting.sort()
            if self._granted(ticket):
                self._take(ticket)
                return
        if on_wait is not None:
            on_wait()
        with self._cond:
            try:
                while not self._granted(ticket):
                    if cancel is not None and cancel.is_set():
                        raise JobCancelled()
                    self._cond.wait(_POLL_S)
            except BaseException:
                self._waiting.remove(ticket)
                self._cond.notify_all()          # the next waiter may now be at the head
                raise
            self._take(ticket)

    def _take(self, ticket: tuple) -> None:
        self._waiting.remove(ticket)
        self._holders += 1
        self._cond.notify_all()

    def release(self) -> None:
        with self._cond:
            if self._holders <= 0:
                raise RuntimeError(f"resource {self.name} released more often than acquired")
            self._holders -= 1
            self._cond.notify_all()

    @contextmanager
    def hold(self, priority=0, cancel: threading.Event | None = None,
             on_wait: Callable[[], None] | None = None) -> Iterator[None]:
        self.acquire(priority, cancel, on_wait)
        try:
            yield
        finally:
            self.release()


class ResourceSet:
    """The gates of one machine. Unknown names get a gate of capacity 1."""

    def __init__(self, capacity: dict[str, int] | None = None):
        self._lock = threading.Lock()
        self._gates: dict[str, ResourceGate] = {}
        for name, value in {**DEFAULT_CAPACITY, **(capacity or {})}.items():
            self._gates[name] = ResourceGate(name, value)

    def gate(self, name: str) -> ResourceGate:
        with self._lock:
            if name not in self._gates:
                self._gates[name] = ResourceGate(name, 1)
            return self._gates[name]

    def configure(self, capacity: dict[str, int]) -> None:
        for name, value in capacity.items():
            self.gate(name).set_capacity(value)

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            gates = dict(self._gates)
        return {name: {"capacity": g.capacity, "in_use": g.in_use, "waiting": g.waiting} for name, g in gates.items()}


#: The gates every job of this application process shares (the GUI's queue runs several jobs at once).
SHARED = ResourceSet()

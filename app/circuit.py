"""
Circuit breaker for external-provider calls.

A :class:`CircuitBreaker` tracks *consecutive* failures of the callable it
wraps. The state machine:

    CLOSED  ──(N consecutive failures)──▶  OPEN
    OPEN    ──(cooldown elapsed)────────▶  HALF_OPEN   (one trial allowed)
    HALF_OPEN ──(trial succeeds)───────▶  CLOSED
    HALF_OPEN ──(trial fails)──────────▶  OPEN         (cooldown restarts)

While OPEN, calls fail fast with :class:`CircuitOpenError` *without* invoking
the wrapped callable — that is what lets the failover layer skip a dead
provider instantly instead of paying its timeout on every request.

The monotonic clock is injectable so tests can drive the cooldown transition
deterministically without sleeping. A lock guards the (small) state mutations;
the wrapped call itself runs outside the lock so a slow network call never
blocks other threads from reading state.
"""
from __future__ import annotations

import logging
import threading
from enum import Enum
from time import monotonic
from typing import Callable, Optional, TypeVar

log = logging.getLogger("funstay.circuit")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is OPEN."""


class CircuitBreaker:
    """A per-provider circuit breaker. Thread-safe."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        # True while the single HALF_OPEN trial is being served — concurrent
        # callers fast-fail so only one probe ever hits a recovering provider.
        self._trial_in_flight = False

    @property
    def state(self) -> CircuitState:
        """Current effective state. A pure read — never advances the machine."""
        with self._lock:
            return self._effective_state()

    def _effective_state(self) -> CircuitState:
        """Read-only view: report HALF_OPEN once the cooldown has elapsed, but do
        NOT mutate. The real OPEN→HALF_OPEN promotion (and the single-trial grant)
        happens only in :meth:`_before_call`, so a health probe can never consume
        the trial. Caller holds the lock."""
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            if self._clock() - self._opened_at >= self.cooldown:
                return CircuitState.HALF_OPEN
        return self._state

    def _before_call(self) -> None:
        with self._lock:
            eff = self._effective_state()
            if eff is CircuitState.OPEN:
                raise CircuitOpenError(f"circuit {self.name} is open")
            if eff is CircuitState.HALF_OPEN:
                if self._trial_in_flight:
                    # A probe is already running; everyone else stays fast-failing.
                    raise CircuitOpenError(f"circuit {self.name} is half-open (trial in flight)")
                if self._state is not CircuitState.HALF_OPEN:
                    log.info("circuit %s: OPEN -> HALF_OPEN (cooldown elapsed, trial granted)", self.name)
                self._state = CircuitState.HALF_OPEN
                self._trial_in_flight = True
            # CLOSED: pass straight through.

    def _on_success(self) -> None:
        with self._lock:
            if self._state is not CircuitState.CLOSED:
                log.info("circuit %s: %s -> CLOSED (trial succeeded)", self.name, self._state.value)
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._trial_in_flight = False

    def _on_failure(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # Trial failed: straight back to OPEN, restart the cooldown.
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._trial_in_flight = False
                log.warning("circuit %s: HALF_OPEN -> OPEN (trial failed)", self.name)
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                log.warning(
                    "circuit %s: CLOSED -> OPEN (%d consecutive failures)",
                    self.name,
                    self._consecutive_failures,
                )

    def call(self, fn: Callable[[], T]) -> T:
        """Invoke ``fn`` through the breaker.

        Raises :class:`CircuitOpenError` immediately if the circuit is OPEN.
        Otherwise runs ``fn``; any exception counts as a failure (and is
        re-raised) while a normal return resets the failure counter.
        """
        self._before_call()
        try:
            result = fn()
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def snapshot(self) -> dict[str, object]:
        """Read-only state for health/observability probes (Lane 5). Non-mutating."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._effective_state().value,
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.failure_threshold,
            }

    def reset(self) -> None:
        """Force the breaker back to CLOSED (test/ops helper)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._trial_in_flight = False

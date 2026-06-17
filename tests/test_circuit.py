"""Tests for app.circuit.CircuitBreaker — state transitions with an injected clock."""
import pytest

from app.circuit import CircuitBreaker, CircuitOpenError, CircuitState


class Clock:
    """Manually-advanced monotonic clock."""
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _boom():
    raise RuntimeError("provider down")


def test_closed_passes_through_and_resets_on_success():
    cb = CircuitBreaker("x", failure_threshold=3, cooldown=10, clock=Clock())
    assert cb.call(lambda: "ok") == "ok"
    assert cb.state is CircuitState.CLOSED


def test_opens_after_n_consecutive_failures():
    cb = CircuitBreaker("x", failure_threshold=3, cooldown=10, clock=Clock())
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(_boom)
    assert cb.state is CircuitState.CLOSED  # not yet at threshold
    with pytest.raises(RuntimeError):
        cb.call(_boom)
    assert cb.state is CircuitState.OPEN


def test_open_fails_fast_without_invoking_fn():
    cb = CircuitBreaker("x", failure_threshold=1, cooldown=10, clock=Clock())
    with pytest.raises(RuntimeError):
        cb.call(_boom)
    assert cb.state is CircuitState.OPEN

    called = {"n": 0}

    def fn():
        called["n"] += 1
        return "ok"

    with pytest.raises(CircuitOpenError):
        cb.call(fn)
    assert called["n"] == 0  # fn never ran while OPEN


def test_success_streak_does_not_open():
    cb = CircuitBreaker("x", failure_threshold=2, cooldown=10, clock=Clock())
    with pytest.raises(RuntimeError):
        cb.call(_boom)
    cb.call(lambda: "ok")  # resets the counter
    with pytest.raises(RuntimeError):
        cb.call(_boom)
    assert cb.state is CircuitState.CLOSED  # only 1 consecutive failure


def test_half_open_trial_success_closes():
    clock = Clock()
    cb = CircuitBreaker("x", failure_threshold=1, cooldown=10, clock=clock)
    with pytest.raises(RuntimeError):
        cb.call(_boom)
    assert cb.state is CircuitState.OPEN

    clock.advance(10)  # cooldown elapsed
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.call(lambda: "ok") == "ok"  # trial succeeds
    assert cb.state is CircuitState.CLOSED


def test_half_open_trial_failure_reopens_and_restarts_cooldown():
    clock = Clock()
    cb = CircuitBreaker("x", failure_threshold=1, cooldown=10, clock=clock)
    with pytest.raises(RuntimeError):
        cb.call(_boom)
    clock.advance(10)
    assert cb.state is CircuitState.HALF_OPEN
    with pytest.raises(RuntimeError):
        cb.call(_boom)  # trial fails
    assert cb.state is CircuitState.OPEN

    clock.advance(9)
    assert cb.state is CircuitState.OPEN       # cooldown restarted, not yet elapsed
    clock.advance(1)
    assert cb.state is CircuitState.HALF_OPEN  # elapses again


def test_snapshot_reports_state():
    cb = CircuitBreaker("tenant:llm:openai", failure_threshold=2, cooldown=5, clock=Clock())
    snap = cb.snapshot()
    assert snap["name"] == "tenant:llm:openai"
    assert snap["state"] == "closed"
    assert snap["failure_threshold"] == 2

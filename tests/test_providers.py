"""Tests for the provider failover layer (app.providers.failover).

Uses fake in-memory providers + real CircuitBreakers (with an injected clock).
No SDK, no network, no keys.
"""
import pytest

from app.circuit import CircuitBreaker, CircuitState
from app.providers.base import ProviderError
from app.providers.failover import AllProvidersFailed, FailoverLLM, FailoverTTS


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeLLM:
    def __init__(self, name, *, available=True, fail=False, reply="reply"):
        self.name = name
        self._available = available
        self.fail = fail
        self.reply = reply
        self.calls = 0

    def available(self):
        return self._available

    def complete(self, *, system, user, temperature, max_tokens):
        self.calls += 1
        if self.fail:
            raise ProviderError(f"{self.name} boom")
        return self.reply


class FakeTTS:
    def __init__(self, name, *, available=True, fail=False, audio=b"AUDIO"):
        self.name = name
        self._available = available
        self.fail = fail
        self.audio = audio

    def available(self):
        return self._available

    def synthesize(self, *, text, language_code):
        if self.fail:
            raise ProviderError(f"{self.name} boom")
        return self.audio


def _llm(provider, threshold=3, cooldown=10, clock=None):
    return provider, CircuitBreaker(provider.name, failure_threshold=threshold,
                                    cooldown=cooldown, clock=clock or Clock())


def test_primary_serves_when_healthy():
    primary = FakeLLM("openai", reply="primary")
    standby = FakeLLM("anthropic", reply="standby")
    fo = FailoverLLM([_llm(primary), _llm(standby)])
    out = fo.complete(system="s", user="u", temperature=0.3, max_tokens=10)
    assert out == "primary"
    assert standby.calls == 0  # standby untouched


def test_falls_over_to_standby_on_primary_failure():
    primary = FakeLLM("openai", fail=True)
    standby = FakeLLM("anthropic", reply="standby")
    fo = FailoverLLM([_llm(primary), _llm(standby)])
    out = fo.complete(system="s", user="u", temperature=0.3, max_tokens=10)
    assert out == "standby"
    assert primary.calls == 1 and standby.calls == 1


def test_skips_unavailable_standby_without_crashing():
    primary = FakeLLM("openai", fail=True)
    standby = FakeLLM("anthropic", available=False)  # e.g. unset ANTHROPIC_API_KEY
    fo = FailoverLLM([_llm(primary), _llm(standby)])
    with pytest.raises(AllProvidersFailed) as ei:
        fo.complete(system="s", user="u", temperature=0.3, max_tokens=10)
    assert standby.calls == 0
    assert ("anthropic", "unavailable") in ei.value.errors


def test_all_failed_raises_with_reasons():
    primary = FakeLLM("openai", fail=True)
    standby = FakeLLM("anthropic", fail=True)
    fo = FailoverLLM([_llm(primary), _llm(standby)])
    with pytest.raises(AllProvidersFailed) as ei:
        fo.complete(system="s", user="u", temperature=0.3, max_tokens=10)
    names = [n for n, _ in ei.value.errors]
    assert names == ["openai", "anthropic"]


def test_circuit_opens_then_fast_fails_to_standby():
    clock = Clock()
    primary = FakeLLM("openai", fail=True)
    standby = FakeLLM("anthropic", reply="standby")
    p_pair = _llm(primary, threshold=2, cooldown=30, clock=clock)
    fo = FailoverLLM([p_pair, _llm(standby)])
    breaker = p_pair[1]

    # Two failures open the primary breaker; each falls over to standby.
    for _ in range(2):
        assert fo.complete(system="s", user="u", temperature=0.3, max_tokens=10) == "standby"
    assert breaker.state is CircuitState.OPEN

    # Now the breaker fast-fails: primary's complete() is NOT invoked again.
    before = primary.calls
    assert fo.complete(system="s", user="u", temperature=0.3, max_tokens=10) == "standby"
    assert primary.calls == before  # circuit short-circuited the primary

    # After cooldown the breaker half-opens and, on a now-healthy primary, recovers.
    primary.fail = False
    clock.advance(30)
    assert fo.complete(system="s", user="u", temperature=0.3, max_tokens=10) == "reply"
    assert breaker.state is CircuitState.CLOSED


def test_tts_failover():
    fo = FailoverTTS([
        _llm(FakeTTS("polly", fail=True)),
        _llm(FakeTTS("openai", audio=b"STANDBY")),
    ])
    assert fo.synthesize(text="hola", language_code="es") == b"STANDBY"

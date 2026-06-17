"""Tests for the NLU failover (app.nlu): local centroid classifier + top-K LLM
disambiguation, and the Dialogflow→local router with a circuit breaker. Mocks
only — no Dialogflow, no embeddings, no LLM, no keys.
"""
from unittest.mock import MagicMock

import app.voice_pipeline as vp
from app.circuit import CircuitBreaker, CircuitState
from app.nlu.classifier import ClassifyResult, LocalIntentClassifier
from app.nlu.router import NLURouter

REF = {"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]}


def fake_embed(text):
    t = (text or "").lower().strip()
    if t.startswith("a"):
        return [1.0, 0.0, 0.0]
    if t.startswith("b"):
        return [0.0, 1.0, 0.0]
    if t.startswith("c"):
        return [0.0, 0.0, 1.0]
    return [0.6, 0.6, 0.6]  # roughly equidistant → not confident


def _clf(llm_fn=None):
    return LocalIntentClassifier(
        embed_fn=fake_embed,
        llm_fn=llm_fn or (lambda s, u: ""),
        reference_phrases=REF,
        top_k=3,
        confident_distance=0.35,
    )


# ---------- classifier ----------

def test_confident_centroid_skips_llm():
    llm = MagicMock(return_value="SHOULD NOT BE CALLED")
    res = _clf(llm).classify("a question about something")
    assert res.intent == "A" and res.method == "centroid"
    llm.assert_not_called()


def test_ambiguous_uses_llm_among_topk():
    llm = MagicMock(return_value="B")
    res = _clf(llm).classify("xx totally ambiguous utterance")
    assert res.method == "llm" and res.intent == "B"
    # LLM saw only the top-K candidate intents
    _, user = llm.call_args[0]
    assert "A" in user and "B" in user and "C" in user
    assert len(res.candidates) == 3


def test_llm_garbage_falls_back_to_nearest_centroid():
    llm = MagicMock(return_value="not-an-intent")
    res = _clf(llm).classify("xx ambiguous")
    # nearest centroid is the first-ranked (stable order) when all equidistant
    assert res.intent == res.candidates[0][0]
    assert res.method == "llm"


def test_llm_error_falls_back_to_centroid():
    def boom(s, u):
        raise RuntimeError("llm down")

    res = _clf(boom).classify("xx ambiguous")
    assert res.intent == res.candidates[0][0]  # nearest centroid


def test_empty_utterance_returns_none():
    res = _clf().classify("   ")
    assert res.intent is None and res.method == "none"


def test_centroids_built_once():
    spy = MagicMock(side_effect=fake_embed)
    clf = LocalIntentClassifier(embed_fn=spy, reference_phrases=REF, top_k=3, confident_distance=0.35)
    clf.classify("a one")
    calls_after_first = spy.call_count
    clf.classify("b two")
    # only one extra embed call (the utterance); centroids cached
    assert spy.call_count == calls_after_first + 1


# ---------- router ----------

class FakeClassifier:
    def __init__(self, intent="LOCAL"):
        self.intent = intent
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        return ClassifyResult(self.intent, [], "centroid")


def _breaker(threshold=2):
    return CircuitBreaker("nlu:test", failure_threshold=threshold, cooldown=999)


def test_router_primary_serves_when_healthy():
    primary = MagicMock(return_value=("booking", "hello from dialogflow"))
    r = NLURouter(primary, classifier=FakeClassifier(), breaker=_breaker())
    res = r.detect("hi", "es", "s1")
    assert res.source == "dialogflow" and res.intent == "booking" and res.reply == "hello from dialogflow"


def test_router_fails_over_to_local():
    def boom(*a):
        raise RuntimeError("dialogflow down")

    clf = FakeClassifier("pool_y_jacuzzi")
    r = NLURouter(boom, classifier=clf, breaker=_breaker())
    res = r.detect("how warm is the pool", "en", "s1")
    assert res.source == "local" and res.intent == "pool_y_jacuzzi" and res.reply == ""
    assert clf.calls == 1


def test_router_local_classifier_failure_degrades(monkeypatch):
    # correlated outage: Dialogflow down AND the local classifier raises.
    def boom(*a):
        raise RuntimeError("dialogflow down")

    bad = MagicMock()
    bad.classify.side_effect = RuntimeError("embeddings down")
    r = NLURouter(boom, classifier=bad, breaker=_breaker())
    res = r.detect("x", "es", "s")
    assert res.source == "local" and res.intent == "" and res.reply == ""


def test_classifier_tolerates_malformed_vector_dims():
    def jagged_embed(text):
        # one phrase returns a wrong-dimension vector; must be skipped, not crash
        return [1.0, 0.0] if text != "a2" else [1.0, 0.0, 0.0]

    clf = LocalIntentClassifier(embed_fn=jagged_embed, llm_fn=lambda s, u: "",
                                reference_phrases={"A": ["a1", "a2"]}, top_k=1, confident_distance=0.5)
    res = clf.classify("a query")
    assert res.intent == "A"


def test_router_circuit_opens_then_fast_fails_to_local():
    primary = MagicMock(side_effect=RuntimeError("down"))
    clf = FakeClassifier()
    breaker = _breaker(threshold=2)
    r = NLURouter(primary, classifier=clf, breaker=breaker)

    r.detect("x", "es", "s")  # fail 1 → local
    r.detect("x", "es", "s")  # fail 2 → local, breaker opens
    assert breaker.state is CircuitState.OPEN
    before = primary.call_count
    r.detect("x", "es", "s")  # circuit open → fast-fail to local, primary not called
    assert primary.call_count == before
    assert clf.calls == 3


# ---------- voice_pipeline integration ----------

def test_voice_failover_knowledge_intent_answers_via_rag(monkeypatch):
    from app.nlu.router import NLUResult

    fake_router = MagicMock()
    fake_router.detect.return_value = NLUResult("pool_y_jacuzzi", "", "local")
    monkeypatch.setattr(vp, "_get_router", lambda: fake_router)

    import app.rag as rag
    monkeypatch.setattr(rag, "answer_with_rag", lambda q, language, topic_filter: f"RAG[{topic_filter}]")

    intent, reply = vp.detect_intent("how warm is the pool", "en", "s1")
    assert intent == "pool_y_jacuzzi"
    assert reply == "RAG[pool]"


def test_voice_failover_nonknowledge_intent_empty_reply(monkeypatch):
    from app.nlu.router import NLUResult

    fake_router = MagicMock()
    fake_router.detect.return_value = NLUResult("solicitar_codigo_puerta", "", "local")
    monkeypatch.setattr(vp, "_get_router", lambda: fake_router)

    import app.rag as rag
    called = MagicMock()
    monkeypatch.setattr(rag, "answer_with_rag", called)

    intent, reply = vp.detect_intent("door code please", "en", "s1")
    assert intent == "solicitar_codigo_puerta" and reply == ""
    called.assert_not_called()


def test_voice_primary_reply_passthrough(monkeypatch):
    from app.nlu.router import NLUResult

    fake_router = MagicMock()
    fake_router.detect.return_value = NLUResult("pool_y_jacuzzi", "DF reply", "dialogflow")
    monkeypatch.setattr(vp, "_get_router", lambda: fake_router)

    import app.rag as rag
    called = MagicMock()
    monkeypatch.setattr(rag, "answer_with_rag", called)

    intent, reply = vp.detect_intent("x", "en", "s1")
    assert reply == "DF reply"
    called.assert_not_called()  # primary reply used; no RAG fallback

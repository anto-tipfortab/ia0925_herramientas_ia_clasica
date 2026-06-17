"""
Local embedding intent classifier — the NLU failover when Dialogflow is down.

Builds one centroid per intent from the reference phrases in ``intents.yaml``
(embedded once, multilingual), then for an utterance:

1. embed it,
2. rank intents by cosine distance to their centroid (nearest-centroid),
3. if the top-1 is confident (distance ≤ threshold) → use it,
4. else ask the LLM provider to disambiguate among ONLY the top-K candidates
   (a small, cheap prompt).

Reference vectors are namespaced by ``TENANT_ID`` so a future multi-tenant build
keeps one classifier per tenant. Embeddings + LLM are reached through the Lane-1
providers (injectable for tests — no network, no keys).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from .. import config
from ..cache import cosine_distance
from ..providers import get_embedding_failover, get_llm_failover

log = logging.getLogger("funstay.nlu")

INTENTS_FILE = Path(__file__).resolve().parent / "intents.yaml"

EmbedFn = Callable[[str], List[float]]
LLMFn = Callable[[str, str], str]


def _default_embed(text: str) -> List[float]:
    return get_embedding_failover().embed(text)


def _default_llm(system: str, user: str) -> str:
    return get_llm_failover().complete(system=system, user=user, temperature=0.0, max_tokens=24)


def load_reference_phrases(path: Path = INTENTS_FILE) -> Dict[str, List[str]]:
    """intent -> flat list of ES+EN reference phrases."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    intents = data.get("intents", {})
    phrases: Dict[str, List[str]] = {}
    for intent, by_lang in intents.items():
        flat: List[str] = []
        for lang_phrases in (by_lang or {}).values():
            flat.extend(lang_phrases or [])
        if flat:
            phrases[intent] = flat
    return phrases


@dataclass
class ClassifyResult:
    intent: Optional[str]
    candidates: List[Tuple[str, float]] = field(default_factory=list)  # (intent, distance), nearest first
    method: str = "none"  # "centroid" | "llm" | "none"


class LocalIntentClassifier:
    def __init__(
        self,
        *,
        tenant_id: str = config.TENANT_ID,
        embed_fn: EmbedFn = _default_embed,
        llm_fn: LLMFn = _default_llm,
        reference_phrases: Optional[Dict[str, List[str]]] = None,
        top_k: int = None,  # type: ignore[assignment]
        confident_distance: float = None,  # type: ignore[assignment]
    ) -> None:
        self.tenant_id = tenant_id
        self._embed = embed_fn
        self._llm = llm_fn
        self._phrases = reference_phrases if reference_phrases is not None else load_reference_phrases()
        self.top_k = top_k if top_k is not None else config.NLU_TOP_K
        self.confident_distance = (
            confident_distance if confident_distance is not None else config.NLU_CONFIDENT_DISTANCE
        )
        self._centroids: Optional[Dict[str, List[float]]] = None
        self._lock = threading.Lock()

    @staticmethod
    def _mean(vectors: List[List[float]]) -> List[float]:
        """Mean vector, ignoring any vector whose dimension differs from the first
        (a malformed embedding must not crash the whole centroid build)."""
        if not vectors:
            return []
        dim = len(vectors[0])
        acc = [0.0] * dim
        n = 0
        for v in vectors:
            if len(v) != dim:
                continue
            for i in range(dim):
                acc[i] += v[i]
            n += 1
        if n == 0:
            return []
        return [x / n for x in acc]

    def centroids(self) -> Dict[str, List[float]]:
        """Per-intent centroid embeddings, built once and cached (tenant-scoped).

        Double-checked locking so concurrent first-failover requests don't each
        re-embed every phrase while a provider is already stressed.
        """
        if self._centroids is None:
            with self._lock:
                if self._centroids is None:
                    built: Dict[str, List[float]] = {}
                    for intent, phrases in self._phrases.items():
                        vecs = [self._embed(p) for p in phrases]
                        centroid = self._mean(vecs)
                        if centroid:
                            built[intent] = centroid
                    self._centroids = built
                    log.info("nlu[%s]: built %d intent centroids", self.tenant_id, len(built))
        return self._centroids

    def _rank(self, utterance: str) -> List[Tuple[str, float]]:
        emb = self._embed(utterance)
        ranked = [(intent, cosine_distance(emb, c)) for intent, c in self.centroids().items()]
        ranked.sort(key=lambda x: x[1])
        return ranked

    def _disambiguate(self, utterance: str, candidates: List[str]) -> Optional[str]:
        """Ask the LLM to pick exactly one intent from the top-K shortlist."""
        system = (
            "You are an intent classifier for a vacation-rental voice concierge. "
            "Choose the single best intent for the user's message from the provided list. "
            "Reply with ONLY the intent name, exactly as written, and nothing else."
        )
        user = (
            f"User message: {utterance}\n"
            f"Candidate intents: {', '.join(candidates)}\n"
            "Answer with exactly one intent name from the list."
        )
        try:
            reply = self._llm(system, user).strip()
        except Exception as e:
            log.warning("nlu: LLM disambiguation failed (%s); using nearest centroid", e)
            return None
        # Token-exact match (not substring): an intent name must appear as a whole
        # token in the reply. If the reply echoes the whole list (multiple matches)
        # we can't tell which it picked → fall back to the nearest centroid rather
        # than guessing the first one.
        tokens = set(re.findall(r"[a-z0-9_]+", reply.lower()))
        matches = [c for c in candidates if c.lower() in tokens]
        if len(matches) == 1:
            return matches[0]
        log.warning("nlu: LLM reply %r gave %d candidate matches; using nearest centroid",
                    reply, len(matches))
        return None

    def classify(self, utterance: str) -> ClassifyResult:
        utterance = (utterance or "").strip()
        if not utterance or not self.centroids():
            return ClassifyResult(None, [], "none")
        ranked = self._rank(utterance)
        top_k = ranked[: self.top_k]
        best_intent, best_dist = top_k[0]
        if best_dist <= self.confident_distance:
            return ClassifyResult(best_intent, top_k, "centroid")
        chosen = self._disambiguate(utterance, [i for i, _ in top_k]) or best_intent
        return ClassifyResult(chosen, top_k, "llm")


_classifier: Optional[LocalIntentClassifier] = None


def get_classifier() -> LocalIntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LocalIntentClassifier()
    return _classifier


def reset() -> None:
    global _classifier
    _classifier = None

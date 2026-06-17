"""
NLU router: Dialogflow PRIMARY → local-classifier FAILOVER.

The Dialogflow call is wrapped in a circuit breaker (its own timeout lives in the
caller's primary_fn). When the breaker is open or the call fails, we fall over to
the local embedding classifier. The breaker is registered in the shared provider
registry so /ready (Lane 5) reports it alongside the model-provider breakers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from ..circuit import CircuitBreaker, CircuitOpenError
from ..providers import get_circuit
from .classifier import LocalIntentClassifier, get_classifier

log = logging.getLogger("funstay.nlu.router")

# primary_fn(text, language_code, session_id) -> (intent_name, reply_text)
PrimaryFn = Callable[[str, str, str], Tuple[str, str]]

NLU_CIRCUIT = "nlu:dialogflow"


@dataclass
class NLUResult:
    intent: str
    reply: str
    source: str  # "dialogflow" | "local"


class NLURouter:
    def __init__(
        self,
        primary_fn: PrimaryFn,
        *,
        classifier: Optional[LocalIntentClassifier] = None,
        breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._primary = primary_fn
        self._classifier = classifier if classifier is not None else get_classifier()
        self._breaker = breaker if breaker is not None else get_circuit(NLU_CIRCUIT)

    def detect(self, text: str, language_code: str, session_id: str) -> NLUResult:
        try:
            intent, reply = self._breaker.call(
                lambda: self._primary(text, language_code, session_id)
            )
            return NLUResult(intent or "", reply or "", "dialogflow")
        except CircuitOpenError:
            log.warning("nlu: Dialogflow circuit open; using local classifier")
        except Exception as e:
            log.warning("nlu: Dialogflow failed (%s); using local classifier", e)

        # The local path must also be fail-soft: during a correlated outage the
        # embedding provider may be down too. A classifier failure degrades to an
        # empty intent (the pipeline substitutes a generic reply) rather than
        # turning the failover into a second crash surface.
        try:
            result = self._classifier.classify(text)
        except Exception as e:
            log.warning("nlu: local classifier failed (%s); no intent", e)
            return NLUResult("", "", "local")
        log.info("nlu(local): intent=%s method=%s", result.intent, result.method)
        # Local path yields an intent only; reply is produced downstream (a
        # knowledge intent is answered from the cache-first RAG path; other
        # intents fall back to a generic message in the voice pipeline).
        return NLUResult(result.intent or "", "", "local")

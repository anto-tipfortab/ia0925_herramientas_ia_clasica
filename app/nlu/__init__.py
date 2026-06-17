"""
NLU failover package.

Dialogflow stays the PRIMARY intent engine; this package adds a circuit breaker
and a local embedding classifier (+ top-K LLM disambiguation) as the failover.
"""
from .classifier import ClassifyResult, LocalIntentClassifier, get_classifier
from .router import NLU_CIRCUIT, NLUResult, NLURouter

__all__ = [
    "ClassifyResult",
    "LocalIntentClassifier",
    "get_classifier",
    "NLURouter",
    "NLUResult",
    "NLU_CIRCUIT",
]

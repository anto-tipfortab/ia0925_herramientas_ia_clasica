"""
Readiness probe (Lane 5).

`/health` is liveness (the process is up). `/ready` is readiness: it inspects the
real dependencies — ChromaDB reachability and every provider/NLU circuit breaker —
and reports 200 ready / 503 degraded with a per-dependency body so a load balancer
or operator can route away from a degraded replica. The service still serves the
common/cached path while degraded; readiness just signals "not fully healthy".
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

from . import config
from .providers import all_circuits, get_circuit

log = logging.getLogger("funstay.health")


def check_chroma() -> Dict[str, object]:
    """Probe ChromaDB by counting the knowledge collection (cheap, local).

    The raw error is logged but NOT returned — the body reports only the
    exception type so an unauthenticated probe can't leak a filesystem path.
    """
    try:
        from .rag import _get_collection

        count = _get_collection().count()
        return {"reachable": True, "documents": count}
    except Exception as e:  # any failure → not reachable
        log.warning("readiness: chroma unreachable: %s", e)
        return {"reachable": False, "error": type(e).__name__}


def _ensure_nlu_circuit() -> None:
    """Register the NLU breaker so /ready reports it even before the first voice turn."""
    try:
        from .nlu.router import NLU_CIRCUIT

        get_circuit(NLU_CIRCUIT)
    except Exception:  # nlu optional; never let readiness fail on it
        pass


def readiness() -> Tuple[bool, Dict[str, object]]:
    """Return ``(ok, body)``. ok is False (→ 503) if Chroma is unreachable or any
    circuit is OPEN."""
    _ensure_nlu_circuit()
    chroma = check_chroma()
    # The probe must report degraded, never crash: a failure building/snapshotting
    # the circuits is itself a degraded signal, not a 500.
    circuits_ok = True
    circuits: Dict[str, object] = {}
    try:
        circuits = {name: cb.snapshot() for name, cb in all_circuits().items()}
    except Exception as e:
        log.warning("readiness: circuit inspection failed: %s", e)
        circuits_ok = False
    open_circuits = sorted(
        name for name, snap in circuits.items()
        if isinstance(snap, dict) and snap.get("state") == "open"
    )
    ok = bool(chroma["reachable"]) and circuits_ok and not open_circuits
    body: Dict[str, object] = {
        "status": "ready" if ok else "degraded",
        "tenant": config.TENANT_ID,
        "dependencies": {"chroma": chroma, "circuits": circuits},
        "open_circuits": open_circuits,
    }
    return ok, body

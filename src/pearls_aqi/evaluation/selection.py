"""Transparent model selection and promotion gating.

Selection is deterministic and fully logged: lowest validation MAE, tie-broken
by hazardous-period MAE, then by artifact size (smaller = cheaper/more reliable
to serve). Promotion adds guardrails so a freshly-trained model only replaces the
approved one when it is meaningfully better and passes smoke tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"promote": self.promote, "reasons": self.reasons}


def _finite(value: float | None) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def select_best_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best candidate by validation MAE → hazard MAE → artifact size.

    Each result must contain ``name`` and ``validation`` (a metrics dict with at
    least ``mae``). Optional: ``hazard`` (``hazard_mae``) and ``artifact_size``.
    Returns the winning result dict augmented with a ``selection_rationale``.
    """
    usable = [r for r in results if _finite(r.get("validation", {}).get("mae"))]
    if not usable:
        raise ValueError("No candidate produced a finite validation MAE.")

    def sort_key(r: dict[str, Any]) -> tuple[float, float, float]:
        val_mae = r["validation"]["mae"]
        hazard_mae = r.get("hazard", {}).get("hazard_mae")
        hazard_key = hazard_mae if _finite(hazard_mae) else float("inf")
        size = r.get("artifact_size") or float("inf")
        return (val_mae, hazard_key, size)

    ranked = sorted(usable, key=sort_key)
    best = ranked[0]
    best = {**best}
    best["selection_rationale"] = (
        f"Lowest validation MAE ({best['validation']['mae']:.3f}) among "
        f"{len(usable)} candidate(s); tie-broken by hazard MAE then artifact size."
    )
    return best


def evaluate_promotion(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    minimum_mae_improvement_percent: float = 1.0,
    maximum_hazard_mae_degradation_percent: float = 2.0,
    smoke_ok: bool = True,
    require_smoke_test: bool = True,
    artifacts_complete: bool = True,
) -> PromotionDecision:
    """Decide whether ``candidate`` should replace ``incumbent`` (the approved model).

    ``candidate``/``incumbent`` carry ``validation.mae`` and (optionally)
    ``hazard.hazard_mae``. If there is no incumbent, the candidate is promoted as
    long as its metrics are valid and smoke tests pass (bootstrap case).
    """
    reasons: list[str] = []

    cand_mae = candidate.get("validation", {}).get("mae")
    if not _finite(cand_mae):
        return PromotionDecision(False, ["Candidate validation MAE is not finite."])
    if require_smoke_test and not smoke_ok:
        return PromotionDecision(False, ["Smoke test failed."])
    if not artifacts_complete:
        return PromotionDecision(False, ["Model artifacts incomplete."])

    if incumbent is None:
        return PromotionDecision(
            True, ["No approved model exists; promoting first valid candidate."]
        )

    inc_mae = incumbent.get("validation", {}).get("mae")
    if not _finite(inc_mae):
        return PromotionDecision(True, ["Incumbent metrics invalid; promoting candidate."])

    improvement_pct = (inc_mae - cand_mae) / inc_mae * 100 if inc_mae else 0.0
    if improvement_pct < minimum_mae_improvement_percent:
        reasons.append(
            f"MAE improvement {improvement_pct:.2f}% < required "
            f"{minimum_mae_improvement_percent:.2f}%."
        )

    cand_hazard = candidate.get("hazard", {}).get("hazard_mae")
    inc_hazard = incumbent.get("hazard", {}).get("hazard_mae")
    if _finite(cand_hazard) and _finite(inc_hazard) and inc_hazard > 0:
        hazard_degradation_pct = (cand_hazard - inc_hazard) / inc_hazard * 100
        if hazard_degradation_pct > maximum_hazard_mae_degradation_percent:
            reasons.append(
                f"Hazard MAE degraded {hazard_degradation_pct:.2f}% > allowed "
                f"{maximum_hazard_mae_degradation_percent:.2f}%."
            )

    if reasons:
        return PromotionDecision(False, reasons)
    return PromotionDecision(
        True,
        [
            f"MAE improved {improvement_pct:.2f}% (>= {minimum_mae_improvement_percent:.2f}%) "
            "and hazard performance within tolerance."
        ],
    )

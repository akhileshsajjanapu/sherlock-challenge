"""Human-readable explanations for identification decisions."""

from __future__ import annotations

from sherlock.models import IdentificationResult, ParticipantScore, SignalContribution


def explain_participant(signals: list[SignalContribution]) -> list[str]:
    """Top contributing signals for a participant."""
    sorted_signals = sorted(signals, key=lambda s: abs(s.weighted_score), reverse=True)
    lines: list[str] = []
    for sig in sorted_signals[:4]:
        direction = "supports" if sig.raw_score > 0 else "against"
        if abs(sig.raw_score) < 0.05:
            continue
        lines.append(f"  • {sig.signal_name}: {direction} candidate ({sig.explanation})")
    return lines


def build_explanation(
    rankings: list[ParticipantScore],
    confidence: float,
    status: str,
    margin: float,
) -> tuple[str, list[str]]:
    """Build overall explanation and evidence summary."""
    if status == "no_participants":
        return "No participants in meeting yet.", []

    if not rankings:
        return "Unable to rank participants.", []

    top = rankings[0]
    evidence: list[str] = explain_participant(top.signals)

    if status == "uncertain":
        header = (
            f"UNCERTAIN: Top candidate is '{top.display_name}' "
            f"at {confidence:.0%} confidence, but margin over second place "
            f"is only {margin:.0%}. Waiting for more evidence."
        )
    else:
        header = (
            f"Identified '{top.display_name}' as candidate "
            f"with {confidence:.0%} confidence."
        )

    if len(rankings) > 1:
        second = rankings[1]
        header += (
            f" Next closest: '{second.display_name}' "
            f"({second.confidence:.0%})."
        )

    return header, evidence

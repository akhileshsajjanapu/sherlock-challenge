"""Evidence fusion — combine weak signals into confidence scores."""

from __future__ import annotations

import math

from sherlock.models import ParticipantScore, SignalContribution


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def fuse_signals(
    participant_id: str,
    display_name: str,
    signals: list[SignalContribution],
    prior_confidence: float = 0.5,
    ema_alpha: float = 0.3,
) -> tuple[float, float]:
    """
    Fuse signal contributions into a confidence score.

    Returns (new_confidence, raw_logit) where confidence is in [0, 1].
    Uses weighted sum -> sigmoid, then EMA with prior for temporal smoothing.
    """
    if not signals:
        return prior_confidence, 0.0

    total_weight = sum(abs(s.weight) for s in signals)
    if total_weight == 0:
        return prior_confidence, 0.0

    weighted_sum = sum(s.weighted_score for s in signals)
    # Normalize by total weight to get average directional evidence
    normalized = weighted_sum / total_weight
    # Scale to logit space (stronger evidence = higher magnitude)
    logit = normalized * 4.0
    instant_confidence = sigmoid(logit)

    # EMA smoothing — continue learning as more data arrives
    new_confidence = ema_alpha * instant_confidence + (1 - ema_alpha) * prior_confidence
    return new_confidence, logit


def rank_participants(
    scores: list[ParticipantScore],
) -> list[ParticipantScore]:
    return sorted(scores, key=lambda s: s.confidence, reverse=True)


def compute_margin(top: float, second: float) -> float:
    """Gap between top two candidates — low margin = ambiguous."""
    return top - second

"""Exclude known interviewers from candidate consideration."""

from __future__ import annotations

from rapidfuzz import fuzz

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor


class InterviewerExclusionSignal(SignalExtractor):
    name = "interviewer_exclusion"
    weight = 0.18
    description = "Penalize participants matching known interviewer names"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}
        interviewers = metadata.interviewer_names or []

        for pid, participant in participants.items():
            if not interviewers:
                results[pid] = SignalContribution(
                    signal_name=self.name,
                    weight=self.weight,
                    raw_score=0.0,
                    weighted_score=0.0,
                    explanation="No interviewer names provided — neutral",
                )
                continue

            best_ratio = max(
                (
                    fuzz.token_sort_ratio(
                        participant.display_name.lower(), iv.lower()
                    )
                    / 100
                    for iv in interviewers
                ),
                default=0.0,
            )

            if best_ratio >= 0.80:
                raw = -0.95
                explanation = (
                    f"Display name matches known interviewer "
                    f"({best_ratio:.0%}) — unlikely candidate"
                )
            elif best_ratio >= 0.55:
                raw = -0.5
                explanation = (
                    f"Partial match with interviewer name ({best_ratio:.0%})"
                )
            else:
                raw = 0.2
                explanation = "Does not match any known interviewer"

            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

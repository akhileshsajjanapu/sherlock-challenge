"""Webcam usage patterns — candidates typically keep camera on."""

from __future__ import annotations

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor


class WebcamBehaviorSignal(SignalExtractor):
    name = "webcam_behavior"
    weight = 0.08
    description = "Candidates usually have webcam enabled during interviews"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}

        for pid, participant in participants.items():
            if participant.webcam_on:
                raw = 0.4
                explanation = "Webcam is on — consistent with interview candidate"
            else:
                raw = -0.2
                explanation = "Webcam off — weak negative signal (not definitive)"

            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

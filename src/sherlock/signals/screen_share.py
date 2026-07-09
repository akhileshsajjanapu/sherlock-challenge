"""Screen share behavior — candidates may share portfolio/code."""

from __future__ import annotations

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor


class ScreenShareSignal(SignalExtractor):
    name = "screen_share"
    weight = 0.05
    description = "Candidates sometimes share screen for coding/portfolio demos"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}
        sharers = [pid for pid, p in participants.items() if p.screen_sharing]

        for pid, participant in participants.items():
            if participant.screen_sharing:
                raw = 0.35
                explanation = "Currently screen sharing — common for candidate demos"
            elif len(sharers) == 1 and pid not in sharers:
                raw = -0.15
                explanation = "Another participant is screen sharing"
            else:
                raw = 0.0
                explanation = "No screen share activity"

            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

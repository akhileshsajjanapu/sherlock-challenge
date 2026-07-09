"""Penalize silent participants — likely observers."""

from __future__ import annotations

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor


class ObserverSilenceSignal(SignalExtractor):
    name = "observer_silence"
    weight = 0.05
    description = "Silent participants are likely observers, not candidates"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}
        transcript_speakers = {s.participant_id for s in transcript}

        for pid, participant in participants.items():
            spoke_in_transcript = pid in transcript_speakers
            has_speech = participant.total_speaking_seconds > 3

            if not spoke_in_transcript and not has_speech:
                raw = -0.6
                explanation = "Completely silent — likely silent observer"
            elif participant.total_speaking_seconds < 10 and not spoke_in_transcript:
                raw = -0.3
                explanation = "Minimal speech activity — possible observer"
            elif has_speech:
                raw = 0.2
                explanation = "Active participant — not a silent observer"
            else:
                raw = 0.0
                explanation = "Insufficient data"

            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

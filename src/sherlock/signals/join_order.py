"""Join order heuristics — candidate often joins near scheduled time."""

from __future__ import annotations

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor


class JoinOrderSignal(SignalExtractor):
    name = "join_order"
    weight = 0.05
    description = "Temporal join patterns relative to interviewers"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}

        join_times = [
            (pid, p.joined_at)
            for pid, p in participants.items()
            if p.joined_at is not None
        ]
        join_times.sort(key=lambda x: x[1])  # type: ignore[arg-type]

        if len(join_times) <= 1:
            for pid in participants:
                results[pid] = SignalContribution(
                    signal_name=self.name,
                    weight=self.weight,
                    raw_score=0.0,
                    weighted_score=0.0,
                    explanation="Single participant — no join order signal",
                )
            return results

        first_joiner = join_times[0][0]
        last_joiner = join_times[-1][0]

        for pid, participant in participants.items():
            if pid == last_joiner and len(join_times) >= 2:
                raw = 0.25
                explanation = "Joined last — candidates often join after interviewers"
            elif pid == first_joiner and len(join_times) >= 3:
                raw = -0.1
                explanation = "Joined first — may be host/interviewer"
            else:
                raw = 0.0
                explanation = "Middle join order — neutral"

            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

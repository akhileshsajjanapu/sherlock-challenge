"""Penalize generic device names like 'MacBook Pro'."""

from __future__ import annotations

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor
from sherlock.signals.name_match import GENERIC_DEVICE_NAMES


class DeviceNameSignal(SignalExtractor):
    name = "device_name"
    weight = 0.07
    description = "Generic device names provide no identity — rely on other signals"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}

        generic_count = sum(
            1
            for p in participants.values()
            if p.display_name.lower().strip() in GENERIC_DEVICE_NAMES
        )

        for pid, participant in participants.items():
            name_lower = participant.display_name.lower().strip()
            if name_lower in GENERIC_DEVICE_NAMES:
                if generic_count == len(participants):
                    raw = 0.0
                    explanation = (
                        f"Generic name '{participant.display_name}' but all "
                        "participants have generic names — defer to other signals"
                    )
                else:
                    raw = -0.4
                    explanation = (
                        f"Generic device name '{participant.display_name}' — "
                        "identity must come from behavior/metadata"
                    )
            else:
                raw = 0.15
                explanation = (
                    f"Human-readable name '{participant.display_name}' "
                    "— slight positive identity signal"
                )

            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

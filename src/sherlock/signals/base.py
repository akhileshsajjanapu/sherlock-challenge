"""Base class for weak signal extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)


class SignalExtractor(ABC):
    """Extracts a weak signal score for each participant."""

    name: str
    weight: float
    description: str

    @abstractmethod
    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        """
        Return per-participant signal contributions.

        raw_score is in [-1, 1]:
          +1 = strong evidence this participant IS the candidate
          -1 = strong evidence this participant is NOT the candidate
           0 = neutral / no evidence
        """

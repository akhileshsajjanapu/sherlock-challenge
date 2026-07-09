"""Fuzzy name and email matching against candidate metadata."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor

GENERIC_DEVICE_NAMES = {
    "macbook pro",
    "macbook air",
    "iphone",
    "ipad",
    "windows",
    "android",
    "guest",
    "participant",
    "user",
    "phone",
    "laptop",
    "desktop",
    "zoom",
    "teams",
    "meet",
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _email_local_part(email: str) -> str:
    return email.split("@")[0].lower().replace(".", " ")


class NameMatchSignal(SignalExtractor):
    name = "name_match"
    weight = 0.22
    description = "Fuzzy match of display name against candidate name/email"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}
        candidate_names: list[str] = []

        if metadata.candidate_name:
            candidate_names.append(metadata.candidate_name)
        if metadata.candidate_email:
            candidate_names.append(_email_local_part(metadata.candidate_email))
            candidate_names.append(metadata.candidate_email.split("@")[0])

        if not candidate_names:
            for pid, p in participants.items():
                results[pid] = SignalContribution(
                    signal_name=self.name,
                    weight=self.weight,
                    raw_score=0.0,
                    weighted_score=0.0,
                    explanation="No candidate name/email in metadata — neutral",
                )
            return results

        for pid, participant in participants.items():
            names_to_check = [participant.display_name] + participant.name_history
            best_ratio = 0.0
            best_match_target = ""

            for display in names_to_check:
                display_norm = display.lower().strip()
                if display_norm in GENERIC_DEVICE_NAMES:
                    continue
                for target in candidate_names:
                    ratio = max(
                        fuzz.token_sort_ratio(display.lower(), target.lower()) / 100,
                        fuzz.partial_ratio(display.lower(), target.lower()) / 100,
                    )
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match_target = target

            if best_ratio >= 0.85:
                raw = 0.6 + (best_ratio - 0.85) * 2.67  # maps 0.85-1.0 -> 0.6-1.0
                explanation = (
                    f"Display name '{participant.display_name}' closely matches "
                    f"candidate '{best_match_target}' ({best_ratio:.0%})"
                )
            elif best_ratio >= 0.55:
                raw = (best_ratio - 0.55) * 2.0  # maps 0.55-0.85 -> 0-0.6
                explanation = (
                    f"Partial name match with candidate '{best_match_target}' "
                    f"({best_ratio:.0%}) — possible nickname"
                )
            elif participant.display_name.lower().strip() in GENERIC_DEVICE_NAMES:
                raw = -0.3
                explanation = (
                    f"Generic device name '{participant.display_name}' — "
                    "no identity signal"
                )
            else:
                raw = -0.15
                explanation = (
                    f"No meaningful match with candidate name "
                    f"(best: {best_ratio:.0%})"
                )

            raw = max(-1.0, min(1.0, raw))
            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

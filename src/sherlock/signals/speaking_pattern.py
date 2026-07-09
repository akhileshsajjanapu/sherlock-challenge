"""Speaking duration and Q&A dynamics — candidates answer more than they ask."""

from __future__ import annotations

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor

QUESTION_PATTERNS = (
    "can you tell me",
    "could you explain",
    "what is your",
    "how would you",
    "walk me through",
    "describe a time",
    "why did you",
    "tell us about",
    "do you have",
    "?",
)

ANSWER_PATTERNS = (
    "in my previous",
    "at my last",
    "i worked on",
    "i implemented",
    "my experience",
    "i would approach",
    "when i was",
    "we used to",
    "i believe",
    "sure,",
    "so basically",
)


class SpeakingPatternSignal(SignalExtractor):
    name = "speaking_pattern"
    weight = 0.15
    description = "Candidates tend to speak longer and answer rather than ask"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}
        total_speech = sum(p.total_speaking_seconds for p in participants.values())

        # Transcript-based Q&A analysis
        question_counts: dict[str, int] = {pid: 0 for pid in participants}
        answer_counts: dict[str, int] = {pid: 0 for pid in participants}

        for seg in transcript:
            text_lower = seg.text.lower()
            if any(p in text_lower for p in QUESTION_PATTERNS):
                question_counts[seg.participant_id] = (
                    question_counts.get(seg.participant_id, 0) + 1
                )
            if any(p in text_lower for p in ANSWER_PATTERNS):
                answer_counts[seg.participant_id] = (
                    answer_counts.get(seg.participant_id, 0) + 1
                )

        for pid, participant in participants.items():
            raw = 0.0
            parts: list[str] = []

            # Duration ratio (candidates often speak 40-70% in technical interviews)
            if total_speech > 0:
                ratio = participant.total_speaking_seconds / total_speech
                if 0.25 <= ratio <= 0.75:
                    raw += 0.3
                    parts.append(
                        f"Speaking share {ratio:.0%} — typical candidate range"
                    )
                elif ratio > 0.75:
                    raw += 0.1
                    parts.append(
                        f"High speaking share ({ratio:.0%}) — may be solo presenter"
                    )
                elif ratio < 0.05 and participant.total_speaking_seconds < 5:
                    raw -= 0.5
                    parts.append("Nearly silent — likely observer")
                else:
                    parts.append(f"Speaking share {ratio:.0%}")

            # Q&A pattern
            q_count = question_counts.get(pid, 0)
            a_count = answer_counts.get(pid, 0)
            if a_count > q_count and a_count >= 2:
                raw += 0.4
                parts.append(
                    f"Answer-heavy speech ({a_count} answers vs {q_count} questions)"
                )
            elif q_count > a_count and q_count >= 2:
                raw -= 0.4
                parts.append(
                    f"Question-heavy speech ({q_count} questions) — interviewer pattern"
                )

            raw = max(-1.0, min(1.0, raw))
            explanation = "; ".join(parts) if parts else "Insufficient speaking data"

            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation=explanation,
            )
        return results

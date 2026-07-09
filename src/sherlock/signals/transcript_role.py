"""Transcript role analysis — who asks interview questions vs answers them."""

from __future__ import annotations

import os
import re

from sherlock.models import (
    MeetingEvent,
    MeetingMetadata,
    Participant,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals.base import SignalExtractor

INTERVIEWER_PHRASES = [
    r"\b(tell me about|walk me through|can you explain|how would you|what is your experience)\b",
    r"\b(next question|let'?s move on|thank you for joining|we'?ll be in touch)\b",
    r"\b(why did you choose|describe a situation|give me an example)\b",
]

CANDIDATE_PHRASES = [
    r"\b(in my (previous|last|current) (role|job|company|project))\b",
    r"\b(i (worked|built|designed|implemented|led|managed))\b",
    r"\b(my (background|experience|approach|team))\b",
    r"\b(when i was at|at (google|amazon|meta|microsoft|startup))\b",
    r"\b(i would (use|approach|solve|design))\b",
]


class TranscriptRoleSignal(SignalExtractor):
    name = "transcript_role"
    weight = 0.15
    description = "Linguistic role detection from speaker-attributed transcript"

    def score(
        self,
        participants: dict[str, Participant],
        metadata: MeetingMetadata,
        transcript: list[TranscriptSegment],
        events: list[MeetingEvent],
    ) -> dict[str, SignalContribution]:
        results: dict[str, SignalContribution] = {}

        if not transcript:
            for pid in participants:
                results[pid] = SignalContribution(
                    signal_name=self.name,
                    weight=self.weight,
                    raw_score=0.0,
                    weighted_score=0.0,
                    explanation="No transcript available yet",
                )
            return results

        interviewer_hits: dict[str, int] = {pid: 0 for pid in participants}
        candidate_hits: dict[str, int] = {pid: 0 for pid in participants}

        for seg in transcript:
            text = seg.text.lower()
            for pattern in INTERVIEWER_PHRASES:
                if re.search(pattern, text):
                    interviewer_hits[seg.participant_id] = (
                        interviewer_hits.get(seg.participant_id, 0) + 1
                    )
            for pattern in CANDIDATE_PHRASES:
                if re.search(pattern, text):
                    candidate_hits[seg.participant_id] = (
                        candidate_hits.get(seg.participant_id, 0) + 1
                    )

        # Optional LLM boost if API key available
        llm_scores = self._llm_role_scores(participants, transcript)

        for pid in participants:
            iv_hits = interviewer_hits.get(pid, 0)
            cand_hits = candidate_hits.get(pid, 0)
            raw = 0.0
            parts: list[str] = []

            if cand_hits > iv_hits and cand_hits >= 1:
                raw += min(0.7, cand_hits * 0.2)
                parts.append(
                    f"Candidate speech patterns detected ({cand_hits} instances)"
                )
            if iv_hits > cand_hits and iv_hits >= 1:
                raw -= min(0.7, iv_hits * 0.2)
                parts.append(
                    f"Interviewer speech patterns detected ({iv_hits} instances)"
                )

            if pid in llm_scores:
                raw = raw * 0.6 + llm_scores[pid] * 0.4
                parts.append(f"LLM role analysis: {llm_scores[pid]:+.2f}")

            if not parts:
                parts.append("Transcript too short for role classification")

            raw = max(-1.0, min(1.0, raw))
            results[pid] = SignalContribution(
                signal_name=self.name,
                weight=self.weight,
                raw_score=raw,
                weighted_score=raw * self.weight,
                explanation="; ".join(parts),
            )
        return results

    def _llm_role_scores(
        self,
        participants: dict[str, Participant],
        transcript: list[TranscriptSegment],
    ) -> dict[str, float]:
        """Optional OpenAI-based role classification."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or len(transcript) < 3:
            return {}

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            name_map = {pid: p.display_name for pid, p in participants.items()}
            lines = [
                f"[{name_map.get(s.participant_id, s.participant_id)}]: {s.text}"
                for s in transcript[-20:]
            ]
            prompt = (
                "Analyze this interview transcript excerpt. For each speaker, "
                "rate -1 to 1 how likely they are the job CANDIDATE (not interviewer).\n"
                "Return JSON: {\"scores\": {\"speaker_name\": score, ...}}\n\n"
                + "\n".join(lines)
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            import json

            data = json.loads(response.choices[0].message.content or "{}")
            name_to_pid = {p.display_name: pid for pid, p in participants.items()}
            result: dict[str, float] = {}
            for name, score in data.get("scores", {}).items():
                pid = name_to_pid.get(name)
                if pid:
                    result[pid] = max(-1.0, min(1.0, float(score)))
            return result
        except Exception:
            return {}

"""Main candidate identification engine — orchestrates signals and fusion."""

from __future__ import annotations

from datetime import datetime

from sherlock.explainer import build_explanation
from sherlock.fusion import compute_margin, fuse_signals, rank_participants
from sherlock.models import (
    EventType,
    IdentificationResult,
    MeetingEvent,
    MeetingMetadata,
    Participant,
    ParticipantScore,
    SignalContribution,
    TranscriptSegment,
)
from sherlock.signals import get_default_signals
from sherlock.signals.base import SignalExtractor

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.12


class CandidateIdentificationEngine:
    """
    Real-time engine that fuses multiple weak signals to identify
    the interview candidate with confidence scoring and explainability.
    """

    def __init__(
        self,
        metadata: MeetingMetadata | None = None,
        signals: list[SignalExtractor] | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
        ema_alpha: float = 0.3,
    ):
        self.metadata = metadata or MeetingMetadata()
        self.signals = signals or get_default_signals()
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.ema_alpha = ema_alpha

        self.participants: dict[str, Participant] = {}
        self.transcript: list[TranscriptSegment] = []
        self.events: list[MeetingEvent] = []
        self._confidence_history: dict[str, float] = {}
        self._results_history: list[IdentificationResult] = []

    def update_metadata(self, metadata: MeetingMetadata) -> None:
        self.metadata = metadata

    def process_event(self, event: MeetingEvent) -> IdentificationResult:
        """Ingest a meeting event and return updated identification."""
        self.events.append(event)
        self._apply_event(event)
        return self.identify()

    def add_transcript(self, segment: TranscriptSegment) -> IdentificationResult:
        self.transcript.append(segment)
        return self.identify()

    def identify(self, timestamp: datetime | None = None) -> IdentificationResult:
        ts = timestamp or datetime.now()

        if not self.participants:
            result = IdentificationResult(
                timestamp=ts,
                candidate_participant_id=None,
                candidate_display_name=None,
                confidence=0.0,
                status="no_participants",
                rankings=[],
                explanation="No participants in meeting yet.",
            )
            self._results_history.append(result)
            return result

        # Collect all signal contributions per participant
        all_signals: dict[str, list[SignalContribution]] = {
            pid: [] for pid in self.participants
        }

        for signal_extractor in self.signals:
            contributions = signal_extractor.score(
                self.participants,
                self.metadata,
                self.transcript,
                self.events,
            )
            for pid, contrib in contributions.items():
                if pid in all_signals:
                    all_signals[pid].append(contrib)

        # Fuse into confidence scores
        scores: list[ParticipantScore] = []
        for pid, participant in self.participants.items():
            prior = self._confidence_history.get(pid, 0.5)
            confidence, _ = fuse_signals(
                pid,
                participant.display_name,
                all_signals[pid],
                prior_confidence=prior,
                ema_alpha=self.ema_alpha,
            )
            self._confidence_history[pid] = confidence

            is_interviewer = any(
                c.raw_score <= -0.5 and c.signal_name == "interviewer_exclusion"
                for c in all_signals[pid]
            )

            scores.append(
                ParticipantScore(
                    participant_id=pid,
                    display_name=participant.display_name,
                    confidence=confidence,
                    signals=all_signals[pid],
                    is_interviewer=is_interviewer,
                )
            )

        rankings = rank_participants(scores)
        top = rankings[0]
        second_conf = rankings[1].confidence if len(rankings) > 1 else 0.0
        margin = compute_margin(top.confidence, second_conf)

        if top.confidence >= self.confidence_threshold and margin >= self.margin_threshold:
            status = "identified"
            candidate_id = top.participant_id
            candidate_name = top.display_name
            confidence = top.confidence
        elif top.confidence >= 0.4:
            status = "uncertain"
            candidate_id = top.participant_id
            candidate_name = top.display_name
            confidence = top.confidence
        else:
            status = "uncertain"
            candidate_id = None
            candidate_name = None
            confidence = top.confidence

        explanation, evidence = build_explanation(
            rankings, confidence, status, margin
        )

        result = IdentificationResult(
            timestamp=ts,
            candidate_participant_id=candidate_id,
            candidate_display_name=candidate_name,
            confidence=confidence,
            status=status,
            rankings=rankings,
            explanation=explanation,
            evidence_summary=evidence,
        )
        self._results_history.append(result)
        return result

    def get_confidence_timeline(self) -> list[dict]:
        """Return confidence evolution for demo visualization."""
        timeline = []
        for result in self._results_history:
            timeline.append(
                {
                    "timestamp": result.timestamp.isoformat(),
                    "status": result.status,
                    "confidence": result.confidence,
                    "candidate": result.candidate_display_name,
                    "rankings": [
                        {
                            "name": r.display_name,
                            "confidence": r.confidence,
                        }
                        for r in result.rankings
                    ],
                }
            )
        return timeline

    def _apply_event(self, event: MeetingEvent) -> None:
        pid = event.participant_id

        if event.event_type == EventType.JOIN:
            self.participants[pid] = Participant(
                participant_id=pid,
                display_name=event.payload.get("display_name", "Unknown"),
                joined_at=event.timestamp,
                webcam_on=event.payload.get("webcam_on", False),
            )

        elif event.event_type == EventType.LEAVE:
            self.participants.pop(pid, None)
            self._confidence_history.pop(pid, None)

        elif pid in self.participants:
            p = self.participants[pid]

            if event.event_type == EventType.WEBCAM_ON:
                p.webcam_on = True
            elif event.event_type == EventType.WEBCAM_OFF:
                p.webcam_on = False
            elif event.event_type == EventType.SCREEN_SHARE_START:
                p.screen_sharing = True
            elif event.event_type == EventType.SCREEN_SHARE_STOP:
                p.screen_sharing = False
            elif event.event_type == EventType.SPEAKING_START:
                p.is_speaking = True
            elif event.event_type == EventType.SPEAKING_STOP:
                p.is_speaking = False
                duration = event.payload.get("duration_seconds", 0.0)
                p.total_speaking_seconds += duration
            elif event.event_type == EventType.DISPLAY_NAME_CHANGE:
                new_name = event.payload.get("new_name", p.display_name)
                if new_name != p.display_name:
                    p.name_history.append(p.display_name)
                    p.display_name = new_name
            elif event.event_type == EventType.TRANSCRIPT:
                self.transcript.append(
                    TranscriptSegment(
                        participant_id=pid,
                        text=event.payload.get("text", ""),
                        start_time=event.timestamp,
                    )
                )

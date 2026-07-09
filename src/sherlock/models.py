"""Data models for meeting events and identification results."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    JOIN = "join"
    LEAVE = "leave"
    WEBCAM_ON = "webcam_on"
    WEBCAM_OFF = "webcam_off"
    SCREEN_SHARE_START = "screen_share_start"
    SCREEN_SHARE_STOP = "screen_share_stop"
    SPEAKING_START = "speaking_start"
    SPEAKING_STOP = "speaking_stop"
    DISPLAY_NAME_CHANGE = "display_name_change"
    TRANSCRIPT = "transcript"


class Participant(BaseModel):
    participant_id: str
    display_name: str
    joined_at: Optional[datetime] = None
    webcam_on: bool = False
    screen_sharing: bool = False
    total_speaking_seconds: float = 0.0
    is_speaking: bool = False
    name_history: List[str] = Field(default_factory=list)


class MeetingMetadata(BaseModel):
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    interviewer_names: List[str] = Field(default_factory=list)
    scheduled_start: Optional[datetime] = None
    calendar_title: Optional[str] = None


class MeetingEvent(BaseModel):
    event_type: EventType
    participant_id: str
    timestamp: datetime
    payload: Dict[str, Any] = Field(default_factory=dict)


class TranscriptSegment(BaseModel):
    participant_id: str
    text: str
    start_time: datetime
    end_time: Optional[datetime] = None


class SignalContribution(BaseModel):
    signal_name: str
    weight: float
    raw_score: float  # -1 to 1 (negative = evidence against being candidate)
    weighted_score: float
    explanation: str


class ParticipantScore(BaseModel):
    participant_id: str
    display_name: str
    confidence: float  # 0-1
    signals: List[SignalContribution] = Field(default_factory=list)
    is_interviewer: bool = False


class IdentificationResult(BaseModel):
    timestamp: datetime
    candidate_participant_id: Optional[str]
    candidate_display_name: Optional[str]
    confidence: float
    status: str  # "identified", "uncertain", "no_participants"
    rankings: List[ParticipantScore]
    explanation: str
    evidence_summary: List[str] = Field(default_factory=list)

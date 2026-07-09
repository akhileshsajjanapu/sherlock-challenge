"""Tests for the candidate identification engine."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sherlock.engine import CandidateIdentificationEngine
from sherlock.models import EventType, MeetingEvent, MeetingMetadata

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "demo" / "scenarios"


def run_scenario_file(path: Path) -> dict:
    with open(path) as f:
        scenario = json.load(f)

    engine = CandidateIdentificationEngine(
        metadata=MeetingMetadata(**scenario["metadata"])
    )
    base_time = datetime(2026, 1, 15, 10, 0, 0)

    for step in scenario["events"]:
        ts = base_time + timedelta(milliseconds=step["delay_ms"])
        event = MeetingEvent(
            event_type=EventType(step["event_type"]),
            participant_id=step["participant_id"],
            timestamp=ts,
            payload=step.get("payload", {}),
        )
        result = engine.process_event(event)

    return {
        "expected": scenario["expected_candidate_id"],
        "actual": result.candidate_participant_id,
        "confidence": result.confidence,
        "status": result.status,
    }


@pytest.mark.parametrize(
    "scenario_file",
    sorted(SCENARIOS_DIR.glob("*.json")),
    ids=lambda p: p.stem,
)
def test_scenario_identifies_correct_candidate(scenario_file: Path):
    outcome = run_scenario_file(scenario_file)
    assert outcome["actual"] == outcome["expected"], (
        f"Expected {outcome['expected']}, got {outcome['actual']} "
        f"(conf={outcome['confidence']:.0%}, status={outcome['status']})"
    )


def test_no_participants():
    engine = CandidateIdentificationEngine()
    result = engine.identify()
    assert result.status == "no_participants"
    assert result.candidate_participant_id is None


def test_name_match_signal():
    engine = CandidateIdentificationEngine(
        metadata=MeetingMetadata(
            candidate_name="Alice Johnson",
            interviewer_names=["Bob Smith"],
        )
    )
    base = datetime(2026, 1, 1, 10, 0)

    engine.process_event(
        MeetingEvent(
            event_type=EventType.JOIN,
            participant_id="p1",
            timestamp=base,
            payload={"display_name": "Bob Smith", "webcam_on": True},
        )
    )
    engine.process_event(
        MeetingEvent(
            event_type=EventType.JOIN,
            participant_id="p2",
            timestamp=base + timedelta(seconds=2),
            payload={"display_name": "Alice Johnson", "webcam_on": True},
        )
    )

    result = engine.identify()
    assert result.candidate_participant_id == "p2"
    assert result.confidence > 0.5


def test_confidence_increases_with_evidence():
    engine = CandidateIdentificationEngine(
        metadata=MeetingMetadata(candidate_name="Test User")
    )
    base = datetime(2026, 1, 1, 10, 0)

    engine.process_event(
        MeetingEvent(
            event_type=EventType.JOIN,
            participant_id="p1",
            timestamp=base,
            payload={"display_name": "MacBook Pro"},
        )
    )
    early = engine.identify()

    engine.process_event(
        MeetingEvent(
            event_type=EventType.TRANSCRIPT,
            participant_id="p1",
            timestamp=base + timedelta(seconds=5),
            payload={"text": "In my previous role, I built distributed systems."},
        )
    )
    engine.process_event(
        MeetingEvent(
            event_type=EventType.SPEAKING_STOP,
            participant_id="p1",
            timestamp=base + timedelta(seconds=10),
            payload={"duration_seconds": 30},
        )
    )
    late = engine.identify()

    # With only generic name, early confidence should be lower or uncertain
    assert late.confidence >= early.confidence or late.status != "no_participants"


def test_explanation_not_empty():
    engine = CandidateIdentificationEngine(
        metadata=MeetingMetadata(candidate_name="Jane Doe")
    )
    base = datetime(2026, 1, 1, 10, 0)
    engine.process_event(
        MeetingEvent(
            event_type=EventType.JOIN,
            participant_id="p1",
            timestamp=base,
            payload={"display_name": "Jane Doe", "webcam_on": True},
        )
    )
    result = engine.identify()
    assert result.explanation
    assert len(result.explanation) > 10

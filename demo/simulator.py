"""Replay interview scenarios through the identification engine."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sherlock.engine import CandidateIdentificationEngine
from sherlock.models import EventType, MeetingEvent, MeetingMetadata


SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def load_scenario(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_scenario(scenario: dict, verbose: bool = True) -> dict:
    """Run a scenario and return final identification result."""
    engine = CandidateIdentificationEngine(
        metadata=MeetingMetadata(**scenario["metadata"])
    )

    base_time = datetime(2026, 1, 15, 10, 0, 0)
    results = []

    for step in scenario["events"]:
        ts = base_time + timedelta(milliseconds=step["delay_ms"])
        event_type = EventType(step["event_type"])

        event = MeetingEvent(
            event_type=event_type,
            participant_id=step["participant_id"],
            timestamp=ts,
            payload=step.get("payload", {}),
        )
        result = engine.process_event(event)
        results.append(result)

        if verbose:
            print(f"\n[{step['delay_ms']:>5}ms] {step['event_type']:>20} | "
                  f"{step['participant_id']} | "
                  f"Status: {result.status:>10} | "
                  f"Candidate: {result.candidate_display_name or '—':>15} | "
                  f"Conf: {result.confidence:.0%}")

    final = results[-1] if results else engine.identify()
    expected = scenario.get("expected_candidate_id")
    actual = final.candidate_participant_id
    correct = actual == expected

    if verbose:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"Expected: {expected} | Identified: {actual} | {'PASS' if correct else 'FAIL'}")
        print(f"Final confidence: {final.confidence:.0%} | Status: {final.status}")
        print(f"Explanation: {final.explanation}")
        if final.evidence_summary:
            print("Evidence:")
            for line in final.evidence_summary:
                print(line)

    return {
        "scenario": scenario["name"],
        "expected": expected,
        "actual": actual,
        "correct": correct,
        "confidence": final.confidence,
        "status": final.status,
        "explanation": final.explanation,
        "timeline": engine.get_confidence_timeline(),
    }


def run_all_scenarios() -> list[dict]:
    results = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        scenario = load_scenario(path)
        result = run_scenario(scenario, verbose=True)
        results.append(result)
        print()
    return results


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"Accuracy: {correct}/{total} ({correct/total:.0%})")
    print()
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"  [{status}] {r['scenario']:40} conf={r['confidence']:.0%}  status={r['status']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario_path = Path(sys.argv[1])
        if not scenario_path.exists():
            scenario_path = SCENARIOS_DIR / sys.argv[1]
        scenario = load_scenario(scenario_path)
        run_scenario(scenario)
    else:
        results = run_all_scenarios()
        print_summary(results)

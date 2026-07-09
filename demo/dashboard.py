"""Streamlit dashboard for live demo of candidate identification."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sherlock.engine import CandidateIdentificationEngine
from sherlock.models import EventType, MeetingEvent, MeetingMetadata

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

st.set_page_config(
    page_title="Sherlock Candidate Identifier",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Sherlock Candidate Identifier")
st.markdown(
    "Real-time interview candidate identification using **multi-signal fusion**. "
    "Combines name matching, speaking patterns, transcript analysis, and more."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Scenario")
    scenario_files = sorted(SCENARIOS_DIR.glob("*.json"))
    scenario_names = [f.stem.replace("_", " ").title() for f in scenario_files]
    selected_idx = st.selectbox(
        "Choose scenario",
        range(len(scenario_files)),
        format_func=lambda i: scenario_names[i],
    )
    speed = st.slider("Replay speed", 0.5, 5.0, 2.0, 0.5)
    run_btn = st.button("▶ Run Scenario", type="primary", use_container_width=True)

with col_right:
    st.subheader("Live Identification")
    status_placeholder = st.empty()
    confidence_placeholder = st.empty()
    explanation_placeholder = st.empty()
    rankings_placeholder = st.empty()
    timeline_placeholder = st.empty()


def render_result(result, step_info: str = ""):
    status_color = {
        "identified": "🟢",
        "uncertain": "🟡",
        "no_participants": "⚪",
    }
    icon = status_color.get(result.status, "⚪")

    status_placeholder.markdown(
        f"### {icon} Status: **{result.status.upper()}**"
        + (f"  \n_{step_info}_" if step_info else "")
    )

    if result.candidate_display_name:
        confidence_placeholder.metric(
            label=f"Candidate: {result.candidate_display_name}",
            value=f"{result.confidence:.0%}",
            delta=f"ID: {result.candidate_participant_id}",
        )
    else:
        confidence_placeholder.metric(label="Candidate", value="—", delta="Not identified")

    explanation_placeholder.info(result.explanation)

    if result.rankings:
        st.subheader("Participant Rankings")
        ranking_data = []
        for r in result.rankings:
            top_signals = sorted(
                r.signals, key=lambda s: abs(s.weighted_score), reverse=True
            )[:3]
            signal_str = " | ".join(
                f"{s.signal_name}: {s.raw_score:+.2f}" for s in top_signals
            )
            ranking_data.append(
                {
                    "Participant": r.display_name,
                    "ID": r.participant_id,
                    "Confidence": f"{r.confidence:.0%}",
                    "Interviewer?": "Yes" if r.is_interviewer else "No",
                    "Top Signals": signal_str,
                }
            )
        rankings_placeholder.dataframe(ranking_data, use_container_width=True)

    if result.evidence_summary:
        with st.expander("Evidence Details"):
            for line in result.evidence_summary:
                st.markdown(line)


if run_btn:
    scenario_path = scenario_files[selected_idx]
    with open(scenario_path) as f:
        scenario = json.load(f)

    st.sidebar.markdown(f"**{scenario['name']}**")
    st.sidebar.markdown(scenario["description"])
    st.sidebar.markdown("**Metadata:**")
    st.sidebar.json(scenario["metadata"])

    engine = CandidateIdentificationEngine(
        metadata=MeetingMetadata(**scenario["metadata"])
    )

    base_time = datetime(2026, 1, 15, 10, 0, 0)
    timeline_data: list[dict] = []
    prev_delay = 0

    progress = st.progress(0)
    total_steps = len(scenario["events"])

    for i, step in enumerate(scenario["events"]):
        wait_ms = (step["delay_ms"] - prev_delay) / speed
        if wait_ms > 0:
            time.sleep(wait_ms / 1000)
        prev_delay = step["delay_ms"]

        ts = base_time + timedelta(milliseconds=step["delay_ms"])
        event = MeetingEvent(
            event_type=EventType(step["event_type"]),
            participant_id=step["participant_id"],
            timestamp=ts,
            payload=step.get("payload", {}),
        )
        result = engine.process_event(event)

        step_info = f"{step['event_type']} — {step['participant_id']}"
        render_result(result, step_info)

        timeline_data.append(
            {
                "step": i + 1,
                "event": step["event_type"],
                "confidence": result.confidence,
                "status": result.status,
            }
        )
        progress.progress((i + 1) / total_steps)

    # Final summary
    expected = scenario.get("expected_candidate_id")
    actual = result.candidate_participant_id
    correct = actual == expected

    st.divider()
    if correct:
        st.success(
            f"Correctly identified candidate `{actual}` "
            f"with {result.confidence:.0%} confidence"
        )
    else:
        st.error(
            f"Expected `{expected}`, got `{actual}` "
            f"(confidence: {result.confidence:.0%})"
        )

    with st.expander("Confidence Timeline"):
        st.line_chart(
            data={r["step"]: r["confidence"] for r in timeline_data},
            x_label="Event Step",
            y_label="Confidence",
        )

else:
    st.info("Select a scenario and click **Run Scenario** to start the demo.")

    st.markdown("---")
    st.subheader("Architecture Overview")
    st.markdown(
        """
        ```
        Meeting Events ──► Signal Extractors ──► Fusion Engine ──► Identification
                              │                      │
                              ├─ Name Match          ├─ Weighted Sum
                              ├─ Interviewer Excl.   ├─ Sigmoid
                              ├─ Speaking Pattern    ├─ EMA Smoothing
                              ├─ Transcript Role     └─ Confidence + Explanation
                              ├─ Webcam Behavior
                              ├─ Device Name
                              ├─ Join Order
                              ├─ Screen Share
                              └─ Observer Silence
        ```
        """
    )

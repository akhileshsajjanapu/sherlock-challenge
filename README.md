# Sherlock Candidate Identifier

Real-time AI system that automatically identifies the interview candidate during live meetings (Google Meet, Microsoft Teams, Zoom) by fusing multiple weak signals with confidence scoring and explainability.

Built for the [Sherlock Internship Challenge](https://sherlock.sh).

## Problem

Sherlock's fraud detectors must analyze the **candidate's** audio and video — not the interviewer or observers. But candidates often join with wrong names, device names ("MacBook Pro"), nicknames, or change names mid-interview. Interviewers may enter incorrect metadata. Multiple participants create ambiguity.

This system solves candidate identification by combining many weak signals rather than relying on any single heuristic.

## Approach

**Multi-signal fusion engine** that:

1. Ingests real-time meeting events (join/leave, webcam, speaking, transcript)
2. Runs 9 independent signal extractors per participant
3. Fuses evidence via weighted sum → sigmoid → EMA temporal smoothing
4. Outputs candidate ID, confidence score (0–1), and human-readable explanation
5. Gracefully handles uncertainty — reports "uncertain" instead of guessing

### Signal Extractors

| Signal | Weight | What it detects |
|--------|--------|-----------------|
| Name Match | 0.22 | Fuzzy match display name vs candidate name/email |
| Interviewer Exclusion | 0.18 | Penalizes known interviewer names |
| Speaking Pattern | 0.15 | Q&A dynamics — candidates answer, interviewers ask |
| Transcript Role | 0.15 | Linguistic patterns ("in my previous role" vs "tell me about") |
| Webcam Behavior | 0.08 | Candidates typically keep camera on |
| Device Name | 0.07 | Penalizes generic names like "MacBook Pro" |
| Join Order | 0.05 | Candidates often join after interviewers |
| Screen Share | 0.05 | Candidates may share portfolio/code |
| Observer Silence | 0.05 | Silent participants are likely observers |

## Quick Start

### Prerequisites

- Python 3.9+
- (Optional) OpenAI API key for enhanced transcript analysis

### Setup

```bash
git clone <your-repo-url>
cd sherlock-candidate-id

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Run Demo (Streamlit Dashboard)

```bash
streamlit run demo/dashboard.py
```

Select a scenario and click **Run Scenario** to watch real-time identification.

### Run Scenario Simulator (CLI)

```bash
# Run all scenarios
python demo/simulator.py

# Run a specific scenario
python demo/simulator.py 02_macbook_pro.json
```

### Run API Server

```bash
uvicorn api.main:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/meetings/{id}/metadata` | Set candidate/interviewer info |
| POST | `/meetings/{id}/events` | Ingest meeting event |
| POST | `/meetings/{id}/transcript` | Add transcript segment |
| GET | `/meetings/{id}/identify` | Get current identification |
| GET | `/meetings/{id}/timeline` | Confidence evolution |
| WS | `/ws/{id}` | Real-time WebSocket stream |

### Run Tests

```bash
pytest tests/ -v
```

## Demo Scenarios

Six scenarios covering real-world edge cases:

1. **Normal Interview** — candidate joins with correct name
2. **MacBook Pro** — generic device name, behavioral identification
3. **Nickname + Wrong Metadata** — interviewer entered wrong name
4. **Multiple Interviewers + Observer** — silent observer exclusion
5. **Display Name Change** — candidate renames mid-interview
6. **Missing Metadata** — no candidate name, pure behavioral signals

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram and data flow.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for test results, edge cases, accuracy metrics, and limitations.

## Assumptions

- Separate audio/video streams per participant are available (provided by Sherlock's meeting bot)
- Speaker-attributed transcript is available (from STT with diarization)
- External metadata (calendar invite, candidate email) may be incomplete or wrong
- System operates in real time — events processed as they arrive
- OpenAI API key is optional; rule-based transcript analysis works without it

## Project Structure

```
sherlock-candidate-id/
├── src/sherlock/           # Core engine
│   ├── engine.py           # Main orchestrator
│   ├── fusion.py           # Evidence fusion + EMA
│   ├── explainer.py        # Human-readable explanations
│   ├── models.py           # Pydantic data models
│   └── signals/            # 9 weak signal extractors
├── api/main.py             # FastAPI + WebSocket server
├── demo/
│   ├── dashboard.py        # Streamlit live demo
│   ├── simulator.py        # CLI scenario replay
│   └── scenarios/          # 6 test scenarios
├── tests/test_engine.py    # Automated tests
├── ARCHITECTURE.md
├── EVALUATION.md
└── README.md
```

## Trade-offs

- **Rule-based transcript analysis** over pure LLM: faster, cheaper, deterministic; LLM is optional boost
- **EMA smoothing** over instant decisions: prevents flip-flopping but adds latency to confidence changes
- **Weighted fusion** over learned model: interpretable and tunable without training data; could be upgraded to learned weights with labeled interviews
- **Graceful uncertainty** over forced classification: better to say "uncertain" than misidentify

## What I'd Improve Next

1. **Learned fusion weights** from labeled interview data
2. **Voice biometrics** — match candidate voice against pre-interview audio sample
3. **Face recognition** — match against LinkedIn/calendar photo
4. **Active learning** — human corrections feed back into weights
5. **Multi-meeting context** — if same candidate had prior interviews, use history
6. **Production hardening** — metrics, alerting, A/B testing framework

## License

MIT

"""FastAPI server for real-time candidate identification."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sherlock.engine import CandidateIdentificationEngine
from sherlock.models import (
    EventType,
    MeetingEvent,
    MeetingMetadata,
    TranscriptSegment,
)

engines: dict[str, CandidateIdentificationEngine] = {}


def get_or_create_engine(meeting_id: str) -> CandidateIdentificationEngine:
    if meeting_id not in engines:
        engines[meeting_id] = CandidateIdentificationEngine()
    return engines[meeting_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    engines.clear()


app = FastAPI(
    title="Sherlock Candidate Identifier",
    description="Real-time interview candidate identification using multi-signal fusion",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MetadataRequest(BaseModel):
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    interviewer_names: List[str] = []
    calendar_title: Optional[str] = None


class EventRequest(BaseModel):
    event_type: str
    participant_id: str
    timestamp: Optional[str] = None
    payload: Dict[str, Any] = {}


class TranscriptRequest(BaseModel):
    participant_id: str
    text: str
    timestamp: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "active_meetings": len(engines)}


@app.post("/meetings/{meeting_id}/metadata")
async def set_metadata(meeting_id: str, body: MetadataRequest):
    engine = get_or_create_engine(meeting_id)
    engine.update_metadata(MeetingMetadata(**body.model_dump()))
    return {"status": "metadata_updated"}


@app.post("/meetings/{meeting_id}/events")
async def post_event(meeting_id: str, body: EventRequest):
    engine = get_or_create_engine(meeting_id)
    ts = datetime.fromisoformat(body.timestamp) if body.timestamp else datetime.now()
    event = MeetingEvent(
        event_type=EventType(body.event_type),
        participant_id=body.participant_id,
        timestamp=ts,
        payload=body.payload,
    )
    result = engine.process_event(event)
    return result.model_dump(mode="json")


@app.post("/meetings/{meeting_id}/transcript")
async def post_transcript(meeting_id: str, body: TranscriptRequest):
    engine = get_or_create_engine(meeting_id)
    ts = datetime.fromisoformat(body.timestamp) if body.timestamp else datetime.now()
    segment = TranscriptSegment(
        participant_id=body.participant_id,
        text=body.text,
        start_time=ts,
    )
    result = engine.add_transcript(segment)
    return result.model_dump(mode="json")


@app.get("/meetings/{meeting_id}/identify")
async def identify(meeting_id: str):
    engine = get_or_create_engine(meeting_id)
    result = engine.identify()
    return result.model_dump(mode="json")


@app.get("/meetings/{meeting_id}/timeline")
async def timeline(meeting_id: str):
    engine = get_or_create_engine(meeting_id)
    return engine.get_confidence_timeline()


@app.get("/api/scenarios")
async def list_scenarios():
    scenarios_dir = Path(__file__).resolve().parent.parent / "demo" / "scenarios"
    if not scenarios_dir.exists():
        return []
    
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
                scenarios.append({
                    "filename": path.name,
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "expected_candidate_id": data.get("expected_candidate_id", ""),
                    "metadata": data.get("metadata", {})
                })
        except Exception:
            pass
    return scenarios


@app.get("/api/scenarios/{filename}")
async def get_scenario(filename: str):
    scenarios_dir = Path(__file__).resolve().parent.parent / "demo" / "scenarios"
    path = scenarios_dir / filename
    try:
        resolved_path = path.resolve()
        resolved_dir = scenarios_dir.resolve()
        if not str(resolved_path).startswith(str(resolved_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Access denied")
        
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    engine = get_or_create_engine(meeting_id)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "metadata":
                engine.update_metadata(MeetingMetadata(**msg["data"]))
                await websocket.send_json({"type": "ack", "message": "metadata_updated"})

            elif msg_type == "event":
                ts = datetime.fromisoformat(msg["timestamp"]) if msg.get("timestamp") else datetime.now()
                event = MeetingEvent(
                    event_type=EventType(msg["event_type"]),
                    participant_id=msg["participant_id"],
                    timestamp=ts,
                    payload=msg.get("payload", {}),
                )
                result = engine.process_event(event)
                await websocket.send_json({"type": "identification", "data": result.model_dump(mode="json")})

            elif msg_type == "transcript":
                ts = datetime.fromisoformat(msg["timestamp"]) if msg.get("timestamp") else datetime.now()
                segment = TranscriptSegment(
                    participant_id=msg["participant_id"],
                    text=msg["text"],
                    start_time=ts,
                )
                result = engine.add_transcript(segment)
                await websocket.send_json({"type": "identification", "data": result.model_dump(mode="json")})

            elif msg_type == "identify":
                result = engine.identify()
                await websocket.send_json({"type": "identification", "data": result.model_dump(mode="json")})

    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

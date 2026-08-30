from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from backend.agent.core import RateLimitException
from backend.agent.session_manager import SessionManager

# Application-level SessionManager instance (Sole orchestration layer)
session_manager = SessionManager()


def __getattr__(name: str):
    if name == "agent":
        return session_manager.get_or_create_session("default")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    session_manager.close()


app = FastAPI(title="P.I.X.I.E. Backend", version="0.1.0", lifespan=lifespan)

# Add CORS middleware to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ActionRequest(BaseModel):
    confirmation_id: str
    tool_name: str
    arguments: Dict[str, Any]


class ChatResponse(BaseModel):
    response: str
    spoken_response: Optional[str] = None
    action_required: Optional[ActionRequest] = None


class ConfirmRequest(BaseModel):
    confirmation_id: str
    approved: bool
    session_id: Optional[str] = None


class ClearRequest(BaseModel):
    session_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"
    session_id: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: str


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float


def resolve_target_session_id(session_id: Optional[str]) -> str:
    """
    Resolves session_id. If explicitly provided, validates that session exists in SessionManager.
    Returns 404 if explicit session_id is unknown.
    If session_id is not provided, defaults to "default" for backward compatibility.
    """
    if session_id:
        existing = session_manager.get_session(session_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Session not found")
        return session_id
    session_manager.get_or_create_session("default")
    return "default"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives user input from the frontend and processes it routed via SessionManager.
    """
    sid = resolve_target_session_id(request.session_id)
    try:
        response_text, spoken_response, action_required = await session_manager.process_intent(
            sid, request.message
        )
        return ChatResponse(
            response=response_text,
            spoken_response=spoken_response,
            action_required=ActionRequest(**action_required) if action_required else None
        )
    except RateLimitException as e:
        return JSONResponse(
            status_code=429,
            content={"response": e.message, "spoken_response": e.message}
        )


@app.post("/voice", response_model=ChatResponse)
async def voice_endpoint(request: ChatRequest):
    """
    Receives voice transcript and processes it routed via SessionManager.
    """
    sid = resolve_target_session_id(request.session_id)
    try:
        response_text, spoken_response, action_required = await session_manager.process_intent(
            sid, request.message
        )
        return ChatResponse(
            response=response_text,
            spoken_response=spoken_response,
            action_required=ActionRequest(**action_required) if action_required else None
        )
    except RateLimitException as e:
        return JSONResponse(
            status_code=429,
            content={"response": e.message, "spoken_response": e.message}
        )


@app.post("/confirm", response_model=ChatResponse)
async def confirm_endpoint(request: ConfirmRequest):
    """
    Handles user confirmation for pending tool executions routed via SessionManager.
    """
    sid = resolve_target_session_id(request.session_id)
    try:
        response_text, spoken_response, action_required = await session_manager.handle_confirmation(
            sid,
            request.confirmation_id,
            request.approved
        )
        return ChatResponse(
            response=response_text,
            spoken_response=spoken_response,
            action_required=ActionRequest(**action_required) if action_required else None
        )
    except RateLimitException as e:
        return JSONResponse(
            status_code=429,
            content={"response": e.message, "spoken_response": e.message}
        )


@app.get("/status")
def status_endpoint():
    """
    Exposes PIXIE's Token Telemetry status safely via SessionManager.
    """
    default_agent = session_manager.get_or_create_session("default")
    gov_status = default_agent.governor.get_status()

    rpm_limit = gov_status["rpm_limit"]
    tpm_limit = gov_status["tpm_limit"]
    rpd_limit = gov_status["rpd_limit"]
    tpd_limit = gov_status["tpd_limit"]

    req_min = gov_status["requests_minute"]
    tok_min = gov_status["tokens_minute"]
    req_day = gov_status["requests_day"]
    tok_day = gov_status["tokens_day"]

    return {
        "status": "online",
        "model": default_agent.model,
        "requests_minute": req_min,
        "tokens_minute": tok_min,
        "requests_day": req_day,
        "tokens_day": tok_day,
        "rpm_limit": rpm_limit,
        "tpm_limit": tpm_limit,
        "rpd_limit": rpd_limit,
        "tpd_limit": tpd_limit,
        "rpm_remaining": max(0, rpm_limit - req_min),
        "tpm_remaining": max(0, tpm_limit - tok_min),
        "rpd_remaining": max(0, rpd_limit - req_day),
        "tpd_remaining": max(0, tpd_limit - tok_day)
    }


@app.get("/usage/history")
def usage_history_endpoint():
    """
    Returns historical usage telemetry for the last 30 days.
    """
    default_agent = session_manager.get_or_create_session("default")
    return default_agent.usage_store.get_daily_history(days=30)


@app.post("/api/clear")
def clear_context(request: Optional[ClearRequest] = None):
    """
    Clears the short-term memory for the specified or default session via SessionManager.
    """
    sid = resolve_target_session_id(request.session_id if request else None)
    target_agent = session_manager.get_session(sid)
    if target_agent:
        target_agent.clear_context()
    return {"status": "Context cleared"}


# ==========================================
# REST /sessions API (Phase 5.2.3)
# ==========================================

@app.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session_endpoint(request: Optional[CreateSessionRequest] = None):
    """
    Creates a new persistent chat session via SessionManager.
    """
    req_title = request.title if request and request.title else "New Chat"
    req_sid = request.session_id if request else None

    if req_sid and session_manager.session_store.get_session(req_sid):
        raise HTTPException(status_code=400, detail="Session ID already exists.")

    try:
        created_agent = session_manager.create_session(title=req_title, session_id=req_sid)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create session: {str(e)}")

    meta = session_manager.session_store.get_session(created_agent.session_id)
    if not meta:
        raise HTTPException(status_code=500, detail="Failed to retrieve created session metadata.")
    return SessionResponse(**meta)


@app.get("/sessions", response_model=List[SessionResponse])
def list_sessions_endpoint(limit: int = 50):
    """
    Lists existing chat sessions ordered by newest updated_at.
    """
    sessions = session_manager.list_sessions(limit=limit)
    return [SessionResponse(**s) for s in sessions]


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session_endpoint(session_id: str):
    """
    Retrieves metadata for an existing session.
    Returns 404 for unknown session IDs.
    """
    meta = session_manager.session_store.get_session(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**meta)


@app.patch("/sessions/{session_id}", response_model=SessionResponse)
def update_session_endpoint(session_id: str, request: UpdateSessionRequest):
    """
    Updates title of an existing session.
    Returns 404 for unknown session IDs.
    """
    updated = session_manager.session_store.update_session_title(session_id, request.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**updated)


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    """
    Deletes a session and its persistent messages, removing cached AgentCore & lock state.
    Returns 404 for unknown session IDs.
    """
    meta = session_manager.session_store.get_session(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    session_manager.remove_session(session_id)
    return {"status": "deleted", "session_id": session_id}


class MessageResponse(BaseModel):
    id: Optional[str] = None
    session_id: str
    role: str
    content: str
    tool_calls_json: Optional[str] = None
    timestamp: float


@app.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
def get_session_messages_endpoint(session_id: str):
    """
    Retrieves stored messages for an existing session.
    """
    meta = session_manager.session_store.get_session(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_manager.session_store.get_messages(session_id)
    return [MessageResponse(**m) for m in messages]


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.1.1", port=8000, reload=True)

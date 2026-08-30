import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent.core import AgentCore, RateLimitException

app = FastAPI(title="P.I.X.I.E. Backend", version="0.1.0")
agent = AgentCore()

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

class ChatResponse(BaseModel):
    response: str
    spoken_response: str | None = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives user input from the frontend and processes it via the Agent Core.
    """
    try:
        response_text, spoken_response = await agent.process_intent(request.message)
        return ChatResponse(response=response_text, spoken_response=spoken_response)
    except RateLimitException as e:
        return JSONResponse(
            status_code=429,
            content={"response": e.message, "spoken_response": e.message}
        )

@app.post("/voice", response_model=ChatResponse)
async def voice_endpoint(request: ChatRequest):
    """
    Receives voice transcript and processes it.
    """
    try:
        response_text, spoken_response = await agent.process_intent(request.message)
        return ChatResponse(response=response_text, spoken_response=spoken_response)
    except RateLimitException as e:
        return JSONResponse(
            status_code=429,
            content={"response": e.message, "spoken_response": e.message}
        )

@app.get("/status")
def status_endpoint():
    """
    Exposes PIXIE's Token Telemetry status safely.
    """
    gov_status = agent.governor.get_status()

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
        "model": agent.model,
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
    return agent.usage_store.get_daily_history(days=30)

@app.post("/api/clear")
def clear_context():
    """
    Clears the short-term memory of the agent.
    """
    agent.clear_context()
    return {"status": "Context cleared"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.1.1", port=8000, reload=True)

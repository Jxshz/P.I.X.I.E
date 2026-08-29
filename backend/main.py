import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent.core import AgentCore

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
    response_text, spoken_response = await agent.process_intent(request.message)
    return ChatResponse(response=response_text, spoken_response=spoken_response)

@app.post("/voice", response_model=ChatResponse)
async def voice_endpoint(request: ChatRequest):
    """
    Receives voice transcript and processes it.
    """
    response_text, spoken_response = await agent.process_intent(request.message)
    return ChatResponse(response=response_text, spoken_response=spoken_response)

@app.post("/api/clear")
def clear_context():
    """
    Clears the short-term memory of the agent.
    """
    agent.clear_context()
    return {"status": "Context cleared"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.1.1", port=8000, reload=True)

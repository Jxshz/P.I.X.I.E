import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple

from backend.agent.core import AgentCore, format_display_response, generate_spoken_response
from backend.storage.session_store import SessionStore


class SessionManager:
    """
    Orchestrates session lifecycle, manages session-scoped AgentCore instances,
    and enforces per-session concurrency isolation.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
    ):
        self.session_store = session_store or SessionStore(db_path=db_path)
        self._active_agents: Dict[str, AgentCore] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._lock = threading.Lock()

    def get_session_lock(self, session_id: str) -> asyncio.Lock:
        """
        Retrieves or creates an asyncio.Lock specific to the session_id.
        """
        with self._lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    def create_session(
        self, title: str = "New Chat", session_id: Optional[str] = None
    ) -> AgentCore:
        """
        Persists a new session in SessionStore and initializes its AgentCore instance.
        """
        with self._lock:
            meta = self.session_store.create_session(title=title, session_id=session_id)
            sid = meta["id"]
            agent = AgentCore(session_store=self.session_store, session_id=sid)
            self._active_agents[sid] = agent
            if sid not in self._session_locks:
                self._session_locks[sid] = asyncio.Lock()
            return agent

    def get_session(self, session_id: str) -> Optional[AgentCore]:
        """
        Retrieves the AgentCore instance for a given session_id.
        Returns None if the session does not exist in SessionStore.
        """
        with self._lock:
            if session_id in self._active_agents:
                return self._active_agents[session_id]

            meta = self.session_store.get_session(session_id)
            if not meta:
                return None

            agent = AgentCore(session_store=self.session_store, session_id=session_id)
            self._active_agents[session_id] = agent
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return agent

    def get_or_create_session(
        self, session_id: Optional[str] = None, title: str = "New Chat"
    ) -> AgentCore:
        """
        Retrieves an existing session or creates a new one if session_id is None or missing.
        """
        if session_id:
            agent = self.get_session(session_id)
            if agent:
                return agent
        return self.create_session(title=title, session_id=session_id)

    async def process_intent(
        self, session_id: str, user_input: str
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Processes a user turn for the specified session serially under that session's lock.
        """
        agent = self.get_or_create_session(session_id)
        lock = self.get_session_lock(agent.session_id)
        async with lock:
            return await agent.process_intent(user_input)

    async def handle_confirmation(
        self, session_id: str, confirmation_id: str, approved: bool
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Handles a pending confirmation resolution for the specified session serially.
        """
        agent = self.get_session(session_id)
        if not agent:
            msg = "Confirmation failed: Unknown, expired, or already used confirmation ID."
            return format_display_response(msg), generate_spoken_response(msg), None

        lock = self.get_session_lock(agent.session_id)
        async with lock:
            return await agent.handle_confirmation(confirmation_id, approved)

    def remove_session(self, session_id: str) -> bool:
        """
        Removes active AgentCore instance & lock from memory and deletes session from SessionStore.
        """
        with self._lock:
            self._active_agents.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            return self.session_store.delete_session(session_id)

    def clear_session_cache(self, session_id: Optional[str] = None) -> None:
        """
        Clears cached AgentCore instances & locks from memory without deleting persistent records.
        """
        with self._lock:
            if session_id:
                self._active_agents.pop(session_id, None)
                self._session_locks.pop(session_id, None)
            else:
                self._active_agents.clear()
                self._session_locks.clear()

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Lists session metadata ordered by newest updated_at timestamp.
        """
        return self.session_store.list_sessions(limit=limit)

    def close(self) -> None:
        """
        Clears cached agents and locks, and closes underlying SessionStore cleanly.
        """
        with self._lock:
            self._active_agents.clear()
            self._session_locks.clear()
            self.session_store.close()

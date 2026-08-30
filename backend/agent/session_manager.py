import threading
from typing import Any, Dict, List, Optional

from backend.agent.core import AgentCore
from backend.storage.session_store import SessionStore


class SessionManager:
    """
    Orchestrates session lifecycle and manages session-scoped AgentCore instances.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
    ):
        self.session_store = session_store or SessionStore(db_path=db_path)
        self._active_agents: Dict[str, AgentCore] = {}
        self._lock = threading.Lock()

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

    def remove_session(self, session_id: str) -> bool:
        """
        Removes the active AgentCore instance from cache and deletes the session from SessionStore.
        """
        with self._lock:
            self._active_agents.pop(session_id, None)
            return self.session_store.delete_session(session_id)

    def clear_session_cache(self, session_id: Optional[str] = None) -> None:
        """
        Clears cached AgentCore instances from memory without deleting persistent records.
        If session_id is provided, removes only that session from memory cache.
        If session_id is None, clears all cached instances.
        """
        with self._lock:
            if session_id:
                self._active_agents.pop(session_id, None)
            else:
                self._active_agents.clear()

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Lists session metadata ordered by newest updated_at timestamp.
        """
        return self.session_store.list_sessions(limit=limit)

    def close(self) -> None:
        """
        Clears cached agents and closes underlying SessionStore cleanly.
        """
        with self._lock:
            self._active_agents.clear()
            self.session_store.close()

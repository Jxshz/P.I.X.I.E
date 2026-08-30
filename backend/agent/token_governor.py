import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

@dataclass
class Reservation:
    id: str
    tokens: int
    timestamp: float

class TokenGovernor:
    """
    In-memory Token Governor for P.I.X.I.E.
    Tracks and limits API usage over minute and daily rolling windows.
    Includes atomic reservation to prevent concurrency double-admission.
    """
    def __init__(self):
        # Load limits from env or use defaults
        self.rpm_limit = int(os.getenv("GROQ_RPM_LIMIT", "30"))
        self.tpm_limit = int(os.getenv("GROQ_TPM_LIMIT", "8000"))
        self.rpd_limit = int(os.getenv("GROQ_RPD_LIMIT", "1000"))
        self.tpd_limit = int(os.getenv("GROQ_TPD_LIMIT", "200000"))
        self.max_completion_tokens = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "1024"))

        # In-memory tracking
        # Each entry is a Reservation object
        self._minute_window: List[Reservation] = []
        self._day_window: List[Reservation] = []

    def _clean_windows(self, now: float) -> None:
        """Removes entries older than 1 minute and 1 day."""
        self._minute_window = [res for res in self._minute_window if now - res.timestamp < 60]
        self._day_window = [res for res in self._day_window if now - res.timestamp < 86400]

    def _get_current_usage(self, now: float) -> Tuple[int, int, int, int]:
        """Returns (requests_minute, tokens_minute, requests_day, tokens_day)"""
        self._clean_windows(now)
        req_min = len(self._minute_window)
        tok_min = sum(res.tokens for res in self._minute_window)
        req_day = len(self._day_window)
        tok_day = sum(res.tokens for res in self._day_window)
        return req_min, tok_min, req_day, tok_day

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Conservatively estimates input tokens based on character count.
        Assumes ~4 chars per token for English, multiplied by a 1.5 safety factor.
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content")
            if content:
                total_chars += len(str(content))

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total_chars += len(str(tool_calls))

        estimate = int((total_chars / 4.0) * 1.5)
        return estimate if estimate > 0 else 1

    def preflight(self, messages: List[Dict[str, str]]) -> Tuple[bool, str, Optional[Reservation]]:
        """
        Checks if the request can be made without exceeding limits, reserving the budget if allowed.
        Returns:
            Tuple[bool, str, Optional[Reservation]]: (is_allowed, error_message, reservation)
        """
        now = time.time()
        req_min, tok_min, req_day, tok_day = self._get_current_usage(now)

        input_estimate = self.estimate_tokens(messages)
        # Total estimate includes the absolute maximum output tokens we allow Groq to generate
        total_estimate = input_estimate + self.max_completion_tokens

        # Check daily limits first
        if req_day >= self.rpd_limit or (tok_day + total_estimate) > self.tpd_limit:
            return False, "Sir, I've reached today's processing budget. We can continue once the daily allowance resets.", None

        # Check minute limits
        if (tok_min + total_estimate) > self.tpm_limit:
            return False, "Sir, I've reached my short-term processing limit. Please give me a moment before we continue.", None

        if req_min >= self.rpm_limit:
            return False, "Sir, I'm processing requests a little too quickly. Give me a moment.", None

        # Reserve the budget atomically
        reservation = Reservation(id=str(uuid.uuid4()), tokens=total_estimate, timestamp=now)
        self._minute_window.append(reservation)
        self._day_window.append(reservation)

        return True, "", reservation

    def record_usage(self, reservation: Reservation, actual_usage: Optional[Any] = None, failed: bool = False) -> None:
        """
        Reconciles the reservation with actual usage.
        If failed is True, the reservation is simply dropped.
        """
        # Remove the exact reservation from both windows
        self._minute_window = [r for r in self._minute_window if r.id != reservation.id]
        self._day_window = [r for r in self._day_window if r.id != reservation.id]

        if failed:
            return

        tokens = reservation.tokens

        # Extract actual usage if present
        if actual_usage and hasattr(actual_usage, 'total_tokens') and actual_usage.total_tokens is not None:
            tokens = actual_usage.total_tokens
        elif isinstance(actual_usage, dict) and actual_usage.get('total_tokens') is not None:
            tokens = actual_usage['total_tokens']

        # Append the finalized usage
        # We preserve the original reservation timestamp
        finalized_reservation = Reservation(id=reservation.id, tokens=tokens, timestamp=reservation.timestamp)
        self._minute_window.append(finalized_reservation)
        self._day_window.append(finalized_reservation)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current limit consumption status."""
        now = time.time()
        req_min, tok_min, req_day, tok_day = self._get_current_usage(now)
        return {
            "requests_minute": req_min,
            "tokens_minute": tok_min,
            "requests_day": req_day,
            "tokens_day": tok_day,
            "rpm_limit": self.rpm_limit,
            "tpm_limit": self.tpm_limit,
            "rpd_limit": self.rpd_limit,
            "tpd_limit": self.tpd_limit
        }

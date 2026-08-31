import json
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional


class MemoryCategory(str, Enum):
    """Categories of persistent memory in P.I.X.I.E."""
    USER_PROFILE = "user_profile"
    USER_PREFERENCE = "user_preference"
    USER_FACT = "user_fact"
    CONTEXT_RULE = "context_rule"


class MemorySource(str, Enum):
    """Source provenance of persistent memory."""
    EXPLICIT_USER_INPUT = "explicit_user_input"
    SYSTEM_INFERRED = "system_inferred"


class MemoryValidationError(ValueError):
    """Raised when a MemoryRecord fails validation rules."""
    pass


@dataclass
class MemoryRecord:
    """
    Data model representing a single persistent memory record in P.I.X.I.E.
    """
    id: str
    category: MemoryCategory
    key: str
    value: str
    source: MemorySource
    confidence: float
    created_at: float
    updated_at: float
    expires_at: Optional[float] = None
    is_active: bool = True
    metadata_json: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.category, str) and not isinstance(self.category, MemoryCategory):
            self.category = MemoryCategory(self.category)
        if isinstance(self.source, str) and not isinstance(self.source, MemorySource):
            self.source = MemorySource(self.source)

    def validate(self) -> None:
        """Enforces schema and semantic validation rules on the MemoryRecord."""
        if not self.id or not isinstance(self.id, str):
            raise MemoryValidationError("Memory record 'id' must be a non-empty string.")

        if not isinstance(self.category, MemoryCategory):
            raise MemoryValidationError(f"Invalid memory category: {self.category}")

        if not self.key or not isinstance(self.key, str) or not self.key.strip():
            raise MemoryValidationError("Memory record 'key' must be a non-empty string.")

        if not isinstance(self.value, str) or not self.value.strip():
            raise MemoryValidationError("Memory record 'value' must be a non-empty string.")

        if not isinstance(self.source, MemorySource):
            raise MemoryValidationError(f"Invalid memory source: {self.source}")

        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise MemoryValidationError(f"Memory confidence must be a float between 0.0 and 1.0, got: {self.confidence}")

        if not isinstance(self.created_at, (int, float)) or self.created_at <= 0:
            raise MemoryValidationError("Memory record 'created_at' must be a valid positive epoch timestamp.")

        if not isinstance(self.updated_at, (int, float)) or self.updated_at <= 0:
            raise MemoryValidationError("Memory record 'updated_at' must be a valid positive epoch timestamp.")

        if self.expires_at is not None:
            if not isinstance(self.expires_at, (int, float)) or self.expires_at <= self.created_at:
                raise MemoryValidationError("Memory record 'expires_at' must be a timestamp greater than created_at.")

        if self.metadata_json is not None:
            try:
                json.loads(self.metadata_json)
            except Exception as e:
                raise MemoryValidationError(f"Invalid JSON string in metadata_json: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the MemoryRecord to a dictionary."""
        self.validate()
        return {
            "id": self.id,
            "category": self.category.value,
            "key": self.key,
            "value": self.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """Factory method to construct a MemoryRecord from a dictionary."""
        if not isinstance(data, dict):
            raise MemoryValidationError("Input data for MemoryRecord must be a dictionary.")

        record = cls(
            id=data.get("id", str(uuid.uuid4())),
            category=MemoryCategory(data["category"]) if "category" in data else MemoryCategory.USER_FACT,
            key=data.get("key", ""),
            value=data.get("value", ""),
            source=MemorySource(data["source"]) if "source" in data else MemorySource.EXPLICIT_USER_INPUT,
            confidence=float(data.get("confidence", 1.0)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            is_active=bool(data.get("is_active", True)),
            metadata_json=data.get("metadata_json"),
        )
        record.validate()
        return record

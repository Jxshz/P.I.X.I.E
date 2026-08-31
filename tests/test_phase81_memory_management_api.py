import os
import tempfile
import time
import pytest

from backend.memory import (
    MemoryCategory,
    MemoryManagementAPI,
    MemoryRecord,
    MemoryService,
    MemorySource,
    MemoryValidationError,
)


@pytest.fixture
def temp_management_api():
    """Provides a fresh MemoryManagementAPI backed by a temporary SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory_mgmt.db")
        service = MemoryService(db_path=db_path)
        api = MemoryManagementAPI(memory_service=service)
        yield api
        api.close()


def test_list_and_search_memories(temp_management_api):
    api = temp_management_api

    api.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    api.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="location",
        value="Chennai",
    )

    # List all
    all_recs = api.list_memories(active_only=True)
    assert len(all_recs) == 2

    # List by category
    pref_recs = api.list_memories(category=MemoryCategory.USER_PREFERENCE)
    assert len(pref_recs) == 1
    assert pref_recs[0].key == "primary_language"

    # Search
    search_res = api.search_memories("Chennai")
    assert len(search_res) == 1
    assert search_res[0].value == "Chennai"

    # Search non-matching query
    assert len(api.search_memories("NonExistentKeyword")) == 0


def test_get_memory_by_id_and_key(temp_management_api):
    api = temp_management_api

    created = api.create_memory(
        category=MemoryCategory.USER_FACT,
        key="favorite_color",
        value="Blue",
    )

    # Get by ID
    by_id = api.get_memory_by_id(created.id)
    assert by_id is not None
    assert by_id.value == "Blue"

    # Get by key
    by_key = api.get_memory_by_key(MemoryCategory.USER_FACT, "favorite_color")
    assert by_key is not None
    assert by_key.id == created.id

    # Invalid ID/key lookups
    assert api.get_memory_by_id("non-existent-id") is None
    assert api.get_memory_by_key(MemoryCategory.USER_FACT, "non_existent_key") is None


def test_create_and_update_memory(temp_management_api):
    api = temp_management_api

    created = api.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="editor",
        value="VSCode",
        confidence=0.9,
    )

    updated = api.update_memory(
        memory_id=created.id,
        value="Neovim",
        confidence=1.0,
    )

    assert updated is not None
    assert updated.value == "Neovim"
    assert updated.confidence == 1.0

    retrieved = api.get_memory_by_id(created.id)
    assert retrieved.value == "Neovim"


def test_forget_and_reactivate_memory(temp_management_api):
    api = temp_management_api

    created = api.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="role",
        value="Developer",
    )

    # Soft-deactivate (forget)
    assert api.forget_memory(created.id) is True
    assert api.get_memory_by_key(MemoryCategory.USER_PROFILE, "role", active_only=True) is None

    # Reactivate
    reactivated = api.reactivate_memory(created.id)
    assert reactivated is not None
    assert reactivated.is_active is True
    assert api.get_memory_by_key(MemoryCategory.USER_PROFILE, "role", active_only=True) is not None


def test_permanently_delete_memory(temp_management_api):
    api = temp_management_api

    created = api.create_memory(
        category=MemoryCategory.CONTEXT_RULE,
        key="temp_rule",
        value="Be concise",
    )

    assert api.permanently_delete_memory(created.id) is True
    assert api.get_memory_by_id(created.id) is None


def test_inspect_metadata_confidence_source(temp_management_api):
    api = temp_management_api

    meta_str = '{"source_detail": "chat_prompt", "tags": ["pref"]}'
    created = api.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="theme",
        value="dark",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.95,
        metadata_json=meta_str,
    )

    # Inspect metadata
    meta_dict = api.inspect_memory_metadata(created.id)
    assert meta_dict.get("source_detail") == "chat_prompt"
    assert meta_dict.get("tags") == ["pref"]

    # Inspect confidence & source
    conf_info = api.inspect_memory_confidence_source(created.id)
    assert conf_info["exists"] is True
    assert conf_info["confidence"] == 0.95
    assert conf_info["source"] == "explicit_user_input"
    assert conf_info["created_at"] > 0


def test_inspect_expiration(temp_management_api):
    api = temp_management_api

    now = time.time()
    created_exp = api.create_memory(
        category=MemoryCategory.CONTEXT_RULE,
        key="ttl_rule",
        value="Temporary Rule",
        expires_at=now + 3600.0,
    )

    info = api.inspect_expiration(created_exp.id)
    assert info["exists"] is True
    assert info["is_expired"] is False
    assert info["remaining_seconds"] > 0.0

    # Expired memory inspection
    created_expired = api.create_memory(
        category=MemoryCategory.CONTEXT_RULE,
        key="expired_rule",
        value="Expired Rule",
        expires_at=now + 0.05,
    )
    time.sleep(0.1)

    info_expired = api.inspect_expiration(created_expired.id)
    assert info_expired["is_expired"] is True
    assert info_expired["remaining_seconds"] == 0.0


def test_inspect_supersession_state(temp_management_api):
    api = temp_management_api

    service = api.memory_service
    rec1 = service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    rec2 = service.supersede_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Python",
    )

    state1 = api.inspect_supersession_state(rec1.id)
    assert state1["exists"] is True
    assert state1["is_current_for_key"] is True
    assert state1["latest_active_memory_id"] == rec2.id


def test_security_sanitization_and_fail_safe(temp_management_api):
    api = temp_management_api

    # Credential creation fails closed
    with pytest.raises(MemoryValidationError):
        api.create_memory(
            category=MemoryCategory.USER_FACT,
            key="api_key",
            value="sk-1234567890abcdef1234567890",
        )

    # Empty query search returns empty list without error
    assert api.search_memories("") == []
    assert api.search_memories(None) == []

    # Non-existent ID inspection fail-safes
    conf_nonexistent = api.inspect_memory_confidence_source("missing-id")
    assert conf_nonexistent["exists"] is False

    exp_nonexistent = api.inspect_expiration("missing-id")
    assert exp_nonexistent["exists"] is False

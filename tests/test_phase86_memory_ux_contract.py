import os
import tempfile
import time
import pytest

from backend.agent.core import AgentCore
from backend.memory import (
    CorrectionCandidate,
    CorrectionDecisionOutcome,
    MemoryCategory,
    MemoryCommandIntent,
    MemoryCommandResult,
    MemoryCorrectionWorkflow,
    MemoryManagementAPI,
    MemoryObservabilityService,
    MemoryRecord,
    MemoryService,
    MemorySource,
    MemoryUXFormatter,
    MemoryUXResponse,
    MemoryUXStatus,
    format_confidence_level,
    format_provenance_source,
)
from backend.storage.session_store import SessionStore


@pytest.fixture
def contract_setup():
    """Fixture providing MemoryUXFormatter, MemoryService, and MemoryObservabilityService."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_p = os.path.join(tmpdir, "test_contract.db")
        obs_p = os.path.join(tmpdir, "test_contract_obs.db")
        obs = MemoryObservabilityService(db_path=obs_p)
        service = MemoryService(db_path=db_p, observability=obs)
        formatter = MemoryUXFormatter()
        yield formatter, service, obs
        service.close()


@pytest.mark.asyncio
async def test_a_list_success(contract_setup):
    formatter, service, _ = contract_setup
    service.create_memory(category=MemoryCategory.USER_PROFILE, key="name", value="Joshva")
    service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")

    recs = service.list_memories(active_only=True)
    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_LIST, message="Listed", data=recs)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.SUCCESS
    assert "You currently have 2 active memories:" in ux_res.response_text
    assert "ABOUT ME" in ux_res.response_text
    assert "PREFERENCES" in ux_res.response_text


@pytest.mark.asyncio
async def test_b_list_empty(contract_setup):
    formatter, service, _ = contract_setup
    recs = service.list_memories(active_only=True)
    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_LIST, message="Listed", data=recs)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.EMPTY
    assert "I don't have any saved memories about you yet." in ux_res.response_text


@pytest.mark.asyncio
async def test_c_search_success(contract_setup):
    formatter, service, _ = contract_setup
    service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")
    api = MemoryManagementAPI(memory_service=service)
    recs = api.search_memories("Java")

    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_SEARCH, message="Found", data=recs)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.SUCCESS
    assert "Java" in ux_res.response_text


@pytest.mark.asyncio
async def test_d_search_empty(contract_setup):
    formatter, _, _ = contract_setup
    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_SEARCH, message="Not found", data=[])
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.EMPTY
    assert "I couldn't find a saved memory matching that." in ux_res.response_text


@pytest.mark.asyncio
async def test_e_lookup_success(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PROFILE, key="name", value="Joshva")
    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_LOOKUP, message="Found", data=[rec])
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.SUCCESS
    assert "Joshva" in ux_res.response_text


@pytest.mark.asyncio
async def test_f_create_success(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")
    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_CREATE, message="Created", data=rec)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.SUCCESS
    assert "I'll remember that you prefer Java." in ux_res.response_text


@pytest.mark.asyncio
async def test_g_update_success(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")
    res = MemoryCommandResult(
        success=True,
        intent=MemoryCommandIntent.MEMORY_UPDATE,
        message="I've updated your preferred primary language from Java to Python.",
        data=rec,
    )
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.UPDATED
    assert "I've updated your preferred primary language from Java to Python." in ux_res.response_text


@pytest.mark.asyncio
async def test_h_forget_success(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")
    service.forget_memory(rec.id)

    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_FORGET, message="Forgotten", data=rec)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.FORGOTTEN
    assert "I've forgotten your primary language preference." in ux_res.response_text


@pytest.mark.asyncio
async def test_i_forget_all_confirmation(contract_setup):
    formatter, _, _ = contract_setup
    res = MemoryCommandResult(
        success=False,
        intent=MemoryCommandIntent.MEMORY_FORGET_ALL,
        message="Confirmation required.",
        confirmation_required=True,
        confirmation_token="token_clean_123",
    )
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.PENDING_CONFIRMATION
    assert "This will remove all active memories." in ux_res.response_text
    assert ux_res.confirmation_required is True
    assert ux_res.confirmation_token == "token_clean_123"
    assert len(ux_res.actions) == 2


@pytest.mark.asyncio
async def test_j_confirmation_cancellation(contract_setup):
    formatter, service, _ = contract_setup
    service.create_memory(category=MemoryCategory.USER_PROFILE, key="name", value="Joshva")

    res = MemoryCommandResult(
        success=False,
        intent=MemoryCommandIntent.MEMORY_FORGET_ALL,
        message="Operation cancelled by user.",
    )
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.ERROR
    assert service.count_memories(active_only=True) == 1


@pytest.mark.asyncio
async def test_k_confirmation_execution(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PROFILE, key="name", value="Joshva")
    service.forget_memory(rec.id)

    res = MemoryCommandResult(
        success=True,
        intent=MemoryCommandIntent.MEMORY_FORGET_ALL,
        message="All memories cleared.",
    )
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.FORGOTTEN
    assert "All stored active memories have been removed." in ux_res.response_text


@pytest.mark.asyncio
async def test_l_reactivate(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")

    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_REACTIVATE, message="Reactivated", data=rec)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.REACTIVATED
    assert "Your primary language preference is active again." in ux_res.response_text


@pytest.mark.asyncio
async def test_m_explain_provenance(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")
    api = MemoryManagementAPI(memory_service=service)
    details = [api.inspect_memory_confidence_source(rec.id)]

    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_EXPLAIN, message="Inspection", data=details)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.SUCCESS
    assert "explicitly provided by you." in ux_res.response_text


@pytest.mark.asyncio
async def test_n_confidence_formatting():
    assert format_confidence_level(0.95) == "High"
    assert format_confidence_level(0.55) == "Medium"
    assert format_confidence_level(0.25) == "Low"


@pytest.mark.asyncio
async def test_o_expiration_formatting(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.CONTEXT_RULE, key="temp_rule", value="temp", expires_at=time.time() + 100)
    api = MemoryManagementAPI(memory_service=service)
    details = [api.inspect_expiration(rec.id)]

    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_EXPIRATION, message="Expiration", data=details)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.SUCCESS
    assert "Status: Active" in ux_res.response_text


@pytest.mark.asyncio
async def test_p_correction_success(contract_setup):
    formatter, service, _ = contract_setup
    service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")
    workflow = MemoryCorrectionWorkflow(memory_service=service)

    cand = CorrectionCandidate(category=MemoryCategory.USER_PREFERENCE, key="primary_language", new_value="Python")
    dec = workflow.execute_correction(cand)

    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_UPDATE, message=dec.message, data=service.get_memory_by_key(MemoryCategory.USER_PREFERENCE, "primary_language"))
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.UPDATED
    assert "Python" in ux_res.response_text


@pytest.mark.asyncio
async def test_q_correction_ambiguity(contract_setup):
    formatter, service, _ = contract_setup
    workflow = MemoryCorrectionWorkflow(memory_service=service)
    cand = CorrectionCandidate(category=MemoryCategory.USER_PREFERENCE, key="ambiguous_key", new_value="Python")
    dec = workflow.execute_correction(cand)

    res = MemoryCommandResult(success=False, intent=MemoryCommandIntent.MEMORY_UPDATE, message=dec.message)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.AMBIGUOUS
    assert "more than one possible memory" in ux_res.response_text


@pytest.mark.asyncio
async def test_r_correction_no_target(contract_setup):
    formatter, service, _ = contract_setup
    res = MemoryCommandResult(success=False, intent=MemoryCommandIntent.MEMORY_UPDATE, message="Target record not found.")
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.EMPTY
    assert "I couldn't find a saved memory to update." in ux_res.response_text


@pytest.mark.asyncio
async def test_s_consent_pending(contract_setup):
    formatter, _, _ = contract_setup
    ux_res = formatter.format_candidate_approval_request("user_preference", "primary_language", "Java")

    assert ux_res.status == MemoryUXStatus.PENDING_APPROVAL
    assert "Would you like me to remember that?" in ux_res.response_text
    assert len(ux_res.actions) == 2


@pytest.mark.asyncio
async def test_t_consent_approval(contract_setup):
    formatter, service, _ = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PREFERENCE, key="primary_language", value="Java")
    res = MemoryCommandResult(success=True, intent=MemoryCommandIntent.MEMORY_CREATE, message="Approved and created.", data=rec)
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.SUCCESS
    assert "I'll remember that you prefer Java." in ux_res.response_text


@pytest.mark.asyncio
async def test_u_consent_rejection(contract_setup):
    formatter, _, _ = contract_setup
    res = MemoryCommandResult(success=False, intent=MemoryCommandIntent.MEMORY_CREATE, message="User declined candidate.")
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.ERROR
    assert "User declined candidate." in ux_res.response_text


@pytest.mark.asyncio
async def test_v_security_rejection(contract_setup):
    formatter, _, _ = contract_setup

    res1 = MemoryCommandResult(success=False, intent=MemoryCommandIntent.MEMORY_CREATE, message="Security Violation: Sensitive credentials detected.")
    ux_res1 = formatter.format_command_result(res1)
    assert ux_res1.status == MemoryUXStatus.SECURITY_REJECTED
    assert "sensitive credential information" in ux_res1.response_text

    res2 = MemoryCommandResult(success=False, intent=MemoryCommandIntent.MEMORY_CREATE, message="Security Violation: System override attempt.")
    ux_res2 = formatter.format_command_result(res2)
    assert ux_res2.status == MemoryUXStatus.SECURITY_REJECTED
    assert "memory instruction" in ux_res2.response_text


@pytest.mark.asyncio
async def test_w_technical_question_false_positive_protection(contract_setup):
    with tempfile.TemporaryDirectory() as tmpdir:
        s = MemoryService(db_path=os.path.join(tmpdir, "fp.db"))
        agent = AgentCore(memory_service=s, enable_memory=True)

        cmd = agent.memory_command_parser.parse("How do I change a variable in Java?")
        assert cmd.intent.value == "unknown"

        s.close()


@pytest.mark.asyncio
async def test_x_malformed_command(contract_setup):
    formatter, _, _ = contract_setup
    res = MemoryCommandResult(success=False, intent=MemoryCommandIntent.UNKNOWN, message="Unknown command.")
    ux_res = formatter.format_command_result(res)

    assert ux_res.status == MemoryUXStatus.ERROR or ux_res.status == MemoryUXStatus.UNKNOWN


@pytest.mark.asyncio
async def test_y_internal_exception_sanitization(contract_setup):
    formatter, _, _ = contract_setup
    safe_err = formatter._sanitize_error("SQLite connection error at /Users/novus/test.db")

    assert "SQLite" not in safe_err
    assert "/Users/novus" not in safe_err
    assert safe_err == "A storage or memory processing error occurred."


@pytest.mark.asyncio
async def test_z_session_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_db = os.path.join(tmpdir, "contract_mem.db")
        sess_db = os.path.join(tmpdir, "contract_sess.db")

        mem_service = MemoryService(db_path=mem_db)
        session_store = SessionStore(db_path=sess_db)

        agent = AgentCore(session_store=session_store, memory_service=mem_service, enable_memory=True)
        await agent.process_intent("remember that I prefer Java")

        msgs = session_store.get_messages(agent.session_id)
        assert len(msgs) == 2
        assert "I'll remember that you prefer Java." in msgs[1]["content"]

        mem_service.close()
        session_store.close()


@pytest.mark.asyncio
async def test_aa_tool_governance_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = MemoryService(db_path=os.path.join(tmpdir, "tool_gov.db"))
        agent = AgentCore(memory_service=s, enable_memory=True)

        assert len(agent.tool_registry.get_all_tool_schemas()) == 1
        await agent.process_intent("what do you remember about me?")
        assert len(agent.tool_registry.get_all_tool_schemas()) == 1

        s.close()


@pytest.mark.asyncio
async def test_ab_token_governor_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = MemoryService(db_path=os.path.join(tmpdir, "tok_gov.db"))
        agent = AgentCore(memory_service=s, enable_memory=True)

        assert agent.governor is not None
        s.close()


@pytest.mark.asyncio
async def test_ac_audit_integrity(contract_setup):
    _, service, obs = contract_setup
    rec = service.create_memory(category=MemoryCategory.USER_PROFILE, key="name", value="Joshva")

    events = obs.store.get_events_for_memory(rec.id)
    assert len(events) > 0
    assert "sk-" not in str(events[0].reason)

# Security Guidelines

## Secrets Management
All sensitive credentials must be stored locally in `.env` or the macOS Keychain. Code must never hardcode secrets.

## API Key Protection
API keys must not be exposed in logs, outputs, or error traces. The logging formatter must mask known secret patterns.

## OAuth Security
Integrations like Gmail/Calendar must use local OAuth2 flows. Tokens should be stored securely with restricted file permissions.

## Tool Permissions
Tools are strictly categorized by risk level. The agent cannot escalate a tool's permission level. 

## Confirmation Requirements
Any "Write", "Delete", "Send", or "Execute" action (e.g., sending an email, deleting a file) strictly requires explicit user confirmation before execution.

## Command Execution Restrictions
The `run_command` or equivalent system tools must use strict sanitization. Dangerous commands (e.g., `rm -rf`, `mkfs`) are blacklisted at the tool level.

## Audit Logging
Every action the agent takes (planning, tool execution, memory read/write) is logged locally for audit and review.

## Prompt Injection Protection
User inputs must be treated as untrusted. System prompts dictate that any instruction contradicting core security rules must be ignored. 

## Emergency Stop
The frontend and backend must support a global "kill switch" that immediately halts all agent reasoning and tool execution.

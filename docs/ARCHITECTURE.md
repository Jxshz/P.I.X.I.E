# P.I.X.I.E. Architecture

## System Architecture
P.I.X.I.E. follows a highly modular, decoupled client-server model to separate the frontend interface (voice/UI) from the backend reasoning core and execution layer.
- **Frontend**: Handles user interactions, voice processing, and UI presentation.
- **Backend**: Python-based central intelligence, orchestrating reasoning, memory, tools, and integrations.

## Agent Architecture
The agent core acts as the "brain".
1. **Understand Intent**: Parses and contextualizes natural language queries.
2. **Plan**: Deconstructs requests into logical execution steps.
3. **Select Tools**: Dynamically maps required capabilities to the tool registry.
4. **Execution & Verification**: Executes actions and observes outputs to ensure goal completion.

## Frontend/Backend Communication
Communication will be handled via asynchronous REST/WebSocket APIs, ensuring that long-running tasks (e.g., web research, file operations) do not block the user interface. Structured JSON payloads will transport intents, status updates, and agent responses.

## Future Tool System
Tools will be implemented as modular plugins with standard interfaces (e.g., `execute()`, `validate()`, `describe()`). The agent will have access to a Tool Registry that exposes descriptions and parameters for dynamic discovery.

## Permission Layer
A rigid permission middleware will sit between the Agent Core and the Tool System. Before any tool executes, the permission layer intercepts the call to verify if the tool requires explicit user confirmation.

## Memory Layer
- **Short-term Memory**: Manages the context window of the current conversation.
- **Persistent/Long-term Memory**: A local vector database (or similar persistence layer) to recall past facts, user preferences, and workflow history.

## Integrations
Integrations (e.g., Gmail, Calendar) will reside in their own isolated modules using official SDKs and OAuth2, completely abstracted from the reasoning core.

## Future iPhone Architecture
The iPhone companion will act as a lightweight client interacting securely with the macOS backend (or a cloud-hosted variant) via encrypted WebSockets or push notifications. Processing will primarily remain on the main system to preserve battery and leverage desktop capabilities.

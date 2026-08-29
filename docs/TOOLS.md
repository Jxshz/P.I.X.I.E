# Planned Tool Catalogue

## Low Risk (Read-Only, No Confirmation Required)
- **Web Search**: Retrieve information from search engines.
- **Read File**: Read local text or configuration files.
- **List Directory**: View contents of the filesystem.
- **Read Emails**: Fetch recent emails.
- **View Calendar**: Check upcoming events.
- **Memory Query**: Retrieve facts from the persistent memory base.

## Medium Risk (State Modification, Confirmation Optional/Configurable)
- **Draft Email**: Prepare an email without sending.
- **Create Task**: Add an item to a local task list.
- **Create File**: Write new files (non-system directories).

## High Risk (Destructive/External Action, Explicit Confirmation REQUIRED)
- **Send Email**: Dispatch communication to external parties.
- **Execute Shell Command**: Run arbitrary commands on macOS.
- **Delete File**: Remove files from the filesystem.
- **System Settings**: Modify macOS system preferences.

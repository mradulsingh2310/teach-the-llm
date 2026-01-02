"""JSON-based conversation logger for agent sessions."""

import json
from pathlib import Path
from typing import Literal

from agent.models import ConversationLog, ConversationTurn, Session, SessionMetadata


class ConversationLogger:
    """Handles logging of conversation sessions to a JSON file."""

    def __init__(self, file_path: str = "conversations.json"):
        """Initialize the logger with a file path.

        Args:
            file_path: Path to the JSON file for storing conversations.
        """
        self.file_path = Path(file_path)
        self.log = self._load_or_create_log()

    def _load_or_create_log(self) -> ConversationLog:
        """Load existing log from file or create a new one."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                return ConversationLog.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not load existing log: {e}. Creating new log.")
                return ConversationLog()
        return ConversationLog()

    def _save(self) -> None:
        """Save the current log to the JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.log.model_dump(), f, indent=2)

    def create_session(
        self,
        model_id: str = "",
        agent_type: str = "property_agent",
        channel: str = "cli",
    ) -> str:
        """Create a new conversation session.

        Args:
            model_id: The model ID being used.
            agent_type: Type of agent (default: property_agent).
            channel: Communication channel (default: cli).

        Returns:
            The session ID of the newly created session.
        """
        session = Session(
            metadata=SessionMetadata(
                agent_type=agent_type,
                model_id=model_id,
                channel=channel,
            )
        )
        self.log.add_session(session)
        self._save()
        return session.id

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by its ID.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The session if found, None otherwise.
        """
        return self.log.get_session(session_id)

    def add_turn(self, session_id: str, turn: ConversationTurn) -> bool:
        """Add a conversation turn to a session.

        Args:
            session_id: The ID of the session to add the turn to.
            turn: The conversation turn to add.

        Returns:
            True if the turn was added successfully, False if session not found.
        """
        session = self.log.get_session(session_id)
        if session is None:
            return False

        session.add_turn(turn)
        self._save()
        return True

    def end_session(
        self,
        session_id: str,
        status: Literal["completed", "abandoned", "error"] = "completed",
    ) -> bool:
        """Mark a session as ended.

        Args:
            session_id: The ID of the session to end.
            status: The final status of the session.

        Returns:
            True if the session was ended successfully, False if session not found.
        """
        session = self.log.get_session(session_id)
        if session is None:
            return False

        session.end_session(status)
        self._save()
        return True

    def add_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        turn_id: int | None = None,
    ) -> bool:
        """Add an error to a session.

        Args:
            session_id: The ID of the session.
            error_type: The type of error.
            error_message: The error message.
            turn_id: Optional turn ID where the error occurred.

        Returns:
            True if the error was added successfully, False if session not found.
        """
        session = self.log.get_session(session_id)
        if session is None:
            return False

        session.add_error(error_type, error_message, turn_id)
        self._save()
        return True

    def get_next_turn_id(self, session_id: str) -> int:
        """Get the next turn ID for a session.

        Args:
            session_id: The ID of the session.

        Returns:
            The next turn ID (1-indexed), or 1 if session not found.
        """
        session = self.log.get_session(session_id)
        if session is None:
            return 1
        return len(session.conversation) + 1

    def get_all_sessions(self) -> list[Session]:
        """Get all sessions.

        Returns:
            List of all sessions.
        """
        return self.log.sessions

    def get_active_sessions(self) -> list[Session]:
        """Get all active sessions.

        Returns:
            List of active sessions.
        """
        return [s for s in self.log.sessions if s.status == "active"]

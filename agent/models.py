"""Data models for conversation logging."""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a single tool call with input and output."""

    tool_name: str
    tool_use_id: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    status: Literal["success", "error"] = "success"
    error_message: str | None = None


class Usage(BaseModel):
    """Token usage for a single turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ConversationTurn(BaseModel):
    """Represents a single turn in the conversation."""

    turn_id: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    role: Literal["user", "assistant"]
    content: str
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"] | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0
    tools_called: list[ToolCall] = Field(default_factory=list)


class ToolsSummary(BaseModel):
    """Summary of tools usage across the session."""

    tool_counts: dict[str, int] = Field(default_factory=dict)
    tool_success_rate: dict[str, float] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    """Summary statistics for a session."""

    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    total_tools_called: int = 0
    tools_summary: ToolsSummary = Field(default_factory=ToolsSummary)


class SessionMetadata(BaseModel):
    """Metadata about the session."""

    agent_type: str = "property_agent"
    model_id: str = ""
    channel: str = "cli"


class SessionError(BaseModel):
    """Represents an error that occurred during the session."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    error_type: str
    error_message: str
    turn_id: int | None = None


class Session(BaseModel):
    """Represents a full conversation session."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(default_factory=lambda: f"Session-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: str | None = None
    status: Literal["active", "completed", "abandoned", "error"] = "active"
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)
    summary: SessionSummary = Field(default_factory=SessionSummary)
    conversation: list[ConversationTurn] = Field(default_factory=list)
    errors: list[SessionError] = Field(default_factory=list)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn and update summary statistics."""
        self.conversation.append(turn)
        self.summary.total_turns = len(self.conversation)
        self.summary.total_input_tokens += turn.usage.input_tokens
        self.summary.total_output_tokens += turn.usage.output_tokens
        self.summary.total_tokens += turn.usage.total_tokens
        self.summary.total_latency_ms += turn.latency_ms

        # Update tools summary
        for tool_call in turn.tools_called:
            self.summary.total_tools_called += 1
            tool_name = tool_call.tool_name

            if tool_name not in self.summary.tools_summary.tool_counts:
                self.summary.tools_summary.tool_counts[tool_name] = 0
            self.summary.tools_summary.tool_counts[tool_name] += 1

    def end_session(self, status: Literal["completed", "abandoned", "error"] = "completed") -> None:
        """Mark the session as ended."""
        self.ended_at = datetime.utcnow().isoformat()
        self.status = status

        # Calculate success rates for tools
        for tool_name in self.summary.tools_summary.tool_counts:
            total_calls = self.summary.tools_summary.tool_counts[tool_name]
            success_count = sum(
                1
                for turn in self.conversation
                for tc in turn.tools_called
                if tc.tool_name == tool_name and tc.status == "success"
            )
            self.summary.tools_summary.tool_success_rate[tool_name] = (
                success_count / total_calls if total_calls > 0 else 0.0
            )

    def add_error(self, error_type: str, error_message: str, turn_id: int | None = None) -> None:
        """Add an error to the session."""
        self.errors.append(
            SessionError(error_type=error_type, error_message=error_message, turn_id=turn_id)
        )


class ConversationLog(BaseModel):
    """The full log structure containing all sessions."""

    sessions: list[Session] = Field(default_factory=list)

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        for session in self.sessions:
            if session.id == session_id:
                return session
        return None

    def add_session(self, session: Session) -> None:
        """Add a new session."""
        self.sessions.append(session)

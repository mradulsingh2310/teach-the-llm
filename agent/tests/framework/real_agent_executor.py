"""
Real Agent Test Executor for AI Agent Testing Framework.

This module provides the RealAgentTestExecutor class that:
- Instantiates and invokes the actual PropertyAgent (LangChain + llama-server)
- Feeds user messages from test scenarios
- Captures which tools were actually called by the LLM
- Captures actual parameters passed to each tool
- Tracks tool execution results and side effects
- Supports multi-turn conversations

Unlike MockAgent tests, this executor tests REAL LLM behavior.
Uses LangChain with llama-server for local model inference (FunctionGemma).
"""

import json
import logging
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv

# Add parent directories for imports
import sys
_agent_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_agent_root))

# Load .env file (safety net in case this module is imported directly)
_env_path = _agent_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

from agent.models import ToolCall, ConversationTurn, Session

from .models import (
    ExpectedToolCall,
    MatchStrategy,
    ParameterMatcher,
    SuiteResult,
    TestCase,
    TestResult,
    TestSuite,
    ToolCallMatchResult,
    ToolMetrics,
)
from .matchers import (
    calculate_tool_metrics,
    match_response,
    match_tool_call,
    validate_sequence,
)


# =============================================================================
# Data Classes for Captured Data
# =============================================================================

@dataclass
class CapturedToolCall:
    """Represents a tool call captured during real agent execution."""
    tool_name: str
    tool_use_id: str
    input_params: dict[str, Any]
    output: dict[str, Any] | None = None
    status: str = "success"
    error_message: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility with existing matchers."""
        return {
            "tool_name": self.tool_name,
            "tool_use_id": self.tool_use_id,
            "input": self.input_params,
            "output": self.output,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class TurnCapture:
    """Captures all data from a single conversation turn."""
    turn_number: int
    user_message: str
    assistant_response: str
    tool_calls: list[CapturedToolCall] = field(default_factory=list)
    latency_ms: int = 0


@dataclass
class ConversationCapture:
    """Captures the full conversation trajectory."""
    session_id: str
    turns: list[TurnCapture] = field(default_factory=list)
    total_tool_calls: list[CapturedToolCall] = field(default_factory=list)

    def get_all_tool_calls(self) -> list[CapturedToolCall]:
        """Get all tool calls across all turns."""
        return self.total_tool_calls

    def get_tool_calls_by_name(self, tool_name: str) -> list[CapturedToolCall]:
        """Get all tool calls for a specific tool."""
        return [tc for tc in self.total_tool_calls if tc.tool_name == tool_name]


# =============================================================================
# Side Effect Tracking
# =============================================================================

@dataclass
class SideEffectCheck:
    """Defines a side effect to check after test execution."""
    check_type: str  # "file_exists", "file_contains", "file_modified"
    path: str
    expected_content: str | None = None
    description: str | None = None


@dataclass
class SideEffectResult:
    """Result of a side effect check."""
    check: SideEffectCheck
    passed: bool
    actual_value: str | None = None
    error_message: str | None = None


class SideEffectTracker:
    """Tracks and verifies side effects from tool execution."""

    # Path to tour bookings file (side effect from create_appointment)
    TOUR_BOOKINGS_PATH = _agent_root / "data" / "tour_bookings.txt"

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._backup_files: dict[str, str] = {}

    def backup_file(self, file_path: str | Path) -> bool:
        """Backup a file before test execution."""
        path = Path(file_path)
        if path.exists():
            backup_path = f"{path}.backup"
            shutil.copy2(path, backup_path)
            self._backup_files[str(path)] = backup_path
            self.logger.debug(f"Backed up {path} to {backup_path}")
            return True
        return False

    def restore_file(self, file_path: str | Path) -> bool:
        """Restore a file from backup."""
        path = Path(file_path)
        backup_path = self._backup_files.get(str(path))
        if backup_path and Path(backup_path).exists():
            shutil.copy2(backup_path, path)
            os.remove(backup_path)
            del self._backup_files[str(path)]
            self.logger.debug(f"Restored {path} from backup")
            return True
        elif str(path) in self._backup_files:
            # File didn't exist before, remove it
            if path.exists():
                os.remove(path)
            del self._backup_files[str(path)]
            return True
        return False

    def cleanup_backups(self) -> None:
        """Remove all backup files."""
        for path, backup in list(self._backup_files.items()):
            if Path(backup).exists():
                os.remove(backup)
            self._backup_files.pop(path, None)

    def check_file_exists(self, file_path: str | Path) -> bool:
        """Check if a file exists."""
        return Path(file_path).exists()

    def check_file_contains(self, file_path: str | Path, content: str) -> tuple[bool, str | None]:
        """Check if a file contains specific content."""
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {path}"

        try:
            with open(path, "r") as f:
                file_content = f.read()
            if content in file_content:
                return True, None
            return False, f"Content not found in file. File contains: {file_content[:500]}..."
        except Exception as e:
            return False, str(e)

    def get_file_content(self, file_path: str | Path) -> str | None:
        """Get the content of a file."""
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception:
            return None

    def check_tour_booking_created(
        self,
        name: str | None = None,
        date: str | None = None,
        confirmation_number: str | None = None
    ) -> tuple[bool, str | None]:
        """Check if a tour booking was written to the bookings file."""
        if not self.TOUR_BOOKINGS_PATH.exists():
            return False, "Tour bookings file does not exist"

        try:
            with open(self.TOUR_BOOKINGS_PATH, "r") as f:
                content = f.read()

            # Check for presence of expected data
            checks_passed = True
            missing = []

            if name and name not in content:
                checks_passed = False
                missing.append(f"name '{name}'")

            if date and date not in content:
                checks_passed = False
                missing.append(f"date '{date}'")

            if confirmation_number and confirmation_number not in content:
                checks_passed = False
                missing.append(f"confirmation '{confirmation_number}'")

            if not checks_passed:
                return False, f"Missing in bookings file: {', '.join(missing)}"

            return True, None

        except Exception as e:
            return False, str(e)

    def verify_side_effects(self, checks: list[SideEffectCheck]) -> list[SideEffectResult]:
        """Verify a list of side effect checks."""
        results = []
        for check in checks:
            result = self._verify_single_check(check)
            results.append(result)
        return results

    def _verify_single_check(self, check: SideEffectCheck) -> SideEffectResult:
        """Verify a single side effect check."""
        try:
            if check.check_type == "file_exists":
                passed = self.check_file_exists(check.path)
                return SideEffectResult(
                    check=check,
                    passed=passed,
                    error_message=None if passed else f"File does not exist: {check.path}"
                )

            elif check.check_type == "file_contains":
                if check.expected_content is None:
                    return SideEffectResult(
                        check=check,
                        passed=False,
                        error_message="expected_content is required for file_contains check"
                    )
                passed, error = self.check_file_contains(check.path, check.expected_content)
                return SideEffectResult(
                    check=check,
                    passed=passed,
                    actual_value=self.get_file_content(check.path),
                    error_message=error
                )

            elif check.check_type == "file_modified":
                # Check if file was modified (compare with backup)
                backup = self._backup_files.get(check.path)
                if backup is None:
                    return SideEffectResult(
                        check=check,
                        passed=False,
                        error_message=f"No backup found for {check.path}. Call backup_file first."
                    )

                current = self.get_file_content(check.path)
                original = self.get_file_content(backup) if Path(backup).exists() else ""

                passed = current != original
                return SideEffectResult(
                    check=check,
                    passed=passed,
                    actual_value=current,
                    error_message=None if passed else "File was not modified"
                )

            else:
                return SideEffectResult(
                    check=check,
                    passed=False,
                    error_message=f"Unknown check type: {check.check_type}"
                )

        except Exception as e:
            return SideEffectResult(
                check=check,
                passed=False,
                error_message=str(e)
            )


# =============================================================================
# Tool Call Interceptor
# =============================================================================

class ToolCallInterceptor:
    """
    Intercepts and captures tool calls made by the real agent.

    This wraps the agent's tool functions to capture:
    - Tool name
    - Input parameters
    - Output results
    - Execution status

    Works with LangChain tools that use the .invoke() method.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.captured_calls: list[CapturedToolCall] = []
        self._original_tools: dict[str, Any] = {}

    def reset(self) -> None:
        """Reset captured calls for a new test."""
        self.captured_calls = []

    def wrap_tool(self, tool_name: str, tool_func: Any) -> Any:
        """
        Wrap a LangChain tool to capture its calls.

        Args:
            tool_name: Name of the tool
            tool_func: The original LangChain tool

        Returns:
            Wrapper object that captures calls and delegates to original tool
        """
        self._original_tools[tool_name] = tool_func
        interceptor = self

        class ToolWrapper:
            """Wrapper class that intercepts tool invocations."""

            def __init__(self, original_tool, tool_name: str, interceptor_ref):
                self._original_tool = original_tool
                self._tool_name = tool_name
                self._interceptor = interceptor_ref
                # Copy attributes from original tool
                self.name = getattr(original_tool, 'name', tool_name)
                self.description = getattr(original_tool, 'description', '')
                self.args_schema = getattr(original_tool, 'args_schema', None)

            def invoke(self, input_params: dict[str, Any], config=None) -> Any:
                """Intercept invoke calls to capture tool usage."""
                call_id = f"capture_{self._tool_name}_{len(self._interceptor.captured_calls)}"

                self._interceptor.logger.debug(f"Tool call intercepted: {self._tool_name}")
                self._interceptor.logger.debug(f"  Input: {json.dumps(input_params, default=str)}")

                captured = CapturedToolCall(
                    tool_name=self._tool_name,
                    tool_use_id=call_id,
                    input_params=input_params
                )

                try:
                    result = self._original_tool.invoke(input_params, config=config)
                    captured.output = result if isinstance(result, dict) else {"result": result}
                    captured.status = "success"
                    self._interceptor.logger.debug(f"  Output: {json.dumps(captured.output, default=str)[:200]}...")
                except Exception as e:
                    captured.status = "error"
                    captured.error_message = str(e)
                    self._interceptor.logger.error(f"  Error: {e}")
                    raise
                finally:
                    self._interceptor.captured_calls.append(captured)

                return result

            def __call__(self, *args, **kwargs):
                """Support direct function call syntax."""
                if args:
                    # If positional args, treat first as input dict
                    input_params = args[0] if isinstance(args[0], dict) else kwargs
                else:
                    input_params = kwargs
                return self.invoke(input_params)

        return ToolWrapper(tool_func, tool_name, interceptor)

    def wrap_agent_tools(self, agent) -> None:
        """
        Wrap all tools on an agent instance.

        Args:
            agent: PropertyAgent instance with LangChain tools
        """
        for tool_name, tool_func in list(agent.tool_mapping.items()):
            wrapped = self.wrap_tool(tool_name, tool_func)
            agent.tool_mapping[tool_name] = wrapped

    def restore_agent_tools(self, agent) -> None:
        """Restore original tool functions on the agent."""
        for tool_name, original_func in self._original_tools.items():
            if tool_name in agent.tool_mapping:
                agent.tool_mapping[tool_name] = original_func
        self._original_tools.clear()

    def get_captured_calls(self) -> list[CapturedToolCall]:
        """Get all captured tool calls."""
        return self.captured_calls.copy()

    def get_calls_by_name(self, tool_name: str) -> list[CapturedToolCall]:
        """Get captured calls for a specific tool."""
        return [c for c in self.captured_calls if c.tool_name == tool_name]


# =============================================================================
# Real Agent Test Executor
# =============================================================================

class RealAgentTestExecutor:
    """
    Executor for running test cases against the REAL PropertyAgent.

    This executor:
    - Instantiates the actual PropertyAgent with LangChain + llama-server backend
    - Feeds user messages from test scenarios
    - Captures actual tool calls made by the LLM
    - Captures actual parameters passed to tools
    - Tracks tool execution results
    - Verifies side effects (e.g., files written)
    - Supports multi-turn conversations

    Example usage:
        executor = RealAgentTestExecutor(
            model_id="functiongemma-270m-it",
            base_url="http://localhost:8080/v1"
        )

        result = executor.run_test_case(test_case)

        # Check what tools were called
        tool_calls = executor.get_captured_tool_calls()

        # Verify side effects
        assert executor.side_effects.check_file_exists("data/tour_bookings.txt")
    """

    def __init__(
        self,
        model_id: str = "functiongemma-270m-it",
        base_url: str = "http://localhost:8080/v1",
        agent_name: str = "Aria",
        property_name: str = "Sunset Apartments",
        logger: logging.Logger | None = None,
        backup_tour_bookings: bool = True,
        temperature: float = 0.7,
    ):
        """
        Initialize the real agent test executor.

        Args:
            model_id: Model name for logging (actual model loaded by llama-server)
            base_url: llama-server URL (default: http://localhost:8080/v1)
            agent_name: Name of the agent
            property_name: Name of the property
            logger: Logger instance
            backup_tour_bookings: Whether to backup/restore tour_bookings.txt
            temperature: Temperature for LLM generation
        """
        self.model_id = model_id
        self.base_url = base_url
        self.agent_name = agent_name
        self.property_name = property_name
        self.logger = logger or logging.getLogger(__name__)
        self.backup_tour_bookings = backup_tour_bookings
        self.temperature = temperature

        # Initialize trackers
        self.tool_interceptor = ToolCallInterceptor(self.logger)
        self.side_effects = SideEffectTracker(self.logger)

        # Will hold the agent instance during test execution
        self._agent = None
        self._current_capture: ConversationCapture | None = None

    def _create_agent(self):
        """Create a fresh PropertyAgent instance."""
        from agent.core import PropertyAgent
        import tempfile

        # Use a temp file for logging to avoid polluting main log
        log_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

        agent = PropertyAgent(
            model_id=self.model_id,
            base_url=self.base_url,
            agent_name=self.agent_name,
            property_name=self.property_name,
            log_file=log_file,
            temperature=self.temperature,
        )

        return agent

    def run_test_case(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case against the real agent.

        Args:
            test_case: The test case to run

        Returns:
            TestResult with detailed results
        """
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"REAL AGENT TEST: {test_case.id} - {test_case.name}")
        self.logger.info(f"{'=' * 60}")

        result = TestResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            passed=False,
        )

        # Check skip condition
        if test_case.skip:
            self.logger.info(f"Test SKIPPED: {test_case.skip_reason}")
            result.passed = True
            result.error_message = f"Skipped: {test_case.skip_reason}"
            return result

        start_time = time.time()
        session_id = None

        try:
            # Setup: backup files that may be modified
            if self.backup_tour_bookings:
                self.side_effects.backup_file(SideEffectTracker.TOUR_BOOKINGS_PATH)

            # Create fresh agent
            self._agent = self._create_agent()

            # Wrap tools for interception
            self.tool_interceptor.reset()
            self.tool_interceptor.wrap_agent_tools(self._agent)

            # Start session
            session_id = self._agent.start_session()
            self._current_capture = ConversationCapture(session_id=session_id)
            self.logger.info(f"Session started: {session_id}")

            # Determine conversation format
            if test_case.is_multi_turn():
                result = self._run_multi_turn_test(test_case, result, session_id)
            else:
                result = self._run_single_turn_test(test_case, result, session_id)

            # Capture all tool calls
            result.actual_tool_calls = [
                tc.to_dict() for tc in self.tool_interceptor.get_captured_calls()
            ]

            # End session
            self._agent.end_session(session_id, "completed")

        except Exception as e:
            self.logger.exception("Exception during real agent test execution")
            result.error_message = str(e)
            result.exception_type = type(e).__name__
            result.stack_trace = traceback.format_exc()
            result.failures.append(f"Exception: {e}")

            if session_id and self._agent:
                try:
                    self._agent.end_session(session_id, "error")
                except Exception:
                    pass

        finally:
            # Restore tools and cleanup
            if self._agent:
                self.tool_interceptor.restore_agent_tools(self._agent)

            # Restore backed up files
            if self.backup_tour_bookings:
                self.side_effects.restore_file(SideEffectTracker.TOUR_BOOKINGS_PATH)

        # Calculate timing
        result.duration_ms = int((time.time() - start_time) * 1000)
        result.ended_at = datetime.utcnow().isoformat()

        # Check latency constraint
        if test_case.max_latency_ms and result.duration_ms > test_case.max_latency_ms:
            result.failures.append(
                f"Latency exceeded: {result.duration_ms}ms > {test_case.max_latency_ms}ms"
            )
            result.passed = False

        # Final pass/fail
        if not result.failures and not result.error_message:
            result.passed = True

        self.logger.info(f"Test {test_case.id}: {'PASSED' if result.passed else 'FAILED'}")
        if result.failures:
            for failure in result.failures:
                self.logger.warning(f"  Failure: {failure}")

        return result

    def _run_single_turn_test(
        self,
        test_case: TestCase,
        result: TestResult,
        session_id: str
    ) -> TestResult:
        """Run a single-turn test case."""
        # Get user message from conversation
        user_messages = [
            msg.content for msg in test_case.conversation
            if msg.role == "user"
        ]

        if not user_messages:
            result.failures.append("No user messages in test case")
            return result

        # Process each user message (usually just one for single-turn)
        final_response = ""
        for idx, user_message in enumerate(user_messages):
            self.logger.info(f"Processing message {idx + 1}: {user_message[:100]}...")

            turn_start = time.time()
            response = self._agent.process_message(user_message, session_id)
            turn_latency = int((time.time() - turn_start) * 1000)

            self.logger.info(f"Response: {response[:200]}...")
            self.logger.info(f"Latency: {turn_latency}ms")

            final_response = response

            # Capture turn
            turn_capture = TurnCapture(
                turn_number=idx + 1,
                user_message=user_message,
                assistant_response=response,
                tool_calls=self.tool_interceptor.get_captured_calls(),
                latency_ms=turn_latency
            )
            self._current_capture.turns.append(turn_capture)

        result.final_response = final_response

        # Evaluate results
        actual_tool_calls = self.tool_interceptor.get_captured_calls()
        self._current_capture.total_tool_calls = actual_tool_calls

        # Convert to ToolCall objects for matching
        actual_as_tool_calls = [
            ToolCall(
                tool_name=tc.tool_name,
                tool_use_id=tc.tool_use_id,
                input=tc.input_params,
                output=tc.output,
                status=tc.status,
                error_message=tc.error_message
            )
            for tc in actual_tool_calls
        ]

        result = self._evaluate_test_case(test_case, result, actual_as_tool_calls)

        return result

    def _run_multi_turn_test(
        self,
        test_case: TestCase,
        result: TestResult,
        session_id: str
    ) -> TestResult:
        """Run a multi-turn test case."""
        from .models import TurnResult

        for turn in test_case.conversation_turns:
            self.logger.info(f"\n--- Turn {turn.turn_number} ---")
            self.logger.info(f"User: {turn.user_message[:100]}...")

            # Reset interceptor for per-turn capture
            turn_start_calls = len(self.tool_interceptor.captured_calls)

            turn_start = time.time()
            response = self._agent.process_message(turn.user_message, session_id)
            turn_latency = int((time.time() - turn_start) * 1000)

            self.logger.info(f"Assistant: {response[:200]}...")
            self.logger.info(f"Latency: {turn_latency}ms")

            # Get tool calls made in this turn
            turn_tool_calls = self.tool_interceptor.captured_calls[turn_start_calls:]

            # Create turn result
            turn_result = TurnResult(
                turn_number=turn.turn_number,
                user_message=turn.user_message,
                assistant_response=response,
                latency_ms=turn_latency,
            )

            # Convert to ToolCall for matching
            turn_actual = [
                ToolCall(
                    tool_name=tc.tool_name,
                    tool_use_id=tc.tool_use_id,
                    input=tc.input_params,
                    output=tc.output,
                    status=tc.status,
                    error_message=tc.error_message
                )
                for tc in turn_tool_calls
            ]

            turn_result.actual_tool_calls = [tc.to_dict() for tc in turn_tool_calls]

            # Match expected tool calls for this turn
            for expected in turn.expected_tool_calls:
                match_result = match_tool_call(expected, turn_actual)
                turn_result.tool_call_results.append(match_result)

                if expected.required and not match_result.matched:
                    result.failures.append(
                        f"Turn {turn.turn_number}: Expected tool '{expected.tool_name}' not called - {match_result.error_message}"
                    )

            # Check negative assertions
            actual_tool_names = {tc.tool_name for tc in turn_actual}
            for negative in turn.negative_assertions:
                if negative.tool_name in actual_tool_names:
                    result.failures.append(
                        f"Turn {turn.turn_number}: Forbidden tool '{negative.tool_name}' was called - {negative.reason}"
                    )

            # Check response assertions
            if turn.response_assertions and response:
                turn_result.response_assertion_results = match_response(
                    turn.response_assertions,
                    response
                )
                for assertion_result in turn_result.response_assertion_results:
                    if not assertion_result.passed:
                        result.failures.append(
                            f"Turn {turn.turn_number}: Response assertion failed: {assertion_result.assertion.assertion_type}"
                        )

            # Check latency
            if turn.max_latency_ms and turn_latency > turn.max_latency_ms:
                turn_result.latency_passed = False
                result.failures.append(
                    f"Turn {turn.turn_number}: Latency {turn_latency}ms > {turn.max_latency_ms}ms"
                )

            turn_result.passed = (
                all(tcr.matched for tcr in turn_result.tool_call_results if tcr.expected.required) and
                turn_result.latency_passed and
                all(rar.passed for rar in turn_result.response_assertion_results)
            )

            result.turn_results.append(turn_result)

            # Capture turn
            turn_capture = TurnCapture(
                turn_number=turn.turn_number,
                user_message=turn.user_message,
                assistant_response=response,
                tool_calls=turn_tool_calls,
                latency_ms=turn_latency
            )
            self._current_capture.turns.append(turn_capture)

        # Set final response from last turn
        if result.turn_results:
            result.final_response = result.turn_results[-1].assistant_response

        # Calculate aggregate metrics
        all_tool_calls = self.tool_interceptor.get_captured_calls()
        self._current_capture.total_tool_calls = all_tool_calls

        all_expected = []
        for turn in test_case.conversation_turns:
            all_expected.extend(turn.expected_tool_calls)

        all_actual = [
            ToolCall(
                tool_name=tc.tool_name,
                tool_use_id=tc.tool_use_id,
                input=tc.input_params,
                output=tc.output,
                status=tc.status,
                error_message=tc.error_message
            )
            for tc in all_tool_calls
        ]

        result.tool_metrics = calculate_tool_metrics(all_expected, all_actual)

        return result

    def _evaluate_test_case(
        self,
        test_case: TestCase,
        result: TestResult,
        actual_tool_calls: list[ToolCall]
    ) -> TestResult:
        """Evaluate test case results."""
        self.logger.info("Evaluating test case results...")

        # Get actual tool names
        actual_tool_names = {tc.tool_name for tc in actual_tool_calls}

        # Check forbidden tools
        for forbidden in test_case.forbidden_tools:
            if forbidden in actual_tool_names:
                result.failures.append(f"Forbidden tool was called: {forbidden}")
                self.logger.warning(f"Forbidden tool called: {forbidden}")

        # Check required tools
        for required in test_case.required_tools:
            if required not in actual_tool_names:
                result.failures.append(f"Required tool was not called: {required}")
                self.logger.warning(f"Required tool missing: {required}")

        # Check tool call count
        if test_case.max_tool_calls is not None:
            if len(actual_tool_calls) > test_case.max_tool_calls:
                result.failures.append(
                    f"Too many tool calls: {len(actual_tool_calls)} > {test_case.max_tool_calls}"
                )

        if test_case.min_tool_calls is not None:
            if len(actual_tool_calls) < test_case.min_tool_calls:
                result.failures.append(
                    f"Too few tool calls: {len(actual_tool_calls)} < {test_case.min_tool_calls}"
                )

        # Match expected tool calls
        tool_call_results = []
        for expected in test_case.expected_tools:
            match_result = match_tool_call(expected, actual_tool_calls)
            tool_call_results.append(match_result)

            if expected.required and not match_result.matched:
                result.failures.append(
                    f"Expected tool not matched: {expected.tool_name} - {match_result.error_message}"
                )

        result.tool_call_results = tool_call_results

        # Calculate metrics
        result.tool_metrics = calculate_tool_metrics(
            test_case.expected_tools,
            actual_tool_calls
        )

        # Validate sequence if required
        if test_case.strict_sequence:
            result.sequence_valid = validate_sequence(
                test_case.expected_tools,
                actual_tool_calls
            )
            if not result.sequence_valid:
                result.failures.append("Tool call sequence did not match expected order")

        # Evaluate response assertions
        if test_case.response_assertions and result.final_response:
            result.response_assertion_results = match_response(
                test_case.response_assertions,
                result.final_response
            )
            for assertion_result in result.response_assertion_results:
                if not assertion_result.passed:
                    result.failures.append(
                        f"Response assertion failed: {assertion_result.assertion.assertion_type}"
                    )

        return result

    def run_suite(self, suite: TestSuite) -> SuiteResult:
        """Run a complete test suite."""
        self.logger.info(f"{'=' * 70}")
        self.logger.info(f"REAL AGENT TEST SUITE: {suite.name}")
        self.logger.info(f"Total test cases: {len(suite.test_cases)}")
        self.logger.info(f"{'=' * 70}")

        result = SuiteResult(
            suite_id=suite.id,
            suite_name=suite.name,
        )

        start_time = time.time()

        # Get tests to run
        focused = [tc for tc in suite.test_cases if tc.focus]
        tests_to_run = focused if focused else suite.test_cases

        if focused:
            self.logger.info(f"Running {len(focused)} focused test(s) only")

        for idx, test_case in enumerate(tests_to_run):
            self.logger.info(f"\n[{idx + 1}/{len(tests_to_run)}] {test_case.name}")

            test_result = self.run_test_case(test_case)
            result.test_results.append(test_result)

            if test_case.skip:
                result.skipped_tests += 1
            elif test_result.passed:
                result.passed_tests += 1
            else:
                result.failed_tests += 1

        # Calculate summary
        result.duration_ms = int((time.time() - start_time) * 1000)
        result.ended_at = datetime.utcnow().isoformat()
        result.total_tests = len(tests_to_run)

        if result.total_tests > 0:
            result.pass_rate = result.passed_tests / result.total_tests

        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"Suite completed: {suite.name}")
        self.logger.info(f"Results: {result.passed_tests} passed, {result.failed_tests} failed, {result.skipped_tests} skipped")
        self.logger.info(f"Pass rate: {result.pass_rate:.1%}")
        self.logger.info(f"Duration: {result.duration_ms}ms")

        return result

    def get_captured_tool_calls(self) -> list[CapturedToolCall]:
        """Get all tool calls captured during the last test."""
        return self.tool_interceptor.get_captured_calls()

    def get_conversation_capture(self) -> ConversationCapture | None:
        """Get the full conversation capture from the last test."""
        return self._current_capture


# =============================================================================
# Assertion Helpers
# =============================================================================

class RealAgentAssertions:
    """
    Assertion helpers for real agent testing.

    Provides fluent assertions for tool calls, parameters, and side effects.
    """

    def __init__(
        self,
        tool_calls: list[CapturedToolCall],
        side_effects: SideEffectTracker,
        logger: logging.Logger | None = None
    ):
        self.tool_calls = tool_calls
        self.side_effects = side_effects
        self.logger = logger or logging.getLogger(__name__)

    def assert_tool_called(self, tool_name: str, message: str = "") -> "RealAgentAssertions":
        """Assert that a specific tool was called."""
        called_tools = [tc.tool_name for tc in self.tool_calls]
        assert tool_name in called_tools, (
            f"Expected tool '{tool_name}' to be called, but it was not. "
            f"Called tools: {called_tools}. {message}"
        )
        return self

    def assert_tool_not_called(self, tool_name: str, message: str = "") -> "RealAgentAssertions":
        """Assert that a specific tool was NOT called."""
        called_tools = [tc.tool_name for tc in self.tool_calls]
        assert tool_name not in called_tools, (
            f"Expected tool '{tool_name}' NOT to be called, but it was. {message}"
        )
        return self

    def assert_tool_called_times(self, tool_name: str, expected_times: int, message: str = "") -> "RealAgentAssertions":
        """Assert a tool was called exactly N times."""
        calls = [tc for tc in self.tool_calls if tc.tool_name == tool_name]
        assert len(calls) == expected_times, (
            f"Expected '{tool_name}' to be called {expected_times} time(s), "
            f"but it was called {len(calls)} time(s). {message}"
        )
        return self

    def assert_parameter(
        self,
        tool_name: str,
        param_name: str,
        expected_value: Any,
        match_strategy: str = "exact"
    ) -> "RealAgentAssertions":
        """Assert a parameter value for a tool call."""
        calls = [tc for tc in self.tool_calls if tc.tool_name == tool_name]
        assert calls, f"Tool '{tool_name}' was not called"

        # Check the first matching call
        call = calls[0]
        actual_value = call.input_params.get(param_name)

        if match_strategy == "exact":
            assert actual_value == expected_value, (
                f"Parameter '{param_name}' for '{tool_name}' mismatch. "
                f"Expected: {expected_value}, Actual: {actual_value}"
            )
        elif match_strategy == "contains":
            if isinstance(expected_value, str):
                assert expected_value in str(actual_value), (
                    f"Parameter '{param_name}' does not contain '{expected_value}'. "
                    f"Actual: {actual_value}"
                )
            elif isinstance(expected_value, list):
                for item in expected_value:
                    assert item in actual_value, (
                        f"Parameter '{param_name}' missing item '{item}'. "
                        f"Actual: {actual_value}"
                    )
        elif match_strategy == "type":
            assert isinstance(actual_value, type(expected_value)), (
                f"Parameter '{param_name}' type mismatch. "
                f"Expected type: {type(expected_value)}, Actual: {type(actual_value)}"
            )

        return self

    def assert_parameter_present(self, tool_name: str, param_name: str) -> "RealAgentAssertions":
        """Assert a parameter is present in the tool call."""
        calls = [tc for tc in self.tool_calls if tc.tool_name == tool_name]
        assert calls, f"Tool '{tool_name}' was not called"

        call = calls[0]
        assert param_name in call.input_params, (
            f"Parameter '{param_name}' not found in '{tool_name}' call. "
            f"Parameters: {list(call.input_params.keys())}"
        )
        return self

    def assert_file_exists(self, file_path: str) -> "RealAgentAssertions":
        """Assert a file was created."""
        assert self.side_effects.check_file_exists(file_path), (
            f"Expected file '{file_path}' to exist, but it does not."
        )
        return self

    def assert_file_contains(self, file_path: str, content: str) -> "RealAgentAssertions":
        """Assert a file contains specific content."""
        passed, error = self.side_effects.check_file_contains(file_path, content)
        assert passed, error
        return self

    def assert_tour_booking_created(
        self,
        name: str | None = None,
        date: str | None = None,
        confirmation_number: str | None = None
    ) -> "RealAgentAssertions":
        """Assert a tour booking was written to the bookings file."""
        passed, error = self.side_effects.check_tour_booking_created(
            name=name, date=date, confirmation_number=confirmation_number
        )
        assert passed, error or "Tour booking not found"
        return self

    def assert_no_tool_errors(self) -> "RealAgentAssertions":
        """Assert no tool calls resulted in errors."""
        errors = [tc for tc in self.tool_calls if tc.status == "error"]
        assert not errors, (
            f"Expected no tool errors, but found {len(errors)}: "
            f"{[(tc.tool_name, tc.error_message) for tc in errors]}"
        )
        return self

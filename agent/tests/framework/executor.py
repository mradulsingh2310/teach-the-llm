"""Test Executor for AI Agent testing framework.

This module provides the TestExecutor class that:
- Loads test cases from YAML
- Runs them against the PropertyAgent (or a mock)
- Captures full trajectory with extensive logging
- Returns TestResult objects
"""

import logging
import time
import traceback
from datetime import datetime
from typing import Any, Protocol

from .models import (
    ExpectedToolCall,
    SuiteResult,
    TestCase,
    TestResult,
    TestSuite,
    ToolCallMatchResult,
)
from .matchers import (
    calculate_tool_metrics,
    match_response,
    match_tool_call,
    validate_sequence,
)

# Import from agent models
from agent.models import ToolCall


class AgentProtocol(Protocol):
    """Protocol defining the interface for an agent."""

    def start_session(self) -> str:
        """Start a new conversation session."""
        ...

    def process_message(self, user_input: str, session_id: str) -> str:
        """Process a user message and return the agent's response."""
        ...

    def end_session(self, session_id: str, status: str = "completed") -> None:
        """End a conversation session."""
        ...


class MockAgent:
    """Mock agent for testing the test harness itself.

    This agent returns predefined responses and simulates tool calls
    based on the mock configuration.
    """

    def __init__(
        self,
        mock_responses: dict[str, str] | None = None,
        mock_tool_calls: dict[str, list[dict[str, Any]]] | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the mock agent.

        Args:
            mock_responses: Mapping of user input patterns to responses.
            mock_tool_calls: Mapping of user input patterns to tool calls.
            logger: Logger instance for logging.
        """
        self.mock_responses = mock_responses or {}
        self.mock_tool_calls = mock_tool_calls or {}
        self.logger = logger or logging.getLogger(__name__)
        self._session_counter = 0
        self._captured_tool_calls: list[ToolCall] = []

    def start_session(self) -> str:
        """Start a new mock session."""
        self._session_counter += 1
        session_id = f"mock-session-{self._session_counter}"
        self._captured_tool_calls = []
        self.logger.info(f"MockAgent: Started session {session_id}")
        return session_id

    def process_message(self, user_input: str, session_id: str) -> str:
        """Process a message and return a mock response."""
        self.logger.info(f"MockAgent: Processing message in session {session_id}")
        self.logger.debug(f"MockAgent: User input: {user_input}")

        # Find matching tool calls
        for pattern, tool_calls in self.mock_tool_calls.items():
            if pattern.lower() in user_input.lower():
                for tc in tool_calls:
                    tool_call = ToolCall(
                        tool_name=tc.get("tool_name", "unknown"),
                        tool_use_id=tc.get("tool_use_id", f"mock-{len(self._captured_tool_calls)}"),
                        input=tc.get("input", {}),
                        output=tc.get("output"),
                        status=tc.get("status", "success"),
                    )
                    self._captured_tool_calls.append(tool_call)
                    self.logger.debug(f"MockAgent: Simulated tool call: {tool_call.tool_name}")
                break

        # Find matching response
        for pattern, response in self.mock_responses.items():
            if pattern.lower() in user_input.lower():
                self.logger.info(f"MockAgent: Found response for pattern '{pattern}'")
                return response

        default_response = "I'm a mock agent. No specific response configured for this input."
        self.logger.info("MockAgent: Using default response")
        return default_response

    def end_session(self, session_id: str, status: str = "completed") -> None:
        """End the mock session."""
        self.logger.info(f"MockAgent: Ended session {session_id} with status {status}")

    def get_captured_tool_calls(self) -> list[ToolCall]:
        """Get the tool calls captured during the session."""
        return self._captured_tool_calls.copy()


class ToolCallCapture:
    """Utility to capture tool calls from an agent's conversation log."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._tool_calls: list[ToolCall] = []

    def capture_from_session(self, agent: Any, session_id: str) -> list[ToolCall]:
        """Extract tool calls from an agent's session.

        Args:
            agent: The agent instance (must have a logger attribute).
            session_id: The session ID to extract from.

        Returns:
            List of ToolCall objects from the session.
        """
        self._tool_calls = []

        try:
            # Try to get tool calls from the agent's conversation logger
            if hasattr(agent, 'logger') and hasattr(agent.logger, 'get_session'):
                session = agent.logger.get_session(session_id)
                if session:
                    for turn in session.conversation:
                        if turn.tools_called:
                            self._tool_calls.extend(turn.tools_called)
                    self.logger.debug(
                        f"Captured {len(self._tool_calls)} tool calls from session {session_id}"
                    )
        except Exception as e:
            self.logger.warning(f"Could not capture tool calls from session: {e}")

        return self._tool_calls


class TestExecutor:
    """Executor for running test cases against an agent."""

    def __init__(
        self,
        agent: AgentProtocol | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the test executor.

        Args:
            agent: The agent to test. If None, a MockAgent will be used.
            logger: Logger instance for extensive logging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.agent = agent
        self._tool_capture = ToolCallCapture(self.logger)

        if self.agent is None:
            self.logger.info("No agent provided, using MockAgent")
            self.agent = MockAgent(logger=self.logger)

    def run_test_case(self, test_case: TestCase) -> TestResult:
        """Run a single test case against the agent.

        Args:
            test_case: The test case to run.

        Returns:
            TestResult with detailed results of the test.
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"Running test case: {test_case.id} - {test_case.name}")
        self.logger.info(f"=" * 60)

        result = TestResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            passed=False,
        )

        # Check if test should be skipped
        if test_case.skip:
            self.logger.info(f"Test case SKIPPED: {test_case.skip_reason or 'No reason provided'}")
            result.passed = True  # Skipped tests don't count as failures
            result.error_message = f"Skipped: {test_case.skip_reason or 'No reason provided'}"
            return result

        start_time = time.time()
        session_id = None

        try:
            # Start agent session
            self.logger.debug("Starting agent session")
            session_id = self.agent.start_session()
            self.logger.info(f"Session started: {session_id}")

            # Apply mock tool responses if using MockAgent
            if isinstance(self.agent, MockAgent) and test_case.setup.mock_tool_responses:
                self.logger.debug("Applying mock tool responses from test setup")
                self.agent.mock_tool_calls.update(test_case.setup.mock_tool_responses)

            # Process each conversation message
            final_response = ""
            for idx, message in enumerate(test_case.conversation):
                if message.role == "user":
                    self.logger.info(f"Processing user message {idx + 1}/{len(test_case.conversation)}")
                    self.logger.debug(f"User input: {message.content[:100]}...")

                    final_response = self.agent.process_message(message.content, session_id)

                    self.logger.debug(f"Agent response: {final_response[:200]}...")

            result.final_response = final_response

            # Capture tool calls
            if isinstance(self.agent, MockAgent):
                actual_tool_calls = self.agent.get_captured_tool_calls()
            else:
                actual_tool_calls = self._tool_capture.capture_from_session(
                    self.agent, session_id
                )

            result.actual_tool_calls = [tc.model_dump() for tc in actual_tool_calls]
            self.logger.info(f"Captured {len(actual_tool_calls)} tool call(s)")

            # Evaluate results
            result = self._evaluate_test_case(test_case, result, actual_tool_calls)

            # End session
            self.agent.end_session(session_id, "completed")
            self.logger.info(f"Session {session_id} ended")

        except Exception as e:
            self.logger.exception(f"Exception during test execution")
            result.error_message = str(e)
            result.exception_type = type(e).__name__
            result.stack_trace = traceback.format_exc()
            result.failures.append(f"Exception: {e}")

            if session_id:
                try:
                    self.agent.end_session(session_id, "error")
                except Exception:
                    pass

        # Calculate timing
        end_time = time.time()
        result.duration_ms = int((end_time - start_time) * 1000)
        result.ended_at = datetime.utcnow().isoformat()

        # Check timing constraint
        if test_case.max_latency_ms and result.duration_ms > test_case.max_latency_ms:
            result.failures.append(
                f"Latency exceeded: {result.duration_ms}ms > {test_case.max_latency_ms}ms"
            )
            result.passed = False

        # Final pass/fail determination
        if not result.failures and not result.error_message:
            result.passed = True

        self.logger.info(f"Test case {test_case.id}: {'PASSED' if result.passed else 'FAILED'}")
        if result.failures:
            for failure in result.failures:
                self.logger.warning(f"  Failure: {failure}")

        return result

    def _evaluate_test_case(
        self,
        test_case: TestCase,
        result: TestResult,
        actual_tool_calls: list[ToolCall],
    ) -> TestResult:
        """Evaluate a test case after execution.

        Args:
            test_case: The test case being evaluated.
            result: The current TestResult to update.
            actual_tool_calls: List of actual tool calls made.

        Returns:
            Updated TestResult with evaluation results.
        """
        self.logger.info("Evaluating test case results")

        # Check forbidden tools
        actual_tool_names = {tc.tool_name for tc in actual_tool_calls}
        for forbidden in test_case.forbidden_tools:
            if forbidden in actual_tool_names:
                result.failures.append(f"Forbidden tool was used: {forbidden}")
                self.logger.warning(f"Forbidden tool used: {forbidden}")

        # Check required tools
        for required in test_case.required_tools:
            if required not in actual_tool_names:
                result.failures.append(f"Required tool was not used: {required}")
                self.logger.warning(f"Required tool missing: {required}")

        # Check tool call count constraints
        if test_case.max_tool_calls is not None:
            if len(actual_tool_calls) > test_case.max_tool_calls:
                result.failures.append(
                    f"Too many tool calls: {len(actual_tool_calls)} > {test_case.max_tool_calls}"
                )
                self.logger.warning(
                    f"Tool call count exceeded: {len(actual_tool_calls)} > {test_case.max_tool_calls}"
                )

        if test_case.min_tool_calls is not None:
            if len(actual_tool_calls) < test_case.min_tool_calls:
                result.failures.append(
                    f"Too few tool calls: {len(actual_tool_calls)} < {test_case.min_tool_calls}"
                )
                self.logger.warning(
                    f"Tool call count below minimum: {len(actual_tool_calls)} < {test_case.min_tool_calls}"
                )

        # Match expected tool calls
        tool_call_results: list[ToolCallMatchResult] = []
        for expected in test_case.expected_tools:
            match_result = match_tool_call(expected, actual_tool_calls)
            tool_call_results.append(match_result)

            if expected.required and not match_result.matched:
                result.failures.append(
                    f"Expected tool call not matched: {expected.tool_name} - {match_result.error_message}"
                )

        result.tool_call_results = tool_call_results

        # Calculate tool metrics
        result.tool_metrics = calculate_tool_metrics(
            test_case.expected_tools,
            actual_tool_calls,
        )

        # Validate sequence if required
        if test_case.strict_sequence:
            result.sequence_valid = validate_sequence(
                test_case.expected_tools,
                actual_tool_calls,
            )
            if not result.sequence_valid:
                result.failures.append("Tool call sequence did not match expected order")
                self.logger.warning("Sequence validation failed")

        # Evaluate response assertions
        if test_case.response_assertions and result.final_response:
            result.response_assertion_results = match_response(
                test_case.response_assertions,
                result.final_response,
            )

            for assertion_result in result.response_assertion_results:
                if not assertion_result.passed:
                    result.failures.append(
                        f"Response assertion failed: {assertion_result.assertion.assertion_type} "
                        f"for value '{assertion_result.assertion.value}'"
                    )

        return result

    def run_suite(self, suite: TestSuite) -> SuiteResult:
        """Run a complete test suite.

        Args:
            suite: The test suite to run.

        Returns:
            SuiteResult with results of all test cases.
        """
        self.logger.info(f"{'=' * 70}")
        self.logger.info(f"Running test suite: {suite.name}")
        self.logger.info(f"Total test cases: {len(suite.test_cases)}")
        self.logger.info(f"{'=' * 70}")

        result = SuiteResult(
            suite_id=suite.id,
            suite_name=suite.name,
        )

        start_time = time.time()

        # Check for focused tests
        focused_tests = [tc for tc in suite.test_cases if tc.focus]
        tests_to_run = focused_tests if focused_tests else suite.test_cases

        if focused_tests:
            self.logger.info(f"Running {len(focused_tests)} focused test(s) only")

        for idx, test_case in enumerate(tests_to_run):
            self.logger.info(f"\n[{idx + 1}/{len(tests_to_run)}] Running: {test_case.name}")

            test_result = self.run_test_case(test_case)
            result.test_results.append(test_result)

            if test_case.skip:
                result.skipped_tests += 1
            elif test_result.passed:
                result.passed_tests += 1
            else:
                result.failed_tests += 1

        # Calculate summary
        end_time = time.time()
        result.duration_ms = int((end_time - start_time) * 1000)
        result.ended_at = datetime.utcnow().isoformat()
        result.total_tests = len(tests_to_run)

        if result.total_tests > 0:
            result.pass_rate = result.passed_tests / result.total_tests

        # Aggregate tool metrics
        result.aggregate_tool_metrics = self._aggregate_tool_metrics(result.test_results)

        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"Suite completed: {suite.name}")
        self.logger.info(
            f"Results: {result.passed_tests} passed, "
            f"{result.failed_tests} failed, "
            f"{result.skipped_tests} skipped"
        )
        self.logger.info(f"Pass rate: {result.pass_rate:.1%}")
        self.logger.info(f"Duration: {result.duration_ms}ms")
        self.logger.info(f"{'=' * 70}")

        return result

    def _aggregate_tool_metrics(self, test_results: list[TestResult]) -> Any:
        """Aggregate tool metrics across all test results.

        Args:
            test_results: List of test results.

        Returns:
            Aggregated ToolMetrics.
        """
        from .models import ToolMetrics

        total_expected = 0
        total_actual = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for tr in test_results:
            total_expected += tr.tool_metrics.total_expected
            total_actual += tr.tool_metrics.total_actual
            total_tp += tr.tool_metrics.true_positives
            total_fp += tr.tool_metrics.false_positives
            total_fn += tr.tool_metrics.false_negatives

        metrics = ToolMetrics(
            total_expected=total_expected,
            total_actual=total_actual,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        )

        # Calculate aggregate precision, recall, F1
        if total_actual > 0:
            metrics.precision = total_tp / total_actual
        else:
            metrics.precision = 1.0 if total_expected == 0 else 0.0

        if total_expected > 0:
            metrics.recall = total_tp / total_expected
        else:
            metrics.recall = 1.0 if total_actual == 0 else 0.0

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = (
                2 * (metrics.precision * metrics.recall) /
                (metrics.precision + metrics.recall)
            )

        return metrics

"""
Test Evaluator for Property Agent Tests.

This module provides the evaluation logic that:
1. Runs test scenarios against the agent
2. Computes detailed per-turn metrics
3. Verifies file writes for create_appointment and create_lead
4. Aggregates results into comprehensive test run reports

Usage:
    evaluator = TestEvaluator(agent, model_id="google/functiongemma-270m-it")
    result = evaluator.evaluate_scenario(scenario)
    run = evaluator.finalize_run()
"""

import logging
import re
from datetime import datetime
from typing import Any, Optional

from .expected_results import (
    enrich_scenario_with_expectations,
    generate_appointment_keywords,
    generate_availability_keywords,
    generate_lead_keywords,
    generate_listing_search_keywords,
    generate_knowledge_search_keywords,
)
from .test_metrics import (
    ExpectedKeywords,
    ExpectedToolParameters,
    FileVerificationResult,
    KeywordMatchResult,
    ScenarioMetrics,
    TestResultsStore,
    TestRun,
    ToolCallEvaluation,
    TurnExpectation,
    TurnMetrics,
    VerificationStatus,
    check_keywords_in_response,
    evaluate_tool_call,
    get_or_create_store,
    save_results_store,
    verify_file_write,
)


logger = logging.getLogger(__name__)


class TestEvaluator:
    """
    Evaluates test scenarios and computes comprehensive metrics.

    This class:
    1. Runs multi-turn conversations against the agent
    2. Evaluates each turn for tool calling, parameters, and response keywords
    3. Verifies file writes for create_appointment and create_lead
    4. Aggregates results into a TestRun with full metrics

    Attributes:
        agent: The agent to test
        model_id: Model identifier for tracking
        agent_type: Type of agent for tracking
        test_suite_name: Name of the test suite
        current_run: The current TestRun being built
        results_store: Storage for persisting results
    """

    def __init__(
        self,
        agent: Any,
        model_id: str = "",
        agent_type: str = "property_agent",
        test_suite_name: str = "Property Agent Tests",
        persist_results: bool = True,
    ):
        """
        Initialize the test evaluator.

        Args:
            agent: The agent instance to test
            model_id: Model identifier
            agent_type: Type of agent
            test_suite_name: Name of test suite
            persist_results: Whether to persist results to disk
        """
        self.agent = agent
        self.model_id = model_id
        self.agent_type = agent_type
        self.test_suite_name = test_suite_name
        self.persist_results = persist_results

        # Initialize current run
        self.current_run = TestRun(
            model_id=model_id,
            agent_type=agent_type,
            test_suite_name=test_suite_name,
        )

        # Load or create results store
        if persist_results:
            self.results_store = get_or_create_store(
                model_id=model_id,
                agent_type=agent_type,
                test_suite_name=test_suite_name,
            )
        else:
            self.results_store = None

    def evaluate_scenario(self, scenario: dict[str, Any]) -> ScenarioMetrics:
        """
        Evaluate a complete test scenario.

        Args:
            scenario: The YAML scenario dict

        Returns:
            ScenarioMetrics with detailed results
        """
        scenario_id = scenario.get("id", "unknown")
        scenario_name = scenario.get("name", "Unnamed")
        category = scenario.get("category", "unknown")

        logger.info(f"Evaluating scenario: [{category}] {scenario_id} - {scenario_name}")

        # Enrich scenario with data-driven expectations
        enriched_scenario = enrich_scenario_with_expectations(scenario)
        turn_expectations = self._parse_turn_expectations(enriched_scenario)

        # Initialize scenario metrics
        metrics = ScenarioMetrics(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            category=category,
        )

        # Start agent session
        try:
            session_id = self.agent.start_session()
            logger.info(f"Started session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            metrics.errors.append(f"Session start failed: {e}")
            return metrics

        try:
            # Process each turn
            turn_num = 0
            conversation = scenario.get("conversation", [])

            if not conversation and "user_input" in scenario:
                # Single-turn scenario
                conversation = [
                    {"role": "user", "content": scenario["user_input"]}
                ]

            for msg in conversation:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    turn_num += 1

                    # Find matching expectation
                    expectation = None
                    for exp in turn_expectations:
                        if exp.turn_number == turn_num:
                            expectation = exp
                            break

                    # Process the turn
                    turn_metrics = self._evaluate_turn(
                        session_id=session_id,
                        user_message=user_message,
                        turn_number=turn_num,
                        expectation=expectation,
                    )

                    metrics.turn_metrics.append(turn_metrics)

            # Calculate aggregates
            metrics.calculate_aggregates()

        except Exception as e:
            logger.error(f"Error during scenario evaluation: {e}")
            metrics.errors.append(f"Evaluation error: {e}")

        finally:
            # End session
            try:
                self.agent.end_session(session_id)
            except Exception as e:
                logger.warning(f"Failed to end session: {e}")

        # Add to current run
        self.current_run.scenario_results.append(metrics)

        logger.info(
            f"Scenario {scenario_id} completed: "
            f"passed={metrics.passed}, "
            f"tool_rate={metrics.tool_calling_rate:.2f}, "
            f"response_rate={metrics.correct_response_rate:.2f}, "
            f"param_rate={metrics.correct_tool_call_rate:.2f}"
        )

        return metrics

    def _evaluate_turn(
        self,
        session_id: str,
        user_message: str,
        turn_number: int,
        expectation: Optional[TurnExpectation],
    ) -> TurnMetrics:
        """
        Evaluate a single conversation turn.

        Args:
            session_id: Active session ID
            user_message: User's message
            turn_number: Turn number (1-indexed)
            expectation: Expected outcomes for this turn

        Returns:
            TurnMetrics with detailed evaluation results
        """
        turn_metrics = TurnMetrics(
            turn_number=turn_number,
            user_message=user_message,
        )

        if expectation:
            turn_metrics.expected_tools_detail = expectation.expected_tools
            turn_metrics.expected_keywords_detail = expectation.expected_keywords
            turn_metrics.tools_expected = [t.tool_name for t in expectation.expected_tools]

        start_time = datetime.utcnow()

        try:
            # Process message through agent
            response = self.agent.process_message(user_message, session_id)
            turn_metrics.assistant_response = response

            # Calculate latency
            end_time = datetime.utcnow()
            turn_metrics.latency_ms = int((end_time - start_time).total_seconds() * 1000)

            # Get tool calls from agent session
            tool_calls = self._extract_tool_calls(session_id)
            turn_metrics.tools_called = [tc.get("tool_name", "") for tc in tool_calls]
            turn_metrics.raw_tool_calls = tool_calls

            # Evaluate tool calls if expectation exists
            if expectation and expectation.expected_tools:
                turn_metrics.tool_calling_success = self._check_tools_called(
                    expected=[t.tool_name for t in expectation.expected_tools],
                    actual=turn_metrics.tools_called,
                )

                # Evaluate each expected tool
                for exp_tool in expectation.expected_tools:
                    evaluation = evaluate_tool_call(exp_tool, tool_calls)
                    turn_metrics.tool_evaluations.append(evaluation)

                # Calculate parameter score
                if turn_metrics.tool_evaluations:
                    called_evals = [e for e in turn_metrics.tool_evaluations if e.was_called]
                    if called_evals:
                        turn_metrics.parameter_score = sum(e.score for e in called_evals) / len(called_evals)
                        turn_metrics.parameters_correct = all(e.parameters_correct for e in called_evals)

            # Evaluate response keywords
            if expectation and expectation.expected_keywords and response:
                keyword_result = check_keywords_in_response(response, expectation.expected_keywords)
                turn_metrics.keyword_results = keyword_result
                turn_metrics.response_correct = keyword_result.passed

            # Verify file writes
            if expectation and expectation.should_write_to_file and expectation.file_search_pattern:
                file_result = verify_file_write(
                    file_type=expectation.should_write_to_file,
                    search_pattern=expectation.file_search_pattern,
                )
                turn_metrics.file_verification = file_result

        except Exception as e:
            logger.error(f"Error processing turn {turn_number}: {e}")
            turn_metrics.assistant_response = f"Error: {e}"

        return turn_metrics

    def _extract_tool_calls(self, session_id: str) -> list[dict[str, Any]]:
        """
        Extract tool calls from the agent's session.

        Args:
            session_id: The session ID

        Returns:
            List of tool call dicts with tool_name, input, output
        """
        tool_calls = []

        try:
            # Try to get tool calls from agent's conversation logger
            if hasattr(self.agent, 'conversation_logger'):
                session = self.agent.conversation_logger.get_session(session_id)
                if session:
                    for turn in session.get("conversation", []):
                        for tc in turn.get("tools_called", []):
                            tool_calls.append({
                                "tool_name": tc.get("tool_name", ""),
                                "input": tc.get("input", {}),
                                "output": tc.get("output", {}),
                                "status": tc.get("status", ""),
                            })

            # Alternative: check for _last_tool_calls attribute
            elif hasattr(self.agent, '_last_tool_calls'):
                tool_calls = self.agent._last_tool_calls or []

        except Exception as e:
            logger.warning(f"Could not extract tool calls: {e}")

        return tool_calls

    def _check_tools_called(
        self,
        expected: list[str],
        actual: list[str],
    ) -> bool:
        """
        Check if expected tools were called.

        Args:
            expected: List of expected tool names
            actual: List of actually called tool names

        Returns:
            True if all expected tools were called
        """
        expected_set = set(expected)
        actual_set = set(actual)

        return expected_set.issubset(actual_set)

    def _parse_turn_expectations(self, scenario: dict[str, Any]) -> list[TurnExpectation]:
        """
        Parse turn expectations from enriched scenario.

        Args:
            scenario: Enriched scenario dict

        Returns:
            List of TurnExpectation objects
        """
        expectations = []

        for exp_data in scenario.get("turn_expectations", []):
            expected_tools = [
                ExpectedToolParameters(**t) for t in exp_data.get("expected_tools", [])
            ]
            expected_keywords = ExpectedKeywords(**exp_data.get("expected_keywords", {}))

            expectations.append(TurnExpectation(
                turn_number=exp_data.get("turn_number", 0),
                expected_tools=expected_tools,
                expected_keywords=expected_keywords,
                should_write_to_file=exp_data.get("should_write_to_file"),
                file_search_pattern=exp_data.get("file_search_pattern"),
            ))

        return expectations

    def finalize_run(self) -> TestRun:
        """
        Finalize the current test run and persist results.

        Returns:
            The completed TestRun
        """
        self.current_run.finalize()

        logger.info(
            f"Test run completed: "
            f"{self.current_run.metrics.passed_scenarios}/{self.current_run.metrics.total_scenarios} passed "
            f"({self.current_run.metrics.overall_pass_rate:.1%})"
        )
        logger.info(
            f"Metrics: tool_calling={self.current_run.metrics.avg_tool_calling_rate:.1%}, "
            f"response={self.current_run.metrics.avg_correct_response_rate:.1%}, "
            f"params={self.current_run.metrics.avg_correct_tool_call_rate:.1%}, "
            f"files={self.current_run.metrics.avg_file_verification_rate:.1%}"
        )

        # Persist to store
        if self.persist_results and self.results_store:
            self.results_store.add_run(self.current_run)
            filepath = save_results_store(self.results_store)
            logger.info(f"Results saved to: {filepath}")

        return self.current_run

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current test run.

        Returns:
            Summary dict suitable for logging or display
        """
        metrics = self.current_run.metrics

        return {
            "run_id": self.current_run.run_id,
            "total_scenarios": metrics.total_scenarios,
            "passed": metrics.passed_scenarios,
            "failed": metrics.failed_scenarios,
            "pass_rate": f"{metrics.overall_pass_rate:.1%}",
            "metrics": {
                "tool_calling_rate": f"{metrics.avg_tool_calling_rate:.1%}",
                "correct_response_rate": f"{metrics.avg_correct_response_rate:.1%}",
                "correct_tool_call_rate": f"{metrics.avg_correct_tool_call_rate:.1%}",
                "file_verification_rate": f"{metrics.avg_file_verification_rate:.1%}",
            },
            "category_breakdown": metrics.category_metrics,
            "tool_breakdown": metrics.tool_metrics,
        }

    def reset_run(self) -> None:
        """Reset for a new test run."""
        self.current_run = TestRun(
            model_id=self.model_id,
            agent_type=self.agent_type,
            test_suite_name=self.test_suite_name,
        )


class SimpleTestEvaluator:
    """
    Simplified evaluator for quick testing without full agent integration.

    This evaluator works with pre-captured tool calls and responses,
    useful for testing the evaluation logic itself.
    """

    def evaluate_turn(
        self,
        response: str,
        tool_calls: list[dict[str, Any]],
        expected_tools: list[ExpectedToolParameters],
        expected_keywords: ExpectedKeywords,
        file_type: Optional[str] = None,
        file_search_pattern: Optional[str] = None,
    ) -> TurnMetrics:
        """
        Evaluate a turn from pre-captured data.

        Args:
            response: The assistant's response
            tool_calls: List of tool calls made
            expected_tools: Expected tool parameters
            expected_keywords: Expected keywords
            file_type: Optional file type to verify
            file_search_pattern: Optional pattern to search in file

        Returns:
            TurnMetrics with evaluation results
        """
        turn_metrics = TurnMetrics(
            turn_number=1,
            user_message="(pre-captured)",
            assistant_response=response,
            tools_expected=[t.tool_name for t in expected_tools],
            tools_called=[tc.get("tool_name", "") for tc in tool_calls],
            raw_tool_calls=tool_calls,
            expected_tools_detail=expected_tools,
            expected_keywords_detail=expected_keywords,
        )

        # Check tool calling
        expected_set = set(turn_metrics.tools_expected)
        actual_set = set(turn_metrics.tools_called)
        turn_metrics.tool_calling_success = expected_set.issubset(actual_set)

        # Evaluate each expected tool
        for exp_tool in expected_tools:
            evaluation = evaluate_tool_call(exp_tool, tool_calls)
            turn_metrics.tool_evaluations.append(evaluation)

        # Calculate parameter score
        if turn_metrics.tool_evaluations:
            called_evals = [e for e in turn_metrics.tool_evaluations if e.was_called]
            if called_evals:
                turn_metrics.parameter_score = sum(e.score for e in called_evals) / len(called_evals)
                turn_metrics.parameters_correct = all(e.parameters_correct for e in called_evals)

        # Evaluate keywords
        keyword_result = check_keywords_in_response(response, expected_keywords)
        turn_metrics.keyword_results = keyword_result
        turn_metrics.response_correct = keyword_result.passed

        # Verify file write
        if file_type and file_search_pattern:
            turn_metrics.file_verification = verify_file_write(file_type, file_search_pattern)

        return turn_metrics


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TestEvaluator",
    "SimpleTestEvaluator",
]

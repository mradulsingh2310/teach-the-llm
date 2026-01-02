"""
GemmaAgent Tests - YAML-Driven Test Suite.

Run with: pytest tests/ -v
Run specific categories:
  pytest tests/ -v -m golden_path
  pytest tests/ -v -m edge_case
  pytest tests/ -v -m negative

Requires: llama-server running with FunctionGemma model
  Start server: llama-server -hf unsloth/functiongemma-270m-it-GGUF:BF16
  Default URL: http://localhost:8080/v1

Test scenarios are loaded from: agent/tests/fixtures/test_scenarios.yaml
"""

import logging
from typing import Any

import pytest

import config
from agent.tests.framework.models import (
    ConversationMessage,
    ExpectedToolCall,
    MatchStrategy,
    ParameterMatcher,
    TestCase,
)


def parse_scenario_to_test_case(scenario: dict[str, Any]) -> TestCase:
    """
    Parse a YAML scenario dict into a TestCase object.
    
    Handles both single-turn (user_input) and multi-turn (conversation) formats.
    """
    test_id = scenario.get("id", "unknown")
    name = scenario.get("name", f"Test {test_id}")
    
    # Build conversation
    conversation = []
    
    if "conversation" in scenario:
        for msg in scenario["conversation"]:
            conversation.append(ConversationMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            ))
    elif "user_input" in scenario:
        conversation.append(ConversationMessage(
            role="user",
            content=scenario["user_input"],
        ))
    
    # Parse expected tools
    expected_tools = []
    expected_tools_data = scenario.get("expected_tools", [])
    
    for tool_data in expected_tools_data:
        parameters = []
        params_data = tool_data.get("parameters", {})
        
        if isinstance(params_data, dict):
            for param_name, param_value in params_data.items():
                strategy = MatchStrategy.EXACT
                match_strategy = tool_data.get("match_strategy", "exact")
                if match_strategy == "contains":
                    strategy = MatchStrategy.CONTAINS
                elif match_strategy == "regex":
                    strategy = MatchStrategy.REGEX
                
                parameters.append(ParameterMatcher(
                    name=param_name,
                    strategy=strategy,
                    expected_value=param_value,
                ))
        
        expected_tools.append(ExpectedToolCall(
            tool_name=tool_data.get("tool_name", tool_data.get("tool", "unknown")),
            parameters=parameters,
            required=tool_data.get("required", True),
        ))
    
    # Get forbidden tools
    forbidden_tools = scenario.get("expected_no_tools", [])
    
    return TestCase(
        id=test_id,
        name=name,
        description=scenario.get("description"),
        category=scenario.get("category"),
        tags=scenario.get("tags", []),
        conversation=conversation,
        expected_tools=expected_tools,
        forbidden_tools=forbidden_tools,
        skip=scenario.get("skip", False),
        skip_reason=scenario.get("skip_reason"),
    )


class TestToolSchemas:
    """Tests for verifying tool schemas."""

    def test_all_tools_have_langchain_attributes(self, test_logger: logging.Logger):
        """Verify all tools have proper LangChain tool attributes."""
        from agent.tools import ALL_TOOLS

        test_logger.info(f"Testing {len(ALL_TOOLS)} tools")

        for tool in ALL_TOOLS:
            assert hasattr(tool, "name"), "Tool missing 'name' attribute"
            assert hasattr(tool, "description"), f"Tool {tool.name} missing 'description'"
            assert hasattr(tool, "args_schema"), f"Tool {tool.name} missing 'args_schema'"

            test_logger.info(f"Verified tool: {tool.name}")

    def test_tool_schemas_are_valid(self, test_logger: logging.Logger):
        """Verify tool schemas have valid JSON schema format."""
        from agent.tools import ALL_TOOLS

        for tool in ALL_TOOLS:
            assert tool.name, "Tool must have a name"
            assert tool.description, f"Tool {tool.name} must have a description"

            if tool.args_schema is not None:
                json_schema = tool.args_schema.model_json_schema()

                assert json_schema.get("type") == "object", (
                    f"Tool {tool.name}: JSON schema type should be 'object'"
                )
                assert "properties" in json_schema, (
                    f"Tool {tool.name}: JSON schema missing 'properties'"
                )

                test_logger.info(
                    f"Validated schema for: {tool.name}\n"
                    f"  Properties: {list(json_schema.get('properties', {}).keys())}"
                )


# =============================================================================
# YAML-Driven Scenario Tests
# =============================================================================

@pytest.mark.integration
class TestYAMLScenarios:
    """
    YAML-driven tests that run scenarios from test_scenarios.yaml.
    
    Each test is parametrized with scenarios loaded from the YAML file.
    The yaml_scenario fixture is populated by pytest_generate_tests hook.
    """

    def test_scenario(
        self,
        yaml_scenario: dict[str, Any],
        real_agent_executor,
        test_logger: logging.Logger,
    ):
        """
        Run a single test scenario from YAML.
        
        This test is parametrized - it runs once per scenario in the YAML file.
        """
        scenario_id = yaml_scenario.get("id", "unknown")
        scenario_name = yaml_scenario.get("name", "Unnamed Scenario")
        category = yaml_scenario.get("category", "unknown")
        
        test_logger.info("=" * 70)
        test_logger.info(f"SCENARIO: [{category}] {scenario_id} - {scenario_name}")
        test_logger.info("=" * 70)
        
        # Parse YAML scenario to TestCase
        test_case = parse_scenario_to_test_case(yaml_scenario)
        
        # Run the test case through the executor
        result = real_agent_executor.run_test_case(test_case)
        
        # Log results
        test_logger.info(f"Result: {'PASSED' if result.passed else 'FAILED'}")
        test_logger.info(f"Duration: {result.duration_ms}ms")
        
        if result.actual_tool_calls:
            test_logger.info(f"Tool calls made: {len(result.actual_tool_calls)}")
            for tc in result.actual_tool_calls:
                test_logger.info(f"  - {tc.get('tool_name', 'unknown')}: {tc.get('input', {})}")
        
        if result.final_response:
            response_preview = result.final_response[:200]
            test_logger.info(f"Final response: {response_preview}...")
        
        if result.failures:
            test_logger.warning("Failures:")
            for failure in result.failures:
                test_logger.warning(f"  - {failure}")
        
        if result.error_message:
            test_logger.error(f"Error: {result.error_message}")
        
        # Assert the test passed
        assert result.passed, f"Scenario {scenario_id} failed: {result.failures or result.error_message}"


@pytest.mark.integration
@pytest.mark.golden_path
class TestGoldenPathScenarios:
    """
    Golden path scenarios - successful complete user journeys.
    
    These test the happy path where users successfully search, inquire, and book tours.
    """

    def test_golden_path_scenario(
        self,
        yaml_scenario: dict[str, Any],
        real_agent_executor,
        test_logger: logging.Logger,
    ):
        """Run golden path scenarios from YAML."""
        scenario_id = yaml_scenario.get("id", "unknown")
        scenario_name = yaml_scenario.get("name", "Unnamed")
        
        test_logger.info(f"[GOLDEN PATH] {scenario_id}: {scenario_name}")
        
        test_case = parse_scenario_to_test_case(yaml_scenario)
        result = real_agent_executor.run_test_case(test_case)
        
        self._log_result(result, test_logger)
        assert result.passed, f"Golden path {scenario_id} failed: {result.failures}"

    def _log_result(self, result, logger):
        """Log test result details."""
        logger.info(f"  Passed: {result.passed}")
        logger.info(f"  Duration: {result.duration_ms}ms")
        if result.actual_tool_calls:
            tools = [tc.get("tool_name") for tc in result.actual_tool_calls]
            logger.info(f"  Tools called: {tools}")


@pytest.mark.integration
@pytest.mark.frustrated_path
class TestFrustratedPathScenarios:
    """
    Frustrated path scenarios - users encounter obstacles.
    
    These test how the agent handles budget mismatches, unavailability,
    and frustrated users requesting human escalation.
    """

    def test_frustrated_path_scenario(
        self,
        yaml_scenario: dict[str, Any],
        real_agent_executor,
        test_logger: logging.Logger,
    ):
        """Run frustrated path scenarios from YAML."""
        scenario_id = yaml_scenario.get("id", "unknown")
        scenario_name = yaml_scenario.get("name", "Unnamed")
        
        test_logger.info(f"[FRUSTRATED PATH] {scenario_id}: {scenario_name}")
        
        test_case = parse_scenario_to_test_case(yaml_scenario)
        result = real_agent_executor.run_test_case(test_case)
        
        test_logger.info(f"  Passed: {result.passed}, Duration: {result.duration_ms}ms")
        assert result.passed, f"Frustrated path {scenario_id} failed: {result.failures}"


@pytest.mark.integration
@pytest.mark.edge_case
class TestEdgeCaseScenarios:
    """
    Edge case scenarios - unusual inputs and complex requests.
    
    These test partial information, unusual date formats, multi-intent messages, etc.
    """

    def test_edge_case_scenario(
        self,
        yaml_scenario: dict[str, Any],
        real_agent_executor,
        test_logger: logging.Logger,
    ):
        """Run edge case scenarios from YAML."""
        scenario_id = yaml_scenario.get("id", "unknown")
        scenario_name = yaml_scenario.get("name", "Unnamed")
        
        test_logger.info(f"[EDGE CASE] {scenario_id}: {scenario_name}")
        
        test_case = parse_scenario_to_test_case(yaml_scenario)
        result = real_agent_executor.run_test_case(test_case)
        
        test_logger.info(f"  Passed: {result.passed}, Duration: {result.duration_ms}ms")
        assert result.passed, f"Edge case {scenario_id} failed: {result.failures}"


@pytest.mark.integration
@pytest.mark.negative
class TestNegativeScenarios:
    """
    Negative scenarios - verify agent does NOT call tools inappropriately.
    
    These test that simple greetings, farewells, and off-topic questions
    don't trigger unnecessary tool calls.
    """

    def test_negative_scenario(
        self,
        yaml_scenario: dict[str, Any],
        real_agent_executor,
        test_logger: logging.Logger,
    ):
        """Run negative scenarios from YAML."""
        scenario_id = yaml_scenario.get("id", "unknown")
        scenario_name = yaml_scenario.get("name", "Unnamed")
        
        test_logger.info(f"[NEGATIVE] {scenario_id}: {scenario_name}")
        
        test_case = parse_scenario_to_test_case(yaml_scenario)
        result = real_agent_executor.run_test_case(test_case)
        
        # For negative tests, also check that forbidden tools were NOT called
        if result.actual_tool_calls:
            tools_called = [tc.get("tool_name") for tc in result.actual_tool_calls]
            test_logger.info(f"  Tools called (should be minimal): {tools_called}")
        
        test_logger.info(f"  Passed: {result.passed}, Duration: {result.duration_ms}ms")
        assert result.passed, f"Negative test {scenario_id} failed: {result.failures}"


# =============================================================================
# Basic Agent Functionality Tests (kept for quick sanity checks)
# =============================================================================

@pytest.mark.integration
class TestBasicAgentFunctionality:
    """Basic sanity check tests for GemmaAgent functionality."""

    def test_agent_can_start_session(self, gemma_agent_helper, test_logger):
        """Test that agent can start a session."""
        test_logger.info(f"Testing session start with model: {config.MODEL_ID}")

        helper = gemma_agent_helper.start()
        assert helper.session_id is not None
        test_logger.info(f"Session started: {helper.session_id}")
        helper.end()


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling behavior."""

    def test_handles_empty_input(self, gemma_agent, test_logger):
        """Test that agent handles empty input gracefully."""
        test_logger.info("Testing empty input handling")

        session_id = gemma_agent.start_session()

        try:
            response = gemma_agent.process_message("", session_id)
            test_logger.info(f"Response to empty input: {response}")
        except Exception as e:
            test_logger.info(f"Exception on empty input (acceptable): {e}")

        gemma_agent.end_session(session_id)

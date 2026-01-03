"""
Pytest configuration for Gemma model tests.

This configures the test environment with model-specific settings
and loads test scenarios from YAML files.
"""

import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import pytest
import yaml

# Add parent directories to path
MODEL_DIR = Path(__file__).parent.parent
PROJECT_ROOT = MODEL_DIR.parent
AGENT_DIR = PROJECT_ROOT / "agent"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(AGENT_DIR))

# Import model config
import config
from gemma_agent import GemmaAgent

# Import test framework from agent/tests
from agent.tests.framework.models import (
    ConversationMessage,
    ExpectedToolCall,
    MatchStrategy,
    ParameterMatcher,
    TestCase,
)

# Import new metrics system
from agent.tests.framework.test_metrics import (
    ExpectedKeywords,
    ExpectedToolParameters,
    ScenarioMetrics,
    TestResultsStore,
    TestRun,
    TurnExpectation,
    TurnMetrics,
    check_keywords_in_response,
    evaluate_tool_call,
    get_or_create_store,
    save_results_store,
    verify_file_write,
)
from agent.tests.framework.expected_results import (
    enrich_scenario_with_expectations,
    generate_turn_expectation_from_yaml,
)
from agent.tests.framework.test_evaluator import TestEvaluator

# Path to test scenarios
SCENARIOS_FILE = AGENT_DIR / "tests" / "fixtures" / "test_scenarios.yaml"

# Configure test results directory
TEST_RESULTS_DIR = MODEL_DIR / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# YAML Scenario Loading
# =============================================================================

def load_all_scenarios_from_yaml(path: Path) -> list[dict[str, Any]]:
    """
    Load all test scenarios from the YAML file.
    
    Handles the structure with multiple scenario categories:
    - golden_path_scenarios
    - frustrated_path_scenarios
    - repeated_query_scenarios
    - edge_case_scenarios
    - negative_scenarios
    
    Returns a flat list of all scenarios with their category.
    """
    if not path.exists():
        raise FileNotFoundError(f"Test scenarios file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return []
    
    all_scenarios = []
    
    # Define scenario category keys to look for
    category_keys = [
        "golden_path_scenarios",
        "frustrated_path_scenarios", 
        "repeated_query_scenarios",
        "edge_case_scenarios",
        "negative_scenarios",
    ]
    
    for category_key in category_keys:
        scenarios = data.get(category_key, [])
        for scenario in scenarios:
            # Ensure category is set
            if "category" not in scenario:
                scenario["category"] = category_key.replace("_scenarios", "")
            all_scenarios.append(scenario)
    
    return all_scenarios


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
        # Multi-turn format
        for msg in scenario["conversation"]:
            conversation.append(ConversationMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            ))
    elif "user_input" in scenario:
        # Single-turn format
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


def get_scenario_test_id(scenario: dict[str, Any]) -> str:
    """Generate a test ID for pytest parametrize."""
    return f"{scenario.get('id', 'unknown')}-{scenario.get('name', 'unnamed')[:40]}"


# =============================================================================
# GemmaAgent Test Executor (replaces RealAgentTestExecutor for HuggingFace models)
# =============================================================================

class GemmaAgentTestExecutor:
    """
    Test executor for the HuggingFace Transformers-based GemmaAgent.

    This executor:
    1. Runs test scenarios against the GemmaAgent
    2. Computes detailed metrics (tool calling %, response %, parameter %)
    3. Verifies file writes for create_appointment and create_lead
    4. Stores results in immutable test run storage for UI visualization
    """

    def __init__(
        self,
        model_id: str = "google/functiongemma-270m-it",
        agent_name: str = "Aria",
        property_name: str = "Sunset Apartments",
        logger: logging.Logger = None,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        device: str = None,
        torch_dtype: str = "auto",
        persist_results: bool = True,
    ):
        self.model_id = model_id
        self.agent_name = agent_name
        self.property_name = property_name
        self.logger = logger or logging.getLogger(__name__)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.torch_dtype = torch_dtype
        self.persist_results = persist_results

        # Create the agent once (model loading is expensive)
        self.logger.info(f"Initializing GemmaAgent with model: {model_id}")
        self._agent = GemmaAgent(
            model_id=model_id,
            agent_name=agent_name,
            property_name=property_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            torch_dtype=torch_dtype,
        )
        self.logger.info("GemmaAgent initialized successfully")

        # Initialize test run for metrics tracking
        self.current_run = TestRun(
            model_id=model_id,
            agent_type="gemma_agent",
            test_suite_name="Property Agent Tests",
        )

        # Load or create results store
        if persist_results:
            self.results_store = get_or_create_store(
                model_id=model_id.replace("/", "_"),
                agent_type="gemma_agent",
                test_suite_name="property_agent_tests",
            )
        else:
            self.results_store = None

    def run_test_case(self, test_case: TestCase, raw_scenario: dict[str, Any] = None):
        """Run a single test case against the GemmaAgent with full metrics.

        Args:
            test_case: The parsed TestCase object
            raw_scenario: Optional raw YAML scenario dict for extracting per-turn expected_tools
        """
        import time
        from dataclasses import dataclass, field

        @dataclass
        class TestResult:
            test_case_id: str
            test_case_name: str
            passed: bool = False
            failures: list = field(default_factory=list)
            actual_tool_calls: list = field(default_factory=list)
            final_response: str = ""
            duration_ms: int = 0
            error_message: str = None
            # New metrics fields
            scenario_metrics: ScenarioMetrics = None

        result = TestResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
        )

        # Check skip condition
        if test_case.skip:
            self.logger.info(f"Test SKIPPED: {test_case.skip_reason}")
            result.passed = True
            result.error_message = f"Skipped: {test_case.skip_reason}"
            return result

        start_time = time.time()

        # Initialize scenario metrics
        scenario_metrics = ScenarioMetrics(
            scenario_id=test_case.id,
            scenario_name=test_case.name,
            category=str(test_case.category) if test_case.category else "unknown",
        )

        try:
            # Start session
            session_id = self._agent.start_session()
            self.logger.info(f"Session started: {session_id}")

            # Get expected tools from test case for comparison
            # Pass raw_scenario to extract per-turn expected_tools from conversation
            expected_tools_by_turn = self._get_expected_tools_by_turn(test_case, raw_scenario)

            # Process each message in the conversation
            turn_num = 0
            for msg in test_case.conversation:
                if msg.role == "user":
                    turn_num += 1
                    turn_start = time.time()

                    self.logger.info(f"Processing turn {turn_num}: {msg.content[:80]}...")
                    response = self._agent.process_message(msg.content, session_id)
                    result.final_response = response

                    turn_latency = int((time.time() - turn_start) * 1000)
                    self.logger.info(f"Response ({turn_latency}ms): {response[:100]}...")

                    # Extract tool calls for this turn
                    turn_tool_calls = self._extract_turn_tool_calls()

                    # Get expected tools for this turn
                    expected_tools = expected_tools_by_turn.get(turn_num, [])
                    expected_keywords = self._get_expected_keywords_for_turn(
                        test_case, turn_num, expected_tools
                    )

                    # Create turn metrics
                    turn_metrics = self._evaluate_turn(
                        turn_number=turn_num,
                        user_message=msg.content,
                        response=response,
                        tool_calls=turn_tool_calls,
                        expected_tools=expected_tools,
                        expected_keywords=expected_keywords,
                        latency_ms=turn_latency,
                    )

                    scenario_metrics.turn_metrics.append(turn_metrics)
                    result.actual_tool_calls.extend(turn_tool_calls)

            # End session
            self._agent.end_session(session_id)

            # Calculate scenario aggregates
            scenario_metrics.calculate_aggregates()
            result.scenario_metrics = scenario_metrics

            # Validate results against expected tools (original validation)
            result.passed = self._validate_results(test_case, result)

            # Also check metrics-based pass
            if not scenario_metrics.passed:
                result.failures.append(
                    f"Metrics threshold not met: "
                    f"tool_rate={scenario_metrics.tool_calling_rate:.2f}, "
                    f"response_rate={scenario_metrics.correct_response_rate:.2f}, "
                    f"param_rate={scenario_metrics.correct_tool_call_rate:.2f}"
                )

        except Exception as e:
            self.logger.exception("Exception during test execution")
            result.error_message = str(e)
            result.failures.append(f"Exception: {e}")
            scenario_metrics.errors.append(str(e))

        result.duration_ms = int((time.time() - start_time) * 1000)
        scenario_metrics.total_duration_ms = result.duration_ms

        # Add to current run
        self.current_run.scenario_results.append(scenario_metrics)

        # Log metrics
        self.logger.info(
            f"Turn Metrics - tool_call: {scenario_metrics.tool_calling_rate:.1%}, "
            f"response: {scenario_metrics.correct_response_rate:.1%}, "
            f"params: {scenario_metrics.correct_tool_call_rate:.1%}, "
            f"files: {scenario_metrics.file_verification_rate:.1%}"
        )

        return result

    def _get_expected_tools_by_turn(self, test_case: TestCase, raw_scenario: dict[str, Any] = None) -> dict:
        """Extract expected tools organized by turn number.

        Args:
            test_case: The parsed TestCase object
            raw_scenario: Optional raw YAML scenario dict for extracting per-turn expected_tools

        Returns:
            dict mapping turn_number to list of ExpectedToolParameters
        """
        tools_by_turn = {}

        # First, try to extract per-turn expected_tools from raw_scenario's conversation
        # This is the correct approach for multi-turn scenarios where expected_tools
        # are defined at each assistant message level in the YAML
        if raw_scenario and "conversation" in raw_scenario:
            turn_num = 0
            for i, msg in enumerate(raw_scenario["conversation"]):
                if msg.get("role") == "user":
                    turn_num += 1
                elif msg.get("role") == "assistant":
                    # Extract expected_tools from this assistant message
                    expected_tools_data = msg.get("expected_tools", [])
                    if expected_tools_data:
                        tools_by_turn[turn_num] = [
                            ExpectedToolParameters(
                                tool_name=tool_data.get("tool_name", tool_data.get("tool", "unknown")),
                                parameters=tool_data.get("parameters", {}),
                                required_params=list(tool_data.get("parameters", {}).keys()),
                            )
                            for tool_data in expected_tools_data
                        ]

        # Also check single-turn format (user_input with top-level expected_tools)
        if raw_scenario and "user_input" in raw_scenario:
            expected_tools_data = raw_scenario.get("expected_tools", [])
            if expected_tools_data:
                tools_by_turn[1] = [
                    ExpectedToolParameters(
                        tool_name=tool_data.get("tool_name", tool_data.get("tool", "unknown")),
                        parameters=tool_data.get("parameters", {}),
                        required_params=list(tool_data.get("parameters", {}).keys()),
                    )
                    for tool_data in expected_tools_data
                ]

        # Fallback: For simple test case format, expected_tools apply to all turns
        if not tools_by_turn and test_case.expected_tools:
            for i, msg in enumerate(test_case.conversation):
                if msg.role == "assistant" or i == 0:
                    continue
                turn_num = sum(1 for m in test_case.conversation[:i+1] if m.role == "user")
                tools_by_turn[turn_num] = [
                    ExpectedToolParameters(
                        tool_name=et.tool_name,
                        parameters={p.name: p.expected_value for p in et.parameters},
                        required_params=[p.name for p in et.parameters if p.required],
                    )
                    for et in test_case.expected_tools
                ]

        return tools_by_turn

    def _get_expected_keywords_for_turn(
        self,
        test_case: TestCase,
        turn_num: int,
        expected_tools: list,
    ) -> ExpectedKeywords:
        """Generate expected keywords for a turn based on expected tools."""
        keywords = ExpectedKeywords()

        for tool in expected_tools:
            tool_name = tool.tool_name
            params = tool.parameters

            if tool_name == "get_available_listings":
                bedrooms = params.get("bedrooms")
                keywords.required.extend(["unit", "available", "bedroom"])
                if bedrooms:
                    for b in (bedrooms if isinstance(bedrooms, list) else [bedrooms]):
                        if b == 0:
                            keywords.optional.append("studio")

            elif tool_name == "get_availability":
                keywords.required.extend(["available", "time", "slot"])
                keywords.optional.extend(["AM", "PM", "morning", "afternoon"])

            elif tool_name == "create_appointment":
                keywords.required.extend(["confirmed", "booked", "scheduled"])
                if "first_name" in params:
                    keywords.optional.append(params["first_name"])

            elif tool_name == "create_lead":
                keywords.required.extend(["contact", "information", "saved"])

            elif tool_name == "search_property_knowledge":
                keywords.required.append("information")

        # Deduplicate
        keywords.required = list(set(keywords.required))
        keywords.optional = list(set(keywords.optional))

        return keywords

    def _evaluate_turn(
        self,
        turn_number: int,
        user_message: str,
        response: str,
        tool_calls: list,
        expected_tools: list,
        expected_keywords: ExpectedKeywords,
        latency_ms: int,
    ) -> TurnMetrics:
        """Evaluate a single turn with detailed metrics."""
        turn_metrics = TurnMetrics(
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=response,
            tools_expected=[t.tool_name for t in expected_tools],
            tools_called=[tc.get("tool_name", "") for tc in tool_calls],
            latency_ms=latency_ms,
            raw_tool_calls=tool_calls,
            expected_tools_detail=expected_tools,
            expected_keywords_detail=expected_keywords,
        )

        # Check if expected tools were called
        expected_set = set(turn_metrics.tools_expected)
        actual_set = set(turn_metrics.tools_called)
        turn_metrics.tool_calling_success = expected_set.issubset(actual_set) if expected_set else True

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

        # Evaluate response keywords
        if expected_keywords.required or expected_keywords.optional:
            keyword_result = check_keywords_in_response(response, expected_keywords)
            turn_metrics.keyword_results = keyword_result
            turn_metrics.response_correct = keyword_result.passed
        else:
            turn_metrics.response_correct = True

        # Check file writes for create_appointment and create_lead
        for tc in tool_calls:
            tool_name = tc.get("tool_name", "")
            tool_input = tc.get("input", {})

            if tool_name == "create_appointment" and "email" in tool_input:
                turn_metrics.file_verification = verify_file_write(
                    "tour_bookings",
                    tool_input["email"]
                )
            elif tool_name == "create_lead" and "email" in tool_input:
                turn_metrics.file_verification = verify_file_write(
                    "leads",
                    tool_input["email"]
                )

        return turn_metrics

    def _extract_turn_tool_calls(self) -> list:
        """Extract tool calls from the agent's last turn."""
        tool_calls = []
        history = self._agent.get_conversation_history()

        if history:
            # Get last assistant message with tool calls
            for entry in reversed(history):
                if entry.get("role") == "assistant" and "tool_calls" in entry:
                    for tc in entry["tool_calls"]:
                        func = tc.get("function", {})
                        tool_calls.append({
                            "tool_name": func.get("name"),
                            "input": func.get("arguments", {}),
                        })
                    break

        return tool_calls

    def _validate_results(self, test_case: TestCase, result) -> bool:
        """Validate test results against expected outcomes."""
        passed = True

        # Check expected tools were called
        actual_tool_names = [tc.get("tool_name") for tc in result.actual_tool_calls]

        for expected in test_case.expected_tools:
            if expected.required:
                if expected.tool_name not in actual_tool_names:
                    result.failures.append(
                        f"Expected tool '{expected.tool_name}' was not called"
                    )
                    passed = False

        # Check forbidden tools were NOT called
        for forbidden in test_case.forbidden_tools:
            if forbidden in actual_tool_names:
                result.failures.append(
                    f"Forbidden tool '{forbidden}' was called"
                )
                passed = False

        return passed

    def finalize_run(self) -> TestRun:
        """Finalize the test run and save results."""
        self.current_run.finalize()

        self.logger.info("=" * 60)
        self.logger.info("TEST RUN SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Run ID: {self.current_run.run_id}")
        self.logger.info(f"Total Scenarios: {self.current_run.metrics.total_scenarios}")
        self.logger.info(f"Passed: {self.current_run.metrics.passed_scenarios}")
        self.logger.info(f"Failed: {self.current_run.metrics.failed_scenarios}")
        self.logger.info(f"Pass Rate: {self.current_run.metrics.overall_pass_rate:.1%}")
        self.logger.info("-" * 60)
        self.logger.info("METRICS:")
        self.logger.info(f"  Tool Calling Rate: {self.current_run.metrics.avg_tool_calling_rate:.1%}")
        self.logger.info(f"  Correct Response Rate: {self.current_run.metrics.avg_correct_response_rate:.1%}")
        self.logger.info(f"  Correct Tool Call Rate: {self.current_run.metrics.avg_correct_tool_call_rate:.1%}")
        self.logger.info(f"  File Verification Rate: {self.current_run.metrics.avg_file_verification_rate:.1%}")
        self.logger.info("-" * 60)

        # Category breakdown
        self.logger.info("BY CATEGORY:")
        for cat, metrics in self.current_run.metrics.category_metrics.items():
            self.logger.info(
                f"  {cat}: {metrics.get('passed', 0)}/{metrics.get('total', 0)} "
                f"({metrics.get('pass_rate', 0):.1%})"
            )

        self.logger.info("-" * 60)

        # Tool breakdown
        self.logger.info("BY TOOL:")
        for tool_name, metrics in self.current_run.metrics.tool_metrics.items():
            self.logger.info(
                f"  {tool_name}: called {metrics.get('total_called', 0)}/{metrics.get('total_expected', 0)} "
                f"(rate: {metrics.get('call_rate', 0):.1%}, param_acc: {metrics.get('param_accuracy', 0):.1%})"
            )

        self.logger.info("=" * 60)

        # Save to store
        if self.persist_results and self.results_store:
            self.results_store.add_run(self.current_run)
            filepath = save_results_store(self.results_store)
            self.logger.info(f"Results saved to: {filepath}")

            # Also save UI-ready JSON
            ui_filepath = filepath.replace(".json", "_ui.json")
            with open(ui_filepath, "w") as f:
                json.dump(self.results_store.to_ui_json(), f, indent=2)
            self.logger.info(f"UI data saved to: {ui_filepath}")

        return self.current_run

    def get_summary(self) -> dict:
        """Get a summary of the current test run."""
        return {
            "run_id": self.current_run.run_id,
            "total_scenarios": self.current_run.metrics.total_scenarios,
            "passed": self.current_run.metrics.passed_scenarios,
            "failed": self.current_run.metrics.failed_scenarios,
            "pass_rate": f"{self.current_run.metrics.overall_pass_rate:.1%}",
            "metrics": {
                "tool_calling_rate": f"{self.current_run.metrics.avg_tool_calling_rate:.1%}",
                "correct_response_rate": f"{self.current_run.metrics.avg_correct_response_rate:.1%}",
                "correct_tool_call_rate": f"{self.current_run.metrics.avg_correct_tool_call_rate:.1%}",
                "file_verification_rate": f"{self.current_run.metrics.avg_file_verification_rate:.1%}",
            },
        }


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def model_config():
    """Provide model-specific configuration."""
    return {
        "model_id": config.MODEL_ID,
        "model_name": config.MODEL_NAME,
        "temperature": config.TEMPERATURE,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "device": config.DEVICE,
        "torch_dtype": config.TORCH_DTYPE,
        "agent_name": config.AGENT_NAME,
        "property_name": config.PROPERTY_NAME,
    }


@pytest.fixture(scope="session")
def test_logger() -> logging.Logger:
    """Session-scoped logger fixture."""
    logger = logging.getLogger("gemma_agent_test")
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    TEST_RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = TEST_RESULTS_DIR / f"test_detailed_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Latest log
    latest_log = TEST_RESULTS_DIR / "latest.log"
    latest_handler = logging.FileHandler(latest_log, mode='w', encoding='utf-8')
    latest_handler.setLevel(logging.DEBUG)
    latest_handler.setFormatter(formatter)
    logger.addHandler(latest_handler)

    logger.info(f"Test logging initialized for {config.MODEL_NAME}")

    return logger


@pytest.fixture(scope="session")
def all_scenarios() -> list[dict[str, Any]]:
    """Load all test scenarios from YAML."""
    return load_all_scenarios_from_yaml(SCENARIOS_FILE)


@pytest.fixture(scope="session")
def golden_path_scenarios(all_scenarios) -> list[dict[str, Any]]:
    """Get only golden path scenarios."""
    return [s for s in all_scenarios if s.get("category") == "golden_path"]


@pytest.fixture(scope="session")
def frustrated_path_scenarios(all_scenarios) -> list[dict[str, Any]]:
    """Get only frustrated path scenarios."""
    return [s for s in all_scenarios if s.get("category") == "frustrated_path"]


@pytest.fixture(scope="session")
def edge_case_scenarios(all_scenarios) -> list[dict[str, Any]]:
    """Get only edge case scenarios."""
    return [s for s in all_scenarios if s.get("category") == "edge_case"]


@pytest.fixture(scope="session")
def negative_scenarios(all_scenarios) -> list[dict[str, Any]]:
    """Get only negative scenarios."""
    return [s for s in all_scenarios if s.get("category") == "negative"]


@pytest.fixture(scope="session")
def temp_log_file() -> str:
    """Provide a temporary log file path (session-scoped for model reuse)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"sessions": []}, f)
        return f.name


@pytest.fixture(scope="session")
def gemma_agent(temp_log_file: str):
    """Provide a GemmaAgent instance for testing.

    Note: Using session scope because model loading is expensive.
    The agent's conversation state is reset between tests via start_session/end_session.
    """
    agent = GemmaAgent(
        model_id=config.MODEL_ID,
        agent_name=config.AGENT_NAME,
        property_name=config.PROPERTY_NAME,
        log_file=temp_log_file,
        temperature=config.TEMPERATURE,
        max_new_tokens=config.MAX_NEW_TOKENS,
        device=config.DEVICE,
        torch_dtype=config.TORCH_DTYPE,
    )
    yield agent


@pytest.fixture(scope="session")
def real_agent_executor(test_logger) -> "GemmaAgentTestExecutor":
    """Provide a test executor using the HuggingFace Transformers-based GemmaAgent."""
    executor = GemmaAgentTestExecutor(
        model_id=config.MODEL_ID,
        agent_name=config.AGENT_NAME,
        property_name=config.PROPERTY_NAME,
        logger=test_logger,
        temperature=config.TEMPERATURE,
        max_new_tokens=config.MAX_NEW_TOKENS,
        device=config.DEVICE,
        torch_dtype=config.TORCH_DTYPE,
    )
    return executor


class GemmaAgentTestHelper:
    """Helper class for running GemmaAgent tests."""

    def __init__(
        self,
        agent: GemmaAgent,
        logger: Optional[logging.Logger] = None
    ):
        self.agent = agent
        self.logger = logger or logging.getLogger("gemma_agent_test")
        self.session_id = None
        self._responses = []
        self._tool_calls = []

    def start(self):
        """Start a new session."""
        self.session_id = self.agent.start_session()
        self._responses = []
        self._tool_calls = []
        return self

    def send(self, message: str) -> str:
        """Send a message and return the response."""
        if not self.session_id:
            self.start()

        response = self.agent.process_message(message, self.session_id)
        self._responses.append(response)
        return response

    def end(self):
        """End the session."""
        if self.session_id:
            self.agent.end_session(self.session_id)
            self.session_id = None

    @property
    def responses(self):
        """Get all responses."""
        return self._responses


@pytest.fixture(scope="function")
def gemma_agent_helper(gemma_agent, test_logger):
    """Provide a GemmaAgentTestHelper for convenient testing."""
    helper = GemmaAgentTestHelper(
        agent=gemma_agent,
        logger=test_logger
    )

    yield helper

    helper.end()


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "tool_calling: marks tests that verify tool calling"
    )
    config.addinivalue_line(
        "markers", "golden_path: marks golden path scenario tests"
    )
    config.addinivalue_line(
        "markers", "frustrated_path: marks frustrated path scenario tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks edge case scenario tests"
    )
    config.addinivalue_line(
        "markers", "negative: marks negative scenario tests"
    )


# =============================================================================
# Pytest Hooks for Dynamic Test Generation
# =============================================================================

def pytest_generate_tests(metafunc):
    """
    Generate parametrized tests from YAML scenarios.

    This hook is called for each test function that uses parametrize markers.
    We use it to dynamically load scenarios from YAML.
    """
    # Load scenarios if the fixture is requested
    if "yaml_scenario" in metafunc.fixturenames:
        try:
            scenarios = load_all_scenarios_from_yaml(SCENARIOS_FILE)

            # Filter by category if marker is present
            markers = [m.name for m in metafunc.definition.iter_markers()]

            if "golden_path" in markers:
                scenarios = [s for s in scenarios if s.get("category") == "golden_path"]
            elif "frustrated_path" in markers:
                scenarios = [s for s in scenarios if s.get("category") == "frustrated_path"]
            elif "edge_case" in markers:
                scenarios = [s for s in scenarios if s.get("category") == "edge_case"]
            elif "negative" in markers:
                scenarios = [s for s in scenarios if s.get("category") == "negative"]

            # Generate test IDs
            ids = [get_scenario_test_id(s) for s in scenarios]

            metafunc.parametrize("yaml_scenario", scenarios, ids=ids)
        except FileNotFoundError:
            # If file not found, skip parametrization
            pass


def pytest_sessionfinish(session, exitstatus):
    """
    Finalize test run and save metrics at end of test session.

    This hook is called after all tests have completed.
    """
    # Try to get the real_agent_executor fixture to finalize results
    # Note: This is a workaround since fixtures aren't directly accessible here
    try:
        # Check if we have test results to finalize
        for item in session.items:
            if hasattr(item, 'funcargs'):
                executor = item.funcargs.get('real_agent_executor')
                if executor and hasattr(executor, 'finalize_run'):
                    # Only finalize if there are results
                    if executor.current_run.scenario_results:
                        executor.finalize_run()
                    break
    except Exception as e:
        # Don't fail the session if finalization fails
        print(f"Warning: Could not finalize test results: {e}")
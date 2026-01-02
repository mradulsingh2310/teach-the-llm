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

    This replaces RealAgentTestExecutor for FunctionGemma tests, using the
    proper transformers pipeline instead of llama.cpp.
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
    ):
        self.model_id = model_id
        self.agent_name = agent_name
        self.property_name = property_name
        self.logger = logger or logging.getLogger(__name__)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.torch_dtype = torch_dtype

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

    def run_test_case(self, test_case: TestCase):
        """Run a single test case against the GemmaAgent."""
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

        try:
            # Start session
            session_id = self._agent.start_session()
            self.logger.info(f"Session started: {session_id}")

            # Process each message in the conversation
            for msg in test_case.conversation:
                if msg.role == "user":
                    self.logger.info(f"Processing message: {msg.content[:80]}...")
                    response = self._agent.process_message(msg.content, session_id)
                    result.final_response = response
                    self.logger.info(f"Response: {response[:100]}...")

            # Get conversation history to extract tool calls
            history = self._agent.get_conversation_history()
            for entry in history:
                if entry.get("role") == "assistant" and "tool_calls" in entry:
                    for tc in entry["tool_calls"]:
                        func = tc.get("function", {})
                        result.actual_tool_calls.append({
                            "tool_name": func.get("name"),
                            "input": func.get("arguments", {}),
                        })

            # End session
            self._agent.end_session(session_id)

            # Validate results against expected tools
            result.passed = self._validate_results(test_case, result)

        except Exception as e:
            self.logger.exception("Exception during test execution")
            result.error_message = str(e)
            result.failures.append(f"Exception: {e}")

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

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
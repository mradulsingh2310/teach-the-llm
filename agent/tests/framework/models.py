"""
Test Data Schema Models for AI Agent Testing Framework.

This module defines comprehensive Pydantic models for test case definition,
execution results, and aggregated reporting for AI agent testing. These models
are specifically designed for testing Property Agent tools:

- get_available_listings: Search properties by bedrooms and max_rent
- get_availability: Get tour time slots for a date
- create_appointment: Book a tour appointment
- search_property_knowledge: FAQ/knowledge base search
- escalate_conversation: Human handoff
- create_lead: Capture prospect information

Key Design Principles:
- Flexible parameter matching (exact, contains, regex, type-based)
- Multi-turn conversation support with per-turn expectations
- Comprehensive metrics (precision, recall, F1, sequence correctness)
- Extensible for future tool additions
- JSON-serializable for file-based test definitions
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enumerations
# =============================================================================


class MatchStrategy(str, Enum):
    """
    Strategy for matching expected values against actual values.

    Attributes:
        EXACT: Value must match exactly (case-sensitive for strings)
        EXACT_IGNORE_CASE: Value must match exactly (case-insensitive)
        CONTAINS: Actual value must contain expected value (substring match)
        CONTAINS_IGNORE_CASE: Case-insensitive substring match
        REGEX: Expected value is treated as a regular expression pattern
        TYPE_ONLY: Only the type of the value is checked, not the content
        PRESENT: Parameter must be present, value is not checked
        ABSENT: Parameter must NOT be present
        NUMERIC_RANGE: Value must be within a numeric range
        LIST_CONTAINS: For list values - actual list must contain expected items
        LIST_EXACT: For list values - must match exactly including order
        LIST_UNORDERED: For list values - must contain same items, any order
        ANY: Any value is acceptable (parameter just needs to exist)
        JSON_SUBSET: Actual JSON must contain all expected keys/values
        STARTS_WITH: Actual value must start with expected value
        ENDS_WITH: Actual value must end with expected value
        GLOB: Glob pattern matching for paths/strings
    """
    EXACT = "exact"
    EXACT_IGNORE_CASE = "exact_ignore_case"
    CONTAINS = "contains"
    CONTAINS_IGNORE_CASE = "contains_ignore_case"
    REGEX = "regex"
    TYPE_ONLY = "type_only"
    PRESENT = "present"
    ABSENT = "absent"
    NUMERIC_RANGE = "numeric_range"
    LIST_CONTAINS = "list_contains"
    LIST_EXACT = "list_exact"
    LIST_UNORDERED = "list_unordered"
    ANY = "any"
    JSON_SUBSET = "json_subset"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GLOB = "glob"


class TestStatus(str, Enum):
    """
    Status of a test case execution.

    Attributes:
        PENDING: Test has not been executed yet
        RUNNING: Test is currently being executed
        PASSED: Test completed and all assertions passed
        FAILED: Test completed but one or more assertions failed
        ERROR: Test could not complete due to an error (not an assertion failure)
        SKIPPED: Test was skipped (e.g., due to skip condition or dependency failure)
        TIMEOUT: Test exceeded the configured timeout
    """
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class TestCategory(str, Enum):
    """
    Categories for organizing test cases.

    Attributes:
        TOOL_SELECTION: Tests that verify correct tool is chosen
        PARAMETER_EXTRACTION: Tests that verify parameters are extracted correctly
        MULTI_TURN: Tests involving multiple conversation turns
        EDGE_CASE: Edge case and boundary condition tests
        ERROR_HANDLING: Tests for error scenarios and graceful degradation
        REGRESSION: Regression tests for previously identified issues
        PERFORMANCE: Tests focused on latency and response time
        INTEGRATION: End-to-end integration tests
        NEGATIVE: Tests that verify tools are NOT called inappropriately
        SMOKE: Basic smoke tests for quick validation
    """
    TOOL_SELECTION = "tool_selection"
    PARAMETER_EXTRACTION = "parameter_extraction"
    MULTI_TURN = "multi_turn"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    REGRESSION = "regression"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    NEGATIVE = "negative"
    SMOKE = "smoke"


class Severity(str, Enum):
    """
    Severity level for test failures.

    Attributes:
        CRITICAL: Test failure indicates critical functionality is broken
        HIGH: Important functionality affected
        MEDIUM: Moderate impact on functionality
        LOW: Minor issues, cosmetic or edge cases
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureType(str, Enum):
    """
    Categories of test failures for analysis.

    Attributes:
        TOOL_SELECTION: Wrong tool was selected
        PARAMETER_EXTRACTION: Correct tool but wrong parameters
        SEQUENCE: Tools called in wrong order
        NEGATIVE_ASSERTION: Forbidden tool was called
        RESPONSE: Response assertion failed
        TIMEOUT: Test exceeded time limit
        ERROR: Runtime error during test
        MISSING_REQUIRED_TOOL: Required tool was not called
    """
    TOOL_SELECTION = "tool_selection"
    PARAMETER_EXTRACTION = "parameter_extraction"
    SEQUENCE = "sequence"
    NEGATIVE_ASSERTION = "negative_assertion"
    RESPONSE = "response"
    TIMEOUT = "timeout"
    ERROR = "error"
    MISSING_REQUIRED_TOOL = "missing_required_tool"


# =============================================================================
# Parameter Matching Models
# =============================================================================


class ParameterMatcher(BaseModel):
    """
    Defines how to match an expected parameter value against actual value.

    This allows flexible matching strategies for parameter validation,
    supporting exact matches, partial matches, regex patterns, and more.

    Attributes:
        name: The parameter name to match
        strategy: The matching strategy to use
        expected_value: The expected value to match against
        min_value: For NUMERIC_RANGE - minimum acceptable value (inclusive)
        max_value: For NUMERIC_RANGE - maximum acceptable value (inclusive)
        pattern: For REGEX/GLOB - the pattern to match
        required: Whether this parameter must be present
        case_sensitive: Override case sensitivity for string comparisons
        weight: Weight for scoring (0.0-1.0) - higher weight means more important
        description: Human-readable description of what this matcher checks

    Example:
        >>> # Exact match for bedroom count
        >>> ParameterMatcher(name="bedrooms", expected_value=[2, 3], strategy=MatchStrategy.LIST_UNORDERED)

        >>> # Email must contain @ symbol
        >>> ParameterMatcher(name="email", expected_value="@", strategy=MatchStrategy.CONTAINS)

        >>> # Max rent in valid range
        >>> ParameterMatcher(name="max_rent", strategy=MatchStrategy.NUMERIC_RANGE, min_value=0, max_value=10000)
    """
    name: str
    strategy: MatchStrategy = MatchStrategy.EXACT
    expected_value: Any = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    required: bool = True
    case_sensitive: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str | None = None

    model_config = {"extra": "forbid"}


class ParameterMatchResult(BaseModel):
    """
    Result of matching a single parameter.

    Attributes:
        parameter_name: Name of the parameter
        matched: Whether the match succeeded
        strategy_used: Which strategy was used for matching
        expected_value: The expected value
        actual_value: The actual value from the tool call
        score: Match score (0.0 to 1.0)
        error_message: Explanation if match failed
    """
    parameter_name: str
    matched: bool
    strategy_used: MatchStrategy
    expected_value: Any = None
    actual_value: Any = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    error_message: str | None = None

    model_config = {"extra": "forbid"}


# =============================================================================
# Expected Tool Call Models
# =============================================================================


class ExpectedToolCall(BaseModel):
    """
    Definition of an expected tool call within a conversation turn.

    This model captures what tool should be called, with what parameters,
    and in what order relative to other expected tool calls.

    Attributes:
        tool_name: The name of the expected tool
        parameters: List of parameter matchers
        order_index: Expected position in the tool call sequence (None = any)
        required: If False, test doesn't fail if this tool isn't called
        allow_additional_params: Whether actual can have extra parameters
        allow_parallel: Whether this tool call can occur in parallel with others
        description: Human-readable description of why this tool should be called
        expected_output_contains: Optional check for expected content in tool output

    Example:
        >>> ExpectedToolCall(
        ...     tool_name="get_available_listings",
        ...     parameters=[
        ...         ParameterMatcher(name="bedrooms", expected_value=[2], strategy=MatchStrategy.LIST_CONTAINS),
        ...         ParameterMatcher(name="max_rent", expected_value=2000, strategy=MatchStrategy.EXACT)
        ...     ],
        ...     order_index=1
        ... )
    """
    tool_name: str
    parameters: list[ParameterMatcher] = Field(default_factory=list)
    order_index: int | None = None
    required: bool = True
    allow_additional_params: bool = True
    allow_parallel: bool = False
    description: str | None = None
    expected_output_contains: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class NegativeToolAssertion(BaseModel):
    """
    Defines a tool that should NOT be called.

    Used for negative testing to ensure the agent doesn't call
    inappropriate tools for a given context.

    Attributes:
        tool_name: The tool that should not be called
        reason: Explanation of why this tool shouldn't be called
        severity: How serious it is if this tool IS called

    Example:
        >>> NegativeToolAssertion(
        ...     tool_name="escalate_conversation",
        ...     reason="Simple listing query should not require escalation",
        ...     severity=Severity.HIGH
        ... )
    """
    tool_name: str
    reason: str
    severity: Severity = Severity.MEDIUM

    model_config = {"extra": "forbid"}


# =============================================================================
# Response Assertion Models
# =============================================================================


class ResponseAssertion(BaseModel):
    """
    Assertion for validating the agent's text response.

    Attributes:
        assertion_type: Type of assertion to apply
        value: The value to check for (string for text checks, int for length checks)
        case_sensitive: Whether string comparison is case-sensitive
        description: Human-readable description of what's being checked

    Example:
        >>> ResponseAssertion(
        ...     assertion_type="contains",
        ...     value="available",
        ...     case_sensitive=False,
        ...     description="Response should mention availability"
        ... )
    """
    assertion_type: Literal[
        "contains", "not_contains", "regex", "starts_with",
        "ends_with", "length_min", "length_max", "exact"
    ]
    value: str | int
    case_sensitive: bool = False
    description: str | None = None

    model_config = {"extra": "forbid"}


class ResponseAssertionResult(BaseModel):
    """
    Result of a response assertion check.

    Attributes:
        assertion: The response assertion that was checked
        passed: Whether the assertion passed
        actual_value: The actual response text (truncated if long)
        error_message: Explanation if assertion failed
    """
    assertion: ResponseAssertion
    passed: bool
    actual_value: str | None = None
    error_message: str | None = None

    model_config = {"extra": "forbid"}


class IntentAssertion(BaseModel):
    """
    Assertion for validating the agent understood the user's intent.

    Attributes:
        expected_intent: Description of the expected understood intent
        keywords: Keywords that should appear in response indicating correct intent
        anti_keywords: Keywords that should NOT appear (indicate misunderstanding)

    Example:
        >>> IntentAssertion(
        ...     expected_intent="User wants to search for 2-bedroom apartments",
        ...     keywords=["2 bedroom", "search", "listings"],
        ...     anti_keywords=["schedule", "tour", "appointment"]
        ... )
    """
    expected_intent: str
    keywords: list[str] = Field(default_factory=list)
    anti_keywords: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


# =============================================================================
# Conversation and Test Turn Models
# =============================================================================


class ConversationMessage(BaseModel):
    """
    A single message in a test conversation (for setup/context).

    Attributes:
        role: Whether this is a user or assistant message
        content: The message text content
    """
    role: Literal["user", "assistant"]
    content: str

    model_config = {"extra": "forbid"}


class TestConversationTurn(BaseModel):
    """
    A single turn in a test conversation scenario.

    Defines what the user says and what the expected agent behavior is
    for that specific turn, including tool calls and response patterns.

    Attributes:
        turn_number: Sequential turn number (1-indexed)
        user_message: The simulated user input for this turn
        expected_tool_calls: Tools expected to be called in this turn
        negative_assertions: Tools that should NOT be called in this turn
        response_assertions: Assertions on the agent's text response
        intent_assertion: Assertion on intent understanding
        max_latency_ms: Maximum acceptable latency for this turn
        strict_sequence: Must tool calls match expected order exactly?
        context_notes: Notes about conversation context for this turn

    Example:
        >>> TestConversationTurn(
        ...     turn_number=1,
        ...     user_message="I'm looking for a 2 bedroom apartment under $2000",
        ...     expected_tool_calls=[
        ...         ExpectedToolCall(
        ...             tool_name="get_available_listings",
        ...             parameters=[
        ...                 ParameterMatcher(name="bedrooms", expected_value=[2])
        ...             ]
        ...         )
        ...     ]
        ... )
    """
    turn_number: int = Field(ge=1)
    user_message: str
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    negative_assertions: list[NegativeToolAssertion] = Field(default_factory=list)
    response_assertions: list[ResponseAssertion] = Field(default_factory=list)
    intent_assertion: IntentAssertion | None = None
    max_latency_ms: int | None = None
    strict_sequence: bool = False
    context_notes: str | None = None

    model_config = {"extra": "forbid"}


# =============================================================================
# Test Setup Models
# =============================================================================


class SetupFile(BaseModel):
    """
    A file to set up before running a test.

    Attributes:
        path: Path where the file should be created
        content: Content to write to the file
    """
    path: str
    content: str

    model_config = {"extra": "forbid"}


class TestSetup(BaseModel):
    """
    Setup configuration for a test case.

    Attributes:
        files: Files to create before test runs
        environment_vars: Environment variables to set
        mock_tool_responses: Predefined responses for tool calls
        initial_context: Initial conversation context/state
    """
    files: list[SetupFile] = Field(default_factory=list)
    environment_vars: dict[str, str] = Field(default_factory=dict)
    mock_tool_responses: dict[str, Any] = Field(default_factory=dict)
    initial_context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


# =============================================================================
# Test Case Model
# =============================================================================


class TestCaseMetadata(BaseModel):
    """
    Metadata for a test case.

    Attributes:
        author: Who created the test case
        created_at: When the test was created
        updated_at: When the test was last updated
        version: Version of the test case
        jira_ticket: Related JIRA ticket if any
        related_tests: IDs of related test cases
    """
    author: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str | None = None
    version: str = "1.0.0"
    jira_ticket: str | None = None
    related_tests: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class TestCase(BaseModel):
    """
    Complete definition of a single test scenario.

    A test case can use either the simple `conversation` format (for quick tests)
    or the detailed `conversation_turns` format (for multi-turn tests with
    per-turn expectations).

    Attributes:
        id: Unique identifier for the test case
        name: Human-readable test name
        description: Detailed description of what's being tested
        category: Category for organizing tests
        tags: Arbitrary tags for filtering and grouping
        severity: Impact severity if this test fails

        # Input - use ONE of these:
        conversation: Simple conversation messages (legacy format)
        conversation_turns: Detailed multi-turn definitions with per-turn expectations

        setup: Test setup configuration

        # Expected behavior (global - used with simple conversation format):
        expected_tools: Tools expected to be called
        forbidden_tools: Tool names that should NOT be called
        required_tools: Tool names that MUST be called
        max_tool_calls: Maximum allowed tool calls
        min_tool_calls: Minimum required tool calls
        strict_sequence: Must tool calls match expected order?

        # Response assertions (global):
        response_assertions: Assertions on final response

        # Timing:
        timeout_ms: Maximum time for entire test execution
        max_latency_ms: Maximum latency per turn

        # Control:
        enabled: Whether the test should be executed
        skip: Whether to skip this test
        skip_reason: If skipped, why
        focus: Run only focused tests (for debugging)

        # Dependencies and metadata:
        dependencies: Test IDs that must pass before this test runs
        retry_count: Number of retries on failure
        metadata: Additional metadata about the test

    Example:
        >>> # Simple format
        >>> TestCase(
        ...     id="TC001",
        ...     name="Basic Listing Search",
        ...     description="Verify agent correctly searches for apartments",
        ...     conversation=[
        ...         ConversationMessage(role="user", content="Show me 2BR apartments")
        ...     ],
        ...     expected_tools=[
        ...         ExpectedToolCall(tool_name="get_available_listings")
        ...     ]
        ... )

        >>> # Multi-turn format
        >>> TestCase(
        ...     id="TC002",
        ...     name="Search and Schedule Tour",
        ...     description="Multi-turn: search then schedule",
        ...     conversation_turns=[
        ...         TestConversationTurn(
        ...             turn_number=1,
        ...             user_message="Show me 2BR apartments",
        ...             expected_tool_calls=[ExpectedToolCall(tool_name="get_available_listings")]
        ...         ),
        ...         TestConversationTurn(
        ...             turn_number=2,
        ...             user_message="Schedule a tour tomorrow at 2pm",
        ...             expected_tool_calls=[ExpectedToolCall(tool_name="create_appointment")]
        ...         )
        ...     ]
        ... )
    """
    id: str = Field(default_factory=lambda: f"TC-{uuid4().hex[:8].upper()}")
    name: str
    description: str | None = None
    category: TestCategory | str | None = None
    tags: list[str] = Field(default_factory=list)
    severity: Severity = Severity.MEDIUM

    # Input - simple format
    conversation: list[ConversationMessage] = Field(default_factory=list)

    # Input - detailed multi-turn format
    conversation_turns: list[TestConversationTurn] = Field(default_factory=list)

    setup: TestSetup = Field(default_factory=TestSetup)

    # Expected behavior (global - for simple conversation format)
    expected_tools: list[ExpectedToolCall] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    min_tool_calls: int | None = None
    strict_sequence: bool = False

    # Response assertions (global)
    response_assertions: list[ResponseAssertion] = Field(default_factory=list)

    # Timing
    timeout_ms: int = Field(default=30000, ge=1000)
    max_latency_ms: int | None = None

    # Control
    enabled: bool = True
    skip: bool = False
    skip_reason: str | None = None
    focus: bool = False

    # Dependencies and metadata
    dependencies: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0, le=5)
    metadata: TestCaseMetadata = Field(default_factory=TestCaseMetadata)

    model_config = {"extra": "forbid"}

    @field_validator("conversation_turns")
    @classmethod
    def validate_turn_numbers(cls, turns: list[TestConversationTurn]) -> list[TestConversationTurn]:
        """Ensure turn numbers are sequential starting from 1."""
        if not turns:
            return turns
        expected_turn = 1
        for turn in turns:
            if turn.turn_number != expected_turn:
                raise ValueError(
                    f"Turn numbers must be sequential. Expected {expected_turn}, got {turn.turn_number}"
                )
            expected_turn += 1
        return turns

    def is_multi_turn(self) -> bool:
        """Check if this test uses the multi-turn format."""
        return len(self.conversation_turns) > 0


# =============================================================================
# Test Suite Model
# =============================================================================


class RetryPolicy(BaseModel):
    """
    Configuration for test retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        retry_delay_ms: Delay between retries in milliseconds
        exponential_backoff: Whether to use exponential backoff
        retry_on_error: Retry on execution errors (not just failures)
        retry_on_timeout: Retry on timeout
    """
    max_retries: int = Field(default=0, ge=0, le=10)
    retry_delay_ms: int = Field(default=1000, ge=0)
    exponential_backoff: bool = False
    retry_on_error: bool = True
    retry_on_timeout: bool = True

    model_config = {"extra": "forbid"}


class SuiteConfiguration(BaseModel):
    """
    Global configuration for a test suite.

    Attributes:
        default_timeout_ms: Default timeout for test cases
        parallel_execution: Whether tests can run in parallel
        max_parallel_tests: Maximum concurrent tests if parallel
        stop_on_failure: Stop suite execution on first failure
        retry_policy: Retry configuration for failed tests
        environment: Target environment (dev, staging, prod)
        model_id: AI model identifier to test against
        base_url: API base URL
        custom_config: Additional configuration key-value pairs
    """
    default_timeout_ms: int = Field(default=30000, ge=1000)
    parallel_execution: bool = False
    max_parallel_tests: int = Field(default=4, ge=1, le=20)
    stop_on_failure: bool = False
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    environment: Literal["dev", "staging", "prod", "local", "test"] = "test"
    model_id: str = ""
    base_url: str = ""
    custom_config: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class Hook(BaseModel):
    """
    Definition of a setup or teardown hook.

    Attributes:
        name: Hook identifier
        hook_type: Type of hook
        script: Script or command to execute
        timeout_ms: Maximum time for hook execution
        continue_on_failure: Whether to continue if hook fails
    """
    name: str
    hook_type: Literal["setup_suite", "teardown_suite", "setup_test", "teardown_test"]
    script: str
    timeout_ms: int = Field(default=10000, ge=1000)
    continue_on_failure: bool = False

    model_config = {"extra": "forbid"}


class TestSuite(BaseModel):
    """
    Collection of test cases with shared configuration.

    A test suite groups related test cases and provides shared
    configuration, setup/teardown hooks, and execution policies.

    Attributes:
        id: Unique identifier for the suite
        name: Human-readable suite name
        description: Description of what the suite tests
        version: Suite version
        created_at: Creation timestamp
        updated_at: Last update timestamp
        configuration: Global suite configuration
        hooks: Setup and teardown hooks
        test_cases: The test cases in this suite
        tags: Tags for filtering suites
        owner: Team or person responsible for the suite

    Example:
        >>> TestSuite(
        ...     name="Property Listing Search Tests",
        ...     description="Tests for get_available_listings tool",
        ...     test_cases=[test_case_1, test_case_2],
        ...     configuration=SuiteConfiguration(
        ...         default_timeout_ms=60000,
        ...         environment="staging"
        ...     )
        ... )
    """
    id: str = Field(default_factory=lambda: f"TS-{uuid4().hex[:8].upper()}")
    name: str
    description: str | None = None
    version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str | None = None
    configuration: SuiteConfiguration = Field(default_factory=SuiteConfiguration)
    hooks: list[Hook] = Field(default_factory=list)
    test_cases: list[TestCase] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    owner: str | None = None

    model_config = {"extra": "forbid"}

    def get_enabled_tests(self) -> list[TestCase]:
        """Return only enabled and not skipped test cases."""
        return [tc for tc in self.test_cases if tc.enabled and not tc.skip]

    def get_focused_tests(self) -> list[TestCase]:
        """Return focused test cases, or all enabled if none focused."""
        focused = [tc for tc in self.test_cases if tc.focus and tc.enabled]
        return focused if focused else self.get_enabled_tests()

    def get_tests_by_category(self, category: TestCategory | str) -> list[TestCase]:
        """Return test cases matching a specific category."""
        return [tc for tc in self.test_cases if tc.category == category]

    def get_tests_by_tag(self, tag: str) -> list[TestCase]:
        """Return test cases with a specific tag."""
        return [tc for tc in self.test_cases if tag in tc.tags]


# =============================================================================
# Test Result Models
# =============================================================================


class ToolCallMatchResult(BaseModel):
    """
    Result of matching an expected tool call against actual calls.

    Attributes:
        expected: The expected tool call definition
        matched: Whether the tool call matched
        matched_actual_index: Index of matching actual call (if any)
        actual_tool_name: Name of the actual tool that was matched
        actual_tool_use_id: ID of the matched actual tool call
        actual_input: Actual input parameters
        parameter_results: Individual parameter match results
        parameters_score: Overall score for parameter matching
        sequence_correct: Whether the call was in the expected sequence position
        error_message: Explanation of the match result
    """
    expected: ExpectedToolCall
    matched: bool
    matched_actual_index: int | None = None
    actual_tool_name: str | None = None
    actual_tool_use_id: str | None = None
    actual_input: dict[str, Any] | None = None
    parameter_results: list[ParameterMatchResult] = Field(default_factory=list)
    parameters_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sequence_correct: bool = True
    error_message: str | None = None

    model_config = {"extra": "forbid"}


class NegativeAssertionResult(BaseModel):
    """
    Result of a negative assertion check.

    Attributes:
        assertion: The negative assertion that was checked
        passed: True if tool was NOT called (as expected)
        actual_call_found: Whether the forbidden tool was actually called
        actual_input: If called, what inputs were provided
    """
    assertion: NegativeToolAssertion
    passed: bool
    actual_call_found: bool = False
    actual_input: dict[str, Any] | None = None

    model_config = {"extra": "forbid"}


class TurnResult(BaseModel):
    """
    Execution result for a single conversation turn.

    Attributes:
        turn_number: Which turn this result is for
        user_message: The user message that was sent
        assistant_response: The agent's response text
        tool_call_results: Results of expected tool call matching
        actual_tool_calls: All actual tool calls made in this turn
        unexpected_tool_calls: Tools called that weren't expected
        negative_assertion_results: Results of negative assertions
        response_assertion_results: Results of response assertions
        latency_ms: Time taken for this turn
        latency_passed: Whether latency was within limits
        passed: Overall pass status for this turn
        errors: Any errors that occurred
    """
    turn_number: int
    user_message: str
    assistant_response: str | None = None
    tool_call_results: list[ToolCallMatchResult] = Field(default_factory=list)
    actual_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    unexpected_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    negative_assertion_results: list[NegativeAssertionResult] = Field(default_factory=list)
    response_assertion_results: list[ResponseAssertionResult] = Field(default_factory=list)
    latency_ms: int = 0
    latency_passed: bool = True
    passed: bool = False
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ToolMetrics(BaseModel):
    """
    Precision, recall, and F1 metrics for tool call evaluation.

    Attributes:
        precision: TP / (TP + FP) - how many called tools were correct
        recall: TP / (TP + FN) - how many expected tools were called
        f1_score: Harmonic mean of precision and recall
        total_expected: Total number of expected tool calls
        total_actual: Total number of actual tool calls
        true_positives: Expected tools that were correctly called
        false_positives: Unexpected tools that were called
        false_negatives: Expected tools that were not called
    """
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_expected: int = 0
    total_actual: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    model_config = {"extra": "forbid"}

    def calculate(self) -> None:
        """Calculate precision, recall, and F1 from TP/FP/FN counts."""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0

        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0

        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0


class TestResult(BaseModel):
    """
    Complete execution result for a single test case.

    Contains all details of the test execution including pass/fail status,
    per-turn results, metrics, and timing information.

    Attributes:
        test_case_id: ID of the test case that was executed
        test_case_name: Name of the test case
        passed: Whether the test passed
        status: Detailed status of the test execution

        # Timing
        started_at: When test execution started
        ended_at: When test execution ended
        duration_ms: Total execution time

        # Turn-by-turn results (for multi-turn tests)
        turn_results: Results for each conversation turn

        # Aggregate results (for simple tests or rollup)
        tool_call_results: Results of expected tool call matching
        tool_metrics: Aggregate tool precision/recall metrics
        parameter_accuracy: Overall parameter matching accuracy
        sequence_valid: Whether tool call sequence was correct
        sequence_accuracy: How often tool call order was correct

        # Response results
        response_assertion_results: Results of response assertions
        final_response: The final response from the agent

        # Actual data captured
        actual_tool_calls: All actual tool calls made

        # Errors and failures
        error_message: Error message if test errored
        exception_type: Type of exception if error occurred
        stack_trace: Stack trace if available
        failures: List of failure descriptions
        failure_reasons: Detailed reasons for failure

        # Retry info
        retry_count: How many times test was retried

        # Full trajectory for debugging
        full_trajectory: Complete conversation log

    Example:
        >>> result = TestResult(
        ...     test_case_id="TC001",
        ...     test_case_name="Basic Listing Search",
        ...     passed=True,
        ...     status=TestStatus.PASSED,
        ...     tool_metrics=ToolMetrics(true_positives=1, precision=1.0, recall=1.0, f1_score=1.0)
        ... )
    """
    test_case_id: str
    test_case_name: str
    passed: bool
    status: TestStatus = TestStatus.PENDING

    # Timing
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: str | None = None
    duration_ms: int = 0

    # Turn-by-turn results (for multi-turn tests)
    turn_results: list[TurnResult] = Field(default_factory=list)

    # Aggregate tool call results
    tool_call_results: list[ToolCallMatchResult] = Field(default_factory=list)
    tool_metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    parameter_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    sequence_valid: bool = True
    sequence_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    # Response results
    response_assertion_results: list[ResponseAssertionResult] = Field(default_factory=list)
    final_response: str | None = None

    # Actual data captured
    actual_tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    # Errors and failures
    error_message: str | None = None
    exception_type: str | None = None
    stack_trace: str | None = None
    failures: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)

    # Retry info
    retry_count: int = 0

    # Full trajectory for debugging
    full_trajectory: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    def mark_completed(self, passed: bool, status: TestStatus | None = None) -> None:
        """Mark the test as completed with final status."""
        self.ended_at = datetime.utcnow().isoformat()
        self.passed = passed
        self.status = status or (TestStatus.PASSED if passed else TestStatus.FAILED)
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.ended_at)
            self.duration_ms = int((end - start).total_seconds() * 1000)


# =============================================================================
# Test Report Models
# =============================================================================


class ToolBreakdown(BaseModel):
    """
    Per-tool statistics in the test report.

    Attributes:
        tool_name: Name of the tool
        total_expected: Total times tool was expected to be called
        total_actual: Total times tool was actually called
        correct_calls: Times tool was correctly called with correct params
        incorrect_calls: Times tool was called but with wrong params
        missed_calls: Times tool was expected but not called
        unexpected_calls: Times tool was called unexpectedly
        accuracy: Overall accuracy for this tool
        parameter_accuracy: Average parameter accuracy when called
    """
    tool_name: str
    total_expected: int = 0
    total_actual: int = 0
    correct_calls: int = 0
    incorrect_calls: int = 0
    missed_calls: int = 0
    unexpected_calls: int = 0
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    parameter_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class CategoryBreakdown(BaseModel):
    """
    Statistics broken down by test category.

    Attributes:
        category: The test category
        total: Total tests in this category
        passed: Number of passed tests
        failed: Number of failed tests
        error: Number of errored tests
        skipped: Number of skipped tests
        pass_rate: Percentage of tests that passed
    """
    category: TestCategory | str
    total: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0
    skipped: int = 0
    pass_rate: float = Field(default=0.0, ge=0.0, le=100.0)

    model_config = {"extra": "forbid"}

    def calculate_pass_rate(self) -> None:
        """Calculate pass rate from counts."""
        executed = self.passed + self.failed
        if executed > 0:
            self.pass_rate = (self.passed / executed) * 100
        else:
            self.pass_rate = 0.0


class FailureAnalysis(BaseModel):
    """
    Analysis of a test failure for reporting.

    Attributes:
        test_case_id: ID of the failed test
        test_case_name: Name of the failed test (optional, for display)
        failure_type: Category of failure
        description: Human-readable failure description
        expected: What was expected
        actual: What actually happened
        severity: Severity of the failure
        suggestions: Potential fixes or investigation steps
        similar_failures: Other tests with similar failures
    """
    test_case_id: str
    test_case_name: str | None = None
    failure_type: FailureType | str
    description: str
    expected: str | None = None
    actual: str | None = None
    severity: Severity = Severity.MEDIUM
    suggestions: list[str] = Field(default_factory=list)
    similar_failures: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class RegressionIndicator(BaseModel):
    """
    Indicator of potential regression from previous runs.

    Attributes:
        test_id: ID of the test showing regression
        test_name: Name of the test
        previous_status: Status in previous run
        current_status: Status in current run
        metric_changes: Changes in key metrics
        first_failed_at: When the test first started failing
    """
    test_id: str
    test_name: str
    previous_status: TestStatus
    current_status: TestStatus
    metric_changes: dict[str, dict[str, float]] = Field(default_factory=dict)
    first_failed_at: str | None = None

    model_config = {"extra": "forbid"}


class SuiteResult(BaseModel):
    """
    Execution result for a complete test suite.

    Attributes:
        suite_id: ID of the suite that was executed
        suite_name: Name of the suite
        started_at: When execution started
        ended_at: When execution ended
        duration_ms: Total execution time
        test_results: Individual test results
        total_tests: Total number of tests
        passed_tests: Number of passed tests
        failed_tests: Number of failed tests
        skipped_tests: Number of skipped tests
        error_tests: Number of errored tests
        pass_rate: Overall pass rate percentage
        aggregate_tool_metrics: Aggregated tool metrics
        configuration_used: Configuration at execution time
    """
    suite_id: str
    suite_name: str
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: str | None = None
    duration_ms: int = 0
    test_results: list[TestResult] = Field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    pass_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    aggregate_tool_metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    configuration_used: SuiteConfiguration | None = None

    model_config = {"extra": "forbid"}

    def calculate_summary(self) -> None:
        """Calculate summary statistics from test results."""
        self.total_tests = len(self.test_results)
        self.passed_tests = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        self.failed_tests = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        self.skipped_tests = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
        self.error_tests = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)

        executed = self.passed_tests + self.failed_tests
        if executed > 0:
            self.pass_rate = (self.passed_tests / executed) * 100
        else:
            self.pass_rate = 0.0


class TestReport(BaseModel):
    """
    Aggregated test report with comprehensive statistics.

    This is the top-level report model containing summary statistics,
    per-tool breakdowns, failure analysis, and regression indicators.

    Attributes:
        report_id: Unique identifier for this report
        generated_at: When the report was generated

        # Suite results
        suite_results: Results from executed test suites

        # Summary across all suites
        total_suites: Total number of suites run
        total_tests: Total tests across all suites
        total_passed: Total passed tests
        total_failed: Total failed tests
        total_error: Total errored tests
        total_skipped: Total skipped tests
        overall_pass_rate: Overall pass rate percentage
        total_duration_ms: Total execution time

        # Aggregate metrics
        overall_tool_metrics: Combined tool metrics
        aggregate_param_accuracy: Overall parameter accuracy
        aggregate_sequence_accuracy: Overall sequence accuracy

        # Breakdowns
        tool_breakdown: Per-tool statistics
        category_breakdown: Per-category statistics
        failure_analysis: Detailed failure analysis
        regression_indicators: Detected regressions

        # Metadata
        agent_type: Type of agent tested
        model_id: Model ID tested
        environment: Environment tests ran against
        metadata: Additional report metadata

    Example:
        >>> report = TestReport(
        ...     suite_results=[suite_result_1],
        ...     tool_breakdown=[
        ...         ToolBreakdown(tool_name="get_available_listings", correct_calls=10)
        ...     ],
        ...     overall_pass_rate=95.0
        ... )
    """
    report_id: str = Field(default_factory=lambda: f"TR-{uuid4().hex[:8].upper()}")
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Suite results
    suite_results: list[SuiteResult] = Field(default_factory=list)

    # Summary across all suites
    total_suites: int = 0
    total_tests: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_error: int = 0
    total_skipped: int = 0
    overall_pass_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    total_duration_ms: int = 0

    # Aggregate metrics
    overall_tool_metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    aggregate_param_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    aggregate_sequence_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    # Breakdowns
    tool_breakdown: list[ToolBreakdown] = Field(default_factory=list)
    category_breakdown: list[CategoryBreakdown] = Field(default_factory=list)
    failure_analysis: list[FailureAnalysis] = Field(default_factory=list)
    regression_indicators: list[RegressionIndicator] = Field(default_factory=list)

    # Metadata
    agent_type: str = ""
    model_id: str = ""
    environment: str = "test"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def calculate_aggregates(self) -> None:
        """Calculate aggregate statistics from suite results."""
        self.total_suites = len(self.suite_results)
        self.total_tests = 0
        self.total_passed = 0
        self.total_failed = 0
        self.total_error = 0
        self.total_skipped = 0
        self.total_duration_ms = 0

        for suite in self.suite_results:
            self.total_tests += suite.total_tests
            self.total_passed += suite.passed_tests
            self.total_failed += suite.failed_tests
            self.total_error += suite.error_tests
            self.total_skipped += suite.skipped_tests
            self.total_duration_ms += suite.duration_ms

        executed = self.total_passed + self.total_failed
        if executed > 0:
            self.overall_pass_rate = (self.total_passed / executed) * 100
        else:
            self.overall_pass_rate = 0.0

    def get_failed_tests(self) -> list[TestResult]:
        """Get all failed test results across suites."""
        failed = []
        for suite in self.suite_results:
            for result in suite.test_results:
                if result.status == TestStatus.FAILED:
                    failed.append(result)
        return failed

    def get_tool_breakdown(self, tool_name: str) -> ToolBreakdown | None:
        """Get breakdown for a specific tool."""
        for breakdown in self.tool_breakdown:
            if breakdown.tool_name == tool_name:
                return breakdown
        return None


# =============================================================================
# Convenience Types for Import
# =============================================================================

__all__ = [
    # Enums
    "MatchStrategy",
    "TestStatus",
    "TestCategory",
    "Severity",
    "FailureType",
    # Parameter Matching
    "ParameterMatcher",
    "ParameterMatchResult",
    # Expected Tool Calls
    "ExpectedToolCall",
    "NegativeToolAssertion",
    # Response Assertions
    "ResponseAssertion",
    "ResponseAssertionResult",
    "IntentAssertion",
    # Conversation
    "ConversationMessage",
    "TestConversationTurn",
    # Setup
    "SetupFile",
    "TestSetup",
    # Test Definition
    "TestCaseMetadata",
    "TestCase",
    # Test Suite
    "RetryPolicy",
    "SuiteConfiguration",
    "Hook",
    "TestSuite",
    # Results
    "ToolCallMatchResult",
    "NegativeAssertionResult",
    "TurnResult",
    "ToolMetrics",
    "TestResult",
    # Reporting
    "ToolBreakdown",
    "CategoryBreakdown",
    "FailureAnalysis",
    "RegressionIndicator",
    "SuiteResult",
    "TestReport",
]

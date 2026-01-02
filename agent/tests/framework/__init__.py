"""Test framework package for AI Agent testing.

This package provides:
- Test models for defining test cases, assertions, and results
- Matchers for parameter and tool call validation
- Test executor for running test cases against agents
- Test reporter for generating reports
- Test loader for loading test cases from YAML
- Real agent executor for integration testing
"""

# Import test models
from .models import (
    # Enums
    MatchStrategy,
    TestStatus,
    TestCategory,
    Severity,
    FailureType,
    # Parameter Matching
    ParameterMatcher,
    ParameterMatchResult,
    # Expected Tool Calls
    ExpectedToolCall,
    NegativeToolAssertion,
    # Response Assertions
    ResponseAssertion,
    ResponseAssertionResult,
    IntentAssertion,
    # Conversation
    ConversationMessage,
    TestConversationTurn,
    # Setup
    SetupFile,
    TestSetup,
    # Test Definition
    TestCaseMetadata,
    TestCase,
    # Test Suite
    RetryPolicy,
    SuiteConfiguration,
    Hook,
    TestSuite,
    # Results
    ToolCallMatchResult,
    NegativeAssertionResult,
    TurnResult,
    ToolMetrics,
    TestResult,
    # Reporting
    ToolBreakdown,
    CategoryBreakdown,
    FailureAnalysis,
    RegressionIndicator,
    SuiteResult,
    TestReport,
)

# Import matchers
from .matchers import (
    match_parameter,
    match_tool_call,
    match_response,
    calculate_tool_metrics,
    validate_sequence,
    calculate_parameter_accuracy,
    calculate_sequence_accuracy,
)

# Import executor
from .executor import (
    TestExecutor,
    MockAgent,
    ToolCallCapture,
    AgentProtocol,
)

# Import reporter
from .reporter import (
    TestReporter,
)

# Import loader
from .loader import (
    load_scenarios_from_yaml,
    load_suite_from_yaml,
    load_multiple_scenarios,
    discover_yaml_files,
    validate_test_case,
    validate_suite,
)

# Real agent testing imports
from .real_agent_executor import (
    RealAgentTestExecutor,
    RealAgentAssertions,
    SideEffectTracker,
    ToolCallInterceptor,
    CapturedToolCall,
    ConversationCapture,
)

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
    # Matchers
    "match_parameter",
    "match_tool_call",
    "match_response",
    "calculate_tool_metrics",
    "validate_sequence",
    "calculate_parameter_accuracy",
    "calculate_sequence_accuracy",
    # Executor
    "TestExecutor",
    "MockAgent",
    "ToolCallCapture",
    "AgentProtocol",
    # Reporter
    "TestReporter",
    # Loader
    "load_scenarios_from_yaml",
    "load_suite_from_yaml",
    "load_multiple_scenarios",
    "discover_yaml_files",
    "validate_test_case",
    "validate_suite",
    # Real Agent
    "RealAgentTestExecutor",
    "RealAgentAssertions",
    "SideEffectTracker",
    "ToolCallInterceptor",
    "CapturedToolCall",
    "ConversationCapture",
]

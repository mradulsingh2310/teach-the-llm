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

# New metrics system imports
from .test_metrics import (
    # Enums
    MetricType,
    VerificationStatus,
    # Expected Result Models
    ExpectedKeywords,
    ExpectedToolParameters,
    TurnExpectation,
    # Turn Result Models
    KeywordMatchResult,
    ToolCallEvaluation,
    FileVerificationResult,
    TurnMetrics,
    # Scenario Result Models
    ScenarioMetrics,
    # Test Run Models
    TestRunMetrics,
    TestRun,
    # Aggregated Results
    AggregatedMetrics,
    TestResultsStore,
    # Utility Functions
    check_keywords_in_response,
    evaluate_tool_call,
    verify_file_write,
    save_results_store,
    load_results_store,
    get_or_create_store,
)

from .expected_results import (
    load_listings,
    load_knowledge_base,
    get_listings_by_criteria,
    generate_listing_search_keywords,
    generate_availability_keywords,
    generate_appointment_keywords,
    generate_lead_keywords,
    generate_knowledge_search_keywords,
    generate_turn_expectation_from_yaml,
    enrich_scenario_with_expectations,
)

from .test_evaluator import (
    TestEvaluator,
    SimpleTestEvaluator,
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
    # New Metrics System - Enums
    "MetricType",
    "VerificationStatus",
    # New Metrics System - Expected Results
    "ExpectedKeywords",
    "ExpectedToolParameters",
    "TurnExpectation",
    # New Metrics System - Turn Results
    "KeywordMatchResult",
    "ToolCallEvaluation",
    "FileVerificationResult",
    "TurnMetrics",
    # New Metrics System - Scenario Results
    "ScenarioMetrics",
    # New Metrics System - Test Run
    "TestRunMetrics",
    "TestRun",
    # New Metrics System - Aggregated Results
    "AggregatedMetrics",
    "TestResultsStore",
    # New Metrics System - Utility Functions
    "check_keywords_in_response",
    "evaluate_tool_call",
    "verify_file_write",
    "save_results_store",
    "load_results_store",
    "get_or_create_store",
    # Expected Results Generation
    "load_listings",
    "load_knowledge_base",
    "get_listings_by_criteria",
    "generate_listing_search_keywords",
    "generate_availability_keywords",
    "generate_appointment_keywords",
    "generate_lead_keywords",
    "generate_knowledge_search_keywords",
    "generate_turn_expectation_from_yaml",
    "enrich_scenario_with_expectations",
    # Evaluator
    "TestEvaluator",
    "SimpleTestEvaluator",
]

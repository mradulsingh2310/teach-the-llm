"""Test Case Loader for AI Agent testing framework.

This module provides functions to load test cases and suites from YAML files,
converting them to Pydantic TestCase models.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from .models import (
    ConversationMessage,
    ExpectedToolCall,
    MatchStrategy,
    ParameterMatcher,
    ResponseAssertion,
    SetupFile,
    TestCase,
    TestSetup,
    TestSuite,
)


logger = logging.getLogger(__name__)


def load_scenarios_from_yaml(path: str) -> list[TestCase]:
    """Load test cases from a YAML file.

    The YAML file should contain a list of test case definitions.

    Args:
        path: Path to the YAML file.

    Returns:
        List of TestCase objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the YAML is malformed or invalid.
    """
    logger.info(f"Loading test scenarios from {path}")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Test scenarios file not found: {path}")

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.exception(f"Failed to parse YAML file: {path}")
        raise ValueError(f"Invalid YAML in {path}: {e}")

    if data is None:
        logger.warning(f"Empty YAML file: {path}")
        return []

    # Handle both list of test cases and dict with test_cases key
    if isinstance(data, dict):
        test_cases_data = data.get("test_cases", data.get("scenarios", []))
    elif isinstance(data, list):
        test_cases_data = data
    else:
        raise ValueError(f"Invalid YAML structure in {path}: expected list or dict")

    test_cases = []
    for idx, tc_data in enumerate(test_cases_data):
        try:
            test_case = _parse_test_case(tc_data, idx)
            test_cases.append(test_case)
            logger.debug(f"Loaded test case: {test_case.id}")
        except Exception as e:
            logger.error(f"Failed to parse test case at index {idx}: {e}")
            raise ValueError(f"Invalid test case at index {idx}: {e}")

    logger.info(f"Loaded {len(test_cases)} test case(s) from {path}")
    return test_cases


def load_suite_from_yaml(path: str) -> TestSuite:
    """Load a test suite from a YAML file.

    The YAML file should contain suite metadata and test cases.

    Args:
        path: Path to the YAML file.

    Returns:
        TestSuite object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the YAML is malformed or invalid.
    """
    logger.info(f"Loading test suite from {path}")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Test suite file not found: {path}")

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.exception(f"Failed to parse YAML file: {path}")
        raise ValueError(f"Invalid YAML in {path}: {e}")

    if data is None:
        raise ValueError(f"Empty YAML file: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid suite format in {path}: expected dict")

    # Extract suite metadata
    suite_name = data.get("name", data.get("suite_name", file_path.stem))
    suite_description = data.get("description", None)
    suite_tags = data.get("tags", [])

    # Load test cases
    test_cases_data = data.get("test_cases", data.get("scenarios", []))
    test_cases = []

    for idx, tc_data in enumerate(test_cases_data):
        try:
            test_case = _parse_test_case(tc_data, idx)
            test_cases.append(test_case)
        except Exception as e:
            logger.error(f"Failed to parse test case at index {idx}: {e}")
            raise ValueError(f"Invalid test case at index {idx}: {e}")

    suite = TestSuite(
        name=suite_name,
        description=suite_description,
        test_cases=test_cases,
        tags=suite_tags,
    )

    # Use provided ID if available
    if "id" in data:
        suite.id = data["id"]

    logger.info(f"Loaded test suite '{suite_name}' with {len(test_cases)} test case(s)")
    return suite


def _parse_test_case(data: dict[str, Any], index: int) -> TestCase:
    """Parse a test case from a dictionary.

    Args:
        data: Dictionary containing test case data.
        index: Index of the test case in the file (for default ID).

    Returns:
        TestCase object.
    """
    # Required fields
    test_id = data.get("id", f"test_{index}")
    name = data.get("name", f"Test Case {index}")

    # Parse conversation
    conversation_data = data.get("conversation", [])
    conversation = [
        ConversationMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
        )
        for msg in conversation_data
    ]

    # Parse setup
    setup = _parse_setup(data.get("setup", {}))

    # Parse expected tools
    expected_tools_data = data.get("expected_tools", data.get("expected", {}).get("tools", []))
    expected_tools = [
        _parse_expected_tool_call(tc_data)
        for tc_data in expected_tools_data
    ]

    # Parse response assertions
    assertions_data = data.get("response_assertions", data.get("assertions", []))
    response_assertions = [
        _parse_response_assertion(a_data)
        for a_data in assertions_data
    ]

    return TestCase(
        id=test_id,
        name=name,
        description=data.get("description"),
        category=data.get("category"),
        tags=data.get("tags", []),
        conversation=conversation,
        setup=setup,
        expected_tools=expected_tools,
        forbidden_tools=data.get("forbidden_tools", data.get("tools_not_used", [])),
        required_tools=data.get("required_tools", data.get("tools_used", [])),
        max_tool_calls=data.get("max_tool_calls", data.get("max_calls")),
        min_tool_calls=data.get("min_tool_calls"),
        strict_sequence=data.get("strict_sequence", False),
        response_assertions=response_assertions,
        max_latency_ms=data.get("max_latency_ms"),
        skip=data.get("skip", False),
        skip_reason=data.get("skip_reason"),
        focus=data.get("focus", False),
    )


def _parse_setup(data: dict[str, Any]) -> TestSetup:
    """Parse test setup from a dictionary.

    Args:
        data: Dictionary containing setup data.

    Returns:
        TestSetup object.
    """
    # Parse files
    files_data = data.get("files", {})
    files = []

    if isinstance(files_data, dict):
        # Dict format: {path: content}
        for path, content in files_data.items():
            files.append(SetupFile(path=path, content=content))
    elif isinstance(files_data, list):
        # List format: [{path: ..., content: ...}]
        for f_data in files_data:
            files.append(SetupFile(
                path=f_data.get("path", ""),
                content=f_data.get("content", ""),
            ))

    return TestSetup(
        files=files,
        environment_vars=data.get("environment_vars", data.get("env", {})),
        mock_tool_responses=data.get("mock_tool_responses", data.get("mocks", {})),
    )


def _parse_expected_tool_call(data: dict[str, Any]) -> ExpectedToolCall:
    """Parse an expected tool call from a dictionary.

    Args:
        data: Dictionary containing expected tool call data.

    Returns:
        ExpectedToolCall object.
    """
    tool_name = data.get("tool", data.get("tool_name", "unknown"))

    # Parse parameters
    params_data = data.get("parameters", data.get("params", {}))
    parameters = []

    if isinstance(params_data, dict):
        # Simple format: {name: value} - all exact match
        for name, value in params_data.items():
            parameters.append(ParameterMatcher(
                name=name,
                strategy=MatchStrategy.EXACT,
                expected_value=value,
            ))
    elif isinstance(params_data, list):
        # Extended format: [{name: ..., strategy: ..., value: ...}]
        for p_data in params_data:
            parameters.append(_parse_parameter_matcher(p_data))

    return ExpectedToolCall(
        tool_name=tool_name,
        parameters=parameters,
        order_index=data.get("order", data.get("order_index")),
        required=data.get("required", True),
        allow_additional_params=data.get("allow_additional_params", True),
        description=data.get("description"),
    )


def _parse_parameter_matcher(data: dict[str, Any]) -> ParameterMatcher:
    """Parse a parameter matcher from a dictionary.

    Args:
        data: Dictionary containing parameter matcher data.

    Returns:
        ParameterMatcher object.
    """
    name = data.get("name", data.get("param", "unknown"))

    # Parse strategy
    strategy_str = data.get("strategy", data.get("match", "exact"))
    try:
        strategy = MatchStrategy(strategy_str.lower())
    except ValueError:
        logger.warning(f"Unknown match strategy '{strategy_str}', using EXACT")
        strategy = MatchStrategy.EXACT

    return ParameterMatcher(
        name=name,
        strategy=strategy,
        expected_value=data.get("value", data.get("expected_value", data.get("expected"))),
        min_value=data.get("min_value", data.get("min")),
        max_value=data.get("max_value", data.get("max")),
        pattern=data.get("pattern", data.get("regex")),
        required=data.get("required", True),
        case_sensitive=data.get("case_sensitive", True),
    )


def _parse_response_assertion(data: dict[str, Any]) -> ResponseAssertion:
    """Parse a response assertion from a dictionary.

    Args:
        data: Dictionary containing assertion data.

    Returns:
        ResponseAssertion object.
    """
    # Handle different input formats
    if isinstance(data, str):
        # Simple string format: "contains:some text"
        if ":" in data:
            assertion_type, value = data.split(":", 1)
            return ResponseAssertion(
                assertion_type=assertion_type.strip().lower(),
                value=value.strip(),
            )
        else:
            # Default to contains
            return ResponseAssertion(
                assertion_type="contains",
                value=data,
            )

    # Dict format
    assertion_type = data.get("type", data.get("assertion_type", "contains"))
    value = data.get("value", data.get("expected", ""))

    return ResponseAssertion(
        assertion_type=assertion_type,
        value=value,
        case_sensitive=data.get("case_sensitive", True),
        description=data.get("description"),
    )


def load_multiple_scenarios(paths: list[str]) -> list[TestCase]:
    """Load test cases from multiple YAML files.

    Args:
        paths: List of paths to YAML files.

    Returns:
        Combined list of TestCase objects.
    """
    all_test_cases = []

    for path in paths:
        try:
            test_cases = load_scenarios_from_yaml(path)
            all_test_cases.extend(test_cases)
            logger.info(f"Loaded {len(test_cases)} test case(s) from {path}")
        except Exception as e:
            logger.error(f"Failed to load test cases from {path}: {e}")

    logger.info(f"Total test cases loaded: {len(all_test_cases)}")
    return all_test_cases


def discover_yaml_files(directory: str, pattern: str = "*.yaml") -> list[str]:
    """Discover YAML files in a directory.

    Args:
        directory: Directory to search.
        pattern: Glob pattern for matching files.

    Returns:
        List of file paths.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    files = list(dir_path.glob(pattern))
    files.extend(dir_path.glob(pattern.replace(".yaml", ".yml")))

    file_paths = [str(f) for f in sorted(files)]
    logger.info(f"Discovered {len(file_paths)} YAML file(s) in {directory}")

    return file_paths


def validate_test_case(test_case: TestCase) -> list[str]:
    """Validate a test case for common issues.

    Args:
        test_case: The test case to validate.

    Returns:
        List of validation warnings/errors.
    """
    issues = []

    # Check for empty conversation
    if not test_case.conversation:
        issues.append(f"Test case '{test_case.id}' has no conversation messages")

    # Check for user message
    user_messages = [m for m in test_case.conversation if m.role == "user"]
    if not user_messages:
        issues.append(f"Test case '{test_case.id}' has no user messages")

    # Check for conflicting tool requirements
    forbidden_set = set(test_case.forbidden_tools)
    required_set = set(test_case.required_tools)
    overlap = forbidden_set & required_set
    if overlap:
        issues.append(
            f"Test case '{test_case.id}' has tools that are both "
            f"required and forbidden: {overlap}"
        )

    # Check expected tools vs required tools
    expected_tool_names = {tc.tool_name for tc in test_case.expected_tools}
    for required in test_case.required_tools:
        if required not in expected_tool_names:
            issues.append(
                f"Test case '{test_case.id}' requires tool '{required}' "
                f"but it's not in expected_tools"
            )

    # Check min/max constraints
    if (test_case.min_tool_calls is not None and
        test_case.max_tool_calls is not None and
        test_case.min_tool_calls > test_case.max_tool_calls):
        issues.append(
            f"Test case '{test_case.id}' has min_tool_calls > max_tool_calls"
        )

    return issues


def validate_suite(suite: TestSuite) -> dict[str, list[str]]:
    """Validate all test cases in a suite.

    Args:
        suite: The test suite to validate.

    Returns:
        Dict mapping test case IDs to their validation issues.
    """
    all_issues = {}

    for test_case in suite.test_cases:
        issues = validate_test_case(test_case)
        if issues:
            all_issues[test_case.id] = issues

    if all_issues:
        logger.warning(f"Suite '{suite.name}' has {len(all_issues)} test case(s) with issues")
    else:
        logger.info(f"Suite '{suite.name}' validation passed")

    return all_issues

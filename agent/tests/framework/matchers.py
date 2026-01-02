"""Parameter and Tool Call Matchers for AI Agent testing.

This module provides functions to match expected vs actual values
for tool calls, parameters, and response assertions.
"""

import fnmatch
import logging
import re
from typing import Any

from .models import (
    ExpectedToolCall,
    MatchStrategy,
    ParameterMatcher,
    ParameterMatchResult,
    ResponseAssertion,
    ResponseAssertionResult,
    ToolCallMatchResult,
    ToolMetrics,
)

# Import ToolCall from the agent's models
from agent.models import ToolCall


logger = logging.getLogger(__name__)


def match_parameter(expected: ParameterMatcher, actual: Any) -> ParameterMatchResult:
    """Match an expected parameter against an actual value.

    Args:
        expected: The parameter matcher defining expected behavior.
        actual: The actual value to match against.

    Returns:
        ParameterMatchResult indicating whether the match succeeded.
    """
    logger.debug(
        f"Matching parameter '{expected.name}' with strategy {expected.strategy.value}: "
        f"expected={expected.expected_value}, actual={actual}"
    )

    result = ParameterMatchResult(
        parameter_name=expected.name,
        matched=False,
        strategy_used=expected.strategy,
        expected_value=expected.expected_value,
        actual_value=actual,
        score=0.0,
    )

    # Handle ABSENT strategy - parameter should NOT be present
    if expected.strategy == MatchStrategy.ABSENT:
        if actual is None:
            result.matched = True
            result.score = 1.0
            logger.debug(f"Parameter '{expected.name}' correctly absent")
        else:
            result.error_message = f"Parameter '{expected.name}' should be absent but was present"
            logger.warning(result.error_message)
        return result

    # Handle PRESENT strategy - parameter should exist with any value
    if expected.strategy == MatchStrategy.PRESENT:
        if actual is not None:
            result.matched = True
            result.score = 1.0
            logger.debug(f"Parameter '{expected.name}' is present as expected")
        else:
            result.error_message = f"Parameter '{expected.name}' should be present but was missing"
            logger.warning(result.error_message)
        return result

    # Handle missing parameter for other strategies
    if actual is None:
        if expected.required:
            result.error_message = f"Required parameter '{expected.name}' is missing"
            logger.warning(result.error_message)
        else:
            result.matched = True
            result.score = 1.0
            logger.debug(f"Optional parameter '{expected.name}' not present, marking as matched")
        return result

    # Apply strategy-specific matching
    try:
        match expected.strategy:
            case MatchStrategy.EXACT:
                result.matched = _match_exact(expected, actual, case_sensitive=True)

            case MatchStrategy.EXACT_IGNORE_CASE:
                result.matched = _match_exact(expected, actual, case_sensitive=False)

            case MatchStrategy.CONTAINS:
                result.matched = _match_contains(expected, actual, case_sensitive=True)

            case MatchStrategy.CONTAINS_IGNORE_CASE:
                result.matched = _match_contains(expected, actual, case_sensitive=False)

            case MatchStrategy.REGEX:
                result.matched = _match_regex(expected, actual)

            case MatchStrategy.TYPE_ONLY:
                result.matched = _match_type_only(expected, actual)

            case MatchStrategy.NUMERIC_RANGE:
                result.matched = _match_numeric_range(expected, actual)

            case MatchStrategy.LIST_CONTAINS:
                result.matched = _match_list_contains(expected, actual)

            case MatchStrategy.LIST_EXACT:
                result.matched = _match_list_exact(expected, actual)

            case MatchStrategy.LIST_UNORDERED:
                result.matched = _match_list_unordered(expected, actual)

            case MatchStrategy.ANY:
                result.matched = True
                logger.debug(f"Parameter '{expected.name}' matched with ANY strategy")

            case MatchStrategy.JSON_SUBSET:
                result.matched = _match_json_subset(expected, actual)

            case MatchStrategy.STARTS_WITH:
                result.matched = _match_starts_with(expected, actual)

            case MatchStrategy.ENDS_WITH:
                result.matched = _match_ends_with(expected, actual)

            case MatchStrategy.GLOB:
                result.matched = _match_glob(expected, actual)

            case _:
                result.error_message = f"Unknown match strategy: {expected.strategy}"
                logger.error(result.error_message)

    except Exception as e:
        result.error_message = f"Error during matching: {str(e)}"
        logger.exception(f"Exception while matching parameter '{expected.name}'")

    # Set score based on match result
    result.score = 1.0 if result.matched else 0.0

    if result.matched:
        logger.info(f"Parameter '{expected.name}' MATCHED using {expected.strategy.value}")
    else:
        if not result.error_message:
            result.error_message = (
                f"Parameter '{expected.name}' did not match: "
                f"expected {expected.expected_value}, got {actual}"
            )
        logger.warning(f"Parameter '{expected.name}' DID NOT MATCH: {result.error_message}")

    return result


def _match_exact(expected: ParameterMatcher, actual: Any, case_sensitive: bool = True) -> bool:
    """Exact match comparison."""
    if isinstance(expected.expected_value, str) and isinstance(actual, str):
        if case_sensitive:
            return expected.expected_value == actual
        return expected.expected_value.lower() == actual.lower()
    return expected.expected_value == actual


def _match_contains(expected: ParameterMatcher, actual: Any, case_sensitive: bool = True) -> bool:
    """Check if actual contains expected substring."""
    if not isinstance(actual, str):
        actual = str(actual)

    expected_val = expected.expected_value
    if not isinstance(expected_val, str):
        expected_val = str(expected_val)

    if case_sensitive:
        return expected_val in actual
    return expected_val.lower() in actual.lower()


def _match_regex(expected: ParameterMatcher, actual: Any) -> bool:
    """Match using regex pattern."""
    pattern = expected.pattern or str(expected.expected_value)
    if not isinstance(actual, str):
        actual = str(actual)

    flags = 0 if expected.case_sensitive else re.IGNORECASE
    return bool(re.search(pattern, actual, flags))


def _match_type_only(expected: ParameterMatcher, actual: Any) -> bool:
    """Match only the type of the value."""
    if expected.expected_value is None:
        return True

    expected_type = expected.expected_value
    if isinstance(expected_type, str):
        type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }
        expected_type = type_mapping.get(expected_type.lower(), type(None))

    return isinstance(actual, expected_type)


def _match_numeric_range(expected: ParameterMatcher, actual: Any) -> bool:
    """Match numeric value within range."""
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False

    if expected.min_value is not None and value < expected.min_value:
        return False
    if expected.max_value is not None and value > expected.max_value:
        return False
    return True


def _match_list_contains(expected: ParameterMatcher, actual: Any) -> bool:
    """Check if actual list contains all expected items."""
    if not isinstance(actual, list):
        return False

    expected_items = expected.expected_value
    if not isinstance(expected_items, list):
        expected_items = [expected_items]

    return all(item in actual for item in expected_items)


def _match_list_exact(expected: ParameterMatcher, actual: Any) -> bool:
    """Check if lists match exactly including order."""
    if not isinstance(actual, list):
        return False

    expected_items = expected.expected_value
    if not isinstance(expected_items, list):
        return False

    return actual == expected_items


def _match_list_unordered(expected: ParameterMatcher, actual: Any) -> bool:
    """Check if lists contain same items regardless of order."""
    if not isinstance(actual, list):
        return False

    expected_items = expected.expected_value
    if not isinstance(expected_items, list):
        return False

    return sorted(actual, key=str) == sorted(expected_items, key=str)


def _match_json_subset(expected: ParameterMatcher, actual: Any) -> bool:
    """Check if actual dict contains all expected keys/values."""
    if not isinstance(actual, dict):
        return False

    expected_dict = expected.expected_value
    if not isinstance(expected_dict, dict):
        return False

    def is_subset(expected_obj: Any, actual_obj: Any) -> bool:
        if isinstance(expected_obj, dict) and isinstance(actual_obj, dict):
            return all(
                key in actual_obj and is_subset(val, actual_obj[key])
                for key, val in expected_obj.items()
            )
        elif isinstance(expected_obj, list) and isinstance(actual_obj, list):
            return all(
                any(is_subset(exp_item, act_item) for act_item in actual_obj)
                for exp_item in expected_obj
            )
        else:
            return expected_obj == actual_obj

    return is_subset(expected_dict, actual)


def _match_starts_with(expected: ParameterMatcher, actual: Any) -> bool:
    """Check if actual starts with expected."""
    if not isinstance(actual, str):
        actual = str(actual)

    expected_val = expected.expected_value
    if not isinstance(expected_val, str):
        expected_val = str(expected_val)

    if expected.case_sensitive:
        return actual.startswith(expected_val)
    return actual.lower().startswith(expected_val.lower())


def _match_ends_with(expected: ParameterMatcher, actual: Any) -> bool:
    """Check if actual ends with expected."""
    if not isinstance(actual, str):
        actual = str(actual)

    expected_val = expected.expected_value
    if not isinstance(expected_val, str):
        expected_val = str(expected_val)

    if expected.case_sensitive:
        return actual.endswith(expected_val)
    return actual.lower().endswith(expected_val.lower())


def _match_glob(expected: ParameterMatcher, actual: Any) -> bool:
    """Match using glob pattern."""
    pattern = expected.pattern or str(expected.expected_value)
    if not isinstance(actual, str):
        actual = str(actual)

    return fnmatch.fnmatch(actual, pattern)


def match_tool_call(
    expected: ExpectedToolCall,
    actual_calls: list[ToolCall],
) -> ToolCallMatchResult:
    """Match an expected tool call against a list of actual calls.

    Args:
        expected: The expected tool call specification.
        actual_calls: List of actual tool calls made.

    Returns:
        ToolCallMatchResult indicating match status and details.
    """
    logger.info(
        f"Matching expected tool call '{expected.tool_name}' against "
        f"{len(actual_calls)} actual call(s)"
    )

    result = ToolCallMatchResult(
        expected=expected,
        matched=False,
        parameter_results=[],
        parameters_score=0.0,
    )

    # Find matching calls by tool name
    matching_calls = [
        (idx, call) for idx, call in enumerate(actual_calls)
        if call.tool_name == expected.tool_name
    ]

    if not matching_calls:
        result.error_message = f"No calls found for tool '{expected.tool_name}'"
        logger.warning(result.error_message)
        return result

    logger.debug(f"Found {len(matching_calls)} call(s) for tool '{expected.tool_name}'")

    # Try to find a call that matches all parameters
    best_match_idx = None
    best_param_results = []
    best_match_score = -1.0
    best_actual_call = None

    for idx, call in matching_calls:
        param_results = []
        total_weight = 0.0
        weighted_score = 0.0

        # Check each expected parameter
        for param_matcher in expected.parameters:
            actual_value = call.input.get(param_matcher.name)
            param_result = match_parameter(param_matcher, actual_value)
            param_results.append(param_result)

            # Calculate weighted score
            weight = param_matcher.weight
            total_weight += weight
            weighted_score += param_result.score * weight

        # Check for unexpected parameters if not allowed
        if not expected.allow_additional_params:
            expected_param_names = {p.name for p in expected.parameters}
            extra_params = set(call.input.keys()) - expected_param_names
            if extra_params:
                logger.debug(f"Found extra parameters not allowed: {extra_params}")
                continue

        # Calculate overall parameter score
        param_score = weighted_score / total_weight if total_weight > 0 else 1.0

        # Track best match
        if param_score > best_match_score:
            best_match_score = param_score
            best_match_idx = idx
            best_param_results = param_results
            best_actual_call = call

        # Check if all parameters matched
        if all(pr.matched for pr in param_results):
            result.matched = True
            result.matched_actual_index = idx
            result.actual_tool_name = call.tool_name
            result.actual_tool_use_id = call.tool_use_id
            result.actual_input = call.input
            result.parameter_results = param_results
            result.parameters_score = param_score
            logger.info(
                f"Tool call '{expected.tool_name}' MATCHED at index {idx} "
                f"with all {len(param_results)} parameters (score: {param_score:.2f})"
            )
            return result

    # If no perfect match, use best partial match for reporting
    result.matched_actual_index = best_match_idx
    result.parameter_results = best_param_results
    result.parameters_score = best_match_score if best_match_score >= 0 else 0.0

    if best_actual_call:
        result.actual_tool_name = best_actual_call.tool_name
        result.actual_tool_use_id = best_actual_call.tool_use_id
        result.actual_input = best_actual_call.input

    failed_params = [pr.parameter_name for pr in best_param_results if not pr.matched]
    result.error_message = (
        f"Tool '{expected.tool_name}' found but parameters didn't match: {failed_params}"
    )
    logger.warning(result.error_message)

    return result


def match_response(
    assertions: list[ResponseAssertion],
    response: str,
) -> list[ResponseAssertionResult]:
    """Evaluate response assertions against the actual response.

    Args:
        assertions: List of assertions to check.
        response: The actual response text.

    Returns:
        List of assertion results.
    """
    logger.info(f"Evaluating {len(assertions)} response assertion(s)")

    results = []

    for assertion in assertions:
        result = ResponseAssertionResult(
            assertion=assertion,
            passed=False,
            actual_value=response[:200] if response else None,  # Truncate for logging
        )

        try:
            # Handle case sensitivity (default is False in updated models)
            check_response = response
            check_value = str(assertion.value) if not isinstance(assertion.value, int) else assertion.value

            if not assertion.case_sensitive and isinstance(check_value, str):
                check_response = response.lower()
                check_value = check_value.lower()

            match assertion.assertion_type:
                case "contains":
                    result.passed = check_value in check_response

                case "not_contains":
                    result.passed = check_value not in check_response

                case "regex":
                    flags = 0 if assertion.case_sensitive else re.IGNORECASE
                    result.passed = bool(re.search(str(assertion.value), response, flags))

                case "length_min":
                    result.passed = len(response) >= int(assertion.value)

                case "length_max":
                    result.passed = len(response) <= int(assertion.value)

                case "starts_with":
                    result.passed = check_response.startswith(check_value)

                case "ends_with":
                    result.passed = check_response.endswith(check_value)

                case "exact":
                    result.passed = check_response == check_value

                case _:
                    result.error_message = f"Unknown assertion type: {assertion.assertion_type}"

        except Exception as e:
            result.error_message = f"Error evaluating assertion: {str(e)}"
            logger.exception(f"Exception while evaluating response assertion")

        if result.passed:
            logger.info(
                f"Response assertion '{assertion.assertion_type}' PASSED "
                f"(checking for: {assertion.value})"
            )
        else:
            logger.warning(
                f"Response assertion '{assertion.assertion_type}' FAILED "
                f"(checking for: {assertion.value})"
            )

        results.append(result)

    return results


def calculate_tool_metrics(
    expected: list[ExpectedToolCall],
    actual: list[ToolCall],
) -> ToolMetrics:
    """Calculate precision, recall, and F1 score for tool calls.

    Args:
        expected: List of expected tool calls.
        actual: List of actual tool calls made.

    Returns:
        ToolMetrics with precision, recall, and F1 scores.
    """
    logger.info(
        f"Calculating tool metrics: {len(expected)} expected, {len(actual)} actual"
    )

    metrics = ToolMetrics(
        total_expected=len(expected),
        total_actual=len(actual),
    )

    if not expected and not actual:
        metrics.precision = 1.0
        metrics.recall = 1.0
        metrics.f1_score = 1.0
        return metrics

    # Track which actual calls have been matched
    matched_actual_indices: set[int] = set()

    # Count true positives (expected calls that were correctly made)
    true_positives = 0
    for exp in expected:
        for idx, act in enumerate(actual):
            if idx in matched_actual_indices:
                continue
            if act.tool_name == exp.tool_name:
                # Check if parameters match
                all_params_match = True
                for param_matcher in exp.parameters:
                    actual_value = act.input.get(param_matcher.name)
                    param_result = match_parameter(param_matcher, actual_value)
                    if not param_result.matched:
                        all_params_match = False
                        break

                if all_params_match or not exp.parameters:
                    true_positives += 1
                    matched_actual_indices.add(idx)
                    logger.debug(f"True positive: {exp.tool_name} matched at index {idx}")
                    break

    # False positives: actual calls that weren't expected
    false_positives = len(actual) - len(matched_actual_indices)

    # False negatives: expected calls that weren't made
    false_negatives = len(expected) - true_positives

    metrics.true_positives = true_positives
    metrics.false_positives = false_positives
    metrics.false_negatives = false_negatives

    # Use the built-in calculate method
    metrics.calculate()

    logger.info(
        f"Tool metrics calculated: precision={metrics.precision:.2f}, "
        f"recall={metrics.recall:.2f}, F1={metrics.f1_score:.2f}"
    )
    logger.debug(
        f"Details: TP={true_positives}, FP={false_positives}, FN={false_negatives}"
    )

    return metrics


def validate_sequence(
    expected: list[ExpectedToolCall],
    actual: list[ToolCall],
) -> bool:
    """Validate that tool calls follow the expected sequence.

    Only considers expected calls that have an order_index set.
    Validates that calls with order_index appear in the correct relative order.

    Args:
        expected: List of expected tool calls (with optional order_index).
        actual: List of actual tool calls made.

    Returns:
        True if the sequence is valid, False otherwise.
    """
    logger.info("Validating tool call sequence")

    # Get expected calls with order_index, sorted by index
    ordered_expected = sorted(
        [e for e in expected if e.order_index is not None],
        key=lambda x: x.order_index or 0
    )

    if not ordered_expected:
        logger.debug("No ordered expectations, sequence is valid by default")
        return True

    # Build a mapping of tool names to their positions in actual calls
    actual_positions: dict[str, list[int]] = {}
    for idx, call in enumerate(actual):
        if call.tool_name not in actual_positions:
            actual_positions[call.tool_name] = []
        actual_positions[call.tool_name].append(idx)

    # Verify sequence
    last_position = -1
    for exp in ordered_expected:
        positions = actual_positions.get(exp.tool_name, [])

        # Find the first position that comes after last_position
        found_valid_position = False
        for pos in positions:
            if pos > last_position:
                last_position = pos
                found_valid_position = True
                logger.debug(
                    f"Sequence check: {exp.tool_name} found at position {pos} "
                    f"(order_index={exp.order_index})"
                )
                break

        if not found_valid_position:
            logger.warning(
                f"Sequence validation FAILED: {exp.tool_name} "
                f"(order_index={exp.order_index}) not found in correct order"
            )
            return False

    logger.info("Sequence validation PASSED")
    return True


def calculate_parameter_accuracy(results: list[ParameterMatchResult]) -> float:
    """Calculate overall parameter matching accuracy from results.

    Args:
        results: List of parameter match results.

    Returns:
        Accuracy score between 0.0 and 1.0.
    """
    if not results:
        return 1.0

    total_weight = sum(1.0 for _ in results)  # Could use weights from matchers
    matched_weight = sum(1.0 for r in results if r.matched)

    return matched_weight / total_weight if total_weight > 0 else 0.0


def calculate_sequence_accuracy(
    expected: list[ExpectedToolCall],
    actual: list[ToolCall],
) -> float:
    """Calculate how many tool calls were in the correct sequence position.

    Args:
        expected: List of expected tool calls.
        actual: List of actual tool calls.

    Returns:
        Accuracy score between 0.0 and 1.0.
    """
    ordered_expected = [e for e in expected if e.order_index is not None]
    if not ordered_expected:
        return 1.0

    correct_sequence = 0
    total_ordered = len(ordered_expected)

    # Sort by order_index
    ordered_expected = sorted(ordered_expected, key=lambda x: x.order_index or 0)

    # Build actual positions
    actual_positions: dict[str, list[int]] = {}
    for idx, call in enumerate(actual):
        if call.tool_name not in actual_positions:
            actual_positions[call.tool_name] = []
        actual_positions[call.tool_name].append(idx)

    last_position = -1
    for exp in ordered_expected:
        positions = actual_positions.get(exp.tool_name, [])
        for pos in positions:
            if pos > last_position:
                correct_sequence += 1
                last_position = pos
                break

    return correct_sequence / total_ordered if total_ordered > 0 else 1.0

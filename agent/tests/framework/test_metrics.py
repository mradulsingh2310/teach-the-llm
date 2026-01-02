"""
Test Metrics and Results Storage System.

This module provides:
1. Comprehensive metrics calculation (tool calling %, correct response %, correct tool call %)
2. Immutable test run storage with run_id tracking
3. Per-turn detailed results with expected vs actual comparisons
4. Support for averaging metrics across multiple runs
5. UI-ready JSON data structures

Metrics Definitions:
- tool_calling_rate: % of scenarios where expected tools were called at all
- correct_response_rate: % of responses containing expected keywords
- correct_tool_call_rate: % of tool calls with correct parameters
- file_verification_rate: % of tool calls that correctly wrote to log files
"""

import json
import os
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Path Configuration
# =============================================================================

# Directory for storing test results
TEST_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "gemma",
    "test_results"
)

# Log files to verify tool execution
TOUR_BOOKINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "tour_bookings.txt"
)

LEADS_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "leads.txt"
)


# =============================================================================
# Enumerations
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics tracked."""
    TOOL_CALLING_RATE = "tool_calling_rate"
    CORRECT_RESPONSE_RATE = "correct_response_rate"
    CORRECT_TOOL_CALL_RATE = "correct_tool_call_rate"
    FILE_VERIFICATION_RATE = "file_verification_rate"


class VerificationStatus(str, Enum):
    """Status of file-based verification."""
    SUCCESS = "success"
    FAILURE = "failure"
    NOT_APPLICABLE = "not_applicable"
    NOT_CHECKED = "not_checked"


# =============================================================================
# Expected Result Models
# =============================================================================

class ExpectedKeywords(BaseModel):
    """Keywords expected in response for a specific context."""
    required: list[str] = Field(default_factory=list, description="Keywords that MUST appear")
    optional: list[str] = Field(default_factory=list, description="Keywords that SHOULD appear")
    forbidden: list[str] = Field(default_factory=list, description="Keywords that must NOT appear")

    model_config = {"extra": "forbid"}


class ExpectedToolParameters(BaseModel):
    """Expected parameters for a tool call."""
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_params: list[str] = Field(default_factory=list, description="Parameters that must be present")
    param_validators: dict[str, str] = Field(default_factory=dict, description="Regex validators for params")

    model_config = {"extra": "forbid"}


class TurnExpectation(BaseModel):
    """Expected outcomes for a single conversation turn."""
    turn_number: int
    expected_tools: list[ExpectedToolParameters] = Field(default_factory=list)
    expected_keywords: ExpectedKeywords = Field(default_factory=ExpectedKeywords)
    should_write_to_file: Optional[str] = Field(None, description="Expected log file (leads.txt, tour_bookings.txt)")
    file_search_pattern: Optional[str] = Field(None, description="Pattern to search in log file to verify write")

    model_config = {"extra": "forbid"}


# =============================================================================
# Turn Result Models
# =============================================================================

class KeywordMatchResult(BaseModel):
    """Result of keyword matching for a response."""
    required_found: list[str] = Field(default_factory=list)
    required_missing: list[str] = Field(default_factory=list)
    optional_found: list[str] = Field(default_factory=list)
    forbidden_found: list[str] = Field(default_factory=list)
    match_score: float = Field(0.0, ge=0.0, le=1.0, description="0-1 score based on keyword matching")
    passed: bool = False

    model_config = {"extra": "forbid"}

    def calculate_score(self, expected: ExpectedKeywords) -> None:
        """Calculate the match score based on keyword results."""
        # All forbidden keywords found = automatic failure
        if self.forbidden_found:
            self.passed = False
            self.match_score = 0.0
            return

        # Required keywords score
        required_total = len(expected.required)
        required_found_count = len(self.required_found)

        if required_total > 0:
            required_score = required_found_count / required_total
        else:
            required_score = 1.0

        # Optional keywords bonus (up to 0.2 extra)
        optional_total = len(expected.optional)
        optional_found_count = len(self.optional_found)

        if optional_total > 0:
            optional_score = (optional_found_count / optional_total) * 0.2
        else:
            optional_score = 0.0

        self.match_score = min(1.0, required_score * 0.8 + optional_score + 0.2)
        self.passed = len(self.required_missing) == 0 and len(self.forbidden_found) == 0


class ToolCallEvaluation(BaseModel):
    """Evaluation of a single tool call."""
    tool_name: str
    was_called: bool = False
    parameters_correct: bool = False
    actual_parameters: dict[str, Any] = Field(default_factory=dict)
    expected_parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_errors: list[str] = Field(default_factory=list, description="Specific parameter mismatches")
    score: float = Field(0.0, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class FileVerificationResult(BaseModel):
    """Result of verifying a file write operation."""
    file_path: str
    status: VerificationStatus = VerificationStatus.NOT_CHECKED
    search_pattern: Optional[str] = None
    entry_found: bool = False
    matched_entry: Optional[str] = None
    error_message: Optional[str] = None

    model_config = {"extra": "forbid"}


class TurnMetrics(BaseModel):
    """Detailed metrics for a single conversation turn."""
    turn_number: int

    # User input
    user_message: str

    # Assistant response
    assistant_response: Optional[str] = None

    # Tool calling metrics
    tools_expected: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    tool_calling_success: bool = False  # Were expected tools called?
    tool_evaluations: list[ToolCallEvaluation] = Field(default_factory=list)

    # Response metrics
    keyword_results: Optional[KeywordMatchResult] = None
    response_correct: bool = False

    # Parameter metrics
    parameters_correct: bool = False
    parameter_score: float = Field(0.0, ge=0.0, le=1.0)

    # File verification
    file_verification: Optional[FileVerificationResult] = None

    # Timing
    latency_ms: int = 0

    # Raw data for UI
    raw_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    expected_tools_detail: list[ExpectedToolParameters] = Field(default_factory=list)
    expected_keywords_detail: Optional[ExpectedKeywords] = None

    model_config = {"extra": "forbid"}


# =============================================================================
# Scenario Result Models
# =============================================================================

class ScenarioMetrics(BaseModel):
    """Aggregated metrics for a single test scenario."""
    scenario_id: str
    scenario_name: str
    category: str

    # Pass/fail
    passed: bool = False

    # Turn-level results
    turn_metrics: list[TurnMetrics] = Field(default_factory=list)
    total_turns: int = 0

    # Aggregated metrics
    tool_calling_rate: float = Field(0.0, ge=0.0, le=1.0, description="% of turns where expected tools were called")
    correct_response_rate: float = Field(0.0, ge=0.0, le=1.0, description="% of turns with correct keywords")
    correct_tool_call_rate: float = Field(0.0, ge=0.0, le=1.0, description="% of tool calls with correct params")
    file_verification_rate: float = Field(0.0, ge=0.0, le=1.0, description="% of file writes verified")

    # Timing
    total_duration_ms: int = 0
    avg_turn_latency_ms: float = 0.0

    # Errors
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    def calculate_aggregates(self) -> None:
        """Calculate aggregated metrics from turn metrics."""
        if not self.turn_metrics:
            return

        self.total_turns = len(self.turn_metrics)

        # Tool calling rate
        turns_with_expected_tools = [t for t in self.turn_metrics if t.tools_expected]
        if turns_with_expected_tools:
            successful_tool_calls = sum(1 for t in turns_with_expected_tools if t.tool_calling_success)
            self.tool_calling_rate = successful_tool_calls / len(turns_with_expected_tools)
        else:
            self.tool_calling_rate = 1.0  # No tools expected = success

        # Correct response rate
        turns_with_keywords = [t for t in self.turn_metrics if t.keyword_results is not None]
        if turns_with_keywords:
            correct_responses = sum(1 for t in turns_with_keywords if t.response_correct)
            self.correct_response_rate = correct_responses / len(turns_with_keywords)
        else:
            self.correct_response_rate = 1.0

        # Correct tool call rate (parameter accuracy)
        all_evaluations = []
        for t in self.turn_metrics:
            all_evaluations.extend(t.tool_evaluations)

        if all_evaluations:
            correct_params = sum(1 for e in all_evaluations if e.was_called and e.parameters_correct)
            called_tools = sum(1 for e in all_evaluations if e.was_called)
            self.correct_tool_call_rate = correct_params / called_tools if called_tools > 0 else 0.0
        else:
            self.correct_tool_call_rate = 1.0

        # File verification rate
        file_verifications = [t.file_verification for t in self.turn_metrics if t.file_verification is not None]
        applicable_verifications = [v for v in file_verifications if v.status != VerificationStatus.NOT_APPLICABLE]
        if applicable_verifications:
            successful_verifications = sum(1 for v in applicable_verifications if v.status == VerificationStatus.SUCCESS)
            self.file_verification_rate = successful_verifications / len(applicable_verifications)
        else:
            self.file_verification_rate = 1.0

        # Timing
        self.total_duration_ms = sum(t.latency_ms for t in self.turn_metrics)
        self.avg_turn_latency_ms = self.total_duration_ms / self.total_turns if self.total_turns > 0 else 0.0

        # Overall pass
        self.passed = (
            self.tool_calling_rate >= 0.8 and
            self.correct_response_rate >= 0.6 and
            self.correct_tool_call_rate >= 0.7 and
            self.file_verification_rate >= 0.8
        )


# =============================================================================
# Test Run Models
# =============================================================================

class TestRunMetrics(BaseModel):
    """Aggregated metrics for a complete test run."""
    # Counts
    total_scenarios: int = 0
    passed_scenarios: int = 0
    failed_scenarios: int = 0

    # Rates (0-1)
    overall_pass_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_tool_calling_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_correct_response_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_correct_tool_call_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_file_verification_rate: float = Field(0.0, ge=0.0, le=1.0)

    # Per-category breakdown
    category_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Per-tool breakdown
    tool_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class TestRun(BaseModel):
    """A complete test run with all results and metrics."""
    # Identity
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    run_number: int = Field(1, description="Sequential run number for averaging")

    # Timestamps
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None
    duration_ms: int = 0

    # Configuration
    model_id: str = ""
    agent_type: str = ""
    test_suite_name: str = ""
    test_suite_version: str = ""

    # Results
    scenario_results: list[ScenarioMetrics] = Field(default_factory=list)

    # Aggregated metrics
    metrics: TestRunMetrics = Field(default_factory=TestRunMetrics)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def calculate_metrics(self) -> None:
        """Calculate aggregated metrics from scenario results."""
        if not self.scenario_results:
            return

        # Calculate each scenario's aggregates first
        for scenario in self.scenario_results:
            scenario.calculate_aggregates()

        self.metrics.total_scenarios = len(self.scenario_results)
        self.metrics.passed_scenarios = sum(1 for s in self.scenario_results if s.passed)
        self.metrics.failed_scenarios = self.metrics.total_scenarios - self.metrics.passed_scenarios

        # Overall pass rate
        self.metrics.overall_pass_rate = (
            self.metrics.passed_scenarios / self.metrics.total_scenarios
            if self.metrics.total_scenarios > 0 else 0.0
        )

        # Average rates
        self.metrics.avg_tool_calling_rate = sum(s.tool_calling_rate for s in self.scenario_results) / len(self.scenario_results)
        self.metrics.avg_correct_response_rate = sum(s.correct_response_rate for s in self.scenario_results) / len(self.scenario_results)
        self.metrics.avg_correct_tool_call_rate = sum(s.correct_tool_call_rate for s in self.scenario_results) / len(self.scenario_results)
        self.metrics.avg_file_verification_rate = sum(s.file_verification_rate for s in self.scenario_results) / len(self.scenario_results)

        # Per-category metrics
        categories: dict[str, list[ScenarioMetrics]] = {}
        for scenario in self.scenario_results:
            cat = scenario.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(scenario)

        for cat, scenarios in categories.items():
            self.metrics.category_metrics[cat] = {
                "total": len(scenarios),
                "passed": sum(1 for s in scenarios if s.passed),
                "pass_rate": sum(1 for s in scenarios if s.passed) / len(scenarios),
                "avg_tool_calling_rate": sum(s.tool_calling_rate for s in scenarios) / len(scenarios),
                "avg_correct_response_rate": sum(s.correct_response_rate for s in scenarios) / len(scenarios),
                "avg_correct_tool_call_rate": sum(s.correct_tool_call_rate for s in scenarios) / len(scenarios),
            }

        # Per-tool metrics
        tool_evals: dict[str, list[ToolCallEvaluation]] = {}
        for scenario in self.scenario_results:
            for turn in scenario.turn_metrics:
                for eval in turn.tool_evaluations:
                    if eval.tool_name not in tool_evals:
                        tool_evals[eval.tool_name] = []
                    tool_evals[eval.tool_name].append(eval)

        for tool_name, evals in tool_evals.items():
            called = [e for e in evals if e.was_called]
            self.metrics.tool_metrics[tool_name] = {
                "total_expected": len(evals),
                "total_called": len(called),
                "call_rate": len(called) / len(evals) if evals else 0.0,
                "param_accuracy": sum(e.score for e in called) / len(called) if called else 0.0,
                "correct_calls": sum(1 for e in called if e.parameters_correct),
            }

    def finalize(self) -> None:
        """Finalize the test run with end time and metrics."""
        self.ended_at = datetime.utcnow().isoformat()
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.ended_at)
            self.duration_ms = int((end - start).total_seconds() * 1000)
        self.calculate_metrics()


# =============================================================================
# Aggregated Results (Multiple Runs)
# =============================================================================

class AggregatedMetrics(BaseModel):
    """Metrics aggregated across multiple test runs."""
    total_runs: int = 0

    # Averages across runs
    avg_pass_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_tool_calling_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_correct_response_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_correct_tool_call_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_file_verification_rate: float = Field(0.0, ge=0.0, le=1.0)

    # Standard deviations
    std_pass_rate: float = 0.0
    std_tool_calling_rate: float = 0.0
    std_correct_response_rate: float = 0.0
    std_correct_tool_call_rate: float = 0.0

    # Min/Max
    min_pass_rate: float = 0.0
    max_pass_rate: float = 0.0

    # Per-scenario breakdown (for trend analysis)
    scenario_trends: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class TestResultsStore(BaseModel):
    """
    Immutable storage for test results across multiple runs.

    Each run is stored independently and can be analyzed individually
    or in aggregate. Supports:
    - Individual run analysis
    - Averaging across n runs
    - Trend analysis over time
    - UI-ready JSON export
    """
    # Identity
    store_id: str = Field(default_factory=lambda: f"store_{uuid4().hex[:8]}")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Configuration
    model_id: str = ""
    agent_type: str = ""
    test_suite_name: str = ""

    # All runs (immutable - append only)
    runs: list[TestRun] = Field(default_factory=list)

    # Aggregated metrics
    aggregated_metrics: AggregatedMetrics = Field(default_factory=AggregatedMetrics)

    model_config = {"extra": "forbid"}

    def add_run(self, run: TestRun) -> None:
        """Add a completed test run to the store."""
        run.run_number = len(self.runs) + 1
        self.runs.append(run)
        self.updated_at = datetime.utcnow().isoformat()
        self._recalculate_aggregates()

    def _recalculate_aggregates(self) -> None:
        """Recalculate aggregated metrics across all runs."""
        if not self.runs:
            return

        n = len(self.runs)
        self.aggregated_metrics.total_runs = n

        # Collect all metrics
        pass_rates = [r.metrics.overall_pass_rate for r in self.runs]
        tool_rates = [r.metrics.avg_tool_calling_rate for r in self.runs]
        response_rates = [r.metrics.avg_correct_response_rate for r in self.runs]
        param_rates = [r.metrics.avg_correct_tool_call_rate for r in self.runs]
        file_rates = [r.metrics.avg_file_verification_rate for r in self.runs]

        # Averages
        self.aggregated_metrics.avg_pass_rate = sum(pass_rates) / n
        self.aggregated_metrics.avg_tool_calling_rate = sum(tool_rates) / n
        self.aggregated_metrics.avg_correct_response_rate = sum(response_rates) / n
        self.aggregated_metrics.avg_correct_tool_call_rate = sum(param_rates) / n
        self.aggregated_metrics.avg_file_verification_rate = sum(file_rates) / n

        # Standard deviations
        if n > 1:
            import statistics
            self.aggregated_metrics.std_pass_rate = statistics.stdev(pass_rates)
            self.aggregated_metrics.std_tool_calling_rate = statistics.stdev(tool_rates)
            self.aggregated_metrics.std_correct_response_rate = statistics.stdev(response_rates)
            self.aggregated_metrics.std_correct_tool_call_rate = statistics.stdev(param_rates)

        # Min/Max
        self.aggregated_metrics.min_pass_rate = min(pass_rates)
        self.aggregated_metrics.max_pass_rate = max(pass_rates)

        # Per-scenario trends
        scenario_data: dict[str, list[dict]] = {}
        for run in self.runs:
            for scenario in run.scenario_results:
                if scenario.scenario_id not in scenario_data:
                    scenario_data[scenario.scenario_id] = []
                scenario_data[scenario.scenario_id].append({
                    "run_number": run.run_number,
                    "passed": scenario.passed,
                    "tool_calling_rate": scenario.tool_calling_rate,
                    "correct_response_rate": scenario.correct_response_rate,
                    "correct_tool_call_rate": scenario.correct_tool_call_rate,
                })

        for scenario_id, data_points in scenario_data.items():
            self.aggregated_metrics.scenario_trends[scenario_id] = {
                "runs": len(data_points),
                "pass_count": sum(1 for d in data_points if d["passed"]),
                "pass_rate": sum(1 for d in data_points if d["passed"]) / len(data_points),
                "avg_tool_calling_rate": sum(d["tool_calling_rate"] for d in data_points) / len(data_points),
                "avg_correct_response_rate": sum(d["correct_response_rate"] for d in data_points) / len(data_points),
                "avg_correct_tool_call_rate": sum(d["correct_tool_call_rate"] for d in data_points) / len(data_points),
                "trend": data_points,  # All data points for charting
            }

    def get_latest_run(self) -> Optional[TestRun]:
        """Get the most recent test run."""
        return self.runs[-1] if self.runs else None

    def get_run_by_id(self, run_id: str) -> Optional[TestRun]:
        """Get a specific run by ID."""
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None

    def to_ui_json(self) -> dict[str, Any]:
        """Export data structure optimized for UI consumption."""
        return {
            "store_id": self.store_id,
            "model_id": self.model_id,
            "agent_type": self.agent_type,
            "test_suite_name": self.test_suite_name,
            "total_runs": self.aggregated_metrics.total_runs,
            "summary": {
                "avg_pass_rate": round(self.aggregated_metrics.avg_pass_rate * 100, 2),
                "avg_tool_calling_rate": round(self.aggregated_metrics.avg_tool_calling_rate * 100, 2),
                "avg_correct_response_rate": round(self.aggregated_metrics.avg_correct_response_rate * 100, 2),
                "avg_correct_tool_call_rate": round(self.aggregated_metrics.avg_correct_tool_call_rate * 100, 2),
                "avg_file_verification_rate": round(self.aggregated_metrics.avg_file_verification_rate * 100, 2),
            },
            "runs": [
                {
                    "run_id": run.run_id,
                    "run_number": run.run_number,
                    "started_at": run.started_at,
                    "duration_ms": run.duration_ms,
                    "pass_rate": round(run.metrics.overall_pass_rate * 100, 2),
                    "total_scenarios": run.metrics.total_scenarios,
                    "passed_scenarios": run.metrics.passed_scenarios,
                }
                for run in self.runs
            ],
            "scenario_trends": self.aggregated_metrics.scenario_trends,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def check_keywords_in_response(response: str, expected: ExpectedKeywords) -> KeywordMatchResult:
    """
    Check if expected keywords are present in the response.

    Args:
        response: The assistant's response text
        expected: The expected keywords configuration

    Returns:
        KeywordMatchResult with detailed findings
    """
    result = KeywordMatchResult()
    response_lower = response.lower()

    # Check required keywords
    for keyword in expected.required:
        if keyword.lower() in response_lower or re.search(keyword, response, re.IGNORECASE):
            result.required_found.append(keyword)
        else:
            result.required_missing.append(keyword)

    # Check optional keywords
    for keyword in expected.optional:
        if keyword.lower() in response_lower or re.search(keyword, response, re.IGNORECASE):
            result.optional_found.append(keyword)

    # Check forbidden keywords
    for keyword in expected.forbidden:
        if keyword.lower() in response_lower or re.search(keyword, response, re.IGNORECASE):
            result.forbidden_found.append(keyword)

    result.calculate_score(expected)
    return result


def evaluate_tool_call(
    expected: ExpectedToolParameters,
    actual_calls: list[dict[str, Any]]
) -> ToolCallEvaluation:
    """
    Evaluate if a tool was called with correct parameters.

    Args:
        expected: Expected tool call specification
        actual_calls: List of actual tool calls made

    Returns:
        ToolCallEvaluation with detailed results
    """
    evaluation = ToolCallEvaluation(
        tool_name=expected.tool_name,
        expected_parameters=expected.parameters,
    )

    # Find matching tool call
    matching_call = None
    for call in actual_calls:
        if call.get("tool_name") == expected.tool_name:
            matching_call = call
            break

    if not matching_call:
        evaluation.was_called = False
        evaluation.score = 0.0
        return evaluation

    evaluation.was_called = True
    evaluation.actual_parameters = matching_call.get("input", {})

    # Check required parameters
    missing_required = []
    for param in expected.required_params:
        if param not in evaluation.actual_parameters:
            missing_required.append(f"Missing required parameter: {param}")

    if missing_required:
        evaluation.parameter_errors.extend(missing_required)

    # Check parameter values
    param_scores = []
    for param_name, expected_value in expected.parameters.items():
        actual_value = evaluation.actual_parameters.get(param_name)

        if actual_value is None:
            evaluation.parameter_errors.append(f"Parameter '{param_name}' not provided (expected: {expected_value})")
            param_scores.append(0.0)
            continue

        # Check regex validator if present
        if param_name in expected.param_validators:
            pattern = expected.param_validators[param_name]
            if isinstance(actual_value, str) and not re.match(pattern, actual_value):
                evaluation.parameter_errors.append(
                    f"Parameter '{param_name}' doesn't match pattern '{pattern}': {actual_value}"
                )
                param_scores.append(0.0)
                continue

        # Compare values
        if _values_match(expected_value, actual_value):
            param_scores.append(1.0)
        else:
            evaluation.parameter_errors.append(
                f"Parameter '{param_name}' mismatch - expected: {expected_value}, got: {actual_value}"
            )
            param_scores.append(0.0)

    # Calculate overall score
    if param_scores:
        evaluation.score = sum(param_scores) / len(param_scores)
    else:
        evaluation.score = 1.0 if not evaluation.parameter_errors else 0.0

    evaluation.parameters_correct = (
        len(evaluation.parameter_errors) == 0 and
        len(missing_required) == 0
    )

    return evaluation


def _values_match(expected: Any, actual: Any) -> bool:
    """Check if two values match, handling various types."""
    # Handle lists (order-independent)
    if isinstance(expected, list) and isinstance(actual, list):
        return set(expected) == set(actual) or sorted(expected) == sorted(actual)

    # Handle numeric comparisons with tolerance
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 0.01

    # Handle strings (case-insensitive contains)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower() in actual.lower() or actual.lower() in expected.lower()

    # Direct comparison
    return expected == actual


def verify_file_write(
    file_type: str,
    search_pattern: str
) -> FileVerificationResult:
    """
    Verify that a tool wrote to the expected log file.

    Args:
        file_type: "leads" or "tour_bookings"
        search_pattern: Pattern to search for in the file

    Returns:
        FileVerificationResult with verification status
    """
    file_path = LEADS_LOG_PATH if file_type == "leads" else TOUR_BOOKINGS_PATH

    result = FileVerificationResult(
        file_path=file_path,
        search_pattern=search_pattern,
    )

    try:
        if not os.path.exists(file_path):
            result.status = VerificationStatus.FAILURE
            result.error_message = f"File does not exist: {file_path}"
            return result

        with open(file_path, "r") as f:
            content = f.read()

        # Search for the pattern
        if search_pattern in content:
            result.entry_found = True
            result.status = VerificationStatus.SUCCESS

            # Find the matching line
            for line in content.split("\n"):
                if search_pattern in line:
                    result.matched_entry = line
                    break
        else:
            result.entry_found = False
            result.status = VerificationStatus.FAILURE
            result.error_message = f"Pattern not found in file: {search_pattern}"

    except Exception as e:
        result.status = VerificationStatus.FAILURE
        result.error_message = str(e)

    return result


def save_results_store(store: TestResultsStore, filename: Optional[str] = None) -> str:
    """
    Save the test results store to a JSON file.

    Args:
        store: The TestResultsStore to save
        filename: Optional filename (defaults to store_id.json)

    Returns:
        Path to the saved file
    """
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    if filename is None:
        filename = f"{store.store_id}.json"

    filepath = os.path.join(TEST_RESULTS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(store.model_dump(), f, indent=2, default=str)

    return filepath


def load_results_store(filepath: str) -> TestResultsStore:
    """
    Load a test results store from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Loaded TestResultsStore
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return TestResultsStore.model_validate(data)


def get_or_create_store(
    model_id: str,
    agent_type: str,
    test_suite_name: str
) -> TestResultsStore:
    """
    Get existing results store or create a new one.

    Args:
        model_id: Model identifier
        agent_type: Type of agent
        test_suite_name: Name of test suite

    Returns:
        TestResultsStore (existing or new)
    """
    # Look for existing store
    store_filename = f"store_{model_id.replace('/', '_')}_{test_suite_name}.json"
    store_path = os.path.join(TEST_RESULTS_DIR, store_filename)

    if os.path.exists(store_path):
        return load_results_store(store_path)

    # Create new store
    return TestResultsStore(
        model_id=model_id,
        agent_type=agent_type,
        test_suite_name=test_suite_name,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "MetricType",
    "VerificationStatus",
    # Expected Result Models
    "ExpectedKeywords",
    "ExpectedToolParameters",
    "TurnExpectation",
    # Turn Result Models
    "KeywordMatchResult",
    "ToolCallEvaluation",
    "FileVerificationResult",
    "TurnMetrics",
    # Scenario Result Models
    "ScenarioMetrics",
    # Test Run Models
    "TestRunMetrics",
    "TestRun",
    # Aggregated Results
    "AggregatedMetrics",
    "TestResultsStore",
    # Utility Functions
    "check_keywords_in_response",
    "evaluate_tool_call",
    "verify_file_write",
    "save_results_store",
    "load_results_store",
    "get_or_create_store",
]

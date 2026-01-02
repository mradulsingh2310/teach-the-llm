"""Test Reporter for AI Agent testing framework.

This module provides comprehensive reporting capabilities:
- Generate test reports from results
- Print formatted console output
- Save reports as JSON files
- Analyze tool usage and failures
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from .models import (
    FailureAnalysis,
    SuiteResult,
    TestReport,
    TestResult,
    ToolBreakdown,
    ToolMetrics,
)


logger = logging.getLogger(__name__)


class TestReporter:
    """Reporter for generating and formatting test results."""

    def __init__(
        self,
        agent_type: str = "property_agent",
        model_id: str = "",
        environment: str = "test",
    ):
        """Initialize the reporter.

        Args:
            agent_type: Type of agent being tested.
            model_id: Model ID being tested.
            environment: Test environment name.
        """
        self.agent_type = agent_type
        self.model_id = model_id
        self.environment = environment
        logger.info(f"TestReporter initialized for {agent_type}")

    def generate_report(self, results: list[TestResult]) -> TestReport:
        """Generate a comprehensive test report from results.

        Args:
            results: List of TestResult objects.

        Returns:
            TestReport with aggregated statistics and analysis.
        """
        logger.info(f"Generating report from {len(results)} test result(s)")

        report = TestReport(
            agent_type=self.agent_type,
            model_id=self.model_id,
            environment=self.environment,
        )

        # Aggregate results
        report.total_tests = len(results)

        for result in results:
            if "Skipped" in (result.error_message or ""):
                report.total_skipped += 1
            elif result.passed:
                report.total_passed += 1
            else:
                report.total_failed += 1

            report.total_duration_ms += result.duration_ms

        # Calculate pass rate
        if report.total_tests > 0:
            report.overall_pass_rate = report.total_passed / report.total_tests

        # Aggregate tool metrics
        report.overall_tool_metrics = self._aggregate_tool_metrics(results)

        # Generate tool breakdown
        report.tool_breakdown = self.generate_tool_breakdown(results)

        # Analyze failures
        report.failure_analysis = self.analyze_failures(results)

        logger.info(
            f"Report generated: {report.total_passed}/{report.total_tests} passed "
            f"({report.overall_pass_rate:.1%})"
        )

        return report

    def _aggregate_tool_metrics(self, results: list[TestResult]) -> ToolMetrics:
        """Aggregate tool metrics across all results.

        Args:
            results: List of test results.

        Returns:
            Aggregated ToolMetrics.
        """
        total_expected = 0
        total_actual = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for result in results:
            total_expected += result.tool_metrics.total_expected
            total_actual += result.tool_metrics.total_actual
            total_tp += result.tool_metrics.true_positives
            total_fp += result.tool_metrics.false_positives
            total_fn += result.tool_metrics.false_negatives

        metrics = ToolMetrics(
            total_expected=total_expected,
            total_actual=total_actual,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        )

        # Calculate metrics
        if total_actual > 0:
            metrics.precision = total_tp / total_actual
        else:
            metrics.precision = 1.0 if total_expected == 0 else 0.0

        if total_expected > 0:
            metrics.recall = total_tp / total_expected
        else:
            metrics.recall = 1.0 if total_actual == 0 else 0.0

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = (
                2 * (metrics.precision * metrics.recall) /
                (metrics.precision + metrics.recall)
            )

        return metrics

    def generate_tool_breakdown(self, results: list[TestResult]) -> list[ToolBreakdown]:
        """Generate a breakdown of results by tool.

        Args:
            results: List of test results.

        Returns:
            List of ToolBreakdown objects, one per tool.
        """
        logger.debug("Generating tool breakdown")

        # Collect data per tool
        tool_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total_expected": 0,
                "total_actual": 0,
                "correct_calls": 0,
                "incorrect_calls": 0,
                "missed_calls": 0,
                "param_match_count": 0,
                "param_total_count": 0,
            }
        )

        for result in results:
            # Track actual calls
            for tc_dict in result.actual_tool_calls:
                tool_name = tc_dict.get("tool_name", "unknown")
                tool_data[tool_name]["total_actual"] += 1

            # Track expected calls and match results
            for tc_result in result.tool_call_results:
                tool_name = tc_result.expected.tool_name
                tool_data[tool_name]["total_expected"] += 1

                if tc_result.matched:
                    tool_data[tool_name]["correct_calls"] += 1
                else:
                    if tc_result.matched_actual_index is not None:
                        tool_data[tool_name]["incorrect_calls"] += 1
                    else:
                        tool_data[tool_name]["missed_calls"] += 1

                # Track parameter accuracy
                for param_result in tc_result.parameter_results:
                    tool_data[tool_name]["param_total_count"] += 1
                    if param_result.matched:
                        tool_data[tool_name]["param_match_count"] += 1

        # Build breakdown objects
        breakdowns = []
        for tool_name, data in sorted(tool_data.items()):
            total = data["total_expected"] or data["total_actual"]
            accuracy = data["correct_calls"] / total if total > 0 else 0.0

            param_accuracy = 0.0
            if data["param_total_count"] > 0:
                param_accuracy = data["param_match_count"] / data["param_total_count"]

            breakdown = ToolBreakdown(
                tool_name=tool_name,
                total_expected=data["total_expected"],
                total_actual=data["total_actual"],
                correct_calls=data["correct_calls"],
                incorrect_calls=data["incorrect_calls"],
                missed_calls=data["missed_calls"],
                accuracy=accuracy,
                parameter_accuracy=param_accuracy,
            )
            breakdowns.append(breakdown)

        logger.debug(f"Generated breakdown for {len(breakdowns)} tool(s)")
        return breakdowns

    def analyze_failures(self, results: list[TestResult]) -> list[FailureAnalysis]:
        """Analyze test failures and provide insights.

        Args:
            results: List of test results.

        Returns:
            List of FailureAnalysis objects.
        """
        logger.debug("Analyzing test failures")

        analyses = []

        for result in results:
            if result.passed:
                continue

            for failure in result.failures:
                analysis = self._analyze_single_failure(result, failure)
                if analysis:
                    analyses.append(analysis)

            # Also analyze exceptions
            if result.exception_type:
                analysis = FailureAnalysis(
                    test_case_id=result.test_case_id,
                    failure_type="exception",
                    description=f"Exception occurred: {result.exception_type}",
                    actual=result.error_message,
                    suggestions=[
                        "Check for runtime errors in the agent code",
                        "Verify mock configurations are correct",
                        "Review the stack trace for details",
                    ],
                )
                analyses.append(analysis)

        logger.debug(f"Analyzed {len(analyses)} failure(s)")
        return analyses

    def _analyze_single_failure(
        self,
        result: TestResult,
        failure: str,
    ) -> FailureAnalysis | None:
        """Analyze a single failure and generate insights.

        Args:
            result: The test result containing the failure.
            failure: The failure message.

        Returns:
            FailureAnalysis or None if cannot analyze.
        """
        # Determine failure type
        failure_type = "tool_mismatch"
        suggestions = []

        if "Forbidden tool was used" in failure:
            failure_type = "forbidden_tool_used"
            tool_name = failure.split(": ")[-1] if ": " in failure else "unknown"
            suggestions = [
                f"Review why '{tool_name}' was called when it shouldn't be",
                "Check if there's an alternative tool that should be used",
                "Verify the agent's tool selection logic",
            ]

        elif "Required tool was not used" in failure:
            failure_type = "missing_required_tool"
            tool_name = failure.split(": ")[-1] if ": " in failure else "unknown"
            suggestions = [
                f"Verify that '{tool_name}' is available to the agent",
                "Check if the prompt clearly requires this tool",
                "Review the agent's understanding of when to use this tool",
            ]

        elif "Expected tool call not matched" in failure:
            failure_type = "tool_mismatch"
            suggestions = [
                "Check parameter values passed to the tool",
                "Verify the tool name is spelled correctly",
                "Review if the matching strategy is appropriate",
            ]

        elif "sequence did not match" in failure:
            failure_type = "sequence_error"
            suggestions = [
                "Review the expected order of tool calls",
                "Check if some tools can actually be called in any order",
                "Consider if strict sequencing is necessary",
            ]

        elif "Response assertion failed" in failure:
            failure_type = "response_assertion_failed"
            suggestions = [
                "Check if the expected response pattern is correct",
                "Review case sensitivity settings",
                "Verify the agent produces the expected output format",
            ]

        elif "Too many tool calls" in failure or "Too few tool calls" in failure:
            failure_type = "tool_mismatch"
            suggestions = [
                "Review the expected number of tool calls",
                "Check if the agent is being efficient",
                "Verify min/max constraints are reasonable",
            ]

        elif "Latency exceeded" in failure:
            failure_type = "timeout"
            suggestions = [
                "Check for performance bottlenecks",
                "Consider increasing the timeout threshold",
                "Review if the test expectations are realistic",
            ]

        return FailureAnalysis(
            test_case_id=result.test_case_id,
            failure_type=failure_type,
            description=failure,
            expected=None,
            actual=result.final_response[:200] if result.final_response else None,
            suggestions=suggestions,
        )

    def print_summary(self, report: TestReport) -> None:
        """Print a formatted summary to the console.

        Args:
            report: The test report to summarize.
        """
        # Header
        print("\n" + "=" * 70)
        print("TEST REPORT SUMMARY")
        print("=" * 70)
        print(f"Generated: {report.generated_at}")
        print(f"Agent: {report.agent_type}")
        print(f"Model: {report.model_id or 'N/A'}")
        print(f"Environment: {report.environment}")
        print()

        # Overall Results
        print("-" * 70)
        print("OVERALL RESULTS")
        print("-" * 70)

        status_emoji = "[PASS]" if report.overall_pass_rate >= 1.0 else "[FAIL]"
        print(f"Status: {status_emoji}")
        print(f"Total Tests: {report.total_tests}")
        print(f"  Passed: {report.total_passed}")
        print(f"  Failed: {report.total_failed}")
        print(f"  Skipped: {report.total_skipped}")
        print(f"Pass Rate: {report.overall_pass_rate:.1%}")
        print(f"Total Duration: {report.total_duration_ms}ms")
        print()

        # Tool Metrics
        print("-" * 70)
        print("TOOL CALL METRICS")
        print("-" * 70)
        metrics = report.overall_tool_metrics
        print(f"Precision: {metrics.precision:.2%}")
        print(f"Recall: {metrics.recall:.2%}")
        print(f"F1 Score: {metrics.f1_score:.2%}")
        print(f"True Positives: {metrics.true_positives}")
        print(f"False Positives: {metrics.false_positives}")
        print(f"False Negatives: {metrics.false_negatives}")
        print()

        # Tool Breakdown
        if report.tool_breakdown:
            print("-" * 70)
            print("TOOL BREAKDOWN")
            print("-" * 70)
            print(f"{'Tool':<30} {'Expected':>10} {'Actual':>10} {'Accuracy':>10}")
            print("-" * 70)
            for tb in report.tool_breakdown:
                accuracy_str = f"{tb.accuracy:.1%}" if tb.total_expected > 0 else "N/A"
                print(f"{tb.tool_name:<30} {tb.total_expected:>10} {tb.total_actual:>10} {accuracy_str:>10}")
            print()

        # Failure Analysis
        if report.failure_analysis:
            print("-" * 70)
            print("FAILURE ANALYSIS")
            print("-" * 70)
            for idx, fa in enumerate(report.failure_analysis, 1):
                print(f"\n{idx}. Test: {fa.test_case_id}")
                print(f"   Type: {fa.failure_type}")
                print(f"   Description: {fa.description}")
                if fa.suggestions:
                    print("   Suggestions:")
                    for suggestion in fa.suggestions:
                        print(f"     - {suggestion}")
            print()

        # Suite Results
        if report.suite_results:
            print("-" * 70)
            print("SUITE RESULTS")
            print("-" * 70)
            for sr in report.suite_results:
                suite_status = "[PASS]" if sr.pass_rate >= 1.0 else "[FAIL]"
                print(f"{suite_status} {sr.suite_name}: {sr.passed_tests}/{sr.total_tests} passed ({sr.pass_rate:.1%})")
            print()

        print("=" * 70)
        print("END OF REPORT")
        print("=" * 70 + "\n")

    def save_report(self, report: TestReport, path: str) -> None:
        """Save the report as a JSON file.

        Args:
            report: The test report to save.
            path: File path to save the report.
        """
        logger.info(f"Saving report to {path}")

        try:
            report_dict = report.model_dump()
            with open(path, "w") as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"Report saved successfully to {path}")
        except Exception as e:
            logger.exception(f"Failed to save report to {path}")
            raise

    def load_report(self, path: str) -> TestReport:
        """Load a report from a JSON file.

        Args:
            path: File path to load the report from.

        Returns:
            TestReport loaded from the file.
        """
        logger.info(f"Loading report from {path}")

        try:
            with open(path, "r") as f:
                report_dict = json.load(f)
            report = TestReport(**report_dict)
            logger.info(f"Report loaded successfully from {path}")
            return report
        except Exception as e:
            logger.exception(f"Failed to load report from {path}")
            raise

    def generate_html_report(self, report: TestReport) -> str:
        """Generate an HTML version of the report.

        Args:
            report: The test report.

        Returns:
            HTML string representation of the report.
        """
        pass_color = "#28a745" if report.overall_pass_rate >= 1.0 else "#dc3545"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {report.generated_at}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: {pass_color}; }}
        .failure-item {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <h1>AI Agent Test Report</h1>

    <div class="summary">
        <p><strong>Generated:</strong> {report.generated_at}</p>
        <p><strong>Agent:</strong> {report.agent_type}</p>
        <p><strong>Model:</strong> {report.model_id or 'N/A'}</p>
        <p><strong>Environment:</strong> {report.environment}</p>
    </div>

    <h2>Overall Results</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{report.overall_pass_rate:.1%}</div>
            <div>Pass Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.total_passed}/{report.total_tests}</div>
            <div>Tests Passed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.total_duration_ms}ms</div>
            <div>Total Duration</div>
        </div>
    </div>

    <h2>Tool Call Metrics</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{report.overall_tool_metrics.precision:.1%}</div>
            <div>Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.overall_tool_metrics.recall:.1%}</div>
            <div>Recall</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.overall_tool_metrics.f1_score:.1%}</div>
            <div>F1 Score</div>
        </div>
    </div>

    <h2>Tool Breakdown</h2>
    <table>
        <tr>
            <th>Tool</th>
            <th>Expected</th>
            <th>Actual</th>
            <th>Correct</th>
            <th>Accuracy</th>
            <th>Param Accuracy</th>
        </tr>
"""

        for tb in report.tool_breakdown:
            accuracy_str = f"{tb.accuracy:.1%}" if tb.total_expected > 0 else "N/A"
            param_accuracy_str = f"{tb.parameter_accuracy:.1%}" if tb.total_expected > 0 else "N/A"
            html += f"""
        <tr>
            <td>{tb.tool_name}</td>
            <td>{tb.total_expected}</td>
            <td>{tb.total_actual}</td>
            <td>{tb.correct_calls}</td>
            <td>{accuracy_str}</td>
            <td>{param_accuracy_str}</td>
        </tr>
"""

        html += """
    </table>
"""

        if report.failure_analysis:
            html += """
    <h2>Failure Analysis</h2>
"""
            for fa in report.failure_analysis:
                html += f"""
    <div class="failure-item">
        <strong>Test:</strong> {fa.test_case_id}<br>
        <strong>Type:</strong> {fa.failure_type}<br>
        <strong>Description:</strong> {fa.description}<br>
"""
                if fa.suggestions:
                    html += "<strong>Suggestions:</strong><ul>"
                    for suggestion in fa.suggestions:
                        html += f"<li>{suggestion}</li>"
                    html += "</ul>"
                html += "</div>\n"

        html += """
</body>
</html>
"""

        return html

    def save_html_report(self, report: TestReport, path: str) -> None:
        """Save the report as an HTML file.

        Args:
            report: The test report.
            path: File path to save the HTML report.
        """
        logger.info(f"Saving HTML report to {path}")

        try:
            html = self.generate_html_report(report)
            with open(path, "w") as f:
                f.write(html)
            logger.info(f"HTML report saved successfully to {path}")
        except Exception as e:
            logger.exception(f"Failed to save HTML report to {path}")
            raise

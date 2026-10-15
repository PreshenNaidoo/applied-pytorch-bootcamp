"""Helpers for simple PASS / TRY_AGAIN messages."""

from __future__ import annotations


def check_threshold(metric_value: float, minimum: float) -> bool:
    """Return True when a metric reaches a minimum target."""
    return float(metric_value) >= float(minimum)


def check_metric_range(metric_value: float, low: float, high: float) -> bool:
    """Return True when a metric stays inside a range."""
    return float(low) <= float(metric_value) <= float(high)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a simple training result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")

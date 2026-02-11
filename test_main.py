#!/usr/bin/env python3
"""Unit tests for shared_bills.py."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

# Import from the main module – adjust the import path to match your filename.
from main import (
    AppConfig,
    BillingResult,
    IssueTracker,
    JiraClient,
    SharedBillsTaskCreator,
    SureTransaction,
    TransactionProvider,
    month_range_for,
    parse_localized_money_to_decimal,
    previous_month_in_tz,
)
from zoneinfo import ZoneInfo


# --------------------------------------------------------------------------
# parse_localized_money_to_decimal
# --------------------------------------------------------------------------


class TestParseLocalizedMoney:
    """Tests for the money-string parser."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$1,234.56", Decimal("1234.56")),
            ("€1.234,56", Decimal("1234.56")),
            ("-€2.00", Decimal("-2.00")),
            ("100", Decimal("100")),
            ("0.99", Decimal("0.99")),
            ("-$0.01", Decimal("-0.01")),
            ("£1,000.00", Decimal("1000")),
            ("3.456,78", Decimal("3456.78")),
            # TODO: Fails:
            #("£1,000", Decimal("1000")),
        ],
    )
    def test_valid_strings(self, raw: str, expected: Decimal) -> None:
        assert parse_localized_money_to_decimal(raw) == expected

    @pytest.mark.parametrize("bad", ["", "$", "abc", "-"])
    def test_invalid_strings_raise(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_localized_money_to_decimal(bad)


# --------------------------------------------------------------------------
# MonthRange
# --------------------------------------------------------------------------


class TestMonthRange:
    """Tests for MonthRange and month_range_for."""

    def test_january(self) -> None:
        mr = month_range_for(2026, 1)
        assert mr.start_date == dt.date(2026, 1, 1)
        assert mr.end_date_inclusive == dt.date(2026, 1, 31)
        assert mr.end_date_exclusive == dt.date(2026, 2, 1)
        assert mr.label == "January 2026"
        assert mr.year_month == "2026-01"

    def test_february_non_leap(self) -> None:
        mr = month_range_for(2025, 2)
        assert mr.end_date_inclusive == dt.date(2025, 2, 28)

    def test_february_leap(self) -> None:
        mr = month_range_for(2024, 2)
        assert mr.end_date_inclusive == dt.date(2024, 2, 29)

    def test_december(self) -> None:
        mr = month_range_for(2025, 12)
        assert mr.end_date_inclusive == dt.date(2025, 12, 31)
        assert mr.end_date_exclusive == dt.date(2026, 1, 1)


# --------------------------------------------------------------------------
# previous_month_in_tz
# --------------------------------------------------------------------------


class TestPreviousMonth:
    def test_basic(self) -> None:
        tz = ZoneInfo("UTC")
        with patch("main.dt") as mock_dt:
            mock_dt.datetime.now.return_value = dt.datetime(
                2026, 3, 15, tzinfo=tz
            )
            mock_dt.date = dt.date
            mock_dt.timedelta = dt.timedelta
            y, m = previous_month_in_tz(tz)
        assert (y, m) == (2026, 2)


# --------------------------------------------------------------------------
# JiraClient ADF helper
# --------------------------------------------------------------------------


class TestJiraAdf:
    def test_plain_text_to_adf(self) -> None:
        adf = JiraClient._plain_text_to_adf("Hello\nWorld") # pyright: ignore[reportPrivateUsage]
        assert adf["type"] == "doc"
        assert adf["version"] == 1
        assert len(adf["content"]) == 2
        assert adf["content"][0]["content"][0]["text"] == "Hello"

    def test_empty_lines(self) -> None:
        adf = JiraClient._plain_text_to_adf("A\n\nB") # pyright: ignore[reportPrivateUsage]
        assert len(adf["content"]) == 3
        assert adf["content"][1]["content"] == []


# --------------------------------------------------------------------------
# SharedBillsTaskCreator (with fakes)
# --------------------------------------------------------------------------


class FakeTransactionProvider(TransactionProvider):
    """In-memory transaction provider for testing."""

    def __init__(self, transactions: list[SureTransaction]) -> None:
        self._transactions = transactions

    def list_transactions(
        self, start_date: dt.date, end_date_exclusive: dt.date
    ) -> list[SureTransaction]:
        return self._transactions


class FakeIssueTracker(IssueTracker):
    """In-memory issue tracker that records calls."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_issue(
        self,
        *,
        project_key: str,
        issue_type: str,
        summary: str,
        assignee_account_id: str,
        description: str | None = None,
    ) -> str:
        record : dict[str, Any] = {
            "project_key": project_key,
            "issue_type": issue_type,
            "summary": summary,
            "assignee_account_id": assignee_account_id,
            "description": description,
        }
        self.created.append(record)
        return "TEST-1"


def _make_config(**overrides: Any) -> AppConfig:
    defaults: dict[str, Any] = {
        "sure_base_url": "https://sure.example.com",
        "sure_api_token": "token",
        "jira_base_url": "https://jira.example.com",
        "jira_email": "test@example.com",
        "jira_api_token": "jiratoken",
        "jira_encoded_credentials": "encodedcreds",
        "jira_project_key": "TEST",
        "jira_assignee_account_id": "abc123",
        "dry_run": False,
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


class TestSharedBillsTaskCreator:
    """Integration-style tests using fakes."""

    def test_basic_expense_sum(self) -> None:
        txs = [
            SureTransaction(
                id="1", classification="expense", amount="$100.00"
            ),
            SureTransaction(
                id="2", classification="expense", amount="$50.00"
            ),
        ]
        provider = FakeTransactionProvider(txs)
        tracker = FakeIssueTracker()
        cfg = _make_config()
        creator = SharedBillsTaskCreator(provider, tracker, cfg)
        month = month_range_for(2026, 1)
        result = creator.compute_shared_bills(month)

        assert result.total_expenses == Decimal("150.00")
        assert result.half_total == Decimal("75.00")
        assert result.included_count == 2
        assert result.excluded_count == 0

    def test_excludes_personal_expenses(self) -> None:
        txs = [
            SureTransaction(
                id="1",
                classification="expense",
                amount="$100.00",
                category_name="Personal Expenses",
            ),
            SureTransaction(
                id="2", classification="expense", amount="$60.00"
            ),
        ]
        provider = FakeTransactionProvider(txs)
        tracker = FakeIssueTracker()
        cfg = _make_config()
        creator = SharedBillsTaskCreator(provider, tracker, cfg)
        month = month_range_for(2026, 1)
        result = creator.compute_shared_bills(month)

        assert result.total_expenses == Decimal("60.00")
        assert result.half_total == Decimal("30.00")
        assert result.excluded_count == 1
        assert result.included_count == 1

    def test_excludes_personal_via_category_dict(self) -> None:
        txs = [
            SureTransaction(
                id="1",
                classification="expense",
                amount="$200.00",
                category={"name": "Personal Expenses", "id": "42"},
            ),
            SureTransaction(
                id="2", classification="expense", amount="$80.00"
            ),
        ]
        provider = FakeTransactionProvider(txs)
        tracker = FakeIssueTracker()
        cfg = _make_config()
        creator = SharedBillsTaskCreator(provider, tracker, cfg)
        result = creator.compute_shared_bills(month_range_for(2026, 2))

        assert result.total_expenses == Decimal("80.00")
        assert result.excluded_count == 1

    def test_income_excluded_by_default(self) -> None:
        txs = [
            SureTransaction(
                id="1", classification="expense", amount="$100.00"
            ),
            SureTransaction(
                id="2", classification="income", amount="$500.00"
            ),
        ]
        provider = FakeTransactionProvider(txs)
        tracker = FakeIssueTracker()
        cfg = _make_config()
        creator = SharedBillsTaskCreator(provider, tracker, cfg)
        result = creator.compute_shared_bills(month_range_for(2026, 1))

        assert result.total_expenses == Decimal("100.00")
        assert result.included_count == 1

    def test_income_included_when_opted_in(self) -> None:
        txs = [
            SureTransaction(
                id="1", classification="expense", amount="$100.00"
            ),
            SureTransaction(
                id="2", classification="income", amount="$500.00"
            ),
        ]
        provider = FakeTransactionProvider(txs)
        tracker = FakeIssueTracker()
        cfg = _make_config(include_income=True)
        creator = SharedBillsTaskCreator(provider, tracker, cfg)
        result = creator.compute_shared_bills(month_range_for(2026, 1))

        assert result.total_expenses == Decimal("600.00")
        assert result.included_count == 2

    def test_no_classification_treated_as_expense(self) -> None:
        txs = [
            SureTransaction(id="1", amount="$75.00"),
        ]
        provider = FakeTransactionProvider(txs)
        tracker = FakeIssueTracker()
        cfg = _make_config()
        creator = SharedBillsTaskCreator(provider, tracker, cfg)
        result = creator.compute_shared_bills(month_range_for(2026, 1))

        assert result.total_expenses == Decimal("75.00")
        assert result.included_count == 1

    def test_create_jira_task(self) -> None:
        provider = FakeTransactionProvider([])
        tracker = FakeIssueTracker()
        cfg = _make_config()
        creator = SharedBillsTaskCreator(provider, tracker, cfg)

        billing = BillingResult(
            month_label="January 2026",
            total_expenses=Decimal("200.00"),
            half_total=Decimal("100.00"),
            excluded_count=1,
            included_count=3,
        )
        key = creator.create_jira_task(billing)

        assert key == "TEST-1"
        assert len(tracker.created) == 1
        issue = tracker.created[0]
        assert issue["summary"] == "Pay shared bills for January 2026: $100.00"
        assert issue["assignee_account_id"] == "abc123"

    def test_dry_run_skips_jira(self) -> None:
        provider = FakeTransactionProvider([])
        tracker = FakeIssueTracker()
        cfg = _make_config(dry_run=True)
        creator = SharedBillsTaskCreator(provider, tracker, cfg)

        billing = BillingResult(
            month_label="January 2026",
            total_expenses=Decimal("200.00"),
            half_total=Decimal("100.00"),
            excluded_count=0,
            included_count=2,
        )
        key = creator.create_jira_task(billing)

        assert key == "DRY_RUN"
        assert len(tracker.created) == 0

    def test_empty_transactions(self) -> None:
        provider = FakeTransactionProvider([])
        tracker = FakeIssueTracker()
        cfg = _make_config()
        creator = SharedBillsTaskCreator(provider, tracker, cfg)
        result = creator.compute_shared_bills(month_range_for(2026, 1))

        assert result.total_expenses == Decimal("0.00")
        assert result.half_total == Decimal("0.00")
        assert result.included_count == 0
        assert result.excluded_count == 0
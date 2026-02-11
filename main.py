#!/usr/bin/env python3
"""Create a monthly Jira task for shared bills based on Sure transactions.

Flow:
    1. Fetch transactions from Sure for a given month.
    2. Exclude transactions where category name == "Personal Expenses".
    3. Sum expenses and compute half.
    4. Create a Jira Task assigned to Alyssa.

Auth assumptions:
    - Sure: API key / token passed as a bearer token (configurable header).
    - Jira Cloud: Basic auth using email + API token.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo
import requests
from pydantic import BaseModel, Field, HttpUrl, SecretStr, ValidationError
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Application configuration loaded from environment variables."""

    # Sure
    sure_base_url: HttpUrl = Field(
        ..., description="Base URL for Sure, e.g. https://sure.example.com"
    )
    sure_api_token: SecretStr = Field(..., description="Sure API token/key")
    sure_auth_header: str = Field(
        default="Authorization",
        description='Header name for Sure token, default "Authorization"',
    )
    sure_auth_prefix: str = Field(
        default="Bearer",
        description='Prefix for Sure auth header value, default "Bearer"',
    )

    # Jira
    jira_base_url: HttpUrl = Field(
        ..., description="Jira base URL, e.g. https://your-org.atlassian.net"
    )
    jira_email: str = Field(..., description="Jira user email for API access")
    jira_api_token: SecretStr = Field(..., description="Jira API token")
    jira_project_key: str = Field(
        ..., description="Jira project key, e.g. HLAB"
    )
    jira_issue_type: str = Field(
        default="Task", description='Issue type name, default "Task"'
    )
    jira_assignee_account_id: str = Field(
        ..., description="Jira accountId for Alyssa"
    )

    # Behaviour
    timezone: str = Field(default="America/New_York", description="IANA timezone name")
    excluded_category_name: str = Field(default="Personal Expenses")
    include_income: bool = Field(
        default=False,
        description="If true, include income transactions in total.",
    )
    dry_run: bool = Field(
        default=False, description="If true, do not create Jira issue."
    )
    currency_symbol: str = Field(
        default="$", description="Currency symbol used in the Jira title."
    )


def load_config_from_env(*, dry_run: bool = False) -> AppConfig:
    """Load configuration from environment variables."""
    data: dict[str, Any] = {
        "sure_base_url": os.environ.get("SURE_BASE_URL", ""),
        "sure_api_token": os.environ.get("SURE_API_TOKEN", ""),
        "sure_auth_header": os.environ.get("SURE_AUTH_HEADER", "Authorization"),
        "sure_auth_prefix": os.environ.get("SURE_AUTH_PREFIX", "Bearer"),
        "jira_base_url": os.environ.get("JIRA_BASE_URL", ""),
        "jira_email": os.environ.get("JIRA_EMAIL", ""),
        "jira_api_token": os.environ.get("JIRA_API_TOKEN", ""),
        "jira_project_key": os.environ.get("JIRA_PROJECT_KEY", ""),
        "jira_issue_type": os.environ.get("JIRA_ISSUE_TYPE", "Task"),
        "jira_assignee_account_id": os.environ.get("JIRA_ASSIGNEE_ACCOUNT_ID", ""),
        "timezone": os.environ.get("APP_TIMEZONE", "America/New_York"),
        "excluded_category_name": os.environ.get(
            "EXCLUDED_CATEGORY_NAME", "Personal Expenses"
        ),
        "include_income": os.environ.get("INCLUDE_INCOME", "false").strip().lower()
        in {"1", "true", "yes", "y"},
        "dry_run": dry_run
        or (
            os.environ.get("DRY_RUN", "false").strip().lower()
            in {"1", "true", "yes", "y"}
        ),
        "currency_symbol": os.environ.get("CURRENCY_SYMBOL", "$"),
    }
    return AppConfig(**data)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonthRange:
    """Inclusive date range for a calendar month."""

    start_date: dt.date
    end_date_inclusive: dt.date

    @property
    def end_date_exclusive(self) -> dt.date:
        return self.end_date_inclusive + dt.timedelta(days=1)

    @property
    def label(self) -> str:
        return self.start_date.strftime("%B %Y")

    @property
    def year_month(self) -> str:
        return self.start_date.strftime("%Y-%m")


class SureTransaction(BaseModel):
    """A single transaction returned by the Sure API."""

    id: str
    name: str | None = None
    classification: str | None = None
    amount: str | None = None
    category: dict[str, Any] | None = None
    category_name: str | None = None
    category_id: str | None = None


class SureTransactionsResponse(BaseModel):
    """Wrapper for paginated Sure transaction responses."""

    transactions: list[SureTransaction] = Field(default_factory=list)
    next_page: int | None = None
    has_more: bool | None = None


@dataclass(frozen=True)
class BillingResult:
    """Result of computing shared bills for a month."""

    month_label: str
    total_expenses: Decimal
    half_total: Decimal
    excluded_count: int
    included_count: int


# ---------------------------------------------------------------------------
# Money parsing
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"[^0-9,.\-]+")


def parse_localized_money_to_decimal(value: str) -> Decimal:
    """Parse a localized currency string to a `Decimal`.

    Handles formats like ``€1.234,56``, ``$1,234.56``, ``-€2.00``.
    """
    cleaned = _AMOUNT_RE.sub("", value.strip().replace("\u00a0", " "))

    if cleaned in {"", "-", ".", ","}:
        raise ValueError(f"Unparseable money string: {value!r}")

    has_comma = "," in cleaned
    has_dot = "." in cleaned

    if has_comma and has_dot:
        last_comma = cleaned.rfind(",")
        last_dot = cleaned.rfind(".")
        dec_sep = "," if last_comma > last_dot else "."
    elif has_comma:
        dec_sep = ","
    else:
        dec_sep = "."

    if dec_sep == ",":
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")

    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(
            f"Unparseable money string: {value!r} -> {cleaned!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def month_range_for(year: int, month: int) -> MonthRange:
    """Compute the inclusive date range for *year*/*month*."""
    start = dt.date(year, month, 1)
    next_month = dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
    return MonthRange(start_date=start, end_date_inclusive=next_month - dt.timedelta(days=1))


def previous_month_in_tz(tz: ZoneInfo) -> tuple[int, int]:
    """Return ``(year, month)`` for the previous calendar month in *tz*."""
    today = dt.datetime.now(tz=tz).date()
    first_of_this = dt.date(today.year, today.month, 1)
    last_of_prev = first_of_this - dt.timedelta(days=1)
    return last_of_prev.year, last_of_prev.month


# ---------------------------------------------------------------------------
# Abstract client interfaces
# ---------------------------------------------------------------------------


class TransactionProvider(ABC):
    """Interface for fetching financial transactions."""

    @abstractmethod
    def list_transactions(
        self,
        start_date: dt.date,
        end_date_exclusive: dt.date,
    ) -> list[SureTransaction]:
        ...


class IssueTracker(ABC):
    """Interface for creating issue-tracker tasks."""

    @abstractmethod
    def create_issue(
        self,
        *,
        project_key: str,
        issue_type: str,
        summary: str,
        assignee_account_id: str,
        description: str | None = None,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Concrete clients
# ---------------------------------------------------------------------------


class SureClient(TransactionProvider):
    """Client for the Sure budgeting API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        auth_header: str,
        auth_prefix: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        header_value = f"{auth_prefix} {token}".strip()
        self._session.headers.update(
            {"Accept": "application/json", auth_header: header_value}
        )

    def list_transactions(
        self,
        start_date: dt.date,
        end_date_exclusive: dt.date,
        *,
        page_size: int = 200,
        max_pages: int = 200,
    ) -> list[SureTransaction]:
        """Fetch transactions for ``[start_date, end_date_exclusive)``."""
        url = f"{self._base_url}/api/v1/transactions"
        transactions: list[SureTransaction] = []

        for page in tqdm(
            range(1, max_pages + 1),
            desc="Fetching Sure transactions",
            unit="page",
        ):
            params: dict[str, str | int] = {
                "start_date": start_date.isoformat(),
                "end_date": (end_date_exclusive - dt.timedelta(days=1)).isoformat(),
                "page": page,
                "per_page": page_size,
            }
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 401:
                raise RuntimeError(
                    "Sure auth failed (401). Check token/header/prefix."
                )
            resp.raise_for_status()
            payload = resp.json()

            try:
                parsed = SureTransactionsResponse(**payload)
            except ValidationError:
                if isinstance(payload, list):
                    parsed = SureTransactionsResponse(
                        transactions=[SureTransaction(**t) for t in payload] # pyright: ignore
                    )
                    # TODO: Temporarily ignore type issue. In the future, validate/enforce payload type
                else:
                    raise

            if not parsed.transactions:
                break
            transactions.extend(parsed.transactions)

            if parsed.has_more is False:
                break
            if parsed.next_page is None and len(parsed.transactions) < page_size:
                break

        return transactions


class JiraClient(IssueTracker):
    """Client for Jira Cloud REST API v3."""

    def __init__(self, base_url: str, email: str, api_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.auth = (email, api_token)
        self._session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )

    def create_issue(
        self,
        *,
        project_key: str,
        issue_type: str,
        summary: str,
        assignee_account_id: str,
        description: str | None = None,
    ) -> str:
        """Create an issue and return its key."""
        url = f"{self._base_url}/rest/api/3/issue"
        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "issuetype": {"name": issue_type},
            "summary": summary,
            "assignee": {"accountId": assignee_account_id},
        }
        if description:
            # Jira Cloud v3 requires Atlassian Document Format (ADF).
            fields["description"] = self._plain_text_to_adf(description)

        resp = self._session.post(
            url, data=json.dumps({"fields": fields}), timeout=30
        )
        if resp.status_code == 401:
            raise RuntimeError("Jira auth failed (401). Check email/token.")
        resp.raise_for_status()
        data = resp.json()
        key = data.get("key")
        if not key:
            raise RuntimeError(
                f"Jira create issue succeeded but no key returned: {data}"
            )
        return str(key)

    @staticmethod
    def _plain_text_to_adf(text: str) -> dict[str, Any]:
        """Convert a plain-text string to Atlassian Document Format."""
        paragraphs: list[dict[str, Any]] = []
        for line in text.split("\n"):
            if line:
                paragraphs.append(
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": line}],
                    }
                )
            else:
                paragraphs.append({"type": "paragraph", "content": []})
        return {"type": "doc", "version": 1, "content": paragraphs}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SharedBillsTaskCreator:
    """Computes shared bills from Sure and creates a Jira task."""

    def __init__(
        self,
        transaction_provider: TransactionProvider,
        issue_tracker: IssueTracker,
        config: AppConfig,
    ) -> None:
        self._transactions = transaction_provider
        self._issues = issue_tracker
        self._config = config

    def _resolve_category_name(self, tx: SureTransaction) -> str | None:
        """Extract the category name from a transaction."""
        if tx.category_name:
            return tx.category_name
        if tx.category:
            return tx.category.get("name")
        return None

    def compute_shared_bills(self, month: MonthRange) -> BillingResult:
        """Sum qualifying transactions and compute the half-split amount."""
        txs = self._transactions.list_transactions(
            month.start_date, month.end_date_exclusive
        )

        excluded = 0
        included = 0
        total = Decimal("0")

        for tx in tqdm(txs, desc="Summing transactions", unit="tx"):
            category_name = self._resolve_category_name(tx)
            if category_name == self._config.excluded_category_name:
                excluded += 1
                continue

            classification = (tx.classification or "").strip().lower()

            # Skip income unless explicitly opted-in.
            if classification == "income" and not self._config.include_income:
                continue

            if not tx.amount:
                continue

            amount_abs = parse_localized_money_to_decimal(tx.amount).copy_abs()

            # Treat as expense if classification says so OR if classification
            # is absent/unknown (most transactions are expenses for bills).
            if classification in {"expense", ""}:
                total += amount_abs
                included += 1
            elif self._config.include_income and classification == "income":
                total += amount_abs
                included += 1

        half = (total / Decimal("2")).quantize(Decimal("0.01"))
        return BillingResult(
            month_label=month.label,
            total_expenses=total.quantize(Decimal("0.01")),
            half_total=half,
            excluded_count=excluded,
            included_count=included,
        )

    def create_jira_task(self, result: BillingResult) -> str:
        """Create (or dry-run) a Jira task for the billing result."""
        sym = self._config.currency_symbol
        summary = (
            f"Pay shared bills for {result.month_label}: {sym}{result.half_total}"
        )
        description = (
            f"Auto-created from Sure.\n\n"
            f"Month: {result.month_label}\n"
            f"Total (expenses): {sym}{result.total_expenses}\n"
            f"Half: {sym}{result.half_total}\n"
            f"Excluded category: {self._config.excluded_category_name}\n"
            f"Included transactions: {result.included_count}\n"
            f"Excluded transactions: {result.excluded_count}\n"
        )

        if self._config.dry_run:
            LOGGER.info(
                "DRY_RUN enabled; would create Jira issue with summary: %s",
                summary,
            )
            return "DRY_RUN"

        return self._issues.create_issue(
            project_key=self._config.jira_project_key,
            issue_type=self._config.jira_issue_type,
            summary=summary,
            assignee_account_id=self._config.jira_assignee_account_id,
            description=description,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class ArgsType(argparse.Namespace):
    month: str | None
    dry_run: bool
    include_income: bool
    log_level: str

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Create monthly shared-bills Jira task from Sure."
    )
    parser.add_argument(
        "--month",
        help="Target month in YYYY-MM format (default: previous month).",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute totals but do not create Jira issue.",
    )
    parser.add_argument(
        "--include-income",
        action="store_true",
        help="Include income transactions in the total.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv, namespace=ArgsType())

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config_from_env(dry_run=args.dry_run)
    except ValidationError as exc:
        LOGGER.error("Config error. Set required env vars. Details:\n%s", exc)
        return 2

    if args.include_income:
        cfg = cfg.model_copy(update={"include_income": True})

    try:
        tz = ZoneInfo(cfg.timezone)
    except Exception as exc:
        LOGGER.error("Invalid timezone %r: %s", cfg.timezone, exc)
        return 2

    if args.month:
        try:
            year_s, month_s = args.month.split("-", 1)
            year, month = int(year_s), int(month_s)
            if not 1 <= month <= 12:
                raise ValueError("month must be 1..12")
        except Exception as exc:
            LOGGER.error(
                "Invalid --month %r (expected YYYY-MM): %s", args.month, exc
            )
            return 2
    else:
        year, month = previous_month_in_tz(tz)

    mrange = month_range_for(year, month)

    sure = SureClient(
        base_url=str(cfg.sure_base_url),
        token=cfg.sure_api_token.get_secret_value(),
        auth_header=cfg.sure_auth_header,
        auth_prefix=cfg.sure_auth_prefix,
    )
    jira = JiraClient(
        base_url=str(cfg.jira_base_url),
        email=cfg.jira_email,
        api_token=cfg.jira_api_token.get_secret_value(),
    )

    runner = SharedBillsTaskCreator(
        transaction_provider=sure, issue_tracker=jira, config=cfg
    )
    result = runner.compute_shared_bills(mrange)

    sym = cfg.currency_symbol
    LOGGER.info(
        "Month=%s total=%s%s half=%s%s excluded=%d included=%d",
        result.month_label,
        sym,
        result.total_expenses,
        sym,
        result.half_total,
        result.excluded_count,
        result.included_count,
    )

    issue_key = runner.create_jira_task(result)
    LOGGER.info("Jira issue: %s", issue_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Create a monthly Jira task for shared bills based on Sure transactions.

Flow:
    1. Fetch transactions from Sure for a given month.
    2. Exclude transactions within given categories.
    3. Sum expenses and compute half.
    4. Create a Jira Task assigned to a specific user
"""
from __future__ import annotations

import argparse
import datetime as dt
import enum
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
    jira_encoded_credentials: SecretStr = Field(..., description="Base64-encoded Jira credentials (email:token)")
    jira_project_key: str = Field(
        ..., description="Jira project key, e.g. HLAB"
    )
    jira_issue_type: str = Field(
        default="Task", description='Issue type name, default "Task"'
    )
    jira_assignee_account_id: str = Field(
        ..., description="Jira accountId for the assignee. Find it here: https://yourdomain.atlassian.net/rest/api/3/user/assignable/search?project=PROJECTKEY&query=username"
    )

    # Behaviour
    timezone: str = Field(default="America/New_York", description="IANA timezone name")
    excluded_category_names: list[str] = Field(default_factory=lambda: ["Personal Expenses"], description="List of category names to exclude from bills")
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
    yes_options = {"1", "true", "yes", "y"}
    include_income_env = os.environ.get("INCLUDE_INCOME", "false").strip().lower() in yes_options
    dry_run_env = dry_run or (os.environ.get("DRY_RUN", "false").strip().lower() in yes_options)
    excluded_categories_env = os.environ.get("EXCLUDED_CATEGORY_NAMES", "Personal Expenses")
    
    data: dict[str, Any] = {
        "sure_base_url": os.environ.get("SURE_BASE_URL", ""),
        "sure_api_token": os.environ.get("SURE_API_TOKEN", ""),
        "sure_auth_header": os.environ.get("SURE_AUTH_HEADER", "Authorization"),
        "sure_auth_prefix": os.environ.get("SURE_AUTH_PREFIX", "Bearer"),
        "jira_base_url": os.environ.get("JIRA_BASE_URL", ""),
        "jira_email": os.environ.get("JIRA_EMAIL", ""),
        "jira_api_token": os.environ.get("JIRA_API_TOKEN", ""),
        "jira_encoded_credentials": os.environ.get("JIRA_ENCODED_CREDENTIALS", ""),
        "jira_project_key": os.environ.get("JIRA_PROJECT_KEY", ""),
        "jira_issue_type": os.environ.get("JIRA_ISSUE_TYPE", "Task"),
        "jira_assignee_account_id": os.environ.get("JIRA_ASSIGNEE_ACCOUNT_ID", ""),
        "timezone": os.environ.get("APP_TIMEZONE", "America/New_York"),
        "excluded_category_names": [x.strip() for x in excluded_categories_env.split(",")],
        "include_income": include_income_env,
        "dry_run": dry_run_env,
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

    def __str__(self) -> str:
        cat = f" in {self.category['name']}" if self.category else ""
        return f"Transaction ({self.id!r}): {self.amount!r} to {self.name[:20] if self.name else 'unknown'!r}{cat}"

    @staticmethod
    def resolve_category_name(transaction: "SureTransaction") -> str | None:
        """
        Extract the category name from a transaction.

        Args:
            transaction: The transaction to extract the category name from.

        Returns:
            The category name if available, otherwise None.
        """
        if transaction.category_name:
            return transaction.category_name
        if transaction.category:
            return transaction.category.get("name")
        return None

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
    """
    Parse a localized currency string to a `Decimal`.

    Handles formats like ``$1.234,56``, ``$1,234.56``, ``-$2.00``.

    Args:
        value: The localized money string to parse.

    Returns:
        A `Decimal` representing the parsed amount.
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
        result = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(
            f"Unparseable money string: {value!r} -> {cleaned!r}"
        ) from exc

    return result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def month_range_for(year: int, month: int) -> MonthRange:
    """
    Compute the inclusive date range for *year*/*month*.

    Args:
        year: The year as a four-digit integer, e.g. 2024.
        month: The month as an integer from 1 (January) to 12 (December).

    Returns:
        A `MonthRange` object representing the inclusive date range for the specified month.
    """
    start = dt.date(year, month, 1)
    next_month = dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
    return MonthRange(start_date=start, end_date_inclusive=next_month - dt.timedelta(days=1))


def previous_month_in_tz(tz: ZoneInfo) -> tuple[int, int]:
    """
    Return ``(year, month)`` for the previous calendar month in *tz*.

    Args:
        tz: The timezone to consider when determining the current date.

    Returns:
        A tuple of (year, month) for the previous calendar month.
    """
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
        page_size: int = 25,
        max_pages: int = 200,
    ) -> list[SureTransaction]:
        """
        Fetch transactions for ``[start_date, end_date_exclusive)``.

        Args:
            start_date: Inclusive start date.
            end_date_exclusive: Exclusive end date.
            page_size: Number of transactions to request per page (Sure may have a max of 25).
            max_pages: Maximum number of pages to fetch to prevent infinite loops.

        Returns:
            A list of SureTransaction objects representing the fetched transactions.

        Raises:
            RuntimeError: If authentication fails or if the API returns an error status.
            ValidationError: If the API response cannot be parsed into the expected format.
        """
        url = f"{self._base_url}/api/v1/transactions"
        transactions: list[SureTransaction] = []

        # 2025-02-11: requesting page_size 200 causes unexpected behavior and only 25 items are returned.
        if page_size > 25:
            LOGGER.warning(
                "Sure API may have a max page size of 25. Requested %d.", page_size
            )

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
            response = self._session.get(url, params=params, timeout=30)
            if response.status_code == 401:
                raise RuntimeError(
                    "Sure auth failed (401). Check token/header/prefix."
                )
            response.raise_for_status()
            payload = response.json()

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
                LOGGER.debug("No transactions found on page %d, stopping pagination.", page)
                break

            transactions.extend(parsed.transactions)

            if parsed.has_more is False:
                LOGGER.debug("No more pages after page %d, stopping pagination.", page)
                break
            if parsed.next_page is None and len(parsed.transactions) < page_size:
                LOGGER.debug("Fewer transactions (%d) than page size (%d) on page %d, assuming last page.", len(parsed.transactions), page_size, page)
                break

        return transactions

class JiraClient(IssueTracker):
    """Client for Jira Cloud REST API v3."""

    def __init__(self, base_url: str, encoded_credentials: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        header_value = f"Basic {encoded_credentials}"
        self._session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json", "Authorization": header_value}
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
        """
        Create an issue and return its key.

        Args:
            project_key: The key of the Jira project, e.g. "HLAB".
            issue_type: The name of the issue type, e.g. "Task".
            summary: The issue summary
            assignee_account_id: The account ID of the assignee (not email; find it here: https://your-domain.atlassian.net/rest/api/3/user/assignable/search?project=PROJECTKEY&query=username).
            description: The issue description (optional)
            
        Returns:
            The key of the created issue, e.g. "HLAB-123".

        Raises:
            RuntimeError: If authentication fails or if the API returns an error status.
            ValueError: If the API response does not contain the expected issue key.
        """
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

        payload = {"fields": fields}

        response = self._session.post(url, json=payload, timeout=30)

        # Always log error bodies from Jira; they are highly informative.
        if response.status_code >= 400:
            LOGGER.error("Jira create issue failed. Status=%s, Url=%s, Fields=%s, Body=%s", response.status_code, url, fields, response.text)

        if response.status_code == 401:
            raise RuntimeError("Jira auth failed (401). Check email/token.")
        response.raise_for_status()

        data = response.json()
        key = data.get("key")
        if not key:
            raise RuntimeError(f"Jira create issue succeeded but no key returned: {data}")
        return str(key)


    @staticmethod
    def _plain_text_to_adf(text: str) -> dict[str, Any]:
        """
        Convert a plain-text string to Atlassian Document Format.

        Args:
            text: The plain text to convert.

        Returns:
            A dict representing the text in Atlassian Document Format.
        """
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

    def compute_shared_bills(self, month: MonthRange) -> BillingResult:
        """
        Sum qualifying transactions and compute the half-split amount.

        Args:
            month: The MonthRange for which to compute the bills.

        Returns:
            A BillingResult containing the total expenses, half total, and counts of included/excluded transactions.
        """
        transactions = self._transactions.list_transactions(
            month.start_date, month.end_date_exclusive
        )

        excluded = 0
        included = 0
        total = Decimal("0")

        for transaction in tqdm(transactions, desc="Summing transactions", unit="tx"):
            category_name = SureTransaction.resolve_category_name(transaction)
            if category_name in self._config.excluded_category_names:
                excluded += 1
                continue

            classification = (transaction.classification or "").strip().lower()

            # Skip income unless explicitly opted-in.
            if classification == "income" and not self._config.include_income:
                continue

            if not transaction.amount:
                continue

            amount_abs = parse_localized_money_to_decimal(transaction.amount).copy_abs()

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

    def list_transactions(self, month: MonthRange) -> list[SureTransaction]:
        """
        List transactions for a month, without any filtering.

        Args:
            month: The MonthRange for which to list transactions.

        Returns:
            A list of SureTransaction objects for the specified month, excluding only those that match the excluded categories.
        """
        transactions = self._transactions.list_transactions(
            month.start_date, month.end_date_exclusive
        )

        results : list[SureTransaction] = []

        for transaction in tqdm(transactions, desc="Listing transactions", unit="tx"):
            category_name = SureTransaction.resolve_category_name(transaction)
            if category_name in self._config.excluded_category_names:
                continue

            classification = (transaction.classification or "").strip().lower()

            # Skip income unless explicitly opted-in.
            if classification == "income" and not self._config.include_income:
                continue

            if not transaction.amount:
                continue

            # Treat as expense if classification says so OR if classification
            # is absent/unknown (most transactions are expenses for bills).
            if classification in {"expense", ""}:
                results.append(transaction)
            elif self._config.include_income and classification == "income":
                results.append(transaction)

        return results
                

    def create_jira_task(self, result: BillingResult) -> str:
        """
        Create (or dry-run) a Jira task for the billing result.

        Args:
            result: The BillingResult containing the computed totals and counts.

        Returns:
            The key of the created Jira issue, or "DRY_RUN" if in dry-run mode.
        """
        currency_symbol = self._config.currency_symbol
        summary = (
            f"Pay shared bills for {result.month_label}: {currency_symbol}{result.half_total}"
        )
        description = (
            f"Calculated from our budgeting app for {result.month_label}.\n\n"
            f"Total (expenses): {currency_symbol}{result.total_expenses}\n"
            f"Half: {currency_symbol}{result.half_total}\n"
            f"Excluded categories: {', '.join(self._config.excluded_category_names)}\n"
            f"Transactions: {result.included_count} included / {result.excluded_count} excluded\n"
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
    action: str
    month: str | None
    dry_run: bool
    include_income: bool
    log_level: str

class Actions(enum.Enum):
    CREATE_ISSUE = "notify"
    LIST_TRANSACTIONS = "list"

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns:
        An argparse.ArgumentParser instance configured with the expected command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create monthly shared-bills Jira task from Sure."
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=[a.value for a in Actions],
        default=Actions.CREATE_ISSUE.value,
        help="Action to perform.",
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
                raise ValueError("month must be between 1 and 12")
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
        encoded_credentials=cfg.jira_encoded_credentials.get_secret_value(),
    )
    runner = SharedBillsTaskCreator(
        transaction_provider=sure, issue_tracker=jira, config=cfg
    )
    if args.action == Actions.LIST_TRANSACTIONS.value:
        transactions = runner.list_transactions(mrange)
        for transaction in transactions:
            print(transaction)
        return 0
    
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
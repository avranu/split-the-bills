# split-the-bills

Automate month-end shared bill reconciliation by:

1. Fetching transactions from **Sure**,
2. Filtering and summing shared expenses,
3. Calculating the 50/50 split,
4. Creating a **Jira** issue with the amount owed.

This project is a small Python CLI designed to run manually or on a schedule (for example via cron, GitHub Actions, or Kestra).

---

## Table of contents

- [split-the-bills](#split-the-bills)
  - [Table of contents](#table-of-contents)
  - [Features](#features)
  - [How it works](#how-it-works)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [Option A (recommended): `uv`](#option-a-recommended-uv)
    - [Option B: standard `pip`](#option-b-standard-pip)
  - [Configuration](#configuration)
    - [Required variables](#required-variables)
    - [Optional variables](#optional-variables)
  - [Usage](#usage)
    - [CLI options](#cli-options)
  - [Examples](#examples)
    - [1) Preview current behavior without creating Jira issue](#1-preview-current-behavior-without-creating-jira-issue)
    - [2) Process a specific month](#2-process-a-specific-month)
    - [3) Inspect included transactions](#3-inspect-included-transactions)
    - [4) Include income in calculation](#4-include-income-in-calculation)
  - [Exit codes](#exit-codes)
  - [Development](#development)
  - [Testing](#testing)
  - [Troubleshooting](#troubleshooting)
- [Known Issues](#known-issues)
  - [TODOs](#todos)

---

## Features

- Pulls monthly transactions from Sure (paginated API).
- Supports custom auth header/prefix for Sure deployments.
- Excludes categories such as `Personal Expenses`.
- Optionally includes income transactions.
- Supports a configurable tag for expenses fully paid by the other partner, which reduces the amount owed.
- Parses localized currency formats into precise `Decimal` values.
- Creates Jira issues with computed totals and contextual details.
- Supports dry-run mode to safely validate behavior.
- Includes unit tests for parser, date logic, filtering, and Jira payload behavior.

---

## How it works

For a target month (explicit via CLI or defaulting to the **previous month** in your configured timezone), the script:

1. Calls Sure transactions endpoint for that month.
2. Keeps transactions classified as `expense` (and optionally `income`).
3. Excludes any transaction whose category matches configured excluded categories.
4. Sums included transaction amounts.
5. Computes half and then subtracts any expenses tagged as fully paid by the other partner.
6. Creates a Jira issue with a summary like:
   - `Pay shared bills for January 2026: $123.45`

---

## Requirements

- Python **3.10+**
- Sure API access
- Jira Cloud/API access with project + assignee permissions

Python dependencies are defined in `pyproject.toml`:
- `pydantic`
- `requests`
- `tqdm`

---

## Installation

### Option A (recommended): `uv`

```bash
git clone <your-fork-or-repo-url>
cd split-the-bills
uv venv .venv
source .venv/bin/activate
uv sync
```

### Option B: standard `pip`

```bash
git clone <your-fork-or-repo-url>
cd split-the-bills
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

Create your runtime env file:

```bash
cp .env-sample .env
```

Then set environment variables (or export them in your shell / secret manager).

### Required variables

| Variable                   | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `SURE_BASE_URL`            | Sure base URL (e.g. `https://sure.yourdomain.com`)    |
| `SURE_API_TOKEN`           | Sure API token                                        |
| `JIRA_BASE_URL`            | Jira base URL (e.g. `https://your-org.atlassian.net`) |
| `JIRA_EMAIL`               | Jira email (used for API identity)                    |
| `JIRA_API_TOKEN`           | Jira API token                                        |
| `JIRA_ENCODED_CREDENTIALS` | Base64 of `email:api_token`                           |
| `JIRA_PROJECT_KEY`         | Jira project key (e.g. `BUDG`)                        |
| `JIRA_ASSIGNEE_ACCOUNT_ID` | Jira assignee account ID                              |

### Optional variables

| Variable                  | Default             | Description                                       |
| ------------------------- | ------------------- | ------------------------------------------------- |
| `SURE_AUTH_HEADER`        | `Authorization`     | Sure auth header name                             |
| `SURE_AUTH_PREFIX`        | `Bearer`            | Sure auth token prefix                            |
| `JIRA_ISSUE_TYPE`         | `Task`              | Jira issue type                                   |
| `APP_TIMEZONE`            | `America/New_York`  | IANA timezone name                                |
| `EXCLUDED_CATEGORY_NAMES` | `Personal Expenses` | Comma-separated category names to exclude         |
| `INCLUDE_INCOME`          | `false`             | Include income transactions in the total          |
| `DRY_RUN`                 | `false`             | Compute values but skip Jira issue creation       |
| `CURRENCY_SYMBOL`         | `$`                 | Symbol used in Jira summary/description           |
| `PAID_BY_PARTNER_TAG`     | `Paid by partner`   | Tag name marks expenses fully paid by the partner |

---

## Usage

```bash
python main.py [--action {notify,list}] [--month YYYY-MM] [--dry-run] [--include-income] [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

### CLI options

- `--action`
  - `notify` (default): compute total and create Jira issue (unless dry-run)
  - `list`: print included transactions only
- `--month YYYY-MM`
  - Target month. If omitted, previous month in `APP_TIMEZONE` is used.
- `--dry-run`
  - Prevents Jira issue creation.
- `--include-income`
  - Includes `income` transactions in computations.
- `--log-level`
  - Logging verbosity (`INFO` default).

---

## Examples

### 1) Preview current behavior without creating Jira issue

```bash
set -a
source .env
set +a
python main.py --dry-run --log-level INFO
```

### 2) Process a specific month

```bash
python main.py --month 2026-01 --dry-run
```

### 3) Inspect included transactions

```bash
python main.py --action list --month 2026-01
```

### 4) Include income in calculation

```bash
python main.py --month 2026-01 --include-income --dry-run
```

---

## Exit codes

- `0`: success
- `2`: configuration/validation/argument error (e.g. invalid env vars, invalid month, invalid timezone)

---

## Development

Install dev dependencies:

```bash
uv sync --group dev
```

or

```bash
pip install -e .
pip install pytest pylint
```

Run lint (if configured in your environment):

```bash
pylint main.py test_main.py
```

---

## Testing

Run the unit test suite:

```bash
pytest -q
```

Tests cover:
- money parsing edge cases,
- month/date calculations,
- Sure transaction filtering behavior,
- Jira ADF conversion helper,
- dry-run and issue creation flow.

---

## Troubleshooting

- **Validation error at startup**
  - Ensure required env vars are set and URLs are valid.
- **Invalid month format**
  - Use `--month YYYY-MM` (e.g. `2026-01`).
- **Timezone errors**
  - Use a valid IANA timezone (`America/New_York`, `Europe/Madrid`, etc.).
- **No Jira issue created**
  - Check whether `--dry-run` or `DRY_RUN=true` is enabled.
- **Unexpected totals**
  - Verify `EXCLUDED_CATEGORY_NAMES`, `INCLUDE_INCOME`, and transaction classifications from Sure.

---

# Known Issues
- Sure API may have a max page size of 25. Requesting more may cause unexpected behavior.
- parse_localized_money_to_decimal("£1,000") returns 1.000

## TODOs
- Pydantic validation
- Additional unit tests
- CI/CD pipeline (unit tests, ruff, mypy)
- Full readme
- Expand error handling and edge cases
- Reduce pylint suppressions
- Stronger config validation and friendlier startup diagnostics.
- Expanded test coverage around API error handling.
- Packaging/release workflow.
- Optional duplicate-issue detection in Jira.

# split-the-bills
A script to automate splitting monthly bills between two partners. Makes use of Sure and Jira, and can be automated with something like Kestra.

# Known Issues
- Sure API may have a max page size of 25. Requesting more may cause unexpected behavior.
- parse_localized_money_to_decimal("£1,000") returns 1.000

# TODOs
- Pydantic validation
- Additional unit tests
- CI/CD pipeline
- Full readme
- Expand error handling and edge cases
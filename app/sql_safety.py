# FILE: app/sql_safety.py
import re

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|commit|rollback)\b",
    re.IGNORECASE,
)

MULTI_STMT = re.compile(r";\s*\S+", re.DOTALL)  # anything after a semicolon


def ensure_safe_select(sql: str) -> str:
    """
    Allow only a single SELECT statement (optionally with WITH).
    Reject anything suspicious.
    """
    if not sql or not isinstance(sql, str):
        raise ValueError("SQL is empty")

    s = sql.strip()

    # Disallow multiple statements
    if ";" in s:
        # allow trailing semicolon only
        if MULTI_STMT.search(s):
            raise ValueError("Multiple SQL statements are not allowed")
        s = s.rstrip(";").strip()

    # Must start with SELECT or WITH
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous keywords anywhere
    if FORBIDDEN.search(s):
        raise ValueError("Forbidden SQL keyword detected")

    return s


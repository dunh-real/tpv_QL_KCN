
MAX_RETRIES: int = 3

FORBIDDEN_STATEMENT_TYPES: set[str] = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "CALL",
}


FORBIDDEN_KEYWORDS_FALLBACK: list[str] = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "MERGE", "GRANT", "REVOKE",
    "EXEC", "EXECUTE", "CALL", "LOAD_FILE", "INTO OUTFILE",
]


DEFAULT_ROW_LIMIT: int = 20

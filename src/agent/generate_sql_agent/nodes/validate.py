

import re
from src.agent.generate_sql_agent.state import GenerateSQLState
from src.agent.generate_sql_agent.constants import FORBIDDEN_STATEMENT_TYPES, FORBIDDEN_KEYWORDS_FALLBACK
from src.core.logger import get_logger

logger = get_logger(__name__)

# Thử import sqlglot
try:
    import sqlglot
    HAS_SQLGLOT = True
    logger.info("[validate_sql] Đã load thư viện sqlglot (bảo mật cao).")
except ImportError:
    HAS_SQLGLOT = False
    logger.warning("[validate_sql] Không tìm thấy sqlglot. Sử dụng Regex fallback (bảo mật thấp hơn). "
                   "Khuyên dùng: pip install sqlglot")


def _strip_sql_comments(sql: str) -> str:
    """Loại bỏ comment đa dòng (/*...*/) và một dòng (--...) để tránh bypass."""
    # Xóa block comments /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Xóa inline comments -- ...
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    return sql


def validate_with_regex(sql: str) -> tuple[bool, str | None, str | None]:
    """Kiểm tra fallback nếu không có sqlglot."""
    clean_sql = _strip_sql_comments(sql).upper()
    
    # Normalize khoảng trắng
    clean_sql = re.sub(r'\s+', ' ', clean_sql)

    for kw in FORBIDDEN_KEYWORDS_FALLBACK:
        # Check keyword độc lập (word boundary)
        pattern = r'\b' + kw.replace(" ", r'\s+') + r'\b'
        if re.search(pattern, clean_sql):
            return False, f"Lỗi bảo mật: Phát hiện từ khóa '{kw}'. AI chỉ được phép SELECT.", "FORBIDDEN"
            
    return True, sql, None


def validate_with_sqlglot(sql: str) -> tuple[bool, str | None, str | None]:
    """Sử dụng AST Parser để siêu kiểm tra."""
    try:
        # Parse SQL thành biểu thức AST
        expressions = sqlglot.parse(sql, read="postgres")
        
        if not expressions:
            return False, "Câu lệnh trống sau khi parse.", "EMPTY"

        # Duyệt qua các câu lệnh (thường chỉ nên có 1)
        for expr in expressions:
            if not expr:
                continue

            expr_type = type(expr).__name__.upper()

            # Chỉ cho phép SELECT đơn thuần hoặc các phép SET (UNION, INTERSECT, EXCEPT)
            # SETOPERATION là lớp cha trong AST của sqlglot cho các phép này
            is_allowed = (
                expr_type in ("SELECT", "SETOPERATION")
                or expr_type.startswith("UNION")
                or expr_type.startswith("INTERSECT")
                or expr_type.startswith("EXCEPT")
            )
            if not is_allowed:
                return False, f"Lỗi bảo mật: Statement type '{expr_type}' không được phép. AI chỉ dùng SELECT.", "FORBIDDEN"
            
            pass

        # Build lại SQL an toàn (normalization)
        safe_sql = sqlglot.transpile(sql, read="postgres", write="postgres")[0]
             
        return True, safe_sql.strip().rstrip(';') + ";", None

    except sqlglot.errors.ParseError as e:
        logger.error(f"[validate_sql] Lỗi cú pháp (sqlglot): {e}")
        return False, f"SQL Syntax Error: {str(e)}. Hãy kiểm tra lại cấu trúc câu lệnh.", "SYNTAX_ERROR"
    except Exception as e:
        logger.error(f"[validate_sql] Lỗi khi xử lý AST: {e}")
        return False, f"Unexpected Validation Error: {e}", "SYNTAX_ERROR"


def validate_sql_node(state: GenerateSQLState) -> dict:
    """
    Kiểm tra cú pháp, tính an toàn và tự động chèn LIMIT vào câu SQL.
    """
    raw_sql = state.get("sql_query", "").strip()
    
    if not raw_sql:
        logger.warning("[validate_sql] SQL trống.")
        return {
            "is_valid": False,
            "error": "Lỗi: Câu lệnh SQL bị trống.",
            "validation_error_type": "EMPTY",
        }
        
    logger.info(f"[validate_sql] Đang kiểm duyệt SQL...")

    if HAS_SQLGLOT:
        is_valid, result_or_error, err_type = validate_with_sqlglot(raw_sql)
    else:
        is_valid, result_or_error, err_type = validate_with_regex(raw_sql)

    if not is_valid:
        logger.warning(f"[validate_sql] Từ chối SQL. Lỗi ({err_type}): {result_or_error}")
        return {
            "is_valid": False,
            "error": result_or_error,
            "validation_error_type": err_type
        }
    
    logger.info("[validate_sql] SQL hợp lệ và chuẩn hóa thành công.")
    
    return {
        "is_valid": True,
        "sql_query": result_or_error,
        "error": None,
        "validation_error_type": None
    }
import streamlit as st
import os, json, re as _re
import snowflake.connector
import pandas as pd
import plotly.express as px
from groq import Groq
import base64, pathlib, time
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
def cfg(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

def _clean_account(raw: str) -> str:
    raw = raw.strip()
    raw = _re.sub(r'\.snowflakecomputing\.com.*$', '', raw, flags=_re.IGNORECASE)
    return raw

SNOWFLAKE_ACCOUNT   = _clean_account(cfg("SNOWFLAKE_ACCOUNT"))
SNOWFLAKE_USER      = cfg("SNOWFLAKE_USER").strip()
SNOWFLAKE_PASSWORD  = cfg("SNOWFLAKE_PASSWORD").strip()
SNOWFLAKE_WAREHOUSE = cfg("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH").strip()
SNOWFLAKE_ROLE      = cfg("SNOWFLAKE_ROLE", "").strip()
GROQ_API_KEY        = cfg("GROQ_API_KEY").strip()

GROQ_MODEL   = "llama-3.1-8b-instant"
COMPANY_NAME = "Techwish AI — Analytics"

DEFAULT_CHART_COLOR   = "#1565C0"
DEFAULT_TITLE_COLOR   = "#1565C0"
DEFAULT_BLUE_SEQUENCE = [
    "#1565C0", "#1976D2", "#1E88E5", "#42A5F5",
    "#90CAF9", "#BBDEFB", "#0D47A1", "#0288D1",
]

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput,
.stButton, .stSelectbox, .stExpander, .stChatMessage,
.stDataFrame, .stCaption, .stSpinner, input, textarea, button {
    font-family: 'Poppins', sans-serif !important;
}
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-top: 0 !important;
}
.sql-block {
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Poppins', monospace !important;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-break: break-all;
}
.logo-row { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.ai-badge {
    background: #1565C0; color: white;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700; font-size: 0.65rem;
    padding: 2px 7px; border-radius: 20px;
    letter-spacing: 0.05em; line-height: 1.4; vertical-align: middle;
}
.ai-welcome-img {
    position: center; top:80px; bottom:55px; left:0; right:0;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    pointer-events:auto; z-index:0; overflow:hidden; flex-shrink:0;
}
.ai-welcome-img img {
    width: clamp(200px,30vw,210px); height:auto;
    object-fit:contain; flex-shrink:0;
}
.ai-welcome-caption {
    margin-top:0.6rem; font-family:'Poppins',sans-serif;
    font-size:clamp(0.7rem,1.1vw,0.9rem); color:gray;
    text-align:center; flex-shrink:0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  DATABASE CONNECTION
# ─────────────────────────────────────────────────────────────────
def _build_conn_kwargs(database: str = None) -> dict:
    kw = dict(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
        session_parameters={"CLIENT_SESSION_KEEP_ALIVE": "TRUE"},
        network_timeout=300,
        login_timeout=60,
    )
    if database:
        kw["database"] = database
    if SNOWFLAKE_ROLE:
        kw["role"] = SNOWFLAKE_ROLE
    return kw

def _new_conn(database: str = None):
    return snowflake.connector.connect(**_build_conn_kwargs(database))

def _get_conn():
    conn = st.session_state.get("_sf_conn")
    if conn is not None:
        try:
            conn.cursor().execute("SELECT 1")
            return conn
        except Exception:
            pass
    conn = _new_conn()
    st.session_state["_sf_conn"] = conn
    return conn

def get_snowflake_conn(database: str = None):
    return _get_conn()

def run_query(sql: str, database: str) -> pd.DataFrame:
    def _execute(conn):
        cur = conn.cursor()
        cur.execute(f'USE DATABASE "{database}"')
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    conn = _get_conn()
    try:
        return _execute(conn)
    except Exception as e:
        err = str(e)
        if any(x in err for x in ("08001","390114","Authentication token","Connection","connection","session")):
            conn = _new_conn()
            st.session_state["_sf_conn"] = conn
            return _execute(conn)
        raise

def list_databases() -> list:
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        rows = cur.fetchall()
        system_dbs = {"SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"}
        dbs = [row[1] for row in rows if row[1] not in system_dbs]
        return sorted(dbs)
    except Exception as e:
        st.error(f"Could not list databases: {e}")
        return []

# ─────────────────────────────────────────────────────────────────
#  SCHEMA LOADER & WHITELIST
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_schema(database: str) -> str:
    schema_sql = f"""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA')
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """
    try:
        df = run_query(schema_sql, database)
        lines = []
        current_table = None
        for _, row in df.iterrows():
            full_name = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"
            if full_name != current_table:
                current_table = full_name
                lines.append(f"\nTable: {full_name}")
            nullable = "nullable" if row["IS_NULLABLE"] == "YES" else "not null"
            lines.append(f"  - {row['COLUMN_NAME']} ({row['DATA_TYPE']}, {nullable})")
        return "\n".join(lines)
    except Exception as e:
        return f"Schema load failed: {e}"

@st.cache_data(show_spinner=False)
def build_whitelist(database: str) -> dict:
    schema_sql = f"""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA')
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """
    try:
        df = run_query(schema_sql, database)
        wl = {}
        for _, row in df.iterrows():
            tbl = row["TABLE_NAME"]
            wl.setdefault(tbl, []).append(row["COLUMN_NAME"])
        return wl
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def build_full_schema_dict(database: str) -> dict:
    """Returns {TABLE_NAME: {COL_NAME: DATA_TYPE}} — all keys uppercase."""
    schema_sql = f"""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA')
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """
    try:
        df = run_query(schema_sql, database)
        schema_dict = {}
        for _, row in df.iterrows():
            tbl = row["TABLE_NAME"].upper()
            col = row["COLUMN_NAME"].upper()
            dtype = row["DATA_TYPE"].upper()
            schema_dict.setdefault(tbl, {})[col] = dtype
        return schema_dict
    except Exception:
        return {}

def whitelist_to_strict_block(wl: dict) -> str:
    """Numbered list of every table and column — used in LLM system prompt."""
    lines = []
    idx = 1
    for tbl, cols in wl.items():
        lines.append(f"\n[TABLE] {tbl}")
        for col in cols:
            lines.append(f"  {idx:04d}. {col}")
            idx += 1
    return "\n".join(lines)

def build_date_type_hints(schema_dict: dict) -> str:
    """
    Scans schema_dict for columns that are declared as DATE or TIMESTAMP.
    Returns a human-readable block telling the LLM exactly which columns
    are native date types (so it must NOT wrap them in TO_DATE / LPAD).
    """
    native_date_cols = []
    native_ts_cols   = []
    numeric_date_cols = []

    DATE_TYPES = {"DATE"}
    TS_TYPES   = {"TIMESTAMP_NTZ","TIMESTAMP_LTZ","TIMESTAMP_TZ","TIMESTAMP","DATETIME","TIME"}
    NUM_TYPES  = {"NUMBER","NUMERIC","INTEGER","INT","BIGINT","SMALLINT","FLOAT","DOUBLE","DECIMAL"}
    TEXT_TYPES = {"TEXT","VARCHAR","CHAR","STRING","NVARCHAR"}

    for tbl, cols in schema_dict.items():
        for col, dtype in cols.items():
            base = dtype.split("(")[0].upper()
            if base in DATE_TYPES:
                native_date_cols.append(f"{tbl}.{col}")
            elif base in TS_TYPES:
                native_ts_cols.append(f"{tbl}.{col}")
            elif base in NUM_TYPES:
                # Could be an integer-encoded date — flag for LLM awareness
                kw = col.lower()
                if any(k in kw for k in ["date","time","dt","day","month","year","_at","_on"]):
                    numeric_date_cols.append(f"{tbl}.{col}")

    lines = ["════════════════════════════════════════════════════════",
             "⛔ ABSOLUTE RULE #2 — DATE COLUMN TYPES (READ CAREFULLY)",
             "════════════════════════════════════════════════════════"]

    if native_date_cols:
        lines.append("\n✅ NATIVE DATE columns — use DIRECTLY with YEAR(), MONTH(), DATE_TRUNC():")
        lines.append("   ⛔ NEVER wrap these in TO_DATE(), LPAD(), or CAST()!")
        for c in native_date_cols:
            lines.append(f"   • {c}")

    if native_ts_cols:
        lines.append("\n✅ NATIVE TIMESTAMP columns — use DATE_TRUNC() or CAST(col AS DATE):")
        lines.append("   ⛔ NEVER wrap these in TO_DATE() or LPAD()!")
        for c in native_ts_cols:
            lines.append(f"   • {c}")

    if numeric_date_cols:
        lines.append("\n⚠️  NUMERIC columns that MAY store encoded dates (INTEGER format):")
        lines.append("   For YYYYMMDD integers: TO_DATE(LPAD(CAST(col AS VARCHAR),8,'0'),'YYYYMMDD')")
        lines.append("   For DDMMYYYY integers: TO_DATE(LPAD(CAST(col AS VARCHAR),8,'0'),'DDMMYYYY')")
        for c in numeric_date_cols:
            lines.append(f"   • {c}")

    lines.append("""
GENERAL DATE RULES:
  • Native DATE   → use col directly: YEAR(col), DATE_TRUNC('MONTH', col), col >= '2025-01-01'
  • Native TS     → use CAST(col AS DATE) first, then apply date functions
  • Integer dates → convert with TO_DATE(LPAD(CAST(col AS VARCHAR), 8, '0'), 'FORMAT') first

TIME-SERIES BEST PRACTICES:
  • Monthly trend: SELECT DATE_TRUNC('MONTH', col) AS MONTH, SUM(...) ... GROUP BY 1 ORDER BY 1
  • Filter year:   WHERE YEAR(col) = 2025
  • Filter range:  WHERE col BETWEEN '2025-01-01' AND '2025-12-31'
  • Always ORDER BY the time column in time-series queries""")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
#  DATE FORMAT DETECTION & SQL REPAIR
# ─────────────────────────────────────────────────────────────────
_date_fmt_cache: dict = {}

def detect_col_date_format(database: str, table_name: str, col_name: str) -> str:
    """
    Returns the storage format of a date-related column.
    Uses INFORMATION_SCHEMA declared type first (authoritative),
    then falls back to sampling real data.

    Return values: 'date' | 'timestamp' | 'yyyymmdd_int' | 'ddmmyyyy_int' |
                   'yyyymmdd_str' | 'ddmmyyyy_str' | 'yyyymmdd_dashed' |
                   'ddmmyyyy_dashed' | 'yyyymm_int' | 'unknown'
    """
    cache_key = (database.upper(), table_name.upper(), col_name.upper())
    if cache_key in _date_fmt_cache:
        return _date_fmt_cache[cache_key]

    def _store(fmt: str) -> str:
        _date_fmt_cache[cache_key] = fmt
        return fmt

    # Step 1 — INFORMATION_SCHEMA declared type (most reliable)
    try:
        parts       = table_name.upper().split(".")
        tbl_only    = parts[-1]
        schema_part = parts[-2] if len(parts) >= 2 else None
        sf          = f"AND TABLE_SCHEMA = '{schema_part}'" if schema_part else ""
        type_sql    = f"""
            SELECT DATA_TYPE FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{tbl_only}' AND COLUMN_NAME = '{col_name.upper()}' {sf}
            LIMIT 1
        """
        tdf = run_query(type_sql, database)
        if not tdf.empty:
            declared = str(tdf.iloc[0]["DATA_TYPE"]).upper().split("(")[0]
            if declared == "DATE":
                return _store("date")
            if declared in ("TIMESTAMP_NTZ","TIMESTAMP_LTZ","TIMESTAMP_TZ","TIMESTAMP","DATETIME","TIME"):
                return _store("timestamp")
            # NUMBER/TEXT → fall through to sampling
    except Exception:
        pass

    # Step 2 — sample actual values
    try:
        sdf = run_query(
            f"SELECT TO_VARCHAR({col_name}) AS V FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 5",
            database,
        )
        if sdf.empty:
            return _store("unknown")
        val = str(sdf.iloc[0]["V"]).strip()

        if _re.match(r'^\d{4}-\d{2}-\d{2}', val):        # "2025-01-15"
            return _store("date")
        if _re.match(r'^\d{8}$', val):
            yr = int(val[:4])
            if 1900 <= yr <= 2100:
                return _store("yyyymmdd_int")
            dd, mm = int(val[:2]), int(val[2:4])
            if 1 <= dd <= 31 and 1 <= mm <= 12:
                return _store("ddmmyyyy_int")
            return _store("yyyymmdd_int")
        if _re.match(r'^\d{6}$', val):
            return _store("yyyymm_int")
        if _re.match(r'^\d{2}[-/]\d{2}[-/]\d{4}$', val):
            return _store("ddmmyyyy_dashed")
        if _re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}$', val):
            return _store("yyyymmdd_dashed")
        return _store("unknown")
    except Exception:
        return _store("unknown")


def _to_date_expr(alias_col: str, fmt: str) -> str:
    """Wraps alias_col in the correct TO_DATE() conversion for its format."""
    if fmt == "date":
        return alias_col
    if fmt == "timestamp":
        return f"CAST({alias_col} AS DATE)"
    if fmt == "yyyymmdd_int":
        return f"TO_DATE(LPAD(CAST({alias_col} AS VARCHAR),8,'0'),'YYYYMMDD')"
    if fmt == "ddmmyyyy_int":
        return f"TO_DATE(LPAD(CAST({alias_col} AS VARCHAR),8,'0'),'DDMMYYYY')"
    if fmt == "yyyymmdd_str":
        return f"TO_DATE(CAST({alias_col} AS VARCHAR),'YYYYMMDD')"
    if fmt == "ddmmyyyy_str":
        return f"TO_DATE(CAST({alias_col} AS VARCHAR),'DDMMYYYY')"
    if fmt == "ddmmyyyy_dashed":
        return f"TO_DATE({alias_col},'DD-MM-YYYY')"
    if fmt == "yyyymmdd_dashed":
        return f"TO_DATE({alias_col},'YYYY-MM-DD')"
    if fmt == "yyyymm_int":
        return f"TO_DATE(CAST({alias_col} AS VARCHAR)||'01','YYYYMMDD')"
    return f"TRY_TO_DATE(TO_VARCHAR({alias_col}))"


def fix_date_filter_in_sql(sql: str, database: str) -> str:
    """
    Two-pass repair:
      Pass 1 — STRIP bad TO_DATE(LPAD(...)) wrappers from native DATE cols.
      Pass 2 — WRAP integer/string date cols with correct conversion.
    """
    # Build alias → table map
    table_aliases: dict = {}
    for m in _re.finditer(
        r'\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b',
        sql, _re.IGNORECASE
    ):
        table_aliases[m.group(2).upper()] = m.group(1).upper()

    # Collect all alias.col pairs that look date-related
    date_refs = _re.findall(
        r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_]*(?:KEY|DATE|TIME|DT|DAY|MONTH|YEAR|AT|ON)[A-Za-z0-9_]*)\b',
        sql, _re.IGNORECASE
    )

    seen: set = set()
    for alias, col in date_refs:
        key = f"{alias.upper()}.{col.upper()}"
        if key in seen:
            continue
        seen.add(key)

        table_name = table_aliases.get(alias.upper(), alias.upper())
        fmt        = detect_col_date_format(database, table_name, col)
        ac         = f"{alias}.{col}"   # e.g. "d.DATE"

        if fmt in ("date", "timestamp"):
            # ── PASS 1: strip any LLM-generated wrapper ──
            # TO_DATE(LPAD(CAST(ac AS VARCHAR), N, '0'), '...')
            sql = _re.sub(
                rf"TO_DATE\s*\(\s*LPAD\s*\(\s*CAST\s*\(\s*{_re.escape(ac)}\s+AS\s+VARCHAR\s*\)"
                rf"\s*,\s*\d+\s*,\s*'0'\s*\)\s*,\s*'[^']+'\s*\)",
                ac, sql, flags=_re.IGNORECASE
            )
            # TO_DATE(CAST(ac AS VARCHAR), '...')
            sql = _re.sub(
                rf"TO_DATE\s*\(\s*CAST\s*\(\s*{_re.escape(ac)}\s+AS\s+VARCHAR\s*\)\s*,\s*'[^']+'\s*\)",
                ac, sql, flags=_re.IGNORECASE
            )
            # TO_DATE(ac, '...')
            sql = _re.sub(
                rf"TO_DATE\s*\(\s*{_re.escape(ac)}\s*,\s*'[^']+'\s*\)",
                ac, sql, flags=_re.IGNORECASE
            )
            # TRY_TO_DATE(TO_VARCHAR(ac))
            sql = _re.sub(
                rf"TRY_TO_DATE\s*\(\s*TO_VARCHAR\s*\(\s*{_re.escape(ac)}\s*\)\s*\)",
                ac, sql, flags=_re.IGNORECASE
            )
            # For TIMESTAMP: wrap with CAST(... AS DATE) inside date functions
            if fmt == "timestamp":
                cast_expr = f"CAST({ac} AS DATE)"
                for fn in ("YEAR","MONTH","DAY"):
                    sql = _re.sub(
                        rf'\b{fn}\s*\(\s*{_re.escape(ac)}\s*\)',
                        f'{fn}({cast_expr})',
                        sql, flags=_re.IGNORECASE
                    )
                sql = _re.sub(
                    rf"DATE_TRUNC\s*\(\s*'([^']+)'\s*,\s*{_re.escape(ac)}\s*\)",
                    lambda m: f"DATE_TRUNC('{m.group(1)}',{cast_expr})",
                    sql, flags=_re.IGNORECASE
                )
        else:
            # ── PASS 2: wrap integer/string dates correctly ──
            date_expr = _to_date_expr(ac, fmt)
            # YEAR(ac) = YYYY
            sql = _re.sub(
                rf'YEAR\s*\(\s*{_re.escape(ac)}\s*\)',
                f'YEAR({date_expr})',
                sql, flags=_re.IGNORECASE
            )
            # MONTH(ac)
            sql = _re.sub(
                rf'MONTH\s*\(\s*{_re.escape(ac)}\s*\)',
                f'MONTH({date_expr})',
                sql, flags=_re.IGNORECASE
            )
            # FLOOR(ac / 10000) = YYYY
            sql = _re.sub(
                rf'FLOOR\s*\(\s*{_re.escape(ac)}\s*/\s*10000\s*\)',
                f'YEAR({date_expr})',
                sql, flags=_re.IGNORECASE
            )
            # DATE_TRUNC('...', ac)
            sql = _re.sub(
                rf"DATE_TRUNC\s*\(\s*'([^']+)'\s*,\s*{_re.escape(ac)}\s*\)",
                lambda m: f"DATE_TRUNC('{m.group(1)}',{date_expr})",
                sql, flags=_re.IGNORECASE
            )
            # TO_CHAR(ac, '...')
            sql = _re.sub(
                rf"TO_CHAR\s*\(\s*{_re.escape(ac)}\s*,\s*'([^']+)'\s*\)",
                lambda m: f"TO_CHAR({date_expr},'{m.group(1)}')",
                sql, flags=_re.IGNORECASE
            )
            # EXTRACT(YEAR FROM ac)
            sql = _re.sub(
                rf"EXTRACT\s*\(\s*(\w+)\s+FROM\s+{_re.escape(ac)}\s*\)",
                lambda m: f"EXTRACT({m.group(1)} FROM {date_expr})",
                sql, flags=_re.IGNORECASE
            )

    return sql

# ─────────────────────────────────────────────────────────────────
#  SQL VALIDATION WHITELIST
# ─────────────────────────────────────────────────────────────────
SQL_KEYWORDS = {
    "select","from","where","join","on","and","or","not","in","is","null","inner",
    "order","by","group","having","top","distinct","as","left","right","outer",
    "cross","union","all","between","like","case","when","then","else","end",
    "count","sum","avg","min","max","cast","convert","varchar","nvarchar","int",
    "bigint","float","date","datetime","bit","char","text","decimal","numeric",
    "desc","asc","with","over","partition","row_number","rank","dense_rank",
    "isnull","coalesce","len","trim","upper","lower","substring","datepart",
    "datediff","dateadd","getdate","year","month","day","exists","values",
    "insert","update","delete","create","drop","alter","table","index","view",
    "limit","offset","fetch","next","rows","only","schema","qualify","sample",
    "ilike","regexp","rlike","startswith","endswith","contains","try_cast",
    "to_date","to_timestamp","to_number","to_varchar","to_char","to_decimal",
    "current_date","current_timestamp","current_time","current_user",
    "current_schema","current_database","current_warehouse","current_role",
    "datetrunc","date_trunc","extract","timediff","timestampdiff",
    "last_day","next_day","previous_day","iff","zeroifnull","nullifzero",
    "object_construct","array_construct","flatten","lateral","any_value",
    "listagg","median","mode","stddev","variance","corr","regr_slope",
    "approx_count_distinct","approx_percentile","approx_top_k",
    "trunc","round","ceil","floor","abs","power","sqrt","log","exp","sin","cos",
    "n","no","yes","true","false","percent","null",
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "o","p","q","r","s","t","u","v","w","x","y","z",
    "1","2","3","4","5","6","7","8","9","0",
    "information_schema","sys","public",
    "lpad","rpad","replace","split","split_part","charindex","position",
    "try_to_date","try_to_timestamp","try_to_number","nvl","nvl2",
    "greatest","least","div0","mod","sign","typeof","array_size",
    "date_part","dayofweek","dayofyear","weekofyear","quarter",
    "monthname","dayname","last_value","first_value","lag","lead",
    "ntile","percent_rank","cume_dist","ratio_to_report",
}

def validate_sql_against_whitelist(sql: str, wl: dict) -> tuple:
    if not sql or not wl:
        return True, []
    valid_tables  = {t.lower() for t in wl}
    valid_columns = {c.lower() for cols in wl.values() for c in cols}
    valid_all     = valid_tables | valid_columns | SQL_KEYWORDS

    # Collect aliases defined in this SQL
    aliases: set = set()
    for a in _re.findall(r'\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b', sql, _re.IGNORECASE):
        aliases.add(a.lower())
    for tbl, alias in _re.findall(
        r'\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b',
        sql, _re.IGNORECASE
    ):
        aliases.add(alias.lower())

    allowed = valid_all | aliases

    # Strip string literals, comments before scanning
    sql_clean = _re.sub(r"'[^']*'", "", sql)
    sql_clean = _re.sub(r"--[^\n]*", "", sql_clean)
    sql_clean = _re.sub(r"/\*.*?\*/", "", sql_clean, flags=_re.DOTALL)

    bad, seen = [], set()
    for word in _re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', sql_clean):
        if word.lower() not in allowed and not word.isdigit() and len(word) > 1:
            if word.lower() not in seen:
                seen.add(word.lower())
                bad.append(word)
    return (len(bad) == 0), bad

# ─────────────────────────────────────────────────────────────────
#  COLOR / AXIS / PERCENTAGE HELPERS
# ─────────────────────────────────────────────────────────────────
COLOR_NAME_MAP = {
    "red":"#E53935","green":"#43A047","blue":"#1565C0","yellow":"#FDD835",
    "orange":"#FB8C00","purple":"#8E24AA","pink":"#E91E63","teal":"#00897B",
    "cyan":"#00ACC1","indigo":"#3949AB","lime":"#C0CA33","amber":"#FFB300",
    "brown":"#6D4C41","grey":"#757575","gray":"#757575","black":"#212121",
    "white":"#FFFFFF","navy":"#1A237E","maroon":"#880E4F","violet":"#6A1B9A",
    "gold":"#F9A825","silver":"#9E9E9E","coral":"#FF7043","magenta":"#D81B60",
    "turquoise":"#00BCD4","lavender":"#7E57C2","rose":"#E91E63",
    "crimson":"#B71C1C","salmon":"#EF9A9A","khaki":"#F9A825",
}

def extract_color_from_question(question: str):
    q = question.lower()
    hx = _re.search(r'#?([0-9a-fA-F]{6})\b', question)
    if hx:
        return f"#{hx.group(1).upper()}"
    for name, hval in COLOR_NAME_MAP.items():
        if _re.search(rf'\b{name}\b', q):
            return hval
    return None

def detect_color_change_target(question: str) -> str:
    q = question.lower()
    if not extract_color_from_question(question):
        return "none"
    title_pat = bool(_re.search(
        r'\btitle\s+(color|colour)\b|\b(color|colour)\s+(of\s+)?(the\s+)?title\b'
        r'|\bchange\b.{0,20}\btitle\b.{0,20}\b(color|colour)\b'
        r'|\b(color|colour)\b.{0,20}\btitle\b', q))
    chart_pat = bool(_re.search(
        r'\b(chart|bar|line|graph|plot|area|scatter|series)\s+(color|colour)\b'
        r'|\b(color|colour)\s+(of\s+)?(the\s+)?(chart|bar|bars|line|graph|plot)\b'
        r'|\bchange\b.{0,30}\b(chart|bar|bars|line|graph|plot)\b.{0,30}\b(color|colour)\b'
        r'|\b(color|colour)\b.{0,20}\b(chart|bar|bars|line|graph|series)\b', q))
    if title_pat and not chart_pat: return "title"
    if chart_pat and not title_pat: return "chart"
    return "both"

def detect_axis_label_change(question: str) -> dict:
    """Parse requests like 'change x axis label to Month' or 'set y axis title to Revenue'."""
    q   = question.lower()
    res = {}
    xp  = [
        r'\bx[\s\-]?axis\s+(?:title|label|name)\s+(?:to|as|=)\s*["\']?(.+?)["\']?\s*$',
        r'\b(?:change|rename|set|update)\s+(?:the\s+)?x[\s\-]?axis\s+(?:title|label|name)\s+(?:to|as)\s*["\']?(.+?)["\']?\s*$',
        r'\b(?:change|rename|set|update)\s+(?:the\s+)?x[\s\-]?(?:axis)?\s+(?:title|label)\s+(?:to|as)\s*["\']?(.+?)["\']?\s*$',
        r'\blabel\s+(?:the\s+)?x[\s\-]?axis\s+(?:as|to)\s*["\']?(.+?)["\']?\s*$',
    ]
    for pat in xp:
        m = _re.search(pat, q)
        if m:
            res["x_label"] = m.group(1).strip().strip("\"'")
            break
    yp = [
        r'\by[\s\-]?axis\s+(?:title|label|name)\s+(?:to|as|=)\s*["\']?(.+?)["\']?\s*$',
        r'\b(?:change|rename|set|update)\s+(?:the\s+)?y[\s\-]?axis\s+(?:title|label|name)\s+(?:to|as)\s*["\']?(.+?)["\']?\s*$',
        r'\b(?:change|rename|set|update)\s+(?:the\s+)?y[\s\-]?(?:axis)?\s+(?:title|label)\s+(?:to|as)\s*["\']?(.+?)["\']?\s*$',
        r'\blabel\s+(?:the\s+)?y[\s\-]?axis\s+(?:as|to)\s*["\']?(.+?)["\']?\s*$',
    ]
    for pat in yp:
        m = _re.search(pat, q)
        if m:
            res["y_label"] = m.group(1).strip().strip("\"'")
            break
    return res

def is_axis_label_only_request(question: str) -> bool:
    if not detect_axis_label_change(question):
        return False
    q = question.lower()
    data_words = ["show","list","count","total","sum","average","how many",
                  "what is","give me","top","bottom","trend","compare"]
    return not any(w in q for w in data_words)

def is_percentage_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "percent","percentage","proportion","share","distribution","breakdown",
        "ratio","composition","what portion","how much of","% of","pie","donut",
        "split","contribution",
    ])

# ─────────────────────────────────────────────────────────────────
#  NUMBER FORMATTING
# ─────────────────────────────────────────────────────────────────
_INR_KW  = ["inr","rupee","rupees"]
_USD_KW  = ["revenue","salary","price","cost","fee","earning","earnings",
            "profit","loss","budget","expense","expenses","payment","payments",
            "usd","dollar","dollars","amount_paid","unit_price","sale_price",
            "list_price","invoice_amount","total_sales","total_revenue","sales"]

def _currency_symbol(col: str):
    c = col.lower()
    if any(k in c for k in _INR_KW): return "₹"
    for kw in _USD_KW:
        if _re.search(rf'(^|_){_re.escape(kw)}($|_)', c): return "$"
    return None

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in display.columns:
        is_float   = pd.api.types.is_float_dtype(display[col])
        is_integer = pd.api.types.is_integer_dtype(display[col])
        if not is_float and not is_integer:
            try:
                display[col] = pd.to_numeric(display[col], errors="raise")
                is_float   = pd.api.types.is_float_dtype(display[col])
                is_integer = pd.api.types.is_integer_dtype(display[col])
            except Exception:
                pass
        if is_float or is_integer:
            sym = _currency_symbol(col)
            try:
                rounded = display[col].round(0).fillna(0).astype(int)
            except Exception:
                try: rounded = display[col].round(0)
                except Exception: continue
            if sym:
                display[col] = rounded.apply(lambda v: f"{sym}{int(v):,}" if pd.notna(v) else "")
            else:
                display[col] = rounded.apply(lambda v: f"{int(v)}" if pd.notna(v) else "")
    return display

# ─────────────────────────────────────────────────────────────────
#  SAMPLE QUESTIONS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_sample_questions(database: str) -> list:
    wl = build_whitelist(database)
    if not wl:
        return ["How many total records do we have?","What does the overall data look like?",
                "Show me a summary of the main numbers","What are the top 5 entries overall?"]
    schema_lines   = [f"Table '{t}': columns → {', '.join(c)}" for t,c in list(wl.items())[:20]]
    schema_hint    = "\n".join(schema_lines)
    table_names_str = ", ".join(list(wl.keys())[:20])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role":"system","content":(
                    "You are a senior business analyst. Generate EXACTLY 4 plain-English questions "
                    "a non-technical manager would ask about their business data.\n"
                    "RULES: 1) Base on real schema. 2) Never mention table/column names. "
                    "3) Each targets a different insight. 4) Under 12 words each. "
                    "5) Output ONLY a valid JSON array of 4 strings.")},
                {"role":"user","content":(
                    f"Database: {database}\nTables: {table_names_str}\n\nSchema:\n{schema_hint}\n\n"
                    "Output ONLY a JSON array of 4 strings.")},
            ],
            temperature=0.3, max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        raw = _re.sub(r"^```[a-z]*\n?","",raw).strip("`").strip()
        m   = _re.search(r'\[.*?\]', raw, _re.DOTALL)
        if m: raw = m.group(0)
        qs = json.loads(raw)
        if isinstance(qs, list) and len(qs) >= 2:
            return [str(q) for q in qs[:4]]
    except Exception:
        pass
    combined = " ".join(wl.keys()).lower() + " " + " ".join(c for cols in wl.values() for c in cols).lower()
    if any(k in combined for k in ["student","class","attendance","fee","grade"]):
        return ["How many students are currently enrolled?","Which class has the best attendance?",
                "Who are the top 10 performing students?","How much fee collection is pending?"]
    if any(k in combined for k in ["order","product","sale","customer","revenue"]):
        return ["What is our total revenue this year?","Who are our top 10 customers?",
                "Which products sell the most?","How many orders are still pending?"]
    if any(k in combined for k in ["employee","staff","department","payroll","salary"]):
        return ["How many employees do we have?","Which department has the most staff?",
                "What is the average salary by department?","Who joined the company this year?"]
    hint = f" in {list(wl.keys())[0]}" if wl else ""
    return [f"How many total records do we have{hint}?","Show me the top 10 entries by count",
            "What is the overall breakdown by category?","Give me a summary of the latest data"]

# ─────────────────────────────────────────────────────────────────
#  LOGO HELPER
# ─────────────────────────────────────────────────────────────────
def img_to_b64(filename: str) -> str:
    for base in [pathlib.Path(__file__).parent, pathlib.Path(".")]:
        for ext in ["",".png",".jpg",".jpeg",".webp",".svg"]:
            p = base / (filename + ext)
            if p.exists() and p.is_file():
                sfx  = p.suffix.lower()
                mime = "image/svg+xml" if sfx == ".svg" else f"image/{sfx.lstrip('.')}"
                return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return ""

# ─────────────────────────────────────────────────────────────────
#  NL → SQL
# ─────────────────────────────────────────────────────────────────
def nl_to_sql(question: str, history: list, database: str) -> dict:
    wl          = build_whitelist(database)
    schema_dict = build_full_schema_dict(database)
    strict_block = whitelist_to_strict_block(wl)
    compact_block = "\n".join(f"  {t}: {', '.join(c)}" for t,c in wl.items())
    date_hints    = build_date_type_hints(schema_dict)

    # Pull last assistant turn
    last_sql=last_chart=last_chart_x=last_chart_y=""; last_chart="none"
    last_chart_title=last_summary=""; last_chart_color=last_title_color=None
    last_x_label=last_y_label=last_df=None
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("sql"):
            last_sql         = m["sql"]
            last_chart       = m.get("chart","none")
            last_chart_x     = m.get("chart_x","")
            last_chart_y     = m.get("chart_y","")
            last_chart_title = m.get("chart_title","")
            last_chart_color = m.get("chart_color")
            last_title_color = m.get("title_color")
            last_x_label     = m.get("x_label")
            last_y_label     = m.get("y_label")
            last_summary     = m.get("summary","")
            last_df          = m.get("df")
            break

    _q = question.strip().lower()

    _COLOR_ONLY = bool(_re.search(
        r'\b(make|change|set|use|turn|switch|update)\b.{0,40}\b(color|colour)\b'
        r'|\bcolor\b.{0,20}\b(to|as|into)\b'
        r'|\b(red|green|blue|yellow|orange|purple|pink|teal|cyan|indigo|lime|amber|'
        r'brown|grey|gray|black|navy|maroon|violet|gold|silver|coral|magenta|'
        r'turquoise|lavender|rose|crimson|salmon|khaki)\b', _q))
    _CHART_TYPE = bool(_re.search(
        r'\b(make|change|convert|switch|turn)\b.{0,30}'
        r'\b(bar|line|pie|area|scatter|donut|histogram|funnel|treemap|sunburst|heatmap|violin|box|seaborn|matplotlib)\b'
        r'|\b(bar|line|pie|area|scatter|donut|histogram|funnel|treemap|sunburst|heatmap|violin|box)\s+chart\b', _q))
    _TITLE_ONLY = bool(_re.search(r'\b(change|set|update|rename)\b.{0,20}\btitle\b', _q))
    _axis_changes    = detect_axis_label_change(question)
    _AXIS_LABEL_ONLY = is_axis_label_only_request(question)

    _appearance_only = last_sql and (_COLOR_ONLY or _CHART_TYPE or _TITLE_ONLY or _AXIS_LABEL_ONLY)

    if _appearance_only:
        extracted_color = extract_color_from_question(question)
        color_target    = detect_color_change_target(question)
        new_chart_color = last_chart_color
        new_title_color = last_title_color
        if extracted_color:
            if color_target == "both":
                new_chart_color = extracted_color; new_title_color = extracted_color
            elif color_target == "chart": new_chart_color = extracted_color
            elif color_target == "title": new_title_color = extracted_color

        new_chart = last_chart
        ct_m = _re.search(
            r'\b(bar|line|pie|area|scatter|donut|histogram|funnel|treemap|sunburst|'
            r'heatmap|violin|box|seaborn_bar|seaborn_line|seaborn_heatmap|seaborn_violin|'
            r'seaborn_box|matplotlib_bar|matplotlib_line|matplotlib_pie|matplotlib_hist)\b', _q)
        if ct_m:
            ct = ct_m.group(1)
            if "seaborn" in _q and ct in ("bar","line","heatmap","violin","box"):
                ct = f"seaborn_{ct}"
            elif "matplotlib" in _q and ct in ("bar","line","pie","hist","histogram"):
                ct = "matplotlib_hist" if ct == "histogram" else f"matplotlib_{ct}"
            new_chart = ct

        new_title = last_chart_title
        tm = _re.search(r'title\s+to\s+["\']?(.+?)["\']?\s*$', _q)
        if tm: new_title = tm.group(1).strip().strip("\"'")

        return {
            "sql":last_sql, "summary":last_summary, "chart":new_chart,
            "chart_x":last_chart_x, "chart_y":last_chart_y, "chart_title":new_title,
            "chart_color":new_chart_color, "title_color":new_title_color,
            "x_label":_axis_changes.get("x_label", last_x_label),
            "y_label":_axis_changes.get("y_label", last_y_label),
            "_reuse_df":last_df,
        }

    # ── Build the system prompt ──
    system_prompt = f"""You are an expert Snowflake SQL generator for a business intelligence assistant.

════════════════════════════════════════════════════════
⛔ RULE 1 — ZERO TOLERANCE FOR INVENTED NAMES
════════════════════════════════════════════════════════
You MUST use ONLY table and column names from the EXACT numbered list below.
- Copy names CHARACTER-FOR-CHARACTER. No guessing, abbreviating, pluralizing, or renaming.
- If a concept has no matching column → set sql to "" and explain.
- The numbered list is the ONLY source of truth for identifiers.

════════════════════════════════════════════════════════
DATABASE SCHEMA — NUMBERED (copy exactly)
════════════════════════════════════════════════════════
{strict_block}

{date_hints}

════════════════════════════════════════════════════════
RULE 3 — SNOWFLAKE SQL SYNTAX
════════════════════════════════════════════════════════
- LIMIT N not TOP N
- Always alias tables: FROM ORDERS o, FROM CUSTOMERS c
- Prefix every column ref with its alias when joins are present
- GROUP BY ALL non-aggregated SELECT expressions
- Never SELECT * — list columns explicitly
- Top-N: ORDER BY metric DESC LIMIT N
- Strings: ILIKE (case-insensitive), || (concat)
- Nulls: COALESCE(col, 0)
- Every computed column MUST have an alias: COUNT(*) AS TOTAL_ORDERS

════════════════════════════════════════════════════════
RULE 4 — CHART SELECTION
════════════════════════════════════════════════════════
- percentage / share / proportion / breakdown / distribution → "donut"
- trend / over time / by month / by year / monthly / yearly → "line"
- category comparison → "bar"
- single scalar result → "none"

════════════════════════════════════════════════════════
RULE 5 — FOLLOW-UP HANDLING
════════════════════════════════════════════════════════
Previous SQL: {last_sql if last_sql else "(none)"}
If the user refines the previous question → modify that SQL only.
If it is a new topic → write a fresh query.

════════════════════════════════════════════════════════
OUTPUT — RAW JSON ONLY, NO MARKDOWN, NO CODE FENCES
════════════════════════════════════════════════════════
{{"sql":"SELECT ...","summary":"One plain-English sentence","chart":"bar|line|donut|pie|scatter|area|none","chart_x":"col_alias","chart_y":"col_alias","chart_title":"Short title"}}
"""

    user_message = f"""TABLES AND EXACT COLUMN NAMES (copy verbatim — no modifications):
{compact_block}

QUESTION: {question}

Steps:
1. Pick the correct table(s) from the list.
2. Copy the exact column names — do NOT rename, guess, or invent.
3. For native DATE columns use them directly (no TO_DATE wrapping).
4. For integer-encoded date columns, apply TO_DATE(LPAD(CAST(col AS VARCHAR),8,'0'),'FORMAT').
5. For percentage/share questions set chart to "donut".
6. For time-series questions set chart to "line" and include ORDER BY time col.
7. Return ONLY valid JSON.
"""

    def _call_groq(extra: str = "") -> dict:
        sys_c = system_prompt + ("\n\n" + extra if extra else "")
        msgs  = []
        for m in history[-6:]:
            msgs.append({"role":m["role"],
                         "content": m["content"] if m["role"]=="user" else m.get("summary","")})
        msgs.append({"role":"user","content":user_message})
        client = Groq(api_key=GROQ_API_KEY)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role":"system","content":sys_c}] + msgs,
                    temperature=0.0, max_tokens=1024,
                )
                raw = resp.choices[0].message.content.strip()
                raw = _re.sub(r"^```[a-z]*\n?","",raw).strip("`").strip()
                jm  = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if jm: raw = jm.group(0)
                if not raw: raise ValueError("Empty response")
                return json.loads(raw)
            except Exception as e:
                es = str(e).lower()
                is_rl = any(k in es for k in ("ratelimit","rate_limit","rate limit","429","too many","tokens per minute","requests per minute"))
                if is_rl and attempt < 2:
                    wt = (attempt+1)*20
                    st.toast(f"⏳ Rate limit — retrying in {wt}s…", icon="⏳")
                    time.sleep(wt)
                elif is_rl:
                    raise RuntimeError("⚠️ AI service rate-limited. Wait 30–60 s and retry.")
                else:
                    raise

    _EMPTY = {"sql":"","chart":"none","chart_x":"","chart_y":"","chart_title":"",
              "chart_color":None,"title_color":None,"x_label":None,"y_label":None}

    try:
        result = _call_groq()
    except RuntimeError as re_err:
        return {**_EMPTY, "summary": str(re_err)}
    except Exception as ex:
        return {**_EMPTY, "summary": f"⚠️ Error generating query: {ex}"}

    sql = result.get("sql","").strip()

    # Post-process: force donut for percentage queries
    if is_percentage_query(question) and result.get("chart","none") not in ("pie","donut"):
        result["chart"] = "donut"

    extracted_color = extract_color_from_question(question)
    result["chart_color"] = extracted_color or result.get("chart_color") or None
    result["title_color"] = None
    result.setdefault("x_label", None)
    result.setdefault("y_label", None)

    # Validate identifiers
    is_valid, bad_cols = validate_sql_against_whitelist(sql, wl)
    if not is_valid and sql:
        correction = (
            f"⛔ VALIDATION FAILED — these identifiers are NOT in the schema: {bad_cols}\n\n"
            "You MUST:\n"
            "1. Look at the numbered schema list provided.\n"
            "2. Copy the EXACT correct identifier character-by-character.\n"
            "3. Rewrite the query using ONLY identifiers from that list.\n"
            "4. If no valid column exists → set sql to \"\" and explain.\n"
            "DO NOT guess. DO NOT use similar-looking names."
        )
        try:
            result = _call_groq(extra=correction)
            result["chart_color"] = extracted_color
            result["title_color"] = None
            result.setdefault("x_label", None)
            result.setdefault("y_label", None)
            sql = result.get("sql","").strip()
            ok2, bad2 = validate_sql_against_whitelist(sql, wl)
            if not ok2 and sql:
                result["sql"]     = ""
                result["summary"] = (
                    f"Couldn't generate a valid query — columns ({', '.join(bad2)}) "
                    "don't exist in the schema. Please rephrase or check the sidebar schema."
                )
                result["chart"] = "none"
        except RuntimeError as re_err:
            result["sql"]=""; result["summary"]=str(re_err); result["chart"]="none"
        except Exception:
            result["sql"]=""; result["summary"]="Query generation failed. Please try rephrasing."; result["chart"]="none"

    return result

# ─────────────────────────────────────────────────────────────────
#  CHART COLUMN RESOLVER
# ─────────────────────────────────────────────────────────────────
def resolve_chart_col(col: str, df_columns: list) -> str:
    if not col: return col
    if col in df_columns: return col
    for c in df_columns:
        if c.lower() == col.lower(): return c
    norm = lambda s: _re.sub(r"[\s_]+","",s.lower())
    for c in df_columns:
        if norm(c) == norm(col): return c
    for c in df_columns:
        if col.lower() in c.lower() or c.lower() in col.lower(): return c
    return col

# ─────────────────────────────────────────────────────────────────
#  CHART RENDERER
# ─────────────────────────────────────────────────────────────────
def _eff_title_color(chart_color, title_color) -> str:
    return title_color or chart_color or DEFAULT_CHART_COLOR

def render_chart(df, chart_type, x, y,
                 chart_color=None, chart_title="", title_color=None,
                 x_label=None, y_label=None):
    if chart_type == "none" or not x: return
    sc  = chart_color or DEFAULT_CHART_COLOR
    seq = [chart_color]+DEFAULT_BLUE_SEQUENCE if chart_color else DEFAULT_BLUE_SEQUENCE
    etc = _eff_title_color(chart_color, title_color)

    if chart_type.startswith("seaborn_"):
        _render_seaborn(df,chart_type,x,y,sc,chart_title,etc,x_label,y_label); return
    if chart_type.startswith("matplotlib_"):
        _render_matplotlib(df,chart_type,x,y,sc,seq,chart_title,etc,x_label,y_label); return

    if y and y not in df.columns and chart_type not in ("histogram","pie","donut"):
        st.warning(f"Chart column '{y}' not found in results."); return
    if x not in df.columns:
        st.warning(f"Chart column '{x}' not found in results."); return

    try:
        # Treat small/year-range integers as categorical on X axis
        if x in df.columns and pd.api.types.is_numeric_dtype(df[x]):
            uv = df[x].dropna().unique()
            if len(uv) <= 50 or (df[x].max() <= 9999 and df[x].min() >= 1000):
                df = df.copy(); df[x] = df[x].astype(int).astype(str)

        eff_xl = x_label or x
        eff_yl = y_label or (y or "")
        lbl    = {x: eff_xl, y: eff_yl} if y else {x: eff_xl}
        ck     = dict(title=chart_title) if chart_title else {}

        if   chart_type == "bar":       fig = px.bar(df,x=x,y=y,color_discrete_sequence=[sc],labels=lbl,**ck)
        elif chart_type == "line":      fig = px.line(df,x=x,y=y,markers=True,color_discrete_sequence=[sc],labels=lbl,**ck)
        elif chart_type == "area":      fig = px.area(df,x=x,y=y,color_discrete_sequence=[sc],labels=lbl,**ck)
        elif chart_type == "scatter":   fig = px.scatter(df,x=x,y=y,color_discrete_sequence=[sc],labels=lbl,**ck)
        elif chart_type == "pie":       fig = px.pie(df,names=x,values=y,color_discrete_sequence=seq,**ck)
        elif chart_type == "donut":     fig = px.pie(df,names=x,values=y,hole=0.45,color_discrete_sequence=seq,**ck)
        elif chart_type == "histogram": fig = px.histogram(df,x=x,color_discrete_sequence=[sc],labels={x:eff_xl},**ck)
        elif chart_type == "box":       fig = px.box(df,x=x,y=y,color_discrete_sequence=[sc],labels=lbl,**ck)
        elif chart_type == "funnel":    fig = px.funnel(df,x=y,y=x,color_discrete_sequence=[sc],labels=lbl,**ck)
        elif chart_type == "treemap":   fig = px.treemap(df,path=[x],values=y,color_discrete_sequence=seq,**ck)
        elif chart_type == "sunburst":  fig = px.sunburst(df,path=[x],values=y,color_discrete_sequence=seq,**ck)
        else: st.info(f"Chart type '{chart_type}' is not supported."); return

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Poppins"),
            margin=dict(t=50 if chart_title else 30,b=30,l=10,r=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True,gridcolor="rgba(200,200,200,0.2)"),
            title=dict(text=chart_title,font=dict(family="Poppins",size=15,color=etc),x=0.02) if chart_title else {},
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render {chart_type} chart: {e}")


def _render_seaborn(df,chart_type,x,y,color,title,title_color,x_label=None,y_label=None):
    try:
        sns.set_theme(style="darkgrid")
        fig,ax = plt.subplots(figsize=(10,5))
        fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        if chart_type == "seaborn_bar":
            if x in df.columns and y in df.columns: sns.barplot(data=df,x=x,y=y,color=color,ax=ax)
            else: sns.countplot(data=df,x=x,color=color,ax=ax)
        elif chart_type == "seaborn_line":   sns.lineplot(data=df,x=x,y=y,color=color,marker="o",ax=ax)
        elif chart_type == "seaborn_heatmap":
            nd = df.select_dtypes(include="number")
            if nd.shape[1] >= 2: sns.heatmap(nd.corr(),annot=True,fmt=".2f",cmap="Blues",ax=ax)
            else: sns.barplot(data=df,x=x,y=y,color=color,ax=ax)
        elif chart_type == "seaborn_violin":
            if y in df.columns: sns.violinplot(data=df,x=x,y=y,color=color,ax=ax)
            else: sns.violinplot(data=df,y=x,color=color,ax=ax)
        elif chart_type == "seaborn_box":
            if y in df.columns: sns.boxplot(data=df,x=x,y=y,color=color,ax=ax)
            else: sns.boxplot(data=df,y=x,color=color,ax=ax)
        else: sns.barplot(data=df,x=x,y=y,color=color,ax=ax)
        if title: ax.set_title(title,fontsize=14,color=title_color,fontweight="bold",pad=12)
        if x_label: ax.set_xlabel(x_label,color="gray",fontsize=9)
        if y_label: ax.set_ylabel(y_label,color="gray",fontsize=9)
        ax.tick_params(colors="gray",labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("rgba(100,100,100,0.3)")
        plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",transparent=True)
        buf.seek(0); st.image(buf,use_container_width=True); plt.close(fig)
    except Exception as e: st.warning(f"Seaborn chart error: {e}")


def _render_matplotlib(df,chart_type,x,y,color,seq_colors,title,title_color,x_label=None,y_label=None):
    try:
        fig,ax = plt.subplots(figsize=(10,5))
        fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        if chart_type == "matplotlib_bar":
            xv = df[x].astype(str).tolist(); yv = df[y].tolist() if y in df.columns else [0]*len(xv)
            ax.bar(xv,yv,color=color,width=0.6,edgecolor="none")
            ax.set_xlabel(x_label or x,color="gray",fontsize=9)
            ax.set_ylabel(y_label or (y or ""),color="gray",fontsize=9)
            plt.xticks(rotation=30,ha="right",fontsize=8,color="gray"); plt.yticks(fontsize=8,color="gray")
        elif chart_type == "matplotlib_line":
            xv = df[x].astype(str).tolist(); yv = df[y].tolist() if y in df.columns else [0]*len(xv)
            ax.plot(xv,yv,color=color,marker="o",linewidth=2,markersize=5)
            ax.set_xlabel(x_label or x,color="gray",fontsize=9)
            ax.set_ylabel(y_label or (y or ""),color="gray",fontsize=9)
            plt.xticks(rotation=30,ha="right",fontsize=8,color="gray"); plt.yticks(fontsize=8,color="gray")
        elif chart_type == "matplotlib_pie":
            lb = df[x].astype(str).tolist(); vl = df[y].tolist() if y in df.columns else [1]*len(lb)
            wc = (seq_colors*((len(lb)//len(seq_colors))+1))[:len(lb)]
            ax.pie(vl,labels=lb,colors=wc,autopct="%1.1f%%",startangle=140,
                   textprops={"color":"gray","fontsize":8}); ax.axis("equal")
        elif chart_type == "matplotlib_hist":
            ax.hist(df[x].dropna().tolist(),color=color,edgecolor="none",bins=20)
            ax.set_xlabel(x_label or x,color="gray",fontsize=9)
            ax.set_ylabel(y_label or "Frequency",color="gray",fontsize=9)
            plt.xticks(fontsize=8,color="gray"); plt.yticks(fontsize=8,color="gray")
        else:
            xv = df[x].astype(str).tolist(); yv = df[y].tolist() if y in df.columns else [0]*len(xv)
            ax.bar(xv,yv,color=color,width=0.6,edgecolor="none")
        if title: ax.set_title(title,fontsize=14,color=title_color,fontweight="bold",pad=12)
        ax.tick_params(colors="gray")
        for sp in ax.spines.values(): sp.set_edgecolor("rgba(100,100,100,0.3)")
        ax.grid(axis="y",color="rgba(200,200,200,0.15)",linestyle="--",linewidth=0.5)
        plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",transparent=True)
        buf.seek(0); st.image(buf,use_container_width=True); plt.close(fig)
    except Exception as e: st.warning(f"Matplotlib chart error: {e}")

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_db" not in st.session_state:
    st.session_state.selected_db = None

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    try: _active_theme = st.context.theme.get("base","light")
    except Exception:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        try:
            _cp = pathlib.Path(__file__).parent/".streamlit"/"config.toml"
            _active_theme = tomllib.loads(_cp.read_text()).get("theme",{}).get("base","light")
        except Exception:
            _active_theme = "light"
    _is_dark = (_active_theme == "dark")

    light_src = img_to_b64("techwish_black_transparent")
    dark_src  = img_to_b64("Techwish-Logo-white (3)")
    logo_src  = (dark_src or light_src) if _is_dark else (light_src or dark_src)

    if logo_src:
        st.markdown(
            f'<div class="logo-row"><img src="{logo_src}" style="max-width:150px;height:auto;"/>'
            f'<span class="ai-badge">AI</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="logo-row"><span style="font-family:Poppins,sans-serif;font-weight:800;'
            'color:#1565C0;font-size:1.2rem;">Techwish</span>'
            '<span class="ai-badge">AI</span></div>', unsafe_allow_html=True)

    st.markdown('<p style="color:gray;font-size:0.8rem;margin-top:2px;font-family:Poppins,sans-serif;">'
                'Ask anything about your data</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border:none;height:1px;background:rgba(128,128,128,0.25);margin:0;">', unsafe_allow_html=True)

    st.markdown("**❄️ Select Database**")
    available_dbs = list_databases()
    if not available_dbs:
        st.error("No databases found or Snowflake connection failed.")
        selected_db = None
    else:
        selected_db = st.selectbox("Database", options=available_dbs, index=0, label_visibility="collapsed")
        if selected_db != st.session_state.selected_db:
            st.session_state.messages = []
            st.session_state.selected_db = selected_db

    st.divider()
    if selected_db:
        with st.expander("📋 View Database Schema", expanded=False):
            st.code(load_schema(selected_db) or "Could not load schema.", language="text")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    if selected_db:
        st.markdown('<p style="font-size:0.8rem;font-family:Poppins,sans-serif;color:gray;margin-bottom:6px;">'
                    '💡 <b>Try asking:</b></p>', unsafe_allow_html=True)
        for q in get_sample_questions(selected_db):
            if st.button(q, key=f"sq_{q}", use_container_width=True):
                st.session_state["_inject_question"] = q
                st.rerun()
    else:
        st.markdown('<p style="color:gray;font-size:0.8rem;">Select a database to see suggested questions.</p>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  GUARD
# ─────────────────────────────────────────────────────────────────
if not selected_db:
    st.info("Please select a database from the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
_ai_icon_src = img_to_b64("ai_icon")
_db_icon_html = (
    f'<img src="{_ai_icon_src}" style="width:36px;height:36px;object-fit:contain;border-radius:6px;flex-shrink:0;"/>'
    if _ai_icon_src else '<span style="font-size:1.6rem;line-height:1;">📊</span>'
)
_badge = (
    '<span style="display:inline-flex;align-items:center;gap:5px;font-size:0.75rem;'
    'font-family:Poppins,sans-serif;color:#2E7D32;font-weight:500;">'
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#2E7D32;"></span>'
    'Connected</span>'
) if selected_db else ""

components.html("""<script>
(function(){
  function align(){
    const mb=window.parent.document.querySelector('[data-testid="stAppViewBlockContainer"]');
    const dv=window.parent.document.getElementById('tw-top-divider');
    if(!mb||!dv)return;
    const r=mb.getBoundingClientRect();
    dv.style.marginLeft=`-${r.left}px`;dv.style.width=`calc(100% + ${r.left}px)`;
  }
  setTimeout(align,200);window.addEventListener('resize',align);
})();
</script>""", height=41, scrolling=False)

st.markdown(f"""
<div style="padding:0.5rem 2rem 0 4rem;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
    {_db_icon_html}
    <h1 style="font-family:Poppins,sans-serif;font-weight:800;font-size:1.6rem;margin:0;color:#1565C0;">
      {selected_db or "Analytics"}</h1>
    {_badge}
    <span style="color:gray;font-size:0.9rem;font-family:Poppins,sans-serif;">| Powered by Techwish AI</span>
  </div>
</div>
<hr id="tw-top-divider" style="border:none;height:1px;background:rgba(128,128,128,0.25);
  margin:0.5rem 0 1rem 0;display:block;position:relative;width:100%;"/>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  VOICE ASSISTANT
# ─────────────────────────────────────────────────────────────────
components.html("""<script>
(function(){
  const D=window.parent?window.parent.document:document;
  const ex=D.getElementById('tw-mic-btn'); if(ex) ex.remove();
  const et=D.getElementById('tw-voice-toast'); if(et) et.remove();
  const s=D.createElement('style');
  s.textContent=`
    #tw-mic-btn{position:fixed;bottom:65px;right:128px;width:40px;height:40px;border-radius:49%;
      border:none;cursor:pointer;background:transparent;color:#1565C0;display:flex;
      align-items:center;justify-content:center;transition:all 0.2s;padding:0;outline:none;z-index:99999;}
    #tw-mic-btn:hover{color:#D22630;background:rgba(255,255,255,0.1);}
    #tw-mic-btn.active{color:#E53935;animation:tw-pulse 1s infinite;}
    #tw-mic-btn svg{width:20px;height:20px;}
    @keyframes tw-pulse{0%{opacity:1}50%{opacity:0.4}100%{opacity:1}}
    #tw-voice-toast{position:fixed;bottom:60px;right:52px;z-index:99999;
      background:rgba(20,20,20,0.92);color:#fff;font-family:'Poppins',sans-serif;
      font-size:0.75rem;padding:8px 12px;border-radius:6px;pointer-events:none;
      opacity:0;transition:opacity 0.3s;white-space:nowrap;}
    #tw-voice-toast.show{opacity:1;}`;
  D.head.appendChild(s);
  const MIC=`<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 2a2 2 0 0 0-2 2v6a2 2 0 1 0 4 0V5a2 2 0 0 0-2-2z"/><path d="M19 11a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.93V20H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.07A7 7 0 0 0 19 11z"/></svg>`;
  const STP=`<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`;
  const btn=D.createElement('button'); btn.id='tw-mic-btn'; btn.title='Click to speak'; btn.innerHTML=MIC;
  D.body.appendChild(btn);
  const toast=D.createElement('div'); toast.id='tw-voice-toast'; D.body.appendChild(toast);
  const SR=window.parent.SpeechRecognition||window.parent.webkitSpeechRecognition||
           window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){btn.title='Voice not supported (use Chrome)';btn.style.opacity='0.35';btn.style.cursor='not-allowed';
    btn.onclick=()=>showToast('⚠️ Voice not supported — use Chrome/Edge',3000);return;}
  const rec=new SR(); rec.lang='en-US'; rec.interimResults=false; rec.maxAlternatives=1; rec.continuous=false;
  let listening=false;
  function showToast(msg,dur){toast.textContent=msg;toast.classList.add('show');
    clearTimeout(toast._t);toast._t=setTimeout(()=>toast.classList.remove('show'),dur||2500);}
  function start(){rec.start();listening=true;btn.classList.add('active');btn.innerHTML=STP;
    btn.title='Listening… click to stop';showToast('🎙️ Listening…',60000);}
  function stop(){rec.stop();}
  btn.addEventListener('click',()=>{if(listening)stop();else start();});
  rec.onend=()=>{listening=false;btn.classList.remove('active');btn.innerHTML=MIC;
    btn.title='Click to speak';toast.classList.remove('show');};
  rec.onerror=(e)=>{listening=false;btn.classList.remove('active');btn.innerHTML=MIC;
    const m={'not-allowed':'🚫 Permission denied','no-speech':'🔇 No speech','audio-capture':'🎙️ No mic','network':'🌐 Network error'};
    showToast(m[e.error]||`⚠️ ${e.error}`,3500);};
  rec.onresult=(e)=>{const t=e.results[0][0].transcript.trim();if(!t)return;
    const ta=D.querySelector('textarea[data-testid="stChatInputTextArea"]');
    if(!ta){showToast('⚠️ Input not found',3000);return;}
    const ns=Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype,'value').set;
    ns.call(ta,t);ta.dispatchEvent(new Event('input',{bubbles:true}));ta.focus();showToast('✅ Ready to send',2000);};
})();
</script>""", height=45, scrolling=False)

# ─────────────────────────────────────────────────────────────────
#  CHAT HISTORY
# ─────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(msg.get("summary", msg["content"]))
            if msg.get("sql"):
                with st.expander("🔍 View SQL Query", expanded=False):
                    st.markdown(f'<div class="sql-block">{msg["sql"]}</div>', unsafe_allow_html=True)
            if msg.get("df") is not None:
                df_h = pd.DataFrame(msg["df"])
                if not df_h.empty:
                    st.dataframe(format_dataframe(df_h), use_container_width=True)
                    st.caption(f"{len(df_h)} row(s) returned")
            if msg.get("df") is not None and msg.get("chart","none") != "none":
                render_chart(
                    pd.DataFrame(msg["df"]), msg["chart"],
                    msg.get("chart_x",""), msg.get("chart_y",""),
                    chart_color=msg.get("chart_color"), chart_title=msg.get("chart_title",""),
                    title_color=msg.get("title_color"),
                    x_label=msg.get("x_label"), y_label=msg.get("y_label"),
                )

# ─────────────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    wi = img_to_b64("ai_icon")
    it = f'<img id="tw-welcome-img" src="{wi}" alt="AI Assistant"/>' if wi else '<span style="font-size:5rem;">🤖</span>'
    st.markdown(f"""
    <div class="ai-welcome-img">
        {it}
        <p class="ai-welcome-caption">Ask anything about your <strong>{selected_db}</strong> data</p>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  CHAT PROCESSING
# ─────────────────────────────────────────────────────────────────
if "_pending_prompt" in st.session_state:
    pending = st.session_state.pop("_pending_prompt")
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = nl_to_sql(pending, st.session_state.messages, selected_db)

        sql         = result.get("sql","").strip()
        summary     = result.get("summary","")
        chart       = result.get("chart","none")
        chart_x     = result.get("chart_x","")
        chart_y     = result.get("chart_y","")
        chart_color = result.get("chart_color")
        title_color = result.get("title_color")
        chart_title = result.get("chart_title","")
        x_label     = result.get("x_label")
        y_label     = result.get("y_label")
        reuse_df    = result.get("_reuse_df")

        st.markdown(summary)

        df    = None
        error = None

        if reuse_df is not None and not sql:
            df = pd.DataFrame(reuse_df)
        elif sql:
            with st.expander("🔍 View SQL Query", expanded=False):
                st.markdown(f'<div class="sql-block">{sql}</div>', unsafe_allow_html=True)
            with st.spinner("Running query..."):
                try:
                    fixed = fix_date_filter_in_sql(sql, selected_db)
                    if fixed != sql:
                        sql = fixed; result["sql"] = fixed
                    df = run_query(sql, selected_db)
                except Exception as e:
                    error = str(e)
            if error:
                st.error(f"Query failed: {error}")

        if df is not None:
            if df.empty and not reuse_df:
                st.info("Query ran successfully but returned no results.")
            elif not df.empty:
                st.dataframe(format_dataframe(df), use_container_width=True)
                st.caption(f"{len(df)} row(s) returned")
                if chart != "none":
                    rx = resolve_chart_col(chart_x, list(df.columns))
                    ry = resolve_chart_col(chart_y, list(df.columns))
                    render_chart(df, chart, rx, ry, chart_color=chart_color,
                                 chart_title=chart_title, title_color=title_color,
                                 x_label=x_label, y_label=y_label)

        rx_s = resolve_chart_col(chart_x, list(df.columns)) if df is not None and not df.empty else chart_x
        ry_s = resolve_chart_col(chart_y, list(df.columns)) if df is not None and not df.empty else chart_y
        st.session_state.messages.append({
            "role":"assistant","content":summary,"summary":summary,"sql":sql,
            "df": df.to_dict("records") if df is not None and not df.empty else None,
            "chart":chart,"chart_x":rx_s,"chart_y":ry_s,
            "chart_color":chart_color,"title_color":title_color,"chart_title":chart_title,
            "x_label":x_label,"y_label":y_label,
        })

# ─────────────────────────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────────────────────────
_new_prompt = None
if "_inject_question" in st.session_state:
    _new_prompt = st.session_state.pop("_inject_question")
elif prompt := st.chat_input("Ask Techwish AI..."):
    _new_prompt = prompt

if _new_prompt:
    st.session_state.messages.append({"role":"user","content":_new_prompt})
    st.session_state["_pending_prompt"] = _new_prompt
    st.rerun()

import streamlit as st
import os, uuid, json, re as _re
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

# ─────────────────────────────────────────────────────────────────
#  DEFAULT CHART COLOR  (Blue theme)
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
#  CSS — Poppins font + sidebar alignment fix
# ─────────────────────────────────────────────────────────────────
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

/* ── Strip default padding so header sits flush ── */
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-top: 0 !important;
}

/* ── Top bar wrapper ── */
.topbar-wrap {
    width: 100%;
    border-bottom: 1px solid rgba(128,128,128,0.25);
    padding: 1rem 2rem 1rem 2rem;
    margin-bottom: 1.5rem;
    box-sizing: border-box;
}

.topbar-inner {
    display: flex;
    align-items: center;
    gap: 15px;
}

.topbar-inner h1 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800;
    font-size: 1.6rem;
    margin: 0;
    color: #1565C0;
}

.main-content {
    padding: 0 1rem 0 1rem;
}

.header-aligned {
    padding: 1rem 1rem 0 1rem;
}

.result-card {
    background: rgba(128,128,128,0.05);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
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

.logo-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.ai-badge {
    background: #1565C0;
    color: white;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    line-height: 1.4;
    vertical-align: middle;
}

/* ── Empty-state AI image ──
   Uses left:0/right:0 so it always centres in the full viewport
   regardless of whether the sidebar is open or collapsed.
   top:95px gives ~5px breathing room below the divider line.     */
.ai-welcome-img {
    position: center;
    top: 80px;
    bottom: 55px;
    left: 0;
    right: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    pointer-events: auto;
    z-index: 0;
    overflow: hidden;
    flex-shrink: 0;
}
.ai-welcome-img img {
    width:     clamp(200px, 30vw, 210px);
    height:    auto;
    max-width:  auto;
    max-height: auto;
    object-fit: auto;
    display: auto;
    flex-shrink: 0;
}
.ai-welcome-caption {
    margin-top: 0.6rem;
    font-family: 'Poppins', sans-serif;
    font-size: clamp(0.7rem, 1.1vw, 0.9rem);
    color: gray;
    text-align: center;
    pointer-events: auto;
    flex-shrink: 0;
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

def _get_conn() -> "snowflake.connector.connection.SnowflakeConnection":
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

def _create_snowflake_conn(database: str = None):
    return _get_conn()

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
        if any(x in err for x in ("08001", "390114", "Authentication token",
                                   "Connection", "connection", "session")):
            conn = _new_conn()
            st.session_state["_sf_conn"] = conn
            return _execute(conn)
        raise

def list_databases() -> list[str]:
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
#  SCHEMA LOADER
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

# ─────────────────────────────────────────────────────────────────
#  WHITELIST BUILDER
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_whitelist(database: str) -> dict[str, list[str]]:
    schema_sql = f"""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA')
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """
    try:
        df = run_query(schema_sql, database)
        wl: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            tbl = row["TABLE_NAME"]
            wl.setdefault(tbl, []).append(row["COLUMN_NAME"])
        return wl
    except Exception as e:
        return {}

@st.cache_data(show_spinner=False)
def build_full_schema_dict(database: str) -> dict:
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
            dtype = row["DATA_TYPE"]
            if tbl not in schema_dict:
                schema_dict[tbl] = {}
            schema_dict[tbl][col] = dtype
        return schema_dict
    except Exception as e:
        return {}

def whitelist_to_strict_block(wl: dict[str, list[str]]) -> str:
    lines = []
    idx = 1
    for tbl, cols in wl.items():
        lines.append(f"\n[TABLE] {tbl}")
        for col in cols:
            lines.append(f"  {idx:04d}. {col}")
            idx += 1
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
#  SQL KEYWORDS
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
    "datetrunc","date_trunc","extract","datediff","timediff","timestampdiff",
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
}

def extract_sql_identifiers(sql: str) -> list[str]:
    sql_clean = _re.sub(r"'[^']*'", "", sql)
    sql_clean = _re.sub(r"--[^\n]*", "", sql_clean)
    sql_clean = _re.sub(r"/\*.*?\*/", "", sql_clean, flags=_re.DOTALL)
    return _re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql_clean)

def validate_sql_against_whitelist(sql: str, wl: dict[str, list[str]]) -> tuple[bool, list[str]]:
    if not sql or not wl:
        return True, []
    valid_tables  = {t.lower() for t in wl}
    valid_columns = {c.lower() for cols in wl.values() for c in cols}
    valid_all     = valid_tables | valid_columns | SQL_KEYWORDS
    aliases = {a.lower() for a in _re.findall(r'\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', sql, _re.IGNORECASE)}
    for tbl, alias in _re.findall(
        r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
        sql, _re.IGNORECASE
    ):
        aliases.add(alias.lower())
    allowed = valid_all | aliases
    bad, seen = [], set()
    for word in extract_sql_identifiers(sql):
        if word.lower() not in allowed and not word.isdigit() and len(word) > 1:
            if word.lower() not in seen:
                seen.add(word.lower())
                bad.append(word)
    return (len(bad) == 0), bad

# ─────────────────────────────────────────────────────────────────
#  COLOR EXTRACTION
# ─────────────────────────────────────────────────────────────────
COLOR_NAME_MAP = {
    "red": "#E53935", "green": "#43A047", "blue": "#1565C0",
    "yellow": "#FDD835", "orange": "#FB8C00", "purple": "#8E24AA",
    "pink": "#E91E63", "teal": "#00897B", "cyan": "#00ACC1",
    "indigo": "#3949AB", "lime": "#C0CA33", "amber": "#FFB300",
    "brown": "#6D4C41", "grey": "#757575", "gray": "#757575",
    "black": "#212121", "white": "#FFFFFF", "navy": "#1A237E",
    "maroon": "#880E4F", "violet": "#6A1B9A", "gold": "#F9A825",
    "silver": "#9E9E9E", "coral": "#FF7043", "magenta": "#D81B60",
    "turquoise": "#00BCD4", "lavender": "#7E57C2", "rose": "#E91E63",
    "crimson": "#B71C1C", "salmon": "#EF9A9A", "khaki": "#F9A825",
}

def extract_color_from_question(question: str) -> str | None:
    q = question.lower()
    hex_match = _re.search(r'#?([0-9a-fA-F]{6})\b', question)
    if hex_match:
        return f"#{hex_match.group(1).upper()}"
    for name, hex_val in COLOR_NAME_MAP.items():
        if _re.search(rf'\b{name}\b', q):
            return hex_val
    return None

def detect_color_change_target(question: str) -> str:
    q = question.lower()
    has_color = bool(extract_color_from_question(question))
    if not has_color:
        return "none"
    title_pattern = bool(_re.search(
        r'\btitle\s+(color|colour)\b'
        r'|\b(color|colour)\s+(of\s+)?(the\s+)?title\b'
        r'|\bchange\b.{0,20}\btitle\b.{0,20}\b(color|colour)\b'
        r'|\b(color|colour)\b.{0,20}\btitle\b',
        q
    ))
    chart_pattern = bool(_re.search(
        r'\b(chart|bar|line|graph|plot|area|scatter|series)\s+(color|colour)\b'
        r'|\b(color|colour)\s+(of\s+)?(the\s+)?(chart|bar|bars|line|graph|plot)\b'
        r'|\bchange\b.{0,30}\b(chart|bar|bars|line|graph|plot)\b.{0,30}\b(color|colour)\b'
        r'|\b(color|colour)\b.{0,20}\b(chart|bar|bars|line|graph|series)\b',
        q
    ))
    if title_pattern and not chart_pattern:
        return "title"
    if chart_pattern and not title_pattern:
        return "chart"
    return "both"

# ─────────────────────────────────────────────────────────────────
#  HELPER UTILS
# ─────────────────────────────────────────────────────────────────
def is_chart_request(text: str) -> bool:
    return any(k in text.lower() for k in ["chart","graph","plot","visualize","line","bar","pie","donut","seaborn","matplotlib","heatmap","violin","box"])

# ─────────────────────────────────────────────────────────────────
#  NUMBER FORMATTING — round floats to 0 dp, add currency symbol
# ─────────────────────────────────────────────────────────────────
# INR keywords in column name → ₹ prefix
_INR_KEYWORDS = ["inr", "rupee", "rupees", "rs", "rs_", "_rs", "indian"]
# Currency keywords → $ by default
_CURRENCY_KEYWORDS = [
    "amount", "revenue", "price", "salary", "cost", "fee", "total",
    "earning", "earnings", "income", "profit", "loss", "budget",
    "expense", "expenses", "payment", "payments", "value", "sales",
    "turnover", "gross", "net", "usd", "dollar", "dollars",
]

def _get_currency_symbol(col_name: str) -> str | None:
    """Return '$' or '₹' if column looks like a currency column, else None."""
    c = col_name.lower()
    if any(k in c for k in _INR_KEYWORDS):
        return "₹"
    if any(k in c for k in _CURRENCY_KEYWORDS):
        return "$"
    return None

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round all numeric float columns to 0 decimal places.
    Add currency prefix ($  or ₹) for currency-like columns.
    Returns a display copy (strings); original df is unchanged.
    """
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            symbol = _get_currency_symbol(col)
            rounded = display[col].round(0).astype("Int64", errors="ignore")
            # Int64 may fail for NaN-heavy cols — fall back to object
            try:
                rounded = display[col].round(0).fillna(0).astype(int)
            except Exception:
                rounded = display[col].round(0)
            if symbol:
                display[col] = rounded.apply(
                    lambda v: f"{symbol}{v:,}" if pd.notna(v) else ""
                )
            else:
                display[col] = rounded.apply(
                    lambda v: f"{v:,}" if pd.notna(v) else ""
                )
        elif pd.api.types.is_integer_dtype(display[col]):
            symbol = _get_currency_symbol(col)
            if symbol:
                display[col] = display[col].apply(
                    lambda v: f"{symbol}{v:,}" if pd.notna(v) else ""
                )
            else:
                display[col] = display[col].apply(
                    lambda v: f"{v:,}" if pd.notna(v) else ""
                )
    return display

@st.cache_data(show_spinner=False)
def get_sample_questions(database: str) -> list[str]:
    wl = build_whitelist(database)
    if not wl:
        return [
            "How many total records do we have?",
            "What does the overall data look like?",
            "Show me a summary of the main numbers",
            "What are the top 5 entries overall?",
        ]
    schema_lines = []
    for tbl, cols in list(wl.items())[:20]:
        schema_lines.append(f"Table '{tbl}': columns → {', '.join(cols)}")
    schema_hint = "\n".join(schema_lines)
    table_names_str = ", ".join(list(wl.keys())[:20])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior business analyst. Your job is to generate EXACTLY 4 "
                        "plain-English questions that a non-technical manager would ask about "
                        "their business data — based on the real database schema provided.\n\n"
                        "STRICT RULES:\n"
                        "1. Read the table names and column names carefully.\n"
                        "2. Generate questions DIRECTLY related to the actual tables and columns.\n"
                        "3. NEVER mention any table name, column name, or SQL term.\n"
                        "4. Each question must target a DIFFERENT insight.\n"
                        "5. Questions must sound like a real person talking to an assistant.\n"
                        "6. Keep each question under 12 words.\n"
                        "7. Output ONLY a valid JSON array of exactly 4 strings.\n\n"
                        "EXAMPLES for a sales database:\n"
                        '["What is our total revenue this month?", '
                        '"Which products are selling the most?", '
                        '"Who are our top 10 customers by spending?", '
                        '"How many orders are still pending?"]'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Database name: {database}\n"
                        f"Tables present: {table_names_str}\n\n"
                        f"Full schema details:\n{schema_hint}\n\n"
                        "Generate 4 relevant business questions. Output ONLY a JSON array of 4 strings."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        raw = _re.sub(r"^```[a-z]*\n?", "", raw).strip("`").strip()
        json_match = _re.search(r'\[.*?\]', raw, _re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        questions = json.loads(raw)
        if isinstance(questions, list) and len(questions) >= 2:
            return [str(q) for q in questions[:4]]
    except Exception:
        pass

    tables_lower = " ".join(wl.keys()).lower()
    all_cols_lower = " ".join(c for cols in wl.values() for c in cols).lower()
    combined = tables_lower + " " + all_cols_lower

    if any(k in combined for k in ["student","class","attendance","fee","grade","subject","mark"]):
        return [
            "How many students are currently enrolled?",
            "Which class has the best attendance?",
            "Who are the top 10 performing students?",
            "How much fee collection is pending?",
        ]
    elif any(k in combined for k in ["order","product","sale","customer","invoice","revenue","amount"]):
        return [
            "What is our total revenue this year?",
            "Who are our top 10 customers?",
            "Which products sell the most?",
            "How many orders are still pending?",
        ]
    elif any(k in combined for k in ["employee","staff","department","payroll","salary"]):
        return [
            "How many employees do we have?",
            "Which department has the most staff?",
            "What is the average salary by department?",
            "Who joined the company this year?",
        ]
    else:
        first_tables = list(wl.keys())[:1]
        hint = f" in {first_tables[0]}" if first_tables else ""
        return [
            f"How many total records do we have{hint}?",
            "Show me the top 10 entries by count",
            "What is the overall breakdown by category?",
            "Give me a summary of the latest data",
        ]

# ─────────────────────────────────────────────────────────────────
#  LOGO HELPER
# ─────────────────────────────────────────────────────────────────
def img_to_b64(filename: str) -> str:
    search_bases = [pathlib.Path(__file__).parent, pathlib.Path(".")]
    extensions   = ["", ".png", ".jpg", ".jpeg", ".webp", ".svg"]
    for base in search_bases:
        for ext in extensions:
            path = base / (filename + ext)
            if path.exists() and path.is_file():
                suffix = path.suffix.lower()
                mime   = "image/svg+xml" if suffix == ".svg" else f"image/{suffix.lstrip('.')}"
                data   = base64.b64encode(path.read_bytes()).decode()
                return f"data:{mime};base64,{data}"
    return ""

# ─────────────────────────────────────────────────────────────────
#  ADVANCED COLUMN RESOLUTION
# ─────────────────────────────────────────────────────────────────
def find_date_columns(schema_dict: dict, table_name: str) -> list[str]:
    if table_name.upper() not in schema_dict:
        return []
    tbl = schema_dict[table_name.upper()]
    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'created', 'updated', 'at']
    date_cols = []
    for col_name, dtype in tbl.items():
        col_lower = col_name.lower()
        dtype_lower = dtype.lower()
        if any(k in dtype_lower for k in ['date', 'timestamp', 'time']):
            date_cols.append(col_name)
        elif any(col_lower.startswith(k) or col_lower.endswith(k) for k in date_keywords):
            date_cols.append(col_name)
    return date_cols

def find_numeric_columns(schema_dict: dict, table_name: str) -> list[str]:
    if table_name.upper() not in schema_dict:
        return []
    tbl = schema_dict[table_name.upper()]
    numeric_types = ['int', 'float', 'decimal', 'numeric', 'bigint', 'smallint', 'double', 'number']
    numeric_cols = []
    for col_name, dtype in tbl.items():
        dtype_lower = dtype.lower()
        if any(nt in dtype_lower for nt in numeric_types):
            numeric_cols.append(col_name)
    return numeric_cols

def find_text_columns(schema_dict: dict, table_name: str) -> list[str]:
    if table_name.upper() not in schema_dict:
        return []
    tbl = schema_dict[table_name.upper()]
    text_types = ['varchar', 'char', 'text', 'string']
    text_cols = []
    for col_name, dtype in tbl.items():
        dtype_lower = dtype.lower()
        if any(tt in dtype_lower for tt in text_types):
            text_cols.append(col_name)
    return text_cols

# ─────────────────────────────────────────────────────────────────
#  NL → SQL  (Snowflake SQL dialect)
# ─────────────────────────────────────────────────────────────────
def nl_to_sql(question: str, history: list, database: str) -> dict:
    wl = build_whitelist(database)
    schema_dict = build_full_schema_dict(database)
    strict_block = whitelist_to_strict_block(wl)
    compact_block = "\n".join(
        f"  {tbl}: {', '.join(cols)}"
        for tbl, cols in wl.items()
    )

    last_sql          = ""
    last_chart        = "none"
    last_chart_x      = ""
    last_chart_y      = ""
    last_chart_title  = ""
    last_chart_color  = None
    last_title_color  = None
    last_summary      = ""
    last_df           = None
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("sql"):
            last_sql          = m["sql"]
            last_chart        = m.get("chart", "none")
            last_chart_x      = m.get("chart_x", "")
            last_chart_y      = m.get("chart_y", "")
            last_chart_title  = m.get("chart_title", "")
            last_chart_color  = m.get("chart_color")
            last_title_color  = m.get("title_color")
            last_summary      = m.get("summary", "")
            last_df           = m.get("df")
            break

    _q = question.strip().lower()

    _COLOR_ONLY = bool(_re.search(
        r'\b(make|change|set|use|turn|switch|update)\b.{0,40}\b(color|colour)\b'
        r'|\bcolor\b.{0,20}\b(to|as|into)\b'
        r'|\b(red|green|blue|yellow|orange|purple|pink|teal|cyan|indigo|lime|amber|'
        r'brown|grey|gray|black|navy|maroon|violet|gold|silver|coral|magenta|'
        r'turquoise|lavender|rose|crimson|salmon|khaki)\b',
        _q
    ))
    _CHART_TYPE = bool(_re.search(
        r'\b(make|change|convert|switch|turn)\b.{0,30}'
        r'\b(bar|line|pie|area|scatter|donut|histogram|funnel|treemap|sunburst|heatmap|violin|box|seaborn|matplotlib)\b'
        r'|\b(bar|line|pie|area|scatter|donut|histogram|funnel|treemap|sunburst|heatmap|violin|box)\s+chart\b',
        _q
    ))
    _TITLE_ONLY = bool(_re.search(
        r'\b(change|set|update|rename)\b.{0,20}\btitle\b',
        _q
    ))

    _is_appearance_only = last_sql and (_COLOR_ONLY or _CHART_TYPE or _TITLE_ONLY)

    if _is_appearance_only:
        extracted_color = extract_color_from_question(question)
        color_target    = detect_color_change_target(question)

        new_chart_color = last_chart_color
        new_title_color = last_title_color

        if extracted_color:
            if color_target == "both":
                new_chart_color = extracted_color
                new_title_color = extracted_color
            elif color_target == "chart":
                new_chart_color = extracted_color
            elif color_target == "title":
                new_title_color = extracted_color

        new_chart = last_chart
        _ct_match = _re.search(
            r'\b(bar|line|pie|area|scatter|donut|histogram|funnel|treemap|sunburst|heatmap|violin|box|seaborn_bar|seaborn_line|seaborn_heatmap|seaborn_violin|seaborn_box|matplotlib_bar|matplotlib_line|matplotlib_pie|matplotlib_hist)\b', _q
        )
        if _ct_match:
            ct = _ct_match.group(1)
            if "seaborn" in _q and ct in ("bar","line","heatmap","violin","box"):
                ct = f"seaborn_{ct}"
            elif "matplotlib" in _q and ct in ("bar","line","pie","hist","histogram"):
                ct = f"matplotlib_{ct}" if ct != "histogram" else "matplotlib_hist"
            new_chart = ct

        new_title = last_chart_title
        _title_match = _re.search(r'title\s+to\s+["\']?(.+?)["\']?\s*$', _q)
        if _title_match:
            new_title = _title_match.group(1).strip().strip("\"'")

        return {
            "sql":         last_sql,
            "summary":     last_summary,
            "chart":       new_chart,
            "chart_x":     last_chart_x,
            "chart_y":     last_chart_y,
            "chart_title": new_title,
            "chart_color": new_chart_color,
            "title_color": new_title_color,
            "_reuse_df":   last_df,
        }

    system_prompt = f"""You are an expert business intelligence assistant and STRICT Snowflake SQL query generator.

════════════════════════════════════════════════════════
CRITICAL INSTRUCTIONS FOR COLUMN & TABLE ACCURACY
════════════════════════════════════════════════════════
1. EXACT MATCHING: Copy table and column names EXACTLY as they appear in the schema.
2. CASE SENSITIVITY: Snowflake identifiers are case-insensitive but you MUST use the exact names from schema.
3. NO GUESSING: If uncertain about a column name, ask the user or use generic aggregates.
4. VERIFY EXISTENCE: Check the numbered list before writing any SQL.

════════════════════════════════════════════════════════
DATABASE SCHEMA (EXACT NAMES — NO CHANGES)
════════════════════════════════════════════════════════
{strict_block}

ABSOLUTE RULES:
✗ Do NOT modify, abbreviate, pluralize, or rename any identifier
✓ Copy names character-for-character from the numbered list above
✓ If no exact match exists → return sql as "" and explain

════════════════════════════════════════════════════════
TIME SERIES & DATE HANDLING — SNOWFLAKE SPECIFICS
════════════════════════════════════════════════════════
DATE FUNCTIONS (case-insensitive):
- DATE_TRUNC('period', date_col): Truncates date to period (year, month, week, day, hour)
- DATEADD('unit', num, date_col): Add/subtract time. Units: year, month, week, day, hour, minute, second
- DATEDIFF('unit', start_date, end_date): Calculate difference between dates
- CURRENT_DATE: Today's date (no parentheses)
- EXTRACT('unit' FROM date_col): Extract year, month, day, quarter, week, dayofweek, etc.
- TO_DATE(string): Convert string to date

TIME SERIES BEST PRACTICES:
- For "trends": GROUP BY DATE_TRUNC('month', date_col) or DATE_TRUNC('day', date_col)
- For "this period": WHERE date_col >= DATE_TRUNC('month', CURRENT_DATE)
- For "last N days": WHERE date_col >= DATEADD('day', -N, CURRENT_DATE)
- For "year-over-year": Use WHERE YEAR(date_col) = YEAR(CURRENT_DATE) - 1
- For "monthly breakdown": SELECT DATE_TRUNC('month', date_col) AS month, SUM(amount) ... GROUP BY 1 ORDER BY 1
- For "daily trend": SELECT DATE(date_col) AS day, COUNT(*) ... GROUP BY 1 ORDER BY 1 DESC
- Always alias date expressions: DATE_TRUNC('month', date_col) AS month_bucket

════════════════════════════════════════════════════════
SNOWFLAKE SQL RULES
════════════════════════════════════════════════════════
- Use LIMIT N (NOT TOP N)
- Column/table names are case-insensitive → use exact names from schema
- Always alias tables: FROM CUSTOMERS c, FROM ORDERS o
- Prefix column refs with alias when joins present: c.customer_id, o.order_date
- GROUP BY all non-aggregated columns when using COUNT/SUM/AVG/MIN/MAX
- Never SELECT * — always list explicit columns
- Use ORDER BY col DESC LIMIT N for "top N"
- For strings: ILIKE for case-insensitive, || for concat
- For nulls: COALESCE(col, 0) or ISNULL(col, default_value)
- Use explicit JOIN syntax (INNER JOIN, LEFT JOIN, etc.)

════════════════════════════════════════════════════════
FOLLOW-UP QUERY HANDLING
════════════════════════════════════════════════════════
Last SQL: {last_sql if last_sql else "(none — first query)"}

If new question modifies last query → edit ONLY that query
If new topic → write fresh query

════════════════════════════════════════════════════════
OUTPUT — RAW JSON ONLY. NO MARKDOWN. NO CODE FENCES.
════════════════════════════════════════════════════════
{{
  "sql": "SELECT ... FROM ... WHERE ...",
  "summary": "One sentence business insight in plain English (no SQL jargon)",
  "chart": "bar|line|pie|donut|scatter|area|histogram|seaborn_bar|none|...",
  "chart_x": "exact_column_name",
  "chart_y": "exact_column_name",
  "chart_title": "Short title (if chart != none)"
}}
"""

    user_message = f"""AVAILABLE TABLES AND COLUMNS (copy names EXACTLY):
{compact_block}

USER QUESTION: {question}

INSTRUCTIONS:
1. Identify which table(s) best match the business intent
2. Find the EXACT column names from the list above
3. Write proper Snowflake SQL using ONLY those exact names
4. For time-series queries: use DATE_TRUNC and DATEADD appropriately
5. Always generate a descriptive chart title
6. Output ONLY valid JSON with no explanation or markdown
"""

    def call_groq(extra_instruction: str = "") -> dict:
        sys_content = system_prompt + ("\n\n" + extra_instruction if extra_instruction else "")
        messages = []
        for m in history[-6:]:
            messages.append({
                "role": m["role"],
                "content": m["content"] if m["role"] == "user" else m.get("summary", ""),
            })
        messages.append({"role": "user", "content": user_message})
        client = Groq(api_key=GROQ_API_KEY)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "system", "content": sys_content}] + messages,
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content.strip()
                raw = _re.sub(r"^```[a-z]*\n?", "", raw).strip("`").strip()
                json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if json_match:
                    raw = json_match.group(0)
                if not raw:
                    raise ValueError("Empty response from model")
                return json.loads(raw)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "ratelimit" in err_str or
                    "rate_limit" in err_str or
                    "rate limit" in err_str or
                    "429" in err_str or
                    "too many" in err_str or
                    "tokens per minute" in err_str or
                    "requests per minute" in err_str
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 20
                    st.toast(f"⏳ Rate limit reached — retrying in {wait_time}s… (attempt {attempt + 1}/{max_retries})", icon="⏳")
                    time.sleep(wait_time)
                    continue
                elif is_rate_limit:
                    raise RuntimeError(
                        "⚠️ The AI service is currently rate-limited (too many requests). "
                        "Please wait 30–60 seconds and try again."
                    )
                else:
                    raise

    try:
        result = call_groq()
    except RuntimeError as rate_err:
        return {
            "sql": "",
            "summary": str(rate_err),
            "chart": "none",
            "chart_x": "",
            "chart_y": "",
            "chart_title": "",
            "chart_color": None,
            "title_color": None,
        }
    except Exception as e:
        return {
            "sql": "",
            "summary": f"⚠️ Unexpected error while generating query: {e}",
            "chart": "none",
            "chart_x": "",
            "chart_y": "",
            "chart_title": "",
            "chart_color": None,
            "title_color": None,
        }

    sql = result.get("sql", "").strip()

    extracted_color = extract_color_from_question(question)
    result["chart_color"] = extracted_color or result.get("chart_color") or None
    result["title_color"] = None

    is_valid, bad_cols = validate_sql_against_whitelist(sql, wl)
    if not is_valid and sql:
        bad_list = ", ".join(bad_cols)
        correction = f"""
⛔ VALIDATION FAILED - COLUMN/TABLE NAMES DO NOT MATCH SCHEMA:
Your SQL referenced: [{bad_list}]

These identifiers are NOT in the schema. You MUST:
1. Check the numbered column list above
2. Find the EXACT correct name (copy character-by-character)
3. Rewrite query using ONLY schema names
4. If no valid columns → return sql as empty string ""

DO NOT GUESS. Use EXACT names from the numbered list only.
"""
        try:
            result = call_groq(extra_instruction=correction)
            result["chart_color"] = extracted_color
            result["title_color"] = None
            sql = result.get("sql", "").strip()
            is_valid2, bad_cols2 = validate_sql_against_whitelist(sql, wl)
            if not is_valid2 and sql:
                result["sql"]     = ""
                result["summary"] = (
                    f"I couldn't generate a valid query — the columns "
                    f"({', '.join(bad_cols2)}) don't exist in the schema. "
                    "Please rephrase your question or check the sidebar schema."
                )
                result["chart"] = "none"
        except RuntimeError as rate_err:
            result["sql"]     = ""
            result["summary"] = str(rate_err)
            result["chart"]   = "none"
        except Exception:
            result["sql"]     = ""
            result["summary"] = "Query generation failed after validation. Please try rephrasing."
            result["chart"]   = "none"

    return result

# ─────────────────────────────────────────────────────────────────
#  COLUMN RESOLVER
# ─────────────────────────────────────────────────────────────────
def resolve_chart_col(col: str, df_columns: list) -> str:
    if not col:
        return col
    if col in df_columns:
        return col
    col_lower = col.lower()
    for c in df_columns:
        if c.lower() == col_lower:
            return c
    def norm(s):
        return _re.sub(r"[\s_]+", "", s.lower())
    col_norm = norm(col)
    for c in df_columns:
        if norm(c) == col_norm:
            return c
    for c in df_columns:
        if col_lower in c.lower() or c.lower() in col_lower:
            return c
    return col

# ─────────────────────────────────────────────────────────────────
#  CHART RENDERER
# ─────────────────────────────────────────────────────────────────
def _effective_title_color(chart_color: str | None, title_color: str | None) -> str:
    if title_color:
        return title_color
    return chart_color if chart_color else DEFAULT_CHART_COLOR

def render_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: str,
    chart_color: str | None = None,
    chart_title: str = "",
    title_color: str | None = None,
):
    if chart_type == "none" or not x:
        return

    single_color    = chart_color if chart_color else DEFAULT_CHART_COLOR
    seq_colors      = [chart_color] + DEFAULT_BLUE_SEQUENCE if chart_color else DEFAULT_BLUE_SEQUENCE
    eff_title_color = _effective_title_color(chart_color, title_color)

    if chart_type.startswith("seaborn_"):
        _render_seaborn(df, chart_type, x, y, single_color, chart_title, eff_title_color)
        return

    if chart_type.startswith("matplotlib_"):
        _render_matplotlib(df, chart_type, x, y, single_color, seq_colors, chart_title, eff_title_color)
        return

    if y and y not in df.columns and chart_type not in ["histogram", "pie", "donut"]:
        st.warning(f"Chart column '{y}' not found in results.")
        return
    if x not in df.columns:
        st.warning(f"Chart column '{x}' not found in results.")
        return

    try:
        common_kwargs = dict(title=chart_title) if chart_title else {}
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color_discrete_sequence=[single_color], **common_kwargs)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, markers=True, color_discrete_sequence=[single_color], **common_kwargs)
        elif chart_type == "area":
            fig = px.area(df, x=x, y=y, color_discrete_sequence=[single_color], **common_kwargs)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color_discrete_sequence=[single_color], **common_kwargs)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, color_discrete_sequence=seq_colors, **common_kwargs)
        elif chart_type == "donut":
            fig = px.pie(df, names=x, values=y, hole=0.45, color_discrete_sequence=seq_colors, **common_kwargs)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x, color_discrete_sequence=[single_color], **common_kwargs)
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color_discrete_sequence=[single_color], **common_kwargs)
        elif chart_type == "funnel":
            fig = px.funnel(df, x=y, y=x, color_discrete_sequence=[single_color], **common_kwargs)
        elif chart_type == "treemap":
            fig = px.treemap(df, path=[x], values=y, color_discrete_sequence=seq_colors, **common_kwargs)
        elif chart_type == "sunburst":
            fig = px.sunburst(df, path=[x], values=y, color_discrete_sequence=seq_colors, **common_kwargs)
        else:
            st.info(f"Chart type '{chart_type}' is not supported.")
            return

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Poppins"),
            margin=dict(t=50 if chart_title else 30, b=30, l=10, r=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
            title=dict(
                text=chart_title,
                font=dict(family="Poppins", size=15, color=eff_title_color),
                x=0.02,
            ) if chart_title else {},
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render {chart_type} chart: {e}")


def _render_seaborn(df, chart_type, x, y, color, title, title_color):
    try:
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        if chart_type == "seaborn_bar":
            if x in df.columns and y in df.columns:
                sns.barplot(data=df, x=x, y=y, color=color, ax=ax)
            else:
                sns.countplot(data=df, x=x, color=color, ax=ax)
        elif chart_type == "seaborn_line":
            sns.lineplot(data=df, x=x, y=y, color=color, marker="o", ax=ax)
        elif chart_type == "seaborn_heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                            linewidths=0.5, linecolor="rgba(200,200,200,0.2)")
            else:
                sns.barplot(data=df, x=x, y=y, color=color, ax=ax)
        elif chart_type == "seaborn_violin":
            if y in df.columns:
                sns.violinplot(data=df, x=x, y=y, color=color, ax=ax)
            else:
                sns.violinplot(data=df, y=x, color=color, ax=ax)
        elif chart_type == "seaborn_box":
            if y in df.columns:
                sns.boxplot(data=df, x=x, y=y, color=color, ax=ax)
            else:
                sns.boxplot(data=df, y=x, color=color, ax=ax)
        else:
            sns.barplot(data=df, x=x, y=y, color=color, ax=ax)

        if title:
            ax.set_title(title, fontsize=14, color=title_color, fontweight="bold", pad=12)

        ax.tick_params(colors="gray", labelsize=9)
        ax.xaxis.label.set_color("gray")
        ax.yaxis.label.set_color("gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("rgba(100,100,100,0.3)")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render seaborn chart: {e}")


def _render_matplotlib(df, chart_type, x, y, color, seq_colors, title, title_color):
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        if chart_type == "matplotlib_bar":
            x_vals = df[x].astype(str).tolist()
            y_vals = df[y].tolist() if y in df.columns else [0] * len(x_vals)
            ax.bar(x_vals, y_vals, color=color, width=0.6, edgecolor="none")
            ax.set_xlabel(x, color="gray", fontsize=9)
            ax.set_ylabel(y if y else "", color="gray", fontsize=9)
            plt.xticks(rotation=30, ha="right", fontsize=8, color="gray")
            plt.yticks(fontsize=8, color="gray")
        elif chart_type == "matplotlib_line":
            x_vals = df[x].astype(str).tolist()
            y_vals = df[y].tolist() if y in df.columns else [0] * len(x_vals)
            ax.plot(x_vals, y_vals, color=color, marker="o", linewidth=2, markersize=5)
            ax.set_xlabel(x, color="gray", fontsize=9)
            ax.set_ylabel(y if y else "", color="gray", fontsize=9)
            plt.xticks(rotation=30, ha="right", fontsize=8, color="gray")
            plt.yticks(fontsize=8, color="gray")
        elif chart_type == "matplotlib_pie":
            labels = df[x].astype(str).tolist()
            vals   = df[y].tolist() if y in df.columns else [1] * len(labels)
            wedge_colors = (seq_colors * ((len(labels) // len(seq_colors)) + 1))[:len(labels)]
            ax.pie(vals, labels=labels, colors=wedge_colors,
                   autopct="%1.1f%%", startangle=140,
                   textprops={"color": "gray", "fontsize": 8})
            ax.axis("equal")
        elif chart_type == "matplotlib_hist":
            vals = df[x].dropna().tolist()
            ax.hist(vals, color=color, edgecolor="none", bins=20)
            ax.set_xlabel(x, color="gray", fontsize=9)
            ax.set_ylabel("Frequency", color="gray", fontsize=9)
            plt.xticks(fontsize=8, color="gray")
            plt.yticks(fontsize=8, color="gray")
        else:
            x_vals = df[x].astype(str).tolist()
            y_vals = df[y].tolist() if y in df.columns else [0] * len(x_vals)
            ax.bar(x_vals, y_vals, color=color, width=0.6, edgecolor="none")

        if title:
            ax.set_title(title, fontsize=14, color=title_color, fontweight="bold", pad=12)

        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("rgba(100,100,100,0.3)")
        ax.grid(axis="y", color="rgba(200,200,200,0.15)", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render matplotlib chart: {e}")

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
    try:
        _active_theme = st.context.theme.get("base", "light")
    except Exception:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        try:
            _cfg_path = pathlib.Path(__file__).parent / ".streamlit" / "config.toml"
            _active_theme = tomllib.loads(_cfg_path.read_text()).get("theme", {}).get("base", "light")
        except Exception:
            _active_theme = "light"

    _is_dark = (_active_theme == "dark")

    LOGO_LIGHT_FILE = "techwish_black_transparent"
    LOGO_DARK_FILE  = "Techwish-Logo-white (3)"

    light_src = img_to_b64(LOGO_LIGHT_FILE)
    dark_src  = img_to_b64(LOGO_DARK_FILE)
    logo_src  = (dark_src or light_src) if _is_dark else (light_src or dark_src)

    if logo_src:
        st.markdown(
            f'<div class="logo-row">'
            f'<img src="{logo_src}" style="max-width:150px; height:auto;" />'
            f'<span class="ai-badge">AI</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="logo-row">'
            '<span style="font-family:Poppins,sans-serif;font-weight:800;color:#1565C0;font-size:1.2rem;">Techwish</span>'
            '<span class="ai-badge">AI</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:gray; font-size:0.8rem; margin-top:2px; font-family:Poppins,sans-serif;">Ask anything about your data</p>',
        unsafe_allow_html=True,
    )

   st.markdown("""
<hr style="border: none; height: 1px; background: rgba(128,128,128,0.25); margin-top: 0px; margin-bottom: 0px;">
""", unsafe_allow_html=True)
    st.markdown("**❄️ Select Database**")
    available_dbs = list_databases()

    if not available_dbs:
        st.error("No databases found or Snowflake connection failed.")
        selected_db = None
    else:
        selected_db = st.selectbox(
            label="Database",
            options=available_dbs,
            index=0,
            label_visibility="collapsed",
        )
        if selected_db != st.session_state.selected_db:
            st.session_state.messages = []
            st.session_state.selected_db = selected_db

    st.divider()

    if selected_db:
        with st.expander("📋 View Database Schema", expanded=False):
            schema_text = load_schema(selected_db)
            st.code(schema_text if schema_text else "Could not load schema.", language="text")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if selected_db:
        st.markdown(
            '<p style="font-size:0.8rem; font-family:Poppins,sans-serif; color:gray; margin-bottom:6px;">💡 <b>Try asking:</b></p>',
            unsafe_allow_html=True,
        )
        questions = get_sample_questions(selected_db)
        for q in questions:
            if st.button(q, key=f"sq_{q}", use_container_width=True):
                st.session_state["_inject_question"] = q
                st.rerun()
    else:
        st.markdown(
            '<p style="color:gray; font-size:0.8rem;">Select a database to see suggested questions.</p>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────
#  GUARD — no database selected
# ─────────────────────────────────────────────────────────────────
if not selected_db:
    st.info("Please select a database from the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
#  LOAD AI ICON
# ─────────────────────────────────────────────────────────────────
_ai_icon_src = img_to_b64("ai_icon")

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
connected_badge = (
    '<span style="display:inline-flex; align-items:center; gap:5px; '
    'font-size:0.75rem; font-family:Poppins,sans-serif; color:#2E7D32; font-weight:500;">'
    '<span style="display:inline-block; width:8px; height:8px; border-radius:50%; '
    'background:#2E7D32;"></span>Connected</span>'
) if selected_db else ''

if _ai_icon_src:
    _db_icon_html = (
        f'<img src="{_ai_icon_src}" '
        f'style="width:36px; height:36px; object-fit:contain; border-radius:6px; flex-shrink:0;" />'
    )
else:
    _db_icon_html = '<span style="font-size:1.6rem; line-height:1;">📊</span>'

components.html("""
<script>
(function() {
  function alignDivider() {
    const mainBlock = window.parent.document.querySelector('[data-testid="stAppViewBlockContainer"]');
    const divider   = window.parent.document.getElementById('tw-top-divider');
    if (!mainBlock || !divider) return;
    const rect  = mainBlock.getBoundingClientRect();
    const leftPx = rect.left;
    divider.style.marginLeft = `-${leftPx}px`;
    divider.style.width      = `calc(100% + ${leftPx}px)`;
  }
  setTimeout(alignDivider, 200);
  window.addEventListener('resize', alignDivider);
})();
</script>
""", height=65, scrolling=False)

# Header block — padding-left matches Streamlit's default main-block indent
# (Streamlit uses ~4rem / 64px left padding in wide-layout; we mirror that
#  so the icon+title visually lines up with the chat message bubbles below.)
st.markdown(f"""
<div style="padding: 0.5rem 2rem 0 4rem;">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
        {_db_icon_html}
        <h1 style="font-family:Poppins,sans-serif; font-weight:800; font-size:1.6rem;
                   margin:0; color:#1565C0;">{selected_db or "Analytics"}</h1>
        {connected_badge}
        <span style="color:gray; font-size:0.9rem; font-family:Poppins,sans-serif;">
            | Powered by Techwish AI
        </span>
    </div>
</div>
<hr id="tw-top-divider" style="
    border: none;
    height: 1px;
    background: rgba(128,128,128,0.25);
    margin: 0.5rem 0 1rem 0;
    display: block;
    position: relative;
    width: 100%;
"/>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  VOICE ASSISTANT — Cloud-compatible
#  Renders entirely inside the iframe (no window.parent DOM write).
#  Uses postMessage to send the transcript to the parent, where a
#  second listener fills the Streamlit chat textarea.
#  Mic icon is blue (#1565C0); recording state pulses red.
# ─────────────────────────────────────────────────────────────────
VOICE_COMPONENT_HTML = """
<script>
(function () {
  const targetDoc = window.parent ? window.parent.document : document;

const existing = targetDoc.getElementById('tw-mic-btn');
if (existing) existing.remove();
const existingToast = targetDoc.getElementById('tw-voice-toast');
if (existingToast) existingToast.remove();

  const style = targetDoc.createElement('style');
  style.textContent = `
    #tw-mic-btn {
      position: fixed;
      bottom: 65px;
      right: 128px;
      width: 40px;
      height: 40px;
      border-radius: 49%;
      border: none;
      cursor: pointer;
      background: transparent;
      color: #1565C0;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
      padding: 0;
      outline: none;
      z-index: 99999;
      flex-shrink: 0;
    }
    #tw-mic-btn:hover { color: #D22630; background: rgba(255,255,255,0.1); }
    #tw-mic-btn.active { color: #E53935; animation: tw-pulse 1s infinite; }
    #tw-mic-btn svg { width: 20px; height: 20px; }
    @keyframes tw-pulse {
      0%   { opacity: 1; }
      50%  { opacity: 0.4; }
      100% { opacity: 1; }
    }
    #tw-voice-toast {
      position: fixed;
      bottom: 60px;
      right: 52px;
      z-index: 99999;
      background: rgba(20,20,20,0.92);
      color: #fff;
      font-family: 'Poppins', sans-serif;
      font-size: 0.75rem;
      padding: 8px 12px;
      border-radius: 6px;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.3s;
      white-space: nowrap;
    }
    #tw-voice-toast.show { opacity: 1; }
  `;
  targetDoc.head.appendChild(style);

  const MIC_ICON = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 2a2 2 0 0 0-2 2v6a2 2 0 1 0 4 0V5a2 2 0 0 0-2-2z"/><path d="M19 11a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.93V20H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.07A7 7 0 0 0 19 11z"/></svg>`;
  const STOP_ICON = `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`;

  const btn = targetDoc.createElement('button');
  btn.id = 'tw-mic-btn';
  btn.title = 'Click to speak your question';
  btn.innerHTML = MIC_ICON;
  targetDoc.body.appendChild(btn);

  const toast = targetDoc.createElement('div');
  toast.id = 'tw-voice-toast';
  targetDoc.body.appendChild(toast);

  const SpeechRecognition =
    window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition ||
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    btn.title = 'Voice input not supported (try Chrome)';
    btn.style.opacity = '0.35';
    btn.style.cursor  = 'not-allowed';
    btn.onclick = () => showToast('⚠️ Voice not supported — use Chrome/Edge', 3000);
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;

  let isListening = false;

  function showToast(msg, duration) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove('show'), duration || 2500);
  }

  function startListening() {
    recognition.start();
    isListening = true;
    btn.classList.add('active');
    btn.innerHTML = STOP_ICON;
    btn.title = 'Listening… click to stop';
    showToast('🎙️ Listening…', 60000);
  }

  function stopListening() { recognition.stop(); }

  btn.addEventListener('click', () => { if (isListening) stopListening(); else startListening(); });

  recognition.onend = () => {
    isListening = false;
    btn.classList.remove('active');
    btn.innerHTML = MIC_ICON;
    btn.title = 'Click to speak your question';
    toast.classList.remove('show');
  };

  recognition.onerror = (e) => {
    isListening = false;
    btn.classList.remove('active');
    btn.innerHTML = MIC_ICON;
    const msgs = { 'not-allowed':'🚫 Permission denied','no-speech':'🔇 No speech detected','audio-capture':'🎙️ No microphone','network':'🌐 Network error' };
    showToast(msgs[e.error] || `⚠️ ${e.error}`, 3500);
  };

  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript.trim();
    if (!transcript) return;
    const textarea = targetDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
    if (!textarea) { showToast('⚠️ Input not found', 3000); return; }
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
    nativeInputValueSetter.call(textarea, transcript);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.focus();
    showToast('✅ Ready to send', 2000);
  };
})();
</script>
"""
components.html(VOICE_COMPONENT_HTML, height=40, scrolling=False)

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
                    st.markdown(
                        f'<div class="sql-block">{msg["sql"]}</div>',
                        unsafe_allow_html=True,
                    )
            if msg.get("df") is not None:
                df = pd.DataFrame(msg["df"])
                if not df.empty:
                    display_df = format_dataframe(df)
                    st.dataframe(display_df, use_container_width=True)
                    st.caption(f"{len(df)} row(s) returned")
            if msg.get("df") is not None and msg.get("chart", "none") != "none":
                df = pd.DataFrame(msg["df"])
                render_chart(
                    df,
                    msg["chart"],
                    msg.get("chart_x", ""),
                    msg.get("chart_y", ""),
                    chart_color=msg.get("chart_color"),
                    chart_title=msg.get("chart_title", ""),
                    title_color=msg.get("title_color"),
                )

# ─────────────────────────────────────────────────────────────────
#  EMPTY-STATE AI IMAGE
#  • left:0 / right:0  →  always centred regardless of sidebar
#  • top:95px          →  ~5px gap below the top divider line
# ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    _welcome_icon = img_to_b64("ai_icon")
    _icon_tag = (
        f'<img id="tw-welcome-img" src="{_welcome_icon}" alt="AI Assistant" />'
        if _welcome_icon
        else '<span style="font-size:5rem; line-height:1;">🤖</span>'
    )
    st.markdown(
        f"""
        <div class="ai-welcome-img" id="tw-welcome-wrap">
            {_icon_tag}
            <p class="ai-welcome-caption">
                Ask anything about your <strong>{selected_db}</strong> data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────
#  CHAT PROCESSING
# ─────────────────────────────────────────────────────────────────
if "_pending_prompt" in st.session_state:
    pending = st.session_state.pop("_pending_prompt")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = nl_to_sql(pending, st.session_state.messages, selected_db)

        sql         = result.get("sql", "").strip()
        summary     = result.get("summary", "")
        chart       = result.get("chart", "none")
        chart_x     = result.get("chart_x", "")
        chart_y     = result.get("chart_y", "")
        chart_color = result.get("chart_color")
        title_color = result.get("title_color")
        chart_title = result.get("chart_title", "")
        reuse_df    = result.get("_reuse_df")

        st.markdown(summary)

        df    = None
        error = None

        if sql:
            with st.expander("🔍 View SQL Query", expanded=False):
                st.markdown(
                    f'<div class="sql-block">{sql}</div>',
                    unsafe_allow_html=True,
                )

            if reuse_df is not None:
                df = pd.DataFrame(reuse_df)
            else:
                with st.spinner("Running query..."):
                    try:
                        df = run_query(sql, selected_db)
                    except Exception as e:
                        error = str(e)

            if error:
                st.error(f"Query failed: {error}")
            elif df is not None:
                if df.empty:
                    st.info("Query ran successfully but returned no results.")
                else:
                    display_df = format_dataframe(df)
                    st.dataframe(display_df, use_container_width=True)
                    st.caption(f"{len(df)} row(s) returned")
                    if chart != "none":
                        resolved_x = resolve_chart_col(chart_x, list(df.columns))
                        resolved_y = resolve_chart_col(chart_y, list(df.columns))
                        render_chart(df, chart, resolved_x, resolved_y,
                                     chart_color=chart_color,
                                     chart_title=chart_title,
                                     title_color=title_color)

        resolved_x_store = resolve_chart_col(chart_x, list(df.columns)) if df is not None and not df.empty else chart_x
        resolved_y_store = resolve_chart_col(chart_y, list(df.columns)) if df is not None and not df.empty else chart_y

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     summary,
            "summary":     summary,
            "sql":         sql,
            "df":          df.to_dict("records") if df is not None and not df.empty else None,
            "chart":       chart,
            "chart_x":     resolved_x_store,
            "chart_y":     resolved_y_store,
            "chart_color": chart_color,
            "title_color": title_color,
            "chart_title": chart_title,
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
    st.session_state.messages.append({"role": "user", "content": _new_prompt})
    st.session_state["_pending_prompt"] = _new_prompt
    st.rerun()

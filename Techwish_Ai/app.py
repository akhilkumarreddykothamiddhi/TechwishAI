import streamlit as st
import streamlit.components.v1 as st_components
import os, uuid, json, re as _re
import snowflake.connector
import pandas as pd
import plotly.express as px
from groq import Groq
import base64, pathlib, time

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
#  CSS — Poppins font  (UNCHANGED)
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

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    background: transparent;
    border-bottom: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 1.5rem;
}
.topbar h1 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800;
    font-size: 1.6rem;
    margin: 0;
    color: #1565C0;
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  DATABASE CONNECTION  — Snowflake
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_snowflake_conn(database: str = None):
    connect_kwargs = dict(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
    )
    if database:
        connect_kwargs["database"] = database
    if SNOWFLAKE_ROLE:
        connect_kwargs["role"] = SNOWFLAKE_ROLE
    return snowflake.connector.connect(**connect_kwargs)

@st.cache_data(show_spinner=False)
def list_databases() -> list[str]:
    try:
        conn = get_snowflake_conn()
        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        rows = cur.fetchall()
        system_dbs = {"SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"}
        dbs = [row[1] for row in rows if row[1] not in system_dbs]
        return sorted(dbs)
    except Exception as e:
        st.error(f"Could not list databases: {e}")
        return []

def run_query(sql: str, database: str) -> pd.DataFrame:
    conn = get_snowflake_conn(database)
    cur = conn.cursor()
    cur.execute(f'USE DATABASE "{database}"')
    cur.execute(sql)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)

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

# ─────────────────────────────────────────────────────────────────
#  HELPER UTILS
# ─────────────────────────────────────────────────────────────────
def is_chart_request(text: str) -> bool:
    return any(k in text.lower() for k in ["chart","graph","plot","visualize","line","bar","pie"])

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
#  NL → SQL  (Snowflake SQL dialect)
# ─────────────────────────────────────────────────────────────────
def nl_to_sql(question: str, history: list, database: str) -> dict:
    wl = build_whitelist(database)
    strict_block = whitelist_to_strict_block(wl)
    compact_block = "\n".join(
        f"  {tbl}: {', '.join(cols)}"
        for tbl, cols in wl.items()
    )
    # Capture last assistant response for follow-up context
    last_sql        = ""
    last_chart      = "none"
    last_chart_x    = ""
    last_chart_y    = ""
    last_chart_title= ""
    last_chart_color= None
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("sql"):
            last_sql         = m["sql"]
            last_chart       = m.get("chart", "none")
            last_chart_x     = m.get("chart_x", "")
            last_chart_y     = m.get("chart_y", "")
            last_chart_title = m.get("chart_title", "")
            last_chart_color = m.get("chart_color")
            break

    system_prompt = f"""You are an expert business intelligence assistant and strict Snowflake SQL query generator
for Snowflake database: "{database}".

════════════════════════════════════════════════════════
CRITICAL RULE — READ THIS FIRST, EVERY TIME
════════════════════════════════════════════════════════
You MUST use ONLY the exact table names and column names listed in STEP 2.
NEVER invent, guess, abbreviate, or assume any identifier.
Before writing SQL, do a mental PRE-FLIGHT CHECK:
  For every table name → confirm it exists EXACTLY in STEP 2.
  For every column name → confirm it exists EXACTLY in STEP 2.
  If even ONE identifier is not in the list → rewrite using only listed names.
  If no valid mapping exists → return sql="" and explain why clearly.

════════════════════════════════════════════════════════
STEP 1 — UNDERSTAND THE USER'S BUSINESS INTENT
════════════════════════════════════════════════════════
The user asks in plain English. Your job:
  1. Understand what business concept they care about.
  2. Find the BEST matching table(s) and column(s) from STEP 2.
  3. Write Snowflake SQL using ONLY exact names from STEP 2.

════════════════════════════════════════════════════════
STEP 2 — THE COMPLETE SCHEMA (EXACT NAMES, LIVE FROM DB)
════════════════════════════════════════════════════════
{strict_block}

ABSOLUTE IDENTIFIER RULES — NO EXCEPTIONS:
✗ NEVER use any table or column name not in the numbered list above.
✗ NEVER guess, abbreviate, pluralize, shorten, or invent names.
✗ NEVER use names like DOCTOR_KEY, STUDENT_ID, BRANCH_CODE, SCHOOL_KEY
  unless they appear EXACTLY in the numbered list above.
✓ Copy every identifier character-for-character from the numbered list.
✓ If no relevant table/column exists → sql="" with a clear explanation.

════════════════════════════════════════════════════════
STEP 3 — SNOWFLAKE SQL RULES
════════════════════════════════════════════════════════
- Snowflake SQL syntax only. Use LIMIT N (never TOP N).
- ⚠️ LIMIT RULE: Only add LIMIT if the user explicitly says "top N", "first N",
  "bottom N", or states a specific number like "show me 5". If the user asks a
  general question like "how many", "which branch", "show all", "per school" →
  do NOT add any LIMIT. Return all rows.
- Always alias tables (e.g. FROM STUDENTS s).
- Prefix all columns with table alias when joins exist.
- Always GROUP BY non-aggregated columns with COUNT/SUM/AVG/MIN/MAX.
- Use explicit column names. Never SELECT *.
- For date functions: DATETRUNC, DATE_TRUNC, DATEDIFF (Snowflake style).
- For JOINs: only join on columns that ACTUALLY EXIST in both tables per STEP 2.
- When user asks to use a NAME column instead of a KEY column → use the name
  column from the correct table, joining if necessary.

════════════════════════════════════════════════════════
STEP 4 — FOLLOW-UP HANDLING (VERY IMPORTANT)
════════════════════════════════════════════════════════
Last SQL in this conversation:
{last_sql if last_sql else "(none — this is the first query)"}

Last chart state: type={last_chart}, x={last_chart_x}, y={last_chart_y},
  title="{last_chart_title}", color={last_chart_color}

RULES for follow-up questions:
A) If user asks to change ONLY chart appearance (color, title, chart type) and
   the data does not need to change → keep the EXACT same SQL as last time,
   update only chart/chart_title fields in the JSON output.
B) If user asks to change the data (different column, filter, grouping) →
   write a new or modified SQL.
C) If user asks to change chart title color → that is a chart_color change.
   Return the same SQL, same chart type, same chart_x/y, update chart_color only.
D) If user says "change to line chart", "make it a bar chart" → same SQL,
   just update "chart" field.
E) If user says "change title to X" → same SQL, update chart_title only.
F) If user says "use school name instead of key" → modify SQL to select the
   name column (joining if needed), update chart_x to the name column.

════════════════════════════════════════════════════════
STEP 5 — CHART COLUMN NAMES
════════════════════════════════════════════════════════
- chart_x and chart_y MUST match the exact column alias used in your SELECT.
  Example: if SQL says "COUNT(*) AS STUDENT_COUNT", chart_y = "STUDENT_COUNT"
  Example: if SQL says "s.SCHOOL_NAME", chart_x = "SCHOOL_NAME"
- Never put a column in chart_x/y that is not in the SELECT output.

════════════════════════════════════════════════════════
OUTPUT — RAW JSON ONLY. NO MARKDOWN. NO CODE FENCES.
════════════════════════════════════════════════════════
{{
  "sql": "SELECT ...",
  "summary": "One sentence business insight in plain English",
  "chart": "bar|line|pie|scatter|area|histogram|none",
  "chart_x": "exact_output_column_name",
  "chart_y": "exact_output_column_name",
  "chart_title": "Short descriptive title for the chart",
  "chart_color": null
}}

Note: chart_color should be null unless user explicitly requests a color.
"""

    user_message = f"""════ VALID TABLES AND COLUMNS — USE ONLY THESE, COPY EXACTLY ════
{compact_block}
════════════════════════════════════════════════════════════════════

USER QUESTION: {question}

PRE-FLIGHT CHECK (mandatory before writing SQL):
1. Identify which table(s) from the list above best answer this question.
2. Identify which column(s) from those tables you need.
3. Verify EVERY identifier appears EXACTLY in the list above.
4. Only then write the SQL.

FOLLOW-UP DETECTION:
- Is this asking to change chart color/title/type only (no data change)?
  → Reuse last SQL exactly. Update only chart/chart_title/chart_color in JSON.
- Is this asking to change the data (columns, grouping, filters)?
  → Write new/modified SQL.
- Does the question mention a color (red, green, yellow, blue, etc.)?
  → Set chart_color to that color's hex value. Do NOT set it to null.

LIMIT RULE — CRITICAL:
- ONLY add LIMIT if user explicitly says "top N", "first N", "bottom N",
  or mentions a specific number like "show 5", "give me 10".
- Questions like "how many", "per school", "by branch", "show all", "which"
  → NO LIMIT. Return all rows.

Additional rules:
- chart_x and chart_y must match the EXACT alias/column name in your SELECT.
- Do not mention table names in the summary.
- If you cannot find matching columns → return sql="" with a clear explanation.
"""

    # ── FIX 4: Groq call with retry + rate-limit error handling ──
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
                # Strip markdown code fences
                raw = _re.sub(r"^```[a-z]*\n?", "", raw).strip("`").strip()
                # If model wrapped JSON in extra text, extract just the JSON object
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
                    wait_time = (attempt + 1) * 20  # 20s, 40s
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
        }

    sql        = result.get("sql", "").strip()
    user_color = extract_color_from_question(question)
    # Priority: explicit color extracted from question > AI-returned color > None
    result["chart_color"] = user_color or result.get("chart_color") or None

    is_valid, bad_cols = validate_sql_against_whitelist(sql, wl)
    if not is_valid and sql:
        bad_list = ", ".join(bad_cols)
        correction = f"""
⛔ SELF-CORRECTION REQUIRED:
Your SQL used these identifiers that DO NOT EXIST in the schema: [{bad_list}]

Go back to the numbered column list in the system prompt.
Find the EXACT correct column names for what the user asked.
Rewrite the query using ONLY those exact names.
If no valid columns exist → return sql as empty string "".
"""
        try:
            result = call_groq(extra_instruction=correction)
            result["chart_color"] = user_color
            sql = result.get("sql", "").strip()
            is_valid2, bad_cols2 = validate_sql_against_whitelist(sql, wl)
            if not is_valid2 and sql:
                result["sql"]     = ""
                result["summary"] = (
                    f"I couldn't generate a valid query — the columns needed "
                    f"({', '.join(bad_cols2)}) don't exist in the schema. "
                    "Please rephrase your question or check the schema in the sidebar."
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
#  CHART RENDERER
#  FIX 2: chart title color now follows chart_color when provided
# ─────────────────────────────────────────────────────────────────
def resolve_chart_col(col: str, df_columns: list) -> str:
    """
    Resolve a chart column name against actual DataFrame columns.
    Tries exact match first, then case-insensitive, then partial match.
    Returns the matched column name or the original if no match found.
    """
    if not col:
        return col
    # Exact match
    if col in df_columns:
        return col
    # Case-insensitive match
    col_lower = col.lower()
    for c in df_columns:
        if c.lower() == col_lower:
            return c
    # Partial match — column contains the hint or vice versa
    for c in df_columns:
        if col_lower in c.lower() or c.lower() in col_lower:
            return c
    return col  # fallback — render_chart will show a clean warning


def render_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: str,
    chart_color: str | None = None,
    chart_title: str = "",
):
    if chart_type == "none" or not x or not y:
        return
    if x not in df.columns or (y not in df.columns and chart_type not in ["histogram", "pie"]):
        st.warning(f"Chart columns '{x}' or '{y}' not found in results.")
        return

    single_color = chart_color if chart_color else DEFAULT_CHART_COLOR
    seq_colors   = [chart_color] + DEFAULT_BLUE_SEQUENCE if chart_color else DEFAULT_BLUE_SEQUENCE
    # FIX 2: title color = user-requested color, fallback to default blue
    title_color  = chart_color if chart_color else "#1565C0"

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
            fig = px.pie(df, names=x, values=y, hole=0.4, color_discrete_sequence=seq_colors, **common_kwargs)
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
                font=dict(family="Poppins", size=15, color=title_color),  # FIX 2 applied here
                x=0.02,
            ) if chart_title else {},
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render {chart_type} chart: {e}")

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
    LOGO_LIGHT_FILE = "techwish_black_transparent"
    LOGO_DARK_FILE  = "Techwish-Logo-white (3)"

    light_src = img_to_b64(LOGO_LIGHT_FILE)
    dark_src  = img_to_b64(LOGO_DARK_FILE)

    initial_src = light_src or dark_src

    if initial_src:
        # ── FIX 3: Robust Streamlit theme detection ──
        # Streamlit sets data-theme="dark"/"light" on <html> but may be slow.
        # We also check the computed background color of the sidebar as a reliable fallback.
        logo_js = f"""
<div id="logo-wrap" style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
  <img id="tw-logo" src="{initial_src}"
       style="max-width:150px; height:auto;" />
  <span style="background:#1565C0; color:white; font-family:'Poppins',sans-serif;
               font-weight:700; font-size:0.65rem; padding:2px 7px; border-radius:20px;
               letter-spacing:0.05em; line-height:1.4; vertical-align:middle;">AI</span>
</div>
<script>
(function() {{
  var light = {json.dumps(light_src)};
  var dark  = {json.dumps(dark_src)};

  function isDarkTheme() {{
    // Method 1: Streamlit's data-theme attribute on <html>
    var htmlTheme = document.documentElement.getAttribute("data-theme");
    if (htmlTheme === "dark") return true;
    if (htmlTheme === "light") return false;

    // Method 2: Check Streamlit's sidebar background color
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {{
      var bg = window.getComputedStyle(sidebar).backgroundColor;
      // Parse rgb values — dark background = low brightness
      var m = bg.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
      if (m) {{
        var brightness = (parseInt(m[1]) * 299 + parseInt(m[2]) * 587 + parseInt(m[3]) * 114) / 1000;
        return brightness < 128;
      }}
    }}

    // Method 3: OS preference as last resort
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }}

  function applyLogo() {{
    var el = document.getElementById("tw-logo");
    if (!el) return;
    var src = isDarkTheme() ? (dark || light) : (light || dark);
    if (src && el.src !== src) el.src = src;
  }}

  // Run immediately and on a short poll until Streamlit finishes rendering
  applyLogo();
  var attempts = 0;
  var interval = setInterval(function() {{
    applyLogo();
    attempts++;
    if (attempts > 30) clearInterval(interval);  // stop after ~15s
  }}, 500);

  // Also react to Streamlit theme toggle (attribute change on <html>)
  var observer = new MutationObserver(applyLogo);
  observer.observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme", "class"] }});

  // Also react to OS theme change
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyLogo);
}})();
</script>
"""
        # Use st_components.html so the <script> actually executes
        # instead of being rendered as visible text by st.markdown
        st_components.html(logo_js, height=50, scrolling=False)
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

    st.divider()
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
#  TOP BAR — title left, theme toggle right
# ─────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

_col_title, _col_spacer, _col_theme = st.columns([6, 2, 2])
with _col_title:
    st.markdown(f"""
<div style="display:flex; align-items:center; gap:15px; padding: 0.6rem 0 0.4rem 0;
            border-bottom: 1px solid rgba(128,128,128,0.2); margin-bottom: 0.5rem;">
    <span style="font-family:'Poppins',sans-serif; font-weight:800; font-size:1.5rem;
                 color:#1565C0;">📊 {selected_db or "Analytics"}</span>
    <span style="color:gray; font-size:0.9rem; font-family:'Poppins',sans-serif;">
        | Powered by Techwish AI</span>
</div>""", unsafe_allow_html=True)

with _col_theme:
    _theme_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
    if st.button(_theme_label, key="topbar_theme_btn", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        _new_base = "light" if not st.session_state.dark_mode else "dark"
        _bg       = "#FFFFFF" if _new_base == "light" else "#0E1117"
        _sbg      = "#F0F2F6" if _new_base == "light" else "#1A1D23"
        _txt      = "#31333F" if _new_base == "light" else "#FAFAFA"
        st_components.html(f"""
<script>
(function(){{
  var theme = {{
    "base": "{_new_base}",
    "primaryColor": "#1565C0",
    "backgroundColor": "{_bg}",
    "secondaryBackgroundColor": "{_sbg}",
    "textColor": "{_txt}",
    "font": "sans serif",
    "widgetBackgroundColor": "",
    "widgetBorderColor": "",
    "skeletonBackgroundColor": ""
  }};
  // Streamlit reads from this localStorage key to set the active theme
  var keys = Object.keys(localStorage).filter(function(k){{
    return k.startsWith('stActiveTheme');
  }});
  var key = keys.length > 0 ? keys[0] : 'stActiveTheme-/';
  localStorage.setItem(key, JSON.stringify(theme));
  // Fire storage event so Streamlit's theme listener picks it up
  window.dispatchEvent(new StorageEvent('storage', {{
    key: key,
    newValue: JSON.stringify(theme),
    storageArea: localStorage
  }}));
}})();
</script>
""", height=0, scrolling=False)
        st.rerun()

st.markdown("<div style='border-bottom:1px solid rgba(128,128,128,0.2); margin-bottom:1rem;'></div>",
            unsafe_allow_html=True)

if not selected_db:
    st.info("Please select a database from the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
#  CHAT HISTORY  (UNCHANGED)
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
                    st.dataframe(df, use_container_width=True)
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
                )

# ─────────────────────────────────────────────────────────────────
#  CHAT PROCESSING
#  USER BUBBLE FIX:
#  We use a two-phase pattern so the bubble always shows:
#    Phase A — append user msg + set _pending_prompt, then rerun.
#              The history loop above will render the user bubble.
#    Phase B — on next run, _pending_prompt exists → run AI + append
#              assistant msg + rerun so history loop renders it too.
# ─────────────────────────────────────────────────────────────────

# ── Phase B: pending prompt is ready, run AI now ──
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
        chart_title = result.get("chart_title", "")

        st.markdown(summary)

        df    = None
        error = None

        if sql:
            with st.expander("🔍 View SQL Query", expanded=False):
                st.markdown(
                    f'<div class="sql-block">{sql}</div>',
                    unsafe_allow_html=True,
                )
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
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(df)} row(s) returned")
                    if chart != "none":
                        resolved_x = resolve_chart_col(chart_x, list(df.columns))
                        resolved_y = resolve_chart_col(chart_y, list(df.columns))
                        render_chart(df, chart, resolved_x, resolved_y,
                                     chart_color=chart_color,
                                     chart_title=chart_title)

        # Store resolved column names so history replay also works
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
            "chart_title": chart_title,
        })

# ─────────────────────────────────────────────────────────────────
#  CHAT INPUT
#  Phase A: capture prompt → append user msg → store pending → rerun
#  This guarantees the history loop renders the user bubble first.
# ─────────────────────────────────────────────────────────────────

# Collect input from either the sidebar button or the chat box
_new_prompt = None

if "_inject_question" in st.session_state:
    _new_prompt = st.session_state.pop("_inject_question")
elif prompt := st.chat_input("Ask anything about your data..."):
    _new_prompt = prompt

if _new_prompt:
    # Phase A: save user message and set pending, then rerun
    st.session_state.messages.append({"role": "user", "content": _new_prompt})
    st.session_state["_pending_prompt"] = _new_prompt
    st.rerun()

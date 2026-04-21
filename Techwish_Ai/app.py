import streamlit as st
import os, uuid, json, re as _re
import snowflake.connector
import pandas as pd
import plotly.express as px
from groq import Groq
import base64, pathlib

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
def cfg(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

def _clean_account(raw: str) -> str:
    """
    Snowflake connector needs just the account identifier, e.g. 'TCFIWLF-SJ78956'.
    Strip any accidental '.snowflakecomputing.com' suffix the user might paste.
    """
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
#  CSS — Poppins font
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
        # Column index 1 = name, skip Snowflake system DBs
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
#  Looks for the file by EXACT name first (with extension),
#  then tries appending common image extensions.
#  Searches: script directory, then cwd.
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
    last_sql = ""
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("sql"):
            last_sql = m["sql"]
            break

    system_prompt = f"""You are an expert business intelligence assistant and strict Snowflake SQL query generator
for Snowflake database: "{database}".

════════════════════════════════════════════════════════
STEP 1 — UNDERSTAND THE USER'S BUSINESS INTENT
════════════════════════════════════════════════════════
The user asks questions in plain English without knowing table or column names.
YOUR JOB is to:
  1. Read the user's question and understand what business concept they care about.
  2. Scan the schema below to find the BEST matching table(s) and column(s).
  3. Write Snowflake SQL using ONLY the exact names found in the schema.

════════════════════════════════════════════════════════
STEP 2 — THE COMPLETE SCHEMA (EXACT NAMES, LIVE FROM DB)
════════════════════════════════════════════════════════
{strict_block}

ABSOLUTE IDENTIFIER RULES — NO EXCEPTIONS:
✗ Do NOT use any table or column name not listed above.
✗ Do NOT guess, abbreviate, pluralize, or invent names.
✓ Copy column names character-for-character from the numbered list.
✓ If no relevant table/column exists → set sql to "" and explain.

════════════════════════════════════════════════════════
STEP 3 — SNOWFLAKE SQL RULES
════════════════════════════════════════════════════════
- Snowflake SQL syntax only. Use LIMIT N, NOT TOP N.
- Column and table names are case-insensitive in Snowflake — use them as-is.
- Always alias tables (e.g. FROM CUSTOMERS c).
- Prefix all column references with alias when joins are present.
- Always GROUP BY non-aggregated columns when using COUNT/SUM/AVG/MIN/MAX.
- Use explicit column names. Never SELECT *.
- For "top N" → use ORDER BY col DESC LIMIT N.
- For date functions: use DATETRUNC, DATE_TRUNC, DATEDIFF (Snowflake style).
- For string functions: use IFF, COALESCE, ZEROIFNULL, NVL as needed.
- Use double quotes around identifiers ONLY if they contain special characters.

════════════════════════════════════════════════════════
STEP 4 — FOLLOW-UP QUERY HANDLING
════════════════════════════════════════════════════════
Last SQL in this conversation:
{last_sql if last_sql else "(none — this is the first query)"}

If the new question is a modification → edit ONLY the last SQL.
If it's a new topic → write a fresh query.

════════════════════════════════════════════════════════
STEP 5 — CHART TITLE
════════════════════════════════════════════════════════
- If the user explicitly mentions a chart title, extract that exact title.
- Otherwise, generate a short descriptive chart title.
- Always populate "chart_title" in the output JSON.

════════════════════════════════════════════════════════
OUTPUT — RAW JSON ONLY. NO MARKDOWN. NO CODE FENCES.
════════════════════════════════════════════════════════
{{
  "sql": "SELECT ...",
  "summary": "One sentence business insight in plain English",
  "chart": "bar|line|pie|scatter|area|histogram|none",
  "chart_x": "exact_column_name",
  "chart_y": "exact_column_name",
  "chart_title": "Short descriptive title for the chart"
}}
"""

    user_message = f"""AVAILABLE TABLES AND COLUMNS (copy names exactly):
{compact_block}

USER QUESTION: {question}

Instructions:
- Infer which table(s) best match the question's topic.
- Use ONLY the column names listed above.
- Do not mention table names in the summary.
- If user mentions a chart title, extract it. Otherwise write a short descriptive title.
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
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": sys_content}] + messages,
            temperature=0.0,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        raw = _re.sub(r"^```[a-z]*\n?", "", raw).strip("`").strip()
        return json.loads(raw)

    result = call_groq()
    sql    = result.get("sql", "").strip()
    user_color = extract_color_from_question(question)
    result["chart_color"] = user_color

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
        except Exception:
            result["sql"]     = ""
            result["summary"] = "Query generation failed after validation. Please try rephrasing."
            result["chart"]   = "none"

    return result

# ─────────────────────────────────────────────────────────────────
#  CHART RENDERER
# ─────────────────────────────────────────────────────────────────
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
                font=dict(family="Poppins", size=15, color="#1565C0"),
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
    # ── Logo filenames in your Git repo (no extension needed — auto-detected) ──
    # Light theme  → black/dark logo  (visible on white background)
    # Dark theme   → white logo       (visible on dark background)
    LOGO_LIGHT_FILE = "techwish_black_transparent"   # e.g. techwish_black_transparent.png
    LOGO_DARK_FILE  = "Techwish-Logo-white (3)"      # e.g. Techwish-Logo-white (3).png

    light_src = img_to_b64(LOGO_LIGHT_FILE)
    dark_src  = img_to_b64(LOGO_DARK_FILE)

    # Use whichever is available as the starting src; JS will swap on theme change
    initial_src = light_src or dark_src

    if initial_src:
        logo_js = f"""
<div id="logo-wrap" class="logo-row">
  <img id="tw-logo" src="{initial_src}"
       style="max-width:150px; height:auto;" />
  <span class="ai-badge">AI</span>
</div>
<script>
(function() {{
  var light = {json.dumps(light_src)};
  var dark  = {json.dumps(dark_src)};
  function applyLogo() {{
    var el = document.getElementById("tw-logo");
    if (!el) return;
    var theme = document.documentElement.getAttribute("data-theme") || "";
    var isDark = theme === "dark" ||
                 (theme === "" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    var src = isDark ? (dark || light) : (light || dark);
    if (src) el.src = src;
  }}
  applyLogo();
  var observer = new MutationObserver(applyLogo);
  observer.observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme"] }});
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyLogo);
}})();
</script>
"""
        st.markdown(logo_js, unsafe_allow_html=True)
    else:
        # Text fallback if neither logo file is found in the repo
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
#  TOP BAR
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div style="display:flex; align-items:center; gap:15px;">
        <h1>📊 {selected_db or "Analytics"}</h1>
        <span style="color:gray; font-size:0.9rem; font-family:Poppins,sans-serif;">| Powered by Techwish AI</span>
    </div>
    <div style="text-align:right;">
        <span style="font-size:0.8rem; color:gray; font-family:Poppins,sans-serif;">
            Database: {selected_db or "None selected"}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

if not selected_db:
    st.info("Please select a database from the sidebar to get started.")
    st.stop()

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
# ─────────────────────────────────────────────────────────────────
def process_question(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = nl_to_sql(prompt, st.session_state.messages, selected_db)

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
                        render_chart(df, chart, chart_x, chart_y,
                                     chart_color=chart_color,
                                     chart_title=chart_title)

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     summary,
            "summary":     summary,
            "sql":         sql,
            "df":          df.to_dict("records") if df is not None and not df.empty else None,
            "chart":       chart,
            "chart_x":     chart_x,
            "chart_y":     chart_y,
            "chart_color": chart_color,
            "chart_title": chart_title,
        })

# ─────────────────────────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────────────────────────
if "_inject_question" in st.session_state:
    injected = st.session_state.pop("_inject_question")
    process_question(injected)
elif prompt := st.chat_input("Ask anything about your data..."):
    process_question(prompt)

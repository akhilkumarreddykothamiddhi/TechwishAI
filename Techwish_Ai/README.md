# 📊 Techwish AI Analytics — Streamlit + Snowflake

Natural-language analytics chat app powered by **Groq LLM** + **Snowflake** + **Streamlit Cloud**.
Ask plain-English questions about your data and get SQL, tables, and charts instantly.

---

## 🗂️ Project Structure

```
techwish-analytics/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── .gitignore                ← Keeps secrets out of Git
├── .streamlit/
│   ├── config.toml           ← Streamlit UI config (safe to commit)
│   └── secrets.toml          ← 🔐 NEVER commit this file with real values
└── README.md
```

---

## 🚀 Deploy to Streamlit Cloud — Step by Step

### 1. Push to GitHub

```bash
git init
git add app.py requirements.txt .streamlit/config.toml .gitignore README.md
# ⚠️  Do NOT add .streamlit/secrets.toml — it has a placeholder only
git commit -m "Initial commit — Techwish AI Analytics"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Create a new app on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **"Advanced settings"** before deploying.

### 3. Add Secrets in Streamlit Cloud UI

In **Advanced settings → Secrets**, paste the following and fill in your real values:

```toml
SNOWFLAKE_ACCOUNT   = "your-org-your-account"
SNOWFLAKE_USER      = "your_username"
SNOWFLAKE_PASSWORD  = "your_password"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_ROLE      = ""
GROQ_API_KEY        = "gsk_xxxxxxxxxxxxxxxxxxxx"
```

> **Finding your Snowflake account identifier:**  
> Log in to Snowflake → Admin → Accounts. Your identifier is in the format `orgname-accountname`  
> (e.g. `mycompany-prod123`). It's also visible in your login URL:  
> `https://mycompany-prod123.snowflakecomputing.com`

5. Click **"Deploy!"**

---

## 🔑 Required Secrets Reference

| Secret | Description | Example |
|---|---|---|
| `SNOWFLAKE_ACCOUNT` | Account identifier (org-account format) | `acme-prod123` |
| `SNOWFLAKE_USER` | Snowflake username | `john_doe` |
| `SNOWFLAKE_PASSWORD` | Snowflake password | `MySecurePass!` |
| `SNOWFLAKE_WAREHOUSE` | Warehouse name (must exist) | `COMPUTE_WH` |
| `SNOWFLAKE_ROLE` | Optional role override (leave `""` for default) | `ANALYST` |
| `GROQ_API_KEY` | Groq API key from console.groq.com | `gsk_xxx...` |

---

## 🏃 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd techwish-analytics

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your secrets locally
#    Edit .streamlit/secrets.toml with your real credentials (already gitignored)

# 5. Run
streamlit run app.py
```

---

## ❄️ Snowflake Permissions Required

The Snowflake user needs:

```sql
-- Minimum required grants
GRANT USAGE ON WAREHOUSE <your_warehouse> TO ROLE <your_role>;
GRANT USAGE ON DATABASE <your_database> TO ROLE <your_role>;
GRANT USAGE ON ALL SCHEMAS IN DATABASE <your_database> TO ROLE <your_role>;
GRANT SELECT ON ALL TABLES IN DATABASE <your_database> TO ROLE <your_role>;

-- To allow listing all databases in the sidebar dropdown:
GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE TO ROLE <your_role>;
-- OR just grant USAGE on specific databases you want to expose
```

---

## 💡 How It Works

1. User types a plain-English question in the chat.
2. App loads the live schema from Snowflake's `INFORMATION_SCHEMA`.
3. Groq LLM (`llama-3.1-8b-instant`) translates the question → Snowflake SQL.
4. SQL is validated against the real schema (hallucination guard).
5. Query runs on Snowflake; results render as a table + optional chart.

---

## 🖼️ Logo Files (Optional)

To show your logo in the sidebar, place these files in the project root:
- `techwish_black_transparent.png` (for light theme)
- `Techwish-Logo-white (3).png` (for dark theme)

Then include them in your git commit:
```bash
git add techwish_black_transparent.png "Techwish-Logo-white (3).png"
```

If no logo files are found, the app falls back to a text logo automatically.

---

## 🔒 Security Notes

- **Never** commit `secrets.toml` with real credentials. It's in `.gitignore`.
- The app is read-only by design — the LLM is prompted to generate only `SELECT` queries.
- All identifiers are validated against the live schema before execution.

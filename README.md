# 🔬 Veritus — Autonomous AI Data Scientist
### *Automated Hypothesis Generation and Testing (AHGT) & Validation*

> **Statistical rigor meets Agentic AI.**  
> Veritus is an autonomous research framework for **Automated Hypothesis Generation and Testing** (often termed *Automated Hypothesis Generation and Validation* in scientific literature). Rather than merely describing datasets or hallucinating correlations, Veritus independently formulates falsifiable scientific hypotheses, subjects them to rigorous statistical tests ($p$-values, effect sizes), refines failed theories, and publishes only statistically validated discoveries with a full audit trail.

---

## ⚡ The Core Problem Veritus Solves

Current generative AI tools (like standard ChatGPT or Claude data plugins) frequently suffer from **statistical hallucination**:
* They generate convincing, authoritative-sounding claims about data correlations.
* They confuse correlation with causation, ignore sample sizes, and skip hypothesis testing.
* They lack a **self-correcting mechanism** when an initial intuition fails.

**Veritus operates like a human research scientist:**
1. **Hypothesize**: Explores data schema and distributions to identify non-obvious relationships.
2. **Experiment**: Runs genuine statistical tests using `scipy` and `statsmodels` (Two-sample T-Tests, Pearson/Spearman correlation, Chi-Square test of independence).
3. **Refine**: If a broad hypothesis fails ($p \ge 0.05$), the agent dissects the failure and generates a more targeted, conditional hypothesis.
4. **Audit & Report**: Logs every test, $p$-value, effect size, and verdict into PostgreSQL, presenting a transparent reasoning trail in an interactive Streamlit UI.

---

## 🔄 Agentic Architecture Loop

```
                     ┌───────────────────────────┐
                     │   User Uploads CSV Data   │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  AI Agent (Ollama/LLM)    │
                     │ Proposes 5-10 Hypotheses │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │     Statistics Engine     │
                     │  (t-test, chi^2, corr)    │
                     └──────┬─────────────┬──────┘
                            │             │
                 Pass (p < 0.05)       Fail (p >= 0.05)
                            │             │
                            ▼             ▼
              ┌──────────────────┐  ┌─────────────────────────┐
              │ Validated Result │  │ LLM Refinement Loop     │
              │  Explanation     │  │ (Forms targeted sub-    │
              └────────┬─────────┘  │  hypotheses & re-tests) │
                       │            └─────────────┬───────────┘
                       │                          │
                       ▼                          ▼
       ┌────────────────────────────────────────────────────────┐
       │   PostgreSQL Database (Audit Log & Provenance Tree)    │
       │   - Datasets, Hypotheses, P-values, Effect Sizes       │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │        Streamlit Frontend Interactive Dashboard        │
       │  (Validated Findings, Rejection Graveyard, Tree View)  │
       └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack ($0 Free Tier Architecture)

| Component | Technology | Role & Justification |
|---|---|---|
| **Language** | Python 3.11 | High performance data science and agent scripting ecosystem |
| **Statistical Engine** | `scipy`, `statsmodels` | Real parametric & non-parametric scientific testing |
| **Data Processing** | `pandas`, `numpy` | In-memory data wrangling, cleaning, and slicing |
| **Local LLM** | Ollama (`llama3` / `mistral`) | 100% free, unlimited, private offline inference |
| **Cloud LLM (Deploy)** | Groq / Gemini Flash | Ultra-fast free-tier cloud inference for deployment |
| **Database** | PostgreSQL 16 | Relational storage for provenance, hypotheses, and runs |
| **Frontend** | Streamlit + Plotly | Reactive UI for file upload, metric cards, and charts |
| **Containerization** | Docker & Docker Compose | Uniform local dev environment and production reproducibility |

---

## 📂 Project Structure

```
Veritus-io/
│
├── backend/
│   ├── __init__.py
│   ├── main.py             # Orchestrates generation -> testing -> refinement -> DB persistence
│   ├── db.py               # PostgreSQL connection pool & relational schema definition
│   ├── stats_engine.py     # Rigorous hypothesis testing suite (t-test, correlation, chi2)
│   ├── ai_agent.py         # LLM interface: hypothesis formulation, refinement, explanation
│   └── requirements.txt    # Production Python dependencies
│
├── frontend/
│   ├── __init__.py
│   └── app.py              # Streamlit dashboard: CSV upload, visual cards, reasoning trail
│
├── Dockerfile              # Production multi-stage container build
├── docker-compose.yml      # Local orchestration (PostgreSQL 16 + Streamlit app)
├── .env                    # Secrets & connection strings (Local / Supabase / Ollama)
├── .env.example            # Environment template for local and production deployment
├── .gitignore              # Protects secrets, cache, and DB volume data
└── README.md               # Project documentation
```

---

## 🚀 Quickstart & Usage

### 1. Prerequisites
* **Python 3.11+** installed
* **Docker Desktop** installed (for PostgreSQL container)
* **Ollama** installed (optional for local LLM inference; [ollama.com](https://ollama.com))

### 2. Start PostgreSQL via Docker
To start the database in the background:
```bash
docker compose up -d postgres
```
This launches a PostgreSQL 16 container exposed at `localhost:5432` with automatic health checks and persistent volume storage in `pgdata`.

### 3. Local Python Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt
```

### 4. Running the Ollama Model Locally
Pull and run the model of your choice:
```bash
ollama run llama3
```

### 5. Launch the Veritus Web Dashboard
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Database Schema

```sql
-- Track uploaded datasets
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

-- Track all hypotheses and test outcomes (full lineage tree)
CREATE TABLE hypotheses (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    hypothesis_text TEXT NOT NULL,
    parent_hypothesis_id INTEGER REFERENCES hypotheses(id), -- For refined/re-tested versions
    test_method TEXT,                                       -- 't-test' | 'correlation' | 'chi-square'
    p_value FLOAT,
    effect_size FLOAT,
    verdict TEXT,                                           -- 'validated' | 'rejected'
    explanation TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---


## 📚 Academic & Research Grounding (AHGT)

Veritus is grounded in the emerging academic field of **Automated Hypothesis Generation and Validation (AHGV)** and Agentic Data Science:
* **HypoGeniC / Hypogenic**: Pioneered automated hypothesis generation and validation from observational data distributions.
* **DS-Agent, BLADE, DiscoveryBench**: Leading academic benchmarks evaluating LLMs on autonomous end-to-end data analysis workflows.
* **OpenScientist**: Open-source autonomous scientific discovery platforms for biomedical and life-science domains.

### 🌟 How Veritus Stands Apart:
While academic systems often focus on complex domain-specific architectures or benchmark evaluation, **Veritus delivers a pragmatic, general-purpose system for everyday tabular and business datasets**. It bridges LLM reasoning with classic frequentist statistical testing (`scipy`, `statsmodels`) in an accessible, open-source web application with zero running costs ($0 stack).

---
## 🎯 Verification & Evaluation Criteria
* **Statistical Validity**: Every claimed finding strictly satisfies $p < 0.05$ with reported effect sizes (e.g., Cohen's $d$, Pearson $r$, Cramér's $V$).
* **Falsification Transparency**: Users can view hypotheses that were rejected, preventing publication bias.
* **Lineage Tracking**: Hierarchical parent-child linking reveals how initial failed theories evolved into validated conditional findings.

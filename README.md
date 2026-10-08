# 🔬 Veritus — Autonomous AI Data Scientist
### *Automated Hypothesis Generation and Testing (AHGT) & Scientific Discovery Engine*

**Veritus** is an autonomous AI data analyst that ingests tabular datasets (CSV), formulates falsifiable scientific hypotheses via an LLM agent, executes rigorous frequentist statistical tests ($p$-values and effect sizes), and autonomously refines failed propositions through an iterative self-correction loop. Rather than hallucinating correlations, Veritus logs complete statistical audit trails into PostgreSQL and surfaces validated findings and rejected theories within a modern, reactive React dashboard backed by a high-performance FastAPI REST service.

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   React 18 + Vite UI                   │
│        (Dataset Preview, Live Polling, Results)        │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP REST / JSON
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│        (Pipeline Orchestration, REST Endpoints)        │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
       SQL / RealDictCursor     HTTP / JSON Prompts
                │                        │
                ▼                        ▼
┌──────────────────────────────┐ ┌───────────────────────┐
│     PostgreSQL Database      │ │   Ollama (Local) /    │
│  (Datasets, Hypotheses Tree, │ │     Groq / Gemini     │
│   Statistical Provenance)    │ │   (Hypothesis Engine) │
└──────────────────────────────┘ └───────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technology | Role & Purpose |
|---|---|---|
| **Frontend** | React 18, Vite | Reactive, modular component architecture with instant HMR |
| **Styling** | Vanilla CSS (Custom Design System) | High-performance styling, responsive layouts, gradient typography, and custom status badges |
| **Backend API** | FastAPI, Uvicorn | Async REST endpoints, background task workers, and OpenAPI docs |
| **Statistical Engine** | `scipy`, `statsmodels` | Frequentist tests: Two-sample $t$-test, ANOVA, Mann-Whitney U, Pearson/Spearman correlation, Chi-Square |
| **Data Processing** | `pandas`, `numpy` | Tabular data profiling, variance checks, and candidate column filtering |
| **LLM Inference** | Ollama (`llama3`), Groq, Gemini Flash | Autonomous scientific reasoning, hypothesis formulation, refinement, and plain-English synthesis |
| **Database** | PostgreSQL 16 | Relational audit log storing dataset runs, hypothesis lineage, $p$-values, and effect sizes |
| **Containerization**| Docker, Docker Compose | Multi-container local orchestration (Postgres + FastAPI + React/Nginx) |

---

## 📂 Project Structure

```text
ai-data-analyst/
├── backend/
│   ├── main.py                 # Core AHGT pipeline orchestrator (generate -> test -> refine)
│   ├── db.py                   # PostgreSQL connection pool and CRUD queries
│   ├── stats_engine.py         # Hypothesis testing engine with frequentist statistical tests
│   ├── ai_agent.py             # LLM agent (dataset profiling, hypothesis generation, refinement)
│   ├── api.py                  # FastAPI REST API exposing endpoints & live progress tracking
│   └── requirements.txt        # Backend dependencies (fastapi, uvicorn, scipy, pandas, etc.)
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx            # React root application mount
│   │   ├── App.jsx             # Main application layout, state coordination, and tabs
│   │   ├── api/
│   │   │   └── client.js       # Fetch client wrapper for FastAPI backend
│   │   ├── components/
│   │   │   ├── Sidebar.jsx         # Service status, CSV upload, agent sliders, and history list
│   │   │   ├── DatasetPreview.jsx  # Interactive tabular preview with row/col/null metrics
│   │   │   ├── RunAnalysisButton.jsx # Interactive action button with loader state
│   │   │   ├── ProgressTracker.jsx # Live progress bar and pipeline step messages
│   │   │   ├── ResultsTabs.jsx     # Validated vs. Rejected tabbed findings container
│   │   │   ├── FindingCard.jsx     # Individual hypothesis cards with badges and explanations
│   │   │   └── SystemStatusBadge.jsx # Ollama and PostgreSQL connectivity pills
│   │   ├── styles/
│   │   │   └── index.css       # Complete design system tokens, typography, and card themes
│   │   └── hooks/
│   │       └── useAnalysis.js  # Custom React hook managing analysis state & status polling
│   ├── index.html              # HTML entry with Inter & JetBrains Mono fonts
│   ├── nginx.conf              # Nginx reverse proxy configuration for production container
│   ├── package.json            # Frontend dependencies and build scripts
│   ├── vite.config.js          # Vite configuration
│   └── .env                    # VITE_API_URL=http://localhost:8000
│
├── Dockerfile.backend          # Builds the FastAPI Python container
├── Dockerfile.frontend         # Multi-stage build (Node build -> Nginx static serving)
├── docker-compose.yml          # 3-tier local service orchestration (postgres + backend + frontend)
├── .env                        # Local backend environment configuration
├── .env.example                # Example environment file template
└── README.md                   # Project documentation
```

---

## 🔄 The AHGT (Automated Hypothesis Generation and Testing) Loop

Veritus operates on a continuous, self-correcting 5-step scientific loop:

1. **Profile**: Computes statistical summaries of the uploaded dataset (data types, distributions, non-null counts, unique values) to identify high-signal candidate variable pairs.
2. **Formulate**: Prompts the LLM agent to propose falsifiable scientific propositions (e.g., *"Customers in Region A spend significantly more than Region B"*), automatically mapping each to an appropriate statistical test method (t-test, ANOVA, correlation, Chi-Square).
3. **Test**: Executes frequentist statistical calculations against the chosen columns using `scipy.stats` and `statsmodels`. Evaluates whether the test meets the significance threshold ($p < 0.05$) and extracts standardized effect sizes ($r$, Cohen's $d$, $\eta^2$, Cramér's $V$).
4. **Refine (Self-Correction)**: If an initial broad hypothesis fails ($p \ge 0.05$), the self-correction engine feeds the failed result back into the LLM to formulate a more specific, conditional hypothesis (e.g., controlling for a third variable or segmenting the sample) and re-tests it.
5. **Report & Lineage**: Persists parent-child hypothesis lineage and statistical metrics into PostgreSQL, generating plain-English executive explanations for validated discoveries and filing rejected hypotheses in an open audit graveyard.

---

## 💻 Local Development Setup

### 1. Prerequisites
- **Docker Desktop** installed and running
- **Ollama** installed on your host machine ([ollama.com](https://ollama.com)) with a model pulled:
  ```bash
  ollama pull llama3
  # or
  ollama pull mistral
  ```

### 2. Environment Configuration (`.env`)
Create a `.env` file in the root directory (based on `.env.example`):
```env
# PostgreSQL Configuration (Local Docker)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ai_data_analyst
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_data_analyst

# LLM Configuration (Ollama running on host)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=120

# Optional Cloud LLM Fallbacks (for cloud deployment or if not using local Ollama)
GROQ_API_KEY=
GEMINI_API_KEY=

# CORS Configuration
FRONTEND_ORIGIN=http://localhost:5173
```

### 3. Run with Docker Compose
Start all three services (`postgres`, `backend`, `frontend`) with one command:
```bash
docker compose up --build
```

### 4. Access URLs
- **React Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🌐 Production Deployment Guide

### 1. Database → Supabase (Free Cloud PostgreSQL)
1. Create a free account at [supabase.com](https://supabase.com) and create a new project.
2. In Project Settings → **Database**, locate the **Connection String** (`URI` mode).
3. Copy the URL string (format: `postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`).
4. Veritus will automatically run `init_db()` upon boot and create the required `datasets` and `hypotheses` tables.

### 2. Backend → Render (Docker Web Service)
1. Push your repository to GitHub.
2. In Render, select **New +** → **Web Service** → Connect your repository.
3. Configure the service:
   - **Environment**: Docker
   - **Dockerfile Path**: `Dockerfile.backend`
   - **Docker Context**: `.`
4. Set Environment Variables in Render:
   - `DATABASE_URL`: Your Supabase PostgreSQL connection string.
   - `GROQ_API_KEY` or `GEMINI_API_KEY`: Free cloud LLM API key for hypothesis generation.
   - `FRONTEND_ORIGIN`: Your deployed frontend URL (e.g., `https://your-app.vercel.app`).
   - `PORT`: `8000`

### 3. Frontend → Vercel or Netlify
1. Import the repository into Vercel or Netlify.
2. Configure build settings:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. Add Environment Variable:
   - `VITE_API_URL`: Your deployed Render backend URL (e.g., `https://veritus-backend.onrender.com`).

---

## 🎓 Academic Context & Related Work

> **Academic Project Note:**  
> Veritus was engineered as a final-year undergraduate capstone project exploring the synthesis of **agentic autonomous reasoning** and **formal frequentist statistics**.

Veritus builds upon and addresses gaps identified in current academic research on autonomous discovery and data science agents:

- **HypoGeniC / Hypogenic** (*Park et al., 2024*): Formalized the paradigm of generating and validating hypotheses from raw observational data distributions. Veritus adopts this principle and equips it with an explicit self-correcting refinement loop.
- **DS-Agent** (*Guo et al., 2024*): Evaluates end-to-end data science agents on iterative code and modeling benchmarks. Veritus mirrors this iterative problem-solving pattern specifically for exploratory data discovery.
- **DiscoveryBench** (*Bhatt et al., 2024*): Assesses LLMs on hypothesis formulation against empirical data, demonstrating that unconstrained generative models regularly succumb to statistical hallucinations without deterministic testing guardrails.

By enforcing rigid statistical testing ($p < 0.05$ thresholding and effect size calculation) before any claim is accepted as valid, Veritus prevents AI hallucination and demonstrates a verifiable approach to autonomous data science.

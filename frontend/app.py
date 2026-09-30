"""
Veritus — AI Data Scientist Web Dashboard (frontend/app.py)
Interactive Streamlit interface for Automated Hypothesis Generation and Testing (AHGT).
"""
import html as _html
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.db import init_db, list_datasets, get_hypotheses_by_dataset, get_dataset
from backend.main import run_analysis_pipeline
from backend.ai_agent import is_ollama_available

st.set_page_config(
    page_title="Veritus — Autonomous AI Data Scientist",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# DESIGN SYSTEM (all colors explicit so cards look the same in light & dark theme)
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---------- Page layout (stable selectors, full width) ---------- */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important; padding-bottom: 3rem !important;
    padding-left: 2rem !important; padding-right: 2rem !important;
    max-width: 100% !important;
}
[data-testid="stTabs"],
[data-baseweb="tab-panel"] { width: 100% !important; }

/* ---------- Header ---------- */
.vt-title {
    font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin: 0 0 .25rem 0;
    background: linear-gradient(90deg, #4F46E5, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.vt-subtitle { font-size: 1rem; color: #64748B; margin: 0 0 1.5rem 0; }

/* ---------- Metrics: equal-size tiles ---------- */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 14px 16px; min-height: 92px;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
[data-testid="stMetricLabel"] p { color: #64748B !important; font-size: .82rem; font-weight: 600; }
[data-testid="stMetricValue"] { color: #0F172A !important; font-size: 1.7rem; font-weight: 700; }

/* ---------- Hero / feature grid ---------- */
.vt-hero {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
    border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.5rem 1.75rem; margin-bottom: 1.25rem;
}
.vt-hero h3 { margin: 0 0 .5rem 0; color: #0F172A; font-size: 1.25rem; }
.vt-hero p  { margin: 0; color: #475569; font-size: 1rem; line-height: 1.6; }
.vt-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 0 0 1.25rem 0; }
@media (max-width: 900px) { .vt-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 520px) { .vt-grid { grid-template-columns: 1fr; } }
.vt-feature {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem 1.1rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
.vt-feature .n { display:inline-block; width: 26px; height: 26px; line-height: 26px; text-align:center;
    border-radius: 50%; background:#E0E7FF; color:#3730A3; font-weight:700; font-size:.8rem; margin-bottom:.5rem; }
.vt-feature .t { color:#0F172A; font-weight:700; font-size:1rem; margin-bottom:.25rem; }
.vt-feature .d { color:#64748B; font-size:.87rem; line-height:1.45; }

/* ---------- Result cards ---------- */
.vt-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-left-width: 6px;
    border-radius: 12px; padding: 1rem 1.25rem; margin: 0 0 .6rem 0;
    box-shadow: 0 1px 3px rgba(15,23,42,.06);
}
.vt-card--valid   { border-left-color: #22C55E; background: #F0FDF4; }
.vt-card--refined { border-left-color: #A855F7; background: #FAF5FF; }
.vt-card--reject  { border-left-color: #EF4444; background: #FEF2F2; }
.vt-card-title { color: #0F172A; font-size: 1.05rem; font-weight: 700; line-height: 1.4; margin: 0 0 .6rem 0; }
.vt-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: .75rem; }
.vt-badge { display: inline-block; font-size: .78rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.vt-b-method  { background: #E0E7FF; color: #3730A3; }
.vt-b-ok      { background: #DCFCE7; color: #166534; }
.vt-b-fail    { background: #FEE2E2; color: #991B1B; }
.vt-b-effect  { background: #FEF3C7; color: #92400E; }
.vt-b-refined { background: #EDE9FE; color: #6D28D9; }
.vt-note {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px;
    padding: .75rem .9rem; color: #334155; font-size: .92rem; line-height: 1.55;
}
.vt-note b { color: #0F172A; }
.vt-child {
    margin: 0 0 .9rem 1.75rem; padding: .7rem .95rem; background: #F8FAFC;
    border: 1px solid #E2E8F0; border-left: 3px dashed #94A3B8; border-radius: 8px;
    color: #334155; font-size: .9rem; line-height: 1.5;
}
.vt-child.ok  { border-left-color: #22C55E; }
.vt-child.bad { border-left-color: #EF4444; }
.vt-child .res { display:block; margin-top:.25rem; font-weight:600; font-size:.83rem; }
.vt-child.ok  .res { color:#166534; }
.vt-child.bad .res { color:#991B1B; }
.vt-spacer { height: .9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def render_html(block: str):
    """Strip indentation and blank lines so Markdown doesn't treat HTML as a code block."""
    cleaned = "\n".join(line.strip() for line in block.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def fmt(v, spec: str = ".4g") -> str:
    """Safe number formatting: returns 'N/A' for None / non-numbers instead of crashing."""
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else "N/A"


def esc(v) -> str:
    return _html.escape(str(v)) if v is not None else ""


def result_card(kind: str, icon: str, title: str, badges: str, note_label: str, note_text: str):
    """kind: valid | refined | reject"""
    render_html(
        f"""
        <div class="vt-card vt-card--{kind}">
        <div class="vt-card-title">{icon} {esc(title)}</div>
        <div class="vt-badges">{badges}</div>
        <div class="vt-note"><b>{note_label}</b><br>{esc(note_text)}</div>
        </div>
        """
    )


# ------------------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------------------
for key, default in {
    "df": None,
    "dataset_name": None,
    "analysis_results": None,
    "is_running": False,
    "last_uploaded_file": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def get_demo_dataset() -> pd.DataFrame:
    """Generates synthetic e-commerce data with known statistical properties."""
    np.random.seed(42)
    n = 180
    ad_spend = np.random.uniform(50, 600, n)
    sales = 3.5 * ad_spend + np.random.normal(0, 120, n)
    tier = np.random.choice(["Premium", "Basic"], size=n, p=[0.45, 0.55])
    satisfaction = np.where(
        tier == "Premium", np.random.normal(8.7, 0.8, n), np.random.normal(6.3, 1.2, n)
    )
    device = np.random.choice(["Desktop", "Mobile"], size=n)
    converted = np.where(
        device == "Desktop",
        np.random.choice(["Yes", "No"], size=n, p=[0.65, 0.35]),
        np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70]),
    )
    random_noise = np.random.normal(50, 15, n)

    return pd.DataFrame(
        {
            "ad_spend": np.round(ad_spend, 2),
            "sales_volume": np.round(sales, 2),
            "customer_tier": tier,
            "satisfaction_score": np.round(satisfaction, 2),
            "device_type": device,
            "converted": converted,
            "noise_metric": np.round(random_noise, 2),
        }
    )


def render_hypothesis_chart(df: Optional[pd.DataFrame], hypo: Dict[str, Any], key: str = "chart"):
    """Renders a Plotly chart (inside an expander so the card list stays even)."""
    if df is None:
        return
    col1 = hypo.get("col1")
    col2 = hypo.get("col2")
    if not col1 or not col2 or col1 not in df.columns or col2 not in df.columns:
        return

    try:
        is_num1 = pd.api.types.is_numeric_dtype(df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(df[col2])
        template = "plotly_white"

        if is_num1 and is_num2:
            try:
                fig = px.scatter(df, x=col1, y=col2, trendline="ols",
                                 color_discrete_sequence=["#4F46E5"], template=template)
            except Exception:  # statsmodels missing
                fig = px.scatter(df, x=col1, y=col2,
                                 color_discrete_sequence=["#4F46E5"], template=template)
        elif is_num1 != is_num2:
            cat_col = col1 if not is_num1 else col2
            num_col = col2 if not is_num1 else col1
            fig = px.box(df, x=cat_col, y=num_col, color=cat_col, points="all", template=template)
            fig.update_layout(showlegend=False)
        else:
            ct = pd.crosstab(df[col1], df[col2]).reset_index()
            ct_melted = ct.melt(id_vars=[col1], var_name=col2, value_name="Count")
            fig = px.bar(ct_melted, x=col1, y="Count", color=col2, barmode="group", template=template)

        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        with st.expander(f"📈 View chart: {col1} vs {col2}"):
            st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.caption(f"Chart note: {e}")


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    ollama_up = is_ollama_available()
    st.markdown(
        f"**LLM Engine:** {'🟢 Ollama' if ollama_up else '🟡 Heuristic / Fallback'}<br>"
        f"**Database:** 🟢 PostgreSQL",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 📂 Dataset Input")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("🧪 Load Demo", use_container_width=True):
            st.session_state.df = get_demo_dataset()
            st.session_state.dataset_name = "synthetic_ecommerce_demo.csv"
            st.session_state.analysis_results = None
            st.rerun()

    with col_demo2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.df = None
            st.session_state.dataset_name = None
            st.session_state.analysis_results = None
            st.session_state.last_uploaded_file = None
            st.rerun()

    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_file") != uploaded_file.name:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.analysis_results = None
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    st.markdown("---")
    st.markdown("### 🎛️ Agent Configuration")
    num_hypotheses = st.slider(
        "Hypotheses to Formulate", min_value=3, max_value=8, value=5,
        help="Number of initial hypotheses formulated by the AI.",
    )
    enable_refinement = st.checkbox(
        "Self-Correcting Refinement Loop", value=True,
        help="When a hypothesis fails testing, the agent refines it and re-tests.",
    )

    st.markdown("---")
    st.markdown("### 📜 Past Experiment Runs")
    try:
        init_db()
        history = list_datasets()
        if history:
            history_options = {
                f"{h['name']} (ID: {h['id']}) - {h['validated_count']} validated": h["id"]
                for h in history
            }
            selected_history_label = st.selectbox("Inspect Past Run", list(history_options.keys()))
            if st.button("📂 Load Past Run Results", use_container_width=True):
                selected_id = history_options[selected_history_label]
                ds_info = get_dataset(selected_id)
                db_hypos = get_hypotheses_by_dataset(selected_id)

                validated_from_db = [h for h in db_hypos if h.get("verdict") == "validated"]
                rejected_from_db = [h for h in db_hypos if h.get("verdict") == "rejected"]

                st.session_state.dataset_name = (
                    ds_info["name"] if ds_info else f"Dataset #{selected_id}"
                )
                st.session_state.analysis_results = {
                    "dataset_id": selected_id,
                    "dataset_name": st.session_state.dataset_name,
                    "total_hypotheses": len(db_hypos),
                    "validated_count": len(validated_from_db),
                    "rejected_count": len(rejected_from_db),
                    "validated": validated_from_db,
                    "rejected": rejected_from_db,
                    "all_hypotheses": db_hypos,
                }
                st.rerun()
        else:
            st.caption("No past analysis runs found in database.")
    except Exception as e:
        st.caption(f"History unavailable: {e}")

# ==============================================================================
# MAIN VIEW
# ==============================================================================
st.markdown('<div class="vt-title">🔬 Veritus — Autonomous AI Data Scientist</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="vt-subtitle">Automated hypothesis generation, rigorous statistical testing '
    "&amp; self-correcting discovery</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# SCENARIO 1: NO DATASET AND NO RESULTS
# ------------------------------------------------------------------------------
if st.session_state.df is None and st.session_state.analysis_results is None:
    render_html(
        """
        <div class="vt-hero">
        <h3>👋 Welcome to Veritus</h3>
        <p>Veritus works like an autonomous scientific researcher. It examines your dataset schema,
        generates falsifiable hypotheses, tests them with frequentist statistics (t-tests, correlations,
        chi-square), and refines failed theories into narrower discoveries.</p>
        </div>
        <div class="vt-grid">
        <div class="vt-feature"><div class="n">1</div><div class="t">Formulate</div>
        <div class="d">The AI agent profiles your data and drafts falsifiable hypotheses.</div></div>
        <div class="vt-feature"><div class="n">2</div><div class="t">Test</div>
        <div class="d">SciPy runs hypothesis tests with p-values and effect sizes.</div></div>
        <div class="vt-feature"><div class="n">3</div><div class="t">Refine</div>
        <div class="d">A self-correction loop turns failures into narrower child hypotheses.</div></div>
        <div class="vt-feature"><div class="n">4</div><div class="t">Audit</div>
        <div class="d">The full provenance tree is persisted into PostgreSQL.</div></div>
        </div>
        """
    )

    st.info("💡 **Get started:** upload a CSV in the sidebar, or load the synthetic demo data below.")

    if st.button("🚀 Load Synthetic Demo Dataset & Try Now", type="primary", use_container_width=True):
        st.session_state.df = get_demo_dataset()
        st.session_state.dataset_name = "synthetic_ecommerce_demo.csv"
        st.session_state.last_uploaded_file = None
        st.session_state.analysis_results = None
        st.rerun()

# ------------------------------------------------------------------------------
# SCENARIO 2: DATASET LOADED -> RUN CONTROLS
# ------------------------------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    ds_name = st.session_state.dataset_name or "Uploaded Dataset"

    num_rows = len(df)
    num_cols = len(df.columns)
    num_numeric = len(df.select_dtypes(include=[np.number]).columns)
    num_categorical = num_cols - num_numeric

    st.markdown(f"**Dataset:** `{ds_name}`")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{num_rows:,}")
    m2.metric("Total Columns", num_cols)
    m3.metric("Numeric Features", num_numeric)
    m4.metric("Categorical Features", num_categorical)

    st.markdown('<div class="vt-spacer"></div>', unsafe_allow_html=True)

    with st.expander("🔍 View Dataset Preview & Schema", expanded=False):
        tab_sample, tab_stats = st.tabs(["Data Sample", "Descriptive Statistics"])
        with tab_sample:
            st.dataframe(df.head(10), use_container_width=True)
        with tab_stats:
            num_df = df.select_dtypes(include=[np.number])
            cat_df = df.select_dtypes(exclude=[np.number])
            if not num_df.empty:
                st.markdown("**Numerical metrics**")
                st.dataframe(num_df.describe(), use_container_width=True)
            if not cat_df.empty:
                st.markdown("**Categorical features**")
                st.dataframe(cat_df.describe().astype(str), use_container_width=True)

    st.markdown("### 🚀 Autonomous Discovery Pipeline")
    st.markdown(
        f"Veritus will formulate **{num_hypotheses} hypotheses**, run statistical tests, "
        f"{'refine failed ones, ' if enable_refinement else ''}"
        f"and record the audit trail to PostgreSQL."
    )

    run_btn = st.button(
        "🔬 Run Autonomous Discovery Experiment", type="primary", use_container_width=True
    )

    if run_btn:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def on_pipeline_progress(msg: str, frac: float):
            progress_bar.progress(min(max(float(frac), 0.0), 1.0))
            status_box.info(f"⏳ **Active step:** {msg}")

        try:
            with st.spinner("Investigating the dataset..."):
                results = run_analysis_pipeline(
                    df=df,
                    dataset_name=ds_name,
                    num_initial_hypotheses=num_hypotheses,
                    enable_refinement=enable_refinement,
                    progress_callback=on_pipeline_progress,
                )
                st.session_state.analysis_results = results
                progress_bar.progress(1.0)
                status_box.success("🎉 Experiment completed! Discoveries are shown below.")
        except Exception as e:
            status_box.error(f"❌ Error during experiment execution: {e}")
            st.exception(e)

# ------------------------------------------------------------------------------
# SCENARIO 3: RESULTS VISUALIZATION (renders whenever results exist)
# ------------------------------------------------------------------------------
if st.session_state.analysis_results is not None:
    res = st.session_state.analysis_results
    validated_items = res.get("validated", [])
    rejected_items = res.get("rejected", [])
    all_hypotheses = res.get("all_hypotheses", [])
    plot_df = st.session_state.df  # may be None for past runs

    refined_count = sum(
        1
        for h in all_hypotheses
        if h.get("is_refined") or h.get("parent_hypothesis_id") or h.get("parent_id")
    )

    st.markdown("---")
    st.markdown("### 📊 Experiment Discovery Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Hypotheses", len(all_hypotheses))
    k2.metric("Validated", len(validated_items))
    k3.metric("Rejected", len(rejected_items))
    k4.metric("Refinements", refined_count)

    st.markdown('<div class="vt-spacer"></div>', unsafe_allow_html=True)

    tab_valid, tab_reject, tab_audit = st.tabs(
        [
            f"🏆 Validated ({len(validated_items)})",
            f"⚰️ Rejection Graveyard ({len(rejected_items)})",
            f"📜 Audit Log ({len(all_hypotheses)})",
        ]
    )

    # ---- TAB 1: VALIDATED ----
    with tab_valid:
        if validated_items:
            for idx, item in enumerate(validated_items):
                parent_ref = item.get("parent_id") or item.get("parent_hypothesis_id")
                is_child = bool(item.get("is_refined") or parent_ref)
                badges = (
                    f'<span class="vt-badge vt-b-method">{esc(item.get("test_method", "N/A"))}</span>'
                    f'<span class="vt-badge vt-b-ok">p = {fmt(item.get("p_value"))}</span>'
                    f'<span class="vt-badge vt-b-effect">{esc(item.get("effect_size_name") or "Effect")}: '
                    f'{fmt(item.get("effect_size"), ".3f")}</span>'
                )
                if is_child:
                    badges += f'<span class="vt-badge vt-b-refined">✨ Refined from #{parent_ref}</span>'
                explanation = (
                    item.get("explanation")
                    or item.get("summary")
                    or "Statistically significant relationship detected."
                )
                result_card(
                    "refined" if is_child else "valid",
                    "✅",
                    item.get("hypothesis_text"),
                    badges,
                    "🧠 AI Scientific Interpretation",
                    explanation,
                )
                render_hypothesis_chart(plot_df, item, key=f"val_chart_{idx}_{item.get('id')}")
                st.markdown('<div class="vt-spacer"></div>', unsafe_allow_html=True)
        else:
            st.info("No hypotheses reached significance (p < 0.05). Check the Graveyard tab.")

    # ---- TAB 2: REJECTED ----
    with tab_reject:
        st.caption(
            "Honest science: reporting rejected hypotheses prevents confirmation bias. "
            "These failed to reach significance (p ≥ 0.05)."
        )
        if rejected_items:
            for item in rejected_items:
                reason = (
                    item.get("summary")
                    or item.get("explanation")
                    or "Failed to reject the null hypothesis at alpha = 0.05."
                )
                badges = (
                    f'<span class="vt-badge vt-b-method">{esc(item.get("test_method", "N/A"))}</span>'
                    f'<span class="vt-badge vt-b-fail">p = {fmt(item.get("p_value"))}</span>'
                    f'<span class="vt-badge vt-b-effect">Effect: {fmt(item.get("effect_size"), ".3f")}</span>'
                )
                result_card(
                    "reject", "❌", item.get("hypothesis_text"), badges,
                    "Statistical reason", reason,
                )
                children = [
                    h
                    for h in all_hypotheses
                    if item.get("id") is not None
                    and (
                        h.get("parent_id") == item.get("id")
                        or h.get("parent_hypothesis_id") == item.get("id")
                    )
                ]
                for child in children:
                    is_val = child.get("verdict") == "validated"
                    render_html(
                        f"""
                        <div class="vt-child {'ok' if is_val else 'bad'}">
                        <b>↳ Refinement (ID #{child.get('id')}):</b> {esc(child.get('hypothesis_text'))}
                        <span class="res">{esc(str(child.get('verdict', '')).upper())} ·
                        p = {fmt(child.get('p_value'))} · effect = {fmt(child.get('effect_size'), '.3f')}</span>
                        </div>
                        """
                    )
                st.markdown('<div class="vt-spacer"></div>', unsafe_allow_html=True)
        else:
            st.info("No hypotheses were rejected in this run.")

    # ---- TAB 3: AUDIT ----
    with tab_audit:
        st.caption("Complete provenance trail registered in PostgreSQL.")
        audit_df = pd.DataFrame(
            [
                {
                    "ID": h.get("id"),
                    "Parent ID": h.get("parent_id") or h.get("parent_hypothesis_id") or "None",
                    "Verdict": str(h.get("verdict", "")).upper(),
                    "Hypothesis": h.get("hypothesis_text"),
                    "Method": h.get("test_method"),
                    "p-value": h.get("p_value"),
                    "Effect Size": h.get("effect_size"),
                    "Scientific Explanation": h.get("explanation"),
                }
                for h in all_hypotheses
            ]
        )
        st.dataframe(audit_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=json.dumps(res, default=str, indent=2),
                file_name=f"veritus_experiment_{res.get('dataset_id', 'export')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "📥 Download Audit Table (CSV)",
                data=audit_df.to_csv(index=False).encode("utf-8"),
                file_name=f"veritus_audit_{res.get('dataset_id', 'export')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
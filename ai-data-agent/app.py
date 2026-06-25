import streamlit as st
import pandas as pd
from modules.ingestion import load_dataset
from modules.cleaning import clean_data
from modules.eda import run_eda
from modules.visualization import generate_charts
from modules.task_detector import detect_task
from modules.model_trainer import train_models
from modules.aggregator import aggregate_results
from modules.llm_client import get_llm_summary, ask_followup

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide", page_icon="🤖")

st.title("🤖 AI Data Analysis Agent")
st.markdown("Upload your dataset and let the AI agent analyze, train models, and explain everything.")

# ── Session state ──────────────────────────────────────────────────────────────
for key in ["df", "clean_df", "eda_stats", "charts", "model_results",
            "aggregated", "summary", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "chat_history" not in st.session_state or st.session_state.chat_history is None:
    st.session_state.chat_history = []

# ── STEP 1: Upload ─────────────────────────────────────────────────────────────
st.header("📁 Step 1 — Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    with st.spinner("Ingesting dataset..."):
        df, schema = load_dataset(uploaded_file)
    st.session_state.df = df
    st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📋 Schema / Column Info"):
        st.dataframe(pd.DataFrame(schema), use_container_width=True)

# ── STEP 2: Clean ─────────────────────────────────────────────────────────────
if st.session_state.df is not None:
    st.header("🧹 Step 2 — Cleaning & Preprocessing")
    if st.button("Run Cleaning"):
        with st.spinner("Cleaning data..."):
            clean_df, cleaning_log = clean_data(st.session_state.df)
        st.session_state.clean_df = clean_df
        st.success("✅ Cleaning complete")
        with st.expander("🔍 Cleaning Log"):
            for entry in cleaning_log:
                st.write(f"- {entry}")
        st.dataframe(clean_df.head(10), use_container_width=True)

# ── STEP 3: EDA ───────────────────────────────────────────────────────────────
if st.session_state.clean_df is not None:
    st.header("📊 Step 3 — Exploratory Data Analysis")
    if st.button("Run EDA"):
        with st.spinner("Running EDA..."):
            eda_stats = run_eda(st.session_state.clean_df)
        st.session_state.eda_stats = eda_stats
        st.success("✅ EDA complete")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Descriptive Stats")
            st.dataframe(eda_stats["describe"], use_container_width=True)
        with col2:
            st.subheader("Correlation Matrix")
            st.dataframe(eda_stats["correlation"], use_container_width=True)

        st.subheader("Missing Values")
        st.dataframe(pd.DataFrame(eda_stats["missing"], index=["Missing Count"]).T, use_container_width=True)

# ── STEP 4: Visualization ─────────────────────────────────────────────────────
if st.session_state.eda_stats is not None:
    st.header("📈 Step 4 — Visualization Engine")
    if st.button("Generate Charts"):
        with st.spinner("Generating charts..."):
            charts = generate_charts(st.session_state.clean_df, st.session_state.eda_stats)
        st.session_state.charts = charts
        st.success(f"✅ Generated {len(charts)} charts")

        cols = st.columns(2)
        for i, (title, img) in enumerate(charts.items()):
            with cols[i % 2]:
                st.image(img, caption=title, use_container_width=True)

# ── STEP 5: ML ────────────────────────────────────────────────────────────────
if st.session_state.clean_df is not None:
    st.header("🤖 Step 5 — Model Training & Evaluation")
    target_col = st.selectbox("Select target column (leave blank for clustering)",
                              ["(None — clustering)"] + list(st.session_state.clean_df.columns))

    if st.button("Train Models"):
        target = None if target_col == "(None — clustering)" else target_col
        with st.spinner("Detecting task type and training models..."):
            task = detect_task(st.session_state.clean_df, target)
            results = train_models(st.session_state.clean_df, target, task)
        st.session_state.model_results = results
        st.success(f"✅ Task detected: **{task}** — Best model: **{results['best_model']}**")

        st.subheader("Model Comparison")
        st.dataframe(pd.DataFrame(results["scores"]), use_container_width=True)

        if results.get("feature_importance") is not None:
            st.subheader("Feature Importance")
            st.dataframe(results["feature_importance"], use_container_width=True)

# ── STEP 6: LLM Summary ───────────────────────────────────────────────────────
if st.session_state.model_results is not None:
    st.header("🧠 Step 6 — AI Insights (Ollama LLM)")
    if st.button("Generate AI Summary & Insights"):
        with st.spinner("Asking the local LLM to summarize findings..."):
            aggregated = aggregate_results(
                st.session_state.eda_stats,
                st.session_state.model_results,
                st.session_state.clean_df
            )
            st.session_state.aggregated = aggregated
            summary = get_llm_summary(aggregated)
        st.session_state.summary = summary
        st.success("✅ AI summary ready")
        st.markdown("### 📝 AI Summary")
        st.markdown(summary)

# ── STEP 7: Q&A ───────────────────────────────────────────────────────────────
if st.session_state.summary is not None:
    st.header("💬 Step 7 — Ask Follow-up Questions")
    st.markdown("Ask anything about your data, the model, or the results.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about your data...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_followup(
                    user_q,
                    st.session_state.aggregated,
                    st.session_state.chat_history
                )
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from pipeline.ingest import normalize_any_csv
from pipeline.clean import clean_feedback
from pipeline.embed import embed_texts
from pipeline.cluster import cluster_embeddings
from pipeline.label import label_clusters_with_llm
from pipeline.insights import add_week_bucket, theme_metrics, exec_summary_and_actions

# Setup
load_dotenv()

st.set_page_config(page_title="Customer Feedback Intelligence", layout="wide")
st.title("Customer Feedback Intelligence System")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY not found. Add it to your .env file to enable embeddings + labeling + summaries.")

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    k_topics = st.slider("Number of themes (topics)", 4, 15, 8)
    run_llm_labeling = st.toggle("Label themes with LLM", value=True)
    run_exec_summary = st.toggle("Generate executive summary", value=True)
    st.divider()
    st.caption("Upload ANY CSV. Only the text column is required.")

# Session state
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

if "analyze" not in st.session_state:
    st.session_state.analyze = False

# Step 1 — Choose data source
st.subheader("Step 1 — Choose your data")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Use sample dataset"):
        st.session_state.use_sample = True
        st.session_state.analyze = False  # reset analysis when switching dataset

with col2:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

# If user uploads, override sample mode
if uploaded is not None:
    st.session_state.use_sample = False
    st.session_state.analyze = False  # reset analysis when switching dataset

# Load raw data
if st.session_state.use_sample:
    try:
        raw_df = pd.read_csv("data/sample_feedback.csv")
    except FileNotFoundError:
        st.error("Could not find data/sample_feedback.csv. Make sure the file exists in the data/ folder.")
        st.stop()
elif uploaded is not None:
    raw_df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to begin, or click **Use sample dataset**.")
    st.stop()

st.write("Detected columns:", list(raw_df.columns))
st.dataframe(raw_df.head(10), use_container_width=True)

# Step 2 — Column mapping
st.subheader("Step 2 — Map your columns")

cols = list(raw_df.columns)

# Required mapping
text_col = st.selectbox("Which column contains the feedback text? (required)", options=cols)

# Optional mappings
date_col = st.selectbox("Date column (optional)", options=["(none)"] + cols)
rating_col = st.selectbox("Rating column (optional)", options=["(none)"] + cols)
source_col = st.selectbox("Source column (optional)", options=["(none)"] + cols)

# Step 3 — Normalize + Analyze
st.subheader("Step 3 — Analyze")

if st.button("Normalize and Analyze"):
    st.session_state.analyze = True

if not st.session_state.analyze:
    st.info("Click **Normalize and Analyze** to run theme detection and dashboards.")
    st.stop()

# Normalize into standard schema
normalized = normalize_any_csv(
    raw_df,
    text_col=text_col,
    date_col=None if date_col == "(none)" else date_col,
    rating_col=None if rating_col == "(none)" else rating_col,
    source_col=None if source_col == "(none)" else source_col,
)

df = clean_feedback(normalized)

if df.empty:
    st.error("No valid rows after cleaning.")
    st.stop()

# Sentiment (rating-based if exists; otherwise unknown)
df["sentiment"] = df["rating"].apply(
    lambda r: "unknown"
    if pd.isna(r)
    else ("negative" if r <= 2 else ("neutral" if r == 3 else "positive"))
)

st.success("Normalized dataset created")
st.write("Standardized columns:", list(df.columns))
st.dataframe(df.head(20), use_container_width=True)

# Theme modeling
st.subheader("Theme Modeling")

with st.spinner("Generating embeddings + clustering..."):
    embeddings = embed_texts(df["text"].tolist())
    df["cluster_id"] = cluster_embeddings(embeddings, k=k_topics)

cluster_to_theme = {int(c): f"Theme {int(c)}" for c in sorted(df["cluster_id"].unique())}

if run_llm_labeling and os.getenv("OPENAI_API_KEY"):
    with st.spinner("Labeling themes with LLM..."):
        cluster_to_theme = label_clusters_with_llm(df)

df["theme"] = df["cluster_id"].map(cluster_to_theme)

# Dashboard
st.subheader("Dashboard")

left, right = st.columns([1, 1])

with left:
    st.markdown("### Sentiment Distribution")
    st.plotly_chart(px.histogram(df, x="sentiment"), use_container_width=True)

with right:
    st.markdown("### Top Themes (Priority)")
    metrics = theme_metrics(df)
    st.dataframe(metrics, use_container_width=True)

# Trends
st.subheader("Trends")

df2 = add_week_bucket(df)
trend = df2.groupby(["week", "theme"], as_index=False).size()
st.plotly_chart(
    px.line(trend, x="week", y="size", color="theme", markers=True),
    use_container_width=True
)

# Theme Deep Dive
st.subheader("Theme Deep Dive")

theme_selected = st.selectbox("Select a theme", sorted(df["theme"].unique().tolist()))
slice_df = df[df["theme"] == theme_selected].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Feedback count", len(slice_df))

if slice_df["rating"].isna().all():
    c2.metric("Avg rating", "N/A")
    c3.metric("Negative % (rating ≤ 2)", "N/A")
else:
    c2.metric("Avg rating", round(slice_df["rating"].mean(), 2))
    c3.metric("Negative % (rating ≤ 2)", f"{(slice_df['rating'].le(2).mean() * 100):.0f}%")

st.write("Example comments")
show_cols = [c for c in ["date", "source", "rating", "sentiment", "text"] if c in slice_df.columns]
st.dataframe(slice_df[show_cols].head(30), use_container_width=True)

# Executive Summary
if run_exec_summary and os.getenv("OPENAI_API_KEY"):
    st.subheader("Executive Summary + Recommended Actions")
    with st.spinner("Generating executive summary..."):
        summary = exec_summary_and_actions(df, top_n=5)
    st.text_area("Copy into a doc / slide", summary, height=280)

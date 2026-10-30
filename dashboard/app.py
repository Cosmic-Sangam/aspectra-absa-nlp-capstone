"""Streamlit alternative dashboard for ABSA."""
import json
import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ABSA Dashboard", layout="wide", page_icon="🔍")

st.title("Aspect-Based Sentiment Analysis (ABSA)")
st.markdown("Fine-grained aspect extraction and sentiment classification with explainability.")

# Sidebar
st.sidebar.header("Configuration")
domain = st.sidebar.selectbox("Domain", ["restaurant", "laptop"], index=0)
model = st.sidebar.selectbox("Model", ["distilbert", "bilstm"], index=0,
                             format_func=lambda x: "DistilBERT (Recommended)" if x == "distilbert" else "BiLSTM (Baseline)")
confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
enable_explain = st.sidebar.checkbox("Token Explainability", value=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Processing", "Evaluation Benchmarks"])

with tab1:
    st.subheader("Single Review Analysis")
    sample_text = (
        "The pizza was delicious and crispy, but the service was extremely slow."
        if domain == "restaurant" else
        "The battery life is exceptional and the display is bright, but the keyboard feels cheap."
    )
    user_input = st.text_area("Review Text", value=sample_text, height=120)

    if st.button("Analyze Review", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(f"{API_URL}/predict", json={
                    "review": user_input,
                    "model": model,
                    "domain": domain,
                    "explain": enable_explain,
                    "confidence_threshold": confidence_thresh,
                }, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Analysis completed in {data['latency_ms']} ms using {model.upper()}")
                    aspects = data.get("aspects", [])
                    if not aspects:
                        st.info("No aspects detected above confidence threshold.")
                    else:
                        for asp in aspects:
                            sent = asp["sentiment"]
                            color = "green" if sent == "positive" else ("red" if sent == "negative" else "gray")
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"### **{asp['aspect'].title()}**")
                                with col2:
                                    st.markdown(f":{color}[**{sent.upper()}**] ({asp['confidence']*100:.1f}%)")
                                st.progress(asp["confidence"])
                                if asp.get("explanation") and "top_influential_words" in asp["explanation"]:
                                    st.markdown("**Influential Words:**")
                                    words = [f"`{w['word']}` ({w['importance']:+.2f})" for w in asp["explanation"]["top_influential_words"]]
                                    st.markdown(" • ".join(words))
                                st.divider()
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server at {API_URL}: {e}")

with tab2:
    st.subheader("Batch Review Processing")
    batch_text = st.text_area(
        "Enter reviews (one per line)",
        value="The pasta was fantastic.\nTerrible customer service and rude waiters.\nAverage coffee, nice atmosphere.",
        height=150
    )
    if st.button("Process Batch"):
        lines = [line.strip() for line in batch_text.split("\n") if line.strip()]
        if lines:
            try:
                res = requests.post(f"{API_URL}/predict/batch", json={
                    "reviews": lines,
                    "model": model,
                    "domain": domain,
                    "confidence_threshold": confidence_thresh,
                }, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    st.info(f"Analyzed {data['total_reviews']} reviews in {data['latency_ms']} ms")
                    rows = []
                    for r in data["results"]:
                        asp_str = "; ".join([f"{a['aspect']} ({a['sentiment']})" for a in r["aspects"]]) or "None"
                        rows.append({"Review": r["review"], "Extracted Aspects": asp_str})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"API Error: {e}")

with tab3:
    st.subheader("Model Performance Comparison (Test Sets)")
    try:
        with open("results/metrics/overall_summary.json") as f:
            metrics = json.load(f)
        for dom, dom_data in metrics.items():
            st.markdown(f"#### Domain: {dom.capitalize()}")
            table = []
            for m_name, m_metrics in dom_data.items():
                table.append({
                    "Model": m_name.upper(),
                    "Extraction Exact F1": f"{m_metrics['extraction_exact_f1']*100:.2f}%",
                    "Extraction Token F1": f"{m_metrics['extraction_token_f1']*100:.2f}%",
                    "Sentiment Accuracy": f"{m_metrics['sentiment_accuracy']*100:.2f}%",
                    "Sentiment Macro-F1": f"{m_metrics['sentiment_macro_f1']*100:.2f}%",
                })
            st.table(pd.DataFrame(table))
    except Exception as e:
        st.warning(f"Could not load metrics summary: {e}")

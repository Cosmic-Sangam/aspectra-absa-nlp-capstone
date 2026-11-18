"""Aspectra — Production Streamlit Dashboard (Native Python ABSA Studio).

Runs entirely in Python without requiring a separate Node.js or external API process.
Features identical Stripe-style styling, color palettes, Leave-One-Out XAI, and batch analytics.
"""
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Directly import core inference components from src
from src.inference.pipeline import ABSAPipeline
from src.inference.explainability import get_explanation

st.set_page_config(
    page_title="Aspectra — ABSA Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Mirroring Aspectra Stripe Design Palette)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #635bff 0%, #4f46e5 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #475569;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-pos {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .badge-neg {
        background-color: #fff1f2;
        color: #9f1239;
        border: 1px solid #fecdd3;
    }
    
    .badge-neu {
        background-color: #f8fafc;
        color: #334155;
        border: 1px solid #cbd5e1;
    }

    .aspect-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(50, 50, 93, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .aspect-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(50, 50, 93, 0.08);
    }
    
    .xai-tag {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .xai-supp {
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .xai-opp {
        background: #fff1f2;
        color: #b91c1c;
        border: 1px solid #fecdd3;
    }

    .stat-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0a2540;
    }
    .stat-lbl {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Pipeline Loader (Lazy Initialization)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLP Checkpoints...")
def load_absa_pipeline(model_type: str, domain: str) -> ABSAPipeline:
    return ABSAPipeline(model_type=model_type, domain=domain)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **Aspectra Studio**")
    st.markdown("<div style='font-size: 0.85rem; color: #64748b; margin-top:-10px; margin-bottom: 20px;'>Fine-Grained Sentiment Engineering</div>", unsafe_allow_html=True)
    
    domain_choice = st.selectbox(
        "Target Domain",
        ["restaurant", "laptop"],
        format_func=lambda x: "🍽️ Restaurant Reviews" if x == "restaurant" else "💻 Laptop Reviews"
    )
    
    model_choice = st.selectbox(
        "Model Architecture",
        ["distilbert", "bilstm"],
        format_func=lambda x: "✨ DistilBERT (Transformer - SOTA)" if x == "distilbert" else "⚡ BiLSTM (Bidirectional Recurrent)"
    )
    
    confidence_slider = st.slider(
        "Confidence Threshold Filter",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Filters out low-confidence predictions. Default is 30% (0.30) to eliminate random hallucinations."
    )
    
    enable_xai = st.toggle("Token Explainability (Leave-One-Out XAI)", value=True)
    
    st.markdown("---")
    st.markdown("""
    **Core Metrics Snapshot:**
    - **DistilBERT Rest F1:** `86.07%`
    - **DistilBERT Laptop F1:** `81.29%`
    - **BiLSTM Rest F1:** `77.62%`
    - **BiLSTM Laptop F1:** `67.00%`
    """)
    st.markdown("<div style='font-size: 0.75rem; color: #94a3b8;'>Trained on SemEval-2014 Task 4 Benchmark</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown('<div class="main-header">Aspectra — ABSA Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fine-grained Aspect Extraction, Sentiment Classification & Token Perturbation Explainability</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <span class="badge-pill badge-neu">Domain: {domain_choice.upper()}</span><br>
        <span class="badge-pill badge-neu" style="margin-top: 5px;">Arch: {model_choice.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab_single, tab_batch, tab_benchmarks, tab_info = st.tabs([
    "🔍 Single Review Inference",
    "📊 Batch Analysis",
    "📈 Benchmark Matrix",
    "💡 How ABSA & XAI Works"
])

# ---------------------------------------------------------
# Tab 1: Single Review Inference
# ---------------------------------------------------------
with tab_single:
    default_text = (
        "The margherita pizza was delicious and crispy, but the waiter was extremely rude and the bill was overpriced."
        if domain_choice == "restaurant" else
        "The battery life is exceptional and the OLED screen is gorgeous, but the cooling fan gets loud under heavy load."
    )
    
    review_input = st.text_area(
        "Enter Review Text:",
        value=default_text,
        height=110,
        placeholder="Type or paste any customer review..."
    )
    
    col_btn, col_empty = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("Analyze Review", type="primary", use_container_width=True)

    if run_btn and review_input.strip():
        with st.spinner("Analyzing aspects and running inference..."):
            pipeline = load_absa_pipeline(model_choice, domain_choice)
            pipeline.threshold = confidence_slider
            
            t_start = time.time()
            result = pipeline.analyze(review_input.strip())
            latency = (time.time() - t_start) * 1000
            
            aspects = result.get("aspects", [])
            
            # Quick summary stats
            st.markdown("#### **Analysis Results**")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="stat-box"><div class="stat-val">{len(aspects)}</div><div class="stat-lbl">Aspects Found</div></div>', unsafe_allow_html=True)
            with m2:
                pos_count = sum(1 for a in aspects if a["sentiment"].lower() == "positive")
                st.markdown(f'<div class="stat-box"><div class="stat-val" style="color: #166534;">{pos_count}</div><div class="stat-lbl">Positive</div></div>', unsafe_allow_html=True)
            with m3:
                neg_count = sum(1 for a in aspects if a["sentiment"].lower() == "negative")
                st.markdown(f'<div class="stat-box"><div class="stat-val" style="color: #9f1239;">{neg_count}</div><div class="stat-lbl">Negative</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="stat-box"><div class="stat-val">{latency:.1f}ms</div><div class="stat-lbl">Inference Latency</div></div>', unsafe_allow_html=True)
            
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            if not aspects:
                st.warning(f"No aspects met the current confidence threshold of {confidence_slider*100:.0f}%. Try lowering the slider in the sidebar.")
            else:
                for idx, asp in enumerate(aspects):
                    sent = asp["sentiment"].lower()
                    conf = asp["confidence"]
                    badge_class = "badge-pos" if sent == "positive" else ("badge-neg" if sent == "negative" else "badge-neu")
                    
                    st.markdown(f"""
                    <div class="aspect-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1.25rem; font-weight: 700; color: #0a2540;">📌 {asp['aspect'].title()}</span>
                            <span class="badge-pill {badge_class}">{sent.upper()} • {conf*100:.1f}% CONFIDENCE</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(conf)
                    
                    # Leave-One-Out Token Perturbation Explainability
                    if enable_xai:
                        with st.spinner(f"Computing Leave-One-Out XAI for '{asp['aspect']}'..."):
                            explanation = get_explanation(pipeline.predictor, review_input.strip(), asp["aspect"], top_k=5)
                            top_words = explanation.get("top_influential_words", [])
                            
                            st.markdown("<div style='margin-top: 10px; font-size: 0.88rem; font-weight: 600; color: #475569;'>Token Attribution (Leave-One-Out Drop):</div>", unsafe_allow_html=True)
                            
                            chips = []
                            for w in top_words:
                                imp = w["importance"]
                                tag_style = "xai-supp" if imp > 0 else "xai-opp"
                                sign = "+" if imp > 0 else ""
                                chips.append(f'<span class="xai-tag {tag_style}">{w["word"]} ({sign}{imp*100:.1f}%)</span>')
                            
                            st.markdown("".join(chips), unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Batch Analysis
# ---------------------------------------------------------
with tab_batch:
    st.markdown("#### **High-Throughput Batch Review Evaluation**")
    st.markdown("Analyze multiple reviews at once. Paste one review per line below:")
    
    sample_batch = (
        "The pasta was fantastic and fresh.\n"
        "Terrible customer service and rude waiters.\n"
        "Average coffee, but the ambiance was cozy.\n"
        "The wait time was over 45 minutes, not worth it."
        if domain_choice == "restaurant" else
        "The battery life easily lasts 12 hours.\n"
        "Overpriced laptop with a disappointing trackpad.\n"
        "The screen resolution is stunning for 4K video editing.\n"
        "Keyboard typing experience is mushy and slow."
    )
    
    batch_input = st.text_area("Batch Input (One review per line):", value=sample_batch, height=150)
    
    if st.button("Run Batch Inference", type="primary"):
        lines = [line.strip() for line in batch_input.split("\n") if line.strip()]
        if lines:
            with st.spinner(f"Processing {len(lines)} reviews with {model_choice.upper()}..."):
                pipeline = load_absa_pipeline(model_choice, domain_choice)
                pipeline.threshold = confidence_slider
                
                t_b_start = time.time()
                batch_results = pipeline.analyze_batch(lines)
                b_latency = (time.time() - t_b_start) * 1000
                
                st.success(f"Processed {len(lines)} reviews in {b_latency:.1f}ms ({b_latency/len(lines):.1f}ms/review)")
                
                rows = []
                for res in batch_results:
                    asps = res.get("aspects", [])
                    asp_formatted = ", ".join([f"{a['aspect']} [{a['sentiment'].upper()}]" for a in asps]) or "None detected"
                    rows.append({
                        "Review": res["review"],
                        "Aspect Count": len(asps),
                        "Extracted Aspects & Sentiments": asp_formatted
                    })
                
                df_batch = pd.DataFrame(rows)
                st.dataframe(df_batch, use_container_width=True)
                
                # CSV Export
                csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Results as CSV",
                    data=csv,
                    file_name=f"aspectra_{domain_choice}_{model_choice}_batch_results.csv",
                    mime="text/csv"
                )

# ---------------------------------------------------------
# Tab 3: Benchmark Matrix
# ---------------------------------------------------------
with tab_benchmarks:
    st.markdown("#### **Empirical Evaluation on SemEval-2014 Task 4 Test Sets**")
    st.markdown("Strict character-level evaluation with exact match span matching and macro-F1 sentiment scoring.")
    
    summary_path = Path("results/metrics/overall_summary.json")
    if summary_path.exists():
        with open(summary_path, "r") as f:
            metrics_data = json.load(f)
            
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### 🍽️ **Restaurant Domain (Gold Benchmark)**")
            rest_m = metrics_data.get("restaurant", {})
            r_rows = [
                {
                    "Model": "DistilBERT (Fine-Tuned)",
                    "Exact F1": f"{rest_m['distilbert']['extraction_exact_f1']*100:.2f}%",
                    "Token F1": f"{rest_m['distilbert']['extraction_token_f1']*100:.2f}%",
                    "Sent Accuracy": f"{rest_m['distilbert']['sentiment_accuracy']*100:.2f}%",
                    "Macro-F1": f"{rest_m['distilbert']['sentiment_macro_f1']*100:.2f}%"
                },
                {
                    "Model": "BiLSTM (Baseline)",
                    "Exact F1": f"{rest_m['bilstm']['extraction_exact_f1']*100:.2f}%",
                    "Token F1": f"{rest_m['bilstm']['extraction_token_f1']*100:.2f}%",
                    "Sent Accuracy": f"{rest_m['bilstm']['sentiment_accuracy']*100:.2f}%",
                    "Macro-F1": f"{rest_m['bilstm']['sentiment_macro_f1']*100:.2f}%"
                }
            ]
            st.table(pd.DataFrame(r_rows))
            
        with col_m2:
            st.markdown("##### 💻 **Laptop Domain (Gold Benchmark)**")
            lap_m = metrics_data.get("laptop", {})
            l_rows = [
                {
                    "Model": "DistilBERT (Fine-Tuned)",
                    "Exact F1": f"{lap_m['distilbert']['extraction_exact_f1']*100:.2f}%",
                    "Token F1": f"{lap_m['distilbert']['extraction_token_f1']*100:.2f}%",
                    "Sent Accuracy": f"{lap_m['distilbert']['sentiment_accuracy']*100:.2f}%",
                    "Macro-F1": f"{lap_m['distilbert']['sentiment_macro_f1']*100:.2f}%"
                },
                {
                    "Model": "BiLSTM (Baseline)",
                    "Exact F1": f"{lap_m['bilstm']['extraction_exact_f1']*100:.2f}%",
                    "Token F1": f"{lap_m['bilstm']['extraction_token_f1']*100:.2f}%",
                    "Sent Accuracy": f"{lap_m['bilstm']['sentiment_accuracy']*100:.2f}%",
                    "Macro-F1": f"{lap_m['bilstm']['sentiment_macro_f1']*100:.2f}%"
                }
            ]
            st.table(pd.DataFrame(l_rows))
    else:
        st.info("Metrics summary file found at results/metrics/overall_summary.json")

# ---------------------------------------------------------
# Tab 4: How ABSA & XAI Works
# ---------------------------------------------------------
with tab_info:
    st.markdown("### **Architectural Breakdown for Academic Review**")
    
    st.markdown("""
    #### **1. Two-Stage Pipeline Architecture**
    Unlike monolithic or generic document-level sentiment classifiers, Aspectra implements a sequential 2-stage decoupling:
    1. **Stage 1: Aspect Term Extraction (ATE)**
       - Modeled as a token-level **BIO sequence tagging task** (`B-ASP`, `I-ASP`, `O`).
       - Tokens are classified into aspect spans.
    2. **Stage 2: Aspect-Level Sentiment Classification (ALSC)**
       - The sentence and the extracted aspect span are constructed into an attention sequence:  
         `[CLS] Review Text [SEP] Aspect Term [SEP]`
       - The model predicts one of 3 sentiment polarities: `positive`, `negative`, or `neutral`.

    #### **2. Leave-One-Out (LOO) Explainability Formulation**
    To eliminate black-box opacity, we compute token importance via perturbation:
    """)
    st.latex(r"I(w_k) = P(s \mid x, a) - P(s \mid x \setminus \{w_k\}, a)")
    st.markdown(r"""
    - If **$I(w_k) > 0$**: Removing token $w_k$ damages predicted confidence $\implies$ **Token supports the prediction**.
    - If **$I(w_k) < 0$**: Removing token $w_k$ boosts predicted confidence $\implies$ **Token contradicts or opposes the prediction**.
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>Aspectra ABSA Studio • Developed by Utkarsh Yadav • Production-Grade NLP System</div>", unsafe_allow_html=True)

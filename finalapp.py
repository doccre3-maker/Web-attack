import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🛡️ Web Attack Detection & Security Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CYBER THEME CSS WITH GLOW EFFECTS ----------
st.markdown("""
<style>
    /* Global dark theme */
    .main { background-color: #0a0e17; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .css-1d391kg { background-color: #111827; border-right: 1px solid #1f2937; }
    .header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 1.2rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.15);
        border: 1px solid rgba(0, 180, 255, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header h1 { margin: 0; font-weight: 700; font-size: 2.2rem; letter-spacing: 1px; text-shadow: 0 0 20px rgba(0,180,255,0.3); }
    .header p { margin: 0; opacity: 0.8; font-size: 1rem; }
    .header-clock { display: none; }
    .glow-card {
        background: rgba(17, 24, 39, 0.8);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 180, 255, 0.2);
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.05), inset 0 0 20px rgba(0, 180, 255, 0.02);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
        color: #e2e8f0;
    }
    .glow-card:hover { transform: translateY(-5px); box-shadow: 0 0 40px rgba(0,180,255,0.2), 0 0 80px rgba(0,180,255,0.05); border-color: rgba(0,180,255,0.6); }
    .metric-glow {
        background: rgba(17, 24, 39, 0.9);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(0, 180, 255, 0.15);
        box-shadow: 0 0 15px rgba(0, 180, 255, 0.05);
        transition: all 0.3s ease;
        color: #e2e8f0;
    }
    .metric-glow:hover { box-shadow: 0 0 30px rgba(0,180,255,0.2); border-color: #00d4ff; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; text-shadow: 0 0 20px rgba(0,180,255,0.3); }
    .metric-label { font-size: 0.9rem; color: #94a3b8; margin-top: 0.2rem; }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #007bff) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 30px !important;
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.3) !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        width: 100%;
    }
    .stButton > button:hover { box-shadow: 0 0 40px rgba(0,180,255,0.6) !important; transform: scale(1.02) !important; }
    .badge-critical { background: #ff0040; color: white; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; box-shadow: 0 0 20px rgba(255,0,64,0.3); }
    .badge-high { background: #ff7700; color: white; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; box-shadow: 0 0 20px rgba(255,119,0,0.3); }
    .badge-medium { background: #ffcc00; color: #0a0e17; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; box-shadow: 0 0 20px rgba(255,204,0,0.3); }
    .badge-low { background: #00cc88; color: white; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; box-shadow: 0 0 20px rgba(0,204,136,0.3); }
    .result-card { background: rgba(17,24,39,0.9); border-radius: 16px; padding: 1.5rem; border-left: 6px solid #00d4ff; box-shadow: 0 0 30px rgba(0,180,255,0.05); color: #e2e8f0; }
    .footer { text-align: center; margin-top: 3rem; color: #64748b; font-size: 0.9rem; border-top: 1px solid #1e293b; padding-top: 1.5rem; }
    div[role="radiogroup"] label { padding: 0.5rem 1rem; border-radius: 10px; transition: all 0.2s; cursor: pointer; color: #cbd5e1; }
    div[role="radiogroup"] label:hover { background: #1e293b; color: white; }
    div[role="radiogroup"] label[data-selected="true"] { background: linear-gradient(135deg, #00d4ff, #007bff); color: white; font-weight: 500; box-shadow: 0 0 30px rgba(0,180,255,0.3); }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0e17; }
    ::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------- PRESETS ----------
presets = {
    "Normal": {"method": "GET", "url_len": 35, "payload_len": 20, "param_count": 1,
               "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 0},
    "SQL Injection": {"method": "POST", "url_len": 90, "payload_len": 280, "param_count": 3,
                      "script": 0, "sql": 1, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 1, "csrf": 0},
    "XSS": {"method": "GET", "url_len": 80, "payload_len": 200, "param_count": 2,
            "script": 1, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 1, "csrf": 0},
    "CRLF Injection": {"method": "POST", "url_len": 60, "payload_len": 130, "param_count": 2,
                       "script": 0, "sql": 0, "crlf": 1, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 0},
    "Path Traversal": {"method": "GET", "url_len": 110, "payload_len": 120, "param_count": 2,
                       "script": 0, "sql": 0, "crlf": 0, "traversal": 1, "overflow": 0, "suspicious": 0, "csrf": 0},
    "Buffer Overflow": {"method": "POST", "url_len": 130, "payload_len": 1300, "param_count": 3,
                        "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 1, "suspicious": 0, "csrf": 0},
    "Parameter Tampering": {"method": "GET", "url_len": 80, "payload_len": 160, "param_count": 5,
                            "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 1, "csrf": 0},
    "Information Gathering": {"method": "GET", "url_len": 230, "payload_len": 95, "param_count": 6,
                              "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 0},
    "CSRF": {"method": "POST", "url_len": 65, "payload_len": 100, "param_count": 2,
             "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 1}
}

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    base = Path("models")
    scaler = joblib.load(base / "scaler.pkl")
    method_encoder = joblib.load(base / "method_encoder.pkl")
    target_encoder = joblib.load(base / "target_encoder.pkl")
    feature_cols = joblib.load(base / "feature_cols.pkl")
    comparison = pd.read_csv(base / "model_comparison.csv")
    model_single = joblib.load(base / "best_model.pkl")

    hybrid_available = False
    model_rf = model_xgb = model_svm = None
    try:
        model_rf = joblib.load(base / "random_forest.pkl")
        model_xgb = joblib.load(base / "xgboost.pkl")
        model_svm = joblib.load(base / "svm.pkl")
        hybrid_available = True
    except FileNotFoundError:
        pass

    return (model_single, model_rf, model_xgb, model_svm,
            scaler, method_encoder, target_encoder, feature_cols, comparison, hybrid_available)

(model_single, model_rf, model_xgb, model_svm,
 scaler, method_encoder, target_encoder, feature_cols, comparison, hybrid_available) = load_artifacts()

# ---------- SEVERITY & RECOMMENDATION ----------
severity_map = {
    'Normal': ('✅ Safe', 'Low', 'No action needed. The request appears legitimate.'),
    'SQL Injection': ('🔴 Critical', 'Critical', 'Use parameterized queries, input validation, and WAF rules.'),
    'XSS': ('🟠 High', 'High', 'Encode output, implement Content Security Policy (CSP), and validate inputs.'),
    'CRLF Injection': ('🟠 High', 'High', 'Sanitize newline characters (CRLF) and validate user inputs.'),
    'Path Traversal': ('🟠 High', 'High', 'Use allowlists for file paths, validate user-supplied filenames.'),
    'Buffer Overflow': ('🔴 Critical', 'Critical', 'Implement bounds checking, use safe functions, and enable stack protection.'),
    'Parameter Tampering': ('🟡 Medium', 'Medium', 'Validate all incoming parameters; use HMAC or signatures.'),
    'Information Gathering': ('🟡 Medium', 'Medium', 'Disable directory listing, hide error details, and restrict info disclosure.'),
    'CSRF': ('🟡 Medium', 'Medium', 'Use anti-CSRF tokens, SameSite cookies, and referer validation.')
}

attack_descriptions = {
    'Normal': 'Legitimate web request with no malicious intent.',
    'SQL Injection': 'Malicious SQL queries injected into input fields to manipulate databases.',
    'XSS': 'Cross-Site Scripting – injection of client-side scripts into web pages.',
    'CRLF Injection': 'Carriage Return Line Feed injection to split HTTP responses.',
    'Path Traversal': 'Accessing files outside the web root directory using "../" sequences.',
    'Buffer Overflow': 'Sending excessive data to overflow memory buffers and execute code.',
    'Parameter Tampering': 'Manipulating URL or form parameters to change application behavior.',
    'Information Gathering': 'Probing for system information, directory listings, or error details.',
    'CSRF': 'Cross-Site Request Forgery – forcing authenticated users to perform unwanted actions.'
}

# ---------- PREDICTION FUNCTIONS ----------
def safe_method_encode(method_list, encoder):
    known = set(encoder.classes_)
    encoded = []
    unknown_count = 0
    for m in method_list:
        if m in known:
            encoded.append(encoder.transform([m])[0])
        else:
            encoded.append(encoder.transform(['GET'])[0])
            unknown_count += 1
    if unknown_count > 0:
        st.warning(f"⚠️ {unknown_count} row(s) had unknown HTTP methods. They were mapped to 'GET'.")
    return np.array(encoded).reshape(-1, 1)

def predict_with_model(model, features_scaled):
    pred_num = model.predict(features_scaled)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
        confidence = max(proba) * 100
    else:
        confidence = 100.0
    return pred_num, confidence

def predict_hybrid(features_scaled):
    proba_rf = model_rf.predict_proba(features_scaled)[0]
    proba_xgb = model_xgb.predict_proba(features_scaled)[0]
    proba_svm = model_svm.predict_proba(features_scaled)[0]
    avg_proba = (proba_rf + proba_xgb + proba_svm) / 3.0
    pred_num = np.argmax(avg_proba)
    confidence = max(avg_proba) * 100
    return pred_num, confidence

def single_prediction(method, url_len, payload_len, param_count, script, sql, crlf, traversal, overflow, suspicious, csrf, model_choice):
    method_enc = method_encoder.transform([method])[0] if method in method_encoder.classes_ else method_encoder.transform(['GET'])[0]
    features = np.array([[method_enc, url_len, payload_len, param_count, script, sql, crlf, traversal, overflow, suspicious, csrf]])
    features_scaled = scaler.transform(features)
    
    if model_choice == "Hybrid (Ensemble)" and hybrid_available:
        pred_num, confidence = predict_hybrid(features_scaled)
    elif model_choice == "Random Forest":
        pred_num, confidence = predict_with_model(model_rf, features_scaled)
    elif model_choice == "XGBoost":
        pred_num, confidence = predict_with_model(model_xgb, features_scaled)
    elif model_choice == "SVM":
        pred_num, confidence = predict_with_model(model_svm, features_scaled)
    else:
        pred_num, confidence = predict_with_model(model_single, features_scaled)
    
    pred_class = target_encoder.inverse_transform([pred_num])[0]
    return pred_class, confidence

def enrich_predictions(df, mode):
    method_enc = safe_method_encode(df['HTTP_Method'].values, method_encoder)
    temp_df = df.copy()
    temp_df['HTTP_Method_Encoded'] = method_enc
    feature_cols_local = ['HTTP_Method_Encoded', 'URL_Length', 'Payload_Length', 'Parameter_Count',
                          'Script_Pattern', 'SQL_Pattern', 'CRLF_Pattern', 'Traversal_Pattern',
                          'Overflow_Pattern', 'Suspicious_Parameter', 'CSRF_Indicator']
    X = temp_df[feature_cols_local].values
    X_scaled = scaler.transform(X)

    if mode == 'single':
        preds = model_single.predict(X_scaled)
        if hasattr(model_single, "predict_proba"):
            probs = model_single.predict_proba(X_scaled)
            confidences = np.max(probs, axis=1) * 100
        else:
            confidences = np.ones(len(preds)) * 100.0
    else:
        if hybrid_available:
            proba_rf = model_rf.predict_proba(X_scaled)
            proba_xgb = model_xgb.predict_proba(X_scaled)
            proba_svm = model_svm.predict_proba(X_scaled)
            avg_proba = (proba_rf + proba_xgb + proba_svm) / 3.0
            preds = np.argmax(avg_proba, axis=1)
            confidences = np.max(avg_proba, axis=1) * 100
        else:
            preds = model_single.predict(X_scaled)
            if hasattr(model_single, "predict_proba"):
                probs = model_single.predict_proba(X_scaled)
                confidences = np.max(probs, axis=1) * 100
            else:
                confidences = np.ones(len(preds)) * 100.0

    pred_classes = target_encoder.inverse_transform(preds)
    df_result = df.copy()
    df_result['Prediction'] = pred_classes
    df_result['Confidence (%)'] = np.round(confidences, 2)
    df_result['Severity'] = df_result['Prediction'].apply(lambda x: severity_map.get(x, ('', 'Unknown', ''))[1])
    df_result['Recommendation'] = df_result['Prediction'].apply(lambda x: severity_map.get(x, ('', '', 'Check logs'))[2])
    df_result['Severity Icon'] = df_result['Prediction'].apply(lambda x: severity_map.get(x, ('', '', ''))[0])
    return df_result

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0;">
    <h2 style="color: #00d4ff; margin-bottom: 0.2rem; text-shadow: 0 0 20px rgba(0,180,255,0.3);">🛡️</h2>
    <h3 style="color: #e2e8f0; margin: 0;">Security Advisor</h3>
    <p style="color: #64748b; font-size: 0.8rem;">v2.0 - Cyber Edition</p>
</div>
<hr style="margin: 0.5rem 0; border-color: #1e293b;">
""", unsafe_allow_html=True)

nav_options = {
    "📊 Dashboard": "dashboard",
    "🔬 Single Predict": "single",
    "📤 Batch Predict": "batch",
    "📜 History": "history",
    "📊 Model Comparison": "comparison",
    "📖 Attack Reference": "reference",
    "ℹ️ About": "about"
}

# Initialize navigation state if not present
if 'nav' not in st.session_state:
    st.session_state.nav = "📊 Dashboard"

# Get current index for radio
current_index = list(nav_options.keys()).index(st.session_state.nav)

# Render radio widget
selected = st.sidebar.radio(
    "Navigate",
    options=list(nav_options.keys()),
    index=current_index,
    key="nav_radio"
)

# Update session state based on radio selection
if selected != st.session_state.nav:
    st.session_state.nav = selected

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Hybrid info
if not hybrid_available:
    st.sidebar.info(
        "ℹ️ **Hybrid mode unavailable** – missing model files. "
        "Run `model.py` to generate `random_forest.pkl`, `xgboost.pkl`, and `svm.pkl`."
    )
else:
    st.sidebar.success("✅ Hybrid mode available (all 3 models loaded).")

if hybrid_available:
    global_mode = st.sidebar.radio(
        "Default Mode for Batch",
        options=["Single Model (XGBoost)", "Hybrid (Ensemble)"],
        index=0,
        key="global_mode"
    )
    global_mode_str = 'single' if global_mode == "Single Model (XGBoost)" else 'hybrid'
else:
    global_mode = "Single Model (XGBoost)"
    global_mode_str = 'single'
    st.sidebar.caption("Batch prediction will use Single Model (XGBoost) only.")

st.sidebar.caption("Built with ❤️ using Streamlit & XGBoost")

# ---------- HEADER (NO CLOCK) ----------
st.markdown(f"""
<div class="header">
    <div>
        <h1>🛡️ Web Attack Detection & Security Advisor</h1>
        <p>AI-powered analysis against 9 attack classes</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- PAGE RENDER ----------
page = nav_options[st.session_state.nav]

# ============================================================
# 1. DASHBOARD
# ============================================================
if page == "dashboard":
    st.markdown("### 📊 Dashboard")
    
    # Load dataset stats
    try:
        data_path = Path("data") / "dataset.csv"
        if data_path.exists():
            df_dash = pd.read_csv(data_path)
            total_samples = len(df_dash)
            attack_types = df_dash['Attack_Type'].nunique()
            class_counts = df_dash['Attack_Type'].value_counts()
        else:
            total_samples = "N/A"
            attack_types = "N/A"
            class_counts = pd.Series()
    except:
        total_samples = "N/A"
        attack_types = "N/A"
        class_counts = pd.Series()
    
    best_model_name = comparison.loc[comparison['F1-Score'].idxmax(), 'Model']
    best_f1 = comparison['F1-Score'].max()
    
    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-glow">
            <div class="metric-value">{total_samples}</div>
            <div class="metric-label">Total Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-glow">
            <div class="metric-value">{attack_types}</div>
            <div class="metric-label">Attack Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-glow">
            <div class="metric-value">{best_model_name}</div>
            <div class="metric-label">Best Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-glow">
            <div class="metric-value">{best_f1:.4f}</div>
            <div class="metric-label">Best F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Class Distribution + Model Performance (resized charts)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Class Distribution")
        if not class_counts.empty:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ['#00d4ff', '#764ba2', '#28a745', '#fd7e14', '#dc3545', '#ffcc00', '#17a2b8', '#6c757d', '#e83e8c']
            bars = ax.bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)], edgecolor='#1e293b', linewidth=0.5)
            ax.set_xlabel('Attack Type', color='#94a3b8', fontsize=9)
            ax.set_ylabel('Count', color='#94a3b8', fontsize=9)
            ax.set_title('Dataset Class Distribution', color='#e2e8f0', fontsize=10)
            ax.tick_params(axis='x', rotation=45, colors='#94a3b8', labelsize=8)
            ax.tick_params(axis='y', colors='#94a3b8', labelsize=8)
            ax.set_facecolor('#0a0e17')
            fig.patch.set_facecolor('#0a0e17')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center', va='bottom', color='#e2e8f0', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Dataset not found.")
    
    with col2:
        st.subheader("📊 Model Performance")
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        models_names = comparison['Model'].tolist()
        x = np.arange(len(metrics))
        width = 0.25
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_m = ['#00d4ff', '#764ba2', '#28a745']
        for i, model in enumerate(models_names):
            values = comparison.loc[comparison['Model'] == model, metrics].values.flatten()
            ax.bar(x + i*width, values, width, label=model, color=colors_m[i], edgecolor='#1e293b', linewidth=0.5)
        ax.set_xlabel('Metrics', color='#94a3b8', fontsize=9)
        ax.set_ylabel('Score', color='#94a3b8', fontsize=9)
        ax.set_title('Model Comparison', color='#e2e8f0', fontsize=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics, color='#94a3b8', fontsize=8)
        ax.legend(loc='upper left', facecolor='#0a0e17', labelcolor='#e2e8f0', fontsize=8)
        ax.tick_params(axis='y', colors='#94a3b8', labelsize=8)
        ax.set_facecolor('#0a0e17')
        fig.patch.set_facecolor('#0a0e17')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Row 3: Session Summary (full width, Quick Actions removed)
    st.subheader("📋 Session Summary")
    if 'history' in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        total_preds = len(hist_df)
        attack_preds = len(hist_df[hist_df['Prediction'] != 'Normal'])
        normal_preds = total_preds - attack_preds
        
        st.markdown(f"""
        <div class="glow-card">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div><span style="color: #94a3b8;">Total</span><br><span style="font-size: 1.8rem; color: #00d4ff;">{total_preds}</span></div>
                <div><span style="color: #94a3b8;">Attacks</span><br><span style="font-size: 1.8rem; color: #ff0040;">{attack_preds}</span></div>
                <div><span style="color: #94a3b8;">Normal</span><br><span style="font-size: 1.8rem; color: #00cc88;">{normal_preds}</span></div>
            </div>
            <hr style="border-color: #1e293b;">
            <div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
                Attack Rate: <span style="color: #00d4ff;">{attack_preds/total_preds*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Recent Predictions**")
        recent = hist_df.tail(5)[['Time', 'Method', 'Prediction', 'Confidence']]
        st.dataframe(recent, use_container_width=True, height=200)
    else:
        st.markdown("""
        <div class="glow-card" style="text-align: center; padding: 2rem;">
            <span style="color: #64748b;">No predictions made in this session.</span><br>
            <span style="color: #94a3b8;">Head to <strong>Single Predict</strong> or <strong>Batch Predict</strong> to start.</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 2. SINGLE PREDICT
# ============================================================
elif page == "single":
    st.markdown("### 🔬 Single Request Analysis")
    
    model_options = ["Random Forest", "XGBoost", "SVM"]
    if hybrid_available:
        model_options.append("Hybrid (Ensemble)")
    default_model = "XGBoost"
    selected_model = st.selectbox(
        "Select Model for this prediction",
        options=model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
        help="Choose which machine learning model to use for this prediction."
    )
    
    with st.expander("📋 Load Example Preset"):
        preset_names = list(presets.keys())
        selected_preset = st.selectbox("Choose a preset", ["None"] + preset_names)
        if selected_preset != "None":
            p = presets[selected_preset]
            st.session_state['preset_method'] = p["method"]
            st.session_state['preset_url'] = p["url_len"]
            st.session_state['preset_payload'] = p["payload_len"]
            st.session_state['preset_param'] = p["param_count"]
            st.session_state['preset_script'] = p["script"]
            st.session_state['preset_sql'] = p["sql"]
            st.session_state['preset_crlf'] = p["crlf"]
            st.session_state['preset_traversal'] = p["traversal"]
            st.session_state['preset_overflow'] = p["overflow"]
            st.session_state['preset_suspicious'] = p["suspicious"]
            st.session_state['preset_csrf'] = p["csrf"]
    
    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("HTTP Method", ['GET', 'POST', 'PUT', 'DELETE'], 
                                  index=['GET','POST','PUT','DELETE'].index(st.session_state.get('preset_method', 'GET')))
            url_len = st.number_input("URL Length", min_value=0, max_value=5000, value=st.session_state.get('preset_url', 50), step=1)
            payload_len = st.number_input("Payload Length", min_value=0, max_value=10000, value=st.session_state.get('preset_payload', 100), step=1)
            param_count = st.number_input("Parameter Count", min_value=0, max_value=50, value=st.session_state.get('preset_param', 2), step=1)
        with col2:
            st.markdown("#### 🔧 Security Pattern Indicators")
            script = st.selectbox("Script Pattern", [0, 1], index=st.session_state.get('preset_script', 0))
            sql = st.selectbox("SQL Pattern", [0, 1], index=st.session_state.get('preset_sql', 0))
            crlf = st.selectbox("CRLF Pattern", [0, 1], index=st.session_state.get('preset_crlf', 0))
            traversal = st.selectbox("Traversal Pattern", [0, 1], index=st.session_state.get('preset_traversal', 0))
            overflow = st.selectbox("Overflow Pattern", [0, 1], index=st.session_state.get('preset_overflow', 0))
            suspicious = st.selectbox("Suspicious Param", [0, 1], index=st.session_state.get('preset_suspicious', 0))
            csrf = st.selectbox("CSRF Indicator", [0, 1], index=st.session_state.get('preset_csrf', 0))
        submitted = st.form_submit_button("🚀 Analyze Request", use_container_width=True)
    
    if submitted:
        if selected_model == "Hybrid (Ensemble)" and not hybrid_available:
            st.error("Hybrid model is not available. Please select another model.")
        else:
            pred_class, confidence = single_prediction(method, url_len, payload_len, param_count,
                                                       script, sql, crlf, traversal, overflow, suspicious, csrf,
                                                       model_choice=selected_model)
            severity_icon, severity_label, recommendation = severity_map.get(pred_class, ("⚠️ Unknown", "Unknown", "Check logs"))
            
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'Time': datetime.now().strftime("%H:%M:%S"), 'Method': method,
                'URL_Length': url_len, 'Payload_Length': payload_len,
                'Prediction': pred_class, 'Confidence': f"{confidence:.2f}%",
                'Severity': severity_label, 'Recommendation': recommendation,
                'Model': selected_model
            })
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.markdown(f"""
                <div class="glow-card" style="text-align:center;">
                    <div style="font-size:0.9rem; color:#94a3b8;">Prediction</div>
                    <div style="font-size:2rem; font-weight:700; color:#00d4ff;">{pred_class}</div>
                    <div style="margin-top:0.5rem;">
                        <span class="badge-{severity_label.lower()}">{severity_icon} {severity_label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <h4 style="color:#e2e8f0;">📌 Security Recommendation</h4>
                    <p style="font-size:1.1rem; color:#cbd5e1;">{recommendation}</p>
                    <hr style="border-color:#1e293b;">
                    <p style="font-size:0.9rem; color:#94a3b8;">
                        <strong>Model used:</strong> {selected_model}<br>
                        <strong>Input summary:</strong> {method} request, URL {url_len}, Payload {payload_len} bytes, {param_count} params.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="glow-card" style="text-align:center;">
                    <div style="font-size:0.9rem; color:#94a3b8;">Confidence</div>
                    <div style="font-size:2rem; font-weight:700; color:#00d4ff;">{confidence:.1f}%</div>
                    <div style="background:#1e293b; border-radius:10px; height:10px; margin-top:0.5rem;">
                        <div style="background:linear-gradient(90deg,#00d4ff,#764ba2); width:{min(confidence,100)}%; height:10px; border-radius:10px; box-shadow: 0 0 20px rgba(0,180,255,0.3);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Fill in the request details and click 'Analyze Request' to see the prediction.")

# ============================================================
# 3. BATCH PREDICT
# ============================================================
elif page == "batch":
    st.markdown("### 📤 Batch Prediction from CSV")
    if hybrid_available:
        st.markdown(f"**Using default model:** {global_mode}")
    else:
        st.markdown("**Using default model:** Single Model (XGBoost) – Hybrid unavailable")
    
    # Generate sample CSV
    sample_rows = []
    for _ in range(6):
        sample_rows.append({
            'HTTP_Method': random.choice(['GET', 'POST']),
            'URL_Length': random.randint(20, 50),
            'Payload_Length': random.randint(10, 40),
            'Parameter_Count': random.randint(0, 2),
            'Script_Pattern': 0,
            'SQL_Pattern': 0,
            'CRLF_Pattern': 0,
            'Traversal_Pattern': 0,
            'Overflow_Pattern': 0,
            'Suspicious_Parameter': 0,
            'CSRF_Indicator': 0
        })
    attack_keys = ['SQL Injection', 'XSS', 'Path Traversal', 'CSRF']
    for key in attack_keys:
        p = presets[key]
        sample_rows.append({
            'HTTP_Method': p['method'],
            'URL_Length': p['url_len'],
            'Payload_Length': p['payload_len'],
            'Parameter_Count': p['param_count'],
            'Script_Pattern': p['script'],
            'SQL_Pattern': p['sql'],
            'CRLF_Pattern': p['crlf'],
            'Traversal_Pattern': p['traversal'],
            'Overflow_Pattern': p['overflow'],
            'Suspicious_Parameter': p['suspicious'],
            'CSRF_Indicator': p['csrf']
        })
    random.shuffle(sample_rows)
    template_df = pd.DataFrame(sample_rows)
    csv_template = template_df.to_csv(index=False)
    
    st.download_button(
        label="📄 Download Sample CSV Template (10 rows)",
        data=csv_template,
        file_name="batch_input_template.csv",
        mime="text/csv"
    )
    st.caption("The template contains 6 Normal and 4 attack samples. You can edit it to add more rows.")
    
    # Upload file and store in session state
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    # Initialize session state for batch file and results
    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = None
    if 'batch_result' not in st.session_state:
        st.session_state.batch_result = None
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.session_state.batch_df = df_batch
            st.success(f"✅ File loaded: {df_batch.shape[0]} rows, {df_batch.shape[1]} columns.")
            
            # Validate columns
            required_cols = ['HTTP_Method', 'URL_Length', 'Payload_Length', 'Parameter_Count',
                             'Script_Pattern', 'SQL_Pattern', 'CRLF_Pattern', 'Traversal_Pattern',
                             'Overflow_Pattern', 'Suspicious_Parameter', 'CSRF_Indicator']
            missing = [c for c in required_cols if c not in df_batch.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.session_state.batch_df = None
            else:
                with st.expander("📋 Preview Uploaded Data"):
                    st.dataframe(df_batch.head(10), use_container_width=True)
                
                # Show Predict button
                if st.button("🔮 Predict Batch", use_container_width=True):
                    with st.spinner("🔄 Processing batch predictions..."):
                        result_df = enrich_predictions(df_batch, mode=global_mode_str)
                        st.session_state.batch_result = result_df
                    st.success("✅ Batch prediction complete!")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.session_state.batch_df = None
    
    # Display results if available
    if st.session_state.batch_result is not None:
        result_df = st.session_state.batch_result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Requests", len(result_df))
        with col2:
            attack_count = len(result_df[result_df['Prediction'] != 'Normal'])
            st.metric("Attacks Detected", attack_count, delta=f"{(attack_count/len(result_df)*100):.1f}%")
        with col3:
            normal_count = len(result_df[result_df['Prediction'] == 'Normal'])
            st.metric("Normal Requests", normal_count)
        
        st.subheader("📊 Prediction Distribution")
        dist = result_df['Prediction'].value_counts()
        if len(dist) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ['#00d4ff', '#764ba2', '#28a745', '#fd7e14', '#dc3545', '#ffcc00', '#17a2b8', '#6c757d', '#e83e8c']
            bars = ax.bar(dist.index, dist.values, color=colors[:len(dist)], edgecolor='#1e293b', linewidth=0.5)
            ax.set_xlabel('Attack Type', color='#94a3b8', fontsize=9)
            ax.set_ylabel('Count', color='#94a3b8', fontsize=9)
            ax.set_title('Batch Prediction Distribution', color='#e2e8f0', fontsize=10)
            ax.tick_params(axis='x', rotation=45, colors='#94a3b8', labelsize=8)
            ax.tick_params(axis='y', colors='#94a3b8', labelsize=8)
            ax.set_facecolor('#0a0e17')
            fig.patch.set_facecolor('#0a0e17')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom', color='#e2e8f0', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("📋 Batch Prediction Results")
        display_cols = ['HTTP_Method', 'URL_Length', 'Payload_Length', 'Parameter_Count',
                        'Prediction', 'Confidence (%)', 'Severity']
        display_cols = [c for c in display_cols if c in result_df.columns]
        st.dataframe(result_df[display_cols], use_container_width=True, height=400)
        
        csv_result = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv_result,
            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Clear results button
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.batch_result = None
            st.rerun()

# ============================================================
# 4. HISTORY (WITH DARK THEME PIE CHART)
# ============================================================
elif page == "history":
    st.markdown("### 📜 Prediction History")
    if 'history' in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.download_button("📥 Download History", data=hist_df.to_csv(index=False).encode('utf-8'),
                           file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        st.dataframe(hist_df, use_container_width=True, height=350)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        
        # Pie chart with dark background (resized)
        pie_data = hist_df['Prediction'].value_counts()
        if len(pie_data) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor('#0a0e17')
            fig.patch.set_facecolor('#0a0e17')
            colors = ['#00d4ff', '#764ba2', '#28a745', '#fd7e14', '#dc3545', '#ffcc00', '#17a2b8', '#6c757d', '#e83e8c']
            wedges, texts, autotexts = ax.pie(
                pie_data.values,
                labels=pie_data.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors[:len(pie_data)],
                wedgeprops={'edgecolor': '#1e293b', 'linewidth': 1},
                textprops={'color': '#e2e8f0', 'fontsize': 9}
            )
            for autotext in autotexts:
                autotext.set_color('#e2e8f0')
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No predictions yet.")

# ============================================================
# 5. MODEL COMPARISON (WITH RESIZED CHARTS)
# ============================================================
elif page == "comparison":
    st.markdown("### 📊 Model Performance Comparison")
    
    st.subheader("Performance Metrics")
    styled = comparison.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen')
    st.dataframe(styled, use_container_width=True)
    st.caption("The best model is highlighted in green.")
    
    st.subheader("📊 Model Comparison Chart")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models_names = comparison['Model'].tolist()
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#00d4ff', '#764ba2', '#28a745']
    for i, model in enumerate(models_names):
        values = comparison.loc[comparison['Model'] == model, metrics].values.flatten()
        ax.bar(x + i*width, values, width, label=model, color=colors[i], edgecolor='#1e293b', linewidth=0.5)
    ax.set_xlabel('Metrics', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Score', color='#94a3b8', fontsize=9)
    ax.set_title('Model Performance Comparison', color='#e2e8f0', fontsize=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, color='#94a3b8', fontsize=8)
    ax.legend(facecolor='#0a0e17', labelcolor='#e2e8f0', fontsize=8)
    ax.tick_params(axis='y', colors='#94a3b8', labelsize=8)
    ax.set_facecolor('#0a0e17')
    fig.patch.set_facecolor('#0a0e17')
    for i, model in enumerate(models_names):
        values = comparison.loc[comparison['Model'] == model, metrics].values.flatten()
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7, color='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🔥 Feature Importance (XGBoost)")
    st.markdown("This shows the relative importance of each feature in the best model (XGBoost).")
    if hasattr(model_single, "feature_importances_"):
        importances = model_single.feature_importances_
        feature_names = ['HTTP Method', 'URL Length', 'Payload Length', 'Parameter Count',
                         'Script Pattern', 'SQL Pattern', 'CRLF Pattern', 'Traversal Pattern',
                         'Overflow Pattern', 'Suspicious Param', 'CSRF Indicator']
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#00d4ff', edgecolor='#1e293b')
        ax.set_xlabel('Importance Score', color='#94a3b8', fontsize=9)
        ax.set_title('Feature Importances (XGBoost)', color='#e2e8f0', fontsize=10)
        ax.tick_params(axis='x', colors='#94a3b8', labelsize=8)
        ax.tick_params(axis='y', colors='#94a3b8', labelsize=8)
        ax.set_facecolor('#0a0e17')
        fig.patch.set_facecolor('#0a0e17')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("**Interpretation:** Features with higher importance have a greater impact on the model's decision.")
    else:
        st.info("Feature importance not available for this model type.")

# ============================================================
# 6. ATTACK REFERENCE
# ============================================================
elif page == "reference":
    st.markdown("### 📖 Attack Type Quick Reference")
    ref_data = []
    for attack, desc in attack_descriptions.items():
        sev = severity_map.get(attack, ('', 'Unknown', ''))[1]
        icon = severity_map.get(attack, ('', '', ''))[0]
        ref_data.append({'Attack': attack, 'Severity': sev, 'Icon': icon, 'Description': desc})
    for idx, row in enumerate(ref_data):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{row['Icon']} {row['Attack']}**")
                st.caption(f"Severity: {row['Severity']}")
            with col2:
                st.write(row['Description'])
            st.divider()

# ============================================================
# 7. ABOUT
# ============================================================
elif page == "about":
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **Machine Learning-Based Web Attack Detection and Security Recommendation System**
    
    This application is developed as a final year project to demonstrate the application of machine learning in cybersecurity.
    
    **Key Features:**
    - Detects 9 types of web attacks (SQL Injection, XSS, CRLF, Path Traversal, Buffer Overflow, Parameter Tampering, Information Gathering, CSRF, and Normal).
    - Uses a balanced synthetic dataset derived from CSIC 2010.
    - Trained and compared three models: Random Forest, XGBoost, SVM.
    - The best model (XGBoost) is used for real-time predictions.
    - Provides severity levels and actionable security recommendations.
    - Supports both single and batch prediction.
    - Interactive and user-friendly interface built with Streamlit.
    
    **Technology Stack:**
    - Python 3.10
    - Streamlit (UI)
    - scikit-learn (ML models)
    - XGBoost
    - Pandas, NumPy, Matplotlib
    
    **Dataset:** CSIC 2010 Web Application Attacks (synthetic balanced version with 9 classes, 1200 samples each).
    
    **Model Performance:** XGBoost achieved ~98.1% accuracy and F1-score.
    
    ---
    **Developer:** [Your Name]  
    **Institution:** [Your Institution]  
    **Year:** 2026
    """)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    Built with ❤️ using Python, Streamlit, scikit-learn, XGBoost | Dataset: CSIC 2010 (balanced synthetic)
</div>
""", unsafe_allow_html=True)
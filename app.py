import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import io

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🛡️ Web Attack Detector & Security Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS (Sleek Modern) ----------
st.markdown("""
<style>
    /* ... (keep previous CSS) ... */
    .main { background-color: #f8f9fc; }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .header h1 { margin: 0; font-weight: 700; font-size: 2.5rem; }
    .header p { margin: 0; opacity: 0.9; font-size: 1.1rem; }
    .card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.3);
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .card:hover { transform: translateY(-3px); }
    .badge-critical { background-color: #dc3545; color: white; padding: 0.3rem 1rem; border-radius: 30px; font-weight: 600; }
    .badge-high { background-color: #fd7e14; color: white; padding: 0.3rem 1rem; border-radius: 30px; font-weight: 600; }
    .badge-medium { background-color: #ffc107; color: #212529; padding: 0.3rem 1rem; border-radius: 30px; font-weight: 600; }
    .badge-low { background-color: #28a745; color: white; padding: 0.3rem 1rem; border-radius: 30px; font-weight: 600; }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 6px solid #667eea;
    }
    .footer { text-align: center; margin-top: 3rem; color: #6c757d; font-size: 0.9rem; border-top: 1px solid #dee2e6; padding-top: 1.5rem; }
    .stProgress > div > div > div > div { background-color: #667eea; }
    div[data-testid="stSidebarNav"] { display: none; }  /* hide default nav */
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    base = Path("models")
    model = joblib.load(base / "best_model.pkl")
    scaler = joblib.load(base / "scaler.pkl")
    method_encoder = joblib.load(base / "method_encoder.pkl")
    target_encoder = joblib.load(base / "target_encoder.pkl")
    feature_cols = joblib.load(base / "feature_cols.pkl")
    comparison = pd.read_csv(base / "model_comparison.csv")
    return model, scaler, method_encoder, target_encoder, feature_cols, comparison

model, scaler, method_encoder, target_encoder, feature_cols, comparison = load_artifacts()

# ---------- SEVERITY & RECOMMENDATION MAPPING ----------
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

# ---------- SAMPLE INPUT PRESETS ----------
presets = {
    "Normal": {
        "method": "GET", "url_len": 35, "payload_len": 20, "param_count": 1,
        "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 0
    },
    "SQL Injection": {
        "method": "POST", "url_len": 90, "payload_len": 280, "param_count": 3,
        "script": 0, "sql": 1, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 1, "csrf": 0
    },
    "XSS": {
        "method": "GET", "url_len": 80, "payload_len": 200, "param_count": 2,
        "script": 1, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 1, "csrf": 0
    },
    "CRLF Injection": {
        "method": "POST", "url_len": 60, "payload_len": 130, "param_count": 2,
        "script": 0, "sql": 0, "crlf": 1, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 0
    },
    "Path Traversal": {
        "method": "GET", "url_len": 110, "payload_len": 120, "param_count": 2,
        "script": 0, "sql": 0, "crlf": 0, "traversal": 1, "overflow": 0, "suspicious": 0, "csrf": 0
    },
    "Buffer Overflow": {
        "method": "POST", "url_len": 130, "payload_len": 1300, "param_count": 3,
        "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 1, "suspicious": 0, "csrf": 0
    },
    "Parameter Tampering": {
        "method": "GET", "url_len": 80, "payload_len": 160, "param_count": 5,
        "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 1, "csrf": 0
    },
    "Information Gathering": {
        "method": "GET", "url_len": 230, "payload_len": 95, "param_count": 6,
        "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 0
    },
    "CSRF": {
        "method": "POST", "url_len": 65, "payload_len": 100, "param_count": 2,
        "script": 0, "sql": 0, "crlf": 0, "traversal": 0, "overflow": 0, "suspicious": 0, "csrf": 1
    }
}

# ---------- SIDEBAR ----------
st.sidebar.markdown("## 🔍 Request Analyzer")

# Preset selector
preset_name = st.sidebar.selectbox(
    "📋 Load Example",
    ["Select a preset..."] + list(presets.keys()),
    index=0
)

with st.sidebar.form("prediction_form"):
    # If preset selected, fill values
    if preset_name != "Select a preset...":
        preset = presets[preset_name]
        method_default = preset["method"]
        url_len_default = preset["url_len"]
        payload_len_default = preset["payload_len"]
        param_count_default = preset["param_count"]
        script_default = preset["script"]
        sql_default = preset["sql"]
        crlf_default = preset["crlf"]
        traversal_default = preset["traversal"]
        overflow_default = preset["overflow"]
        suspicious_default = preset["suspicious"]
        csrf_default = preset["csrf"]
    else:
        method_default = "GET"
        url_len_default = 50
        payload_len_default = 100
        param_count_default = 2
        script_default = 0
        sql_default = 0
        crlf_default = 0
        traversal_default = 0
        overflow_default = 0
        suspicious_default = 0
        csrf_default = 0

    method = st.selectbox("HTTP Method", ['GET', 'POST', 'PUT', 'DELETE'], index=['GET','POST','PUT','DELETE'].index(method_default))
    url_len = st.number_input("URL Length", min_value=0, max_value=5000, value=url_len_default, step=1)
    payload_len = st.number_input("Payload Length", min_value=0, max_value=10000, value=payload_len_default, step=1)
    param_count = st.number_input("Parameter Count", min_value=0, max_value=50, value=param_count_default, step=1)

    st.markdown("---")
    st.markdown("#### 🔧 Security Pattern Indicators")
    col1, col2 = st.columns(2)
    with col1:
        script = st.selectbox("Script Pattern (<script>)", [0, 1], index=script_default)
        sql = st.selectbox("SQL Pattern (SELECT, UNION)", [0, 1], index=sql_default)
        crlf = st.selectbox("CRLF Pattern (%0d%0a)", [0, 1], index=crlf_default)
    with col2:
        traversal = st.selectbox("Traversal Pattern (../)", [0, 1], index=traversal_default)
        overflow = st.selectbox("Overflow Pattern (long payload)", [0, 1], index=overflow_default)
        suspicious = st.selectbox("Suspicious Parameter", [0, 1], index=suspicious_default)
        csrf = st.selectbox("CSRF Indicator", [0, 1], index=csrf_default)

    submitted = st.form_submit_button("🚀 Analyze Request", use_container_width=True)

# ---------- PREDICTION FUNCTION ----------
def make_prediction(method, url_len, payload_len, param_count, script, sql, crlf, traversal, overflow, suspicious, csrf):
    method_enc = method_encoder.transform([method])[0]
    features = [method_enc, url_len, payload_len, param_count, 
                script, sql, crlf, traversal, overflow, suspicious, csrf]
    scaled = scaler.transform([features])
    pred_num = model.predict(scaled)[0]
    pred_class = target_encoder.inverse_transform([pred_num])[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled)[0]
        confidence = max(probs) * 100
    else:
        confidence = 100.0
    return pred_class, confidence

# ---------- MAIN UI ----------
st.markdown("""
<div class="header">
    <h1>🛡️ Web Attack Detection & Security Advisor</h1>
    <p>AI-powered analysis of web request parameters against 9 attack classes</p>
</div>
""", unsafe_allow_html=True)

# Tabs for better organization
tab1, tab2, tab3 = st.tabs(["🔬 Analyze Request", "📜 Prediction History", "📊 Model Comparison"])

# ---------- TAB 1: ANALYSIS ----------
with tab1:
    if submitted:
        pred_class, confidence = make_prediction(method, url_len, payload_len, param_count,
                                                 script, sql, crlf, traversal, overflow, suspicious, csrf)
        severity_icon, severity_label, recommendation = severity_map.get(pred_class, ("⚠️ Unknown", "Unknown", "Check logs"))

        # Store history
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Method': method,
            'URL_Length': url_len,
            'Payload_Length': payload_len,
            'Prediction': pred_class,
            'Confidence': f"{confidence:.2f}%",
            'Severity': severity_label,
            'Recommendation': recommendation
        })

        # Display results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h3>🔮 Prediction</h3>
                <h2 style="font-weight:700; color:#2c3e50;">{pred_class}</h2>
                <div style="margin: 0.5rem 0;">
                    <span class="badge-{severity_label.lower()}">{severity_icon} {severity_label}</span>
                </div>
                <p style="font-size:1.2rem;">Confidence: <strong>{confidence:.2f}%</strong></p>
                <!-- Confidence bar -->
                <div style="background:#e9ecef; border-radius:10px; height:10px; margin-top:0.5rem;">
                    <div style="background:linear-gradient(90deg, #667eea, #764ba2); width:{min(confidence,100)}%; height:10px; border-radius:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <h4>📌 Security Recommendation</h4>
                <p style="font-size:1.1rem;">{recommendation}</p>
                <hr>
                <p style="font-size:0.9rem; color:#495057;">
                    <strong>Input summary:</strong> {method} request, URL length {url_len}, payload {payload_len} bytes, {param_count} parameters.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Expandable feature details
        with st.expander("📊 Input Features Details"):
            features_dict = {
                'HTTP Method': method,
                'URL Length': url_len,
                'Payload Length': payload_len,
                'Parameter Count': param_count,
                'Script Pattern': "Yes" if script else "No",
                'SQL Pattern': "Yes" if sql else "No",
                'CRLF Pattern': "Yes" if crlf else "No",
                'Traversal Pattern': "Yes" if traversal else "No",
                'Overflow Pattern': "Yes" if overflow else "No",
                'Suspicious Parameter': "Yes" if suspicious else "No",
                'CSRF Indicator': "Yes" if csrf else "No"
            }
            feat_df = pd.DataFrame([features_dict]).T
            feat_df.columns = ['Value']
            st.dataframe(feat_df, use_container_width=True)

# ---------- TAB 2: HISTORY ----------
with tab2:
    st.subheader("📜 Prediction History (Current Session)")

    if 'history' in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        # Download button
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Show table
        st.dataframe(hist_df, use_container_width=True, height=300)

        # Clear button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

        # Pie chart of predictions
        st.subheader("📊 Prediction Distribution")
        pie_data = hist_df['Prediction'].value_counts()
        if len(pie_data) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#667eea', '#764ba2', '#28a745', '#fd7e14', '#dc3545', '#ffc107', '#17a2b8', '#6c757d', '#e83e8c']
            wedges, texts, autotexts = ax.pie(
                pie_data.values, 
                labels=pie_data.index, 
                autopct='%1.1f%%',
                colors=colors[:len(pie_data)],
                startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("No predictions yet.")
    else:
        st.info("No predictions made yet. Use the 'Analyze Request' tab to start.")

# ---------- TAB 3: MODEL COMPARISON ----------
with tab3:
    st.subheader("📊 Model Performance Comparison")
    st.markdown("The following table shows the performance metrics of all three models on the test set.")
    # Highlight best
    styled = comparison.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen')
    st.dataframe(styled, use_container_width=True)
    st.caption("The best model is automatically used for predictions.")

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    Built with ❤️ using Python, Streamlit, scikit-learn, XGBoost | Dataset: CSIC 2010 (balanced synthetic)
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🛡️ Web Attack Detection & Security Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- ADVANCED CUSTOM CSS ----------
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .css-1d391kg { background-color: #ffffff; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    .header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.2rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header h1 { margin: 0; font-weight: 700; font-size: 2.2rem; letter-spacing: 1px; }
    .header p { margin: 0; opacity: 0.8; font-size: 1rem; }
    .header-clock { font-size: 1.1rem; background: rgba(255,255,255,0.15); padding: 0.4rem 1.2rem; border-radius: 30px; }
    .card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.4);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,0,0,0.12); }
    .badge-critical { background: #dc3545; color: white; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; }
    .badge-high { background: #fd7e14; color: white; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; }
    .badge-medium { background: #ffc107; color: #212529; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; }
    .badge-low { background: #28a745; color: white; padding: 0.3rem 1.2rem; border-radius: 30px; font-weight: 600; display: inline-block; }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 6px solid #667eea;
    }
    .footer { text-align: center; margin-top: 3rem; color: #6c757d; font-size: 0.9rem; border-top: 1px solid #dee2e6; padding-top: 1.5rem; }
    div[role="radiogroup"] { gap: 0.3rem; }
    div[role="radiogroup"] label {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: all 0.2s;
        cursor: pointer;
    }
    div[role="radiogroup"] label:hover { background: #e9ecef; }
    div[role="radiogroup"] label[data-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
    }
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
    # Load essential common artifacts
    scaler = joblib.load(base / "scaler.pkl")
    method_encoder = joblib.load(base / "method_encoder.pkl")
    target_encoder = joblib.load(base / "target_encoder.pkl")
    feature_cols = joblib.load(base / "feature_cols.pkl")
    comparison = pd.read_csv(base / "model_comparison.csv")
    model_single = joblib.load(base / "best_model.pkl")

    # Try to load the three individual models for hybrid mode
    hybrid_available = True
    try:
        model_rf = joblib.load(base / "random_forest.pkl")
        model_xgb = joblib.load(base / "xgboost.pkl")
        model_svm = joblib.load(base / "svm.pkl")
    except FileNotFoundError:
        hybrid_available = False
        model_rf = model_xgb = model_svm = None
        st.warning("⚠️ Hybrid mode is disabled because one or more individual model files are missing. "
                   "Run `model.py` again to generate them, or use Single mode.")

    return (model_single, model_rf, model_xgb, model_svm,
            scaler, method_encoder, target_encoder, feature_cols, comparison, hybrid_available)

(model_single, model_rf, model_xgb, model_svm,
 scaler, method_encoder, target_encoder, feature_cols, comparison, hybrid_available) = load_artifacts()

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

# ---------- HELPER FUNCTIONS ----------
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

def predict_single(features_scaled):
    pred_num = model_single.predict(features_scaled)[0]
    if hasattr(model_single, "predict_proba"):
        proba = model_single.predict_proba(features_scaled)[0]
        confidence = max(proba) * 100
    else:
        confidence = 100.0
    return pred_num, confidence

def predict_hybrid(features_scaled):
    if not hybrid_available:
        # fallback to single
        return predict_single(features_scaled)
    proba_rf = model_rf.predict_proba(features_scaled)[0]
    proba_xgb = model_xgb.predict_proba(features_scaled)[0]
    proba_svm = model_svm.predict_proba(features_scaled)[0]
    avg_proba = (proba_rf + proba_xgb + proba_svm) / 3.0
    pred_num = np.argmax(avg_proba)
    confidence = max(avg_proba) * 100
    return pred_num, confidence

def single_prediction(method, url_len, payload_len, param_count, script, sql, crlf, traversal, overflow, suspicious, csrf, mode='single'):
    method_enc = method_encoder.transform([method])[0] if method in method_encoder.classes_ else method_encoder.transform(['GET'])[0]
    features = np.array([[method_enc, url_len, payload_len, param_count, script, sql, crlf, traversal, overflow, suspicious, csrf]])
    features_scaled = scaler.transform(features)
    if mode == 'single':
        pred_num, confidence = predict_single(features_scaled)
    else:  # hybrid
        pred_num, confidence = predict_hybrid(features_scaled)
    pred_class = target_encoder.inverse_transform([pred_num])[0]
    return pred_class, confidence

def enrich_predictions(df, mode='single'):
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
    else:  # hybrid
        if hybrid_available:
            proba_rf = model_rf.predict_proba(X_scaled)
            proba_xgb = model_xgb.predict_proba(X_scaled)
            proba_svm = model_svm.predict_proba(X_scaled)
            avg_proba = (proba_rf + proba_xgb + proba_svm) / 3.0
            preds = np.argmax(avg_proba, axis=1)
            confidences = np.max(avg_proba, axis=1) * 100
        else:
            # fallback to single
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

# ---------- SIDEBAR: NAVIGATION + MODE SELECTOR ----------
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0;">
    <h2 style="color: #2c3e50; margin-bottom: 0.2rem;">🛡️</h2>
    <h3 style="color: #2c3e50; margin: 0;">Security Advisor</h3>
    <p style="color: #6c757d; font-size: 0.8rem;">v2.0 - ML Powered</p>
</div>
<hr style="margin: 0.5rem 0;">
""", unsafe_allow_html=True)

nav_options = {
    "🔬 Single Predict": "single",
    "📤 Batch Predict": "batch",
    "📜 History": "history",
    "📊 Model Comparison": "comparison",
    "🔥 Feature Importance": "importance",
    "📖 Attack Reference": "reference",
    "ℹ️ About": "about"
}

selection = st.sidebar.radio(
    "Navigate",
    options=list(nav_options.keys()),
    index=0,
    key="nav",
    label_visibility="collapsed"
)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# ----- Prediction Mode Toggle (only if hybrid available) -----
if hybrid_available:
    mode = st.sidebar.radio(
        "Prediction Mode",
        options=["Single Model (XGBoost)", "Hybrid (Ensemble)"],
        index=0,
        key="mode",
        help="Single uses the best model; Hybrid averages predictions from all three."
    )
else:
    st.sidebar.info("ℹ️ Hybrid mode unavailable (missing model files). Using Single model only.")
    mode = "Single Model (XGBoost)"

st.sidebar.caption("Built with ❤️ using Streamlit & XGBoost")

mode_str = 'single' if mode == "Single Model (XGBoost)" else 'hybrid'

# ---------- REST OF APP ----------
page = nav_options[selection]

current_time = datetime.now().strftime("%H:%M:%S")

st.markdown(f"""
<div class="header">
    <div>
        <h1>🛡️ Web Attack Detection & Security Advisor</h1>
        <p>AI-powered analysis against 9 attack classes • {current_time}</p>
    </div>
    <div class="header-clock">⚡ Live</div>
</div>
""", unsafe_allow_html=True)

# ---------- PAGE RENDER ----------
if page == "single":
    st.markdown("### 🔬 Single Request Analysis")
    st.markdown(f"**Current Mode:** {mode}")
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
            st.markdown("#### Security Pattern Indicators")
            script = st.selectbox("Script Pattern", [0, 1], index=st.session_state.get('preset_script', 0))
            sql = st.selectbox("SQL Pattern", [0, 1], index=st.session_state.get('preset_sql', 0))
            crlf = st.selectbox("CRLF Pattern", [0, 1], index=st.session_state.get('preset_crlf', 0))
            traversal = st.selectbox("Traversal Pattern", [0, 1], index=st.session_state.get('preset_traversal', 0))
            overflow = st.selectbox("Overflow Pattern", [0, 1], index=st.session_state.get('preset_overflow', 0))
            suspicious = st.selectbox("Suspicious Param", [0, 1], index=st.session_state.get('preset_suspicious', 0))
            csrf = st.selectbox("CSRF Indicator", [0, 1], index=st.session_state.get('preset_csrf', 0))
        submitted = st.form_submit_button("🚀 Analyze Request", use_container_width=True)
    
    if submitted:
        pred_class, confidence = single_prediction(method, url_len, payload_len, param_count,
                                                   script, sql, crlf, traversal, overflow, suspicious, csrf,
                                                   mode=mode_str)
        severity_icon, severity_label, recommendation = severity_map.get(pred_class, ("⚠️ Unknown", "Unknown", "Check logs"))
        
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            'Time': datetime.now().strftime("%H:%M:%S"), 'Method': method,
            'URL_Length': url_len, 'Payload_Length': payload_len,
            'Prediction': pred_class, 'Confidence': f"{confidence:.2f}%",
            'Severity': severity_label, 'Recommendation': recommendation,
            'Mode': mode
        })
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h3>🔮 Prediction</h3>
                <h2 style="font-weight:700;">{pred_class}</h2>
                <div><span class="badge-{severity_label.lower()}">{severity_icon} {severity_label}</span></div>
                <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                <div style="background:#e9ecef; border-radius:10px; height:10px; margin-top:0.5rem;">
                    <div style="background:linear-gradient(90deg,#667eea,#764ba2); width:{min(confidence,100)}%; height:10px; border-radius:10px;"></div>
                </div>
                <p style="margin-top:0.5rem; font-size:0.85rem; color:#6c757d;">Mode: {mode}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <h4>📌 Security Recommendation</h4>
                <p style="font-size:1.1rem;">{recommendation}</p>
                <hr><p style="font-size:0.9rem;"><strong>Input:</strong> {method} request, URL {url_len}, Payload {payload_len} bytes, {param_count} params.</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "batch":
    st.markdown("### 📤 Batch Prediction from CSV")
    st.markdown(f"**Current Mode:** {mode}")
    template_df = pd.DataFrame({
        'HTTP_Method': ['GET', 'POST'],
        'URL_Length': [45, 120],
        'Payload_Length': [30, 350],
        'Parameter_Count': [1, 3],
        'Script_Pattern': [0, 1],
        'SQL_Pattern': [0, 1],
        'CRLF_Pattern': [0, 0],
        'Traversal_Pattern': [0, 0],
        'Overflow_Pattern': [0, 0],
        'Suspicious_Parameter': [0, 1],
        'CSRF_Indicator': [0, 0]
    })
    csv_template = template_df.to_csv(index=False)
    st.download_button("📄 Download Sample CSV Template", data=csv_template, file_name="batch_input_template.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {df_batch.shape[0]} rows, {df_batch.shape[1]} columns.")
            required_cols = ['HTTP_Method', 'URL_Length', 'Payload_Length', 'Parameter_Count',
                             'Script_Pattern', 'SQL_Pattern', 'CRLF_Pattern', 'Traversal_Pattern',
                             'Overflow_Pattern', 'Suspicious_Parameter', 'CSRF_Indicator']
            missing = [c for c in required_cols if c not in df_batch.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
            else:
                with st.expander("📋 Preview Uploaded Data"):
                    st.dataframe(df_batch.head(10), use_container_width=True)
                with st.spinner("🔄 Processing batch predictions..."):
                    result_df = enrich_predictions(df_batch, mode=mode_str)
                st.success("✅ Batch prediction complete!")
                
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
                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['#667eea', '#764ba2', '#28a745', '#fd7e14', '#dc3545', '#ffc107', '#17a2b8', '#6c757d', '#e83e8c']
                    bars = ax.bar(dist.index, dist.values, color=colors[:len(dist)], edgecolor='black', linewidth=0.5)
                    ax.set_xlabel('Attack Type'); ax.set_ylabel('Count'); ax.set_title('Batch Prediction Distribution')
                    ax.tick_params(axis='x', rotation=45)
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom')
                    st.pyplot(fig)
                
                st.subheader("📋 Batch Prediction Results")
                display_cols = ['HTTP_Method', 'URL_Length', 'Payload_Length', 'Parameter_Count',
                                'Prediction', 'Confidence (%)', 'Severity']
                display_cols = [c for c in display_cols if c in result_df.columns]
                st.dataframe(result_df[display_cols], use_container_width=True, height=400)
                
                csv_result = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions (CSV)", data=csv_result,
                                   file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ---------- HISTORY, COMPARISON, IMPORTANCE, REFERENCE, ABOUT ----------
# (same as before – I'll include them for completeness)

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
        pie_data = hist_df['Prediction'].value_counts()
        if len(pie_data) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
            ax.axis('equal')
            st.pyplot(fig)
    else:
        st.info("No predictions yet.")

elif page == "comparison":
    st.markdown("### 📊 Model Performance Comparison")
    styled = comparison.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen')
    st.dataframe(styled, use_container_width=True)
    st.caption("The best model is automatically used for predictions when in Single mode.")

elif page == "importance":
    st.markdown("### 🔥 Feature Importance Analysis")
    st.markdown("This shows the relative importance of each feature in the best model (XGBoost).")
    if hasattr(model_single, "feature_importances_"):
        importances = model_single.feature_importances_
        feature_names = ['HTTP Method', 'URL Length', 'Payload Length', 'Parameter Count',
                         'Script Pattern', 'SQL Pattern', 'CRLF Pattern', 'Traversal Pattern',
                         'Overflow Pattern', 'Suspicious Param', 'CSRF Indicator']
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#667eea', edgecolor='black')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importances (XGBoost)')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        st.markdown("**Interpretation:** Features with higher importance have a greater impact on the model's decision.")
    else:
        st.info("Feature importance not available for this model type.")

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
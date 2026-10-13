# model.py – Train and save all models including individual ones for hybrid mode

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# -------------------- CONFIGURATION --------------------
DATA_FOLDER = r"C:\Users\user\Desktop\web_attack\data"   # adjust if needed
MODEL_FOLDER = "models"
Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

# Find the dataset CSV
csv_files = list(Path(DATA_FOLDER).glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV found in {DATA_FOLDER}")
DATA_FILE = csv_files[0]
print(f"[INFO] Using dataset: {DATA_FILE.name}")

# -------------------- LOAD DATA --------------------
df = pd.read_csv(DATA_FILE)
print(f"[INFO] Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------- PREPROCESSING --------------------
target_col = 'Attack_Type'
if target_col not in df.columns:
    possible = [c for c in df.columns if 'attack' in c.lower() or 'class' in c.lower()]
    target_col = possible[0] if possible else df.columns[-1]
print(f"[INFO] Target column: '{target_col}'")

# Encode HTTP Method
method_encoder = LabelEncoder()
df['HTTP_Method_Encoded'] = method_encoder.fit_transform(df['HTTP_Method'])

# Feature columns (as per abstract)
feature_cols = [
    'HTTP_Method_Encoded', 'URL_Length', 'Payload_Length', 'Parameter_Count',
    'Script_Pattern', 'SQL_Pattern', 'CRLF_Pattern', 'Traversal_Pattern',
    'Overflow_Pattern', 'Suspicious_Parameter', 'CSRF_Indicator'
]

X = df[feature_cols]
y = df[target_col]  # string labels

# Encode target for training (XGBoost expects numeric)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- DEFINE MODELS --------------------
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                             use_label_encoder=False, eval_metric='mlogloss'),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
}

# -------------------- TRAIN & EVALUATE EACH MODEL --------------------
results = []
best_model = None
best_f1 = -1
best_name = ""

for name, model in models.items():
    print(f"\n[INFO] Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    
    # Save each model individually (for hybrid)
    model_filename = name.lower().replace(' ', '_') + '.pkl'
    joblib.dump(model, Path(MODEL_FOLDER) / model_filename)
    print(f"  [SAVED] {model_filename}")
    
    # Track best
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# -------------------- SAVE BEST MODEL --------------------
joblib.dump(best_model, Path(MODEL_FOLDER) / "best_model.pkl")
print(f"\n[INFO] Best model: {best_name} with F1 = {best_f1:.4f}")

# -------------------- SAVE COMMON ARTIFACTS --------------------
joblib.dump(scaler, Path(MODEL_FOLDER) / "scaler.pkl")
joblib.dump(method_encoder, Path(MODEL_FOLDER) / "method_encoder.pkl")
joblib.dump(target_encoder, Path(MODEL_FOLDER) / "target_encoder.pkl")
joblib.dump(feature_cols, Path(MODEL_FOLDER) / "feature_cols.pkl")

# Save comparison results
comparison_df = pd.DataFrame(results)
comparison_df.to_csv(Path(MODEL_FOLDER) / "model_comparison.csv", index=False)

print(f"[OK] All artifacts saved to '{MODEL_FOLDER}/'")

# -------------------- DETAILED REPORT FOR BEST MODEL --------------------
print("\n" + "="*60)
print(f"CLASSIFICATION REPORT FOR BEST MODEL ({best_name})")
print("="*60)
y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best, target_names=target_encoder.classes_))

# Save a summary report
report_path = Path(MODEL_FOLDER) / "training_summary.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("MODEL TRAINING SUMMARY\n")
    f.write("="*40 + "\n")
    f.write(f"Dataset: {DATA_FILE.name}\n")
    f.write(f"Total samples: {df.shape[0]}\n")
    f.write(f"Attack classes: {len(target_encoder.classes_)}\n")
    f.write(f"Train/Test split: {len(X_train)} / {len(X_test)}\n\n")
    f.write("Model Performance:\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\nBest model: " + best_name + f" (F1 = {best_f1:.4f})")
    f.write("\n\nDetailed Classification Report (Best Model):\n")
    f.write(classification_report(y_test, y_pred_best, target_names=target_encoder.classes_))

print(f"[OK] Training summary saved to {report_path}")
print("\n[SUCCESS] model.py finished. All models are ready for the Streamlit app!")
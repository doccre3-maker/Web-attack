import pandas as pd
import numpy as np
import random
from pathlib import Path

# -------------------- SETUP --------------------
np.random.seed(2026)          # reproducible
random.seed(2026)
SAMPLES_PER_CLASS = 200       # adjust as needed (e.g., 200 per class = 1800 total)
NOISE_RATE = 0.02             # 2% noise to avoid perfect patterns

# Define all classes (same as training)
attack_types = [
    'Normal',
    'SQL Injection',
    'XSS',
    'CRLF Injection',
    'Path Traversal',
    'Buffer Overflow',
    'Parameter Tampering',
    'Information Gathering',
    'CSRF'
]

# Severity mapping (just for reference, will be added to CSV)
severity_map = {
    'Normal': 'Low',
    'SQL Injection': 'Critical',
    'XSS': 'High',
    'CRLF Injection': 'High',
    'Path Traversal': 'High',
    'Buffer Overflow': 'Critical',
    'Parameter Tampering': 'Medium',
    'Information Gathering': 'Medium',
    'CSRF': 'Medium'
}

# -------------------- GENERATION FUNCTION --------------------
def generate_class_data(class_name, n):
    data = []
    
    # Base distributions for each class (similar to training but with slight variations)
    if class_name == 'Normal':
        method_probs = {'GET': 0.85, 'POST': 0.15}
        url_len = np.random.normal(30, 12, n).astype(int)
        payload_len = np.random.normal(20, 12, n).astype(int)
        param_count = np.random.poisson(1.2, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': NOISE_RATE,
            'CRLF': NOISE_RATE,
            'Traversal': NOISE_RATE,
            'Overflow': NOISE_RATE,
            'Suspicious': NOISE_RATE,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'SQL Injection':
        method_probs = {'GET': 0.4, 'POST': 0.6}
        url_len = np.random.normal(80, 35, n).astype(int)
        payload_len = np.random.normal(250, 110, n).astype(int)
        param_count = np.random.poisson(3, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': 0.98,
            'CRLF': NOISE_RATE,
            'Traversal': NOISE_RATE,
            'Overflow': NOISE_RATE,
            'Suspicious': 0.95,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'XSS':
        method_probs = {'GET': 0.7, 'POST': 0.3}
        url_len = np.random.normal(70, 30, n).astype(int)
        payload_len = np.random.normal(180, 90, n).astype(int)
        param_count = np.random.poisson(2.2, n).astype(int)
        patterns = {
            'Script': 0.98,
            'SQL': NOISE_RATE,
            'CRLF': NOISE_RATE,
            'Traversal': NOISE_RATE,
            'Overflow': NOISE_RATE,
            'Suspicious': 0.90,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'CRLF Injection':
        method_probs = {'GET': 0.5, 'POST': 0.5}
        url_len = np.random.normal(60, 25, n).astype(int)
        payload_len = np.random.normal(120, 60, n).astype(int)
        param_count = np.random.poisson(2, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': NOISE_RATE,
            'CRLF': 0.98,
            'Traversal': NOISE_RATE,
            'Overflow': NOISE_RATE,
            'Suspicious': 0.15,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'Path Traversal':
        method_probs = {'GET': 0.8, 'POST': 0.2}
        url_len = np.random.normal(100, 35, n).astype(int)
        payload_len = np.random.normal(110, 50, n).astype(int)
        param_count = np.random.poisson(2, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': NOISE_RATE,
            'CRLF': NOISE_RATE,
            'Traversal': 0.98,
            'Overflow': NOISE_RATE,
            'Suspicious': 0.25,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'Buffer Overflow':
        method_probs = {'GET': 0.3, 'POST': 0.7}
        url_len = np.random.normal(120, 35, n).astype(int)
        payload_len = np.random.normal(1200, 350, n).astype(int)
        param_count = np.random.poisson(3, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': NOISE_RATE,
            'CRLF': NOISE_RATE,
            'Traversal': NOISE_RATE,
            'Overflow': 0.98,
            'Suspicious': 0.35,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'Parameter Tampering':
        method_probs = {'GET': 0.6, 'POST': 0.4}
        url_len = np.random.normal(70, 25, n).astype(int)
        payload_len = np.random.normal(150, 70, n).astype(int)
        param_count = np.random.poisson(5, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': NOISE_RATE,
            'CRLF': NOISE_RATE,
            'Traversal': NOISE_RATE,
            'Overflow': NOISE_RATE,
            'Suspicious': 0.98,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'Information Gathering':
        method_probs = {'GET': 0.9, 'POST': 0.1}
        url_len = np.random.normal(220, 55, n).astype(int)
        payload_len = np.random.normal(90, 35, n).astype(int)
        param_count = np.random.poisson(6, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': NOISE_RATE,
            'CRLF': NOISE_RATE,
            'Traversal': 0.15,
            'Overflow': NOISE_RATE,
            'Suspicious': 0.15,
            'CSRF': NOISE_RATE
        }

    elif class_name == 'CSRF':
        method_probs = {'GET': 0.1, 'POST': 0.9}
        url_len = np.random.normal(60, 25, n).astype(int)
        payload_len = np.random.normal(90, 45, n).astype(int)
        param_count = np.random.poisson(2, n).astype(int)
        patterns = {
            'Script': NOISE_RATE,
            'SQL': NOISE_RATE,
            'CRLF': NOISE_RATE,
            'Traversal': NOISE_RATE,
            'Overflow': NOISE_RATE,
            'Suspicious': 0.15,
            'CSRF': 0.98
        }

    # Clip values to reasonable ranges
    url_len = np.clip(url_len, 5, 2000)
    payload_len = np.clip(payload_len, 5, 5000)
    param_count = np.clip(param_count, 0, 20)

    # Generate rows
    for i in range(n):
        method = np.random.choice(list(method_probs.keys()), p=list(method_probs.values()))
        # Apply noise with a small chance of flipping the bit
        script = 1 if np.random.rand() < patterns['Script'] else 0
        sql = 1 if np.random.rand() < patterns['SQL'] else 0
        crlf = 1 if np.random.rand() < patterns['CRLF'] else 0
        traversal = 1 if np.random.rand() < patterns['Traversal'] else 0
        overflow = 1 if np.random.rand() < patterns['Overflow'] else 0
        suspicious = 1 if np.random.rand() < patterns['Suspicious'] else 0
        csrf = 1 if np.random.rand() < patterns['CSRF'] else 0

        data.append({
            'HTTP_Method': method,
            'URL_Length': url_len[i],
            'Payload_Length': payload_len[i],
            'Parameter_Count': param_count[i],
            'Script_Pattern': script,
            'SQL_Pattern': sql,
            'CRLF_Pattern': crlf,
            'Traversal_Pattern': traversal,
            'Overflow_Pattern': overflow,
            'Suspicious_Parameter': suspicious,
            'CSRF_Indicator': csrf,
            'Attack_Type': class_name,
            'Severity': severity_map[class_name]
        })
    
    return pd.DataFrame(data)

# -------------------- GENERATE FULL TEST SET --------------------
print("🔄 Generating test dataset (200 samples per class)...")
all_dfs = []
for attack in attack_types:
    print(f"   - Generating {attack}...")
    df = generate_class_data(attack, SAMPLES_PER_CLASS)
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
# Shuffle
final_df = final_df.sample(frac=1, random_state=2026).reset_index(drop=True)

# Save to CSV
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "test_dataset.csv"
final_df.to_csv(output_path, index=False)

print(f"✅ Test dataset saved to: {output_path}")
print(f"   Total rows: {len(final_df)}")
print(f"   Class distribution:\n{final_df['Attack_Type'].value_counts()}")
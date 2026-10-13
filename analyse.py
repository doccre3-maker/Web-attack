import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -------------------- CONFIGURATION --------------------
DATA_FOLDER = r"C:\Users\user\Desktop\web_attack\data"   # your dataset path
OUTPUT_FOLDER = "analyze"  # will be created inside the project root

# Create output folder if it doesn't exist
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# -------------------- FIND THE CSV FILE --------------------
csv_files = list(Path(DATA_FOLDER).glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER}")
# Use the first CSV found (or you can manually specify the filename)
CSV_FILE = csv_files[0]
print(f"[OK] Found dataset: {CSV_FILE.name}")

# -------------------- LOAD DATA --------------------
df = pd.read_csv(CSV_FILE)
print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------- 1. BASIC INFORMATION --------------------
info = {
    "File Name": CSV_FILE.name,
    "Total Rows": df.shape[0],
    "Total Columns": df.shape[1],
    "Missing Values": df.isnull().sum().sum(),
    "Duplicate Rows": df.duplicated().sum(),
}

# -------------------- 2. CLASS BALANCE --------------------
# Assuming the target column is 'Attack_Type' (as per your abstract)
target_col = 'Attack_Type'
if target_col not in df.columns:
    # fallback: try to find column with "attack" or "class" in name
    possible = [col for col in df.columns if 'attack' in col.lower() or 'class' in col.lower()]
    if possible:
        target_col = possible[0]
    else:
        target_col = df.columns[-1]  # last column as fallback

print(f"[INFO] Using target column: '{target_col}'")

class_counts = df[target_col].value_counts()
class_percentages = df[target_col].value_counts(normalize=True) * 100
balance_df = pd.DataFrame({
    'Count': class_counts,
    'Percentage': class_percentages
})
# Mark classes as balanced if count is between 1000 and 1500 (adjust as needed)
balance_df['Is Balanced'] = balance_df['Count'].apply(
    lambda x: 'YES' if (x >= 1000 and x <= 1500) else 'NO'
)

# -------------------- 3. FEATURE STATISTICS --------------------
# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_stats = df[numeric_cols].describe().T
numeric_stats['skew'] = df[numeric_cols].skew()
numeric_stats['kurtosis'] = df[numeric_cols].kurtosis()

# -------------------- 4. SAVE TEXT SUMMARY (FIXED: UTF-8 encoding) --------------------
summary_path = Path(OUTPUT_FOLDER) / "dataset_analysis_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:  # <--- THIS FIXES THE ERROR!
    f.write("="*60 + "\n")
    f.write("DATASET ANALYSIS REPORT\n")  # Removed emoji to be extra safe
    f.write("="*60 + "\n\n")
    
    f.write("BASIC INFO\n")
    f.write("-"*40 + "\n")
    for k, v in info.items():
        f.write(f"{k}: {v}\n")
    f.write("\n")
    
    f.write("CLASS DISTRIBUTION (BALANCE CHECK)\n")
    f.write("-"*40 + "\n")
    f.write(balance_df.to_string())
    f.write("\n\n")
    
    f.write("NUMERIC FEATURE STATISTICS\n")
    f.write("-"*40 + "\n")
    f.write(numeric_stats.to_string())
    f.write("\n\n")
    
    f.write("CATEGORICAL COLUMNS\n")
    f.write("-"*40 + "\n")
    for col in categorical_cols:
        f.write(f"{col}: {df[col].nunique()} unique values\n")
        if df[col].nunique() <= 10:
            f.write(df[col].value_counts().to_string() + "\n\n")
        else:
            f.write(f"   (Top 5 values):\n{df[col].value_counts().head().to_string()}\n\n")
    
    f.write("="*60 + "\n")
    f.write("END OF REPORT\n")

print(f"[OK] Text summary saved to {summary_path}")

# -------------------- 5. GENERATE PLOTS --------------------
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 5.1 Class Distribution Bar Chart
plt.figure()
ax = balance_df['Count'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Class Distribution (Attack Types)', fontsize=14)
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
# Add percentage labels on bars
for i, (idx, row) in enumerate(balance_df.iterrows()):
    ax.text(i, row['Count'] + 10, f"{row['Percentage']:.1f}%", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(Path(OUTPUT_FOLDER) / "class_distribution.png", dpi=300)
plt.close()

# 5.2 Feature Distributions (histograms) for numeric features
if numeric_cols:
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3  # 3 per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows*4))
    # Flatten axes safely
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], color='lightgreen', edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_FOLDER) / "feature_distributions.png", dpi=300)
    plt.close()

# 5.3 Correlation Heatmap (only numeric)
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_FOLDER) / "correlation_heatmap.png", dpi=300)
    plt.close()

# 5.4 Boxplots for each numeric feature grouped by target (if target is categorical)
if target_col in df.columns and len(numeric_cols) > 0:
    # Only if target has reasonable number of classes (<=15)
    if df[target_col].nunique() <= 15:
        # Plot boxplots for each numeric feature vs target
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target_col, y=col, data=df, palette='Set2')
            plt.title(f'Boxplot of {col} by {target_col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(Path(OUTPUT_FOLDER) / f"boxplot_{col}_by_target.png", dpi=300)
            plt.close()

print(f"[OK] All plots saved to '{OUTPUT_FOLDER}/' folder")

# -------------------- 6. SUMMARY FOR CONSOLE --------------------
print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Total samples: {df.shape[0]}")
print(f"Attack types: {df[target_col].nunique()} unique classes")
print("\nClass balance (top 5):")
print(balance_df.head())
print(f"\n[OK] Check the '{OUTPUT_FOLDER}' folder for detailed report and visualizations.")
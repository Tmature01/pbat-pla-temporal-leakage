"""
Cross-material validation using raw Excel data.
- Extract 6 materials (PBAT+PLA, PLA, PBAT1, PBAT2, PBAT+PGA, PBAT+Talc)
- Train LS-MPR on PBAT+PLA, predict on the other 5
- Generate paper-ready comparison tables and metrics
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import PolyModel

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")


DATA_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'original_workbooks')
DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEP = "=" * 70

def extract_material_data(excel_path, sheet_name):
    """Extract clean Days + Sample CO2 + Degradation % from raw Excel"""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # Structure: row 0=headers, row 1=sub-header, rows 2-181=data
    raw_days = pd.to_numeric(df.iloc[2:, 0], errors='coerce')
    # Col 1: blank CO2, Col 2: cellulose CO2, Col 5: sample net CO2, Col 7: sample degradation %
    blank_co2 = pd.to_numeric(df.iloc[2:, 1], errors='coerce')
    cellulose_co2 = pd.to_numeric(df.iloc[2:, 2], errors='coerce')
    sample_co2_raw = pd.to_numeric(df.iloc[2:, 3], errors='coerce')
    sample_net_co2 = pd.to_numeric(df.iloc[2:, 5], errors='coerce')
    sample_degrad_pct = pd.to_numeric(df.iloc[2:, 7], errors='coerce')

    data = pd.DataFrame({
        'Days': raw_days,
        'Blank_CO2': blank_co2,
        'Cellulose_CO2': cellulose_co2,
        'Sample_CO2_raw': sample_co2_raw,
        'Sample_CO2_net': sample_net_co2,
        'Degradation_Pct': sample_degrad_pct,
    })
    data = data[data['Days'].notna()].copy()
    # Ensure non-negative
    for col in ['Sample_CO2_net', 'Degradation_Pct']:
        if col in data.columns:
            data[col] = data[col].clip(lower=0)
    return data


# ================================================================
# STEP 1: Extract all materials
# ================================================================
f2_path = os.path.join(DATA_SRC, '1-几种PBAT膜样品数据.xlsx')
xls2 = pd.ExcelFile(f2_path)
sheets_available = [s for s in xls2.sheet_names if s != 'PBAT+PLA']
sheets_available.insert(0, 'PBAT+PLA')  # Train first, then test others

all_data = {}
for sn in xls2.sheet_names:
    material_name = sn.replace('~', '_').replace('%', 'pct')
    df_clean = extract_material_data(f2_path, sn)
    all_data[material_name] = df_clean
    # Save clean CSV
    out_path = os.path.join(DATA_OUT, f'validation_{material_name}.csv')
    df_clean.to_csv(out_path, index=False)
    print(f"Extracted {sn}: {len(df_clean)} rows, CO2 net range [{df_clean['Sample_CO2_net'].min():.1f}, {df_clean['Sample_CO2_net'].max():.1f}]")

# ================================================================
# STEP 2: Train Days-only model on PBAT+PLA, validate on others
# ================================================================
lines = []
def p(s):
    print(s); lines.append(s)

p(SEP)
p("CROSS-MATERIAL VALIDATION")
p("Model: LS-MPR d=2, trained on PBAT+PLA raw data (Days-only)")
p(SEP)

# Train model: PBAT+PLA, Days → CO2_net
df_train = all_data['PBAT+PLA'].copy()
X_train_full = df_train[['Days']].values
y_train_full = df_train[['Sample_CO2_net']].values

# Remove outliers
y_mean, y_std = y_train_full.mean(), y_train_full.std()
mask = np.abs(y_train_full.flatten() - y_mean) <= 3 * y_std
X_train = X_train_full[mask]
y_train = y_train_full[mask]

# Standardize
y_zs = StandardScaler()
y_train_s = y_zs.fit_transform(y_train)

# LS-MPR d=2 (Days → CO2)
poly = PolyModel(degree=2)
X_train_poly = poly.create_features(X_train)

ols = LinearRegression(fit_intercept=False)
ols.fit(X_train_poly, y_train_s.flatten())

# Report training performance (y_train is already in original scale)
y_pred_train = ols.predict(X_train_poly)
y_pred_train_real = y_zs.inverse_transform(y_pred_train.reshape(-1, 1))
train_r2 = r2_score(y_train, y_pred_train_real)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_real))
train_rho, _ = spearmanr(y_train.flatten(), y_pred_train_real.flatten())

p(f"\nPBAT+PLA Training (Days-only model, d=2):")
p(f"  R2={train_r2:.4f}  RMSE={train_rmse:.2f}  Spearman rho={train_rho:.4f}")

# Coefficients
coefs = ols.coef_
feature_names = poly.poly.get_feature_names_out(['Days'])
p(f"  Model: y = {' + '.join([f'{c:.4f}*{n}' for c,n in zip(coefs, feature_names)])}")

# ================================================================
# STEP 3: Validate on other 5 materials
# ================================================================
p(f"\n{SEP}")
p("Validation on other materials (same experimental conditions):")
p(SEP)

results = []

for material_name, df_test in all_data.items():
    X_test = df_test[['Days']].values
    y_test_raw = df_test[['Sample_CO2_net']].values

    # Remove outliers
    ym, ys = y_test_raw.mean(), y_test_raw.std()
    mask_t = np.abs(y_test_raw.flatten() - ym) <= 3 * ys
    X_test_c = X_test[mask_t]
    y_test = y_test_raw[mask_t]
    y_test_s = y_zs.transform(y_test)

    # Predict
    X_test_poly = poly.create_features(X_test_c)
    y_pred_s = ols.predict(X_test_poly)
    y_pred = y_zs.inverse_transform(y_pred_s.reshape(-1, 1))

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rho, _ = spearmanr(y_test.flatten(), y_pred.flatten())

    tag = " (TRAIN)" if material_name == 'PBAT+PLA' else ""
    p(f"  {material_name:25s}{tag}: R2={r2:.4f}  RMSE={rmse:.2f}  rho={rho:.4f}")
    results.append({
        'Material': material_name.replace('PBAT+PLA', 'PBAT+PLA (train)'),
        'Samples': len(y_test),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'Spearman_rho': round(rho, 4),
    })

# ================================================================
# STEP 4: Per-material independent model comparison
# ================================================================
p(f"\n{SEP}")
p("PER-MATERIAL INDEPENDENT LS-MPR (each material fitted separately):")
p(SEP)

for material_name, df in all_data.items():
    X = df[['Days']].values
    y = df[['Sample_CO2_net']].values

    ym, ys = y.mean(), y.std()
    mask = np.abs(y.flatten() - ym) <= 3 * ys
    X_c, y_c = X[mask], y[mask]

    zs = StandardScaler()
    y_s = zs.fit_transform(y_c)

    p_local = PolyModel(degree=2)
    X_poly = p_local.create_features(X_c)
    ols_local = LinearRegression(fit_intercept=False)
    ols_local.fit(X_poly, y_s.flatten())
    y_pred_s = ols_local.predict(X_poly)
    y_pred = zs.inverse_transform(y_pred_s.reshape(-1, 1))

    r2 = r2_score(y_c, y_pred)
    rmse = np.sqrt(mean_squared_error(y_c, y_pred))
    rho, _ = spearmanr(y_c.flatten(), y_pred.flatten())

    p(f"  {material_name:25s}: R2={r2:.4f}  RMSE={rmse:.2f}  rho={rho:.4f}  CO2_range=[{y_c.min():.0f},{y_c.max():.0f}]")

# ================================================================
# STEP 5: Degradation rate comparison
# ================================================================
p(f"\n{SEP}")
p("DEGRADATION RATE COMPARISON (CO2 at Day 90 and Day 180):")
p(SEP)

for material_name, df in all_data.items():
    # Get CO2 at specific days
    for target_day in [30, 90, 180]:
        row = df[df['Days'].round() == target_day]
        if len(row) > 0:
            co2 = row['Sample_CO2_net'].values[0]
            degrad = row['Degradation_Pct'].values[0] if 'Degradation_Pct' in row.columns else 0
            p(f"  {material_name:25s}  Day{target_day:3d}: CO2={co2:7.2f}g   Degradation={degrad:.2f}%")

# Save report
report_path = os.path.join(RESULTS_DIR, 'cross_material_validation.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\n{SEP}")
print(f"Report: {report_path}")

# Save results CSV
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(RESULTS_DIR, 'cross_material_validation.csv'), index=False)
print(f"CSV: {RESULTS_DIR}/cross_material_validation.csv")
print("DONE")

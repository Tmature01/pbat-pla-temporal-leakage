"""
Figure: 时间数据泄漏概念示意图 (Conceptual: Why random split leaks on temporal data)
Generates data for:
  1. Protocol comparison: same degradation curve, 4 train/test partitions
  2. Temporal overlap: KDE of Days in train vs test under Random vs Temporal
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Load C1 (Pure PBAT) as reference curve ──
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

backup = str(_REPO_ROOT / "data" / "pbat_pla_experimental_543.csv")
df = pd.read_csv(backup).dropna(subset=['Days','Temperature','Humidity','Ratio','Compost_Volume','CO2_Release'])

c1 = df[(df['Temperature'] == 58) & (df['Humidity'] == 60) &
        (df['Ratio'] == 100) & (np.isclose(df['Compost_Volume'], 0.5))].sort_values('Days').copy()

print(f'C1 (Pure PBAT): {len(c1)} rows, Days=[{c1.Days.min():.0f},{c1.Days.max():.0f}]')

days = c1['Days'].values
co2 = c1['CO2_Release'].values

# ─────────────────────────────────────────────────────────
# 1. Four protocol partitions on the SAME C1 curve
# ─────────────────────────────────────────────────────────
np.random.seed(42)

# (a) Random 80/20 split
n = len(c1)
idx = np.arange(n)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
protocol_random = np.full(n, '', dtype=object)
protocol_random[train_idx] = 'Train'
protocol_random[test_idx] = 'Test'

# (b) Temporal holdout (train <= 120, test > 120)
protocol_temporal = np.where(days <= 120, 'Train', 'Test')

# (c) Temporal shuffle (permute Days, then random 80/20)
days_shuffled = days.copy()
np.random.shuffle(days_shuffled)
# The "train" on shuffled data — all points used for fitting (no separate test in shuffle diagnostic)
# For the illustration: show that after shuffle, Days no longer relate to CO2
protocol_shuffle = np.full(n, 'Shuffled', dtype=object)

# (d) LOCO — for C1, train = C2+C3, test = C1
# But for the schematic, use C1's own timeline
protocol_loco = np.full(n, 'Test (held-out)', dtype=object)

# ── Build DataFrame ──
rows = []
for i in range(n):
    # Original
    rows.append({
        'Day': days[i], 'CO2': co2[i],
        'Protocol': 'All data',
        'Partition': 'Data point',
        'Split': 'original',
    })
    # Random split
    rows.append({
        'Day': days[i], 'CO2': co2[i] if protocol_random[i] == 'Train' else np.nan,
        'Protocol': 'Random split',
        'Partition': protocol_random[i],
        'Split': 'random',
    })
    rows.append({
        'Day': days[i], 'CO2': co2[i] if protocol_random[i] == 'Test' else np.nan,
        'Protocol': 'Random split',
        'Partition': protocol_random[i],
        'Split': 'random',
    })
    # Actually simplify: one row per protocol per day
    # Let me restructure...

# Redo — simpler structure
rows_simple = []
for i in range(n):
    rows_simple.append({
        'Day': days[i],
        'CO2': co2[i],
        'Random_Partition': 'Train' if i in train_idx else 'Test',
        'Temporal_Partition': 'Train' if days[i] <= 120 else 'Test',
    })

df_protocols = pd.DataFrame(rows_simple)
df_protocols.to_csv(str(FIGURE_DATA / "Fig_DataLeakage_Protocols.csv"), index=False)
print(f'Saved: Fig_DataLeakage_Protocols.csv ({len(df_protocols)} rows)')

# ─────────────────────────────────────────────────────────
# 2. Temporal overlap: KDE of Days in train vs test
# ─────────────────────────────────────────────────────────
# Use ALL 3 conditions pooled for realistic split
all_data = df.copy()
all_days = all_data['Days'].values
all_n = len(all_data)

# Random split
idx_all = np.arange(all_n)
train_r, test_r = train_test_split(idx_all, test_size=0.2, random_state=42)
days_train_r = all_days[train_r]
days_test_r = all_days[test_r]

# Temporal split
train_t_mask = all_days <= 120
days_train_t = all_days[train_t_mask]
days_test_t = all_days[~train_t_mask]

# Generate KDE data for each
from scipy.stats import gaussian_kde

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")


day_grid = np.linspace(0, 185, 200)

kde_data = {'Day': day_grid}

for name, train_d, test_d in [
    ('Random', days_train_r, days_test_r),
    ('Temporal', days_train_t, days_test_t),
]:
    if len(train_d) > 2:
        kde_tr = gaussian_kde(train_d)(day_grid)
    else:
        kde_tr = np.zeros_like(day_grid)
    if len(test_d) > 2:
        kde_te = gaussian_kde(test_d)(day_grid)
    else:
        kde_te = np.zeros_like(day_grid)

    # Normalize to peak=1 for visualization
    kde_tr = kde_tr / kde_tr.max() if kde_tr.max() > 0 else kde_tr
    kde_te = kde_te / kde_te.max() if kde_te.max() > 0 else kde_te

    kde_data[f'{name}_Train_Density'] = kde_tr
    kde_data[f'{name}_Test_Density'] = kde_te

df_kde = pd.DataFrame(kde_data)
df_kde.to_csv(str(FIGURE_DATA / "Fig_DataLeakage_KDE.csv"), index=False)
print(f'Saved: Fig_DataLeakage_KDE.csv ({len(df_kde)} grid points)')

# ─────────────────────────────────────────────────────────
# 3. Summary stats for annotation
# ─────────────────────────────────────────────────────────
print(f'\n--- Summary stats ---')
print(f'Random split: Train n={len(days_train_r)}, Test n={len(days_test_r)}')
print(f'  Train Days range: [{days_train_r.min():.0f}, {days_train_r.max():.0f}]')
print(f'  Test Days range: [{days_test_r.min():.0f}, {days_test_r.max():.0f}]')
print(f'  Overlap: {min(days_train_r.max(), days_test_r.max()) - max(days_train_r.min(), days_test_r.min()):.0f} days')

print(f'Temporal holdout: Train n={len(days_train_t)}, Test n={len(days_test_t)}')
print(f'  Train Days range: [{days_train_t.min():.0f}, {days_train_t.max():.0f}]')
print(f'  Test Days range: [{days_test_t.min():.0f}, {days_test_t.max():.0f}]')
print(f'  Overlap: 0 days (clean separation)')

# ─────────────────────────────────────────────────────────
# 4. Illustration data: interpolate vs extrapolate
# ─────────────────────────────────────────────────────────
# For one test point in random split, show its nearest train neighbors
# For one test point in temporal, show the gap to nearest train point

# Find a test point near Day 100 in random split
test_days_r = days_test_r
test_day_example_r = test_days_r[np.argmin(np.abs(test_days_r - 100))]
train_days_r = days_train_r
# Find nearest train points
distances_r = np.abs(train_days_r - test_day_example_r)
nearest_idx = np.argsort(distances_r)[:2]
nearest_train_days = train_days_r[nearest_idx]

print(f'\n--- Interpolation vs Extrapolation ---')
print(f'Random split example:')
print(f'  Test point at Day={test_day_example_r:.0f}')
print(f'  Nearest train points at Days={nearest_train_days[0]:.0f}, {nearest_train_days[1]:.0f}')
print(f'  Gap: {np.min(distances_r):.0f} days → model can INTERPOLATE')

# For temporal, test starts at Day 121
test_day_example_t = 150.0
print(f'Temporal holdout example:')
print(f'  Test point at Day={test_day_example_t:.0f}')
print(f'  Nearest train point at Day=120')
print(f'  Gap: {test_day_example_t - 120:.0f} days → model must EXTRAPOLATE')

# Save illustration data
illustration = pd.DataFrame({
    'Concept': ['Random split', 'Random split', 'Temporal holdout', 'Temporal holdout'],
    'Role': ['Test point', 'Nearest train', 'Test point', 'Nearest train'],
    'Day': [test_day_example_r, nearest_train_days[0], test_day_example_t, 120.0],
    'CO2_approx': [co2[np.argmin(np.abs(days - test_day_example_r))],
                    co2[np.argmin(np.abs(days - nearest_train_days[0]))],
                    co2[np.argmin(np.abs(days - test_day_example_t))],
                    co2[np.argmin(np.abs(days - 120.0))]],
    'Gap_days': [np.min(distances_r), np.min(distances_r), 30.0, 30.0],
})
illustration.to_csv(str(FIGURE_DATA / "Fig_DataLeakage_Illustration.csv"), index=False)
print(f'\nSaved: Fig_DataLeakage_Illustration.csv')

print('\n=== All data saved ===')
print('Fig_DataLeakage_Protocols.csv   — C1 curve with Random & Temporal partition labels')
print('Fig_DataLeakage_KDE.csv         — Train/Test Day distributions under both protocols')
print('Fig_DataLeakage_Illustration.csv — Interpolation vs Extrapolation demo points')

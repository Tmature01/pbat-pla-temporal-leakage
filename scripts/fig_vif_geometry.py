"""
Figure: VIF=∞ rank deficiency geometric visualization
Panel (a): 3D scatter — 3 conditions in environmental feature space
Panel (b): 2D analogy — why 3 points can't determine 4D space
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.linalg import svd

# ─────────────────────────────────────────────────────────
# PANEL (a): 3D feature space coordinates
# ─────────────────────────────────────────────────────────

# Condition coordinates (environmental features only, excluding Days)
cond_points = {
    'C1 (T=50, H=70, R=70, CV=0.90)':  [50, 70, 70, 0.90],
    'C2 (T=58, H=60, R=100, CV=0.50)': [58, 60, 100, 0.50],
    'C3 (T=58, H=70, R=0, CV=0.75)':   [58, 70, 0, 0.75],
}

feature_names = ['Temperature (°C)', 'Humidity (%)', 'Ratio (%)', 'Compost_Volume']

# Normalize each feature to [0,1] for visualization
feat_matrix = np.array(list(cond_points.values()))  # 3x4
feat_min = feat_matrix.min(axis=0)
feat_max = feat_matrix.max(axis=0)
feat_norm = (feat_matrix - feat_min) / (feat_max - feat_min + 1e-10)

# Save 3D coordinates (using first 3 features for axes)
df_3d = pd.DataFrame({
    'Condition': list(cond_points.keys()),
    'T_raw': feat_matrix[:, 0],
    'H_raw': feat_matrix[:, 1],
    'Ratio_raw': feat_matrix[:, 2],
    'CV_raw': feat_matrix[:, 3],
    'X_T_norm': feat_norm[:, 0],
    'Y_H_norm': feat_norm[:, 1],
    'Z_Ratio_norm': feat_norm[:, 2],
    'CV_norm': feat_norm[:, 3],
})
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df_3d.to_csv(str(FIGURE_DATA / "Fig_VIF_3D_Points.csv"), index=False)
print('Saved: Fig_VIF_3D_Points.csv')
print(df_3d[['Condition', 'T_raw', 'H_raw', 'Ratio_raw', 'CV_raw']].to_string(index=False))

# ── Rank analysis of environmental feature matrix ──
# 3 conditions x 4 environmental features
env_matrix = feat_matrix  # 3x4
rank_env = np.linalg.matrix_rank(env_matrix)
print(f'\nEnvironmental feature matrix: {env_matrix.shape[0]} conditions x {env_matrix.shape[1]} features')
print(f'Matrix rank: {rank_env} (need >= {env_matrix.shape[1]} for unique solution)')
print(f'Rank deficiency: {env_matrix.shape[1] - rank_env} missing dimensions')

# Full design matrix with polynomial terms and Days
# 3 conditions x 181 days each = 543 observations
# After polynomial expansion (degree 2, 5 features): 21 terms
# But effective rank is much lower due to condition-level confounding

# Environmental submatrix: 3 unique condition rows
# With polynomial expansion of environmental features only (4 features, degree 2): C(4+2,2) = 15 terms
# But only 3 unique rows → max rank = 3 out of 15

print(f'\nWith polynomial expansion (degree 2):')
print(f'  Environmental terms only (4 features): C(6,2) = 15 polynomial terms')
print(f'  Unique condition rows: 3')
print(f'  Max rank of env-only design matrix: min(3, 15) = 3')
print(f'  Rank deficiency: 15 - 3 = 12')

# SVD of the normalized environmental matrix
U, s, Vt = svd(feat_norm, full_matrices=False)
print(f'\nSVD of normalized environmental matrix (3x4):')
print(f'  Singular values: {s}')
print(f'  Non-zero singular values: {np.sum(s > 1e-10)}')
print(f'  Condition number: {s[0]/s[-1]:.2e}')

# ─────────────────────────────────────────────────────────
# PANEL (b): 2D analogy data
# ─────────────────────────────────────────────────────────

# Conceptual illustration data
analogy_data = pd.DataFrame({
    'Scenario': [
        '2 points → 1D line (solvable)',
        '2 points → 1D line (solvable)',
        '3 points → 2D plane (solvable)',
        '3 points → 2D plane (solvable)',
        '3 points → 2D plane (solvable)',
        'This study: PBAT/PLA',
    ],
    'n_points': [2, 2, 3, 3, 3, 3],
    'ambient_dim': [2, 3, 3, 4, 5, 4],
    'target_rank': [1, 1, 2, 2, 2, 4],
    'achievable_rank': [1, 1, 2, 2, 2, 3],
    'solvable': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No (rank 3 < 4 needed)'],
})
analogy_data.to_csv(str(FIGURE_DATA / "Fig_VIF_Analogy.csv"), index=False)
print(f'\nSaved: Fig_VIF_Analogy.csv')

# ── Correlation matrix of environmental features ──
# All features are observed at discrete Days, so pairwise correlation is high
# due to the experimental design structure
env_df = pd.DataFrame(feat_matrix, columns=feature_names,
                       index=['C1', 'C2', 'C3'])
corr_env = pd.DataFrame(np.corrcoef(feat_matrix.T),
                         index=feature_names, columns=feature_names)

print(f'\nEnvironmental feature correlation (n=3 conditions):')
print(corr_env.round(4).to_string())

# ── The plane equation (for the 3D visualization) ──
# 3 points in 3D define a plane. Compute its equation.
# Points: C1(50,70,70), C2(58,60,100), C3(58,70,0)
p1 = feat_matrix[0, :3]  # C1
p2 = feat_matrix[1, :3]  # C2
p3 = feat_matrix[2, :3]  # C3

# Two vectors in the plane
v1 = p2 - p1
v2 = p3 - p1
# Normal vector
normal = np.cross(v1, v2)
normal = normal / np.linalg.norm(normal)
d = -np.dot(normal, p1)

print(f'\nPlane in (T, H, Ratio) space:')
print(f'  {normal[0]:.3f}*T + {normal[1]:.3f}*H + {normal[2]:.3f}*Ratio + ({d:.3f}) = 0')
print(f'  Normal vector: {normal}')

# Generate grid points on the plane for visualization
# Use T and H as free variables, solve for Ratio
t_grid = np.linspace(48, 60, 15)
h_grid = np.linspace(58, 72, 15)
TT, HH = np.meshgrid(t_grid, h_grid)
# Plane equation: nx*T + ny*H + nz*Ratio + d = 0
# Ratio = -(nx*T + ny*H + d) / nz
plane_ratio = -(normal[0]*TT + normal[1]*HH + d) / normal[2]

plane_df = pd.DataFrame({
    'T': TT.flatten(),
    'H': HH.flatten(),
    'Ratio': plane_ratio.flatten(),
})
plane_df.to_csv(str(FIGURE_DATA / "Fig_VIF_Plane_Grid.csv"), index=False)
print(f'\nSaved: Fig_VIF_Plane_Grid.csv ({len(plane_df)} grid points for the 2D plane)')

# Also compute the plane including CV as 4th dimension
# The 3 points in 4D span a 2D subspace
# Any linear combination of the 3 points gives a vector in this subspace
print(f'\nGeometric summary:')
print(f'  3 condition points in 4D environmental space')
print(f'  They span a 2D plane (not a 3D volume)')
print(f'  Any 4th condition would determine a unique 3D hyperplane')
print(f'  Thus: VIF = infinity (features perfectly collinear within the 3-condition design)')
print(f'\nMinimum conditions needed for 4 environmental features: 4')
print(f'Current conditions: 3')
print(f'Missing: 1 additional condition to break collinearity')

# ─────────────────────────────────────────────────────────
# Summary table for annotation
# ─────────────────────────────────────────────────────────
summary = pd.DataFrame({
    'Item': [
        'Number of experimental conditions',
        'Number of environmental features',
        'Rank of environmental design matrix',
        'Minimum conditions needed',
        'Condition number (env matrix)',
        'Effective dimension spanned',
        'Remaining degrees of freedom',
    ],
    'Value': [
        '3 (C1, C2, C3)',
        '4 (T, H, Ratio, CV)',
        f'{rank_env}',
        f'{feat_matrix.shape[1]} (= number of features)',
        f'{s[0]/s[-1]:.2e} (near-singular)',
        f'{rank_env} (a 2D plane in 4D space)',
        f'{feat_matrix.shape[1] - rank_env} (unobservable dimensions)',
    ],
})
summary.to_csv(str(FIGURE_DATA / "Fig_VIF_Summary.csv"), index=False)
print(f'\nSaved: Fig_VIF_Summary.csv')
print(summary.to_string(index=False))
print('\nDone.')
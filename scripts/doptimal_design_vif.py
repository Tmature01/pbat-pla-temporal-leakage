# -*- coding: utf-8 -*-
"""Reproducible D-optimal five-condition design and VIF computation.

Manuscript (sc-2026-08315g, ACS Sustainable Chemistry & Engineering):
"Temporal Data Leakage Inflates Machine Learning Performance in Compostable
PBAT/PLA Polymer Biodegradation Modeling".

Reconstructs the claim in Section 4 and the SI Table S3 note:

    "a D-optimal selection of only five of the 36 candidate condition
    combinations restores full-rank estimation of the four environmental
    main effects and reduces their VIFs from infinity to ~1.07."

Method
------
1. Enumerate the 2x2x3x3 = 36 full-factorial candidate conditions for the
   four environmental features:
       Temperature     in {50, 58}            (deg C)
       Humidity        in {60, 70}            (%)
       PBAT Ratio      in {0, 70, 100}        (%)
       Compost Volume  in {0.50, 0.75, 0.90}  (kg)
2. For every 5-condition subset (C(36,5) = 376,992), build the main-effects
   design matrix  X = [1 | T | H | R | CV]  (5 rows x 5 columns).
3. Keep the subset maximizing det(X'X)  (D-optimality).
4. Report the selected conditions and their variance inflation factors.

Only numpy is required.  Runtime is a few seconds.

Note: the D-optimal design is not unique (factor-level symmetry); every
D-optimal design yields VIF ~= 1.07 for all four features.  The design printed
here is the first optimum in the canonical enumeration order and matches the
five conditions listed in the SI Table S3 note.
"""

import sys
import io
from itertools import combinations

import numpy as np

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# --- factor levels (manuscript Section 2.2, 2 x 2 x 3 x 3 = 36) ---
T = [50, 58]
H = [60, 70]
R = [0, 70, 100]
CV = [0.50, 0.75, 0.90]

candidates = [(t, h, r, cv) for t in T for h in H for r in R for cv in CV]
assert len(candidates) == 36, "expected 2 x 2 x 3 x 3 = 36 candidates"

DEG = "\u00b0"  # degree sign


def vif(rows):
    """VIF of the four features (T, H, R, CV); inf = perfect collinearity.

    VIF_j = 1 / (1 - R2_j), where R2_j comes from regressing feature j on the
    other three features (with an intercept) via least squares.
    """
    X = np.asarray(rows, dtype=float)
    n, p = X.shape
    out = []
    for j in range(p):
        y = X[:, j]
        others = [X[:, k] for k in range(p) if k != j]
        Xo = np.column_stack([np.ones(n)] + others)
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        resid = y - Xo @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append(1.0 / (1.0 - r2) if r2 < 1.0 else np.inf)
    return np.asarray(out)


def d_criterion(rows):
    X = np.column_stack([np.ones(len(rows)), np.asarray(rows, dtype=float)])
    return float(np.linalg.det(X.T @ X))


def main():
    # Sanity check: the three conditions actually used are rank-deficient.
    current = [(50, 70, 70, 0.90), (58, 60, 100, 0.50), (58, 70, 0, 0.75)]
    print("Current 3 conditions VIF =", vif(current), "(expect all inf)")

    # Exhaustive D-optimal search over all 5-condition subsets.
    best_det, best_idx = -1.0, None
    for idx in combinations(range(36), 5):
        d = d_criterion([candidates[i] for i in idx])
        if d > best_det:
            best_det, best_idx = d, idx

    best_rows = [candidates[i] for i in best_idx]
    best_vif = vif(best_rows)

    print("D-optimal det(X'X) =", best_det)
    print("Selected 5 conditions (T, H, R, CV):")
    for t, h, r, cv in best_rows:
        print(f"  ({t} {DEG}C, {h}%, {r}%, {cv:.2f} kg)")
    print("VIF (Temperature, Humidity, Ratio, Compost Volume) =",
          np.round(best_vif, 4))
    print("VIF range =", round(best_vif.min(), 3), "-", round(best_vif.max(), 3))


if __name__ == "__main__":
    main()

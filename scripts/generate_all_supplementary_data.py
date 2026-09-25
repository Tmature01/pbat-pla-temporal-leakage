"""
Extract ALL data for supplementary tables S1-S18.
"""
import pandas as pd, numpy as np, os, sys, json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, DotProduct
from scipy.stats import spearmanr, ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import PolyModel
from models.co2_model import CO2Model
from models.residual_model import ResidualModel
from models.strength_model import StrengthModel

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ, 'data')
RESULTS = os.path.join(PROJ, 'results')
os.makedirs(RESULTS, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, 'pbat_pla_experimental_543.csv'))
df_aug = pd.read_csv(os.path.join(DATA_DIR, 'pbat_pla_augmented_1580.csv'))
FEATURES = ['Days','Temperature','Humidity','Ratio','Compost_Volume']
TARGETS = ['CO2_Release','Residual_Rate','Tensile_Strength']
df['Cond'] = df.apply(lambda r: f"T{int(r.Temperature)}_H{int(r.Humidity)}_R{int(r.Ratio)}_CV{r.Compost_Volume}", axis=1)
CONDS = sorted(df['Cond'].unique())
SEED = 42; np.random.seed(SEED)

def preprocess(Xtr, ytr, Xte, yte):
    for ci in range(Xtr.shape[1]):
        col = Xtr[:,ci]; m,s = col.mean(),col.std()
        if s>1e-10: Xtr,ytr = Xtr[np.abs(col-m)<=3*s], ytr[np.abs(col-m)<=3*s]
    ym,ys = ytr.mean(),ytr.std()
    if ys>1e-10: Xtr,ytr = Xtr[np.abs(ytr.flatten()-ym)<=3*ys], ytr[np.abs(ytr.flatten()-ym)<=3*ys]
    xz = StandardScaler().fit(Xtr); yz = StandardScaler().fit(ytr)
    xm = MinMaxScaler().fit(xz.transform(Xtr))
    return xm.transform(xz.transform(Xtr)), xm.transform(xz.transform(Xte)), yz.transform(ytr), yz.transform(yte), yz

# ===== S1: 3-sigma outlier removal =====
print("S1: 3-sigma removal...")
s1_rows = []
for target in TARGETS:
    for test_cond in CONDS:
        train = df[df['Cond']!=test_cond]; Xtr = train[FEATURES].values; ytr = train[[target]].values
        n_before = len(Xtr)
        for ci in range(Xtr.shape[1]):
            col=Xtr[:,ci]; m,s=col.mean(),col.std()
            if s>1e-10: Xtr,ytr = Xtr[np.abs(col-m)<=3*s], ytr[np.abs(col-m)<=3*s]
        ym,ys=ytr.mean(),ytr.std()
        if ys>1e-10: Xtr,ytr = Xtr[np.abs(ytr.flatten()-ym)<=3*ys], ytr[np.abs(ytr.flatten()-ym)<=3*ys]
        s1_rows.append([target, test_cond[:35], n_before, len(Xtr), n_before-len(Xtr)])

# ===== S2: Condition number =====
print("S2: Condition number...")
poly = PolynomialFeatures(degree=2)
X_poly_raw = poly.fit_transform(df[FEATURES].values)
cond_raw = np.linalg.cond(X_poly_raw)
xz_cn = StandardScaler().fit(df[FEATURES].values)
X_z = xz_cn.transform(df[FEATURES].values)
cond_z = np.linalg.cond(poly.fit_transform(X_z))

# ===== S3: AIC/BIC =====
print("S3: AIC/BIC...")
s2_data = [
    ('PBAT/PLA','CO2',1,6,-30.1,-20.0,0.534),('PBAT/PLA','CO2',2,21,-55.2,-19.8,0.994),('PBAT/PLA','CO2',3,56,68.5,129.0,0.978),
    ('PBAT/PLA','Residual',1,6,-28.5,-18.3,0.548),('PBAT/PLA','Residual',2,21,-53.7,-18.2,0.994),('PBAT/PLA','Residual',3,56,72.3,132.9,0.980),
    ('PBAT/PLA','Tensile',1,6,-25.3,-15.2,0.512),('PBAT/PLA','Tensile',2,21,-49.9,-14.4,0.978),('PBAT/PLA','Tensile',3,56,78.1,138.7,0.965),
    ('PBAT/TPS','CO2',1,6,-26.3,-16.2,0.534),('PBAT/TPS','CO2',2,21,-51.4,-16.0,0.883),('PBAT/TPS','CO2',3,56,61.7,122.3,0.646),
    ('PBAT/TPS','Residual',1,6,-24.2,-14.0,0.501),('PBAT/TPS','Residual',2,21,-45.9,-10.4,0.882),('PBAT/TPS','Residual',3,56,55.7,116.2,0.672),
    ('PBAT/PBS','CO2',1,6,-47.0,-36.8,0.705),('PBAT/PBS','CO2',2,21,-94.7,-59.2,0.958),('PBAT/PBS','CO2',3,56,72.1,132.7,0.932),
]

# ===== S4: Computational cost =====
print("S4: Computational cost...")
s4_rows = [
    ['Persistence','<0.001','N/A','N/A','0','Zero-parameter baseline'],
    ['LS-MPR','~0.003','N/A','N/A','21','Analytical solution'],
    ['Ridge','~0.2','5-fold CV','Log-space grid (10 values)','21','Grid search + CV'],
    ['Lasso','~0.2','5-fold CV','Log-space grid (10 values)','21','Grid search + CV'],
    ['GA','~35','N/A','Population 200 x 500 gen','21','DEAP library'],
    ['SVR','~2.3','3-fold CV','C x gamma grid','N/A','RBF kernel'],
    ['RF','~0.16','N/A','n_estimators=200','N/A','Default sklearn'],
    ['GPR-Poly','~6','Marginal likelihood','5 random restarts','N/A (kernel params)','L-BFGS-B optimizer'],
    ['Temporal shuffle','~0.1 x 30','N/A','N/A','N/A','30 realizations'],
    ['Block bootstrap','~5 x 4','N/A','B=500 per block size','N/A','4 block sizes tested'],
]

# ===== S5: LOCO bootstrap intervals =====
print("S5: LOCO bootstrap...")
target = 'CO2_Release'
s5_rows = []
for test_cond in CONDS:
    train = df[df['Cond']!=test_cond]; test = df[df['Cond']==test_cond]
    Xtr = train[FEATURES].values; ytr = train[[target]].values
    Xte = test[FEATURES].values; yte = test[[target]].values
    Xtr_s, Xte_s, ytr_s, yte_s, yz = preprocess(Xtr, ytr, Xte, yte)
    pm = PolyModel(degree=2)
    ols = LinearRegression(fit_intercept=False); ols.fit(pm.create_features(Xtr_s), ytr_s.flatten())
    yp = yz.inverse_transform(ols.predict(pm.create_features(Xte_s)).reshape(-1,1)).flatten()
    yt = yz.inverse_transform(yte_s).flatten()
    # Bootstrap
    B=500; n_test=len(yp); r2_boots=np.zeros(B)
    for b in range(B):
        idx=np.random.choice(n_test,n_test,replace=True)
        r2_boots[b]=r2_score(yt[idx],yp[idx])
    s5_rows.append([test_cond[:35], r2_score(yt,yp), np.mean(r2_boots),
                    np.percentile(r2_boots,2.5), np.percentile(r2_boots,97.5)])

# ===== S7: VIF + correlation =====
print("S7: VIF analysis...")
X_std = StandardScaler().fit_transform(df[FEATURES].values)
X_with_const = np.column_stack([np.ones(len(X_std)), X_std])
s7_vif = []
for i, name in enumerate(['Intercept']+FEATURES):
    vif = variance_inflation_factor(X_with_const, i)
    s7_vif.append([name, f'{vif:.1f}' if not np.isinf(vif) else 'infinity'])

# Correlation matrix
corr = df[FEATURES].corr()
s7_corr = []
for c1 in FEATURES:
    s7_corr.append([c1] + [f'{corr.loc[c1,c2]:.4f}' for c2 in FEATURES])

# ===== S10: Persistence for Residual/Tensile =====
print("S10: Persistence all targets...")
s10_rows = []
for target in TARGETS:
    train = df[df['Days']<=119]; test = df[df['Days']>119]
    yp_p = np.zeros(len(test))
    for cond in test['Cond'].unique():
        ct = train[train['Cond']==cond]
        if len(ct)>0:
            yp_p[test['Cond']==cond] = ct[ct['Days']==ct['Days'].max()][target].values[0]
    r2_p = r2_score(test[target].values, yp_p)
    rmse_p = np.sqrt(mean_squared_error(test[target].values, yp_p))
    # LS-MPR
    Xtr = train[FEATURES].values; ytr = train[[target]].values
    Xte = test[FEATURES].values; yte = test[[target]].values
    Xtr_s, Xte_s, ytr_s, yte_s, yz = preprocess(Xtr, ytr, Xte, yte)
    pm = PolyModel(degree=2); Xtr_p = pm.create_features(Xtr_s); Xte_p = pm.create_features(Xte_s)
    ols = LinearRegression(fit_intercept=False); ols.fit(Xtr_p, ytr_s.flatten())
    yp_ls = yz.inverse_transform(ols.predict(Xte_p).reshape(-1,1)).flatten()
    r2_ls = r2_score(yte, yp_ls)
    rmse_ls = np.sqrt(mean_squared_error(yte, yp_ls))
    s10_rows.append([target, f'{r2_p:+.4f}', f'{rmse_p:.2f}', f'{r2_ls:+.4f}', f'{rmse_ls:.2f}'])

# ===== S11: Kernel ablation =====
print("S11: Kernel ablation...")
kernels = {
    'Dot-Product+White': ConstantKernel(1.0)*DotProduct(sigma_0=1.0)+WhiteKernel(noise_level=0.01),
    'RBF+White': ConstantKernel(1.0)*RBF(length_scale=1.0)+WhiteKernel(noise_level=0.01),
    'Dot-Product+RBF+White': ConstantKernel(1.0)*DotProduct(sigma_0=1.0)+RBF(length_scale=1.0)+WhiteKernel(noise_level=0.01),
}
s11_rows = []
for kname, kernel in kernels.items():
    loco_r2s = [[], [], []]
    for target in TARGETS:
        r2s = []
        for test_cond in CONDS:
            train = df[df['Cond']!=test_cond]; test = df[df['Cond']==test_cond]
            Xtr = train[FEATURES].values; ytr = train[[target]].values
            Xte = test[FEATURES].values; yte = test[[target]].values
            Xtr_s, Xte_s, ytr_s, yte_s, yz = preprocess(Xtr, ytr, Xte, yte)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=5, random_state=SEED)
            gpr.fit(Xtr_s, ytr_s.flatten())
            yp = yz.inverse_transform(gpr.predict(Xte_s).reshape(-1,1)).flatten()
            yt = yz.inverse_transform(yte_s).flatten()
            r2s.append(r2_score(yt,yp))
        loco_r2s[TARGETS.index(target)] = r2s
    for tidx, target in enumerate(TARGETS):
        vals = loco_r2s[tidx]
        s11_rows.append([kname, target, f'{vals[0]:+.4f}', f'{vals[1]:+.4f}', f'{vals[2]:+.4f}', f'{np.mean(vals):+.4f}+/-{np.std(vals):.4f}'])

# ===== S13: Per-material =====
print("S13: Per-material...")
val_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('validation_')])
s13_rows = []
for vf in val_files:
    vdf = pd.read_csv(os.path.join(DATA_DIR, vf))
    mat_name = vf.replace('validation_','').replace('.csv','')
    X = vdf[['Days']].values; y = vdf[['Sample_CO2_net']].values
    xz = StandardScaler().fit(X); yz = StandardScaler().fit(y)
    pm = PolyModel(degree=2); X_p = pm.create_features(xz.transform(X))
    ols = LinearRegression(fit_intercept=False); ols.fit(X_p, yz.transform(y).flatten())
    yp = yz.inverse_transform(ols.predict(X_p).reshape(-1,1)).flatten()
    co2_180 = vdf[vdf['Days'].round()==180]['Sample_CO2_net'].values[0] if 180 in vdf['Days'].values else np.nan
    s13_rows.append([mat_name, f'{r2_score(y.flatten(),yp):.4f}', f'{co2_180:.2f}'])

# ===== S14-S15: Augmentation =====
print("S14-S15: Augmentation...")
df_aug['Cond'] = df_aug.apply(lambda r: f"T{int(r.Temperature)}_H{int(r.Humidity)}_R{int(r.Ratio)}_CV{r.Compost_Volume}", axis=1)
s14_rows = []
for target in TARGETS:
    X = df[FEATURES].values; y = df[[target]].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    r2_orig,_,_,_,_ = preprocess(Xtr,ytr,Xte,yte)
    pm = PolyModel(degree=2); ols = LinearRegression(fit_intercept=False)
    Xtr_s, Xte_s, ytr_s, yte_s, yz = preprocess(Xtr,ytr,Xte,yte)
    ols.fit(pm.create_features(Xtr_s), ytr_s.flatten())
    yp = yz.inverse_transform(ols.predict(pm.create_features(Xte_s)).reshape(-1,1)).flatten(); yt = yz.inverse_transform(yte_s).flatten()
    r2_orig = r2_score(yt,yp)
    # Augmented
    Xa = df_aug[FEATURES].values; ya = df_aug[[target]].values
    Xtr_a, Xte_a, ytr_a, yte_a = train_test_split(Xa, ya, test_size=0.2, random_state=SEED)
    Xtr_as, Xte_as, ytr_as, yte_as, yza = preprocess(Xtr_a,ytr_a,Xte_a,yte_a)
    ols_a = LinearRegression(fit_intercept=False)
    pm_a = PolyModel(degree=2); ols_a.fit(pm_a.create_features(Xtr_as), ytr_as.flatten())
    ypa = yza.inverse_transform(ols_a.predict(pm_a.create_features(Xte_as)).reshape(-1,1)).flatten()
    yta = yza.inverse_transform(yte_as).flatten()
    s14_rows.append([target, f'{r2_orig:.4f}', f'{r2_score(yta,ypa):.4f}', f'{r2_orig-r2_score(yta,ypa):+.4f}'])

# Temporal holdout augmentation
s15_rows = []
for target in [TARGETS[0]]:
    for days_cut in [119,89]:
        for label, data in [('Original 543',df),('Augmented 1580',df_aug)]:
            train_d = data[data['Days']<=days_cut]; test_d = data[data['Days']>days_cut]
            Xtr_d = train_d[FEATURES].values; ytr_d = train_d[[target]].values
            Xte_d = test_d[FEATURES].values; yte_d = test_d[[target]].values
            Xtr_sd, Xte_sd, ytr_sd, yte_sd, yzd = preprocess(Xtr_d, ytr_d, Xte_d, yte_d)
            pm_d = PolyModel(degree=2); ols_d = LinearRegression(fit_intercept=False)
            ols_d.fit(pm_d.create_features(Xtr_sd), ytr_sd.flatten())
            ypd = yzd.inverse_transform(ols_d.predict(pm_d.create_features(Xte_sd)).reshape(-1,1)).flatten()
            ytd = yzd.inverse_transform(yte_sd).flatten()
            r2d = r2_score(ytd,ypd)
            s15_rows.append([label, f'0-{days_cut}->{days_cut+1}-180', f'{r2d:+.4f}', len(Xtr_d), len(test_d)])

# ===== S16-S18: LS coefficients =====
print("S16-S18: LS coefficients...")
s16_rows = []; s17_rows = []; s18_rows = []
for target, Model, storage in [('CO2_Release',CO2Model,s16_rows),('Residual_Rate',ResidualModel,s17_rows),('Tensile_Strength',StrengthModel,s18_rows)]:
    m = Model(os.path.join(DATA_DIR,'pbat_pla_experimental_543.csv'))
    coefs = m.get_coefficients()
    for feat, val in zip(coefs['features'], coefs['LS']):
        storage.append([feat, f'{val:+.6f}'])

# ===== SAVE ALL =====
import json

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")

all_data = {
    's1': s1_rows, 's2_raw': cond_raw, 's2_z': cond_z, 's3': s2_data,
    's4': s4_rows, 's5': s5_rows, 's7_vif': s7_vif, 's7_corr': s7_corr,
    's10': s10_rows, 's11': s11_rows, 's13': s13_rows,
    's14': s14_rows, 's15': s15_rows,
    's16': s16_rows, 's17': s17_rows, 's18': s18_rows,
}
# Save as JSON for the docx generator
with open(os.path.join(RESULTS, 'all_supplementary_data.json'), 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
print(f'\nAll data saved to {RESULTS}/all_supplementary_data.json')
print(f'Tables: S1({len(s1_rows)}), S2, S3({len(s2_data)}), S4({len(s4_rows)}), S5({len(s5_rows)}), S7({len(s7_vif)}+{len(s7_corr)}), S10({len(s10_rows)}), S11({len(s11_rows)}), S13({len(s13_rows)}), S14({len(s14_rows)}), S15({len(s15_rows)}), S16-18({len(s16_rows)} each)')

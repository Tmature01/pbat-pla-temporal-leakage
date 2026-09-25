"""
Supplementary experiments addressing reviewer comments:
1. Temporal shuffle (separate leakage from extrapolation weakness)
2. Augmentation under temporal holdout
3. Ridge optimal alpha
4. Condition number
5. Block bootstrap
"""
import pandas as pd, numpy as np, sys, os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import PolyModel

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
df = pd.read_csv(os.path.join(DATA_DIR, 'pbat_pla_experimental_543.csv'))
df_aug = pd.read_csv(os.path.join(DATA_DIR, 'pbat_pla_augmented_1580.csv'))
FEATURES = ['Days','Temperature','Humidity','Ratio','Compost_Volume']
TARGET = 'CO2_Release'
SEED=42; np.random.seed(SEED)

def preprocess_and_fit(Xtr, ytr, Xte, yte):
    for ci in range(Xtr.shape[1]):
        col=Xtr[:,ci]; m,s=col.mean(),col.std()
        if s>1e-10: Xtr,ytr=Xtr[np.abs(col-m)<=3*s],ytr[np.abs(col-m)<=3*s]
    ym,ys=ytr.mean(),ytr.std()
    if ys>1e-10: Xtr,ytr=Xtr[np.abs(ytr.flatten()-ym)<=3*ys],ytr[np.abs(ytr.flatten()-ym)<=3*ys]
    xz=StandardScaler().fit(Xtr); yz=StandardScaler().fit(ytr)
    xm=MinMaxScaler().fit(xz.transform(Xtr))
    Xtr_s=xm.transform(xz.transform(Xtr)); Xte_s=xm.transform(xz.transform(Xte))
    ytr_s=yz.transform(ytr); yte_s=yz.transform(yte)
    pm=PolyModel(degree=2); Xtr_p=pm.create_features(Xtr_s); Xte_p=pm.create_features(Xte_s)
    ols=LinearRegression(fit_intercept=False); ols.fit(Xtr_p,ytr_s.flatten())
    yp=yz.inverse_transform(ols.predict(Xte_p).reshape(-1,1)).flatten()
    yt=yz.inverse_transform(yte_s).flatten()
    return r2_score(yt,yp), np.sqrt(mean_squared_error(yt,yp)), spearmanr(yt,yp)[0], pm, ols, Xtr_s, ytr_s, Xte_s, yte_s, yz

# Add condition labels
df['Cond'] = df.apply(lambda r: "T{}_H{}_R{}_CV{}".format(int(r.Temperature),int(r.Humidity),int(r.Ratio),r.Compost_Volume), axis=1)
df_aug['Cond'] = df_aug.apply(lambda r: "T{}_H{}_R{}_CV{}".format(int(r.Temperature),int(r.Humidity),int(r.Ratio),r.Compost_Volume), axis=1)

out = []
def p(s):
    print(s); out.append(s)

p('='*65)
p('SUPPLEMENTARY EXPERIMENTS FOR REVIEWER RESPONSE')
p('='*65)

# ================================================================
# 1. TEMPORAL SHUFFLE
# ================================================================
p('\n--- 1. TEMPORAL SHUFFLE: Separate leakage from extrapolation ---')
train=df[df['Days']<=119]; test=df[df['Days']>119]
Xtr=train[FEATURES].values; ytr=train[[TARGET]].values
Xte=test[FEATURES].values; yte=test[[TARGET]].values
r2_real,rmse_real,rho_real,_,_,_,_,_,_,_ = preprocess_and_fit(Xtr,ytr,Xte,yte)
p('Temporal holdout (real ordering):     R2={:+.4f}  RMSE={:.2f}'.format(r2_real,rmse_real))

# Shuffle Days within each condition
df_shuf = df.copy()
for cond in df_shuf['Cond'].unique():
    mask = df_shuf['Cond']==cond
    df_shuf.loc[mask,'Days'] = np.random.permutation(df_shuf.loc[mask,'Days'].values)
train_s=df_shuf[df_shuf['Days']<=119]; test_s=df_shuf[df_shuf['Days']>119]
Xtr_s=train_s[FEATURES].values; ytr_s=train_s[[TARGET]].values
Xte_s=test_s[FEATURES].values; yte_s=test_s[[TARGET]].values
r2_shuf,rmse_shuf,rho_shuf,_,_,_,_,_,_,_ = preprocess_and_fit(Xtr_s,ytr_s,Xte_s,yte_s)
p('Temporal holdout (shuffled Days):     R2={:+.4f}  RMSE={:.2f}'.format(r2_shuf,rmse_shuf))
p('Leakage contribution (shuffled-real): Delta-R2 = {:+.4f}'.format(r2_shuf - r2_real))
p('Extrapolation weakness (real R2):     R2 = {:+.4f}'.format(r2_real))

# Random split on shuffled to confirm
X_all=df_shuf[FEATURES].values; y_all=df_shuf[[TARGET]].values
Xtr_r,Xte_r,ytr_r,yte_r = train_test_split(X_all,y_all,test_size=0.2,random_state=SEED)
r2_rand,rmse_rand,rho_rand,_,_,_,_,_,_,_ = preprocess_and_fit(Xtr_r,ytr_r,Xte_r,yte_r)
p('Random split on shuffled data:        R2={:+.4f}'.format(r2_rand))
p('=> Leakage effect: Delta-R2(shuffled temporal - real temporal) = {:+.4f}'.format(r2_shuf-r2_real))
p('=> Extrapolation weakness: real temporal R2 = {:+.4f}'.format(r2_real))
p('=> Combined: shuffled temporal R2 - real temporal R2 isolates leakage from extrapolation ability')

# ================================================================
# 2. AUGMENTATION UNDER TEMPORAL HOLDOUT
# ================================================================
p('\n--- 2. AUGMENTATION UNDER TEMPORAL HOLDOUT ---')
for label, data in [('Original 543', df), ('Augmented 1580', df_aug)]:
    train_d = data[data['Days']<=119]; test_d = data[data['Days']>119]
    Xtr_d = train_d[FEATURES].values; ytr_d = train_d[[TARGET]].values
    Xte_d = test_d[FEATURES].values; yte_d = test_d[[TARGET]].values
    r2_d,rmse_d,rho_d,_,_,_,_,_,_,_ = preprocess_and_fit(Xtr_d,ytr_d,Xte_d,yte_d)
    p('{}: R2={:+.4f}  RMSE={:.2f}  rho={:+.4f}  (train={}, test={})'.format(
        label, r2_d, rmse_d, rho_d, len(Xtr_d), len(Xte_d)))

# ================================================================
# 3. RIDGE ALPHA
# ================================================================
p('\n--- 3. RIDGE OPTIMAL ALPHA ---')
train=df[df['Days']<=119]; test=df[df['Days']>119]
Xtr=train[FEATURES].values; ytr=train[[TARGET]].values
Xte=test[FEATURES].values; yte=test[[TARGET]].values
r2_ls,rmse_ls,rho_ls,pm,ols,Xtr_s,ytr_s,Xte_s,yte_s,yz = preprocess_and_fit(Xtr,ytr,Xte,yte)
Xtr_p = pm.create_features(Xtr_s); Xte_p = pm.create_features(Xte_s)

alphas = np.logspace(-3, 3, 20)
ridge_cv = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
ridge_cv.fit(Xtr_p, ytr_s.flatten())
best_alpha = ridge_cv.alpha_
p('RidgeCV best alpha: {:.6f}'.format(best_alpha))
for a in [1e-6, 1e-4, 1e-2, 1.0, 1e2, best_alpha]:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(Xtr_p, ytr_s.flatten())
    yp = yz.inverse_transform(ridge.predict(Xte_p).reshape(-1,1)).flatten()
    yt = yz.inverse_transform(yte_s).flatten()
    marker = ' <-- BEST' if abs(a-best_alpha)<1e-10 else ''
    p('  alpha={:.0e}: R2={:+.6f}{}'.format(a, r2_score(yt,yp), marker))

# ================================================================
# 4. CONDITION NUMBER
# ================================================================
p('\n--- 4. POLYNOMIAL DESIGN MATRIX CONDITION NUMBER ---')
poly_direct = PolynomialFeatures(degree=2)
X_poly_raw = poly_direct.fit_transform(df[FEATURES].values)
cond_raw = np.linalg.cond(X_poly_raw)
p('Unscaled X_poly (21 cols): condition number = {:.1e}'.format(cond_raw))

# Get scalers from a proper fit
xz_cn = StandardScaler().fit(df[FEATURES].values)
xm_cn = MinMaxScaler().fit(xz_cn.transform(df[FEATURES].values))
X_all_s_cn = xm_cn.transform(xz_cn.transform(df[FEATURES].values))
X_poly_scaled = poly_direct.fit_transform(X_all_s_cn)
cond_scaled = np.linalg.cond(X_poly_scaled)
p('Scaled X_poly (21 cols):   condition number = {:.1f}'.format(cond_scaled))
p('Scaling reduces condition number by factor of {:.1e}x'.format(cond_raw/cond_scaled))

# ================================================================
# 5. BLOCK BOOTSTRAP
# ================================================================
p('\n--- 5. BLOCK BOOTSTRAP vs STANDARD BOOTSTRAP ---')
# Train on full data
X=df[FEATURES].values; y=df[[TARGET]].values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=SEED)
xz2=StandardScaler().fit(Xtr); yz2=StandardScaler().fit(ytr)
xm2=MinMaxScaler().fit(xz2.transform(Xtr))
Xtr_s2=xm2.transform(xz2.transform(Xtr)); Xte_s2=xm2.transform(xz2.transform(Xte))
ytr_s2=yz2.transform(ytr); yte_s2=yz2.transform(yte)
pm2=PolyModel(degree=2); Xtr_p2=pm2.create_features(Xtr_s2); Xte_p2=pm2.create_features(Xte_s2)
ols2=LinearRegression(fit_intercept=False); ols2.fit(Xtr_p2,ytr_s2.flatten())
yp=yz2.inverse_transform(ols2.predict(Xte_p2).reshape(-1,1)).flatten()
yt=yz2.inverse_transform(yte_s2).flatten()

B=1000; n_test=len(yp)

# Standard bootstrap
cov_std=np.zeros(B)
for b in range(B):
    idx=np.random.choice(n_test,n_test,replace=True)
    resid=yp[idx]-yt[idx]
    ci=np.percentile(resid,[2.5,97.5])
    cov_std[b]=(ci[0]<=0)&(ci[1]>=0)
p('Standard bootstrap CI coverage: {:.1f}% (B={})'.format(np.mean(cov_std)*100,B))

# Block bootstrap
block_sizes=[5,10,20]
for bs in block_sizes:
    n_blocks=n_test//bs
    cov_block=np.zeros(min(B,500))
    for b in range(min(B,500)):
        blocks=np.random.choice(n_blocks,n_blocks,replace=True)
        idx=np.concatenate([np.arange(bl*bs,min((bl+1)*bs,n_test)) for bl in blocks])
        if len(idx)>n_test: idx=idx[:n_test]
        resid=yp[idx]-yt[idx]
        ci=np.percentile(resid,[2.5,97.5])
        cov_block[b]=(ci[0]<=0)&(ci[1]>=0)
    p('Block bootstrap CI coverage (block={:2d}, B=500): {:.1f}%'.format(bs,np.mean(cov_block)*100))

p('\nDONE - all supplementary experiments complete')

# Save
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'reviewer_supplementary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

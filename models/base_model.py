from utils.data_processor import DataProcessor
from utils.model import PolyModel
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from sklearn.model_selection import KFold


class BaseModel:
    """基类：统一管理 LS 和 GA 两种参数估计方法"""

    def __init__(self, data_path, target_variable):
        self.target_variable = target_variable
        self.processor = DataProcessor(data_path)
        self.model = PolyModel(degree=2)

        # 划分训练集与测试集
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.processor.prepare_train_test_split(target_variable)
        self.X_train_poly = self.model.create_features(self.X_train)
        self.X_test_poly = self.model.create_features(self.X_test)

        # LS 训练
        self.coeff_ls = self.model.ols_optimization(self.X_train_poly, self.y_train)

        # GA 训练
        self.coeff_ga = self.model.genetic_optimization(self.X_train_poly, self.y_train)

    def _get_scores(self, X_poly, y_norm, coeff):
        """计算单个数据集的评估指标（X_poly 已经是多项式展开后的特征矩阵）"""
        y_pred = np.dot(X_poly, coeff)
        y_true = self.processor.scalers['y_zscore'].inverse_transform(y_norm)
        y_pred_real = self.processor.scalers['y_zscore'].inverse_transform(y_pred.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_true, y_pred_real))
        r2 = r2_score(y_true, y_pred_real)
        rho, _ = spearmanr(y_true.flatten(), y_pred_real.flatten())
        return {"rmse": round(rmse, 4), "r2": round(r2, 4), "rho": round(rho, 4)}

    def calculate_metrics(self):
        """返回 LS 和 GA 两种方法在训练集/测试集上的对比指标"""
        results = {}
        for name, coeff in [("LS", self.coeff_ls), ("GA", self.coeff_ga)]:
            results[name] = {
                "train": self._get_scores(self.X_train_poly, self.y_train, coeff),
                "test": self._get_scores(self.X_test_poly, self.y_test, coeff),
            }
        return results

    def get_coefficients(self):
        """返回 LS 和 GA 的多项式系数及其特征名"""
        feature_names = self.model.poly.get_feature_names_out(self.processor.feature_columns).tolist()
        ls_list = [round(float(c), 6) for c in self.coeff_ls]
        ga_list = [round(float(c), 6) for c in self.coeff_ga]
        return {"features": feature_names, "LS": ls_list, "GA": ga_list}

    def predict(self, inputs, method="LS"):
        """网页预测：输入原始值，返回反归一化后的预测值"""
        coeff = self.coeff_ga if method == "GA" else self.coeff_ls
        X_z = self.processor.scalers['X_zscore'].transform([inputs])
        X_norm = self.processor.scalers['X_minmax'].transform(X_z)
        y_pred_norm = self.model.predict(X_norm, coeff)
        return self.processor.scalers['y_zscore'].inverse_transform(
            y_pred_norm.reshape(-1, 1)
        ).flatten()[0]

    def compare_degrees(self, max_degree=3):
        """对比不同多项式阶数 d=1..max_degree 的 AIC/BIC，为 degree=2 提供定量依据"""
        from sklearn.preprocessing import PolynomialFeatures
        results = []
        for d in range(1, max_degree + 1):
            poly = PolynomialFeatures(degree=d)
            X_tr_poly = poly.fit_transform(self.X_train)
            X_te_poly = poly.transform(self.X_test)

            ols = LinearRegression(fit_intercept=False)
            ols.fit(X_tr_poly, self.y_train.flatten())
            y_pred = ols.predict(X_te_poly)

            n = len(self.y_test)
            k = X_te_poly.shape[1]  # 参数个数
            sse = np.sum((self.y_test.flatten() - y_pred) ** 2)
            # 避免 log(0)
            if sse < 1e-12:
                sse = 1e-12
            aic = n * np.log(sse / n) + 2 * k
            bic = n * np.log(sse / n) + k * np.log(n)

            y_true_real = self.processor.scalers['y_zscore'].inverse_transform(self.y_test)
            y_pred_real = self.processor.scalers['y_zscore'].inverse_transform(y_pred.reshape(-1, 1))
            r2 = r2_score(y_true_real, y_pred_real)

            results.append({
                "degree": d,
                "n_features": k,
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "r2": round(r2, 4),
            })
        return results

    def cross_validate(self, n_folds=5):
        """K折交叉验证，返回 LS 和 GA 的均值 ± 标准差"""
        import pandas as pd
        df = pd.read_csv(self.processor.filepath)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        df = self.processor.remove_outliers_3sigma(
            df, self.processor.feature_columns + [self.target_variable]
        )

        X = df[self.processor.feature_columns].values
        y = df[[self.target_variable]].values

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = {"LS": {"rmse": [], "r2": [], "rho": []},
                      "GA": {"rmse": [], "r2": [], "rho": []}}

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 标准化（每折独立 fit）
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            x_zs = StandardScaler().fit(X_train)
            y_zs = StandardScaler().fit(y_train)
            x_mm = MinMaxScaler().fit(x_zs.transform(X_train))

            X_train_s = x_mm.transform(x_zs.transform(X_train))
            X_test_s = x_mm.transform(x_zs.transform(X_test))
            y_train_s = y_zs.transform(y_train)
            y_test_s = y_zs.transform(y_test)

            model = PolyModel(degree=2)
            X_train_poly = model.create_features(X_train_s)
            X_test_poly = model.create_features(X_test_s)

            coeff_ls = model.ols_optimization(X_train_poly, y_train_s)
            coeff_ga = model.genetic_optimization(X_train_poly, y_train_s)

            for name, coeff in [("LS", coeff_ls), ("GA", coeff_ga)]:
                y_pred = np.dot(X_test_poly, coeff)
                y_true = y_zs.inverse_transform(y_test_s)
                y_pred_real = y_zs.inverse_transform(y_pred.reshape(-1, 1))

                cv_results[name]["rmse"].append(
                    np.sqrt(mean_squared_error(y_true, y_pred_real)))
                cv_results[name]["r2"].append(r2_score(y_true, y_pred_real))
                rho, _ = spearmanr(y_true.flatten(), y_pred_real.flatten())
                cv_results[name]["rho"].append(rho)

        # 汇总均值和标准差
        summary = {}
        for name in ["LS", "GA"]:
            summary[name] = {}
            for metric in ["rmse", "r2", "rho"]:
                vals = cv_results[name][metric]
                summary[name][metric] = {
                    "mean": round(np.mean(vals), 4),
                    "std": round(np.std(vals), 4)
                }
        return summary

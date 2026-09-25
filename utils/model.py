from sklearn.preprocessing import PolynomialFeatures
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import random
import numpy as np

RANDOM_SEED = 42

# DEAP 遗传算法环境初始化
if not hasattr(creator, 'FitnessMin'):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class PolyModel:
    def __init__(self, degree=2, random_seed=RANDOM_SEED):
        self.degree = degree
        self.random_seed = random_seed
        self.poly = PolynomialFeatures(degree)
        self._poly_fitted = False

    def create_features(self, X):
        """首次调用时 fit_transform，后续调用只 transform"""
        if not self._poly_fitted:
            result = self.poly.fit_transform(X)
            self._poly_fitted = True
            return result
        return self.poly.transform(X)

    def ols_optimization(self, X_poly, y_norm):
        """最小二乘法（LS）参数估计"""
        ols_model = LinearRegression(fit_intercept=False)
        ols_model.fit(X_poly, y_norm.flatten())
        return np.array(ols_model.coef_)

    def genetic_optimization(self, X_poly, y_norm, pop_size=200, generations=500):
        """遗传算法（GA）参数优化"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -20, 20)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=X_poly.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        y_true = y_norm.flatten()
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        def eval_func(individual):
            y_pred = np.dot(X_poly, individual)

            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

            ss_res = np.sum((y_true - y_pred) ** 2)
            r2_loss = ss_res / (ss_tot + 1e-6)

            if np.std(y_pred) > 1e-6:
                corr, _ = spearmanr(y_pred, y_true)
            else:
                corr = 0
            corr_loss = 1 - corr

            # 论文权重：R²=0.4（主要地位）, RMSE=0.3, Spearman=0.3
            score = (0.3 * rmse) + (0.4 * r2_loss) + (0.3 * corr_loss)
            return (score,)

        toolbox.register("evaluate", eval_func)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations, verbose=False)

        best_ind = tools.selBest(pop, k=1)[0]
        return np.array(best_ind)

    def predict(self, X_norm, coeff):
        X_poly = self.create_features(X_norm)
        return np.dot(X_poly, coeff)
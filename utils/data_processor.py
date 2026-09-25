import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scalers = {}
        self.feature_columns = ['Days', 'Temperature', 'Humidity', 'Ratio', 'Compost_Volume']

    def remove_outliers_3sigma(self, df, columns):
        """拉依达准则（3σ准则）异常值检测与剔除"""
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] - mean).abs() <= 3 * std]
        return df

    def prepare_data(self, target_variable=None):
        """数据准备：异常值处理 → Z-score标准化 → Min-Max归一化"""
        df = pd.read_csv(self.filepath)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        cols_to_check = self.feature_columns + [target_variable] if target_variable else self.feature_columns
        df = self.remove_outliers_3sigma(df, cols_to_check)

        X = df[self.feature_columns].values
        y = df[[target_variable]].values if target_variable else None

        # Z-score标准化
        self.scalers['X_zscore'] = StandardScaler()
        X_zscore = self.scalers['X_zscore'].fit_transform(X)

        # Min-Max归一化 → [0,1]
        self.scalers['X_minmax'] = MinMaxScaler()
        X_norm = self.scalers['X_minmax'].fit_transform(X_zscore)

        if target_variable:
            self.scalers['y_zscore'] = StandardScaler()
            y_norm = self.scalers['y_zscore'].fit_transform(y)
            return X_norm, y_norm
        return X_norm

    def prepare_train_test_split(self, target_variable, test_size=0.2, random_state=RANDOM_STATE):
        """划分训练集/测试集，异常值处理 → Z-score → Min-Max"""
        df = pd.read_csv(self.filepath)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        df = self.remove_outliers_3sigma(df, self.feature_columns + [target_variable])

        X = df[self.feature_columns].values
        y = df[[target_variable]].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Z-score标准化
        self.scalers['X_zscore'] = StandardScaler()
        self.scalers['y_zscore'] = StandardScaler()
        X_train_z = self.scalers['X_zscore'].fit_transform(X_train)
        X_test_z = self.scalers['X_zscore'].transform(X_test)
        y_train_z = self.scalers['y_zscore'].fit_transform(y_train)
        y_test_z = self.scalers['y_zscore'].transform(y_test)

        # Min-Max归一化
        self.scalers['X_minmax'] = MinMaxScaler()
        X_train_scaled = self.scalers['X_minmax'].fit_transform(X_train_z)
        X_test_scaled = self.scalers['X_minmax'].transform(X_test_z)

        return X_train_scaled, X_test_scaled, y_train_z, y_test_z
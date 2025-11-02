"""
机器学习模型定义和训练
用于ERA5天气预测的各种机器学习模型
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
from config import *

class WeatherPredictor:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def load_data(self):
        """加载预处理的数据"""
        print("加载预处理数据...")
        
        features_path = f"{DATA_DIR}/features.csv"
        targets_path = f"{DATA_DIR}/targets.csv"
        
        if not os.path.exists(features_path) or not os.path.exists(targets_path):
            raise FileNotFoundError("预处理数据不存在，请先运行数据预处理器")
        
        self.features = pd.read_csv(features_path, index_col=0)
        self.targets = pd.read_csv(targets_path, index_col=0)
        
        print(f"加载特征数据: {self.features.shape}")
        print(f"加载目标数据: {self.targets.shape}")
    
    def split_data(self):
        """划分训练集和测试集"""
        print("划分训练集和测试集...")
        
        # 移除包含NaN的行
        valid_mask = ~(self.features.isna().any(axis=1) | self.targets.isna().any(axis=1))
        X = self.features[valid_mask]
        y = self.targets[valid_mask]
        
        # 划分数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
    
    def train_temperature_model(self):
        """训练温度预测模型"""
        print("训练温度预测模型...")
        
        # 目标变量：温度
        y_temp_train = self.y_train['target_temp']
        y_temp_test = self.y_test['target_temp']
        
        # 移除NaN值
        train_mask = ~y_temp_train.isna()
        test_mask = ~y_temp_test.isna()
        
        X_train_temp = self.X_train[train_mask]
        y_train_temp = y_temp_train[train_mask]
        X_test_temp = self.X_test[test_mask]
        y_test_temp = y_temp_test[test_mask]
        
        # 训练XGBoost回归模型
        model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
        
        model.fit(X_train_temp, y_train_temp)
        
        # 预测和评估
        y_pred = model.predict(X_test_temp)
        
        metrics = {
            'mse': mean_squared_error(y_test_temp, y_pred),
            'mae': mean_absolute_error(y_test_temp, y_pred),
            'r2': r2_score(y_test_temp, y_pred)
        }
        
        self.models['temperature'] = model
        self.metrics['temperature'] = metrics
        
        print(f"温度模型 - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
    
    def train_precipitation_model(self):
        """训练降水预测模型"""
        print("训练降水预测模型...")
        
        # 目标变量：降水
        y_precip_train = self.y_train['target_precip']
        y_precip_test = self.y_test['target_precip']
        
        # 移除NaN值
        train_mask = ~y_precip_train.isna()
        test_mask = ~y_precip_test.isna()
        
        X_train_precip = self.X_train[train_mask]
        y_train_precip = y_precip_train[train_mask]
        X_test_precip = self.X_test[test_mask]
        y_test_precip = y_precip_test[test_mask]
        
        # 训练Random Forest回归模型
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE
        )
        
        model.fit(X_train_precip, y_train_precip)
        
        # 预测和评估
        y_pred = model.predict(X_test_precip)
        
        metrics = {
            'mse': mean_squared_error(y_test_precip, y_pred),
            'mae': mean_absolute_error(y_test_precip, y_pred),
            'r2': r2_score(y_test_precip, y_pred)
        }
        
        self.models['precipitation'] = model
        self.metrics['precipitation'] = metrics
        
        print(f"降水模型 - MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R2: {metrics['r2']:.4f}")
    
    def train_weather_type_model(self):
        """训练天气类型分类模型"""
        print("训练天气类型分类模型...")
        
        # 目标变量：天气类型
        y_weather_train = self.y_train['target_weather']
        y_weather_test = self.y_test['target_weather']
        
        # 移除NaN值
        train_mask = ~y_weather_train.isna()
        test_mask = ~y_weather_test.isna()
        
        X_train_weather = self.X_train[train_mask]
        y_train_weather = y_weather_train[train_mask]
        X_test_weather = self.X_test[test_mask]
        y_test_weather = y_weather_test[test_mask]
        
        # 训练Random Forest分类模型
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE
        )
        
        model.fit(X_train_weather, y_train_weather)
        
        # 预测和评估
        y_pred = model.predict(X_test_weather)
        
        metrics = {
            'accuracy': accuracy_score(y_test_weather, y_pred)
        }
        
        self.models['weather_type'] = model
        self.metrics['weather_type'] = metrics
        
        print(f"天气类型模型 - 准确率: {metrics['accuracy']:.4f}")
        print("分类报告:")
        print(classification_report(y_test_weather, y_pred))
    
    def train_pressure_level_model(self):
        """训练等压面预测模型"""
        print("训练等压面预测模型...")
        
        # 目标变量：500hPa位势高度
        y_z500_train = self.y_train['target_z500']
        y_z500_test = self.y_test['target_z500']
        
        # 移除NaN值
        train_mask = ~y_z500_train.isna()
        test_mask = ~y_z500_test.isna()
        
        X_train_z500 = self.X_train[train_mask]
        y_train_z500 = y_z500_train[train_mask]
        X_test_z500 = self.X_test[test_mask]
        y_test_z500 = y_z500_test[test_mask]
        
        # 训练XGBoost回归模型
        model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
        
        model.fit(X_train_z500, y_train_z500)
        
        # 预测和评估
        y_pred = model.predict(X_test_z500)
        
        metrics = {
            'mse': mean_squared_error(y_test_z500, y_pred),
            'mae': mean_absolute_error(y_test_z500, y_pred),
            'r2': r2_score(y_test_z500, y_pred)
        }
        
        self.models['pressure_level'] = model
        self.metrics['pressure_level'] = metrics
        
        print(f"等压面模型 - MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}")
    
    def train_all_models(self):
        """训练所有模型"""
        print("开始训练所有模型...")
        
        self.load_data()
        self.split_data()
        
        # 训练各个模型
        self.train_temperature_model()
        self.train_precipitation_model()
        self.train_weather_type_model()
        self.train_pressure_level_model()
        
        # 保存模型
        self.save_models()
        
        print("所有模型训练完成!")
    
    def save_models(self):
        """保存训练好的模型"""
        print("保存模型...")
        
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        
        for name, model in self.models.items():
            model_path = f"{MODELS_DIR}/{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"模型已保存: {model_path}")
        
        # 保存评估指标
        metrics_path = f"{MODELS_DIR}/metrics.pkl"
        joblib.dump(self.metrics, metrics_path)
        print(f"评估指标已保存: {metrics_path}")
    
    def load_models(self):
        """加载已训练的模型"""
        print("加载模型...")
        
        model_files = {
            'temperature': f"{MODELS_DIR}/temperature_model.pkl",
            'precipitation': f"{MODELS_DIR}/precipitation_model.pkl",
            'weather_type': f"{MODELS_DIR}/weather_type_model.pkl",
            'pressure_level': f"{MODELS_DIR}/pressure_level_model.pkl"
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"模型已加载: {name}")
        
        # 加载评估指标
        metrics_path = f"{MODELS_DIR}/metrics.pkl"
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
            print("评估指标已加载")
    
    def predict(self, features):
        """使用训练好的模型进行预测"""
        if not self.models:
            self.load_models()
        
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(features)
            predictions[name] = pred
        
        return predictions
    
    def get_feature_importance(self, model_name):
        """获取特征重要性"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.features.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None

if __name__ == "__main__":
    predictor = WeatherPredictor()
    
    # 训练所有模型
    predictor.train_all_models()
    
    # 显示模型性能
    print("\n模型性能总结:")
    for model_name, metrics in predictor.metrics.items():
        print(f"\n{model_name} 模型:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
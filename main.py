"""
ERA5机器学习天气预测主程序
整合数据下载、预处理、模型训练和预测的完整流程
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_downloader import ERA5DataDownloader
from data_preprocessor import ERA5DataPreprocessor
from ml_models import WeatherPredictor
from visualization import WeatherVisualizer

class ERA5WeatherPrediction:
    def __init__(self):
        self.downloader = ERA5DataDownloader()
        self.preprocessor = ERA5DataPreprocessor()
        self.predictor = WeatherPredictor()
        self.visualizer = WeatherVisualizer()
    
    def run_full_pipeline(self):
        """运行完整的机器学习流水线"""
        print("=" * 60)
        print("ERA5机器学习天气预测系统")
        print("=" * 60)
        
        try:
            # 步骤1: 数据下载
            if not self._check_data_exists():
                print("\n步骤1: 下载ERA5数据...")
                self.downloader.download_era5_data()
                self.downloader.combine_data_files()
            else:
                print("\n步骤1: 数据已存在，跳过下载")
            
            # 步骤2: 数据预处理
            if not self._check_processed_data_exists():
                print("\n步骤2: 数据预处理...")
                self.preprocessor.load_data()
                self.preprocessor.clean_data()
                features, targets = self.preprocessor.prepare_ml_data()
                self.preprocessor.save_processed_data(features, targets)
            else:
                print("\n步骤2: 预处理数据已存在，跳过预处理")
            
            # 步骤3: 模型训练
            if not self._check_models_exist():
                print("\n步骤3: 训练机器学习模型...")
                self.predictor.train_all_models()
            else:
                print("\n步骤3: 模型已存在，加载现有模型")
                self.predictor.load_models()
            
            # 步骤4: 模型评估和可视化
            print("\n步骤4: 模型评估和可视化...")
            self._evaluate_and_visualize()
            
            # 步骤5: 示例预测
            print("\n步骤5: 示例预测...")
            self._make_sample_predictions()
            
            print("\n" + "=" * 60)
            print("ERA5天气预测系统运行完成!")
            print("=" * 60)
            
        except Exception as e:
            print(f"运行过程中出错: {e}")
            sys.exit(1)
    
    def _check_data_exists(self):
        """检查原始数据是否存在"""
        surface_file = f"{DATA_DIR}/era5_surface_combined.nc"
        pressure_file = f"{DATA_DIR}/era5_pressure_combined.nc"
        return os.path.exists(surface_file) and os.path.exists(pressure_file)
    
    def _check_processed_data_exists(self):
        """检查预处理数据是否存在"""
        features_file = f"{DATA_DIR}/features.csv"
        targets_file = f"{DATA_DIR}/targets.csv"
        return os.path.exists(features_file) and os.path.exists(targets_file)
    
    def _check_models_exist(self):
        """检查训练好的模型是否存在"""
        model_files = [
            f"{MODELS_DIR}/temperature_model.pkl",
            f"{MODELS_DIR}/precipitation_model.pkl",
            f"{MODELS_DIR}/weather_type_model.pkl",
            f"{MODELS_DIR}/pressure_level_model.pkl"
        ]
        return all(os.path.exists(f) for f in model_files)
    
    def _evaluate_and_visualize(self):
        """评估模型并生成可视化"""
        # 确保结果目录存在
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        # 加载测试数据进行预测
        self.predictor.load_data()
        self.predictor.split_data()
        
        # 进行预测
        predictions = self.predictor.predict(self.predictor.X_test)
        
        # 绘制预测结果
        self.visualizer.plot_prediction_results(predictions, self.predictor.y_test)
        
        # 绘制等压面图
        if self.visualizer.pressure_data is not None:
            self.visualizer.plot_pressure_level_map(time_idx=0, level=500)
            self.visualizer.plot_multiple_pressure_levels(time_idx=0)
        
        # 绘制模型性能对比
        self.visualizer.plot_model_performance(self.predictor.metrics)
        
        print("评估和可视化完成!")
    
    def _make_sample_predictions(self):
        """进行示例预测"""
        # 加载测试数据
        features_df = pd.read_csv(f"{DATA_DIR}/features.csv", index_col=0)
        
        # 选择前10个样本进行预测
        sample_features = features_df.head(10)
        
        # 进行预测
        predictions = self.predictor.predict(sample_features)
        
        print("\n示例预测结果:")
        print("-" * 50)
        
        for i in range(len(sample_features)):
            print(f"\n样本 {i+1}:")
            
            if 'temperature' in predictions:
                temp_pred = predictions['temperature'][i]
                temp_c = temp_pred - 273.15  # 转换为摄氏度
                print(f"  预测温度: {temp_pred:.2f} K ({temp_c:.2f}°C)")
            
            if 'precipitation' in predictions:
                precip_pred = predictions['precipitation'][i]
                precip_mm = precip_pred * 1000  # 转换为毫米
                print(f"  预测降水: {precip_pred:.6f} m ({precip_mm:.2f} mm)")
            
            if 'weather_type' in predictions:
                weather_pred = int(predictions['weather_type'][i])
                weather_labels = ['晴天', '多云', '小雨', '中雨', '大雨', '雪天']
                print(f"  预测天气: {weather_labels[weather_pred]}")
            
            if 'pressure_level' in predictions:
                z500_pred = predictions['pressure_level'][i]
                print(f"  预测500hPa位势高度: {z500_pred:.2f} m")
    
    def predict_specific_date(self, date_str):
        """预测特定日期的天气"""
        print(f"\n预测 {date_str} 的天气...")
        
        try:
            # 这里应该根据特定日期提取相应的特征
            # 为了简化，我们使用现有数据进行示例预测
            features_df = pd.read_csv(f"{DATA_DIR}/features.csv", index_col=0)
            sample_features = features_df.head(1)  # 使用第一个样本作为示例
            
            # 进行预测
            predictions = self.predictor.predict(sample_features)
            
            print(f"\n{date_str} 天气预测结果:")
            print("-" * 40)
            
            if 'temperature' in predictions:
                temp_pred = predictions['temperature'][0]
                temp_c = temp_pred - 273.15
                print(f"温度: {temp_c:.2f}°C")
            
            if 'precipitation' in predictions:
                precip_pred = predictions['precipitation'][0]
                precip_mm = precip_pred * 1000
                if precip_mm < 0.1:
                    precip_str = "无降水"
                elif precip_mm < 2:
                    precip_str = f"小雨 {precip_mm:.2f}mm"
                elif precip_mm < 10:
                    precip_str = f"中雨 {precip_mm:.2f}mm"
                else:
                    precip_str = f"大雨 {precip_mm:.2f}mm"
                print(f"降水: {precip_str}")
            
            if 'weather_type' in predictions:
                weather_pred = int(predictions['weather_type'][0])
                weather_labels = ['晴天', '多云', '小雨', '中雨', '大雨', '雪天']
                print(f"天气: {weather_labels[weather_pred]}")
            
            if 'pressure_level' in predictions:
                z500_pred = predictions['pressure_level'][0]
                print(f"500hPa位势高度: {z500_pred:.0f} m")
            
        except Exception as e:
            print(f"预测失败: {e}")
    
    def show_model_summary(self):
        """显示模型摘要信息"""
        print("\n模型摘要信息:")
        print("=" * 50)
        
        if not self._check_models_exist():
            print("模型尚未训练")
            return
        
        # 加载模型和指标
        self.predictor.load_models()
        
        for model_name, metrics in self.predictor.metrics.items():
            print(f"\n{model_name} 模型:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # 显示特征重要性
        print(f"\n温度模型特征重要性 (前10):")
        importance_df = self.predictor.get_feature_importance('temperature')
        if importance_df is not None:
            print(importance_df.head(10).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='ERA5机器学习天气预测系统')
    parser.add_argument('--mode', choices=['full', 'download', 'preprocess', 'train', 'predict', 'summary'], 
                       default='full', help='运行模式')
    parser.add_argument('--date', type=str, help='预测特定日期 (格式: YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    system = ERA5WeatherPrediction()
    
    if args.mode == 'full':
        system.run_full_pipeline()
    elif args.mode == 'download':
        system.downloader.download_era5_data()
        system.downloader.combine_data_files()
    elif args.mode == 'preprocess':
        system.preprocessor.load_data()
        system.preprocessor.clean_data()
        features, targets = system.preprocessor.prepare_ml_data()
        system.preprocessor.save_processed_data(features, targets)
    elif args.mode == 'train':
        system.predictor.train_all_models()
    elif args.mode == 'predict':
        if args.date:
            system.predict_specific_date(args.date)
        else:
            system._make_sample_predictions()
    elif args.mode == 'summary':
        system.show_model_summary()

if __name__ == "__main__":
    main()
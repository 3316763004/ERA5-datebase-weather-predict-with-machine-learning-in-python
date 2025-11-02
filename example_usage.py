"""
ERA5天气预测系统使用示例
演示如何使用各个模块进行天气预测
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入项目模块
from config import *
from data_downloader import ERA5DataDownloader
from data_preprocessor import ERA5DataPreprocessor
from ml_models import WeatherPredictor
from visualization import WeatherVisualizer

def example_1_download_data():
    """示例1: 下载数据"""
    print("=" * 50)
    print("示例1: 下载ERA5数据")
    print("=" * 50)
    
    downloader = ERA5DataDownloader()
    
    # 检查数据是否已存在
    surface_file = f"{DATA_DIR}/era5_surface_combined.nc"
    pressure_file = f"{DATA_DIR}/era5_pressure_combined.nc"
    
    if not (os.path.exists(surface_file) and os.path.exists(pressure_file)):
        print("开始下载ERA5数据...")
        downloader.download_era5_data()
        downloader.combine_data_files()
        print("数据下载完成!")
    else:
        print("数据已存在，跳过下载")

def example_2_preprocess_data():
    """示例2: 数据预处理"""
    print("=" * 50)
    print("示例2: 数据预处理")
    print("=" * 50)
    
    preprocessor = ERA5DataPreprocessor()
    
    try:
        # 加载数据
        preprocessor.load_data()
        print(f"地表数据维度: {preprocessor.surface_data.dims}")
        print(f"等压面数据维度: {preprocessor.pressure_data.dims}")
        
        # 清理数据
        preprocessor.clean_data()
        
        # 创建特征和目标
        features, targets = preprocessor.prepare_ml_data()
        
        print(f"特征数据形状: {features.shape}")
        print(f"目标数据形状: {targets.shape}")
        print(f"特征列名: {list(features.columns)[:5]}...")  # 显示前5个
        print(f"目标列名: {list(targets.columns)}")
        
        # 保存处理后的数据
        preprocessor.save_processed_data(features, targets)
        
        print("数据预处理完成!")
        
    except Exception as e:
        print(f"预处理失败: {e}")

def example_3_train_models():
    """示例3: 训练机器学习模型"""
    print("=" * 50)
    print("示例3: 训练机器学习模型")
    print("=" * 50)
    
    predictor = WeatherPredictor()
    
    try:
        # 训练所有模型
        predictor.train_all_models()
        
        # 显示模型性能
        print("\n模型性能总结:")
        for model_name, metrics in predictor.metrics.items():
            print(f"\n{model_name} 模型:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # 显示特征重要性
        print(f"\n温度模型特征重要性 (前5):")
        importance_df = predictor.get_feature_importance('temperature')
        if importance_df is not None:
            print(importance_df.head(5).to_string(index=False))
        
    except Exception as e:
        print(f"模型训练失败: {e}")

def example_4_make_predictions():
    """示例4: 进行预测"""
    print("=" * 50)
    print("示例4: 进行天气预测")
    print("=" * 50)
    
    predictor = WeatherPredictor()
    
    try:
        # 加载已训练的模型
        predictor.load_data()
        predictor.load_models()
        
        # 准备测试数据
        predictor.split_data()
        
        # 进行预测
        predictions = predictor.predict(predictor.X_test)
        
        print("预测结果示例:")
        print("-" * 40)
        
        # 显示前5个预测结果
        for i in range(5):
            print(f"\n样本 {i+1}:")
            
            if 'temperature' in predictions:
                temp_pred = predictions['temperature'][i]
                temp_c = temp_pred - 273.15
                print(f"  预测温度: {temp_c:.2f}°C")
            
            if 'precipitation' in predictions:
                precip_pred = predictions['precipitation'][i]
                precip_mm = precip_pred * 1000
                if precip_mm < 0.1:
                    precip_str = "无降水"
                else:
                    precip_str = f"{precip_mm:.2f}mm"
                print(f"  预测降水: {precip_str}")
            
            if 'weather_type' in predictions:
                weather_pred = int(predictions['weather_type'][i])
                weather_labels = ['晴天', '多云', '小雨', '中雨', '大雨', '雪天']
                print(f"  预测天气: {weather_labels[weather_pred]}")
        
    except Exception as e:
        print(f"预测失败: {e}")

def example_5_visualization():
    """示例5: 数据可视化"""
    print("=" * 50)
    print("示例5: 数据可视化")
    print("=" * 50)
    
    visualizer = WeatherVisualizer()
    
    try:
        # 加载预测结果进行可视化
        predictor = WeatherPredictor()
        predictor.load_data()
        predictor.load_models()
        predictor.split_data()
        
        predictions = predictor.predict(predictor.X_test)
        
        # 绘制预测结果
        visualizer.plot_prediction_results(predictions, predictor.y_test)
        
        # 绘制等压面图
        if visualizer.pressure_data is not None:
            visualizer.plot_pressure_level_map(time_idx=0, level=500)
            visualizer.plot_multiple_pressure_levels(time_idx=0)
        
        # 绘制模型性能
        visualizer.plot_model_performance(predictor.metrics)
        
        print("可视化完成! 图片保存在 results/ 目录")
        
    except Exception as e:
        print(f"可视化失败: {e}")

def example_6_custom_prediction():
    """示例6: 自定义预测"""
    print("=" * 50)
    print("示例6: 自定义天气预测")
    print("=" * 50)
    
    # 创建一个简单的天气场景
    print("创建一个典型的夏季场景...")
    
    # 这里我们使用现有的数据进行示例
    # 在实际应用中，您需要根据当前的天气条件创建特征
    
    predictor = WeatherPredictor()
    predictor.load_data()
    predictor.load_models()
    
    # 加载特征数据
    features_df = pd.read_csv(f"{DATA_DIR}/features.csv", index_col=0)
    
    # 选择一个夏季样本作为示例
    summer_sample = features_df.head(1)
    
    # 进行预测
    predictions = predictor.predict(summer_sample)
    
    print("\n夏季天气预测结果:")
    print("-" * 30)
    
    if 'temperature' in predictions:
        temp_pred = predictions['temperature'][0]
        temp_c = temp_pred - 273.15
        print(f"温度: {temp_c:.1f}°C")
        
        # 温度描述
        if temp_c > 30:
            temp_desc = "炎热"
        elif temp_c > 20:
            temp_desc = "温暖"
        elif temp_c > 10:
            temp_desc = "凉爽"
        else:
            temp_desc = "寒冷"
        print(f"体感: {temp_desc}")
    
    if 'precipitation' in predictions:
        precip_pred = predictions['precipitation'][0]
        precip_mm = precip_pred * 1000
        if precip_mm < 0.1:
            rain_desc = "无降水，适合户外活动"
        elif precip_mm < 5:
            rain_desc = "可能有小雨，记得带伞"
        elif precip_mm < 15:
            rain_desc = "有中雨，建议室内活动"
        else:
            rain_desc = "有大雨，注意安全"
        print(f"降水: {rain_desc}")
    
    if 'weather_type' in predictions:
        weather_pred = int(predictions['weather_type'][0])
        weather_labels = ['晴天', '多云', '小雨', '中雨', '大雨', '雪天']
        weather = weather_labels[weather_pred]
        
        # 天气建议
        suggestions = {
            '晴天': '阳光明媚，适合户外运动，注意防晒',
            '多云': '天气舒适，适合各种活动',
            '小雨': '记得带伞，路面湿滑注意安全',
            '中雨': '建议室内活动，如需外出请做好防护',
            '大雨': '避免外出，注意防范',
            '雪天': '注意保暖，路面结冰小心行走'
        }
        
        print(f"天气: {weather}")
        print(f"建议: {suggestions[weather]}")

def main():
    """运行所有示例"""
    print("ERA5天气预测系统 - 使用示例")
    print("=" * 60)
    
    examples = [
        ("下载数据", example_1_download_data),
        ("数据预处理", example_2_preprocess_data),
        ("训练模型", example_3_train_models),
        ("进行预测", example_4_make_predictions),
        ("数据可视化", example_5_visualization),
        ("自定义预测", example_6_custom_prediction)
    ]
    
    print("可用的示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\n选择要运行的示例 (输入数字，或 'all' 运行所有):")
    choice = input().strip()
    
    if choice.lower() == 'all':
        for name, func in examples:
            try:
                func()
                print(f"\n✅ {name} 示例完成\n")
            except Exception as e:
                print(f"\n❌ {name} 示例失败: {e}\n")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                func()
                print(f"\n✅ {name} 示例完成")
            else:
                print("无效的选择")
        except ValueError:
            print("请输入有效的数字")

if __name__ == "__main__":
    main()
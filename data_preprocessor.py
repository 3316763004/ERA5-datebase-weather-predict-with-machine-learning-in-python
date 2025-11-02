"""
ERA5数据预处理器
用于处理和准备ERA5数据以供机器学习模型使用
"""

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from config import *

class ERA5DataPreprocessor:
    def __init__(self):
        self.surface_data = None
        self.pressure_data = None
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
    
    def load_data(self):
        """加载ERA5数据"""
        print("加载ERA5数据...")
        
        surface_file = f"{DATA_DIR}/era5_surface_combined.nc"
        pressure_file = f"{DATA_DIR}/era5_pressure_combined.nc"
        
        if not os.path.exists(surface_file) or not os.path.exists(pressure_file):
            raise FileNotFoundError("ERA5数据文件不存在，请先运行数据下载器")
        
        self.surface_data = xr.open_dataset(surface_file)
        self.pressure_data = xr.open_dataset(pressure_file)
        
        print(f"地表数据形状: {self.surface_data.dims}")
        print(f"等压面数据形状: {self.pressure_data.dims}")
    
    def clean_data(self):
        """清理数据，处理缺失值"""
        print("清理数据...")
        
        # 处理地表数据
        for var in self.surface_data.data_vars:
            # 用前向填充处理缺失值
            self.surface_data[var] = self.surface_data[var].ffill(dim='time')
            # 如果还有缺失值，用后向填充
            self.surface_data[var] = self.surface_data[var].bfill(dim='time')
        
        # 处理等压面数据
        for var in self.pressure_data.data_vars:
            self.pressure_data[var] = self.pressure_data[var].ffill(dim='time')
            self.pressure_data[var] = self.pressure_data[var].bfill(dim='time')
        
        print("数据清理完成")
    
    def create_features(self):
        """创建特征变量"""
        print("创建特征变量...")
        
        # 从地表数据提取特征
        surface_features = []
        
        # 温度相关特征
        if 't2m' in self.surface_data:
            surface_features.append(self.surface_data['t2m'])
            # 温度梯度
            temp_grad_lat = self.surface_data['t2m'].differentiate('latitude')
            temp_grad_lon = self.surface_data['t2m'].differentiate('longitude')
            surface_features.extend([temp_grad_lat, temp_grad_lon])
        
        # 降水相关特征
        if 'tp' in self.surface_data:
            surface_features.append(self.surface_data['tp'])
            # 12小时累积降水
            precip_12h = self.surface_data['tp'].rolling(time=2).sum()
            surface_features.append(precip_12h)
        
        # 风场相关特征
        if 'u10' in self.surface_data and 'v10' in self.surface_data:
            surface_features.extend([self.surface_data['u10'], self.surface_data['v10']])
            # 风速
            wind_speed = np.sqrt(self.surface_data['u10']**2 + self.surface_data['v10']**2)
            surface_features.append(wind_speed)
        
        # 气压相关特征
        if 'msl' in self.surface_data:
            surface_features.append(self.surface_data['msl'])
            # 气压梯度
            press_grad_lat = self.surface_data['msl'].differentiate('latitude')
            press_grad_lon = self.surface_data['msl'].differentiate('longitude')
            surface_features.extend([press_grad_lat, press_grad_lon])
        
        # 从等压面数据提取特征
        pressure_features = []
        
        # 主要等压面的位势高度
        if 'z' in self.pressure_data:
            # 选择关键等压面
            key_levels = [500, 700, 850]  # 关键等压面
            for level in key_levels:
                if level in self.pressure_data.level:
                    z_at_level = self.pressure_data['z'].sel(level=level)
                    pressure_features.append(z_at_level)
        
        # 相对湿度
        if 'r' in self.pressure_data:
            # 选择关键等压面
            key_levels = [850, 700, 500]
            for level in key_levels:
                if level in self.pressure_data.level:
                    r_at_level = self.pressure_data['r'].sel(level=level)
                    pressure_features.append(r_at_level)
        
        # 时间特征
        time_features = self._create_time_features()
        
        # 合并所有特征
        all_features = surface_features + pressure_features + time_features
        
        # 转换为DataFrame
        feature_df = self._convert_to_dataframe(all_features)
        
        self.feature_columns = list(feature_df.columns)
        print(f"创建了 {len(self.feature_columns)} 个特征")
        
        return feature_df
    
    def _create_time_features(self):
        """创建时间相关特征"""
        time_features = []
        
        # 获取时间坐标
        times = self.surface_data.time
        
        # 小时（循环编码）
        hour_sin = np.sin(2 * np.pi * times.dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * times.dt.hour / 24)
        
        # 月份（循环编码）
        month_sin = np.sin(2 * np.pi * times.dt.month / 12)
        month_cos = np.cos(2 * np.pi * times.dt.month / 12)
        
        # 日年（循环编码）
        dayofyear_sin = np.sin(2 * np.pi * times.dt.dayofyear / 365.25)
        dayofyear_cos = np.cos(2 * np.pi * times.dt.dayofyear / 365.25)
        
        time_features = [hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos]
        
        return time_features
    
    def _convert_to_dataframe(self, features):
        """将xarray特征转换为pandas DataFrame"""
        # 将所有特征堆叠成一个数组
        stacked_features = []
        
        for feature in features:
            # 展平空间维度
            flattened = feature.stack(points=('latitude', 'longitude'))
            stacked_features.append(flattened)
        
        # 合并所有特征
        combined = xr.concat(stacked_features, dim='variable')
        
        # 转换为DataFrame
        df = combined.to_pandas()
        df.columns = [f'feature_{i}' for i in range(len(df.columns))]
        
        return df
    
    def create_targets(self):
        """创建目标变量"""
        print("创建目标变量...")
        
        targets = []
        
        # 目标1: 24小时后的温度
        if 't2m' in self.surface_data:
            temp_future = self.surface_data['t2m'].shift(time=-1)  # 向前移动1个时间步
            targets.append(temp_future)
        
        # 目标2: 24小时后的降水
        if 'tp' in self.surface_data:
            precip_future = self.surface_data['tp'].shift(time=-1)
            targets.append(precip_future)
        
        # 目标3: 天气类型（基于温度和降水分类）
        if 't2m' in self.surface_data and 'tp' in self.surface_data:
            weather_type = self._classify_weather(self.surface_data['t2m'], self.surface_data['tp'])
            targets.append(weather_type)
        
        # 目标4: 500hPa位势高度（用于等压面预测）
        if 'z' in self.pressure_data and 500 in self.pressure_data.level:
            z500_future = self.pressure_data['z'].sel(level=500).shift(time=-1)
            targets.append(z500_future)
        
        # 转换为DataFrame
        target_df = self._convert_targets_to_dataframe(targets)
        
        self.target_columns = list(target_df.columns)
        print(f"创建了 {len(self.target_columns)} 个目标变量")
        
        return target_df
    
    def _classify_weather(self, temperature, precipitation):
        """简单天气分类"""
        # 基于温度和降水量的简单分类
        # 0: 晴天, 1: 多云, 2: 小雨, 3: 中雨, 4: 大雨, 5: 雪
        
        weather = xr.full_like(temperature, 0, dtype=int)
        
        # 温度阈值（摄氏度）
        temp_c = temperature - 273.15  # 转换为摄氏度
        
        # 降水阈值（米）
        precip_m = precipitation * 1000  # 转换为毫米
        
        # 分类逻辑
        # 大雨
        weather = xr.where(precip_m > 10, 4, weather)
        # 中雨
        weather = xr.where((precip_m > 2) & (precip_m <= 10), 3, weather)
        # 小雨
        weather = xr.where((precip_m > 0.1) & (precip_m <= 2), 2, weather)
        # 雪天（温度<0且有降水）
        weather = xr.where((temp_c < 0) & (precip_m > 0.1), 5, weather)
        # 多云（温度适中但无降水）
        weather = xr.where((temp_c >= 0) & (temp_c <= 25) & (precip_m <= 0.1), 1, weather)
        
        return weather
    
    def _convert_targets_to_dataframe(self, targets):
        """将目标变量转换为DataFrame"""
        stacked_targets = []
        
        for target in targets:
            flattened = target.stack(points=('latitude', 'longitude'))
            stacked_targets.append(flattened)
        
        combined = xr.concat(stacked_targets, dim='variable')
        df = combined.to_pandas()
        df.columns = ['target_temp', 'target_precip', 'target_weather', 'target_z500']
        
        return df
    
    def prepare_ml_data(self):
        """准备机器学习数据"""
        print("准备机器学习数据...")
        
        # 创建特征和目标
        features_df = self.create_features()
        targets_df = self.create_targets()
        
        # 移除包含NaN的行
        valid_indices = ~(features_df.isna().any(axis=1) | targets_df.isna().any(axis=1))
        
        features_clean = features_df[valid_indices]
        targets_clean = targets_df[valid_indices]
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # 保存标准化器
        self.scalers['feature_scaler'] = scaler
        
        # 转换回DataFrame
        features_final = pd.DataFrame(
            features_scaled,
            columns=features_clean.columns,
            index=features_clean.index
        )
        
        print(f"最终数据形状: 特征 {features_final.shape}, 目标 {targets_clean.shape}")
        
        return features_final, targets_clean
    
    def save_processed_data(self, features, targets):
        """保存处理后的数据"""
        print("保存处理后的数据...")
        
        features.to_csv(f"{DATA_DIR}/features.csv")
        targets.to_csv(f"{DATA_DIR}/targets.csv")
        
        # 保存标准化器
        import joblib
        joblib.dump(self.scalers, f"{DATA_DIR}/scalers.pkl")
        
        print("数据保存完成")

if __name__ == "__main__":
    preprocessor = ERA5DataPreprocessor()
    
    try:
        preprocessor.load_data()
        preprocessor.clean_data()
        features, targets = preprocessor.prepare_ml_data()
        preprocessor.save_processed_data(features, targets)
        
        print("数据预处理完成!")
        
    except Exception as e:
        print(f"数据预处理失败: {e}")
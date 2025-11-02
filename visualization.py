"""
可视化模块
用于展示ERA5天气预测结果和等压面图
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import os
from config import *

class WeatherVisualizer:
    def __init__(self):
        self.setup_style()
        self.load_data()
    
    def setup_style(self):
        """设置绘图样式"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        sns.set_palette("husl")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self):
        """加载原始ERA5数据用于可视化"""
        try:
            surface_file = f"{DATA_DIR}/era5_surface_combined.nc"
            pressure_file = f"{DATA_DIR}/era5_pressure_combined.nc"
            
            if os.path.exists(surface_file):
                self.surface_data = xr.open_dataset(surface_file)
            else:
                self.surface_data = None
            
            if os.path.exists(pressure_file):
                self.pressure_data = xr.open_dataset(pressure_file)
            else:
                self.pressure_data = None
                
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.surface_data = None
            self.pressure_data = None
    
    def plot_prediction_results(self, predictions, actual=None, save_dir=RESULTS_DIR):
        """绘制预测结果对比图"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 温度预测结果
        if 'temperature' in predictions:
            self._plot_temperature_predictions(predictions['temperature'], actual, save_dir)
        
        # 降水预测结果
        if 'precipitation' in predictions:
            self._plot_precipitation_predictions(predictions['precipitation'], actual, save_dir)
        
        # 天气类型预测结果
        if 'weather_type' in predictions:
            self._plot_weather_predictions(predictions['weather_type'], actual, save_dir)
    
    def _plot_temperature_predictions(self, pred_temp, actual, save_dir):
        """绘制温度预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 预测值分布
        axes[0, 0].hist(pred_temp, bins=50, alpha=0.7, label='预测值', color='skyblue')
        if actual is not None and 'target_temp' in actual.columns:
            axes[0, 0].hist(actual['target_temp'].dropna(), bins=50, alpha=0.7, label='实际值', color='orange')
        axes[0, 0].set_xlabel('温度 (K)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('温度预测值分布')
        axes[0, 0].legend()
        
        # 预测值 vs 实际值散点图
        if actual is not None and 'target_temp' in actual.columns:
            valid_mask = ~(pd.Series(pred_temp).isna() | actual['target_temp'].isna())
            if valid_mask.any():
                axes[0, 1].scatter(
                    actual.loc[valid_mask, 'target_temp'], 
                    pd.Series(pred_temp)[valid_mask], 
                    alpha=0.5, s=1
                )
                axes[0, 1].plot([actual['target_temp'].min(), actual['target_temp'].max()], 
                               [actual['target_temp'].min(), actual['target_temp'].max()], 
                               'r--', label='完美预测线')
                axes[0, 1].set_xlabel('实际温度 (K)')
                axes[0, 1].set_ylabel('预测温度 (K)')
                axes[0, 1].set_title('温度预测 vs 实际')
                axes[0, 1].legend()
        
        # 残差图
        if actual is not None and 'target_temp' in actual.columns:
            valid_mask = ~(pd.Series(pred_temp).isna() | actual['target_temp'].isna())
            if valid_mask.any():
                residuals = pd.Series(pred_temp)[valid_mask] - actual.loc[valid_mask, 'target_temp']
                axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='coral')
                axes[1, 0].set_xlabel('残差 (K)')
                axes[1, 0].set_ylabel('频次')
                axes[1, 0].set_title('温度预测残差分布')
                axes[1, 0].axvline(x=0, color='red', linestyle='--')
        
        # 时间序列（如果有时间信息）
        axes[1, 1].plot(pred_temp[:1000], label='预测值', alpha=0.7)
        if actual is not None and 'target_temp' in actual.columns:
            axes[1, 1].plot(actual['target_temp'].iloc[:1000], label='实际值', alpha=0.7)
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('温度 (K)')
        axes[1, 1].set_title('温度预测时间序列')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/temperature_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"温度预测图已保存: {save_dir}/temperature_predictions.png")
    
    def _plot_precipitation_predictions(self, pred_precip, actual, save_dir):
        """绘制降水预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 预测值分布
        axes[0, 0].hist(pred_precip, bins=50, alpha=0.7, label='预测值', color='lightblue', log=True)
        if actual is not None and 'target_precip' in actual.columns:
            axes[0, 0].hist(actual['target_precip'].dropna(), bins=50, alpha=0.7, label='实际值', color='darkblue', log=True)
        axes[0, 0].set_xlabel('降水量 (m)')
        axes[0, 0].set_ylabel('频次 (log)')
        axes[0, 0].set_title('降水预测值分布')
        axes[0, 0].legend()
        
        # 预测值 vs 实际值散点图
        if actual is not None and 'target_precip' in actual.columns:
            valid_mask = ~(pd.Series(pred_precip).isna() | actual['target_precip'].isna())
            # 只显示有降水的点
            if valid_mask.any():
                precip_mask = (pd.Series(pred_precip)[valid_mask] > 0.001) | (actual.loc[valid_mask, 'target_precip'] > 0.001)
                if precip_mask.any():
                    axes[0, 1].scatter(
                        actual.loc[valid_mask, 'target_precip'][precip_mask], 
                        pd.Series(pred_precip)[valid_mask][precip_mask], 
                        alpha=0.5, s=1
                    )
                    max_val = max(actual['target_precip'].max(), pd.Series(pred_precip).max())
                    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='完美预测线')
                    axes[0, 1].set_xlabel('实际降水 (m)')
                    axes[0, 1].set_ylabel('预测降水 (m)')
                    axes[0, 1].set_title('降水预测 vs 实际')
                    axes[0, 1].legend()
        
        # 累积降水分布
        axes[1, 0].hist(pred_precip[pred_precip > 0.001], bins=30, alpha=0.7, label='预测值', color='green')
        if actual is not None and 'target_precip' in actual.columns:
            actual_precip = actual['target_precip'].dropna()
            axes[1, 0].hist(actual_precip[actual_precip > 0.001], bins=30, alpha=0.7, label='实际值', color='darkgreen')
        axes[1, 0].set_xlabel('降水量 (m)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('有降水时的分布')
        axes[1, 0].legend()
        
        # 时间序列
        axes[1, 1].plot(pred_precip[:1000] * 1000, label='预测值 (mm)', alpha=0.7)
        if actual is not None and 'target_precip' in actual.columns:
            axes[1, 1].plot(actual['target_precip'].iloc[:1000] * 1000, label='实际值 (mm)', alpha=0.7)
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('降水量 (mm)')
        axes[1, 1].set_title('降水预测时间序列')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/precipitation_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"降水预测图已保存: {save_dir}/precipitation_predictions.png")
    
    def _plot_weather_predictions(self, pred_weather, actual, save_dir):
        """绘制天气类型预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        weather_labels = ['晴天', '多云', '小雨', '中雨', '大雨', '雪天']
        colors = ['gold', 'lightgray', 'lightblue', 'blue', 'darkblue', 'white']
        
        # 预测值分布
        pred_counts = pd.Series(pred_weather).value_counts().sort_index()
        axes[0, 0].bar(range(len(pred_counts)), pred_counts.values, color=colors[:len(pred_counts)])
        axes[0, 0].set_xticks(range(len(pred_counts)))
        axes[0, 0].set_xticklabels([weather_labels[i] for i in pred_counts.index])
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('天气类型预测分布')
        
        # 实际值分布
        if actual is not None and 'target_weather' in actual.columns:
            actual_counts = actual['target_weather'].value_counts().sort_index()
            axes[0, 1].bar(range(len(actual_counts)), actual_counts.values, color=colors[:len(actual_counts)])
            axes[0, 1].set_xticks(range(len(actual_counts)))
            axes[0, 1].set_xticklabels([weather_labels[i] for i in actual_counts.index])
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].set_title('天气类型实际分布')
            
            # 混淆矩阵
            from sklearn.metrics import confusion_matrix
            valid_mask = ~(pd.Series(pred_weather).isna() | actual['target_weather'].isna())
            if valid_mask.any():
                cm = confusion_matrix(actual.loc[valid_mask, 'target_weather'], pd.Series(pred_weather)[valid_mask])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                           xticklabels=weather_labels[:len(cm)], yticklabels=weather_labels[:len(cm)])
                axes[1, 0].set_xlabel('预测天气类型')
                axes[1, 0].set_ylabel('实际天气类型')
                axes[1, 0].set_title('天气类型混淆矩阵')
        
        # 时间序列
        axes[1, 1].plot(pred_weather[:1000], 'o-', markersize=2, label='预测值')
        if actual is not None and 'target_weather' in actual.columns:
            axes[1, 1].plot(actual['target_weather'].iloc[:1000], 'o-', markersize=2, alpha=0.7, label='实际值')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('天气类型')
        axes[1, 1].set_title('天气类型预测时间序列')
        axes[1, 1].set_yticks(range(6))
        axes[1, 1].set_yticklabels(weather_labels)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/weather_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"天气类型预测图已保存: {save_dir}/weather_predictions.png")
    
    def plot_pressure_level_map(self, time_idx=0, level=500, save_dir=RESULTS_DIR):
        """绘制等压面图"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.pressure_data is None:
            print("等压面数据不存在")
            return
        
        if 'z' not in self.pressure_data:
            print("位势高度数据不存在")
            return
        
        # 选择特定时间和等压面
        try:
            geopotential = self.pressure_data['z'].isel(time=time_idx).sel(level=level)
            
            # 转换位势高度（从米到位势米）
            geopotential_gpm = geopotential / 9.80665
            
            # 创建地图
            fig = plt.figure(figsize=(12, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # 添加地理特征
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            
            # 绘制等值线
            contour = ax.contour(
                geopotential_gpm.longitude, 
                geopotential_gpm.latitude, 
                geopotential_gpm,
                levels=20,
                colors='black',
                linewidths=1.0,
                transform=ccrs.PlateCarree()
            )
            
            # 添加等值线标签
            ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
            
            # 填充颜色
            contourf = ax.contourf(
                geopotential_gpm.longitude,
                geopotential_gpm.latitude,
                geopotential_gpm,
                levels=20,
                cmap='viridis',
                alpha=0.6,
                transform=ccrs.PlateCarree()
            )
            
            # 添加颜色条
            cbar = plt.colorbar(contourf, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label(f'{level}hPa 位势高度 (gpm)', fontsize=10)
            
            # 设置地图范围
            ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], crs=ccrs.PlateCarree())
            
            # 添加网格线
            ax.gridlines(draw_labels=True, alpha=0.3)
            
            # 标题
            time_str = str(self.pressure_data.isel(time=time_idx).time.values)
            plt.title(f'{level}hPa 等压面图\n{time_str}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/pressure_level_{level}hPa.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"等压面图已保存: {save_dir}/pressure_level_{level}hPa.png")
            
        except Exception as e:
            print(f"绘制等压面图时出错: {e}")
    
    def plot_multiple_pressure_levels(self, time_idx=0, save_dir=RESULTS_DIR):
        """绘制多个等压面的组合图"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.pressure_data is None:
            print("等压面数据不存在")
            return
        
        levels_to_plot = [1000, 850, 700, 500, 300, 200]
        n_levels = len(levels_to_plot)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, level in enumerate(levels_to_plot):
            if i >= len(axes):
                break
                
            try:
                # 选择特定时间和等压面
                if level in self.pressure_data.level:
                    geopotential = self.pressure_data['z'].isel(time=time_idx).sel(level=level)
                    geopotential_gpm = geopotential / 9.80665
                    
                    # 在子图中绘制
                    ax = axes[i]
                    
                    # 绘制等值线
                    contour = ax.contour(
                        geopotential_gpm.longitude,
                        geopotential_gpm.latitude,
                        geopotential_gpm,
                        levels=15,
                        colors='black',
                        linewidths=0.8
                    )
                    
                    # 填充颜色
                    contourf = ax.contourf(
                        geopotential_gpm.longitude,
                        geopotential_gpm.latitude,
                        geopotential_gpm,
                        levels=15,
                        cmap='viridis',
                        alpha=0.7
                    )
                    
                    ax.set_title(f'{level} hPa', fontsize=12, fontweight='bold')
                    ax.set_xlabel('经度')
                    ax.set_ylabel('纬度')
                    
                    # 添加等值线标签
                    ax.clabel(contour, inline=True, fontsize=6, fmt='%1.0f')
                    
            except Exception as e:
                axes[i].text(0.5, 0.5, f'错误: {e}', transform=axes[i].transAxes,
                           ha='center', va='center')
                axes[i].set_title(f'{level} hPa', fontsize=12)
        
        # 整体标题
        time_str = str(self.pressure_data.isel(time=time_idx).time.values) if self.pressure_data else "未知时间"
        fig.suptitle(f'多等压面图 - {time_str}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/multiple_pressure_levels.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"多等压面图已保存: {save_dir}/multiple_pressure_levels.png")
    
    def plot_model_performance(self, metrics, save_dir=RESULTS_DIR):
        """绘制模型性能对比"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 回归模型性能对比
        regression_models = ['temperature', 'precipitation', 'pressure_level']
        regression_metrics = ['mse', 'mae', 'r2']
        
        for i, metric in enumerate(regression_metrics):
            ax = axes[i//2, i%2]
            values = []
            labels = []
            
            for model in regression_models:
                if model in metrics and metric in metrics[model]:
                    values.append(metrics[model][metric])
                    labels.append(model)
            
            if values:
                bars = ax.bar(labels, values, alpha=0.7)
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} 对比')
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # 分类模型性能
        if 'weather_type' in metrics:
            ax = axes[1, 1]
            accuracy = metrics['weather_type'].get('accuracy', 0)
            ax.bar(['准确率'], [accuracy], color='green', alpha=0.7)
            ax.set_ylabel('准确率')
            ax.set_title('天气类型分类性能')
            ax.text(0, accuracy + 0.01, f'{accuracy:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"模型性能图已保存: {save_dir}/model_performance.png")

if __name__ == "__main__":
    visualizer = WeatherVisualizer()
    
    # 绘制等压面图示例
    visualizer.plot_pressure_level_map(time_idx=0, level=500)
    visualizer.plot_multiple_pressure_levels(time_idx=0)
    
    print("可视化完成!")
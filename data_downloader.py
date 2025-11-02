"""
ERA5数据下载器
用于下载小规模的ERA5数据集用于机器学习天气预测
"""

import os
import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import *

class ERA5DataDownloader:
    def __init__(self):
        self.cds = cdsapi.Client()
        self.ensure_directories()
    
    def ensure_directories(self):
        """创建必要的目录"""
        for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")
    
    def download_era5_data(self):
        """下载ERA5数据"""
        print("开始下载ERA5数据...")
        
        # 转换日期格式
        start_date = datetime.strptime(TIME_RANGE['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(TIME_RANGE['end_date'], '%Y-%m-%d')
        
        # 生成日期列表
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # 分批下载数据以避免请求过大
        batch_size = 30  # 每批30天
        
        for i in range(0, len(dates), batch_size):
            batch_dates = dates[i:i+batch_size]
            print(f"下载批次 {i//batch_size + 1}: 日期 {batch_dates[0]} 到 {batch_dates[-1]}")
            
            try:
                # 下载地表数据
                self._download_surface_data(batch_dates, i//batch_size + 1)
                
                # 下载等压面数据
                self._download_pressure_level_data(batch_dates, i//batch_size + 1)
                
            except Exception as e:
                print(f"下载批次 {i//batch_size + 1} 时出错: {e}")
                continue
        
        print("ERA5数据下载完成!")
    
    def _download_surface_data(self, dates, batch_num):
        """下载地表数据"""
        request = {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature',
                'total_precipitation',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'mean_sea_level_pressure'
            ],
            'year': list(set([d[:4] for d in dates])),
            'month': list(set([d[5:7] for d in dates])),
            'day': list(set([d[8:10] for d in dates])),
            'time': [
                '00:00', '06:00', '12:00', '18:00'
            ],
            'area': LAT_RANGE + LON_RANGE,  # North, West, South, East
            'format': 'netcdf',
            'grid': [GRID_RESOLUTION, GRID_RESOLUTION]
        }
        
        output_file = f"{DATA_DIR}/era5_surface_batch_{batch_num}.nc"
        
        self.cds.retrieve(
            'reanalysis-era5-single-levels',
            request,
            output_file
        )
        
        print(f"地表数据已保存到: {output_file}")
    
    def _download_pressure_level_data(self, dates, batch_num):
        """下载等压面数据"""
        request = {
            'product_type': 'reanalysis',
            'variable': [
                'geopotential',
                'relative_humidity',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind'
            ],
            'pressure_level': PRESSURE_LEVELS,
            'year': list(set([d[:4] for d in dates])),
            'month': list(set([d[5:7] for d in dates])),
            'day': list(set([d[8:10] for d in dates])),
            'time': [
                '00:00', '06:00', '12:00', '18:00'
            ],
            'area': LAT_RANGE + LON_RANGE,
            'format': 'netcdf',
            'grid': [GRID_RESOLUTION, GRID_RESOLUTION]
        }
        
        output_file = f"{DATA_DIR}/era5_pressure_batch_{batch_num}.nc"
        
        self.cds.retrieve(
            'reanalysis-era5-pressure-levels',
            request,
            output_file
        )
        
        print(f"等压面数据已保存到: {output_file}")
    
    def combine_data_files(self):
        """合并下载的数据文件"""
        print("合并数据文件...")
        
        # 查找所有地表数据文件
        surface_files = []
        pressure_files = []
        
        for file in os.listdir(DATA_DIR):
            if file.startswith('era5_surface_') and file.endswith('.nc'):
                surface_files.append(os.path.join(DATA_DIR, file))
            elif file.startswith('era5_pressure_') and file.endswith('.nc'):
                pressure_files.append(os.path.join(DATA_DIR, file))
        
        # 合并地表数据
        if surface_files:
            surface_datasets = [xr.open_dataset(f) for f in sorted(surface_files)]
            combined_surface = xr.concat(surface_datasets, dim='time')
            combined_surface.to_netcdf(f"{DATA_DIR}/era5_surface_combined.nc")
            print(f"合并地表数据保存到: {DATA_DIR}/era5_surface_combined.nc")
        
        # 合并等压面数据
        if pressure_files:
            pressure_datasets = [xr.open_dataset(f) for f in sorted(pressure_files)]
            combined_pressure = xr.concat(pressure_datasets, dim='time')
            combined_pressure.to_netcdf(f"{DATA_DIR}/era5_pressure_combined.nc")
            print(f"合并等压面数据保存到: {DATA_DIR}/era5_pressure_combined.nc")
        
        # 删除临时文件
        for f in surface_files + pressure_files:
            os.remove(f)
        
        print("数据合并完成!")

if __name__ == "__main__":
    downloader = ERA5DataDownloader()
    
    # 检查是否已有数据
    if os.path.exists(f"{DATA_DIR}/era5_surface_combined.nc") and \
       os.path.exists(f"{DATA_DIR}/era5_pressure_combined.nc"):
        print("数据文件已存在，跳过下载")
    else:
        downloader.download_era5_data()
        downloader.combine_data_files()
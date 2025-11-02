import os

# ERA5 API Configuration
CDS_API_URL = "https://cds.climate.copernicus.eu/api/v2"

# Data Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# ERA5 Variables to Download
ERA5_VARIABLES = [
    '2m_temperature',      # 2米温度
    'total_precipitation',  # 总降水量
    '10m_u_component_of_wind',  # 10米风速U分量
    '10m_v_component_of_wind',  # 10米风速V分量
    'mean_sea_level_pressure',  # 海平面气压
    'geopotential',        # 位势高度
    'relative_humidity'    # 相对湿度
]

# Pressure Levels for Isobaric Surfaces
PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 200, 100]

# Geographic Extent (smaller region for smaller dataset)
# Example: East Asia region
LAT_RANGE = [20, 50]  # 纬度范围
LON_RANGE = [100, 140]  # 经度范围

# Time Range (small period for smaller dataset)
TIME_RANGE = {
    'start_date': '2020-01-01',
    'end_date': '2020-12-31'  # One year of data
}

# Model Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Grid Resolution
GRID_RESOLUTION = 2.5  # degrees (coarser resolution for smaller dataset)
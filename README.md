# ERA5机器学习天气预测系统

基于ERA5再分析数据的机器学习天气预测项目，使用Python实现温度、降水量、天气类型和等压面的预测。

## 项目特点

- ✅ 使用小规模ERA5数据集，适合学习和研究
- ✅ 完整的机器学习流水线：数据下载→预处理→训练→预测→可视化
- ✅ 多种预测目标：温度、降水、天气类型、等压面
- ✅ 多种机器学习模型：XGBoost、Random Forest
- ✅ 专业的气象可视化：等压面图、预测结果对比图
- ✅ 模块化设计，易于扩展和维护

## 系统架构

```
ERA5天气预测系统/
├── main.py              # 主程序入口
├── config.py            # 配置文件
├── data_downloader.py   # ERA5数据下载器
├── data_preprocessor.py # 数据预处理器
├── ml_models.py         # 机器学习模型
├── visualization.py     # 可视化模块
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明
```

## 安装和设置

### 1. 环境要求

- Python 3.8+
- CDS API账户（用于下载ERA5数据）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置CDS API

在用户主目录创建`.cdsapirc`文件：

```ini
url: https://cds.climate.copernicus.eu/api/v2
key: 你的UID:你的API密钥
```

获取API密钥：
1. 访问 [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. 注册账户
3. 在用户设置中找到API密钥

## 使用方法

### 快速开始

运行完整流水线（推荐）：

```bash
python main.py --mode full
```

这将自动完成：
1. 下载ERA5数据（约1年的东亚区域数据）
2. 数据预处理和特征工程
3. 训练多个机器学习模型
4. 模型评估和可视化
5. 生成预测示例

### 分步运行

如果需要分步控制，可以使用以下命令：

```bash
# 仅下载数据
python main.py --mode download

# 仅预处理数据
python main.py --mode preprocess

# 仅训练模型
python main.py --mode train

# 仅进行预测
python main.py --mode predict

# 预测特定日期
python main.py --mode predict --date 2020-06-15

# 查看模型摘要
python main.py --mode summary
```

## 数据说明

### ERA5数据集

- **时间范围**: 2020年1月1日 - 2020年12月31日
- **空间范围**: 东亚地区 (20°N-50°N, 100°E-140°E)
- **时间分辨率**: 6小时间隔
- **空间分辨率**: 2.5° × 2.5°

### 预测变量

1. **温度** (2m_temperature)
   - 单位：开尔文(K)
   - 预测24小时后的温度

2. **降水量** (total_precipitation)
   - 单位：米(m)
   - 预测24小时后的累积降水量

3. **天气类型** (weather_type)
   - 分类：晴天、多云、小雨、中雨、大雨、雪天
   - 基于温度和降水的自动分类

4. **等压面** (500hPa位势高度)
   - 单位：米(m)
   - 用于高空天气分析

## 机器学习模型

### 模型类型

1. **XGBoost回归器**
   - 用于温度和等压面预测
   - 处理复杂的非线性关系

2. **Random Forest回归器**
   - 用于降水量预测
   - 对异常值鲁棒

3. **Random Forest分类器**
   - 用于天气类型分类
   - 提供概率输出

### 特征工程

- **气象特征**: 温度梯度、气压梯度、风速、湿度等
- **时间特征**: 小时、月份、日年的循环编码
- **空间特征**: 不同等压面的位势高度
- **衍生特征**: 12小时累积降水、风场合成等

### 模型评估

- **回归指标**: MSE、MAE、R²
- **分类指标**: 准确率、混淆矩阵、分类报告

## 可视化输出

系统会在`results/`目录生成以下图表：

1. **预测结果图**
   - `temperature_predictions.png`: 温度预测对比
   - `precipitation_predictions.png`: 降水预测对比
   - `weather_predictions.png`: 天气类型预测对比

2. **等压面图**
   - `pressure_level_500hPa.png`: 500hPa等压面图
   - `multiple_pressure_levels.png`: 多等压面组合图

3. **模型性能图**
   - `model_performance.png`: 各模型性能对比

## 配置说明

主要配置在`config.py`文件中：

```python
# 数据配置
LAT_RANGE = [20, 50]        # 纬度范围
LON_RANGE = [100, 140]      # 经度范围
TIME_RANGE = {
    'start_date': '2020-01-01',
    'end_date': '2020-12-31'
}

# 模型配置
TEST_SIZE = 0.2             # 测试集比例
N_ESTIMATORS = 100          # 树的数量
MAX_DEPTH = 10              # 树的最大深度
```

## 输出示例

```
ERA5机器学习天气预测系统
============================================================

步骤1: 数据已存在，跳过下载

步骤2: 预处理数据已存在，跳过预处理

步骤3: 模型已存在，加载现有模型

步骤4: 模型评估和可视化...
温度预测图已保存: results/temperature_predictions.png
降水预测图已保存: results/precipitation_predictions.png
天气类型预测图已保存: results/weather_predictions.png
等压面图已保存: results/pressure_level_500hPa.png
多等压面图已保存: results/multiple_pressure_levels.png
模型性能图已保存: results/model_performance.png
评估和可视化完成!

步骤5: 示例预测...

示例预测结果:
--------------------------------------------------

样本 1:
  预测温度: 285.32 K (12.17°C)
  预测降水: 0.000000 m (0.00 mm)
  预测天气: 晴天
  预测500hPa位势高度: 5620.45 m
```

## 扩展功能

### 添加新的预测变量

1. 在`config.py`中添加新的ERA5变量
2. 在`data_preprocessor.py`中创建相应的特征
3. 在`ml_models.py`中添加新的模型训练方法
4. 在`visualization.py`中添加可视化代码

### 尝试其他机器学习模型

- LSTM/GRU（时间序列预测）
- CNN（空间特征提取）
- Transformer（序列建模）
- 集成方法（投票、堆叠）

### 提高预测精度

- 增加更多历史数据
- 使用更高分辨率的数据
- 添加更多物理特征
- 使用超参数优化

## 常见问题

### Q: 数据下载失败怎么办？

A: 检查以下几点：
1. CDS API密钥是否正确配置
2. 网络连接是否正常
3. 是否有足够的磁盘空间
4. CDS服务是否可用

### Q: 模型训练时间很长？

A: 可以：
1. 减少数据时间范围
2. 降低空间分辨率
3. 减少模型复杂度（N_ESTIMATORS, MAX_DEPTH）
4. 使用更小的数据集进行测试

### Q: 预测精度不高？

A: 尝试：
1. 增加更多特征工程
2. 使用更长时间的数据
3. 调整模型超参数
4. 尝试其他模型算法

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目！

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
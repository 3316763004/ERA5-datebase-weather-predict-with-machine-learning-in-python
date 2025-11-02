"""
ERA5天气预测系统 - 快速开始指南
这个脚本帮助用户快速测试系统功能
"""

import os
import sys

def print_banner():
    """打印横幅"""
    print("=" * 60)
    print("ERA5机器学习天气预测系统 - 快速开始")
    print("=" * 60)

def check_dependencies():
    """检查依赖包"""
    print("\n1. 检查依赖包...")
    
    required_packages = [
        'numpy', 'pandas', 'xarray', 'netcdf4', 
        'sklearn', 'matplotlib', 'seaborn', 
        'xgboost', 'cdsapi', 'cartopy', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'netcdf4':
                import netCDF4
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (未安装)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("  ✅ 所有依赖包都已安装")
        return True

def check_cds_api():
    """检查CDS API配置"""
    print("\n2. 检查CDS API配置...")
    
    # 检查环境变量
    cds_api_key = os.environ.get('CDS_API_KEY')
    
    # 检查配置文件
    config_files = [
        os.path.expanduser('~/.cdsapirc'),
        './.cdsapirc'
    ]
    
    config_found = False
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  ✅ 找到配置文件: {config_file}")
            config_found = True
            break
    
    if not config_found:
        print("  ❌ 未找到CDS API配置文件")
        print("  请在用户主目录创建 ~/.cdsapirc 文件")
        print("  文件内容:")
        print("    url: https://cds.climate.copernicus.eu/api/v2")
        print("    key: 你的UID:你的API密钥")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    print("\n3. 创建目录结构...")
    
    directories = ['data', 'models', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✅ 创建目录: {directory}")
        else:
            print(f"  ✅ 目录已存在: {directory}")

def run_basic_test():
    """运行基本测试"""
    print("\n4. 运行基本测试...")
    
    try:
        # 测试配置导入
        import config
        print("  ✅ 配置文件加载成功")
        
        # 测试类定义（不实例化）
        from data_downloader import ERA5DataDownloader
        from data_preprocessor import ERA5DataPreprocessor
        from ml_models import WeatherPredictor
        from visualization import WeatherVisualizer
        print("  ✅ 所有模块导入成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n5. 使用示例:")
    print("  完整流程:")
    print("    python main.py --mode full")
    print()
    print("  分步执行:")
    print("    python main.py --mode download      # 下载数据")
    print("    python main.py --mode preprocess    # 预处理数据")
    print("    python main.py --mode train         # 训练模型")
    print("    python main.py --mode predict       # 进行预测")
    print("    python main.py --mode summary       # 查看摘要")
    print()
    print("  运行示例:")
    print("    python example_usage.py")
    print()
    print("  注意: 首次运行需要下载数据，可能需要较长时间")

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 检查API配置
    api_ok = check_cds_api()
    
    # 创建目录
    create_directories()
    
    # 运行测试
    test_ok = run_basic_test()
    
    # 显示使用示例
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("快速开始检查完成!")
    
    if deps_ok and api_ok and test_ok:
        print("🎉 系统准备就绪，可以开始使用！")
        print("\n建议首先运行:")
        print("  python main.py --mode full")
    else:
        print("⚠️  请先解决上述问题后再使用系统")
        if not deps_ok:
            print("  - 安装依赖包: pip install -r requirements.txt")
        if not api_ok:
            print("  - 配置CDS API: 创建 ~/.cdsapirc 文件")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
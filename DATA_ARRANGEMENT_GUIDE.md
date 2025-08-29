# 数据安排指南 (Data Arrangement Guide)

## 📊 数据集概览

### 雄安新区高光谱数据集规格
- **光谱范围**: 400-1000nm (250个波段)
- **空间分辨率**: 0.5m
- **影像大小**: 3750×1580像元
- **地物类别**: 19种 (水稻茬、草地、榆树、白蜡、国槐、菜地、杨树、大豆、刺槐、水稻、水体、柳树、复叶槭槭、栾树、桃树、玉米、梨树、荷叶、建筑)
- **数据格式**: MATLAB .mat文件

## 🗂️ 当前数据目录结构

```
项目根目录/
├── dataset/                           # 数据集目录
│   ├── xiongan.mat                   # 主要高光谱影像数据 (2.3GB)
│   ├── xiongan_gt.mat                # 地物类别标注数据 (114KB)
│   ├── datainstruction.md            # 详细数据说明文档
│   └── 雄安新区高光谱数据集简介.pdf    # 数据集介绍
│
└── hyperspectral_reconstruction/      # 重建系统目录
    ├── config/default_config.json    # 配置文件(已配置正确路径)
    ├── src/                          # 源代码
    ├── main.py                       # 主执行脚本
    └── results/                      # 结果输出目录
```

## ⚙️ 数据配置状态

### ✅ 已正确配置的路径
```json
{
  "data_config": {
    "data_source": "synthetic",           # 当前设置为合成数据
    "xiong_an_data_path": "../dataset/xiongan.mat",      # ✅ 正确
    "xiong_an_gt_path": "../dataset/xiongan_gt.mat",     # ✅ 正确
    "num_samples": 2000,                  # 样本数量
    "sampling_method": "random"           # 采样方法
  }
}
```

## 🚀 使用数据的三种方式

### 1. 使用雄安真实数据集
```bash
cd hyperspectral_reconstruction
python main.py --data-source xiong_an
```

### 2. 使用合成数据 (默认)
```bash
cd hyperspectral_reconstruction
python main.py --data-source synthetic
```

### 3. 自定义配置运行
```bash
cd hyperspectral_reconstruction
python main.py --config config/custom_config.json --data-source xiong_an --num-samples 1000
```

## 📋 数据加载流程详解

### Step 1: 数据检测与加载
系统会自动检测数据文件存在性：
```python
# 系统会检查以下文件
../dataset/xiongan.mat      # 主数据文件
../dataset/xiongan_gt.mat   # 标注文件(可选)
```

### Step 2: 数据预处理
- **格式转换**: MATLAB .mat → NumPy arrays
- **归一化**: MinMax归一化到[0,1]范围
- **坏波段移除**: 自动检测并移除噪声波段
- **样本提取**: 随机或均匀采样指定数量的像元

### Step 3: 波长对齐
- **雄安数据**: 400-1000nm, 250波段
- **探测器响应**: 自动生成15个高斯探测器覆盖相同范围
- **一致性检查**: 确保波长数组匹配

## 🔧 数据配置选项详解

### 数据源选择
```json
"data_source": "xiong_an"    # 使用雄安真实数据
"data_source": "synthetic"   # 使用合成数据
```

### 采样配置
```json
"num_samples": 2000,         # 提取样本数量
"sampling_method": "random"  # 采样方法: "random" 或 "uniform"
```

### 预处理配置
```json
"normalization": "minmax",        # 归一化方法: "minmax", "standard", "none"
"remove_bad_bands": true,         # 是否移除坏波段
"noise_threshold": 0.01           # 噪声阈值
```

## 📊 数据质量验证

### 自动验证检查
系统启动时会自动验证：
1. **文件存在性**: 检查数据文件是否存在
2. **数据完整性**: 验证数据格式和大小
3. **波长一致性**: 确保波长范围匹配
4. **样本有效性**: 检查提取的样本质量

### 验证命令
```bash
# 快速验证数据加载
cd hyperspectral_reconstruction
python simple_test.py

# 详细验证包括数据质量
python test_system.py
```

## 🎯 针对不同研究目标的数据配置建议

### 1. 算法性能测试
```json
{
  "data_source": "xiong_an",
  "num_samples": 5000,
  "sampling_method": "random",
  "noise_level": 0.01
}
```

### 2. 快速原型验证
```json
{
  "data_source": "synthetic",
  "num_samples": 500,
  "synthetic_height": 50,
  "synthetic_width": 50
}
```

### 3. 探测器设计优化
```json
{
  "data_source": "xiong_an",
  "num_samples": 10000,
  "detector_config": {
    "num_detectors": 15,
    "detector_fwhm": 50.0
  }
}
```

## 📈 性能优化建议

### 内存优化
- **大数据集**: 减少 `num_samples` 到 1000-2000
- **小内存**: 使用 `synthetic` 数据源
- **批处理**: 分批处理大型数据集

### 速度优化
- **快速测试**: 使用合成数据
- **并行处理**: 启用多核CPU处理
- **缓存机制**: 预处理结果可保存重用

## ⚠️ 常见问题与解决方案

### 问题1: 数据文件未找到
```bash
Error: Could not find ../dataset/xiongan.mat
```
**解决方案**: 检查数据文件路径，确保在正确目录运行

### 问题2: 内存不足
```bash
Error: Cannot allocate memory
```
**解决方案**: 减少 `num_samples` 或使用合成数据

### 问题3: 波长不匹配
```bash
Error: Expected 250 wavelengths, got XXX
```
**解决方案**: 检查数据格式，确保波段数正确

## 🔍 数据探索命令

### 查看数据基本信息
```bash
cd hyperspectral_reconstruction
python -c "
from src.data_utils import HyperspectralDataLoader
loader = HyperspectralDataLoader()
data, gt = loader.load_xiong_an_data('../dataset/xiongan.mat', '../dataset/xiongan_gt.mat')
print(f'数据形状: {data.shape}')
print(f'标注形状: {gt.shape}')
print(f'数据范围: [{data.min():.3f}, {data.max():.3f}]')
"
```

### 生成数据摘要报告
```bash
python main.py --data-source xiong_an --num-samples 100 --quiet
# 查看 results/ 目录中的 experiment_results.json
```

## 📚 相关文档

- **详细数据说明**: `dataset/datainstruction.md`
- **系统文档**: `README.md`
- **配置指南**: `config/default_config.json`
- **API文档**: `src/data_utils.py` 中的函数文档

---

**💡 提示**: 首次使用建议先运行 `python simple_test.py` 验证系统配置正确，然后根据研究需求选择合适的数据配置方案。
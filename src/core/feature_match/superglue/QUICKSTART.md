# SuperGlue C++ 快速入门指南

## 设计概述

本项目为 SuperGlue 提供了一个完整的 C++ LibTorch 封装，专为 GUI 集成设计，具有以下特点：

### 四类参数设计

```cpp
struct SuperGlueConfig {
    // ========== 基础参数 ==========
    std::string model_path;        // 模型路径
    float match_threshold;         // 匹配阈值 [0.0-1.0]
    
    // ========== 高级参数 ==========
    int batch_size;               // 批处理大小
    int sinkhorn_iterations;      // Sinkhorn迭代次数
    int max_keypoints;            // 最大关键点数量
    
    // ========== 系统参数 ==========
    bool use_cuda;                // 是否使用CUDA
    int cuda_device_id;           // CUDA设备ID
    int num_threads;              // CPU线程数
    
    // ========== 调试参数 ==========
    bool enable_csv_output;       // CSV输出开关
    std::string csv_output_path;  // CSV文件路径
    bool enable_visualization;    // 可视化开关
    std::string visualization_output_path; // 可视化路径
    bool verbose;                 // 详细日志开关
};
```

## 快速开始

### 第一步：导出模型

```bash
cd /path/to/SuperGluePretrainedNetwork-master

# 导出CPU版本（适合无GPU环境）
python3 superglue_cpp/export_torchscript.py --weights outdoor --device cpu

# 导出CUDA版本（适合GPU环境）
python3 superglue_cpp/export_torchscript.py --weights outdoor --device cuda

# 或导出所有变体
python3 superglue_cpp/export_torchscript.py --all
```

导出后会生成：
- `superglue_outdoor_cpu.pt` - CPU版本
- `superglue_outdoor_cuda.pt` - CUDA版本
- `superglue_indoor_cpu.pt` - Indoor CPU版本
- `superglue_indoor_cuda.pt` - Indoor CUDA版本

### 第二步：编译项目

```bash
cd superglue_cpp
mkdir build && cd build

# 设置LibTorch路径（修改为你的实际路径）
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# 编译
make -j4
```

### 第三步：基本使用

```cpp
#include "SuperGlueMatcher.h"

int main() {
    // 方式1: 最简单的使用
    SuperGlueMatcher matcher("superglue_outdoor_cpu.pt", false);
    
    // 方式2: 使用完整配置（推荐用于GUI）
    SuperGlueConfig config;
    config.model_path = "superglue_outdoor_cpu.pt";
    config.match_threshold = 0.2f;
    config.enable_csv_output = true;
    config.enable_visualization = true;
    config.verbose = true;
    
    SuperGlueMatcher matcher2(config);
    
    // 执行匹配
    MatchResult result = matcher.match(img0, img1, kpts0, kpts1);
    
    std::cout << "找到 " << result.num_matches << " 对匹配" << std::endl;
    
    return 0;
}
```

## GUI 集成示例

### 参数配置界面设计

```
┌─────────────────────────────────────────────┐
│         SuperGlue 匹配器设置                │
├─────────────────────────────────────────────┤
│ [基础参数]                                  │
│   模型路径: [superglue_outdoor_cpu.pt] [浏览]│
│   匹配阈值: [0.20] (0.0 - 1.0)             │
│                                             │
│ [高级参数]                                  │
│   批处理大小: [1]                           │
│   最大关键点: [2000] (-1表示无限制)        │
│   Sinkhorn迭代: [100]                      │
│                                             │
│ [系统参数]                                  │
│   ☑ 使用CUDA   设备ID: [0]                │
│   CPU线程数: [4] (-1表示自动)              │
│                                             │
│ [调试参数]                                  │
│   ☑ 输出CSV    路径: [matches.csv]         │
│   ☑ 输出可视化  路径: [matches.jpg]        │
│   ☑ 详细日志                               │
│                                             │
│            [应用设置]  [取消]               │
└─────────────────────────────────────────────┘
```

### Qt 集成代码示例

```cpp
// 在Qt GUI中的集成示例
class MatcherSettingsDialog : public QDialog {
    Q_OBJECT
    
private:
    SuperGlueConfig config_;
    SuperGlueMatcher* matcher_;
    
    // UI控件
    QLineEdit* modelPathEdit_;
    QDoubleSpinBox* thresholdSpinBox_;
    QSpinBox* batchSizeSpinBox_;
    QCheckBox* useCudaCheckBox_;
    QCheckBox* csvOutputCheckBox_;
    QCheckBox* visualizationCheckBox_;
    
public slots:
    void onApplyClicked() {
        // 从UI更新配置
        config_.model_path = modelPathEdit_->text().toStdString();
        config_.match_threshold = thresholdSpinBox_->value();
        config_.batch_size = batchSizeSpinBox_->value();
        config_.use_cuda = useCudaCheckBox_->isChecked();
        config_.enable_csv_output = csvOutputCheckBox_->isChecked();
        config_.enable_visualization = visualizationCheckBox_->isChecked();
        
        // 更新或重新创建匹配器
        if (matcher_) {
            delete matcher_;
        }
        matcher_ = new SuperGlueMatcher(config_);
        
        QMessageBox::information(this, "成功", "配置已应用");
    }
};
```

## 调试功能详解

### 1. CSV 输出

启用后，每次匹配会自动追加到CSV文件：

```csv
image_pair,keypoint0_idx,keypoint1_idx,matching_score,distance
pair_01,0,5,0.856,0.144
pair_01,1,12,0.923,0.077
pair_01,2,8,0.789,0.211
```

用途：
- 离线分析匹配质量
- 导入到Excel/Python进行统计分析
- 调试匹配算法参数

### 2. 可视化输出

启用后自动保存匹配可视化图像，包含：
- 两张图像并排显示
- 匹配点连接线
- 匹配数量统计

用途：
- 快速检查匹配效果
- 对比不同参数的影响
- 制作报告和演示

### 3. 详细日志

启用后在控制台输出：
```
[SuperGlue 16:40:35] 使用 CPU 设备
[SuperGlue 16:40:36] 加载模型: superglue_outdoor_cpu.pt
[SuperGlue 16:40:37] 模型加载成功
[SuperGlue 16:40:38] 匹配完成: 156 对
```

用途：
- 实时监控匹配进度
- 调试模型加载问题
- 性能分析

## 批处理使用

处理大量图像对时，使用批处理可以显著提高效率：

```cpp
SuperGlueConfig config;
config.model_path = "superglue_outdoor_cuda.pt";
config.batch_size = 8;  // 一次处理8对图像
config.use_cuda = true;

SuperGlueMatcher matcher(config);

std::vector<cv::Mat> images0, images1;
std::vector<KeypointData> kpts0_batch, kpts1_batch;

// 加载多组图像...
for (int i = 0; i < 100; ++i) {
    // 加载数据...
}

// 批量匹配
auto results = matcher.matchBatch(images0, images1, 
                                 kpts0_batch, kpts1_batch);

for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "批次 " << i << ": " 
              << results[i].num_matches << " 对匹配" << std::endl;
}
```

## 性能优化建议

### GPU 环境（推荐）

```cpp
SuperGlueConfig config;
config.model_path = "superglue_outdoor_cuda.pt";
config.use_cuda = true;
config.cuda_device_id = 0;
config.batch_size = 8;         // 批处理
config.max_keypoints = 1000;   // 限制关键点
```

预期性能：~50-100对/秒（取决于硬件）

### CPU 环境

```cpp
SuperGlueConfig config;
config.model_path = "superglue_outdoor_cpu.pt";
config.use_cuda = false;
config.num_threads = 8;        // 使用多线程
config.batch_size = 1;
config.max_keypoints = 500;    // 限制关键点提高速度
```

预期性能：~5-10对/秒（取决于CPU）

## 常见问题

### Q1: 如何选择 indoor 还是 outdoor 模型？

- **outdoor**: 适合室外场景、自然风景、建筑物
- **indoor**: 适合室内场景、纹理较少的环境

### Q2: 如何确定合适的 match_threshold？

- 默认 `0.2` 适合大多数场景
- 减小（如 `0.15`）：获得更多匹配，但可能增加误匹配
- 增大（如 `0.3`）：减少误匹配，但可能遗漏正确匹配

### Q3: 批处理大小如何设置？

- CPU: 建议 `1`（批处理对CPU提升有限）
- GPU: 根据显存，通常 `4-16`

### Q4: CSV文件何时写入？

每次调用 `match()` 后立即写入并 flush，确保数据不丢失。

### Q5: 如何在GUI中实时更新参数？

使用 `updateConfig()` 方法更新运行时参数，或使用 `set*()` 系列方法。

## 文件说明

| 文件 | 说明 |
|------|------|
| `SuperGlueMatcher.h` | 主类头文件，包含配置结构体 |
| `SuperGlueMatcher.cpp` | 主类实现 |
| `export_torchscript.py` | 模型导出脚本（支持CPU/CUDA）|
| `usage_examples.cpp` | 完整使用示例集合 |
| `example_usage.cpp` | 基本使用示例 |
| `test_superglue.cpp` | 单元测试 |
| `CMakeLists.txt` | CMake构建配置 |
| `README.md` | 详细文档 |
| `QUICKSTART.md` | 本文件 |

## 下一步

1. 查看 `usage_examples.cpp` 了解各种使用场景
2. 阅读 `README.md` 获取完整API文档
3. 运行 `test_superglue` 验证安装
4. 在你的GUI项目中集成 `SuperGlueMatcher`

## 技术支持

如有问题，请检查：
1. LibTorch 版本是否与CUDA版本匹配
2. 模型文件是否正确导出
3. OpenCV 是否正确安装

祝你使用愉快！🚀

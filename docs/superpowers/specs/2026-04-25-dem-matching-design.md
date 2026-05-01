# PlaScan GUI 优化设计文档

**日期**: 2026-04-25  
**范围**: DEM 工作流 GUI 双路径 + 新增深度学习匹配算法 + 现有对话框问题修复

---

## 一、DEM 工作流 GUI 优化（CreateDemDialog）

### 现状

`CreateDemDialog` 只支持"已有点云 → DEM"一条路径，参数仅有输出目录、分辨率、数据类型。

### 目标

支持两条路径，用户通过 Tab 或 RadioButton 切换：

**路径 A：已有点云 → DEM**（现有功能，保持不变）
- 前置条件：已完成"创建点云"步骤
- 参数：输出目录、分辨率、数据类型

**路径 B：选择影像 → 自动 MVS → DEM**（新增）
- 前置条件：已完成 SfM（有相机参数）
- 用户选择 2 张或多张影像（QListWidget + checkbox）
- 自动串联：深度图估计 → 深度融合 → 点云 → DEM
- 参数：影像选择、输出目录、分辨率、数据类型
- 相机参数自动从项目中读取，不需要用户干预

### UI 结构

```
CreateDemDialog
├── QTabWidget
│   ├── Tab "从点云生成"  (路径 A，现有逻辑)
│   │   └── 参数表单（输出目录、分辨率、类型）
│   └── Tab "从影像生成"  (路径 B，新增)
│       ├── 说明标签（"需要已完成 SfM，相机参数自动读取"）
│       ├── 影像列表（QListWidget + checkbox，全选/清除按钮）
│       └── 参数表单（输出目录、分辨率、类型）
└── 底部按钮（运行、关闭）
```

### 信号设计

新增信号：
```cpp
void requestRunImagesToDem(const QStringList &selectedImages,
                           const QString &outputDir,
                           double demResolution,
                           const QString &demType);
```

原有信号 `requestRunStereoAndPoint2Dem` 保持不变（路径 A 继续使用）。

### 接入点

`ProjectTerrainProductsManager` 接收 `requestRunImagesToDem` 信号，串联调用：
1. `DepthMapEstimate`（对选中影像对）
2. `DepthFusion`
3. `DenseCloud`（可选，直接用融合结果）
4. `CreateDem`

---

## 二、新增深度学习特征提取算法（SuperPointDialog）

### 现状

`SuperPointDialog::m_algorithmCombo` 只有 SuperPoint / ORB / SIFT 三项。

### 新增算法

| 算法 | 模型文件 | 描述子维度 | 备注 |
|------|---------|-----------|------|
| DISK | `disk_cuda.pt` / `disk_cpu.pt` | 128 | 旋转不变性强 |
| ALIKED | `aliked_cuda.pt` / `aliked_cpu.pt` | 128 | 轻量，速度快 |

### 变更

1. `m_algorithmCombo` 新增两项：`"DISK"` / `"ALIKED"`
2. 选择 DISK/ALIKED 时，`m_descriptorDimSpin` 自动设为 128（SuperPoint 默认 256）
3. `onAlgorithmChanged` 槽（新增）：根据算法自动调整描述子维度默认值
4. `collectSettings()` 输出 `feature_algorithm` 字段值为 `"disk"` / `"aliked"`

### 现有问题修复（SuperPointDialog）

1. **`m_algorithmCombo` 切换后描述子维度不联动**：DISK/ALIKED 是 128 维，SuperPoint 是 256 维，切换时应自动更新 `m_descriptorDimSpin` 默认值
2. **`onResetDefaults` 弹出 MessageBox**：打断用户操作，改为静默重置（去掉 `QMessageBox::information`）
3. **`applySettings` 中 `output_dir` 逻辑**：当前只在 `m_outputLine` 为空时才设置，但恢复项目时应无条件覆盖（去掉空判断）

---

## 三、新增深度学习匹配算法（FeatureMatchingDialog）

### 新增算法

| 算法 | 类型 | 模型文件 | 备注 |
|------|------|---------|------|
| LoFTR | 无检测器（直接输入图像） | `loftr_outdoor.pt` / `loftr_indoor.pt` | 有 outdoor/indoor 区分 |
| RoMa | 无检测器（直接输入图像） | `roma_outdoor.pt` / `roma_indoor.pt` | 大旋转角效果最好 |

### UI 变更

`m_matchAlgorithmCombo` 新增两项：
- `"LoFTR"` → data `"loftr"`
- `"RoMa"` → data `"roma"`

`m_paramStack` 新增两页：

**Page 3: LoFTR**
- 模型类型（outdoor/indoor）
- 匹配阈值（默认 0.2）
- 提示标签：`"⚠ LoFTR 直接处理原始图像，不使用已有特征提取结果"`

**Page 4: RoMa**
- 模型类型（outdoor/indoor）
- 匹配阈值（默认 0.05）
- 最大关键点数（默认 10000）
- 提示标签：`"⚠ RoMa 直接处理原始图像，不使用已有特征提取结果"`

### 现有问题修复（FeatureMatchingDialog）

1. **`applySettings` 中 LightGlue 面板复用 SuperGlue 的 JSON key**：
   `m_lgMatchThresholdSpin` 读取的是 `"match_threshold"`，与 SuperGlue 共用同一 key，导致切换算法后参数互相覆盖。
   修复：LightGlue 专属 key 改为 `"lg_match_threshold"` / `"lg_batch_size"` / `"lg_input_width"` / `"lg_input_height"`

2. **`onAlgorithmChanged` 对传统算法统一跳到 Page 2**：
   新增 LoFTR/RoMa 后需要扩展分支，避免 LoFTR/RoMa 也跳到传统算法页

3. **`onResetDefaults` 未重置 LightGlue 专属控件**：已有，但需确认新算法也加入重置

4. **`m_featureList` 对 LoFTR/RoMa 应禁用**：这两个算法不使用预提取特征，选中特征文件无意义，应在切换到这两个算法时禁用特征列表和生成匹配对按钮，改为显示影像列表

---

## 四、LoFTR/RoMa 的影像输入问题

LoFTR/RoMa 不使用 `.sp` 特征文件，而是直接输入原始图像对。

**方案**：`FeatureMatchingDialog` 在选择 LoFTR/RoMa 时：
- 隐藏特征文件列表，显示影像文件列表（`m_imageList`，新增）
- 匹配对格式不变（`img1__img2`），但 `FeatureMatchRunner` 根据算法名走不同分支

`FeatureMatchRunner` 新增分支：
```cpp
if (algo == "loftr" || algo == "roma") {
    // 直接加载原始图像，不读取 .sp 文件
    // 调用 LoFTRMatcher / RoMaMatcher
}
```

---

## 五、新增 Matcher 类

### LoFTRMatcher

```
src/core/feature_match/loftr/
├── LoFTRMatcher.h
└── LoFTRMatcher.cpp
```

接口：
```cpp
class LoFTRMatcher : public IFeatureMatcher {
    // match(FeatureData, FeatureData) — 忽略描述子，仅用关键点坐标（或直接用图像）
    // matchImages(cv::Mat, cv::Mat) — 主要接口
};
```

由于 LoFTR 输入是图像而非特征，`IFeatureMatcher::match(FeatureData, FeatureData)` 接口不适用。
扩展方案：在 `IFeatureMatcher` 中新增可选虚函数：
```cpp
virtual MatchResult matchImages(const cv::Mat &img0, const cv::Mat &img1) {
    throw std::runtime_error("not supported");
}
virtual bool supportsRawImages() const { return false; }
```

### RoMaMatcher

结构同 LoFTRMatcher，位于 `src/core/feature_match/roma/`。

---

## 六、实现顺序

1. **修复现有 Dialog 问题**（无风险，先做）
2. **SuperPointDialog 新增 DISK/ALIKED**（仅 GUI，不涉及后端）
3. **FeatureMatchingDialog 新增 LoFTR/RoMa 页面**（仅 GUI）
4. **CreateDemDialog 双路径重构**（GUI + 信号接入）
5. **LoFTRMatcher / RoMaMatcher 后端实现**（需要模型文件）
6. **FeatureMatchRunner 新增 LoFTR/RoMa 分支**

步骤 1-4 可以立即实现，步骤 5-6 依赖模型文件，可以先做 stub 实现。

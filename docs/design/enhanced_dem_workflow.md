# 增强版 DEM 创建工作流程设计

## 1. 需求概述

当前 `CreateDemDialog` 只能基于已有的密集点云生成 DEM，用户需要手动完成前置步骤：
- 特征提取 (SuperPoint)
- 特征匹配 (SuperGlue/LightGlue)
- 三角化生成稀疏点云
- MVS 生成密集点云

**新需求**：
- 支持"傻瓜式"操作：仅需 2 张影像 + 2 个相机文件即可生成 DEM
- 支持手动指定中间数据，灵活控制流程起点
- 智能检测已有中间结果，避免重复计算

## 2. 设计方案

### 2.1 工作模式

#### 自动模式 (Automatic Mode)
- **输入**：2 张影像 + 2 个相机文件
- **流程**：自动执行完整流水线
  1. 特征提取 (SuperPoint)
  2. 特征匹配 (SuperGlue/LightGlue)
  3. 前方交会/三角化 (生成稀疏点云)
  4. MVS 深度估计 + 融合 (生成密集点云)
  5. DEM 生成
- **智能检测**：检测项目中是否已有中间结果，提示用户是否复用

#### 手动模式 (Manual Mode)
- **输入**：用户指定任意中间数据
- **选项**：
  - 使用已有特征点文件 (.pt)
  - 使用已有匹配文件 (.match)
  - 使用已有稀疏点云 (.xyz)
  - 使用已有密集点云 (.ply)
- **流程**：从指定步骤开始执行

### 2.2 UI 设计

```
┌─────────────────────────────────────────────────────┐
│  创建相对 DEM                                        │
├─────────────────────────────────────────────────────┤
│  ○ 自动模式  ● 手动模式                              │
├─────────────────────────────────────────────────────┤
│  [自动模式面板]                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │ 说明：仅需选择 2 张影像和对应相机文件，        │  │
│  │       系统将自动完成全部流水线步骤。          │  │
│  ├───────────────────────────────────────────────┤  │
│  │ 影像列表：                                    │  │
│  │  □ image1.tif                                 │  │
│  │  □ image2.tif                                 │  │
│  │  [浏览影像...]                                │  │
│  ├───────────────────────────────────────────────┤  │
│  │ 相机文件：                                    │  │
│  │  影像 1: [/path/to/cam1.tsai] [浏览...]       │  │
│  │  影像 2: [/path/to/cam2.tsai] [浏览...]       │  │
│  ├───────────────────────────────────────────────┤  │
│  │ 流水线状态：                                  │  │
│  │  ✓ 特征点: 已检测到 2 个文件                  │  │
│  │  ✓ 匹配: 已检测到 1 对                        │  │
│  │  ✗ 稀疏点云: 未找到                           │  │
│  │  ✗ 密集点云: 未找到                           │  │
│  │  [检测已有数据]                               │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  [手动模式面板]                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │ 说明：手动指定中间数据，从任意步骤开始。      │  │
│  ├───────────────────────────────────────────────┤  │
│  │ □ 使用已有特征点                              │  │
│  │   [/path/to/features/] [浏览...]              │  │
│  │ □ 使用已有匹配结果                            │  │
│  │   [/path/to/matches/] [浏览...]               │  │
│  │ □ 使用已有稀疏点云                            │  │
│  │   [/path/to/sparse.xyz] [浏览...]             │  │
│  │ ☑ 使用已有密集点云                            │  │
│  │   [/path/to/dense.ply] [浏览...]              │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ DEM 参数 ────────────────────────────────────┐  │
│  │ 输出目录: [assets/dem/] [浏览...]             │  │
│  │ DEM 分辨率: [0.0] (0 = 自动)                  │  │
│  │ 数据类型: [float32 ▼]                         │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│                          [运行]  [关闭]              │
└─────────────────────────────────────────────────────┘
```

### 2.3 流水线编排

#### 自动模式流程

```python
def runAutomaticPipeline(images, cameras, outputDir):
    # 1. 检测已有数据
    status = detectPipelineStatus(images)
    
    # 2. 特征提取
    if not status.hasFeatures:
        features = runSuperPoint(images)
    else:
        features = status.featuresPath
        if askUserReuseData("特征点"):
            # 复用
        else:
            features = runSuperPoint(images)
    
    # 3. 特征匹配
    if not status.hasMatches:
        matches = runSuperGlue(features)
    else:
        matches = status.matchesPath
        if askUserReuseData("匹配结果"):
            # 复用
        else:
            matches = runSuperGlue(features)
    
    # 4. 三角化
    if not status.hasSparseCloud:
        sparseCloud = runTriangulation(images, cameras, matches)
    else:
        sparseCloud = status.sparseCloudPath
        if askUserReuseData("稀疏点云"):
            # 复用
        else:
            sparseCloud = runTriangulation(images, cameras, matches)
    
    # 5. 密集点云
    if not status.hasDenseCloud:
        denseCloud = runMVS(images, cameras, sparseCloud)
    else:
        denseCloud = status.denseCloudPath
        if askUserReuseData("密集点云"):
            # 复用
        else:
            denseCloud = runMVS(images, cameras, sparseCloud)
    
    # 6. DEM 生成
    dem = runDemGeneration(denseCloud, outputDir)
    return dem
```

### 2.4 数据检测逻辑

```cpp
PipelineStepStatus detectPipelineStatus(const QStringList &images)
{
    PipelineStepStatus status;
    
    // 从项目元数据中查询
    QJsonObject meta = projectData->metadata();
    
    // 1. 检测特征点
    QJsonArray ipfindResults = meta["ipfind_results"].toArray();
    for (const auto &result : ipfindResults) {
        QString input = result["input"].toString();
        QString output = result["output"].toString();
        if (images.contains(input) && QFile::exists(output)) {
            status.hasFeatures = true;
            status.featuresPath = output;
        }
    }
    
    // 2. 检测匹配结果
    QJsonArray ipmatchResults = meta["ipmatch_results"].toArray();
    // ... 类似逻辑
    
    // 3. 检测稀疏点云
    QJsonArray atResults = meta["aerial_triangulation_results"].toArray();
    // ... 类似逻辑
    
    // 4. 检测密集点云
    QString denseCloud;
    if (resolveLatestDenseCloudPath(projectData, &denseCloud, nullptr)) {
        status.hasDenseCloud = true;
        status.denseCloudPath = denseCloud;
    }
    
    return status;
}
```

## 3. 实现计划

### 3.1 文件修改

1. **CreateDemDialog.h/cpp** - 增强对话框
   - 添加模式选择 (QRadioButton)
   - 添加自动模式 UI (影像列表、相机文件选择)
   - 添加手动模式 UI (中间数据选择)
   - 添加流水线状态显示
   - 添加智能检测逻辑

2. **ProjectTerrainProductsManager.h/cpp** - 流水线编排
   - 添加 `startFullDemPipelineAsync()` 方法
   - 实现流水线步骤编排
   - 实现中间结果检测和复用

3. **新增工具类** (可选)
   - `DemPipelineOrchestrator` - 流水线编排器
   - 负责步骤调度、进度跟踪、错误处理

### 3.2 信号流

```
CreateDemDialog::onRunClicked()
  ↓
  emit requestRunFullPipeline(images, outputDir, settings)
  ↓
ProjectManager::startFullDemPipelineAsync()
  ↓
ProjectTerrainProductsManager::startFullDemPipelineAsync()
  ↓
  1. detectPipelineStatus()
  2. runSuperPoint() [if needed]
  3. runSuperGlue() [if needed]
  4. runTriangulation() [if needed]
  5. runMVS() [if needed]
  6. runDemGeneration()
  ↓
  emit mvsProgressChanged() / atProgressChanged()
  ↓
MainWindow 状态栏更新
```

## 4. 用户体验

### 4.1 自动模式典型流程

1. 用户打开"创建 DEM"对话框
2. 选择"自动模式"
3. 从项目中选择 2 张影像
4. 指定 2 个相机文件路径
5. 点击"检测已有数据"按钮
6. 系统显示：
   ```
   ✓ 特征点: 已检测到 2 个文件 (复用)
   ✓ 匹配: 已检测到 1 对 (复用)
   ✗ 稀疏点云: 未找到 (需要生成)
   ✗ 密集点云: 未找到 (需要生成)
   ```
7. 点击"运行"
8. 系统自动执行：三角化 → MVS → DEM
9. 完成后显示结果路径

### 4.2 手动模式典型流程

1. 用户打开"创建 DEM"对话框
2. 选择"手动模式"
3. 勾选"使用已有密集点云"
4. 浏览选择密集点云文件
5. 设置 DEM 参数
6. 点击"运行"
7. 系统直接从密集点云生成 DEM

## 5. 技术要点

### 5.1 异步任务链

使用 Qt 信号槽机制串联异步任务：

```cpp
connect(superPointRunner, &SuperPointRunner::finished,
        this, [this]() {
    // SuperPoint 完成，启动 SuperGlue
    startSuperGlue();
});

connect(superGlueRunner, &SuperGlueRunner::finished,
        this, [this]() {
    // SuperGlue 完成，启动三角化
    startTriangulation();
});

// ... 依此类推
```

### 5.2 进度跟踪

```cpp
struct PipelineProgress {
    int totalSteps = 5;
    int currentStep = 0;
    QString currentStepName;
    int stepProgress = 0;  // 0-100
};

void updateProgress(int step, const QString &name, int progress) {
    m_progress.currentStep = step;
    m_progress.currentStepName = name;
    m_progress.stepProgress = progress;
    
    int overallProgress = (step * 100 + progress) / m_progress.totalSteps;
    emit pipelineProgressChanged(name, overallProgress);
}
```

### 5.3 错误处理

每个步骤失败时：
1. 停止流水线
2. 显示错误信息
3. 保存已完成步骤的中间结果
4. 允许用户从失败点重新开始

## 6. 测试计划

1. **自动模式测试**
   - 从零开始：2 张新影像 → DEM
   - 部分复用：已有特征点 → DEM
   - 全部复用：已有密集点云 → DEM

2. **手动模式测试**
   - 指定密集点云 → DEM
   - 指定稀疏点云 → 密集点云 → DEM
   - 指定匹配结果 → 稀疏点云 → 密集点云 → DEM

3. **边界情况测试**
   - 影像数量不是 2 张
   - 相机文件缺失或格式错误
   - 中间数据路径无效
   - 流水线中途取消

## 7. 后续优化

1. **参数预设**：为每个步骤提供"快速/标准/精细"预设
2. **批处理**：支持多对立体像对批量生成 DEM
3. **可视化**：实时显示中间结果（特征点、匹配、点云）
4. **配置保存**：记忆用户的流水线配置

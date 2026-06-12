# 词汇树候选重叠对设计方案

## 背景

PlaScan 的稀疏重建流程目前包含：

1. 特征点提取
2. 创建连接点
3. 构建观测网络模型
4. 初始化相机位姿
5. 生成初始稀疏点云
6. 光束法平差优化

当影像数量较多时，创建连接点若采用全连接匹配会产生 `N * (N - 1) / 2` 个影像对，时间成本较高。词汇树检索可以基于已提取特征快速召回可能重叠的候选影像对，用于约束后续特征匹配。

该功能应定位为“候选重叠对生成”，不是严格几何重叠面积计算。严格几何重叠仍由现有 `OverlapAnalyzer` 基于相机、DEM 或固定高程计算。

## 菜单位置

新增菜单项：

```text
重建
  稀疏重建
    特征点提取
    获取重叠对...
    创建连接点
    构建观测网络模型...
    初始化相机位姿...
    生成初始稀疏点云...
    光束法平差优化...
    稀疏点云后处理...
```

菜单项名称使用 `获取重叠对...`，对话框标题使用 `获取重叠对`。

## 推荐流程

第一版采用：

```text
已提取特征 -> 构建项目级词汇树 -> BoW/TF-IDF 相似度检索 -> 候选对排序
           -> 可选几何验证 -> 保存候选对 -> 应用到创建连接点
```

默认启用几何验证。原因是行星、遥感、地形纹理中可能存在重复纹理，仅靠 BoW 相似度容易召回纹理相似但不重叠的影像对。

## 方案选择

### 方案 A：纯词汇树召回

只使用特征描述子构建词汇树并计算 BoW/TF-IDF 相似度。

优点：
- 不依赖相机参数。
- 速度快。
- 适合无 `.tsai` 或相机姿态不可信的项目。

缺点：
- 重复地貌或大面积纹理相似区域容易误召回。
- 输出是视觉相似候选对，不是几何重叠对。

### 方案 B：词汇树召回 + 几何验证

先用词汇树召回 Top-K 候选，再用已有特征做轻量匹配与 RANSAC 验证。

优点：
- 兼顾速度和可靠性。
- 能直接为“创建连接点”提供更可信的候选对。
- 不需要一开始调用完整匹配流程。

缺点：
- 比纯词汇树略慢。
- 需要实现轻量匹配与验证统计。

### 方案 C：几何 footprint + 词汇树混合

如果项目有可靠相机和 DEM，先用几何重叠生成候选；词汇树作为补充召回或交叉验证。

优点：
- 对有 `.tsai` 的行星影像更符合摄影测量逻辑。
- 可减少词汇树误召回。

缺点：
- 对相机、DEM、固定高程配置有依赖。
- 第一版实现范围更大。

第一版推荐采用方案 B，后续再把方案 C 作为增强模式。

## Dialog UI 设计

窗口：

- 类名：`VocabularyOverlapDialog`
- UI 文件：`VocabularyOverlapDialog.ui`
- 标题：`获取重叠对`
- 推荐尺寸：`1100 x 720`
- 布局：上方左右分栏，下方结果表，底部固定操作按钮

### 布局草图

```text
┌──────────────────────────── 获取重叠对 ────────────────────────────┐
│ ┌────────────── 输入影像 / 特征文件 ──────────────┐ ┌──────── 参数 ────────┐ │
│ │ [全选] [清除] [仅显示缺特征]                    │ │ 特征来源              │ │
│ │                                                │ │ 算法: DISK ▼          │ │
│ │ ☑ img_001.tif    .dsk  8192点                  │ │ 特征目录: assets/ip   │ │
│ │ ☑ img_002.tif    .dsk  8021点                  │ │ [自动检测] [浏览...]  │ │
│ │ ☑ img_003.tif    .dsk  7900点                  │ │                       │ │
│ │ ⚠ img_004.tif    缺少 .dsk                     │ │ 词汇树参数            │ │
│ │                                                │ │ 分支数: 10            │ │
│ │                                                │ │ 深度: 4               │ │
│ │                                                │ │ 每图采样描述子: 2000 │ │
│ │                                                │ │                       │ │
│ │                                                │ │ 候选对筛选            │ │
│ │                                                │ │ 每图 Top-K: 8         │ │
│ │                                                │ │ 最小相似度: 0.05      │ │
│ │                                                │ │ ☑ TF-IDF 权重         │ │
│ │                                                │ │ ☑ 几何验证            │ │
│ │                                                │ │ 最小内点数: 30        │ │
│ └────────────────────────────────────────────────┘ │                       │ │
│ ┌──────────────────── 候选重叠对结果 ───────────────────────────────┐ │
│ │ 影像A              影像B              BoW分数  内点数  状态        │ │
│ │ img_001.tif        img_002.tif        0.83     421     通过        │ │
│ │ img_002.tif        img_003.tif        0.76     388     通过        │ │
│ │ img_001.tif        img_006.tif        0.22     12      剔除        │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│ 摘要: 6张影像，15个全连接候选，保留9对，几何验证通过7对              │
│ [恢复默认] [导出 pairs.lis] [应用到创建连接点] [开始计算] [关闭]     │
└─────────────────────────────────────────────────────────────────────┘
```

### 左侧：输入影像 / 特征文件

控件：

- `QListWidget m_imageList`
- `QPushButton m_selectAllBtn`
- `QPushButton m_clearSelectionBtn`
- `QCheckBox m_showMissingOnlyCheck`

每个影像项显示：

```text
影像名    特征后缀    关键点数 / 状态
```

状态包括：

- `.dsk 8192点`
- `.alk 4096点`
- `.sp 2048点`
- `缺少 .dsk`
- `描述子维度异常`
- `特征文件无法读取`

### 右侧：参数区

右侧使用 `QScrollArea`，避免在小屏幕上底部按钮不可见。

分组一：特征来源

- `QComboBox m_featureAlgorithmCombo`
  - DISK
  - ALIKED
  - SuperPoint
  - SIFT
- `QLineEdit m_featureDirEdit`
- `QPushButton m_autoDetectFeatureDirBtn`
- `QPushButton m_browseFeatureDirBtn`

第一版优先支持 float 描述子：

- DISK
- ALIKED
- SuperPoint
- SIFT

ORB 属于二进制描述子，词汇树距离和量化方式不同，第一版不作为默认支持对象。

分组二：词汇树参数

- `QSpinBox m_branchFactorSpin`
  - 默认：10
  - 范围：2 到 64
- `QSpinBox m_treeDepthSpin`
  - 默认：4
  - 范围：1 到 8
- `QSpinBox m_samplePerImageSpin`
  - 默认：2000
  - 范围：100 到 20000
- `QSpinBox m_maxTrainingDescriptorsSpin`
  - 默认：200000
  - 范围：10000 到 2000000

分组三：候选对筛选

- `QSpinBox m_topKSpin`
  - 默认：8
  - 范围：1 到 100
- `QDoubleSpinBox m_minSimilaritySpin`
  - 默认：0.05
  - 范围：0.0 到 1.0
- `QCheckBox m_useTfidfCheck`
  - 默认：开启
- `QCheckBox m_mutualTopKCheck`
  - 默认：开启
  - 含义：只有 A 的 Top-K 包含 B 且 B 的 Top-K 包含 A 才保留

分组四：几何验证

- `QCheckBox m_enableGeometryCheck`
  - 默认：开启
- `QSpinBox m_minInliersSpin`
  - 默认：30
  - 范围：0 到 10000
- `QDoubleSpinBox m_ransacThresholdSpin`
  - 默认：2.0 像素
- `QComboBox m_geometryModelCombo`
  - 自动
  - Fundamental
  - Homography

分组五：输出

- `QLineEdit m_outputJsonEdit`
- `QLineEdit m_outputLisEdit`
- `QCheckBox m_applyToMatchingCheck`
  - 默认：开启
  - 含义：计算结束后可将通过的候选对写入“创建连接点”的候选对约束

### 下方：候选重叠对结果表

控件：

- `QTableWidget m_pairTable`
- `QLabel m_summaryLabel`

列：

1. 勾选
2. 影像 A
3. 影像 B
4. BoW 分数
5. 共享视觉词数
6. 几何内点数
7. 状态
8. 备注

状态包括：

- `通过`
- `BoW分数过低`
- `几何验证失败`
- `缺少特征`
- `读取失败`

### 底部按钮

- `QPushButton m_resetBtn`
- `QPushButton m_exportLisBtn`
- `QPushButton m_applyToMatchingBtn`
- `QPushButton m_runBtn`
- `QPushButton m_closeBtn`

按钮行为：

- `开始计算`：执行词汇树召回和可选几何验证。
- `应用到创建连接点`：把当前勾选候选对写入匹配对约束。
- `导出 pairs.lis`：导出普通文本配对文件。
- `恢复默认`：恢复推荐参数。
- `关闭`：关闭对话框。

## 数据结构

新增候选对结果结构：

```cpp
struct VocabularyOverlapPair
{
    QString image0;
    QString image1;
    double bowScore = 0.0;
    int sharedWordCount = 0;
    int geometricInliers = 0;
    bool accepted = false;
    QString rejectReason;
};
```

新增请求配置：

```cpp
struct VocabularyOverlapConfig
{
    QString featureAlgorithm;
    QString featureSuffix;
    QString featureDir;
    int branchFactor = 10;
    int treeDepth = 4;
    int samplePerImage = 2000;
    int maxTrainingDescriptors = 200000;
    int topK = 8;
    double minSimilarity = 0.05;
    bool useTfidf = true;
    bool mutualTopK = true;
    bool geometryCheck = true;
    int minInliers = 30;
    double ransacThreshold = 2.0;
};
```

## 输出文件

默认输出目录：

```text
assets/overlap/
```

输出文件：

```text
assets/overlap/vocab_overlap_pairs.json
assets/overlap/vocab_overlap_pairs.lis
assets/overlap/vocab_tree_cache.bin
```

`vocab_overlap_pairs.json` 示例：

```json
{
  "type": "vocabulary_overlap",
  "feature_algorithm": "disk",
  "feature_suffix": ".dsk",
  "branch_factor": 10,
  "tree_depth": 4,
  "top_k": 8,
  "min_similarity": 0.05,
  "geometry_check": true,
  "pair_count": 7,
  "pairs": [
    {
      "image0": "/path/img_001.tif",
      "image1": "/path/img_002.tif",
      "bow_score": 0.83,
      "shared_word_count": 521,
      "geometric_inliers": 421,
      "accepted": true,
      "reject_reason": ""
    }
  ]
}
```

## 与创建连接点的集成

对话框点击 `应用到创建连接点` 后，应将通过的候选对写入当前项目的特征匹配设置，复用现有 `generated_pairs` 机制。

后续用户打开 `创建连接点` 时：

- 默认加载这些候选对；
- 匹配对预览显示来自 `获取重叠对`；
- 用户仍可清空或重新生成全连接对。

这样可以避免改动匹配器核心逻辑，只让候选对生成来源更多。

## 与现有几何重叠工具的关系

现有 `OverlapAnalysisDialog` 和 `OverlapAnalyzer` 基于相机、DEM 或固定高程计算几何重叠。

新功能基于特征描述子计算视觉候选重叠。

两者定位不同：

- 几何重叠：有可靠相机参数时更可信。
- 词汇树重叠：相机缺失、不准或需要快速视觉召回时更有用。

第一版不合并两个对话框。后续可以在词汇树对话框中增加 `融合几何重叠结果` 选项。

## 错误处理

对话框应明确报告：

- 未打开项目。
- 没有可用影像。
- 所选算法没有对应特征文件。
- 特征文件无法读取。
- 描述子维度不一致。
- 可用于训练的描述子数量过少。
- 词汇树训练失败。
- 候选对为空。

错误信息需要包含路径、算法和后缀，例如：

```text
缺少 DISK 特征文件: /path/assets/ip/img_001.dsk
```

## 测试计划

单元测试：

- 词汇树节点分裂与叶节点分配。
- BoW 直方图归一化。
- TF-IDF 权重计算。
- Top-K 候选对生成。
- mutual Top-K 过滤。
- JSON / LIS 输出格式。

GUI 测试：

- 菜单中存在 `获取重叠对...`，位置在 `特征点提取` 后、`创建连接点` 前。
- Dialog 默认参数正确。
- 缺少特征文件时影像列表显示缺失状态。
- 结果表能显示候选对。
- `应用到创建连接点` 写入 `generated_pairs`。

集成测试：

- 使用小型三图数据集，构造可分辨描述子，验证候选对数量和排序。
- 使用已有 `.dsk` 或 `.alk` 特征文件，验证后续 `创建连接点` 只处理候选对。

## 非目标

第一版不做：

- 大规模持久化通用预训练词汇树。
- ORB 二进制描述子的专用词汇树。
- 复杂网络图可视化。
- 替代现有几何 `OverlapAnalyzer`。
- 自动执行完整特征匹配。

## 实施顺序

1. 新增核心 `VocabularyOverlapRetriever`，完成纯词汇树候选召回。
2. 新增 JSON/LIS 输出工具。
3. 新增 `VocabularyOverlapDialog.ui/.h/.cpp`。
4. 在 `MainMenu` 和 `MenuWorkflowController` 中接入菜单。
5. 将候选对应用到 `FeatureMatchingDialog` 的 `generated_pairs`。
6. 增加可选几何验证。
7. 补充测试和文档。

## 默认推荐

第一版默认配置：

```text
特征算法: 当前项目可用特征中优先 DISK
分支数: 10
深度: 4
每图采样描述子: 2000
最大训练描述子: 200000
每图 Top-K: 8
最小相似度: 0.05
TF-IDF: 开启
mutual Top-K: 开启
几何验证: 开启
最小内点数: 30
RANSAC 阈值: 2.0 像素
```


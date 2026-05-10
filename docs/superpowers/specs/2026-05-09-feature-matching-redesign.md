# 特征匹配系统重设计 — 设计规格

日期: 2026-05-09 | 状态: 已确认

## 背景

当前特征匹配系统存在三个核心问题：

1. **多特征类型调度缺失**：当多种提取器运行后（SuperPoint → .sp, DISK → .dsk, ALIKED → .alk），匹配系统只能使用 `findFeatureForImage()` 扫描到的第一个文件，用户无法选择用哪个特征进行匹配。

2. **端到端算法（LoFTR/RoMa）无法选取匹配对**：LoFTR/RoMa 直接在原始影像上工作，但 FeatureMatchingDialog 切换到影像列表后生成的 pair string 无法被 FeatureMatchRunner 解析，后者总是尝试查找 `.sp` 文件。

3. **算法后端缺失**：UI 列出 7 种算法，但 RoMa、BF-Hamming、BF-L2、FLANN 没有后端实现。RoMa 的 Python 脚本 CLI 与现有接口完全不兼容。

## 目标

- 用户能根据匹配算法选择对应的特征文件类型
- LoFTR/RoMa 能正确选取影像匹配对并执行
- 所有 7 种 UI 列出的匹配算法都有可工作的后端
- 端到端算法和基于特征的算法共享统一的配对和执行流程

## 设计决策

### 决策 1: 算法-特征兼容性约束

每种匹配算法有固定的兼容特征类型。用户在 UI 中选择算法后，只能在兼容的特征中选取：

| 匹配算法 | 兼容特征 | 说明 |
|---------|---------|------|
| SuperGlue | `.sp` (SuperPoint) | SuperGlue 仅支持 SuperPoint |
| LightGlue | `.sp` / `.dsk` / `.alk` | 支持多种特征 |
| LoFTR | *无* (端到端) | 直接读原始影像 |
| RoMa | *无* (端到端) | 直接读原始影像 |
| BF-Hamming | `.orb` | ORB 的 Hamming 距离 |
| BF-L2 | `.sift` | SIFT 的 L2 距离 |
| FLANN | `.sift` | SIFT 的 FLANN |

### 决策 2: 特征选择 UI

在 FeatureMatchingDialog 中新增 `QComboBox`（特征类型选择器），根据所选算法动态更新可选后缀列表，只列出已有对应文件的类型。端到端算法（LoFTR/RoMa）时隐藏。

### 决策 3: 配对数据扩展

每对 match pair 携带上下文信息（特征后缀），替代当前的纯文件名格式：

```
pair:  "IMG_0001__IMG_0002"
suffix: ".alk"          // 空字符串表示端到端（直接读影像）
```

FeatureMatchRunner 根据 suffix 查找对应的特征文件，而非调用 `findFeatureForImage()`。

### 决策 4: LoFTR/RoMa 配对方式

- 手动选对：用户在影像列表中勾选匹配对
- 可选自动生成：全排列按钮（N×(N-1)/2）或基于重叠度分析

端到端算法的 pair 标记 suffix 为空，runner 检测到后直接传入原始影像路径给 Python 脚本。

### 决策 5: 缺失后端补全

- **RoMa**: 封装 `match_roma.py` 为统一 CLI `-L img -R img -o out`，与 LoFTR 接口一致
- **BF/FLANN**: 在 FeatureMatchRunner 中用 OpenCV `cv::BFMatcher` / `cv::FlannBasedMatcher` 直接实现
- **FeatureMatchRunner**: 统一 4 条分支 —— SuperGlueAdapter / LightGlueAdapter / 端到端PythonAdapter / OpenCV传统

---

## 变更清单

### FeatureMatchingDialog 变更

| 变更 | 说明 |
|------|------|
| 新增 `QComboBox m_featureSuffixCombo` | 列出当前算法兼容且文件存在的特征后缀 |
| `onAlgorithmChanged()` 更新 | 切换算法时更新后缀列表；端到端时隐藏特征选择区 |
| 影像配对列表重构 | LoFTR/RoMa 时显示影像勾选列表 + "全排列生成"按钮 |
| `collectSettings()` 更新 | 输出中携带 `feature_suffix` 字段 |
| match pair 格式 | 从纯文件名改为 `base__base` + 单独传递 suffix map |

### FeatureMatchRunner 变更

| 变更 | 说明 |
|------|------|
| 根据 suffix 查找特征文件 | 替换 `findFeatureForImage()` 为 `featureFileForSuffix()` |
| 端到端分支 | suffix 为空时跳过特征文件加载，直接传影像路径给 Python 脚本 |
| BF/FLANN 实现 | 新增 OpenCV 传统匹配分支（cv::BFMatcher / cv::FlannBasedMatcher） |
| RoMa 支持 | 与 LoFTR 复用同一 PythonAdapter，调用封装后的 `match_roma.py` |

### Python 脚本变更

| 变更 | 说明 |
|------|------|
| `match_roma.py` 重构 | 统一 CLI 为 `-L img -R img -o out --scene outdoor/indoor` |
| `match_roma.py` 输出格式 | 统一为二进制 `.match` 格式（与 LoFTR/SuperGlue 一致） |

### 不涉及的模块

- SFMService — 其匹配逻辑单独处理，本次不改
- CLI 工具 (`feature_match_cli`) — MatcherFactory 已有 LoFTR/DISK/ALIKED 支持，本次不扩展
- 特征提取端 — 不做修改

---

## 算法调度流程（新）

```
用户打开 FeatureMatchingDialog
  ├─ 选择算法 (7种)
  │   ├─ 端到端 (LoFTR/RoMa)
  │   │   ├─ 隐藏特征选择器
  │   │   ├─ 显示影像列表 + 手动勾选
  │   │   └─ 可选: 全排列 / 重叠度筛选
  │   └─ 基于特征 (其余5种)
  │       ├─ 显示特征后缀下拉 (仅可用项)
  │       ├─ 显示特征文件列表
  │       └─ 生成配对 (携带 suffix)
  ├─ 生成 pairs + suffix map
  └─ 提交给 FeatureMatchRunner
      ├─ suffix 非空 → 读特征文件 → 匹配
      └─ suffix 为空 → 读影像路径 → Python 端到端脚本
```

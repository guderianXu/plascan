# 特征匹配系统重设计 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户在多特征提取结果中选择匹配所用的特征类型，端到端算法(LoFTR/RoMa)正确选取影像对并执行，补全全部7种匹配算法后端。

**Architecture:** 算法-特征兼容性映射表驱动 UI 和后端调度。配对流新增 suffix 上下文（空=端到端）。FeatureMatchRunner 新增 4 路分支：SuperGlue/LightGlue/Python端到端/OpenCV传统。

**Tech Stack:** C++17, Qt6, OpenCV, LibTorch, Python (LoFTR/RoMa 后端脚本)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/core/feature_match/AlgorithmCompat.h` | 算法-特征兼容性映射表 + 查询函数 |
| Create | `src/core/feature_match/tests/test_algorithm_compat.cpp` | 兼容性映射单元测试 |
| Modify | `src/gui/dialogs/FeatureMatchingDialog.h` | 新增 m_featureSuffixCombo, m_imageCheckList |
| Modify | `src/gui/dialogs/FeatureMatchingDialog.cpp` | 特征选择UI + 配对逻辑重构 |
| Modify | `src/core/pipeline/FeatureMatchRunner.h` | 接口新增 suffix 参数 |
| Modify | `src/core/pipeline/FeatureMatchRunner.cpp` | 4路分支 + suffix文件解析 + BF/FLANN |
| Modify | `scripts/match_roma.py` | CLI统一为 -L/-R/-o |
| Modify | `src/gui/main_window/MainWindow.cpp` | 信号连接适配新pair格式 |
| Modify | `src/gui/cmake/GuiSources.cmake` | 新增源文件 |

---

### Task 1: 算法-特征兼容性映射表

**Files:**
- Create: `src/core/feature_match/AlgorithmCompat.h`
- Create: `src/core/feature_match/tests/test_algorithm_compat.cpp`

- [ ] **Step 1: Write the failing test**

Create `src/core/feature_match/tests/test_algorithm_compat.cpp`:

```cpp
#include <gtest/gtest.h>
#include "AlgorithmCompat.h"

using namespace xjw::feature_match;

TEST(AlgorithmCompatTest, SuperGlueOnlySupportsSP)
{
    auto suffixes = compatibleFeatureSuffixes("superglue");
    ASSERT_EQ(suffixes.size(), 1);
    EXPECT_EQ(suffixes[0], ".sp");
}

TEST(AlgorithmCompatTest, LightGlueSupportsMultiple)
{
    auto suffixes = compatibleFeatureSuffixes("lightglue");
    EXPECT_GE(suffixes.size(), 3);
    EXPECT_TRUE(suffixes.contains(".sp"));
    EXPECT_TRUE(suffixes.contains(".dsk"));
    EXPECT_TRUE(suffixes.contains(".alk"));
}

TEST(AlgorithmCompatTest, EndToEndHasNoFeatures)
{
    for (const auto &algo : {"loftr", "roma"})
    {
        auto suffixes = compatibleFeatureSuffixes(algo);
        EXPECT_TRUE(suffixes.isEmpty()) << algo << " should have no feature deps";
    }
}

TEST(AlgorithmCompatTest, BfHammingOnlyOrb)
{
    auto suffixes = compatibleFeatureSuffixes("orb_bf_hamming");
    ASSERT_EQ(suffixes.size(), 1);
    EXPECT_EQ(suffixes[0], ".orb");
}

TEST(AlgorithmCompatTest, BfL2AndFlannOnlySift)
{
    for (const auto &algo : {"sift_bf_l2", "sift_flann"})
    {
        auto suffixes = compatibleFeatureSuffixes(algo);
        ASSERT_EQ(suffixes.size(), 1) << algo;
        EXPECT_EQ(suffixes[0], ".sift") << algo;
    }
}

TEST(AlgorithmCompatTest, IsEndToEnd)
{
    EXPECT_TRUE(isEndToEndAlgorithm("loftr"));
    EXPECT_TRUE(isEndToEndAlgorithm("roma"));
    EXPECT_FALSE(isEndToEndAlgorithm("superglue"));
    EXPECT_FALSE(isEndToEndAlgorithm("lightglue"));
    EXPECT_FALSE(isEndToEndAlgorithm("orb_bf_hamming"));
}

TEST(AlgorithmCompatTest, AlgorithmDisplayName)
{
    EXPECT_FALSE(algorithmDisplayName("superglue").isEmpty());
    EXPECT_FALSE(algorithmDisplayName("loftr").isEmpty());
    EXPECT_FALSE(algorithmDisplayName("orb_bf_hamming").isEmpty());
}
```

- [ ] **Step 2: Verify tests fail**

```bash
cd /home/guderian/code/plascan/build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) --target test_algorithm_compat 2>&1 | tail -5
```
Expected: compile error — AlgorithmCompat.h not found.

- [ ] **Step 3: Implement AlgorithmCompat.h**

Create `src/core/feature_match/AlgorithmCompat.h`:

```cpp
#pragma once

#include <QString>
#include <QStringList>
#include <QMap>

namespace xjw {
namespace feature_match {

// 算法 → 兼容的特征文件后缀列表 (空列表 = 端到端, 不需要特征文件)
inline const QMap<QString, QStringList> &algorithmFeatureMap()
{
    static const QMap<QString, QStringList> map = {
        {"superglue",       {".sp"}},
        {"lightglue",       {".sp", ".dsk", ".alk"}},
        {"loftr",           {}},   // 端到端
        {"roma",            {}},   // 端到端
        {"orb_bf_hamming",  {".orb"}},
        {"sift_bf_l2",      {".sift"}},
        {"sift_flann",      {".sift"}},
    };
    return map;
}

inline QStringList compatibleFeatureSuffixes(const QString &algo)
{
    return algorithmFeatureMap().value(algo);
}

inline bool isEndToEndAlgorithm(const QString &algo)
{
    const auto &map = algorithmFeatureMap();
    return map.contains(algo) && map.value(algo).isEmpty();
}

inline QString algorithmDisplayName(const QString &algo)
{
    static const QMap<QString, QString> names = {
        {"superglue",       "SuperGlue"},
        {"lightglue",       "LightGlue"},
        {"loftr",           "LoFTR"},
        {"roma",            "RoMa"},
        {"orb_bf_hamming",  "BF-Hamming (ORB)"},
        {"sift_bf_l2",      "BF-L2 (SIFT)"},
        {"sift_flann",      "FLANN (SIFT)"},
    };
    return names.value(algo, algo);
}

} // namespace feature_match
} // namespace xjw
```

- [ ] **Step 4: Add test target to CMake**

In `src/core/feature_match/CMakeLists.txt`, add after existing entries:

```cmake
if(BUILD_TESTS)
  add_executable(test_algorithm_compat tests/test_algorithm_compat.cpp)
  target_link_libraries(test_algorithm_compat PRIVATE GTest::gtest GTest::gtest_main Qt6::Core)
  target_include_directories(test_algorithm_compat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME AlgorithmCompatTest COMMAND test_algorithm_compat)
endif()
```

- [ ] **Step 5: Build and run tests**

```bash
cd /home/guderian/code/plascan/build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) --target test_algorithm_compat && ./src/core/feature_match/test_algorithm_compat
```
Expected: 7/7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/feature_match/AlgorithmCompat.h \
        src/core/feature_match/tests/test_algorithm_compat.cpp \
        src/core/feature_match/CMakeLists.txt
git commit -m "feat: add algorithm-feature compatibility map"
```

---

### Task 2: FeatureMatchingDialog — 特征后缀选择器

**Files:**
- Modify: `src/gui/dialogs/FeatureMatchingDialog.h`
- Modify: `src/gui/dialogs/FeatureMatchingDialog.cpp`

- [ ] **Step 1: 读取现有头文件确认需要添加的成员**

Read `src/gui/dialogs/FeatureMatchingDialog.h` to find the existing member variables section.

- [ ] **Step 2: 在头文件中新增成员**

Add after existing combo box declarations:

```cpp
    QComboBox*      m_featureSuffixCombo{nullptr};  // 特征文件后缀选择
    QLabel*         m_featureSuffixLabel{nullptr};
```

Add new method declaration in public section:

```cpp
    void setAvailableFeatureSuffixes(const QStringList &suffixes);
    QString selectedFeatureSuffix() const;
```

Add new slot in private slots:

```cpp
    void onAlgorithmOrFeatureChanged();
```

- [ ] **Step 3: 在 cpp 的 setupUi 中添加后缀选择器**

In `FeatureMatchingDialog.cpp`, find the section where algorithm combo is laid out (around line 136-148). ADD after the algorithm row:

```cpp
    // 特征文件后缀选择（端到端算法时隐藏）
    m_featureSuffixLabel = new QLabel(tr("特征类型:"), inputGroup);
    m_featureSuffixCombo = new QComboBox(inputGroup);
    // ... add to layout
```

- [ ] **Step 4: 实现 setAvailableFeatureSuffixes 和联动逻辑**

```cpp
void FeatureMatchingDialog::setAvailableFeatureSuffixes(const QStringList &suffixes)
{
    m_featureSuffixCombo->blockSignals(true);
    m_featureSuffixCombo->clear();
    m_featureSuffixCombo->addItems(suffixes);
    m_featureSuffixCombo->blockSignals(false);
    
    bool visible = !suffixes.isEmpty();
    m_featureSuffixLabel->setVisible(visible);
    m_featureSuffixCombo->setVisible(visible);
}

QString FeatureMatchingDialog::selectedFeatureSuffix() const
{
    return m_featureSuffixCombo->currentText();
}

void FeatureMatchingDialog::onAlgorithmOrFeatureChanged()
{
    const QString algo = m_matchAlgorithmCombo->currentData().toString();
    const bool isE2E = xjw::feature_match::isEndToEndAlgorithm(algo);
    
    // 端到端: 隐藏特征选择, 显示影像列表
    m_inputStack->setCurrentIndex(isE2E ? 1 : 0);
    
    // 非端到端: 更新可用后缀列表
    if (!isE2E)
        setAvailableFeatureSuffixes(xjw::feature_match::compatibleFeatureSuffixes(algo));
    
    updatePreview();
}
```

- [ ] **Step 5: 更新 collectSettings 以包含 feature_suffix**

In `collectSettings()`, add:

```cpp
    config["feature_suffix"] = selectedFeatureSuffix();
```

- [ ] **Step 6: Build to verify compilation**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target plascan_gui 2>&1 | tail -5
```
Expected: compiles clean.

- [ ] **Step 7: Commit**

```bash
git add src/gui/dialogs/FeatureMatchingDialog.h src/gui/dialogs/FeatureMatchingDialog.cpp
git commit -m "feat: add feature suffix selector to FeatureMatchingDialog"
```

---

### Task 3: 配对格式扩展 + runner 接口变更

**Files:**
- Modify: `src/core/pipeline/FeatureMatchRunner.h`
- Modify: `src/core/pipeline/FeatureMatchRunner.cpp`

- [ ] **Step 1: 更新 FeatureMatchRunner 签名**

In `FeatureMatchRunner.h`, add new overload:

```cpp
    // 带特征后缀信息的匹配 (suffix为空=端到端)
    static void run(const QJsonObject &config,
                    const QStringList &imagePairs,
                    const QString &featureSuffix,
                    ProjectManager *projectManager,
                    std::atomic<bool> &cancelFlag,
                    std::atomic<int> &progressCount);
```

Keep existing overloads for backward compatibility (delegate to new one).

- [ ] **Step 2: 重构 dispatch 逻辑**

In `FeatureMatchRunner.cpp`, replace line 87-90 dispatch with:

```cpp
    const QString matchAlgorithm = config.value("match_algorithm").toString("superglue").trimmed().toLower();
    const QString featureSuffix = config.value("feature_suffix").toString();
    const bool isSuperGlueMatch = (matchAlgorithm == "superglue");
    const bool isLightGlueMatch = (matchAlgorithm == "lightglue");
    const bool isLoftrMatch     = (matchAlgorithm == "loftr");
    const bool isRomaMatch      = (matchAlgorithm == "roma");
    const bool isPythonE2E      = isLoftrMatch || isRomaMatch;
    const bool isBfFlann        = (matchAlgorithm == "orb_bf_hamming" ||
                                   matchAlgorithm == "sift_bf_l2" ||
                                   matchAlgorithm == "sift_flann");
```

- [ ] **Step 3: 修改特征文件解析为 suffix-based**

Replace the `resolveProjectFeaturePathFromToken` call (line ~358) with:

```cpp
    const QString sp0Path = featureSuffix.isEmpty()
        ? QString()  // 端到端不需要特征文件
        : xjw::gui::project::resolveFeaturePathBySuffix(plascanPath, currentMeta, imageToken0, featureSuffix);
    const QString sp1Path = featureSuffix.isEmpty()
        ? QString()
        : xjw::gui::project::resolveFeaturePathBySuffix(plascanPath, currentMeta, imageToken1, featureSuffix);
```

- [ ] **Step 4: 新增 resolveFeaturePathBySuffix 在 ProjectSupportUtils**

In `src/gui/project/support/ProjectSupportUtils.h`, add:

```cpp
QString resolveFeaturePathBySuffix(const QString &plascanPath, const QJsonObject &meta,
                                   const QString &token, const QString &suffix);
```

In `.cpp`, implement using `ProjectIO::featureFileForSuffix()`.

- [ ] **Step 5: Build and verify**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target plascan_gui 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add src/core/pipeline/FeatureMatchRunner.h src/core/pipeline/FeatureMatchRunner.cpp \
        src/gui/project/support/ProjectSupportUtils.h src/gui/project/support/ProjectSupportUtils.cpp
git commit -m "refactor: suffix-based feature file resolution + runner dispatch for all 7 algos"
```

---

### Task 4: Python 端到端分支 (LoFTR/RoMa) in FeatureMatchRunner

**Files:**
- Modify: `src/core/pipeline/FeatureMatchRunner.cpp`

- [ ] **Step 1: 实现 Python 端到端调用**

After the dispatch, add a new branch before the existing SuperGlue/LightGlue processing:

```cpp
    if (isPythonE2E)
    {
        LOG_INFO("端到端匹配(%s): 直接使用原始影像", qUtf8Printable(matchAlgorithm));
        
        // 对每对影像, 调用 Python 脚本
        for (const QString &pairStr : imagePairs)
        {
            // ... resolve img paths from pair tokens
            const QString imgPath0 = xjw::gui::project::resolveProjectImagePathFromToken(currentMeta, token0);
            const QString imgPath1 = xjw::gui::project::resolveProjectImagePathFromToken(currentMeta, token1);
            
            const QString scriptName = isLoftrMatch ? "run_loftr.py" : "match_roma.py";
            const QString scriptPath = QDir(QCoreApplication::applicationDirPath())
                .filePath("../scripts/" + scriptName);
            
            // 调用 Python: python3 script.py -L img0 -R img1 -o outPath
            QProcess proc;
            proc.start("python3", {scriptPath, "-L", imgPath0, "-R", imgPath1,
                                   "-o", matchOutPath, "--threshold",
                                   QString::number(sgConfig.match_threshold)});
            proc.waitForFinished(600000); // 10 min timeout
            // ... error handling
        }
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/core/pipeline/FeatureMatchRunner.cpp
git commit -m "feat: add Python end-to-end branch for LoFTR/RoMa in FeatureMatchRunner"
```

---

### Task 5: OpenCV BF/FLANN 传统匹配分支

**Files:**
- Modify: `src/core/pipeline/FeatureMatchRunner.cpp`

- [ ] **Step 1: 实现 BF/FLANN 分支**

Add after the isPythonE2E block:

```cpp
    else if (isBfFlann)
    {
        LOG_INFO("传统匹配(%s): 使用 OpenCV", qUtf8Printable(matchAlgorithm));
        
        const int normType = (matchAlgorithm == "orb_bf_hamming")
            ? cv::NORM_HAMMING : cv::NORM_L2;
        const bool useFlann = (matchAlgorithm == "sift_flann");
        
        for (const QString &pairStr : imagePairs)
        {
            // 读取特征文件
            FeatureOutput sp0, sp1;
            QString imgName;
            FeatureFileIO::read(sp0Path, imgName, sp0);
            FeatureFileIO::read(sp1Path, imgName, sp1);
            
            cv::Mat desc0 = FeatureData::fromFeatureOutput(sp0, "sift").toCvDescriptors(
                useFlann ? "sift" : "sift");
            cv::Mat desc1 = FeatureData::fromFeatureOutput(sp1, "sift").toCvDescriptors(
                useFlann ? "sift" : "sift");
            
            std::vector<cv::DMatch> matches;
            if (useFlann)
            {
                cv::FlannBasedMatcher matcher;
                matcher.match(desc0, desc1, matches);
            }
            else
            {
                cv::BFMatcher matcher(normType, true); // crossCheck
                matcher.match(desc0, desc1, matches);
            }
            
            // 保存为 .match 格式
            // ... convert cv::DMatch to match format and save
        }
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/core/pipeline/FeatureMatchRunner.cpp
git commit -m "feat: add OpenCV BF/FLANN traditional matching branch"
```

---

### Task 6: match_roma.py CLI 统一

**Files:**
- Modify: `scripts/match_roma.py`

- [ ] **Step 1: 重构 match_roma.py CLI**

在文件末尾添加新的 CLI 入口，兼容 `-L` / `-R` / `-o` 参数：

```python
def match_pair(imgL_path, imgR_path, out_path, scene="outdoor", threshold=0.8, max_kpts=2048):
    """单对影像匹配 (统一CLI入口)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = roma_outdoor(device=device) if scene == "outdoor" else roma_indoor(device=device)
    
    imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)
    
    tL = torch.from_numpy(imgL.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    tR = torch.from_numpy(imgR.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        warped, certainty = model.match(tL, tR)
    
    matches, certainty = model.sample(warped, certainty, num=max_kpts)
    kptsL = matches[..., :2].cpu().numpy()
    kptsR = matches[..., 2:].cpu().numpy()
    conf = certainty.cpu().numpy()
    
    mask = conf > threshold
    kptsL, kptsR, conf = kptsL[mask], kptsR[mask], conf[mask]
    
    # 写二进制 .match 格式
    with open(out_path, "wb") as f:
        f.write(struct.pack(">i", len(kptsL)))
        for i in range(len(kptsL)):
            f.write(struct.pack(">ffff", float(kptsL[i,0]), float(kptsL[i,1]),
                               float(kptsR[i,0]), float(kptsR[i,1])))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-L", required=True)
    ap.add_argument("-R", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("--scene", default="outdoor")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--max-keypoints", type=int, default=2048)
    args = ap.parse_args()
    match_pair(args.L, args.R, args.o, args.scene, args.threshold, args.max_keypoints)
```

- [ ] **Step 2: 保留旧的 --pairs 批量接口 (向后兼容)**

保持 `main_legacy()` 不动，只是新增 `match_pair()` 和新 CLI。

- [ ] **Step 3: Commit**

```bash
git add scripts/match_roma.py
git commit -m "refactor: add unified -L/-R/-o CLI to match_roma.py"
```

---

### Task 7: MainWindow 信号适配 + 端到端选对UI

**Files:**
- Modify: `src/gui/main_window/MainWindow.cpp`
- Modify: `src/gui/dialogs/FeatureMatchingDialog.cpp` (端到端选对)

- [ ] **Step 1: 端到端选对UI**

在 `FeatureMatchingDialog` 中, 当端到端算法选中时:
- 显示 checkbox-list (m_imageList 已经是 checkable)
- 添加 "全选" 和 "生成所有配对 (N×(N-1)/2)" 按钮
- `onGeneratePairs()` 中端到端分支: 收集勾选的影像, 生成所有 pair

修改 `onGeneratePairs()`:

```cpp
} else if (isRawImage) {
    QStringList selected;
    for (int i = 0; i < m_imageList->count(); ++i) {
        QListWidgetItem *item = m_imageList->item(i);
        if (item->checkState() == Qt::Checked)
            selected.append(QFileInfo(item->data(Qt::UserRole).toString()).completeBaseName());
    }
    m_currentPairs.clear();
    for (int i = 0; i < selected.size(); ++i)
        for (int j = i + 1; j < selected.size(); ++j)
            m_currentPairs.append(QString("%1__%2").arg(selected[i], selected[j]));
}
```

- [ ] **Step 2: 更新 MainWindow 的 runRequested 连接**

In `MainWindow.cpp` (line ~460), update the connection to pass feature suffix:

```cpp
    connect(dlg, &FeatureMatchingDialog::runRequested, this,
        [this, dlg](const QJsonObject &config, const QStringList &pairs)
    {
        // 注入 feature_suffix 到 config
        QJsonObject cfg = config;
        cfg["feature_suffix"] = dlg->selectedFeatureSuffix();
        
        QtConcurrent::run([cfg, pairs, pm = m_projectManager]() {
            std::atomic<bool> cancel{false};
            std::atomic<int> progress{0};
            FeatureMatchRunner::run(cfg, pairs, pm, cancel, progress);
        });
    });
```

- [ ] **Step 3: Build, fix errors, commit**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target plascan_gui 2>&1 | tail -5
git add src/gui/dialogs/FeatureMatchingDialog.cpp src/gui/main_window/MainWindow.cpp
git commit -m "feat: end-to-end pair selection UI + MainWindow suffix injection"
```

---

### Task 8: 完整构建 + 测试验证

- [ ] **Step 1: 全量构建**

```bash
cd /home/guderian/code/plascan/build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) 2>&1 | tail -5
```
Expected: 100% build, zero errors.

- [ ] **Step 2: 全量测试**

```bash
cd /home/guderian/code/plascan/build && ctest --output-on-failure 2>&1 | tail -10
```
Expected: all tests pass (including new AlgorithmCompat tests).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: finalize feature matching redesign — all tests pass"
git push origin main
```

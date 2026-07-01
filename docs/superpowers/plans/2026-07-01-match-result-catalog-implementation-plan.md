# Match Result Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make match visualization group results by image pair, default to the variant with the most geometric inliers, allow algorithm switching in the detail viewer, and make SfM match-cache usage explicit without changing the default SfM selection policy.

**Architecture:** Add a shared Qt/Core catalog layer that scans `assets/matches`, reads `.match` and `.match.json` sidecars, groups variants by canonical image pair, and computes best display variants. GUI consumes grouped results; SfM uses the same catalog for diagnostics while still loading only the configured `feature_algorithm + match_algorithm`.

**Tech Stack:** C++17, Qt6 Core/Widgets, existing `.match` SGMT format, JSON sidecars, CMake/GTest.

---

## File Structure

- Create `src/core/pipeline/MatchResultCatalog.h`: data structures and public catalog API.
- Create `src/core/pipeline/MatchResultCatalog.cpp`: file scanning, sidecar parsing, SGMT count reading, grouping, sorting, compatibility checks.
- Create `tests/test_match_result_catalog.cpp`: catalog unit tests with temporary `.match` and sidecar fixtures.
- Modify `tests/CMakeLists.txt`: add `test_match_result_catalog`.
- Modify `src/gui/cmake/GuiSources.cmake`: include `../core/pipeline/MatchResultCatalog.cpp` for GUI.
- Modify `src/gui/dialogs/MatchPairSelectorDialog.h/.cpp`: replace flat per-file list with grouped pair list and expose variants to the detail viewer.
- Modify `src/gui/dialogs/MatchViewerDialog.h/.cpp/.ui`: add match variant combo and reload selected match file.
- Modify `src/core/pipeline/SFMService.cpp`: add catalog-based diagnostics before SfM matching, without changing match selection.
- Modify `tests/test_gui_project_utils.cpp`: structural tests for grouped match viewer UI and SfM diagnostics strings.

---

### Task 1: Add MatchResultCatalog Unit Tests

**Files:**
- Create: `tests/test_match_result_catalog.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing catalog tests**

Create `tests/test_match_result_catalog.cpp`:

```cpp
#include "MatchResultCatalog.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QTextStream>

namespace
{

QString writeTextFile(const QString &path, const QString &text)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write(text.toUtf8());
    file.close();
    return path;
}

QString writeSgmtMatchFile(const QString &path,
                           const QString &image0,
                           const QString &image1,
                           int numMatches,
                           int numKp0,
                           int numKp1)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly));
    QDataStream out(&file);
    out.setVersion(QDataStream::Qt_5_15);
    file.write("SGMT", 4);
    out << quint32(1);
    const QByteArray img0Bytes = image0.toUtf8();
    const QByteArray img1Bytes = image1.toUtf8();
    out << quint32(img0Bytes.size());
    file.write(img0Bytes);
    out << quint32(img1Bytes.size());
    file.write(img1Bytes);
    out << qint32(numMatches) << qint32(numKp0) << qint32(numKp1);
    for (int i = 0; i < numMatches; ++i)
    {
        out << qint32(i % qMax(1, numKp0)) << qint32(i % qMax(1, numKp1)) << float(1.0f);
    }
    file.close();
    return path;
}

void writeSidecar(const QString &matchFile,
                  const QString &featureAlgorithm,
                  const QString &matchAlgorithm,
                  int totalMatches,
                  int inlierMatches,
                  const QString &feature0,
                  const QString &feature1)
{
    QJsonObject root;
    root.insert(QStringLiteral("feature_algorithm"), featureAlgorithm);
    root.insert(QStringLiteral("match_algorithm"), matchAlgorithm);
    root.insert(QStringLiteral("num_matches"), totalMatches);
    root.insert(QStringLiteral("num_inliers"), inlierMatches);
    root.insert(QStringLiteral("feature_format_version"), 2);
    root.insert(QStringLiteral("feature0_path"), feature0);
    root.insert(QStringLiteral("feature1_path"), feature1);

    QJsonArray idx0;
    QJsonArray idx1;
    for (int i = 0; i < totalMatches; ++i)
    {
        idx0.append(i);
        idx1.append(i);
    }
    root.insert(QStringLiteral("matched_indices0"), idx0);
    root.insert(QStringLiteral("matched_indices1"), idx1);

    QFile file(matchFile + QStringLiteral(".json"));
    EXPECT_TRUE(file.open(QIODevice::WriteOnly));
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
}

} // namespace

TEST(MatchResultCatalogTest, GroupsMultipleAlgorithmsByImagePairAndSelectsMostInliers)
{
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    const QString projectRoot = temp.path();
    const QString plascanPath = QDir(projectRoot).filePath(QStringLiteral("case.plascan"));
    writeTextFile(plascanPath, QStringLiteral("{}"));
    QDir().mkpath(QDir(projectRoot).filePath(QStringLiteral("assets/matches")));

    const QString imageA = QDir(projectRoot).filePath(QStringLiteral("Images/A.JPG"));
    const QString imageB = QDir(projectRoot).filePath(QStringLiteral("Images/B.JPG"));
    QDir().mkpath(QFileInfo(imageA).absolutePath());
    writeTextFile(imageA, QStringLiteral("a"));
    writeTextFile(imageB, QStringLiteral("b"));

    const QString featureA = QDir(projectRoot).filePath(QStringLiteral("assets/ip/A.sift"));
    const QString featureB = QDir(projectRoot).filePath(QStringLiteral("assets/ip/B.sift"));
    QDir().mkpath(QFileInfo(featureA).absolutePath());
    writeTextFile(featureA, QStringLiteral("fa"));
    writeTextFile(featureB, QStringLiteral("fb"));

    const QString matchDir = QDir(projectRoot).filePath(QStringLiteral("assets/matches"));
    const QString weak = writeSgmtMatchFile(QDir(matchDir).filePath(QStringLiteral("A__B_lightglue.match")),
                                            imageA, imageB, 200, 400, 400);
    writeSidecar(weak, QStringLiteral("sift"), QStringLiteral("lightglue"), 200, 50, featureA, featureB);

    const QString strong = writeSgmtMatchFile(QDir(matchDir).filePath(QStringLiteral("A__B_sift_bf_l2.match")),
                                              imageA, imageB, 160, 400, 400);
    writeSidecar(strong, QStringLiteral("sift"), QStringLiteral("sift_bf_l2"), 160, 120, featureA, featureB);

    xjw::pipeline::MatchResultCatalogConfig cfg;
    cfg.plascanPath = plascanPath;
    cfg.images = QStringList{imageA, imageB};
    cfg.currentFeatureAlgorithm = QStringLiteral("sift");
    cfg.currentMatchAlgorithm = QStringLiteral("sift_bf_l2");
    cfg.currentFeaturePaths.insert(QDir::cleanPath(imageA), QDir::cleanPath(featureA));
    cfg.currentFeaturePaths.insert(QDir::cleanPath(imageB), QDir::cleanPath(featureB));

    const xjw::pipeline::MatchResultCatalog catalog = xjw::pipeline::MatchResultCatalog::scan(cfg);
    ASSERT_EQ(catalog.groups.size(), 1);
    const xjw::pipeline::MatchPairGroup &group = catalog.groups.first();
    EXPECT_EQ(group.variants.size(), 2);
    ASSERT_TRUE(group.bestVariantIndex >= 0);
    EXPECT_EQ(group.variants[group.bestVariantIndex].matchAlgorithm, QStringLiteral("sift_bf_l2"));
    EXPECT_EQ(group.variants[group.bestVariantIndex].inlierMatches, 120);
    EXPECT_TRUE(group.variants[group.bestVariantIndex].compatibleWithCurrentSfm);
}

TEST(MatchResultCatalogTest, MarksMissingSidecarAsVisibleButNotSfmCompatible)
{
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    const QString projectRoot = temp.path();
    const QString plascanPath = QDir(projectRoot).filePath(QStringLiteral("case.plascan"));
    writeTextFile(plascanPath, QStringLiteral("{}"));
    QDir().mkpath(QDir(projectRoot).filePath(QStringLiteral("assets/matches")));

    const QString imageA = QDir(projectRoot).filePath(QStringLiteral("Images/A.JPG"));
    const QString imageB = QDir(projectRoot).filePath(QStringLiteral("Images/B.JPG"));
    QDir().mkpath(QFileInfo(imageA).absolutePath());
    writeTextFile(imageA, QStringLiteral("a"));
    writeTextFile(imageB, QStringLiteral("b"));

    writeSgmtMatchFile(QDir(projectRoot).filePath(QStringLiteral("assets/matches/A__B_sift_bf_l2.match")),
                       imageA, imageB, 40, 100, 100);

    xjw::pipeline::MatchResultCatalogConfig cfg;
    cfg.plascanPath = plascanPath;
    cfg.images = QStringList{imageA, imageB};
    cfg.currentFeatureAlgorithm = QStringLiteral("sift");
    cfg.currentMatchAlgorithm = QStringLiteral("sift_bf_l2");

    const xjw::pipeline::MatchResultCatalog catalog = xjw::pipeline::MatchResultCatalog::scan(cfg);
    ASSERT_EQ(catalog.groups.size(), 1);
    ASSERT_EQ(catalog.groups.first().variants.size(), 1);
    const xjw::pipeline::MatchVariant &variant = catalog.groups.first().variants.first();
    EXPECT_EQ(variant.totalMatches, 40);
    EXPECT_FALSE(variant.hasSidecar);
    EXPECT_FALSE(variant.compatibleWithCurrentSfm);
    EXPECT_FALSE(variant.failureReason.isEmpty());
}
```

- [ ] **Step 2: Register the failing test**

Add to `tests/CMakeLists.txt` near the other lightweight Qt Core tests:

```cmake
add_executable(test_match_result_catalog
    test_match_result_catalog.cpp
    ${CMAKE_SOURCE_DIR}/src/core/pipeline/MatchResultCatalog.cpp
)
target_link_libraries(test_match_result_catalog PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_match_result_catalog PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
    ${CMAKE_SOURCE_DIR}/src/gui/project/io
)
gtest_discover_tests(test_match_result_catalog)
```

- [ ] **Step 3: Run the test and verify it fails**

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_match_result_catalog -j 8
```

Expected: build fails because `MatchResultCatalog.h` and `MatchResultCatalog.cpp` do not exist.

---

### Task 2: Implement MatchResultCatalog

**Files:**
- Create: `src/core/pipeline/MatchResultCatalog.h`
- Create: `src/core/pipeline/MatchResultCatalog.cpp`
- Modify: `tests/CMakeLists.txt` if include paths need adjustment

- [ ] **Step 1: Add the public header**

Create `src/core/pipeline/MatchResultCatalog.h`:

```cpp
#pragma once

#include <QDateTime>
#include <QList>
#include <QMap>
#include <QString>
#include <QStringList>

namespace xjw::pipeline
{

struct MatchVariant
{
    QString imageA;
    QString imageB;
    QString imageAName;
    QString imageBName;
    QString featureAlgorithm;
    QString matchAlgorithm;
    QString matchFilePath;
    QString sidecarPath;
    QString feature0Path;
    QString feature1Path;
    int totalMatches = 0;
    int inlierMatches = -1;
    int outlierMatches = -1;
    bool hasSidecar = false;
    bool hasMatchedIndices = false;
    bool compatibleWithCurrentSfm = false;
    QString failureReason;
    QDateTime modifiedTime;
};

struct MatchPairGroup
{
    QString imageA;
    QString imageB;
    QString imageAName;
    QString imageBName;
    QString pairKey;
    QList<MatchVariant> variants;
    int bestVariantIndex = -1;
    bool overlapCandidate = false;
    double overlapScore = 0.0;
    QString overlapSource;
};

struct MatchResultCatalogConfig
{
    QString plascanPath;
    QStringList images;
    QString currentFeatureAlgorithm;
    QString currentMatchAlgorithm;
    QMap<QString, QString> currentFeaturePaths;
    bool includeOverlapCandidates = true;
};

struct MatchResultCatalogSummary
{
    int pairGroups = 0;
    int variants = 0;
    int compatibleVariants = 0;
    int missingSidecarVariants = 0;
    int missingMatchedIndexVariants = 0;
    int incompatibleAlgorithmVariants = 0;
};

class MatchResultCatalog
{
public:
    QList<MatchPairGroup> groups;
    MatchResultCatalogSummary summary;

    static MatchResultCatalog scan(const MatchResultCatalogConfig &config);
    static QString canonicalPairKey(const QString &imageA, const QString &imageB);
    static int readSgmtMatchCount(const QString &matchFilePath);

private:
    static void sortGroupVariants(MatchPairGroup *group);
};

} // namespace xjw::pipeline
```

- [ ] **Step 2: Add the implementation**

Create `src/core/pipeline/MatchResultCatalog.cpp` with these responsibilities:

```cpp
#include "MatchResultCatalog.h"

#include "ProjectIO.h"

#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>

#include <algorithm>
#include <cstring>

namespace xjw::pipeline
{

namespace
{

QString cleanPath(const QString &path)
{
    return QDir::cleanPath(path.trimmed());
}

QString normalizedAlgorithm(const QString &value)
{
    return value.trimmed().toLower();
}

QJsonObject readJsonObject(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return {};
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    return doc.isObject() ? doc.object() : QJsonObject{};
}

QString algorithmFromFileName(QString *partB)
{
    static const QStringList algorithms{
        QStringLiteral("orb_bf_hamming"),
        QStringLiteral("sift_bf_l2"),
        QStringLiteral("sift_flann"),
        QStringLiteral("superglue"),
        QStringLiteral("lightglue"),
        QStringLiteral("loftr"),
        QStringLiteral("roma")
    };
    for (const QString &algorithm : algorithms)
    {
        const QString suffix = QStringLiteral("_") + algorithm;
        if (partB->endsWith(suffix, Qt::CaseInsensitive))
        {
            partB->chop(suffix.size());
            return algorithm;
        }
    }
    return QString();
}

QString sidecarString(const QJsonObject &sidecar, const QString &key)
{
    QString value = sidecar.value(key).toString().trimmed();
    if (value.isEmpty())
    {
        value = sidecar.value(QStringLiteral("settings")).toObject().value(key).toString().trimmed();
    }
    return value;
}

int sidecarInt(const QJsonObject &sidecar, const QStringList &keys, int fallback)
{
    for (const QString &key : keys)
    {
        if (sidecar.contains(key))
        {
            return sidecar.value(key).toInt(fallback);
        }
    }
    return fallback;
}

bool pathsMatchEitherOrder(const QString &a0,
                           const QString &a1,
                           const QString &b0,
                           const QString &b1)
{
    if (a0.isEmpty() || a1.isEmpty() || b0.isEmpty() || b1.isEmpty())
    {
        return false;
    }
    const QString ca0 = cleanPath(a0);
    const QString ca1 = cleanPath(a1);
    const QString cb0 = cleanPath(b0);
    const QString cb1 = cleanPath(b1);
    return (ca0 == cb0 && ca1 == cb1) || (ca0 == cb1 && ca1 == cb0);
}

} // namespace

QString MatchResultCatalog::canonicalPairKey(const QString &imageA, const QString &imageB)
{
    QString a = cleanPath(imageA);
    QString b = cleanPath(imageB);
    if (a > b)
    {
        std::swap(a, b);
    }
    return a + QStringLiteral("\n") + b;
}

int MatchResultCatalog::readSgmtMatchCount(const QString &matchFilePath)
{
    QFile file(matchFilePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        return 0;
    }
    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);
    char magic[4] = {};
    if (in.readRawData(magic, 4) != 4 || std::strncmp(magic, "SGMT", 4) != 0)
    {
        return 0;
    }
    quint32 version = 0;
    in >> version;
    if (version != 1)
    {
        return 0;
    }
    quint32 img0Len = 0;
    quint32 img1Len = 0;
    in >> img0Len;
    file.seek(file.pos() + static_cast<qint64>(img0Len));
    in >> img1Len;
    file.seek(file.pos() + static_cast<qint64>(img1Len));
    qint32 numMatches = 0;
    qint32 numKp0 = 0;
    qint32 numKp1 = 0;
    in >> numMatches >> numKp0 >> numKp1;
    Q_UNUSED(numKp0);
    Q_UNUSED(numKp1);
    return qMax(0, static_cast<int>(numMatches));
}

void MatchResultCatalog::sortGroupVariants(MatchPairGroup *group)
{
    if (!group)
    {
        return;
    }
    std::sort(group->variants.begin(), group->variants.end(),
              [](const MatchVariant &left, const MatchVariant &right)
              {
                  const int leftInliers = left.inlierMatches >= 0 ? left.inlierMatches : -1;
                  const int rightInliers = right.inlierMatches >= 0 ? right.inlierMatches : -1;
                  if (leftInliers != rightInliers)
                  {
                      return leftInliers > rightInliers;
                  }
                  if (left.totalMatches != right.totalMatches)
                  {
                      return left.totalMatches > right.totalMatches;
                  }
                  return left.modifiedTime > right.modifiedTime;
              });
    group->bestVariantIndex = group->variants.isEmpty() ? -1 : 0;
}

MatchResultCatalog MatchResultCatalog::scan(const MatchResultCatalogConfig &config)
{
    MatchResultCatalog catalog;
    if (config.plascanPath.trimmed().isEmpty())
    {
        return catalog;
    }

    QMap<QString, QString> baseToPath;
    for (const QString &image : config.images)
    {
        const QString base = QFileInfo(image).completeBaseName();
        if (!baseToPath.contains(base))
        {
            baseToPath.insert(base, cleanPath(image));
        }
    }

    QMap<QString, MatchPairGroup> groupsByKey;
    const QString matchDirPath = QDir(ProjectIO::projectAssetsDir(config.plascanPath))
                                     .filePath(QStringLiteral("matches"));
    const QDir matchDir(matchDirPath);
    const QStringList matchFiles = matchDir.entryList(QStringList{QStringLiteral("*.match")},
                                                      QDir::Files,
                                                      QDir::Name);

    for (const QString &matchFileName : matchFiles)
    {
        const QString stem = QFileInfo(matchFileName).completeBaseName();
        const QStringList parts = stem.split(QStringLiteral("__"));
        if (parts.size() != 2)
        {
            continue;
        }
        QString partA = parts.at(0);
        QString partB = parts.at(1);
        const QString fileAlgorithm = algorithmFromFileName(&partB);
        const QString imageA = baseToPath.value(partA);
        const QString imageB = baseToPath.value(partB);
        if (imageA.isEmpty() || imageB.isEmpty())
        {
            continue;
        }

        const QString matchPath = cleanPath(matchDir.filePath(matchFileName));
        const QString sidecarPath = matchPath + QStringLiteral(".json");
        const QJsonObject sidecar = readJsonObject(sidecarPath);

        MatchVariant variant;
        variant.imageA = imageA;
        variant.imageB = imageB;
        variant.imageAName = QFileInfo(imageA).fileName();
        variant.imageBName = QFileInfo(imageB).fileName();
        variant.matchFilePath = matchPath;
        variant.sidecarPath = sidecarPath;
        variant.modifiedTime = QFileInfo(matchPath).lastModified();
        variant.totalMatches = readSgmtMatchCount(matchPath);
        variant.hasSidecar = !sidecar.isEmpty();
        variant.featureAlgorithm = normalizedAlgorithm(sidecarString(sidecar, QStringLiteral("feature_algorithm")));
        variant.matchAlgorithm = normalizedAlgorithm(sidecarString(sidecar, QStringLiteral("match_algorithm")));
        if (variant.matchAlgorithm.isEmpty())
        {
            variant.matchAlgorithm = fileAlgorithm;
        }
        variant.inlierMatches = sidecarInt(sidecar,
                                           QStringList{QStringLiteral("num_inliers"),
                                                       QStringLiteral("inlier_matches"),
                                                       QStringLiteral("valid_matches")},
                                           variant.hasSidecar ? variant.totalMatches : -1);
        variant.outlierMatches = variant.inlierMatches >= 0
            ? qMax(0, variant.totalMatches - variant.inlierMatches)
            : -1;
        variant.feature0Path = sidecarString(sidecar, QStringLiteral("feature0_path"));
        variant.feature1Path = sidecarString(sidecar, QStringLiteral("feature1_path"));
        if (variant.feature0Path.isEmpty())
        {
            variant.feature0Path = sidecarString(sidecar, QStringLiteral("sp0_path"));
        }
        if (variant.feature1Path.isEmpty())
        {
            variant.feature1Path = sidecarString(sidecar, QStringLiteral("sp1_path"));
        }

        const QJsonArray matched0 = sidecar.value(QStringLiteral("matched_indices0")).toArray();
        const QJsonArray matched1 = sidecar.value(QStringLiteral("matched_indices1")).toArray();
        variant.hasMatchedIndices = sidecar.value(QStringLiteral("feature_format_version")).toInt(0) >= 2
                                  && !matched0.isEmpty()
                                  && matched0.size() == matched1.size();

        const QString currentFeature = normalizedAlgorithm(config.currentFeatureAlgorithm);
        const QString currentMatch = normalizedAlgorithm(config.currentMatchAlgorithm);
        const bool algorithmMatches = !currentFeature.isEmpty()
                                   && !currentMatch.isEmpty()
                                   && variant.featureAlgorithm == currentFeature
                                   && variant.matchAlgorithm == currentMatch;
        const bool featurePathsMatch = config.currentFeaturePaths.isEmpty()
            || pathsMatchEitherOrder(variant.feature0Path,
                                     variant.feature1Path,
                                     config.currentFeaturePaths.value(cleanPath(imageA)),
                                     config.currentFeaturePaths.value(cleanPath(imageB)));

        variant.compatibleWithCurrentSfm = variant.hasSidecar
                                        && variant.hasMatchedIndices
                                        && algorithmMatches
                                        && featurePathsMatch;
        if (!variant.hasSidecar)
        {
            variant.failureReason = QStringLiteral("missing_sidecar");
            ++catalog.summary.missingSidecarVariants;
        }
        else if (!algorithmMatches)
        {
            variant.failureReason = QStringLiteral("algorithm_mismatch");
            ++catalog.summary.incompatibleAlgorithmVariants;
        }
        else if (!variant.hasMatchedIndices)
        {
            variant.failureReason = QStringLiteral("missing_matched_indices");
            ++catalog.summary.missingMatchedIndexVariants;
        }
        else if (!featurePathsMatch)
        {
            variant.failureReason = QStringLiteral("feature_path_mismatch");
        }

        const QString key = canonicalPairKey(imageA, imageB);
        MatchPairGroup &group = groupsByKey[key];
        if (group.pairKey.isEmpty())
        {
            group.pairKey = key;
            group.imageA = imageA;
            group.imageB = imageB;
            group.imageAName = QFileInfo(imageA).fileName();
            group.imageBName = QFileInfo(imageB).fileName();
        }
        group.variants.append(variant);
        ++catalog.summary.variants;
        if (variant.compatibleWithCurrentSfm)
        {
            ++catalog.summary.compatibleVariants;
        }
    }

    catalog.groups = groupsByKey.values();
    for (MatchPairGroup &group : catalog.groups)
    {
        sortGroupVariants(&group);
    }
    catalog.summary.pairGroups = catalog.groups.size();
    std::sort(catalog.groups.begin(), catalog.groups.end(),
              [](const MatchPairGroup &left, const MatchPairGroup &right)
              {
                  return left.imageBName < right.imageBName;
              });
    return catalog;
}

} // namespace xjw::pipeline
```

- [ ] **Step 3: Run catalog tests**

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_match_result_catalog -j 8
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R MatchResultCatalog
```

Expected: all `MatchResultCatalogTest.*` cases pass.

- [ ] **Step 4: Commit catalog core**

Run:

```powershell
git add src/core/pipeline/MatchResultCatalog.h src/core/pipeline/MatchResultCatalog.cpp tests/test_match_result_catalog.cpp tests/CMakeLists.txt
git commit -m "feat: add match result catalog"
```

---

### Task 3: Refactor MatchPairSelectorDialog to Use Grouped Results

**Files:**
- Modify: `src/gui/dialogs/MatchPairSelectorDialog.h`
- Modify: `src/gui/dialogs/MatchPairSelectorDialog.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add structural failing tests**

Append to `tests/test_gui_project_utils.cpp`:

```cpp
TEST(GuiSourceStructureTest, MatchPairSelectorUsesMatchResultCatalog)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchPairSelectorDialog.cpp"));
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"MatchResultCatalog.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("MatchResultCatalog::scan")));
    EXPECT_TRUE(source.contains(QStringLiteral("bestVariantIndex")));
    EXPECT_TRUE(source.contains(QStringLiteral("可用算法")));
}

TEST(GuiSourceStructureTest, MatchPairSelectorNoLongerShowsOneRowPerAlgorithmFile)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchPairSelectorDialog.cpp"));
    EXPECT_FALSE(source.contains(QStringLiteral("matches.append(info);")));
    EXPECT_TRUE(source.contains(QStringLiteral("appendGroupRow")));
}
```

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils -j 8
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "GuiSourceStructureTest.*MatchPairSelector"
```

Expected: tests fail because selector still uses flat `matches.append(info)`.

- [ ] **Step 2: Update header data model**

In `src/gui/dialogs/MatchPairSelectorDialog.h`, include catalog types and replace `_currentMatches` with grouped rows:

```cpp
#include "MatchResultCatalog.h"
```

Replace `MatchInfo` with:

```cpp
struct MatchRow
{
    xjw::pipeline::MatchPairGroup group;
    int bestVariantIndex = -1;
};
```

Replace:

```cpp
QList<MatchInfo> _currentMatches;
```

with:

```cpp
QList<MatchRow> _currentRows;
```

Add private helpers:

```cpp
QList<MatchRow> parseMatchGroupsForImage(const QString &imagePath);
void appendGroupRow(int row, const MatchRow &matchRow);
```

- [ ] **Step 3: Replace table setup**

In `setupTable()`, use six columns:

```cpp
_matchTable->setColumnCount(6);
_matchTable->setHorizontalHeaderLabels(QStringList{
    tr("图像"),
    tr("最佳算法"),
    tr("有效内点"),
    tr("总匹配"),
    tr("可用算法"),
    tr("状态")
});
```

- [ ] **Step 4: Implement grouped loading**

Replace `loadMatchPairsForImage()` body with logic that fills `_currentRows`:

```cpp
_matchTable->setRowCount(0);
_currentRows.clear();
_selectedMatchIndex = -1;
_viewDetailBtn->setEnabled(false);

_currentRows = parseMatchGroupsForImage(imagePath);
if (_currentRows.isEmpty())
{
    _statusLabel->setText(tr("该图像没有匹配数据"));
    return;
}

_matchTable->setRowCount(_currentRows.size());
for (int i = 0; i < _currentRows.size(); ++i)
{
    appendGroupRow(i, _currentRows[i]);
}

int variantCount = 0;
for (const MatchRow &row : _currentRows)
{
    variantCount += row.group.variants.size();
}
_statusLabel->setText(tr("找到 %1 个影像对，%2 个算法结果").arg(_currentRows.size()).arg(variantCount));
```

- [ ] **Step 5: Implement catalog-backed parsing**

Implement:

```cpp
QList<MatchPairSelectorDialog::MatchRow> MatchPairSelectorDialog::parseMatchGroupsForImage(const QString &imagePath)
{
    QList<MatchRow> rows;
    if (!_projectManager)
    {
        return rows;
    }

    xjw::pipeline::MatchResultCatalogConfig cfg;
    cfg.plascanPath = _projectManager->currentProjectPath();
    cfg.images = _allImages;
    cfg.includeOverlapCandidates = true;

    const QJsonObject meta = _projectManager->currentMeta();
    const QJsonObject sfmSettings = meta.value(QStringLiteral("sfm_settings")).toObject();
    cfg.currentFeatureAlgorithm = sfmSettings.value(QStringLiteral("feature_algorithm")).toString(QStringLiteral("disk"));
    cfg.currentMatchAlgorithm = sfmSettings.value(QStringLiteral("match_algorithm")).toString(QStringLiteral("lightglue"));

    const xjw::pipeline::MatchResultCatalog catalog = xjw::pipeline::MatchResultCatalog::scan(cfg);
    const QString current = QDir::cleanPath(imagePath);
    for (const xjw::pipeline::MatchPairGroup &group : catalog.groups)
    {
        if (QDir::cleanPath(group.imageA) != current && QDir::cleanPath(group.imageB) != current)
        {
            continue;
        }
        MatchRow row;
        row.group = group;
        row.bestVariantIndex = group.bestVariantIndex;
        rows.append(row);
    }
    return rows;
}
```

- [ ] **Step 6: Implement row rendering**

Add:

```cpp
void MatchPairSelectorDialog::appendGroupRow(int row, const MatchRow &matchRow)
{
    const xjw::pipeline::MatchPairGroup &group = matchRow.group;
    const QString otherPath = QDir::cleanPath(group.imageA) == QDir::cleanPath(_currentImage)
        ? group.imageB
        : group.imageA;
    const QString otherName = QFileInfo(otherPath).fileName();

    const xjw::pipeline::MatchVariant *best = nullptr;
    if (matchRow.bestVariantIndex >= 0 && matchRow.bestVariantIndex < group.variants.size())
    {
        best = &group.variants[matchRow.bestVariantIndex];
    }

    _matchTable->setItem(row, 0, new QTableWidgetItem(otherName));
    _matchTable->item(row, 0)->setToolTip(otherPath);
    _matchTable->setItem(row, 1, new QTableWidgetItem(best ? best->matchAlgorithm : tr("重叠候选")));
    _matchTable->setItem(row, 2, new QTableWidgetItem(best ? QString::number(best->inlierMatches) : QStringLiteral("-")));
    _matchTable->setItem(row, 3, new QTableWidgetItem(best ? QString::number(best->totalMatches) : QStringLiteral("-")));
    _matchTable->setItem(row, 4, new QTableWidgetItem(QString::number(group.variants.size())));
    QString status = best ? tr("已匹配") : tr("仅重叠候选");
    if (best && !best->failureReason.isEmpty())
    {
        status = best->failureReason;
    }
    _matchTable->setItem(row, 5, new QTableWidgetItem(status));
}
```

- [ ] **Step 7: Update selection and open detail**

Replace `_currentMatches` references with `_currentRows`. In `onViewDetailedMatch()`, build variant list and pass it to `MatchViewerDialog`:

```cpp
const MatchRow &row = _currentRows[_selectedMatchIndex];
if (row.group.variants.isEmpty())
{
    QMessageBox::information(this, tr("尚无匹配"), tr("该影像对目前只有重叠候选，没有可查看的匹配文件。"));
    return;
}
const xjw::pipeline::MatchVariant &best = row.group.variants[qMax(0, row.bestVariantIndex)];
const QString otherPath = QDir::cleanPath(row.group.imageA) == QDir::cleanPath(_currentImage)
    ? row.group.imageB
    : row.group.imageA;
auto *viewer = new MatchViewerDialog(_currentImage, otherPath, best.matchFilePath, this);
viewer->setMatchVariants(row.group.variants, qMax(0, row.bestVariantIndex));
```

- [ ] **Step 8: Add catalog source to GUI build**

In `src/gui/cmake/GuiSources.cmake`, add:

```cmake
../core/pipeline/MatchResultCatalog.cpp
```

near the other `../core/pipeline/*.cpp` entries.

- [ ] **Step 9: Run GUI structural tests**

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils -j 8
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "GuiSourceStructureTest.*MatchPairSelector"
```

Expected: selector structural tests pass.

- [ ] **Step 10: Commit selector integration**

Run:

```powershell
git add src/gui/dialogs/MatchPairSelectorDialog.h src/gui/dialogs/MatchPairSelectorDialog.cpp src/gui/cmake/GuiSources.cmake tests/test_gui_project_utils.cpp
git commit -m "feat: group match viewer pairs by image"
```

---

### Task 4: Add Algorithm Variant Switching to MatchViewerDialog

**Files:**
- Modify: `src/gui/dialogs/MatchViewerDialog.h`
- Modify: `src/gui/dialogs/MatchViewerDialog.cpp`
- Modify: `src/gui/dialogs/MatchViewerDialog.ui`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add failing structural tests**

Append to `tests/test_gui_project_utils.cpp`:

```cpp
TEST(GuiSourceStructureTest, MatchViewerSupportsAlgorithmVariantSwitching)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchViewerDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchViewerDialog.cpp"));
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchViewerDialog.ui"));
    EXPECT_TRUE(header.contains(QStringLiteral("setMatchVariants")));
    EXPECT_TRUE(source.contains(QStringLiteral("onMatchVariantChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("_variantCombo")));
    EXPECT_TRUE(ui.contains(QStringLiteral("m_variantCombo")));
}
```

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils -j 8
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "GuiSourceStructureTest.*MatchViewerSupports"
```

Expected: test fails because no variant combo exists.

- [ ] **Step 2: Update header**

In `MatchViewerDialog.h`, include catalog and add:

```cpp
#include "MatchResultCatalog.h"
```

Public method:

```cpp
void setMatchVariants(const QList<xjw::pipeline::MatchVariant> &variants, int selectedIndex);
```

Private slot:

```cpp
void onMatchVariantChanged(int index);
```

Private members:

```cpp
QComboBox *_variantCombo = nullptr;
QList<xjw::pipeline::MatchVariant> _variants;
```

- [ ] **Step 3: Add combo to UI**

In `MatchViewerDialog.ui`, add a label and combo to the sparse display controls group:

```xml
<widget class="QLabel" name="variantLabel">
 <property name="text">
  <string>匹配结果:</string>
 </property>
</widget>
<widget class="QComboBox" name="m_variantCombo">
 <property name="minimumWidth">
  <number>260</number>
 </property>
</widget>
```

- [ ] **Step 4: Wire combo in constructor**

In `MatchViewerDialog.cpp` constructor:

```cpp
_variantCombo = form.m_variantCombo;
_variantCombo->setVisible(false);
connect(_variantCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &MatchViewerDialog::onMatchVariantChanged);
```

- [ ] **Step 5: Implement setMatchVariants**

Add:

```cpp
void MatchViewerDialog::setMatchVariants(const QList<xjw::pipeline::MatchVariant> &variants, int selectedIndex)
{
    _variants = variants;
    if (!_variantCombo)
    {
        return;
    }
    QSignalBlocker blocker(_variantCombo);
    _variantCombo->clear();
    for (int i = 0; i < _variants.size(); ++i)
    {
        const auto &variant = _variants[i];
        const QString label = tr("%1 + %2：有效 %3 / 总 %4")
            .arg(variant.featureAlgorithm.isEmpty() ? QStringLiteral("?") : variant.featureAlgorithm.toUpper())
            .arg(variant.matchAlgorithm.isEmpty() ? QStringLiteral("?") : variant.matchAlgorithm)
            .arg(variant.inlierMatches >= 0 ? QString::number(variant.inlierMatches) : QStringLiteral("-"))
            .arg(variant.totalMatches);
        _variantCombo->addItem(label, variant.matchFilePath);
    }
    _variantCombo->setVisible(_variantCombo->count() > 1);
    if (!_variants.isEmpty())
    {
        const int safeIndex = qBound(0, selectedIndex, _variants.size() - 1);
        _variantCombo->setCurrentIndex(safeIndex);
        _matchFile = _variants[safeIndex].matchFilePath;
    }
}
```

- [ ] **Step 6: Implement reload on variant change**

Add:

```cpp
void MatchViewerDialog::onMatchVariantChanged(int index)
{
    if (index < 0 || index >= _variants.size() || !_viewer)
    {
        return;
    }
    const xjw::pipeline::MatchVariant &variant = _variants[index];
    if (variant.matchFilePath == _matchFile)
    {
        return;
    }
    _matchFile = variant.matchFilePath;
    _totalMatches = 0;
    const QString left = variant.imageA;
    const QString right = variant.imageB;
    _viewer->loadMatchPair(left, right, _matchFile);
    updateStatusBar();
}
```

- [ ] **Step 7: Run tests**

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils plascan_gui -j 8
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "GuiSourceStructureTest.*MatchViewerSupports"
```

Expected: tests pass and `plascan_gui` builds.

- [ ] **Step 8: Commit viewer switching**

Run:

```powershell
git add src/gui/dialogs/MatchViewerDialog.h src/gui/dialogs/MatchViewerDialog.cpp src/gui/dialogs/MatchViewerDialog.ui tests/test_gui_project_utils.cpp
git commit -m "feat: switch match algorithms in viewer"
```

---

### Task 5: Add SfM Match Catalog Diagnostics Without Changing Selection

**Files:**
- Modify: `src/core/pipeline/SFMService.cpp`
- Modify: `tests/test_sfm_service_default_algorithms.py`

- [ ] **Step 1: Add failing source-level tests**

Append to `tests/test_sfm_service_default_algorithms.py`:

```python
    def test_sfm_reports_match_catalog_but_keeps_configured_algorithm_policy(self):
        cli = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        self.assertIn("MatchResultCatalog::scan", cli)
        self.assertIn("匹配缓存目录诊断", cli)
        self.assertIn("SfM 默认仍按当前 feature_algorithm + match_algorithm 选择匹配", cli)
        self.assertNotIn("autoSelectBestVariantForSfm = true", cli)
```

Run:

```powershell
python tests\test_sfm_service_default_algorithms.py
```

Expected: test fails because SfMService does not call the catalog yet.

- [ ] **Step 2: Include catalog in SFMService**

Add near other includes in `SFMService.cpp`:

```cpp
#include "MatchResultCatalog.h"
```

- [ ] **Step 3: Build current feature path map**

After `featureFilePaths` is populated and before pair generation, add:

```cpp
QMap<QString, QString> currentFeaturePathByImage;
for (auto it = featureFilePaths.constBegin(); it != featureFilePaths.constEnd(); ++it)
{
    currentFeaturePathByImage.insert(normalizePath(idToPath.value(it.key())),
                                     normalizePath(it.value()));
}
```

- [ ] **Step 4: Log catalog diagnostics**

Before `logSfmMatchDiagnostics(QStringLiteral("预检查"), validIds, allPairs);`, add:

```cpp
xjw::pipeline::MatchResultCatalogConfig catalogConfig;
catalogConfig.plascanPath = opts.plascanPath;
catalogConfig.images = opts.images;
catalogConfig.currentFeatureAlgorithm = featureAlgorithm;
catalogConfig.currentMatchAlgorithm = matchAlgorithm;
catalogConfig.currentFeaturePaths = currentFeaturePathByImage;
catalogConfig.includeOverlapCandidates = false;
const xjw::pipeline::MatchResultCatalog matchCatalog =
    xjw::pipeline::MatchResultCatalog::scan(catalogConfig);
LOG_INFO(QStringLiteral(
    "  匹配缓存目录诊断: 影像对=%1, 算法结果=%2, 当前配置可用于 SfM=%3, 缺sidecar=%4, 缺V2索引=%5, 算法不兼容=%6")
    .arg(matchCatalog.summary.pairGroups)
    .arg(matchCatalog.summary.variants)
    .arg(matchCatalog.summary.compatibleVariants)
    .arg(matchCatalog.summary.missingSidecarVariants)
    .arg(matchCatalog.summary.missingMatchedIndexVariants)
    .arg(matchCatalog.summary.incompatibleAlgorithmVariants));
LOG_INFO(QStringLiteral(
    "  SfM 默认仍按当前 feature_algorithm + match_algorithm 选择匹配: %1 + %2")
    .arg(featureAlgorithm, matchAlgorithm));
```

- [ ] **Step 5: Add source to builds**

If `SFMService.cpp` is compiled into targets that do not already include `MatchResultCatalog.cpp`, add `src/core/pipeline/MatchResultCatalog.cpp` to those target source lists:

- `src/gui/cmake/GuiSources.cmake`
- `src/cli/CMakeLists.txt` target `reconstruct_pipeline_cli` if it lists pipeline source files explicitly
- `tests/CMakeLists.txt` targets that compile `SFMService.cpp` directly

- [ ] **Step 6: Run tests**

Run:

```powershell
python tests\test_sfm_service_default_algorithms.py
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target reconstruct_pipeline_cli plascan_gui -j 8
```

Expected: Python source test passes and both targets build.

- [ ] **Step 7: Commit SfM diagnostics**

Run:

```powershell
git add src/core/pipeline/SFMService.cpp src/cli/CMakeLists.txt src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt tests/test_sfm_service_default_algorithms.py
git commit -m "feat: report sfm match catalog diagnostics"
```

---

### Task 6: Validate on the 9-Image Aerial Test Set

**Files:**
- No source changes expected.

- [ ] **Step 1: Build all affected targets**

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target reconstruct_pipeline_cli plascan_gui test_match_result_catalog test_gui_project_utils test_sfm_params -j 8
```

Expected: exit code 0.

- [ ] **Step 2: Run targeted tests**

Run:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "MatchResultCatalog|GuiSourceStructureTest|IncrementalSfmParams"
python tests\test_sfm_service_default_algorithms.py
```

Expected: all targeted tests pass.

- [ ] **Step 3: Run 9-image SFM regression**

Run:

```powershell
. E:\code\plascan\scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch -Quiet
$out="E:\code\test\agisoft_aerial_gcps_small_9_match_catalog_verify"
& E:\code\plascan\build\windows-vcpkg-cuda-release\bin\reconstruct_pipeline_cli.exe `
  E:\code\plascan\testData\photogrammetry_benchmarks\agisoft_aerial_gcps_small\extracted\aerial_images_with_gcps\image_camera.lis `
  --output-dir $out `
  --device cuda `
  --sfm-feature-algorithm sift `
  --sfm-match-algorithm sift_bf_l2 `
  --sfm-guided-rematching `
  --quality 2 `
  --threads 8 `
  --feature-max-image-dim 0 `
  --stop-after-sfm `
  --force
```

Expected:

- `status=ok`
- `SFM 完成: 注册 9 张影像`
- report `sfm.success == true`
- report `sfm.registered_images == 9`
- sparse cloud exists at `$out\sparse\sfm_sparse.ply`

- [ ] **Step 4: Confirm git status excludes generated models**

Run:

```powershell
git status --short
```

Expected: only unrelated untracked local model artifacts may remain:

```text
?? resources/models/lightglue_sift_cpu.torchscript
?? resources/models/lightglue_sift_cuda.torchscript
```

No generated test output under `E:\code\test\...` is inside the repository.

---

### Task 7: Final Review and Push

**Files:**
- No new source files expected beyond previous tasks.

- [ ] **Step 1: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. LF/CRLF warnings are acceptable if no error is reported.

- [ ] **Step 2: Review final diff**

Run:

```powershell
git status --short
git log --oneline -5
```

Expected: source/test changes are committed. Only generated TorchScript files may remain untracked.

- [ ] **Step 3: Push**

Run:

```powershell
git push origin main
```

Expected: `main -> main`.

---

## Self-Review

- Spec coverage: the plan implements pair grouping, best-by-inliers display, detail-view algorithm switching, SfM explicit diagnostics, and keeps the SfM default selection policy unchanged.
- Placeholder scan: no task contains placeholder markers or unspecified implementation steps.
- Type consistency: `MatchVariant`, `MatchPairGroup`, `MatchResultCatalogConfig`, `MatchResultCatalogSummary`, and `MatchResultCatalog::scan()` are introduced in Task 2 and reused consistently in GUI and SfM tasks.
- Scope check: automatic mixed-algorithm SfM selection is intentionally excluded and documented as a future explicit option.

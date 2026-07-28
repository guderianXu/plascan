#pragma once

#include "Camera.h"
#include "common/SfmTypes.h"

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>
#include <memory>

namespace xjw::aerial_triangulation
{

// 面向 GUI/CLI 的“对齐照片”参数。连接点前端参数在 Workflow 中转换为
// MatchPhotosOptions，不允许继续传入纯 SfM 重建管线。
struct AerialTriangulationOptions
{
    QStringList images;
    QStringList cameraPaths;
    QString projectPath;
    QString outputDir;
    QJsonObject projectMeta;

    QString quality = QStringLiteral("high");
    bool genericPreselection = true;
    bool referencePreselection = false;
    QString referenceMode = QStringLiteral("source_code");
    bool resetAlignment = true;
    // 重置相机对齐与重建连接点是两个独立动作。默认保留兼容的特征、匹配和连接点缓存，
    // 让用户可以在相同观测网络上重新尝试 SfM/BA。
    bool reuseExistingMatches = true;
    bool saveAfterEachStep = false;

    int keypointLimit = 40000;
    int tiepointLimit = 4000;
    QString maskApplyMode = QStringLiteral("none");
    bool excludeFixedTiePoints = true;
    bool guidedImageMatching = false;
    bool adaptiveCameraModelFitting = true;
    // Benchmark/reference-camera workflows can request exact external poses.
    // Normal GUI alignment keeps the existing soft-prior refinement behavior.
    bool lockInputCameraPoses = false;
    bool useInitialPairHint = false;
    ImageId initialImageId1 = kInvalidImageId;
    ImageId initialImageId2 = kInvalidImageId;

    QString featureAlgorithm = QStringLiteral("sift");
    QString matchAlgorithm = QStringLiteral("lightglue");
    QString matchPipeline;
    QString device = QStringLiteral("auto");
    int threads = 8;
    int featureMaxImageDim = 0;
    int cudaParallelPairs = 0;
    bool autoGenerateMissingMatches = false;
    bool restrictPairs = false;
    QStringList allowedPairs;
    QString assetsDir;
    QString featureDir;
    QString matchDir;
    QMap<QString, QString> maskPaths;
    QMap<QString, Camera> referenceCameras;
    float featureGrayscaleMin = 5.0f / 255.0f;
    float featureGrayscaleMax = 1.0f;

    std::shared_ptr<std::atomic<bool>> cancelFlag;
    std::function<void(const QString &stage, int percent)> progressFn;
    std::function<void(const QString &img0,
                       const QString &img1,
                       const QString &matchPath,
                       int numMatches)> pairMatchedFn;
};

// 已完成连接点准备后的 SfM/BA 输入。该类型刻意不包含任何特征提取和匹配参数。
struct PreparedAerialTriangulationInput
{
    QStringList images;
    QStringList cameraPaths;
    QString projectPath;
    QString markerSetPath;
    QString tiePointPath;
    QString outputDir;
    QJsonObject projectMeta;

    int quality = 2;
    int threads = 8;
    QString device = QStringLiteral("auto");
    bool useProjectCameraIntrinsics = true;
    bool useProjectCameraPoses = false;
    bool adaptiveCameraModelFitting = true;
    bool lockInputCameraPoses = false;
    bool enforceSequencePoseConsistency = false;
    bool sequenceLoopClosure = false;
    bool useInitialPairHint = false;
    ImageId initialImageId1 = kInvalidImageId;
    ImageId initialImageId2 = kInvalidImageId;
    // 无标定相机的初始焦距，以“焦距像素 / 影像最长边”表示。
    double estimatedFocalScale = 1.2;

    std::shared_ptr<std::atomic<bool>> cancelFlag;
    std::function<void(const QString &stage, int percent)> progressFn;
};

} // namespace xjw::aerial_triangulation

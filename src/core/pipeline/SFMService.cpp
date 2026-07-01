// =============================================================================
// 文件名: SFMService.cpp
// 描述:   增量式 SFM 一站式服务实现，详细说明见 SFMService.h。
//
//         全自动流水线：DISK 特征提取 → LightGlue 匹配 → 增量式 SFM
//         本文件不依赖任何 Qt Widget，所有 GUI 提示由调用方负责。
// =============================================================================

// ── LibTorch / OpenCV 头文件必须在 Qt 头文件之前引入，避免宏冲突 ────────
#include "compat/QtTorchMacroGuard.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "ExtractorFactory.h"
#include "FeatureData.h"
#include "FeatureFileIO.h"
#include "SuperGlueMatchIO.h"
#include "MatchOutlierRejector.h"
#include "AlgorithmCompat.h"
#include "lightglue/LightGlueMatcher.h"
#include "tradition/TraditionalFeatureMatcher.h"
#include <opencv2/opencv.hpp>
#if defined(PLASCAN_TORCH_HAS_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <torch/torch.h>

// ── 项目 / 服务头文件 ──────────────────────────────────────────────────────
#include "SFMService.h"
#include "SfmPairPlanner.h"
#include "SfmMatchDiagnostics.h"
#include "GuidedRematchService.h"
#include "MatchResultCatalog.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "Logger.h"
#include "Camera.h"
#include "OverlapAnalyzer.h"
#include "project/SparseResultQuality.h"
#include "pipeline/IncrementalSfm.h"
#include "common/SfmTypes.h"
#include "quality/SfmQualityReport.h"
#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTextStream>
#include <QDateTime>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QImageReader>
#include <QProcess>
#include <QSet>
#include <QStandardPaths>
#include <QStringConverter>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace xjw
{
namespace gui
{

// ══════════════════════════════════════════════════════════════════════════════
// 匿名命名空间：文件内部辅助工具
// ══════════════════════════════════════════════════════════════════════════════
namespace
{

// ── 路径工具 ─────────────────────────────────────────────────────────────────

QString normalizePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

bool shouldReportIndexedProgress(int doneCount, int totalCount, int maxReports = 200)
{
    if (totalCount <= 0)
    {
        return false;
    }
    if (doneCount <= 1 || doneCount >= totalCount)
    {
        return true;
    }
    if (totalCount <= maxReports)
    {
        return true;
    }
    const int stride = std::max(1, totalCount / std::max(1, maxReports));
    return (doneCount % stride) == 0;
}

void logSfmMatchCacheCatalogDiagnostics(const QString &matchDir)
{
    if (matchDir.trimmed().isEmpty())
    {
        LOG_INFO(QStringLiteral("匹配缓存目录诊断: 未提供匹配缓存目录，跳过"));
        return;
    }

    const QDir dir(matchDir);
    if (!dir.exists())
    {
        LOG_INFO(QStringLiteral("匹配缓存目录诊断: 目录不存在，跳过: %1").arg(matchDir));
        return;
    }

    xjw::pipeline::MatchResultCatalogConfig catalogConfig;
    catalogConfig.matchDirectory = matchDir;
    const xjw::pipeline::MatchResultCatalog catalog(catalogConfig);
    const xjw::pipeline::MatchResultCatalogSummary summary = catalog.scan();
    if (summary.variantCount == 0)
    {
        LOG_INFO(QStringLiteral("匹配缓存目录诊断: 目录为空或没有 .match 文件: %1").arg(matchDir));
        return;
    }

    int compatiblePairCount = 0;
    int multiAlgorithmPairCount = 0;
    for (const xjw::pipeline::MatchPairGroup &group : summary.pairGroups)
    {
        QSet<QString> algorithmKeys;
        bool hasCompatibleVariant = false;
        for (const xjw::pipeline::MatchVariant &variant : group.variants)
        {
            if (!variant.compatible)
            {
                continue;
            }

            hasCompatibleVariant = true;
            algorithmKeys.insert(QStringLiteral("%1 + %2")
                                     .arg(variant.featureAlgorithm.trimmed().toLower(),
                                          variant.matchAlgorithm.trimmed().toLower()));
        }

        if (hasCompatibleVariant)
        {
            ++compatiblePairCount;
        }
        if (algorithmKeys.size() > 1)
        {
            ++multiAlgorithmPairCount;
        }

        if (group.bestVariantIndex >= 0 && group.variants.size() > 1)
        {
            const xjw::pipeline::MatchVariant &bestVariant = group.variants.at(group.bestVariantIndex);
            LOG_INFO(QStringLiteral(
                "匹配缓存目录诊断: pair=%1 variant=%2 个, best variant 只是展示/诊断用途: %3 + %4, inliers=%5, total=%6")
                .arg(group.pairKey)
                .arg(group.variants.size())
                .arg(bestVariant.featureAlgorithm.isEmpty() ? QStringLiteral("unknown") : bestVariant.featureAlgorithm,
                     bestVariant.matchAlgorithm.isEmpty() ? QStringLiteral("unknown") : bestVariant.matchAlgorithm)
                .arg(bestVariant.hasInlierStats ? bestVariant.geometricVerifiedInliers : -1)
                .arg(bestVariant.totalMatches));
        }
    }

    LOG_INFO(QStringLiteral(
        "匹配缓存目录诊断: 总 pair 数=%1, 可查看/兼容 pair 数=%2, 多算法 pair 数=%3, 不可用 variant 数=%4, variant 总数=%5")
        .arg(summary.pairGroupCount)
        .arg(compatiblePairCount)
        .arg(multiAlgorithmPairCount)
        .arg(summary.incompatibleVariantCount)
        .arg(summary.variantCount));
    if (multiAlgorithmPairCount > 0)
    {
        LOG_INFO(QStringLiteral(
            "匹配缓存目录诊断: 检测到同一影像对存在多个算法结果，catalog best variant 只是展示/诊断用途"));
    }
    LOG_INFO(QStringLiteral("SfM 默认仍按当前 feature_algorithm + match_algorithm 选择匹配"));
}

struct SparseExportColorRequest
{
    std::size_t pointIndex = 0;
    FeatureIdx featureIdx = 0;
};

void sampleSparseExportColorsByImage(
    const QMap<ImageId, std::vector<SparseExportColorRequest>> &colorRequestsByImage,
    const QMap<ImageId, std::vector<cv::KeyPoint>> &kptPositions,
    const QMap<ImageId, QString> &idToPath,
    std::vector<uint8_t> *colorsData)
{
    if (colorsData == nullptr || colorsData->empty())
    {
        return;
    }

    const std::size_t pointCount = colorsData->size() / 3;
    std::vector<unsigned char> colorFilled(pointCount, 0);

    for (auto requestIt = colorRequestsByImage.constBegin(); requestIt != colorRequestsByImage.constEnd(); ++requestIt)
    {
        const ImageId imageId = requestIt.key();
        const QString imagePath = idToPath.value(imageId);
        if (imagePath.isEmpty())
        {
            continue;
        }

        auto kptIt = kptPositions.find(imageId);
        if (kptIt == kptPositions.end())
        {
            continue;
        }

        const cv::Mat image = cv::imread(imagePath.toStdString(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            continue;
        }

        const auto &kpts = kptIt.value();
        for (const SparseExportColorRequest &request : requestIt.value())
        {
            if (request.pointIndex >= pointCount || colorFilled[request.pointIndex] != 0)
            {
                continue;
            }
            if (request.featureIdx >= static_cast<FeatureIdx>(kpts.size()))
            {
                continue;
            }

            const int px = static_cast<int>(std::round(kpts[request.featureIdx].pt.x));
            const int py = static_cast<int>(std::round(kpts[request.featureIdx].pt.y));
            if (px < 0 || px >= image.cols || py < 0 || py >= image.rows)
            {
                continue;
            }

            const cv::Vec3b &pix = image.at<cv::Vec3b>(py, px);
            const std::size_t colorBase = request.pointIndex * 3;
            (*colorsData)[colorBase] = pix[2];
            (*colorsData)[colorBase + 1] = pix[1];
            (*colorsData)[colorBase + 2] = pix[0];
            colorFilled[request.pointIndex] = 1;
        }
    }
}

QString canonicalPairKey(const QString &pathA, const QString &pathB)
{
    const QString normA = normalizePath(pathA);
    const QString normB = normalizePath(pathB);
    if (normA.isEmpty() || normB.isEmpty() || normA == normB)
    {
        return QString();
    }
    return (normA < normB)
        ? (normA + QStringLiteral("\n") + normB)
        : (normB + QStringLiteral("\n") + normA);
}

QJsonArray ipmatchResultsFromMeta(const QJsonObject &meta)
{
    const QJsonArray direct = meta.value(QStringLiteral("ipmatch_results")).toArray();
    if (!direct.isEmpty())
    {
        return direct;
    }

    const QJsonObject resultsObj = meta.value(QStringLiteral("project_results")).toObject();
    if (!resultsObj.isEmpty())
    {
        return resultsObj.value(QStringLiteral("ipmatch_results")).toArray();
    }

    return QJsonArray();
}

QString resolvePathFromProjectRoot(const QString &projectRoot, const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    if (QDir::isAbsolutePath(trimmed))
    {
        return QDir::cleanPath(trimmed);
    }

    if (projectRoot.isEmpty())
    {
        return QDir::cleanPath(trimmed);
    }

    return QDir(projectRoot).filePath(trimmed);
}

QString resolveImagePathTokenFromMeta(const QString &token, const QJsonObject &projectMeta)
{
    const QString trimmed = token.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    const QString resolved = xjw::gui::project::resolveProjectImagePathFromToken(trimmed, projectMeta);
    if (!resolved.isEmpty())
    {
        return normalizePath(resolved);
    }

    return normalizePath(trimmed);
}

QString canonicalNamePairKey(const QString &nameA, const QString &nameB)
{
    const QString a = nameA.trimmed().toLower();
    const QString b = nameB.trimmed().toLower();
    if (a.isEmpty() || b.isEmpty() || a == b)
    {
        return QString();
    }
    return (a < b) ? (a + QStringLiteral("\n") + b) : (b + QStringLiteral("\n") + a);
}

std::array<double, 3> cameraViewingDirection(const Camera &camera)
{
    const auto rotation = camera.cameraToWorldRotation();
    return {rotation[2], rotation[5], rotation[8]};
}

std::vector<std::array<double, 3>> loadKnownCameraCentersFromPaths(const QStringList &images,
                                                                   const QStringList &cameraPaths,
                                                                   QString *errorMessage)
{
    std::vector<std::array<double, 3>> centers;
    if (images.isEmpty() || cameraPaths.size() != images.size())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("影像数量和相机文件数量不一致");
        }
        return centers;
    }

    centers.reserve(static_cast<std::size_t>(cameraPaths.size()));
    for (int i = 0; i < cameraPaths.size(); ++i)
    {
        const QString cameraPath = cameraPaths.at(i).trimmed();
        if (cameraPath.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("第 %1 张影像缺少相机文件: %2").arg(i + 1).arg(images.value(i));
            }
            centers.clear();
            return centers;
        }

        Camera camera;
        if (!camera.loadFromFile(cameraPath.toStdString()) || !camera.isValid())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取相机文件: %1").arg(cameraPath);
            }
            centers.clear();
            return centers;
        }

        centers.push_back(camera.cameraCenter());
    }

    return centers;
}

std::vector<std::array<double, 3>> loadKnownCameraViewingDirectionsFromPaths(const QStringList &images,
                                                                            const QStringList &cameraPaths,
                                                                            QString *errorMessage)
{
    std::vector<std::array<double, 3>> directions;
    if (images.isEmpty() || cameraPaths.size() != images.size())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("影像数量和相机文件数量不一致");
        }
        return directions;
    }

    directions.reserve(static_cast<std::size_t>(cameraPaths.size()));
    for (int i = 0; i < cameraPaths.size(); ++i)
    {
        const QString cameraPath = cameraPaths.at(i).trimmed();
        if (cameraPath.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("第 %1 张影像缺少相机文件: %2").arg(i + 1).arg(images.value(i));
            }
            directions.clear();
            return directions;
        }

        Camera camera;
        if (!camera.loadFromFile(cameraPath.toStdString()) || !camera.isValid())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取相机文件: %1").arg(cameraPath);
            }
            directions.clear();
            return directions;
        }

        directions.push_back(cameraViewingDirection(camera));
    }

    return directions;
}

std::vector<std::array<double, 3>> loadKnownCameraCentersFromProjectMeta(const QStringList &images,
                                                                        const QJsonObject &projectMeta,
                                                                        int *matchedCount)
{
    std::vector<std::array<double, 3>> centers;
    if (matchedCount)
    {
        *matchedCount = 0;
    }
    if (images.isEmpty() || projectMeta.isEmpty())
    {
        return centers;
    }

    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::gui::project::projectImageMetaByPath(projectMeta, true);
    if (imageMetaByPath.isEmpty())
    {
        return centers;
    }

    centers.reserve(static_cast<std::size_t>(images.size()));
    int matched = 0;
    for (const QString &imagePath : images)
    {
        const QString normalizedImagePath = xjw::gui::project::normalizePath(imagePath);
        const QJsonObject imageMeta = imageMetaByPath.value(normalizedImagePath);
        Camera camera;
        if (imageMeta.isEmpty() || !xjw::gui::project::imageCameraFromEntry(imageMeta, &camera) || !camera.isValid())
        {
            centers.clear();
            if (matchedCount)
            {
                *matchedCount = matched;
            }
            return centers;
        }

        centers.push_back(camera.cameraCenter());
        ++matched;
    }

    if (matchedCount)
    {
        *matchedCount = matched;
    }
    return centers;
}

std::vector<std::array<double, 3>> loadKnownCameraViewingDirectionsFromProjectMeta(const QStringList &images,
                                                                                  const QJsonObject &projectMeta,
                                                                                  int *matchedCount)
{
    std::vector<std::array<double, 3>> directions;
    if (matchedCount)
    {
        *matchedCount = 0;
    }
    if (images.isEmpty() || projectMeta.isEmpty())
    {
        return directions;
    }

    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::gui::project::projectImageMetaByPath(projectMeta, true);
    if (imageMetaByPath.isEmpty())
    {
        return directions;
    }

    directions.reserve(static_cast<std::size_t>(images.size()));
    int matched = 0;
    for (const QString &imagePath : images)
    {
        const QString normalizedImagePath = xjw::gui::project::normalizePath(imagePath);
        const QJsonObject imageMeta = imageMetaByPath.value(normalizedImagePath);
        Camera camera;
        if (imageMeta.isEmpty() || !xjw::gui::project::imageCameraFromEntry(imageMeta, &camera) || !camera.isValid())
        {
            directions.clear();
            if (matchedCount)
            {
                *matchedCount = matched;
            }
            return directions;
        }

        directions.push_back(cameraViewingDirection(camera));
        ++matched;
    }

    if (matchedCount)
    {
        *matchedCount = matched;
    }
    return directions;
}

std::vector<std::array<int, 2>> loadKnownCameraOverlapPairsFromPaths(const QStringList &images,
                                                                     const QStringList &cameraPaths,
                                                                     double neighborFactor,
                                                                     QString *detailMessage,
                                                                     QString *errorMessage)
{
    std::vector<std::array<int, 2>> pairs;
    if (detailMessage)
    {
        detailMessage->clear();
    }
    if (errorMessage)
    {
        errorMessage->clear();
    }

    if (images.isEmpty() || cameraPaths.size() != images.size())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("影像数量和相机文件数量不一致");
        }
        return pairs;
    }

    std::vector<OverlapImageInput> inputs;
    inputs.reserve(static_cast<std::size_t>(images.size()));
    for (int i = 0; i < images.size(); ++i)
    {
        const QString imagePath = images.at(i).trimmed();
        const QString cameraPath = cameraPaths.at(i).trimmed();
        if (imagePath.isEmpty() || cameraPath.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("第 %1 张影像缺少影像或相机路径").arg(i + 1);
            }
            return pairs;
        }

        Camera camera;
        if (!camera.loadFromFile(cameraPath.toStdString()) || !camera.isValid())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取相机文件: %1").arg(cameraPath);
            }
            return pairs;
        }

        QSize imageSize = QImageReader(imagePath).size();
        if (!imageSize.isValid() || imageSize.width() <= 0 || imageSize.height() <= 0)
        {
            const cv::Mat image = cv::imread(imagePath.toStdString(), cv::IMREAD_GRAYSCALE);
            if (!image.empty())
            {
                imageSize = QSize(image.cols, image.rows);
            }
        }

        if (!imageSize.isValid() || imageSize.width() <= 0 || imageSize.height() <= 0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取影像尺寸: %1").arg(imagePath);
            }
            return pairs;
        }

        OverlapImageInput input;
        input.imagePath = imagePath.toStdString();
        input.camera = camera;
        input.width = imageSize.width();
        input.height = imageSize.height();
        inputs.push_back(std::move(input));
    }

    OverlapAnalysisOptions overlapOptions;
    overlapOptions.groundModel = OverlapGroundModel::ReferenceSphere;
    overlapOptions.neighborFactor = std::max(0.1, neighborFactor);
    overlapOptions.referenceSphere.body = ReferenceBody::Earth;
    overlapOptions.referenceSphere.radiusMeters = referenceBodyRadiusMeters(ReferenceBody::Earth);
    overlapOptions.referenceSphere.centerMode = ReferenceSphereCenterMode::Auto;
    overlapOptions.referenceSphere.autoLocalTangentHeight = true;

    OverlapAnalysisResult overlapResult;
    std::string overlapError;
    if (!OverlapAnalyzer::analyze(inputs, overlapOptions, &overlapResult, &overlapError))
    {
        if (errorMessage)
        {
            *errorMessage = QString::fromStdString(overlapError);
        }
        return pairs;
    }

    pairs.reserve(overlapResult.pairs.size());
    for (const OverlapPairResult &pair : overlapResult.pairs)
    {
        pairs.push_back({pair.indexA, pair.indexB});
    }

    if (detailMessage)
    {
        *detailMessage = QString::fromStdString(overlapResult.detail);
    }
    return pairs;
}

// ── 相机 JSON 序列化 ─────────────────────────────────────────────────────────

QJsonObject cameraToJson(const Camera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonObject obj;
    obj[QStringLiteral("model")]       = QStringLiteral("tsai");
    obj[QStringLiteral("imported_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    obj[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    obj[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    obj[QStringLiteral("pitch")] = camera.pixelPitch();
    obj[QStringLiteral("fu")] = camera.focalXMillimeters();
    obj[QStringLiteral("fv")] = camera.focalYMillimeters();
    obj[QStringLiteral("cu")] = camera.principalXMillimeters();
    obj[QStringLiteral("cv")] = camera.principalYMillimeters();
    obj[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
    obj[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
    obj[QStringLiteral("depth_axis_flipped")] = camera.depthAxisFlipped();

    QJsonArray cArr;
    cArr.append(center[0]);
    cArr.append(center[1]);
    cArr.append(center[2]);
    obj[QStringLiteral("C")] = cArr;

    QJsonArray rArr;
    for (int i = 0; i < 9; ++i)
    {
        rArr.append(rotation[i]);
    }
    obj[QStringLiteral("R")] = rArr;

    // 欧拉角（供显示用）
    const double r00 = rotation[0], r10 = rotation[3], r20 = rotation[6];
    const double r21 = rotation[7], r22 = rotation[8];
    const double pitch = std::asin(std::clamp(-r20, -1.0, 1.0));
    double yaw = 0.0, roll = 0.0;
    if (std::abs(std::cos(pitch)) > 1e-8)
    {
        yaw  = std::atan2(r10, r00);
        roll = std::atan2(r21, r22);
    }
    else
    {
        yaw = std::atan2(-rotation[1], rotation[4]);
    }
    constexpr double kRad2Deg = 180.0 / M_PI;
    obj[QStringLiteral("yaw_deg")]   = yaw   * kRad2Deg;
    obj[QStringLiteral("pitch_deg")] = pitch * kRad2Deg;
    obj[QStringLiteral("roll_deg")]  = roll  * kRad2Deg;

    return obj;
}

// ── 模型文件查找 ─────────────────────────────────────────────────────────────

QString findModelFile(const QString &modelName)
{
    QStringList candidates;
    QStringList modelNames;
    modelNames.append(modelName);

    const QString envModelDir = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!envModelDir.isEmpty())
    {
        for (const QString &name : modelNames)
        {
            candidates.append(QDir(envModelDir).filePath(name));
        }
    }

#ifdef PLASCAN_SOURCE_DIR
    for (const QString &name : modelNames)
    {
        candidates.append(
            QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("resources/models/%1").arg(name)));
    }
#endif

    const QString exeDir = QCoreApplication::applicationDirPath();
    for (const QString &name : modelNames)
    {
        candidates.append(QDir(exeDir).filePath(QStringLiteral("../models/%1").arg(name)));
        candidates.append(QDir(exeDir).filePath(QStringLiteral("../resources/models/%1").arg(name)));
        candidates.append(QDir(exeDir).filePath(QStringLiteral("../../resources/models/%1").arg(name)));
        candidates.append(QStringLiteral("models/%1").arg(name));
    }

    for (const QString &candidate : candidates)
    {
        if (QFile::exists(candidate))
        {
            return candidate;
        }
    }

    return QString();
}

QString findFirstModelFile(const QStringList &modelNames, QString *pickedModelName = nullptr)
{
    for (const QString &modelName : modelNames)
    {
        const QString path = findModelFile(modelName);
        if (!path.isEmpty())
        {
            if (pickedModelName)
            {
                *pickedModelName = modelName;
            }
            return path;
        }
    }
    return QString();
}

QString normalizedAlgorithm(QString value, const QString &fallback)
{
    value = value.trimmed().toLower();
    return value.isEmpty() ? fallback : value;
}

bool isSfmFeatureAlgorithm(const QString &featureAlgorithm)
{
    return featureAlgorithm == QStringLiteral("superpoint") ||
           featureAlgorithm == QStringLiteral("disk") ||
           featureAlgorithm == QStringLiteral("aliked") ||
           featureAlgorithm == QStringLiteral("sift") ||
           featureAlgorithm == QStringLiteral("orb") ||
           featureAlgorithm == QStringLiteral("akaze");
}

bool isLightGlueSfmMatch(const QString &featureAlgorithm, const QString &matchAlgorithm)
{
    return matchAlgorithm == QStringLiteral("lightglue") &&
           (featureAlgorithm == QStringLiteral("disk") ||
            featureAlgorithm == QStringLiteral("aliked") ||
            featureAlgorithm == QStringLiteral("sift"));
}

bool isTraditionalSiftMatch(const QString &featureAlgorithm, const QString &matchAlgorithm)
{
    return featureAlgorithm == QStringLiteral("sift") &&
           (matchAlgorithm == QStringLiteral("sift_flann") ||
            matchAlgorithm == QStringLiteral("sift_bf_l2"));
}

bool sfmFeatureNeedsModel(const QString &featureAlgorithm)
{
    return featureAlgorithm == QStringLiteral("disk") ||
           featureAlgorithm == QStringLiteral("aliked") ||
           featureAlgorithm == QStringLiteral("superpoint");
}

QStringList featureModelCandidates(const QString &featureAlgorithm, bool useCuda)
{
    const QString suffix = useCuda ? QStringLiteral("cuda") : QStringLiteral("cpu");
    if (featureAlgorithm == QStringLiteral("disk"))
    {
        return {
            QStringLiteral("disk_extractor_%1_8192.torchscript").arg(suffix),
            QStringLiteral("disk_extractor_%1_8192.pt").arg(suffix),
            QStringLiteral("disk_extractor_%1_1200.torchscript").arg(suffix),
            QStringLiteral("disk_extractor_%1_1200.pt").arg(suffix),
            QStringLiteral("disk_extractor.torchscript"),
            QStringLiteral("disk_extractor.pt")
        };
    }
    if (featureAlgorithm == QStringLiteral("aliked"))
    {
        return {
            QStringLiteral("aliked_extractor_%1_480.torchscript").arg(suffix),
            QStringLiteral("aliked_extractor_%1_480.pt").arg(suffix),
            QStringLiteral("aliked_extractor.torchscript"),
            QStringLiteral("aliked_extractor.pt")
        };
    }
    return {};
}

QStringList lightGlueModelCandidates(const QString &featureAlgorithm, bool useCuda)
{
    const QString suffix = useCuda ? QStringLiteral("cuda") : QStringLiteral("cpu");
    if (featureAlgorithm == QStringLiteral("disk"))
    {
        return {
            QStringLiteral("lightglue_disk_%1.torchscript").arg(suffix)
        };
    }
    if (featureAlgorithm == QStringLiteral("aliked"))
    {
        return {
            QStringLiteral("lightglue_aliked_%1.torchscript").arg(suffix)
        };
    }
    if (featureAlgorithm == QStringLiteral("sift"))
    {
        return {
            QStringLiteral("lightglue_sift_%1.torchscript").arg(suffix)
        };
    }
    return {
        QStringLiteral("lightglue_matcher_%1.torchscript").arg(suffix),
        QStringLiteral("lightglue_matcher.torchscript")
    };
}

QString findScriptFile(const QString &scriptName)
{
    QStringList candidates;

    const QString envScriptDir = qEnvironmentVariable("PLASCAN_SCRIPT_DIR").trimmed();
    if (!envScriptDir.isEmpty())
    {
        candidates.append(QDir(envScriptDir).filePath(scriptName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("scripts/%1").arg(scriptName)));
#endif

    const QString exeDir = QCoreApplication::applicationDirPath();
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../scripts/%1").arg(scriptName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../../scripts/%1").arg(scriptName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../../../scripts/%1").arg(scriptName)));
    candidates.append(QDir(QDir::currentPath()).filePath(QStringLiteral("scripts/%1").arg(scriptName)));

    for (const QString &candidate : candidates)
    {
        if (QFileInfo::exists(candidate))
        {
            return QDir::cleanPath(QFileInfo(candidate).absoluteFilePath());
        }
    }

    return QString();
}

QString resolvedExecutablePath(const QString &candidate)
{
    const QString trimmed = candidate.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    if (trimmed.contains(QLatin1Char('/')) || trimmed.contains(QLatin1Char('\\')))
    {
        const QFileInfo info(trimmed);
        if (info.exists() && info.isFile() && info.isExecutable())
        {
            return QDir::cleanPath(info.absoluteFilePath());
        }
        return QString();
    }

    const QString resolved = QStandardPaths::findExecutable(trimmed);
    return resolved.isEmpty() ? QString() : QDir::cleanPath(QFileInfo(resolved).absoluteFilePath());
}

QString pythonInEnvironmentPrefix(const QString &prefix)
{
    const QString trimmed = prefix.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    QStringList candidates;
#ifdef Q_OS_WIN
    candidates << QDir(trimmed).filePath(QStringLiteral("python.exe"))
               << QDir(trimmed).filePath(QStringLiteral("Scripts/python.exe"));
#else
    candidates << QDir(trimmed).filePath(QStringLiteral("bin/python"));
#endif

    for (const QString &candidate : candidates)
    {
        const QString resolved = resolvedExecutablePath(candidate);
        if (!resolved.isEmpty())
        {
            return resolved;
        }
    }
    return QString();
}

QStringList defaultPythonEnvironmentPrefixes()
{
    QStringList prefixes;

    const QString mambaRoot = qEnvironmentVariable("MAMBA_ROOT_PREFIX").trimmed();
    if (!mambaRoot.isEmpty())
    {
        prefixes << QDir(mambaRoot).filePath(QStringLiteral("envs/plascan"));
    }

    const QString home = QDir::homePath();
    prefixes << QDir(home).filePath(QStringLiteral(".local/share/mamba/envs/plascan"))
             << QDir(home).filePath(QStringLiteral("mambaforge/envs/plascan"))
             << QDir(home).filePath(QStringLiteral("miniforge3/envs/plascan"))
             << QDir(home).filePath(QStringLiteral("miniconda3/envs/plascan"))
             << QDir(home).filePath(QStringLiteral("anaconda3/envs/plascan"));

    prefixes.removeDuplicates();
    return prefixes;
}

bool pythonHasLightGlueBackend(const QString &executable)
{
    QProcess process;
    process.start(executable, {QStringLiteral("-c"), QStringLiteral("import torch, lightglue")});
    if (!process.waitForFinished(10000))
    {
        process.kill();
        process.waitForFinished(3000);
        return false;
    }
    return process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0;
}

QString resolvePythonExecutableForLightGlue()
{
    QStringList candidates;
    candidates << qEnvironmentVariable("PLASCAN_PYTHON_EXECUTABLE").trimmed()
               << qEnvironmentVariable("PLASCAN_PYTHON").trimmed()
               << pythonInEnvironmentPrefix(qEnvironmentVariable("VIRTUAL_ENV"))
               << pythonInEnvironmentPrefix(qEnvironmentVariable("CONDA_PREFIX"));

    for (const QString &prefix : defaultPythonEnvironmentPrefixes())
    {
        candidates << pythonInEnvironmentPrefix(prefix);
    }

    const QString pythonEnv = qEnvironmentVariable("PYTHON").trimmed();
    if (!pythonEnv.isEmpty())
    {
        candidates << pythonEnv;
    }
    candidates << QStringLiteral("python3") << QStringLiteral("python");
    candidates.removeAll(QString());
    candidates.removeDuplicates();

    QString firstResolved;
    for (const QString &candidate : candidates)
    {
        const QString resolved = resolvedExecutablePath(candidate);
        if (resolved.isEmpty())
        {
            continue;
        }
        if (firstResolved.isEmpty())
        {
            firstResolved = resolved;
        }
        if (pythonHasLightGlueBackend(resolved))
        {
            return resolved;
        }
    }
    return firstResolved.isEmpty() ? QStringLiteral("python3") : firstResolved;
}

QString pythonExecutable()
{
    static const QString cached = resolvePythonExecutableForLightGlue();
    return cached;
}

QString processOutputSummary(QProcess &process)
{
    const QString stdoutText = QString::fromUtf8(process.readAllStandardOutput()).trimmed();
    const QString stderrText = QString::fromUtf8(process.readAllStandardError()).trimmed();
    if (!stderrText.isEmpty() && !stdoutText.isEmpty())
    {
        return QStringLiteral("%1\n%2").arg(stderrText, stdoutText);
    }
    if (!stderrText.isEmpty())
    {
        return stderrText;
    }
    return stdoutText;
}

QString lightGlueModelOutputDir(QString *error)
{
    const QString envModelDir = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!envModelDir.isEmpty())
    {
        return QDir::cleanPath(QFileInfo(envModelDir).absoluteFilePath());
    }

#ifdef PLASCAN_SOURCE_DIR
    return QDir::cleanPath(QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("resources/models")));
#else
    const QString exeDir = QCoreApplication::applicationDirPath();
    const QString installedModels = QDir(exeDir).filePath(QStringLiteral("../models"));
    if (!installedModels.trimmed().isEmpty())
    {
        return QDir::cleanPath(QFileInfo(installedModels).absoluteFilePath());
    }
    if (error)
    {
        *error = QStringLiteral("无法确定模型输出目录");
    }
    return QString();
#endif
}

QString lightGlueTorchScriptModelName(const QString &featureAlgorithm, bool useCuda)
{
    const QString suffix = useCuda ? QStringLiteral("cuda") : QStringLiteral("cpu");
    return QStringLiteral("lightglue_%1_%2.torchscript").arg(featureAlgorithm, suffix);
}

QString ensureLightGlueTorchScriptModel(const QString &featureAlgorithm,
                                        bool useCuda,
                                        QString *pickedModelName,
                                        QString *error)
{
    const QString modelName = lightGlueTorchScriptModelName(featureAlgorithm, useCuda);
    const QString existingPath = findModelFile(modelName);
    if (!existingPath.isEmpty())
    {
        if (pickedModelName)
        {
            *pickedModelName = modelName;
        }
        return existingPath;
    }

    const QString scriptPath = findScriptFile(QStringLiteral("export_lightglue_torchscript.py"));
    if (scriptPath.isEmpty())
    {
        if (error)
        {
            *error = QStringLiteral("未找到自动导出脚本 export_lightglue_torchscript.py");
        }
        return QString();
    }

    QString outputDirError;
    const QString outputDir = lightGlueModelOutputDir(&outputDirError);
    if (outputDir.isEmpty())
    {
        if (error)
        {
            *error = outputDirError;
        }
        return QString();
    }

    QDir dir;
    if (!dir.mkpath(outputDir))
    {
        if (error)
        {
            *error = QStringLiteral("无法创建模型输出目录: %1").arg(outputDir);
        }
        return QString();
    }

    const QString deviceName = useCuda ? QStringLiteral("cuda") : QStringLiteral("cpu");
    QStringList args;
    args << scriptPath
         << QStringLiteral("--features") << featureAlgorithm
         << QStringLiteral("--devices") << deviceName
         << QStringLiteral("--output-dir") << outputDir;

    LOG_INFO(QStringLiteral("  LightGlue %1 TorchScript 缺失，自动导出: %2")
        .arg(featureAlgorithm.toUpper(), modelName));

    QProcess process;
    process.setWorkingDirectory(QFileInfo(scriptPath).absolutePath());
    process.start(pythonExecutable(), args);
    if (!process.waitForStarted(30000))
    {
        if (error)
        {
            *error = QStringLiteral("无法启动 LightGlue TorchScript 自动导出: %1").arg(process.errorString());
        }
        return QString();
    }
    if (!process.waitForFinished(900000))
    {
        process.kill();
        process.waitForFinished(5000);
        if (error)
        {
            *error = QStringLiteral("LightGlue TorchScript 自动导出超时: %1").arg(processOutputSummary(process));
        }
        return QString();
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
    {
        if (error)
        {
            const QString details = processOutputSummary(process);
            *error = details.isEmpty()
                ? QStringLiteral("LightGlue TorchScript 自动导出失败，退出码: %1").arg(process.exitCode())
                : QStringLiteral("LightGlue TorchScript 自动导出失败，退出码: %1\n%2")
                    .arg(process.exitCode())
                    .arg(details);
        }
        return QString();
    }

    const QString generatedPath = findModelFile(modelName);
    if (!generatedPath.isEmpty())
    {
        if (pickedModelName)
        {
            *pickedModelName = modelName;
        }
        LOG_INFO(QStringLiteral("  LightGlue TorchScript 已生成: %1").arg(generatedPath));
        return generatedPath;
    }

    const QString directPath = QDir(outputDir).filePath(modelName);
    if (QFile::exists(directPath))
    {
        if (pickedModelName)
        {
            *pickedModelName = modelName;
        }
        LOG_INFO(QStringLiteral("  LightGlue TorchScript 已生成: %1").arg(directPath));
        return QDir::cleanPath(QFileInfo(directPath).absoluteFilePath());
    }

    if (error)
    {
        *error = QStringLiteral("自动导出完成但未找到模型文件: %1").arg(modelName);
    }
    return QString();
}

bool runPythonLightGlue(const QString &scriptPath,
                        const QString &featurePath0,
                        const QString &featurePath1,
                        const QString &matchPath,
                        bool useCuda,
                        float threshold,
                        const QString &featureAlgorithm,
                        const QString &matchAlgorithm,
                        xjw::feature_match::MatchResult *matchResult,
                        QString *error)
{
    if (!matchResult)
    {
        if (error) *error = QStringLiteral("内部错误：匹配结果输出为空");
        return false;
    }

    QStringList args;
    args << scriptPath
         << QStringLiteral("-f1") << featurePath0
         << QStringLiteral("-f2") << featurePath1
         << QStringLiteral("-o") << matchPath
         << QStringLiteral("--threshold") << QString::number(threshold, 'g', 6)
         << QStringLiteral("--feature-algorithm") << featureAlgorithm
         << QStringLiteral("--match-algorithm") << matchAlgorithm;
    if (useCuda)
    {
        args << QStringLiteral("--cuda");
    }

    QProcess process;
    process.setWorkingDirectory(QFileInfo(scriptPath).absolutePath());
    process.start(pythonExecutable(), args);
    if (!process.waitForStarted(30000))
    {
        if (error) *error = QStringLiteral("无法启动 Python LightGlue: %1").arg(process.errorString());
        return false;
    }
    if (!process.waitForFinished(600000))
    {
        process.kill();
        process.waitForFinished(5000);
        if (error) *error = QStringLiteral("Python LightGlue 超时: %1").arg(processOutputSummary(process));
        return false;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
    {
        if (error)
        {
            const QString details = processOutputSummary(process);
            *error = details.isEmpty()
                ? QStringLiteral("Python LightGlue 退出码: %1").arg(process.exitCode())
                : QStringLiteral("Python LightGlue 退出码: %1\n%2").arg(process.exitCode()).arg(details);
        }
        return false;
    }

    QString image0Name;
    QString image1Name;
    if (!SuperGlueMatchIO::read(matchPath, image0Name, image1Name, *matchResult))
    {
        if (error) *error = QStringLiteral("Python LightGlue 已退出但无法读取匹配文件: %1").arg(matchPath);
        return false;
    }
    LOG_INFO(QStringLiteral("  Python LightGlue 输出: %1 对匹配 (%2)")
        .arg(matchResult->numMatches)
        .arg(pythonExecutable()));
    matchResult->sourceAlgorithm = matchAlgorithm.toStdString();
    return true;
}

float pythonLightGlueFallbackThreshold(const QString &featureAlgorithm, float presetThreshold)
{
    if (featureAlgorithm == QStringLiteral("disk") || featureAlgorithm == QStringLiteral("aliked"))
    {
        return 0.0f;
    }
    return presetThreshold;
}

bool allowPythonLightGlueFallback()
{
    const QString value = qEnvironmentVariable("PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK")
        .trimmed()
        .toLower();
    return value == QStringLiteral("1")
        || value == QStringLiteral("true")
        || value == QStringLiteral("yes")
        || value == QStringLiteral("on");
}

xjw::feature_extractors::FeatureData withHalfTurnRotatedKeypoints(
    const xjw::feature_extractors::FeatureData &input)
{
    xjw::feature_extractors::FeatureData rotated = input;
    if (rotated.imageWidth <= 0 || rotated.imageHeight <= 0)
    {
        return rotated;
    }

    const float maxX = static_cast<float>(rotated.imageWidth - 1);
    const float maxY = static_cast<float>(rotated.imageHeight - 1);
    for (cv::KeyPoint &keypoint : rotated.keypoints)
    {
        keypoint.pt.x = maxX - keypoint.pt.x;
        keypoint.pt.y = maxY - keypoint.pt.y;
    }
    return rotated;
}

xjw::feature_extractors::FeatureData limitedFeatureData(
    const xjw::feature_extractors::FeatureData &input,
    int maxKeypoints)
{
    if (maxKeypoints <= 0 || input.size() <= maxKeypoints)
    {
        return input;
    }

    xjw::feature_extractors::FeatureData limited = input;
    const int safeLimit = std::max(0, maxKeypoints);
    limited.keypoints.resize(static_cast<std::size_t>(safeLimit));
    if (static_cast<int>(limited.scores.size()) > safeLimit)
    {
        limited.scores.resize(static_cast<std::size_t>(safeLimit));
    }
    if (!limited.descriptors.empty() && limited.descriptors.rows > safeLimit)
    {
        limited.descriptors = limited.descriptors.rowRange(0, safeLimit).clone();
    }
    return limited;
}

bool shouldRunLightGlueHalfTurnRetry(const QString &featureAlgorithm,
                                     const QString &matchAlgorithm)
{
    return matchAlgorithm == QStringLiteral("lightglue")
        && (featureAlgorithm == QStringLiteral("disk")
            || featureAlgorithm == QStringLiteral("aliked"));
}

QJsonObject readJsonObjectFile(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return QJsonObject();
    }
    return QJsonDocument::fromJson(file.readAll()).object();
}

float matchScoreForDMatch(const xjw::feature_match::MatchResult &result, const cv::DMatch &match)
{
    if (match.queryIdx >= 0 &&
        match.queryIdx < static_cast<int>(result.matchingScores0.size()))
    {
        const float score = result.matchingScores0[static_cast<std::size_t>(match.queryIdx)];
        if (std::isfinite(score) && score > 0.0f)
        {
            return std::max(0.0f, std::min(1.0f, score));
        }
    }

    if (std::isfinite(match.distance))
    {
        return std::max(0.0f, std::min(1.0f, 1.0f / (1.0f + std::max(0.0f, match.distance))));
    }
    return 1.0f;
}

xjw::feature_match::MatchResult runTraditionalSiftFallback(
    const xjw::feature_extractors::FeatureData &fdA,
    const xjw::feature_extractors::FeatureData &fdB,
    const std::vector<cv::KeyPoint> &keypointsA,
    const std::vector<cv::KeyPoint> &keypointsB,
    const superglue::OutlierFilterConfig &outlierCfg,
    bool useCuda,
    int cudaDevice,
    int *rawMatchCount)
{
    xjw::feature_match::tradition::TraditionalMatchConfig traditionalCfg;
    traditionalCfg.algorithmName = "sift_bf_l2";
    traditionalCfg.ratioTestThreshold = 0.75f;
    traditionalCfg.requireMutualConsistency = true;
    traditionalCfg.useCuda = useCuda;
    traditionalCfg.cudaDevice = cudaDevice;

    xjw::feature_match::MatchResult fallback =
        xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
            fdA.toCvDescriptors(traditionalCfg.algorithmName),
            fdB.toCvDescriptors(traditionalCfg.algorithmName),
            fdA.size(),
            fdB.size(),
            traditionalCfg);

    if (rawMatchCount)
    {
        *rawMatchCount = fallback.numMatches;
    }

    int inlierCount = fallback.numMatches;
    return superglue::MatchOutlierRejector::filter(
        fallback, keypointsA, keypointsB, outlierCfg, &inlierCount);
}

// ── 精度等级参数预设 ─────────────────────────────────────────────────────────

struct QualityPresets
{
    // SFM
    int   initMinNumMatches;
    int   initMinNumInliers;
    int   localBAInterval;
    int   globalBAInterval;
    // 深度特征
    float featureDetectionThreshold;   ///< 检测阈值，越低特征点越多
    int   featureMaxKeypoints;         ///< 最大关键点数，<=0 不限制
    int   featureNmsRadius;            ///< NMS 半径（像素），越小特征越密
    int   featureRemoveBorders;        ///< 边界移除宽度（像素）
    int   featureNeighborhoodRadius;   ///< 邻域黑边检测半径，0=禁用
    float featureNeighborhoodThresh;   ///< 邻域检测灰度阈值
    int   featureMaxImageDim;          ///< 输入图像最长边限制，<=0 不缩放
    // LightGlue
    float matchThreshold;              ///< 匹配置信度阈值，越低匹配越多
    int   reservedMatcherIterations;   ///< 兼容旧预设表，LightGlue 当前不使用
    // 粗差剔除
    double outlierReprojThresh;   ///< RANSAC 重投影误差阈值（像素）
    int    minInliers;            ///< 粗差剔除后最少内点数；低于此阈值则舍弃该影像对
};

QualityPresets presetsForLevel(int quality)
{
    // 注意：initMinNumMatches / initMinNumInliers 不宜过高——
    // 自动流水线使用估算内参（非精确标定），相对定向内点率会偏低。
    // 阈值过高会导致完全无法初始化。
    //
    // featureMaxImageDim 控制最长边上限：提取器内部按需缩放并映射回原图。
    //
    //                           SFM                              Feature                                                      LightGlue          Outlier
    //                   minMatch inl  lBA gBA   thresh  maxKp nms brd  nbR  nbT   maxDim  sgThr  sink  reproj  minInl
    switch (quality)
    {
    case 0: return {       15,    5,  5,  15,   0.005f, 2048,  3,  4,   3,  0.05f, 1200,  0.20f, 100,  4.0,    15  };   // 低精度 — 快速
    case 1: return {       20,    5,  4,  12,   0.003f, 4096,  2,  3,   2,  0.03f, 1600,  0.15f, 120,  3.0,    20  };   // 中精度
    case 2: return {       25,    5,  3,  10,   0.002f, 8192,  2,  2,   0,  0.00f, 2000,  0.10f, 150,  2.5,    20  };   // 高精度
    case 3:
    default: return {      30,    5,  3,  10,   0.001f,   -1,  2,  2,   0,  0.00f,    0,  0.20f, 200,  2.0,    25  };   // 最高精度 — 不缩放
    }
}

int safeDefaultFeatureMaxImageDim(const QString &featureAlgorithm)
{
    if (featureAlgorithm == QStringLiteral("disk"))
    {
        return 4096;
    }
    if (featureAlgorithm == QStringLiteral("aliked"))
    {
        return 2048;
    }
    return 2048;
}

int resolveFeatureMaxImageDim(const SFMServiceOptions &opts,
                              const QualityPresets &presets,
                              const QString &featureAlgorithm)
{
    if (opts.featureMaxImageDim < 0)
    {
        return 0;
    }
    if (opts.featureMaxImageDim > 0)
    {
        return opts.featureMaxImageDim;
    }

    const int safeMax = safeDefaultFeatureMaxImageDim(featureAlgorithm);
    if (presets.featureMaxImageDim <= 0)
    {
        return 0;
    }
    if ((featureAlgorithm == QStringLiteral("disk") ||
         featureAlgorithm == QStringLiteral("aliked")) &&
        presets.featureMaxImageDim > safeMax)
    {
        return safeMax;
    }
    return presets.featureMaxImageDim;
}

QString featureMaxImageDimLabel(int maxImageDim)
{
    return maxImageDim > 0 ? QStringLiteral("%1 px").arg(maxImageDim)
                           : QStringLiteral("原始尺寸");
}

bool isCudaOutOfMemoryError(const QString &message)
{
    const QString lower = message.toLower();
    return lower.contains(QStringLiteral("cuda out of memory"))
        || lower.contains(QStringLiteral("cuda error: out of memory"))
        || (lower.contains(QStringLiteral("cuda")) && lower.contains(QStringLiteral("out of memory")));
}

bool isCudaOutOfMemoryError(const std::exception &error)
{
    return isCudaOutOfMemoryError(QString::fromUtf8(error.what()));
}

void clearTorchCudaCache()
{
#if defined(PLASCAN_TORCH_HAS_CUDA)
    if (!torch::cuda::is_available())
    {
        return;
    }
    try
    {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    catch (...)
    {
        // 清理缓存只是 OOM 后的辅助动作，失败时不要遮蔽原始错误。
    }
#else
    // CPU-only LibTorch does not ship c10 CUDA allocator symbols.
#endif
}

int alignFeatureRetryDim(int value)
{
    if (value <= 0)
    {
        return 0;
    }
    return std::max(256, (value / 16) * 16);
}

QVector<int> adaptiveFeatureMaxImageDims(int initialMaxImageDim,
                                         int imageMaxSide,
                                         const QString &featureAlgorithm)
{
    QVector<int> dims;
    auto appendDim = [&dims](int value) {
        value = alignFeatureRetryDim(value);
        if (!dims.contains(value))
        {
            dims.append(value);
        }
    };

    appendDim(initialMaxImageDim);
    if (imageMaxSide <= 0)
    {
        return dims;
    }

    const int currentEffectiveSide = initialMaxImageDim > 0
        ? std::min(alignFeatureRetryDim(initialMaxImageDim), imageMaxSide)
        : imageMaxSide;
    const QVector<int> candidates = featureAlgorithm == QStringLiteral("aliked")
        ? QVector<int>{2048, 1792, 1600, 1536, 1280, 1200, 1024, 960}
        : QVector<int>{4096, 3584, 3072, 2560, 2048, 1792, 1600, 1536, 1280, 1200, 1024};

    for (const int candidate : candidates)
    {
        if (candidate < currentEffectiveSide)
        {
            appendDim(candidate);
        }
    }

    return dims;
}

FeatureOutput extractFeatureWithAdaptiveRetry(const QString &featureAlgorithm,
                                              const QString &imageName,
                                              const cv::Mat &image,
                                              const QString &cpuExtractorModelPath,
                                              ExtractorConfig *extractorCfg,
                                              std::unique_ptr<IExtractor> *extractor)
{
    if (!extractorCfg || !extractor)
    {
        throw std::invalid_argument("extractFeatureWithAdaptiveRetry received null state");
    }

    auto recreateExtractor = [&]() {
        *extractor = xjw::feature_extractors::createExtractor(featureAlgorithm.toStdString(), *extractorCfg);
    };

    if (!*extractor)
    {
        recreateExtractor();
    }

    const QVector<int> retryDims = adaptiveFeatureMaxImageDims(extractorCfg->maxImageDim,
                                                               std::max(image.cols, image.rows),
                                                               featureAlgorithm);
    QString lastCudaOom;
    for (int attempt = 0; attempt < retryDims.size(); ++attempt)
    {
        const int retryMaxImageDim = retryDims.at(attempt);
        if (extractorCfg->maxImageDim != retryMaxImageDim || !*extractor)
        {
            extractorCfg->maxImageDim = retryMaxImageDim;
            recreateExtractor();
        }

        try
        {
            return (*extractor)->extract(image);
        }
        catch (const c10::Error &error)
        {
            if (!(extractorCfg->useCuda && isCudaOutOfMemoryError(error)))
            {
                throw;
            }
            lastCudaOom = QString::fromUtf8(error.what());
        }
        catch (const std::exception &error)
        {
            if (!(extractorCfg->useCuda && isCudaOutOfMemoryError(error)))
            {
                throw;
            }
            lastCudaOom = QString::fromUtf8(error.what());
        }

        clearTorchCudaCache();
        if (attempt + 1 < retryDims.size())
        {
            LOG_WARN(QStringLiteral("  %1 CUDA OOM: %2 使用 %3 失败，自动降到 %4 重试")
                .arg(featureAlgorithm.toUpper(),
                     imageName,
                     featureMaxImageDimLabel(retryMaxImageDim),
                     featureMaxImageDimLabel(retryDims.at(attempt + 1))));
            continue;
        }
    }

    if (extractorCfg->useCuda && !cpuExtractorModelPath.isEmpty())
    {
        LOG_WARN(QStringLiteral("  %1 CUDA OOM: %2 GPU 自适应降级仍失败，切换 CPU 模型重试")
            .arg(featureAlgorithm.toUpper(), imageName));
        extractorCfg->useCuda = false;
        extractorCfg->modelPath = cpuExtractorModelPath.toStdString();
        if (extractorCfg->maxImageDim <= 0)
        {
            extractorCfg->maxImageDim = safeDefaultFeatureMaxImageDim(featureAlgorithm);
        }
        *extractor = nullptr;
        recreateExtractor();
        return (*extractor)->extract(image);
    }

    throw std::runtime_error(lastCudaOom.isEmpty()
        ? "feature extraction failed after adaptive retry"
        : lastCudaOom.toStdString());
}

// ── 内部数据结构 ─────────────────────────────────────────────────────────────

/// 单张影像的缓存特征数据（含描述子，Phase 2 匹配需要）
struct ImageFeatureCache
{
    FeatureOutput featureOutput;     ///< keypoints, scores, descriptors
    int imgH = 0;                    ///< 图像高度（LightGlue 位置编码需要）
    int imgW = 0;                    ///< 图像宽度
};

/// 一对影像的匹配数据
struct PairMatchData
{
    ImageId idA = 0;
    ImageId idB = 0;
    std::vector<FeatureMatch> matches;
    bool loaded = false;
    bool skippedByNoMatchCache = false;
};

struct GuidedRematchExecutionStats
{
    int plannedPairCount = 0;
    int attemptedPairCount = 0;
    int invalidGeometryPairCount = 0;
    int generatedMatchCount = 0;
    int addedMatchCount = 0;
    int skippedExistingMatchCount = 0;
    int skippedInvalidMatchCount = 0;
    bool secondPassAttempted = false;
    bool secondPassAccepted = false;

    QJsonObject toJson() const
    {
        QJsonObject object;
        object[QStringLiteral("guided_matching_planned_pair_count")] = plannedPairCount;
        object[QStringLiteral("guided_matching_attempted_pair_count")] = attemptedPairCount;
        object[QStringLiteral("guided_matching_invalid_geometry_pair_count")] = invalidGeometryPairCount;
        object[QStringLiteral("guided_matching_generated_match_count")] = generatedMatchCount;
        object[QStringLiteral("guided_matching_added_match_count")] = addedMatchCount;
        object[QStringLiteral("guided_matching_skipped_existing_match_count")] = skippedExistingMatchCount;
        object[QStringLiteral("guided_matching_skipped_invalid_match_count")] = skippedInvalidMatchCount;
        object[QStringLiteral("guided_matching_second_pass_attempted")] = secondPassAttempted;
        object[QStringLiteral("guided_matching_second_pass_accepted")] = secondPassAccepted;
        return object;
    }
};

QVector<int> makeSfmDiagnosticImageIds(const QVector<ImageId> &validIds)
{
    QVector<int> imageIds;
    imageIds.reserve(validIds.size());
    for (const ImageId id : validIds)
    {
        imageIds.append(static_cast<int>(id));
    }
    return imageIds;
}

QVector<SfmMatchDiagnosticPair> makeSfmDiagnosticPairs(const QVector<PairMatchData> &pairs)
{
    QVector<SfmMatchDiagnosticPair> diagnosticPairs;
    diagnosticPairs.reserve(pairs.size());
    for (const PairMatchData &pair : pairs)
    {
        SfmMatchDiagnosticPair diagnosticPair;
        diagnosticPair.imageA = static_cast<int>(pair.idA);
        diagnosticPair.imageB = static_cast<int>(pair.idB);
        diagnosticPair.matchCount = static_cast<int>(pair.matches.size());
        diagnosticPair.loaded = pair.loaded;
        diagnosticPair.skippedByNoMatchCache = pair.skippedByNoMatchCache;
        diagnosticPairs.append(diagnosticPair);
    }
    return diagnosticPairs;
}

QString formatSfmGraphSummary(const SfmMatchGraphStats &stats)
{
    return QStringLiteral("图像=%1, 边=%2, 分量=%3, 最大分量=%4, 孤立=%5, Top=%6")
        .arg(stats.nodeCount)
        .arg(stats.edgeCount)
        .arg(stats.componentCount)
        .arg(stats.largestComponentSize)
        .arg(stats.isolatedNodeCount)
        .arg(formatSfmComponentSizes(stats.componentSizes));
}

QJsonArray intVectorToJsonArray(const QVector<int> &values)
{
    QJsonArray array;
    for (const int value : values)
    {
        array.append(value);
    }
    return array;
}

QJsonObject sfmGraphStatsToJson(const SfmMatchGraphStats &stats)
{
    QJsonObject object;
    object[QStringLiteral("node_count")] = stats.nodeCount;
    object[QStringLiteral("edge_count")] = stats.edgeCount;
    object[QStringLiteral("component_count")] = stats.componentCount;
    object[QStringLiteral("largest_component_size")] = stats.largestComponentSize;
    object[QStringLiteral("isolated_node_count")] = stats.isolatedNodeCount;
    object[QStringLiteral("component_sizes")] = intVectorToJsonArray(stats.componentSizes);
    object[QStringLiteral("summary")] = formatSfmGraphSummary(stats);
    return object;
}

QStringList sfmPairPlanSourceTypes(const SfmPairPlan &pairPlan)
{
    QStringList sourceTypes;
    if (pairPlan.usedCameraOverlapPairs)
    {
        sourceTypes.append(QStringLiteral("known_camera_overlap"));
    }
    if (pairPlan.usedSpatialCameraCenters)
    {
        sourceTypes.append(QStringLiteral("known_camera_spatial_neighbors"));
    }
    if (pairPlan.knownCameraPairWindow > 0)
    {
        sourceTypes.append(QStringLiteral("sequence_window"));
    }
    if (!pairPlan.restrictPairs)
    {
        sourceTypes.append(QStringLiteral("exhaustive"));
    }
    if (sourceTypes.isEmpty())
    {
        sourceTypes.append(pairPlan.autoRestricted
            ? QStringLiteral("auto_restricted")
            : QStringLiteral("manual_restricted"));
    }
    return sourceTypes;
}

QJsonObject sfmPairPlanToJson(const SfmPairPlan &pairPlan)
{
    constexpr int kCandidateSampleLimit = 500;
    QJsonObject object;
    object[QStringLiteral("restrict_pairs")] = pairPlan.restrictPairs;
    object[QStringLiteral("auto_restricted")] = pairPlan.autoRestricted;
    object[QStringLiteral("all_pair_count")] = pairPlan.allPairCount;
    object[QStringLiteral("planned_pair_count")] =
        pairPlan.restrictPairs ? pairPlan.allowedPairKeys.size() : pairPlan.allPairCount;
    object[QStringLiteral("known_camera_pair_window")] = pairPlan.knownCameraPairWindow;
    object[QStringLiteral("known_camera_spatial_neighbor_count")] = pairPlan.knownCameraSpatialNeighborCount;
    object[QStringLiteral("known_camera_overlap_pair_count")] = pairPlan.knownCameraOverlapPairCount;
    object[QStringLiteral("used_camera_overlap_pairs")] = pairPlan.usedCameraOverlapPairs;
    object[QStringLiteral("used_spatial_camera_centers")] = pairPlan.usedSpatialCameraCenters;
    object[QStringLiteral("source_types")] = QJsonArray::fromStringList(sfmPairPlanSourceTypes(pairPlan));
    object[QStringLiteral("candidate_count")] = static_cast<int>(pairPlan.pairCandidates.size());
    object[QStringLiteral("candidate_sample_limit")] = kCandidateSampleLimit;
    object[QStringLiteral("candidate_samples_truncated")] =
        static_cast<int>(pairPlan.pairCandidates.size()) > kCandidateSampleLimit;

    QJsonArray candidateSamples;
    QJsonObject sourceTypeCounts;
    for (const SfmPairCandidate &candidate : pairPlan.pairCandidates)
    {
        for (const QString &sourceType : candidate.sourceTypes)
        {
            if (!sourceType.isEmpty())
            {
                sourceTypeCounts[sourceType] = sourceTypeCounts.value(sourceType).toInt() + 1;
            }
        }

        if (candidateSamples.size() >= kCandidateSampleLimit)
        {
            continue;
        }

        QJsonObject candidateObject;
        candidateObject[QStringLiteral("pair_key")] = candidate.pairKey;
        candidateObject[QStringLiteral("image_index_a")] = candidate.indexA;
        candidateObject[QStringLiteral("image_index_b")] = candidate.indexB;
        candidateObject[QStringLiteral("source_types")] = QJsonArray::fromStringList(candidate.sourceTypes);
        candidateObject[QStringLiteral("priority_score")] = candidate.priorityScore;
        candidateObject[QStringLiteral("overlap_score")] = candidate.overlapScore;
        candidateObject[QStringLiteral("sequence_score")] = candidate.sequenceScore;
        candidateObject[QStringLiteral("spatial_score")] = candidate.spatialScore;
        candidateObject[QStringLiteral("baseline_score")] = candidate.baselineScore;
        candidateObject[QStringLiteral("orientation_score")] = candidate.orientationScore;
        if (candidate.sequenceDistance > 0)
        {
            candidateObject[QStringLiteral("sequence_distance")] = candidate.sequenceDistance;
        }
        if (candidate.centerDistance >= 0.0)
        {
            candidateObject[QStringLiteral("center_distance")] = candidate.centerDistance;
        }
        if (candidate.orientationAngleDeg >= 0.0)
        {
            candidateObject[QStringLiteral("orientation_angle_deg")] = candidate.orientationAngleDeg;
        }
        candidateSamples.append(candidateObject);
    }
    object[QStringLiteral("source_type_counts")] = sourceTypeCounts;
    object[QStringLiteral("candidate_samples")] = candidateSamples;
    return object;
}

QHash<QString, SfmPairCandidate> sfmPairCandidateByKey(const SfmPairPlan &pairPlan)
{
    QHash<QString, SfmPairCandidate> candidates;
    candidates.reserve(static_cast<int>(pairPlan.pairCandidates.size()));
    for (const SfmPairCandidate &candidate : pairPlan.pairCandidates)
    {
        if (!candidate.pairKey.isEmpty())
        {
            candidates.insert(candidate.pairKey, candidate);
        }
    }
    return candidates;
}

QString sfmPairStatus(const PairMatchData &pair, const QSet<QString> &failedPairKeys)
{
    const QString pairKey = QString::number(pair.idA) + QStringLiteral("\n") + QString::number(pair.idB);
    if (failedPairKeys.contains(pairKey))
    {
        return QStringLiteral("failed_geometric_verification");
    }
    if (pair.skippedByNoMatchCache)
    {
        return QStringLiteral("skipped_no_match_cache");
    }
    if (!pair.loaded)
    {
        return QStringLiteral("pending");
    }
    if (pair.matches.empty())
    {
        return QStringLiteral("empty_loaded_match");
    }
    return QStringLiteral("matched");
}

QJsonObject sfmPairToJson(const PairMatchData &pair,
                          const QMap<ImageId, QString> &idToPath,
                          const QSet<QString> &failedPairKeys,
                          const QStringList &fallbackSourceTypes,
                          const SfmPairCandidate *candidate)
{
    QJsonObject object;
    const QString imageA = idToPath.value(pair.idA);
    const QString imageB = idToPath.value(pair.idB);
    object[QStringLiteral("image_a")] = imageA;
    object[QStringLiteral("image_b")] = imageB;
    object[QStringLiteral("pair_key")] = canonicalPairKey(imageA, imageB);
    const QStringList sourceTypes =
        (candidate && !candidate->sourceTypes.isEmpty()) ? candidate->sourceTypes : fallbackSourceTypes;
    object[QStringLiteral("source_types")] = QJsonArray::fromStringList(sourceTypes);
    if (candidate)
    {
        object[QStringLiteral("priority_score")] = candidate->priorityScore;
        object[QStringLiteral("overlap_score")] = candidate->overlapScore;
        object[QStringLiteral("sequence_score")] = candidate->sequenceScore;
        object[QStringLiteral("spatial_score")] = candidate->spatialScore;
        object[QStringLiteral("baseline_score")] = candidate->baselineScore;
        object[QStringLiteral("orientation_score")] = candidate->orientationScore;
        if (candidate->sequenceDistance > 0)
        {
            object[QStringLiteral("sequence_distance")] = candidate->sequenceDistance;
        }
        if (candidate->centerDistance >= 0.0)
        {
            object[QStringLiteral("center_distance")] = candidate->centerDistance;
        }
        if (candidate->orientationAngleDeg >= 0.0)
        {
            object[QStringLiteral("orientation_angle_deg")] = candidate->orientationAngleDeg;
        }
    }
    object[QStringLiteral("status")] = sfmPairStatus(pair, failedPairKeys);
    object[QStringLiteral("match_count")] = static_cast<int>(pair.matches.size());
    object[QStringLiteral("geometric_inlier_count")] = static_cast<int>(pair.matches.size());
    object[QStringLiteral("loaded")] = pair.loaded;
    object[QStringLiteral("skipped_by_no_match_cache")] = pair.skippedByNoMatchCache;
    if (object.value(QStringLiteral("status")).toString() == QLatin1String("failed_geometric_verification"))
    {
        object[QStringLiteral("failure_reason")] = QStringLiteral("geometric_inliers_below_threshold");
    }
    else if (pair.skippedByNoMatchCache)
    {
        object[QStringLiteral("failure_reason")] = QStringLiteral("cached_no_match_pair");
    }
    else if (!pair.loaded)
    {
        object[QStringLiteral("failure_reason")] = QStringLiteral("match_file_missing_or_not_generated");
    }
    else if (pair.matches.empty())
    {
        object[QStringLiteral("failure_reason")] = QStringLiteral("loaded_match_file_empty");
    }
    return object;
}

QJsonObject buildSfmPairDiagnosticsJson(const QString &label,
                                        const QVector<ImageId> &validIds,
                                        const QVector<PairMatchData> &pairs,
                                        const SfmPairPlan &pairPlan,
                                        const QMap<ImageId, QString> &idToPath,
                                        const QVector<FailedPairRecord> &failedPairs)
{
    constexpr int kPairSampleLimit = 500;
    QSet<QString> failedPairKeysByPath;
    for (const FailedPairRecord &failedPair : failedPairs)
    {
        const QString pairKey = canonicalPairKey(failedPair.imagePath0, failedPair.imagePath1);
        if (!pairKey.isEmpty())
        {
            failedPairKeysByPath.insert(pairKey);
        }
    }

    QSet<QString> failedPairKeysById;
    for (const PairMatchData &pair : pairs)
    {
        const QString pathKey = canonicalPairKey(idToPath.value(pair.idA), idToPath.value(pair.idB));
        if (failedPairKeysByPath.contains(pathKey))
        {
            failedPairKeysById.insert(QString::number(pair.idA) + QStringLiteral("\n") + QString::number(pair.idB));
        }
    }

    const SfmMatchDiagnostics diagnostics =
        analyzeSfmMatchDiagnostics(makeSfmDiagnosticImageIds(validIds), makeSfmDiagnosticPairs(pairs));

    QJsonObject object;
    object[QStringLiteral("label")] = label;
    object[QStringLiteral("pair_plan")] = sfmPairPlanToJson(pairPlan);
    object[QStringLiteral("total_pairs")] = diagnostics.totalPairs;
    object[QStringLiteral("actual_match_pairs")] = diagnostics.actualMatchPairs;
    object[QStringLiteral("no_match_cache_skipped_pairs")] = diagnostics.noMatchCacheSkippedPairs;
    object[QStringLiteral("pending_pairs")] = diagnostics.pendingPairs;
    object[QStringLiteral("empty_loaded_pairs")] = diagnostics.emptyLoadedPairs;
    object[QStringLiteral("failed_pairs")] = failedPairs.size();
    object[QStringLiteral("candidate_graph")] = sfmGraphStatsToJson(diagnostics.candidateGraph);
    object[QStringLiteral("actual_match_graph")] = sfmGraphStatsToJson(diagnostics.actualMatchGraph);

    QJsonArray pairSamples;
    QJsonArray failedPairSamples;
    QJsonArray pendingPairSamples;
    QJsonArray skippedPairSamples;
    const QStringList sourceTypes = sfmPairPlanSourceTypes(pairPlan);
    const QHash<QString, SfmPairCandidate> candidatesByKey = sfmPairCandidateByKey(pairPlan);
    for (const PairMatchData &pair : pairs)
    {
        const QString pairKey = canonicalPairKey(idToPath.value(pair.idA), idToPath.value(pair.idB));
        const auto candidateIt = candidatesByKey.constFind(pairKey);
        const SfmPairCandidate *candidate =
            candidateIt == candidatesByKey.constEnd() ? nullptr : &candidateIt.value();
        const QJsonObject pairObject =
            sfmPairToJson(pair, idToPath, failedPairKeysById, sourceTypes, candidate);
        const QString status = pairObject.value(QStringLiteral("status")).toString();
        if (pairSamples.size() < kPairSampleLimit)
        {
            pairSamples.append(pairObject);
        }
        if (status == QLatin1String("failed_geometric_verification") && failedPairSamples.size() < kPairSampleLimit)
        {
            failedPairSamples.append(pairObject);
        }
        else if (status == QLatin1String("pending") && pendingPairSamples.size() < kPairSampleLimit)
        {
            pendingPairSamples.append(pairObject);
        }
        else if (status == QLatin1String("skipped_no_match_cache") && skippedPairSamples.size() < kPairSampleLimit)
        {
            skippedPairSamples.append(pairObject);
        }
    }
    object[QStringLiteral("pair_samples")] = pairSamples;
    object[QStringLiteral("failed_pair_samples")] = failedPairSamples;
    object[QStringLiteral("pending_pair_samples")] = pendingPairSamples;
    object[QStringLiteral("skipped_pair_samples")] = skippedPairSamples;
    object[QStringLiteral("pair_sample_limit")] = kPairSampleLimit;
    object[QStringLiteral("pair_samples_truncated")] = pairs.size() > kPairSampleLimit;
    return object;
}

QJsonObject sfmGuidedMatchPlanToJson(const SfmGuidedMatchPlan &plan,
                                     const QMap<ImageId, QString> &idToPath,
                                     bool enabled)
{
    constexpr int kCandidateSampleLimit = 500;
    QJsonObject object;
    object[QStringLiteral("guided_matching_enabled")] = enabled;
    object[QStringLiteral("guided_match_candidate_count")] = plan.candidates.size();
    object[QStringLiteral("seed_pair_count")] = plan.seedPairCount;
    object[QStringLiteral("skipped_healthy_pairs")] = plan.skippedHealthyPairs;
    object[QStringLiteral("skipped_unregistered_pairs")] = plan.skippedUnregisteredPairs;
    object[QStringLiteral("candidate_sample_limit")] = kCandidateSampleLimit;
    object[QStringLiteral("candidate_samples_truncated")] = plan.candidates.size() > kCandidateSampleLimit;

    QJsonArray samples;
    for (const SfmGuidedMatchCandidate &candidate : plan.candidates)
    {
        if (samples.size() >= kCandidateSampleLimit)
        {
            break;
        }

        QJsonObject row;
        row[QStringLiteral("image_a")] = idToPath.value(static_cast<ImageId>(candidate.imageA));
        row[QStringLiteral("image_b")] = idToPath.value(static_cast<ImageId>(candidate.imageB));
        row[QStringLiteral("image_index_a")] = candidate.imageA;
        row[QStringLiteral("image_index_b")] = candidate.imageB;
        row[QStringLiteral("match_count")] = candidate.matchCount;
        row[QStringLiteral("reason")] = candidate.reason;
        row[QStringLiteral("priority_score")] = candidate.priorityScore;
        row[QStringLiteral("can_use_epipolar_band")] = candidate.canUseEpipolarBand;
        samples.append(row);
    }
    object[QStringLiteral("candidate_samples")] = samples;
    return object;
}

QString csvEscape(const QString &value)
{
    QString escaped = value;
    escaped.replace(QStringLiteral("\""), QStringLiteral("\"\""));
    if (escaped.contains(QLatin1Char(',')) ||
        escaped.contains(QLatin1Char('\n')) ||
        escaped.contains(QLatin1Char('\r')) ||
        escaped.contains(QLatin1Char('"')))
    {
        return QStringLiteral("\"%1\"").arg(escaped);
    }
    return escaped;
}

QStringList jsonStringArrayToStringList(const QJsonArray &array)
{
    QStringList result;
    for (const QJsonValue &value : array)
    {
        const QString text = value.toString();
        if (!text.isEmpty())
        {
            result.append(text);
        }
    }
    return result;
}

QJsonObject writeSfmMatchingQualityReports(const QString &assetsDir,
                                           const QJsonObject &diagnostics,
                                           const QVector<PairMatchData> &pairs,
                                           const SfmPairPlan &pairPlan,
                                           const QMap<ImageId, QString> &idToPath,
                                           const QVector<FailedPairRecord> &failedPairs)
{
    QJsonObject result;
    if (assetsDir.isEmpty() || diagnostics.isEmpty())
    {
        return result;
    }

    const QString reportsDir = QDir(assetsDir).filePath(QStringLiteral("reports"));
    QDir().mkpath(reportsDir);
    const QString jsonPath = QDir(reportsDir).filePath(QStringLiteral("matching_quality_report.json"));
    const QString csvPath = QDir(reportsDir).filePath(QStringLiteral("matching_quality_report.csv"));

    QJsonObject jsonReport = diagnostics;
    jsonReport[QStringLiteral("report_type")] = QStringLiteral("matching_quality_report");
    jsonReport[QStringLiteral("csv_path")] = csvPath;
    jsonReport[QStringLiteral("json_path")] = jsonPath;

    QFile jsonFile(jsonPath);
    if (jsonFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        jsonFile.write(QJsonDocument(jsonReport).toJson(QJsonDocument::Indented));
        jsonFile.close();
        result[QStringLiteral("json_path")] = jsonPath;
    }
    else
    {
        LOG_WARN(QStringLiteral("SFM: 写出匹配质量 JSON 失败 → %1").arg(jsonPath));
    }

    QSet<QString> failedPairKeysByPath;
    for (const FailedPairRecord &failedPair : failedPairs)
    {
        const QString pairKey = canonicalPairKey(failedPair.imagePath0, failedPair.imagePath1);
        if (!pairKey.isEmpty())
        {
            failedPairKeysByPath.insert(pairKey);
        }
    }

    QSet<QString> failedPairKeysById;
    for (const PairMatchData &pair : pairs)
    {
        const QString pathKey = canonicalPairKey(idToPath.value(pair.idA), idToPath.value(pair.idB));
        if (failedPairKeysByPath.contains(pathKey))
        {
            failedPairKeysById.insert(QString::number(pair.idA) + QStringLiteral("\n") + QString::number(pair.idB));
        }
    }

    const QHash<QString, SfmPairCandidate> candidatesByKey = sfmPairCandidateByKey(pairPlan);
    const QStringList fallbackSourceTypes = sfmPairPlanSourceTypes(pairPlan);

    QFile csvFile(csvPath);
    if (csvFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        QTextStream stream(&csvFile);
        stream.setEncoding(QStringConverter::Utf8);
        stream << "pair_key,image_a,image_b,status,match_count,geometric_inlier_count,loaded,"
                  "skipped_by_no_match_cache,source_types,priority_score,overlap_score,sequence_score,"
                  "spatial_score,baseline_score,orientation_score,sequence_distance,center_distance,"
                  "orientation_angle_deg,failure_reason\n";

        for (const PairMatchData &pair : pairs)
        {
            const QString imageA = idToPath.value(pair.idA);
            const QString imageB = idToPath.value(pair.idB);
            const QString pairKey = canonicalPairKey(imageA, imageB);
            const auto candidateIt = candidatesByKey.constFind(pairKey);
            const SfmPairCandidate *candidate =
                candidateIt == candidatesByKey.constEnd() ? nullptr : &candidateIt.value();
            const QJsonObject row =
                sfmPairToJson(pair, idToPath, failedPairKeysById, fallbackSourceTypes, candidate);
            const QStringList sourceTypes =
                jsonStringArrayToStringList(row.value(QStringLiteral("source_types")).toArray());

            stream << csvEscape(row.value(QStringLiteral("pair_key")).toString()) << ','
                   << csvEscape(row.value(QStringLiteral("image_a")).toString()) << ','
                   << csvEscape(row.value(QStringLiteral("image_b")).toString()) << ','
                   << csvEscape(row.value(QStringLiteral("status")).toString()) << ','
                   << row.value(QStringLiteral("match_count")).toInt() << ','
                   << row.value(QStringLiteral("geometric_inlier_count")).toInt() << ','
                   << (row.value(QStringLiteral("loaded")).toBool() ? 1 : 0) << ','
                   << (row.value(QStringLiteral("skipped_by_no_match_cache")).toBool() ? 1 : 0) << ','
                   << csvEscape(sourceTypes.join(QStringLiteral("|"))) << ','
                   << row.value(QStringLiteral("priority_score")).toDouble(0.0) << ','
                   << row.value(QStringLiteral("overlap_score")).toDouble(0.0) << ','
                   << row.value(QStringLiteral("sequence_score")).toDouble(0.0) << ','
                   << row.value(QStringLiteral("spatial_score")).toDouble(0.0) << ','
                   << row.value(QStringLiteral("baseline_score")).toDouble(0.0) << ','
                   << row.value(QStringLiteral("orientation_score")).toDouble(0.0) << ','
                   << row.value(QStringLiteral("sequence_distance")).toInt(0) << ','
                   << row.value(QStringLiteral("center_distance")).toDouble(-1.0) << ','
                   << row.value(QStringLiteral("orientation_angle_deg")).toDouble(-1.0) << ','
                   << csvEscape(row.value(QStringLiteral("failure_reason")).toString()) << '\n';
        }
        csvFile.close();
        result[QStringLiteral("csv_path")] = csvPath;
    }
    else
    {
        LOG_WARN(QStringLiteral("SFM: 写出匹配质量 CSV 失败 → %1").arg(csvPath));
    }

    return result;
}

double computeTrackMaxTriangulationAngleDeg(const SfmReconstruction &reconstruction,
                                            const ScenePoint3D &point)
{
    double maxAngle = 0.0;
    const auto &observations = point.track.elements;
    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        if (!reconstruction.hasCamera(observations[i].imageId))
        {
            continue;
        }
        const auto centerI = reconstruction.camera(observations[i].imageId).cameraCenter();
        for (std::size_t j = i + 1; j < observations.size(); ++j)
        {
            if (!reconstruction.hasCamera(observations[j].imageId))
            {
                continue;
            }
            const auto centerJ = reconstruction.camera(observations[j].imageId).cameraCenter();
            const double rayI[3] = {
                point.xyz[0] - centerI[0],
                point.xyz[1] - centerI[1],
                point.xyz[2] - centerI[2],
            };
            const double rayJ[3] = {
                point.xyz[0] - centerJ[0],
                point.xyz[1] - centerJ[1],
                point.xyz[2] - centerJ[2],
            };
            const double lenI = std::sqrt(rayI[0] * rayI[0] + rayI[1] * rayI[1] + rayI[2] * rayI[2]);
            const double lenJ = std::sqrt(rayJ[0] * rayJ[0] + rayJ[1] * rayJ[1] + rayJ[2] * rayJ[2]);
            if (lenI <= 1e-9 || lenJ <= 1e-9)
            {
                continue;
            }
            double cosAngle = (rayI[0] * rayJ[0] + rayI[1] * rayJ[1] + rayI[2] * rayJ[2]) / (lenI * lenJ);
            cosAngle = std::clamp(cosAngle, -1.0, 1.0);
            maxAngle = std::max(maxAngle, std::acos(cosAngle) * 180.0 / M_PI);
        }
    }
    return maxAngle;
}

QSize inferSfmQualityImageSize(const SfmReconstruction &reconstruction, const QStringList &imagePaths)
{
    for (const QString &imagePath : imagePaths)
    {
        QImageReader reader(imagePath);
        const QSize size = reader.size();
        if (size.isValid() && size.width() > 0 && size.height() > 0)
        {
            return size;
        }
    }

    double maxX = 0.0;
    double maxY = 0.0;
    for (ImageId imageId : reconstruction.allImageIds())
    {
        if (!reconstruction.hasImage(imageId))
        {
            continue;
        }

        const ImageData &image = reconstruction.image(imageId);
        for (const FeatureKeypoint &keypoint : image.keypoints)
        {
            maxX = std::max(maxX, static_cast<double>(keypoint.x));
            maxY = std::max(maxY, static_cast<double>(keypoint.y));
        }
    }

    if (maxX > 0.0 && maxY > 0.0)
    {
        return QSize(static_cast<int>(std::ceil(maxX + 1.0)), static_cast<int>(std::ceil(maxY + 1.0)));
    }
    return QSize();
}

void logSfmMatchDiagnostics(const QString &label,
                            const QVector<ImageId> &validIds,
                            const QVector<PairMatchData> &pairs)
{
    const SfmMatchDiagnostics diagnostics =
        analyzeSfmMatchDiagnostics(makeSfmDiagnosticImageIds(validIds), makeSfmDiagnosticPairs(pairs));

    LOG_INFO(QStringLiteral("  %1配对诊断: 候选=%2, 实际有效匹配=%3, no_match无匹配=%4, 待生成=%5, 空匹配缓存=%6")
        .arg(label)
        .arg(diagnostics.totalPairs)
        .arg(diagnostics.actualMatchPairs)
        .arg(diagnostics.noMatchCacheSkippedPairs)
        .arg(diagnostics.pendingPairs)
        .arg(diagnostics.emptyLoadedPairs));
    LOG_INFO(QStringLiteral("  %1候选图: %2")
        .arg(label, formatSfmGraphSummary(diagnostics.candidateGraph)));
    LOG_INFO(QStringLiteral("  %1实际匹配图: %2")
        .arg(label, formatSfmGraphSummary(diagnostics.actualMatchGraph)));

    if (diagnostics.actualMatchGraph.nodeCount > 0 &&
        diagnostics.actualMatchGraph.componentCount > 1)
    {
        const double largestRatio =
            100.0 * static_cast<double>(diagnostics.actualMatchGraph.largestComponentSize) /
            static_cast<double>(diagnostics.actualMatchGraph.nodeCount);
        LOG_WARN(QStringLiteral("  %1实际匹配图不连通: 最大分量 %2/%3 (%4%)；增量 SfM 只能从初始分量向外注册，"
                                "请检查 no_match_pairs.json、匹配阈值或重叠配对结果")
            .arg(label)
            .arg(diagnostics.actualMatchGraph.largestComponentSize)
            .arg(diagnostics.actualMatchGraph.nodeCount)
            .arg(largestRatio, 0, 'f', 1));
    }
}

bool resolveIndexedSidecarOrder(const QJsonObject &sidecar,
                                const QString &imagePathA,
                                const QString &imagePathB,
                                const QString &featurePathA,
                                const QString &featurePathB,
                                bool *direct)
{
    auto matchesOrder = [](const QString &side0,
                           const QString &side1,
                           const QString &currentA,
                           const QString &currentB,
                           bool *directOut) -> bool
    {
        if (side0.isEmpty() || side1.isEmpty() || currentA.isEmpty() || currentB.isEmpty())
        {
            return false;
        }

        const QString norm0 = normalizePath(side0);
        const QString norm1 = normalizePath(side1);
        const QString normA = normalizePath(currentA);
        const QString normB = normalizePath(currentB);
        if (norm0 == normA && norm1 == normB)
        {
            if (directOut)
            {
                *directOut = true;
            }
            return true;
        }
        if (norm0 == normB && norm1 == normA)
        {
            if (directOut)
            {
                *directOut = false;
            }
            return true;
        }
        return false;
    };

    const QString image0 = sidecar.value(QStringLiteral("image0_path")).toString().trimmed();
    const QString image1 = sidecar.value(QStringLiteral("image1_path")).toString().trimmed();
    if (matchesOrder(image0, image1, imagePathA, imagePathB, direct))
    {
        return true;
    }

    QString feature0 = sidecar.value(QStringLiteral("feature0_path")).toString().trimmed();
    QString feature1 = sidecar.value(QStringLiteral("feature1_path")).toString().trimmed();
    if (feature0.isEmpty())
    {
        feature0 = sidecar.value(QStringLiteral("sp0_path")).toString().trimmed();
    }
    if (feature1.isEmpty())
    {
        feature1 = sidecar.value(QStringLiteral("sp1_path")).toString().trimmed();
    }
    return matchesOrder(feature0, feature1, featurePathA, featurePathB, direct);
}

bool appendIndexedMatchesFromSidecar(const QJsonObject &sidecar,
                                     bool direct,
                                     const ImageFeatureCache &fcA,
                                     const ImageFeatureCache &fcB,
                                     std::vector<FeatureMatch> *matches)
{
    if (!matches)
    {
        return false;
    }

    const QJsonArray indices0 = sidecar.value(QStringLiteral("matched_indices0")).toArray();
    const QJsonArray indices1 = sidecar.value(QStringLiteral("matched_indices1")).toArray();
    if (sidecar.value(QStringLiteral("feature_format_version")).toInt(0) < 2 ||
        indices0.isEmpty() ||
        indices0.size() != indices1.size())
    {
        return false;
    }

    const QJsonArray scores = sidecar.value(QStringLiteral("matched_scores")).toArray();
    const std::size_t originalSize = matches->size();
    matches->reserve(originalSize + static_cast<std::size_t>(indices0.size()));
    for (int i = 0; i < indices0.size(); ++i)
    {
        const int idx0 = indices0.at(i).toInt(-1);
        const int idx1 = indices1.at(i).toInt(-1);
        const int mappedA = direct ? idx0 : idx1;
        const int mappedB = direct ? idx1 : idx0;
        if (mappedA < 0 ||
            mappedB < 0 ||
            mappedA >= static_cast<int>(fcA.featureOutput.keypoints.size()) ||
            mappedB >= static_cast<int>(fcB.featureOutput.keypoints.size()))
        {
            continue;
        }

        FeatureMatch fm;
        fm.idx1 = static_cast<FeatureIdx>(mappedA);
        fm.idx2 = static_cast<FeatureIdx>(mappedB);
        if (i < scores.size())
        {
            const double score = scores.at(i).toDouble(1.0);
            if (std::isfinite(score))
            {
                fm.score = static_cast<float>(std::clamp(score, 0.0, 1.0));
            }
        }
        matches->push_back(fm);
    }
    return matches->size() > originalSize;
}

QString guidedPairKey(ImageId imageA, ImageId imageB)
{
    const ImageId a = std::min(imageA, imageB);
    const ImageId b = std::max(imageA, imageB);
    return QStringLiteral("%1:%2").arg(static_cast<qulonglong>(a)).arg(static_cast<qulonglong>(b));
}

std::vector<cv::Point2f> guidedKeypointPoints(const std::vector<cv::KeyPoint> &keypoints)
{
    std::vector<cv::Point2f> points;
    points.reserve(keypoints.size());
    for (const cv::KeyPoint &keypoint : keypoints)
    {
        points.push_back(keypoint.pt);
    }
    return points;
}

std::vector<std::pair<int, int>> guidedExistingMatches(const PairMatchData &pair,
                                                       const ImageFeatureCache &featureA,
                                                       const ImageFeatureCache &featureB)
{
    std::vector<std::pair<int, int>> existingMatches;
    existingMatches.reserve(pair.matches.size());
    const int countA = static_cast<int>(featureA.featureOutput.keypoints.size());
    const int countB = static_cast<int>(featureB.featureOutput.keypoints.size());

    for (const FeatureMatch &match : pair.matches)
    {
        if (match.idx1 == kInvalidFeatureIdx || match.idx2 == kInvalidFeatureIdx)
        {
            continue;
        }

        const int queryIndex = static_cast<int>(match.idx1);
        const int trainIndex = static_cast<int>(match.idx2);
        if (queryIndex < 0 || trainIndex < 0 || queryIndex >= countA || trainIndex >= countB)
        {
            continue;
        }
        existingMatches.emplace_back(queryIndex, trainIndex);
    }
    return existingMatches;
}

cv::Mat estimateFundamentalFromExistingMatches(const PairMatchData &pair,
                                               const ImageFeatureCache &featureA,
                                               const ImageFeatureCache &featureB)
{
    std::vector<cv::Point2f> pointsA;
    std::vector<cv::Point2f> pointsB;
    pointsA.reserve(pair.matches.size());
    pointsB.reserve(pair.matches.size());

    const int countA = static_cast<int>(featureA.featureOutput.keypoints.size());
    const int countB = static_cast<int>(featureB.featureOutput.keypoints.size());
    for (const FeatureMatch &match : pair.matches)
    {
        if (match.idx1 == kInvalidFeatureIdx || match.idx2 == kInvalidFeatureIdx)
        {
            continue;
        }

        const int queryIndex = static_cast<int>(match.idx1);
        const int trainIndex = static_cast<int>(match.idx2);
        if (queryIndex < 0 || trainIndex < 0 || queryIndex >= countA || trainIndex >= countB)
        {
            continue;
        }

        pointsA.push_back(featureA.featureOutput.keypoints[static_cast<std::size_t>(queryIndex)].pt);
        pointsB.push_back(featureB.featureOutput.keypoints[static_cast<std::size_t>(trainIndex)].pt);
    }

    if (pointsA.size() < 8 || pointsB.size() < 8)
    {
        return cv::Mat();
    }

    cv::Mat inlierMask;
    cv::Mat fundamental = cv::findFundamentalMat(pointsA, pointsB, cv::FM_RANSAC, 2.0, 0.99, inlierMask);
    if (fundamental.empty() || fundamental.rows != 3 || fundamental.cols != 3)
    {
        return cv::Mat();
    }

    cv::Mat fundamental64;
    fundamental.convertTo(fundamental64, CV_64F);
    return fundamental64;
}

cv::Matx33d cameraIntrinsicMatrix(const Camera &camera)
{
    return cv::Matx33d(camera.uAxisSign() * camera.focalX(), 0.0, camera.principalX(),
                       0.0, camera.vAxisSign() * camera.focalY(), camera.principalY(),
                       0.0, 0.0, 1.0);
}

cv::Matx33d worldToCameraMatrix(const Camera &camera)
{
    const auto rotation = camera.worldToCameraRotation();
    return cv::Matx33d(rotation[0], rotation[1], rotation[2],
                       rotation[3], rotation[4], rotation[5],
                       rotation[6], rotation[7], rotation[8]);
}

cv::Vec3d cameraCenterVector(const Camera &camera)
{
    const auto center = camera.cameraCenter();
    return cv::Vec3d(center[0], center[1], center[2]);
}

cv::Matx33d skewSymmetric(const cv::Vec3d &vector)
{
    return cv::Matx33d(0.0, -vector[2], vector[1],
                       vector[2], 0.0, -vector[0],
                       -vector[1], vector[0], 0.0);
}

cv::Mat fundamentalFromRegisteredCameras(const Camera &cameraA,
                                         const Camera &cameraB)
{
    if (!cameraA.isValid() || !cameraB.isValid() ||
        std::abs(cameraA.focalX()) <= 1e-9 ||
        std::abs(cameraA.focalY()) <= 1e-9 ||
        std::abs(cameraB.focalX()) <= 1e-9 ||
        std::abs(cameraB.focalY()) <= 1e-9)
    {
        return cv::Mat();
    }

    const cv::Matx33d rotationA = worldToCameraMatrix(cameraA);
    const cv::Matx33d rotationB = worldToCameraMatrix(cameraB);
    const cv::Vec3d centerA = cameraCenterVector(cameraA);
    const cv::Vec3d centerB = cameraCenterVector(cameraB);
    const cv::Vec3d translationAB = rotationB * (centerA - centerB);
    const double baseline = cv::norm(translationAB);
    if (baseline <= 1e-9)
    {
        return cv::Mat();
    }

    const cv::Matx33d relativeRotation = rotationB * rotationA.t();
    const cv::Matx33d essential = skewSymmetric(translationAB) * relativeRotation;
    const cv::Matx33d intrinsicA = cameraIntrinsicMatrix(cameraA);
    const cv::Matx33d intrinsicB = cameraIntrinsicMatrix(cameraB);
    const cv::Matx33d fundamental = intrinsicB.inv().t() * essential * intrinsicA.inv();

    double frobeniusNorm = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            frobeniusNorm += fundamental(row, col) * fundamental(row, col);
        }
    }
    frobeniusNorm = std::sqrt(frobeniusNorm);
    if (frobeniusNorm <= 1e-12)
    {
        return cv::Mat();
    }

    cv::Mat matrix(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            matrix.at<double>(row, col) = fundamental(row, col) / frobeniusNorm;
        }
    }
    return matrix;
}

cv::Mat guidedDescriptorsFromFeatureOutput(const FeatureOutput &featureOutput,
                                           const QString &featureAlgorithm)
{
    const xjw::feature_extractors::FeatureData featureData =
        xjw::feature_extractors::FeatureData::fromFeatureOutput(featureOutput,
                                                                featureAlgorithm.toStdString());
    return featureData.descriptors;
}

GuidedRematchExecutionStats appendGuidedRematchCandidatesToPairs(
    const SfmGuidedMatchPlan &guidedPlan,
    QVector<PairMatchData> *pairs,
    const QMap<ImageId, ImageFeatureCache> &featureCache,
    const SfmReconstruction &reconstruction,
    const QString &featureAlgorithm,
    int targetInlierCount)
{
    GuidedRematchExecutionStats stats;
    stats.plannedPairCount = guidedPlan.candidates.size();
    if (!pairs || guidedPlan.candidates.isEmpty())
    {
        return stats;
    }

    QHash<QString, int> pairIndexByKey;
    for (int index = 0; index < pairs->size(); ++index)
    {
        const PairMatchData &pair = pairs->at(index);
        pairIndexByKey.insert(guidedPairKey(pair.idA, pair.idB), index);
    }

    GuidedRematchOptions options;
    options.minOverlapScore = 0.0;
    options.targetInlierCount = std::max(1, targetInlierCount);
    options.epipolarBandPx = 2.0;
    options.maxMatches = 200;

    for (const SfmGuidedMatchCandidate &candidate : guidedPlan.candidates)
    {
        const ImageId imageA = static_cast<ImageId>(candidate.imageA);
        const ImageId imageB = static_cast<ImageId>(candidate.imageB);
        auto pairIndexIt = pairIndexByKey.constFind(guidedPairKey(imageA, imageB));
        if (pairIndexIt == pairIndexByKey.constEnd())
        {
            PairMatchData newPair;
            newPair.idA = imageA;
            newPair.idB = imageB;
            pairIndexByKey.insert(guidedPairKey(imageA, imageB), pairs->size());
            pairs->push_back(std::move(newPair));
            pairIndexIt = pairIndexByKey.constFind(guidedPairKey(imageA, imageB));
        }

        PairMatchData &pair = (*pairs)[pairIndexIt.value()];
        if (!reconstruction.isRegistered(pair.idA) || !reconstruction.isRegistered(pair.idB))
        {
            continue;
        }

        auto featureAIt = featureCache.constFind(pair.idA);
        auto featureBIt = featureCache.constFind(pair.idB);
        if (featureAIt == featureCache.constEnd() || featureBIt == featureCache.constEnd())
        {
            continue;
        }

        cv::Mat fundamental = estimateFundamentalFromExistingMatches(pair, featureAIt.value(), featureBIt.value());
        if (fundamental.empty())
        {
            fundamental = fundamentalFromRegisteredCameras(reconstruction.camera(pair.idA),
                                                          reconstruction.camera(pair.idB));
        }
        if (fundamental.empty())
        {
            ++stats.invalidGeometryPairCount;
            continue;
        }

        GuidedRematchInput input;
        input.pair.hasRegisteredCameraA = true;
        input.pair.hasRegisteredCameraB = true;
        input.pair.overlapScore = 1.0;
        input.pair.geometricInlierCount = static_cast<int>(pair.matches.size());
        input.pair.permanentlyRejected = false;
        input.options = options;
        input.fundamentalMatrix = fundamental;
        input.keypointsA = guidedKeypointPoints(featureAIt.value().featureOutput.keypoints);
        input.keypointsB = guidedKeypointPoints(featureBIt.value().featureOutput.keypoints);
        input.descriptorsA = guidedDescriptorsFromFeatureOutput(featureAIt.value().featureOutput, featureAlgorithm);
        input.descriptorsB = guidedDescriptorsFromFeatureOutput(featureBIt.value().featureOutput, featureAlgorithm);
        input.existingMatches = guidedExistingMatches(pair, featureAIt.value(), featureBIt.value());

        GuidedRematchResult guidedResult = generateGuidedRematchCandidates(input);
        for (GuidedRematchMatch &guidedMatch : guidedResult.matches)
        {
            guidedMatch.replacesExistingMatch = false;
        }

        if (!guidedResult.executed)
        {
            continue;
        }

        ++stats.attemptedPairCount;
        stats.generatedMatchCount += static_cast<int>(guidedResult.matches.size());

        const GuidedRematchMergeResult mergeResult =
            mergeGuidedRematchMatches(pair.matches, guidedResult);
        pair.matches = mergeResult.matches;
        pair.loaded = !pair.matches.empty();
        pair.skippedByNoMatchCache = false;

        stats.addedMatchCount += mergeResult.addedMatchCount;
        stats.skippedExistingMatchCount += mergeResult.skippedExistingMatchCount;
        stats.skippedInvalidMatchCount += mergeResult.skippedInvalidMatchCount;
    }

    return stats;
}

} // anonymous namespace

// ══════════════════════════════════════════════════════════════════════════════
// SFMService::run  — 一站式增量 SFM 全自动主入口（同步阻塞）
// ══════════════════════════════════════════════════════════════════════════════
SFMServiceResult runSingleSfmAttempt(const SFMServiceOptions &opts)
{
    SFMServiceResult result;
    QElapsedTimer elapsedTimer;
    elapsedTimer.start();

    auto reportProgress = [&](const QString &stage, int percent)
    {
        if (opts.progressFn)
        {
            opts.progressFn(stage, percent);
        }
    };

    // 取消检查辅助 lambda
    auto isCancelled = [&]() -> bool
    {
        return opts.cancelFlag && opts.cancelFlag->load();
    };

    // ══════════════════════════════════════════════════════════════════════════
    // 0. 基本校验
    // ══════════════════════════════════════════════════════════════════════════
    if (opts.images.size() < 2)
    {
        result.errorMessage = QStringLiteral("至少需要选择两张影像");
        result.summary      = result.errorMessage;
        return result;
    }

    const QualityPresets presets = presetsForLevel(opts.quality);

    // ══════════════════════════════════════════════════════════════════════════
    // 1. 确定计算设备
    // ══════════════════════════════════════════════════════════════════════════
    bool useCuda = false;
    if (opts.device == QStringLiteral("cuda") ||
        (opts.device == QStringLiteral("auto") && torch::cuda::is_available()))
    {
        useCuda = true;
    }
    LOG_INFO(QStringLiteral("SFM 流水线: 使用 %1 设备, 精度等级 %2")
        .arg(useCuda ? QStringLiteral("CUDA") : QStringLiteral("CPU"))
        .arg(opts.quality));

    const QString featureAlgorithm = normalizedAlgorithm(opts.featureAlgorithm, QStringLiteral("disk"));
    const QString matchAlgorithm = normalizedAlgorithm(opts.matchAlgorithm, QStringLiteral("lightglue"));
    const bool isExistingMatchOnlyMode = !opts.autoGenerateMissingMatches;

    if (!isSfmFeatureAlgorithm(featureAlgorithm))
    {
        result.errorMessage = QStringLiteral("SFM 当前不支持该特征类型: %1")
            .arg(featureAlgorithm);
        result.summary = result.errorMessage;
        return result;
    }

    const QString featureSuffix = QString::fromLatin1(
        ExtractorSuffix::forAlgorithm(featureAlgorithm.toStdString()));
    const QStringList compatibleSuffixes = xjw::feature_match::compatibleFeatureSuffixes(matchAlgorithm);
    if (compatibleSuffixes.isEmpty())
    {
        result.errorMessage = QStringLiteral("SFM 初始化需要显式特征文件轨迹，当前不支持端到端或未知匹配算法: %1")
            .arg(matchAlgorithm);
        result.summary = result.errorMessage;
        return result;
    }
    if (!compatibleSuffixes.contains(featureSuffix))
    {
        result.errorMessage = QStringLiteral("匹配算法 %1 不兼容特征类型 %2")
            .arg(matchAlgorithm, featureSuffix);
        result.summary = result.errorMessage;
        return result;
    }

    const bool useLightGlueSfmMatch = isLightGlueSfmMatch(featureAlgorithm, matchAlgorithm);
    const bool useTraditionalSiftSfmMatch = isTraditionalSiftMatch(featureAlgorithm, matchAlgorithm);
    if (!isExistingMatchOnlyMode && !useLightGlueSfmMatch && !useTraditionalSiftSfmMatch)
    {
        result.errorMessage = QStringLiteral(
            "自动补全匹配当前支持 DISK/ALIKED/SIFT + LightGlue，或 SIFT + FLANN/BF-L2，收到: %1 + %2")
            .arg(featureAlgorithm, matchAlgorithm);
        result.summary = result.errorMessage;
        return result;
    }

    const int featureMaxImageDim = resolveFeatureMaxImageDim(opts, presets, featureAlgorithm);

    LOG_INFO(QStringLiteral("SFM 流水线算法: %1 + %2, 特征后缀=%3")
        .arg(featureAlgorithm.toUpper(), matchAlgorithm, featureSuffix));
    if (featureMaxImageDim > 0)
    {
        LOG_INFO(QStringLiteral("SFM 特征提取图像最长边上限: %1 px").arg(featureMaxImageDim));
    }
    else
    {
        LOG_INFO(QStringLiteral("SFM 特征提取图像最长边: 自动自适应（先尝试原始尺寸，CUDA OOM 时自动降低）"));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. 设置目录
    // ══════════════════════════════════════════════════════════════════════════
    const QString assetsDir = ProjectIO::projectAssetsDir(opts.plascanPath);
    const QString projectRootDir = ProjectIO::projectRootFromPlascan(opts.plascanPath);
    const QString ipDir     = QDir(assetsDir).filePath(QStringLiteral("ip"));
    const QString matchDir  = QDir(assetsDir).filePath(QStringLiteral("matches"));
    const QString outDir    = QDir::cleanPath(opts.outputDir);
    QDir().mkpath(ipDir);
    QDir().mkpath(matchDir);
    logSfmMatchCacheCatalogDiagnostics(matchDir);
    if (!outDir.isEmpty())
    {
        QDir().mkpath(outDir);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3. 分配 ImageId
    // ══════════════════════════════════════════════════════════════════════════
    QMap<QString, ImageId> imageIdMap;   // normPath → id
    QMap<ImageId, QString> idToPath;     // id → normPath
    ImageId nextId = 0;
    for (const QString &imgPath : opts.images)
    {
        const QString norm = normalizePath(imgPath);
        imageIdMap[norm] = nextId;
        idToPath[nextId] = norm;
        ++nextId;
    }
    const int N = static_cast<int>(opts.images.size());

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1: 确保所有影像的特征文件存在
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("检查特征文件..."), 2);
    LOG_INFO(QStringLiteral("SFM Phase 1: 检查 / 提取特征..."));

    QMap<ImageId, ImageFeatureCache> featureCache;
    QMap<ImageId, QString> featureFilePaths;     // id → 当前算法特征文件路径
    QVector<ImageId> missingFeatureIds;

    // 1a. 查找已有的当前算法特征文件
    for (auto it = imageIdMap.constBegin(); it != imageIdMap.constEnd(); ++it) {
        const QString &imgPath = it.key();
        const ImageId id       = it.value();
        const QString featurePath = ProjectIO::featureFileForSuffix(opts.plascanPath, imgPath, featureSuffix);
        if (!featurePath.isEmpty() &&
            QString::fromStdString(FeatureFileIO::peekAlgorithm(featurePath)) == featureAlgorithm)
        {
            featureFilePaths[id] = featurePath;
        }
        else
        {
            missingFeatureIds.append(id);
        }
    }

    LOG_INFO(QStringLiteral("  已有特征: %1/%2, 需提取: %3")
        .arg(featureFilePaths.size()).arg(N).arg(missingFeatureIds.size()));

    // 1b. 提取缺失的当前算法特征
    if (!missingFeatureIds.isEmpty())
    {
        LOG_INFO(QStringLiteral("  启动 %1 特征提取...").arg(featureAlgorithm.toUpper()));

        QString pickedExtractorModelName;
        const bool featureNeedsModel = sfmFeatureNeedsModel(featureAlgorithm);
        const QString extractorModelPath = findFirstModelFile(
            featureModelCandidates(featureAlgorithm, useCuda),
            &pickedExtractorModelName);
        if (featureNeedsModel && extractorModelPath.isEmpty())
        {
            result.errorMessage = QStringLiteral("未找到 %1 模型文件: %2")
                .arg(featureAlgorithm.toUpper(),
                     featureModelCandidates(featureAlgorithm, useCuda).join(QStringLiteral(", ")));
            result.summary      = result.errorMessage;
            return result;
        }
        if (featureNeedsModel)
        {
            LOG_INFO(QStringLiteral("  特征提取模型: %1").arg(pickedExtractorModelName));
        }
        else
        {
            if (featureAlgorithm == QStringLiteral("sift") && useCuda)
            {
                LOG_INFO(QStringLiteral("  SIFT 使用 OpenCV 标准特征提取器；CUDA 仅用于后续匹配阶段"));
            }
            else
            {
                LOG_INFO(QStringLiteral("  %1 使用 OpenCV 传统特征提取器，无需模型文件")
                    .arg(featureAlgorithm.toUpper()));
            }
        }

        try
        {
            QString cpuExtractorModelPath;
            if (useCuda)
            {
                cpuExtractorModelPath = findFirstModelFile(featureModelCandidates(featureAlgorithm, false));
            }

            ExtractorConfig extractorCfg;
            extractorCfg.modelPath     = extractorModelPath.toStdString();
            extractorCfg.maxKeypoints  = presets.featureMaxKeypoints;
            extractorCfg.detThreshold  = presets.featureDetectionThreshold;
            extractorCfg.nmsRadius     = presets.featureNmsRadius;
            extractorCfg.removeBorder  = presets.featureRemoveBorders;
            extractorCfg.maxImageDim   = featureMaxImageDim;
            extractorCfg.grayscaleMin  = opts.featureGrayscaleMin;
            extractorCfg.grayscaleMax  = opts.featureGrayscaleMax;
            extractorCfg.useCuda       = useCuda && featureAlgorithm != QStringLiteral("sift");
            extractorCfg.cudaDevice    = 0;

            std::unique_ptr<IExtractor> extractor =
                xjw::feature_extractors::createExtractor(featureAlgorithm.toStdString(), extractorCfg);

            int featureDoneCount = 0;
            const int featureTotalCount = static_cast<int>(missingFeatureIds.size());
            for (const ImageId id : missingFeatureIds)
            {
                // 取消检查
                if (isCancelled())
                {
                    result.errorMessage = QStringLiteral("用户取消");
                    result.summary = result.errorMessage;
                    return result;
                }

                // 报告特征提取进度（2%~35% 区间映射）
                if (shouldReportIndexedProgress(featureDoneCount + 1, featureTotalCount))
                {
                    int pct = 2 + (featureDoneCount * 33) / std::max(1, featureTotalCount);
                    reportProgress(QStringLiteral("正在查找特征点... %1/%2")
                        .arg(featureDoneCount + 1).arg(featureTotalCount), pct);
                }

                const QString &imgPath = idToPath[id];
                const QFileInfo fi(imgPath);

                cv::Mat image = cv::imread(imgPath.toStdString(), cv::IMREAD_GRAYSCALE);
                if (image.empty()) 
                {
                    LOG_WARN(QStringLiteral("  无法读取图像: %1").arg(fi.fileName()));
                    continue;
                }

                FeatureOutput featureOut = extractFeatureWithAdaptiveRetry(featureAlgorithm,
                                                                           fi.fileName(),
                                                                           image,
                                                                           cpuExtractorModelPath,
                                                                           &extractorCfg,
                                                                           &extractor);

                const QString featurePath = QDir(ipDir).filePath(fi.completeBaseName() + featureSuffix);
                if (!FeatureFileIO::write(featurePath, fi.fileName(), featureOut,
                                          featureAlgorithm.toStdString()))
                {
                    LOG_WARN(QStringLiteral("  保存特征文件失败: %1").arg(featurePath));
                    continue;
                }

                LOG_INFO(QStringLiteral("  提取 %1 个特征点: %2")
                    .arg(featureOut.keypoints.size()).arg(fi.fileName()));

                featureFilePaths[id] = featurePath;

                // 缓存特征及图像尺寸（Phase 2 匹配可直接使用）
                ImageFeatureCache &fc = featureCache[id];
                fc.featureOutput = std::move(featureOut);
                fc.imgH     = image.rows;
                fc.imgW     = image.cols;

                // 记录新生成的文件
                result.newFeatureFiles.append({imgPath, featurePath});
                ++featureDoneCount;
            }
        }
        catch (const std::exception &e)
        {
            result.errorMessage = QStringLiteral("%1 特征提取失败: %2")
                .arg(featureAlgorithm.toUpper())
                .arg(QString::fromStdString(e.what()));
            result.summary = result.errorMessage;
            return result;
        }
    }

    // 1c. 加载已有特征文件到缓存（仅加载尚未缓存的）
    for (auto it = featureFilePaths.constBegin(); it != featureFilePaths.constEnd(); ++it)
    {
        const ImageId id     = it.key();
        const QString &featurePath = it.value();
        if (featureCache.contains(id)) continue;   // 刚提取的已在缓存中

        QString imageName;
        FeatureOutput featureOut;
        if (!FeatureFileIO::read(featurePath, imageName, featureOut))
        {
            LOG_WARN(QStringLiteral("  读取特征文件失败: %1").arg(featurePath));
            continue;
        }

        ImageFeatureCache &fc = featureCache[id];
        fc.featureOutput = std::move(featureOut);
        // 图像尺寸未知，Phase 2 匹配时按需读取
    }

    // 1d. 检查可用特征数量
    if (featureCache.size() < 2) 
    {
        result.errorMessage = QStringLiteral("有效影像不足 2 张（仅 %1 张有特征）")
            .arg(featureCache.size());
        result.summary = result.errorMessage;
        return result;
    }

    // 1e. 取消检查
    if (isCancelled())
    {
        result.errorMessage = QStringLiteral("用户取消");
        result.summary = result.errorMessage;
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2: 确保所有影像对的匹配结果存在
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("检查匹配文件..."), 35);
    LOG_INFO(QStringLiteral("SFM Phase 2: 检查 / 生成匹配..."));

    // 有效 Id 列表（只保留有特征的）
    QVector<ImageId> validIds;
    for (auto it = featureCache.constBegin(); it != featureCache.constEnd(); ++it)
        validIds.append(it.key());
    std::sort(validIds.begin(), validIds.end());

    // ── 加载 no_match_pairs.json：上次已确认无匹配的影像对 ──────────────────────
    // key: "normPath0|normPath1"（两方向均存），value 未使用（以文件 mtime 比较代替）
    QSet<QString> noMatchSet;
    const QString noMatchFilePath = QDir(matchDir).filePath(QStringLiteral("no_match_pairs.json"));
    const QDateTime noMatchFileMtime = QFileInfo(noMatchFilePath).lastModified();
    {
        QFile nmf(noMatchFilePath);
        if (nmf.open(QIODevice::ReadOnly)) 
        {
            const QJsonArray arr = QJsonDocument::fromJson(nmf.readAll()).array();
            nmf.close();
            for (const QJsonValue &v : arr) 
            {
                const QJsonObject o = v.toObject();
                const QString recFeatureAlgorithm =
                    o.value(QStringLiteral("feature_algorithm")).toString().trimmed().toLower();
                const QString recMatchAlgorithm =
                    o.value(QStringLiteral("match_algorithm")).toString().trimmed().toLower();
                if (recFeatureAlgorithm != featureAlgorithm || recMatchAlgorithm != matchAlgorithm)
                {
                    continue;
                }
                const QString p0 = normalizePath(o.value(QStringLiteral("image0")).toString());
                const QString p1 = normalizePath(o.value(QStringLiteral("image1")).toString());
                if (!p0.isEmpty() && !p1.isEmpty()) 
                {
                    noMatchSet.insert(p0 + QStringLiteral("|") + p1);
                    noMatchSet.insert(p1 + QStringLiteral("|") + p0);
                }
            }
            LOG_INFO(QStringLiteral("  已加载 no_match_pairs.json: %1 条无匹配记录")
                .arg(noMatchSet.size() / 2));
            if (noMatchFileMtime.isValid())
            {
                LOG_INFO(QStringLiteral("  no_match_pairs.json 修改时间: %1")
                    .arg(noMatchFileMtime.toString(Qt::ISODateWithMs)));
            }
        }
    }

    SfmPairPlannerOptions pairPlanOptions;
    pairPlanOptions.restrictPairs = opts.restrictPairs;
    pairPlanOptions.allowedPairs = opts.allowedPairs;
    pairPlanOptions.autoRestrictKnownCameraPairs = opts.autoRestrictKnownCameraPairs;
    pairPlanOptions.knownCameraPairWindow = opts.knownCameraPairWindow;
    pairPlanOptions.knownCameraSpatialNeighborCount = opts.knownCameraSpatialNeighborCount;
    pairPlanOptions.knownCameraAllPairsMaxImages = opts.knownCameraAllPairsMaxImages;
    int projectMetaCameraCenterCount = 0;
    const std::vector<std::array<double, 3>> projectMetaCameraCenters =
        loadKnownCameraCentersFromProjectMeta(opts.images, opts.projectMeta, &projectMetaCameraCenterCount);
    int projectMetaViewingDirectionCount = 0;
    const std::vector<std::array<double, 3>> projectMetaViewingDirections =
        loadKnownCameraViewingDirectionsFromProjectMeta(opts.images,
                                                        opts.projectMeta,
                                                        &projectMetaViewingDirectionCount);
    const bool hasCameraPathRestrictionInputs = hasCompleteCameraPathList(opts.images, opts.cameraPaths);
    const bool hasProjectMetaCameraCenters =
        hasCompleteKnownCameraCenters(static_cast<int>(opts.images.size()), projectMetaCameraCenters);
    const bool canUseKnownCameraAutoPairs =
        !opts.restrictPairs &&
        opts.autoRestrictKnownCameraPairs &&
        opts.images.size() > std::max(0, opts.knownCameraAllPairsMaxImages) &&
        (hasCameraPathRestrictionInputs || hasProjectMetaCameraCenters);
    if (canUseKnownCameraAutoPairs && opts.useKnownCameraOverlapPairs && hasCameraPathRestrictionInputs)
    {
        QString overlapDetail;
        QString overlapError;
        pairPlanOptions.knownCameraOverlapPairs =
            loadKnownCameraOverlapPairsFromPaths(opts.images,
                                                 opts.cameraPaths,
                                                 opts.knownCameraOverlapNeighborFactor,
                                                 &overlapDetail,
                                                 &overlapError);
        if (!pairPlanOptions.knownCameraOverlapPairs.empty())
        {
            LOG_INFO(QStringLiteral("  已知相机足迹重叠分析: %1")
                .arg(overlapDetail.isEmpty() ? QStringLiteral("完成") : overlapDetail));
        }
        else
        {
            LOG_WARN(QStringLiteral("  已知相机足迹重叠配对不可用，将回退顺序/中心邻域: %1")
                .arg(overlapError.isEmpty() ? QStringLiteral("没有可用重叠对") : overlapError));
        }
    }
    else if (canUseKnownCameraAutoPairs && opts.useKnownCameraOverlapPairs && hasProjectMetaCameraCenters)
    {
        LOG_INFO(QStringLiteral("  项目元数据相机中心可用于配对裁剪；未提供 .tsai 文件，跳过足迹重叠分析"));
    }
    if (!opts.restrictPairs &&
        opts.autoRestrictKnownCameraPairs &&
        opts.knownCameraSpatialNeighborCount > 0 &&
        canUseKnownCameraAutoPairs)
    {
        if (hasCameraPathRestrictionInputs)
        {
            QString knownCameraCenterError;
            QString knownCameraDirectionError;
            pairPlanOptions.knownCameraCenters = loadKnownCameraCentersFromPaths(opts.images,
                                                                                 opts.cameraPaths,
                                                                                 &knownCameraCenterError);
            pairPlanOptions.knownCameraViewingDirections =
                loadKnownCameraViewingDirectionsFromPaths(opts.images,
                                                          opts.cameraPaths,
                                                          &knownCameraDirectionError);
            if (pairPlanOptions.knownCameraCenters.empty())
            {
                LOG_WARN(QStringLiteral("  已知相机空间配对不可用，将回退顺序邻域: %1").arg(knownCameraCenterError));
            }
            if (pairPlanOptions.knownCameraViewingDirections.empty())
            {
                LOG_WARN(QStringLiteral("  已知相机视线方向评分不可用，将只使用中心距离/顺序评分: %1")
                    .arg(knownCameraDirectionError));
            }
        }
        else if (hasProjectMetaCameraCenters)
        {
            pairPlanOptions.knownCameraCenters = projectMetaCameraCenters;
            pairPlanOptions.knownCameraViewingDirections = projectMetaViewingDirections;
            LOG_INFO(QStringLiteral("  从项目元数据读取相机中心用于配对裁剪: %1/%2")
                .arg(projectMetaCameraCenterCount)
                .arg(opts.images.size()));
            if (!projectMetaViewingDirections.empty())
            {
                LOG_INFO(QStringLiteral("  从项目元数据读取相机视线方向用于配对评分: %1/%2")
                    .arg(projectMetaViewingDirectionCount)
                    .arg(opts.images.size()));
            }
        }
    }

    const SfmPairPlan pairPlan = planSfmMatchPairs(opts.images, opts.cameraPaths, pairPlanOptions);

    QSet<QString> allowedPairSet;
    for (const QString &pairKey : pairPlan.allowedPairKeys)
    {
        const QString trimmedKey = pairKey.trimmed();
        if (!trimmedKey.isEmpty())
        {
            allowedPairSet.insert(trimmedKey);
        }
    }

    if (pairPlan.restrictPairs)
    {
        LOG_INFO(QStringLiteral("  匹配对约束已启用: %1 对").arg(allowedPairSet.size()));
        if (pairPlan.autoRestricted)
        {
            if (pairPlan.usedCameraOverlapPairs)
            {
                LOG_INFO(QStringLiteral("  已知相机足迹重叠配对裁剪: 原始 %1 对 -> %2 对, 重叠候选=%3, 邻域系数=%4")
                    .arg(pairPlan.allPairCount)
                    .arg(allowedPairSet.size())
                    .arg(pairPlan.knownCameraOverlapPairCount)
                    .arg(opts.knownCameraOverlapNeighborFactor, 0, 'g', 4));
            }
            else if (pairPlan.usedSpatialCameraCenters)
            {
                if (pairPlan.knownCameraOverlapPairCount > 0)
                {
                    LOG_WARN(QStringLiteral("  已知相机足迹重叠候选过密(%1 对)，已回退为空间+顺序邻域")
                        .arg(pairPlan.knownCameraOverlapPairCount));
                }
                LOG_INFO(QStringLiteral("  已知相机空间+顺序配对裁剪: 原始 %1 对 -> %2 对, 顺序窗口=%3, 空间邻居=%4")
                    .arg(pairPlan.allPairCount)
                    .arg(allowedPairSet.size())
                    .arg(pairPlan.knownCameraPairWindow)
                    .arg(pairPlan.knownCameraSpatialNeighborCount));
            }
            else
            {
                LOG_INFO(QStringLiteral("  已知相机顺序配对裁剪: 原始 %1 对 -> %2 对, 邻域窗口=%3")
                    .arg(pairPlan.allPairCount)
                    .arg(allowedPairSet.size())
                    .arg(pairPlan.knownCameraPairWindow));
            }
        }
    }

    // 从项目元数据复用已有匹配记录（覆盖“仅按 assets/matches 规范命名扫描”的限制）
    // 适配场景：ipmatch 使用了自定义 output_dir，文件不在 assets/matches 下。
    QMap<QString, QString> metaMatchPathByPair;
    QMap<QString, QString> metaMatchPathByBasePair;
    {
        const QJsonArray ipmatchResults = ipmatchResultsFromMeta(opts.projectMeta);
        for (const QJsonValue &val : ipmatchResults)
        {
            const QJsonObject rec = val.toObject();
            const QString rawOutput = rec.value(QStringLiteral("output")).toString();
            const QString outputPath = resolvePathFromProjectRoot(projectRootDir, rawOutput);
            if (outputPath.isEmpty() || !QFile::exists(outputPath))
            {
                continue;
            }

            QString image0 = resolveImagePathTokenFromMeta(rec.value(QStringLiteral("image0")).toString(),
                                                           opts.projectMeta);
            QString image1 = resolveImagePathTokenFromMeta(rec.value(QStringLiteral("image1")).toString(),
                                                           opts.projectMeta);

            if (image0.isEmpty() || image1.isEmpty())
            {
                const QJsonArray imageFiles = rec.value(QStringLiteral("settings"))
                                                 .toObject()
                                                 .value(QStringLiteral("image_files"))
                                                 .toArray();
                if (imageFiles.size() >= 2)
                {
                    image0 = resolveImagePathTokenFromMeta(imageFiles.at(0).toString(), opts.projectMeta);
                    image1 = resolveImagePathTokenFromMeta(imageFiles.at(1).toString(), opts.projectMeta);
                }
            }

            if (image0.isEmpty() || image1.isEmpty())
            {
                const QString sidecarPath = outputPath + QStringLiteral(".json");
                QFile sidecarFile(sidecarPath);
                if (sidecarFile.open(QIODevice::ReadOnly))
                {
                    const QJsonObject sidecarObj = QJsonDocument::fromJson(sidecarFile.readAll()).object();
                    sidecarFile.close();
                    image0 = resolveImagePathTokenFromMeta(sidecarObj.value(QStringLiteral("image0_path")).toString(),
                                                           opts.projectMeta);
                    image1 = resolveImagePathTokenFromMeta(sidecarObj.value(QStringLiteral("image1_path")).toString(),
                                                           opts.projectMeta);
                }
            }

            const QString pairKey = canonicalPairKey(image0, image1);
            const QString basePairKey = canonicalNamePairKey(QFileInfo(image0).completeBaseName(),
                                                             QFileInfo(image1).completeBaseName());
            if (pairKey.isEmpty() && basePairKey.isEmpty())
            {
                continue;
            }

            auto upsertNewest = [&](QMap<QString, QString> *pathMap, const QString &key)
            {
                if (!pathMap || key.isEmpty())
                {
                    return;
                }
                const QString existingPath = pathMap->value(key);
                if (existingPath.isEmpty())
                {
                    pathMap->insert(key, outputPath);
                    return;
                }

                const QDateTime oldTime = QFileInfo(existingPath).lastModified();
                const QDateTime newTime = QFileInfo(outputPath).lastModified();
                if (newTime > oldTime)
                {
                    (*pathMap)[key] = outputPath;
                }
            };

            upsertNewest(&metaMatchPathByPair, pairKey);
            upsertNewest(&metaMatchPathByBasePair, basePairKey);
        }
        if (!metaMatchPathByPair.isEmpty())
        {
            LOG_INFO(QStringLiteral("  元数据可复用匹配: %1 对").arg(metaMatchPathByPair.size()));
        }
    }

    int metaFoundPathCount = 0;
    int metaReadOkCount = 0;
    int metaBaseHitCount = 0;

    auto existingMatchCompatible = [&](const QString &matchPath, ImageId idA, ImageId idB) -> bool
    {
        const QString sidecarPath = matchPath + QStringLiteral(".json");
        if (!QFile::exists(sidecarPath))
        {
            return false;
        }

        const QJsonObject sc = readJsonObjectFile(sidecarPath);
        if (sc.isEmpty())
        {
            return false;
        }

        QString scFeatureAlgorithm = sc.value(QStringLiteral("feature_algorithm")).toString().trimmed().toLower();
        if (scFeatureAlgorithm.isEmpty())
        {
            scFeatureAlgorithm = sc.value(QStringLiteral("settings")).toObject()
                .value(QStringLiteral("feature_algorithm")).toString().trimmed().toLower();
        }
        if (!scFeatureAlgorithm.isEmpty() && scFeatureAlgorithm != featureAlgorithm)
        {
            return false;
        }

        QString scMatchAlgorithm = sc.value(QStringLiteral("match_algorithm")).toString().trimmed().toLower();
        if (scMatchAlgorithm.isEmpty())
        {
            scMatchAlgorithm = sc.value(QStringLiteral("settings")).toObject()
                .value(QStringLiteral("match_algorithm")).toString().trimmed().toLower();
        }
        if (!scMatchAlgorithm.isEmpty() && scMatchAlgorithm != matchAlgorithm)
        {
            return false;
        }

        QString feature0 = sc.value(QStringLiteral("feature0_path")).toString().trimmed();
        QString feature1 = sc.value(QStringLiteral("feature1_path")).toString().trimmed();
        if (feature0.isEmpty())
        {
            feature0 = sc.value(QStringLiteral("sp0_path")).toString().trimmed();
        }
        if (feature1.isEmpty())
        {
            feature1 = sc.value(QStringLiteral("sp1_path")).toString().trimmed();
        }

        if (!feature0.isEmpty() || !feature1.isEmpty())
        {
            if (feature0.isEmpty() || feature1.isEmpty())
            {
                return false;
            }
            const QString sideFeature0 = normalizePath(feature0);
            const QString sideFeature1 = normalizePath(feature1);
            const QString currentA = normalizePath(featureFilePaths.value(idA));
            const QString currentB = normalizePath(featureFilePaths.value(idB));
            if (!((sideFeature0 == currentA && sideFeature1 == currentB) ||
                  (sideFeature0 == currentB && sideFeature1 == currentA)))
            {
                return false;
            }
        }
        else if (scFeatureAlgorithm.isEmpty())
        {
            return false;
        }

        const QJsonArray indices0 = sc.value(QStringLiteral("matched_indices0")).toArray();
        const QJsonArray indices1 = sc.value(QStringLiteral("matched_indices1")).toArray();
        if (sc.value(QStringLiteral("feature_format_version")).toInt(0) < 2 ||
            indices0.isEmpty() ||
            indices0.size() != indices1.size())
        {
            LOG_INFO(QStringLiteral("  匹配缓存缺少 V2 特征索引，不能用于正式 SfM track 合并: %1")
                .arg(matchPath));
            return false;
        }

        return true;
    };

    // 生成所有需要处理的影像对并检查已有匹配文件
    QVector<PairMatchData> allPairs;
    QVector<int> missingPairIndices;

    auto appendCandidatePair = [&](ImageId idA, ImageId idB)
    {
            const QString baseA = QFileInfo(idToPath[idA]).completeBaseName();
            const QString baseB = QFileInfo(idToPath[idB]).completeBaseName();

            PairMatchData pd;
            pd.idA = idA;
            pd.idB = idB;

            auto findExistingMatchCache = [&](const QString &leftBase,
                                              const QString &rightBase) -> QString
            {
                const QDir dir(matchDir);
                const QStringList patterns{
                    QStringLiteral("%1__%2.match").arg(leftBase, rightBase),
                    QStringLiteral("%1__%2*.match").arg(leftBase, rightBase)
                };

                QFileInfoList candidates;
                QSet<QString> seenPaths;
                for (const QString &pattern : patterns)
                {
                    const QFileInfoList files = dir.entryInfoList(QStringList{pattern},
                                                                  QDir::Files,
                                                                  QDir::Time);
                    for (const QFileInfo &fileInfo : files)
                    {
                        const QString path = QDir::cleanPath(fileInfo.absoluteFilePath());
                        if (!seenPaths.contains(path))
                        {
                            seenPaths.insert(path);
                            candidates.append(fileInfo);
                        }
                    }
                }

                std::sort(candidates.begin(), candidates.end(),
                          [](const QFileInfo &left, const QFileInfo &right)
                          {
                              return left.lastModified() > right.lastModified();
                          });

                for (const QFileInfo &fileInfo : candidates)
                {
                    const QString candidatePath = QDir::cleanPath(fileInfo.absoluteFilePath());
                    if (existingMatchCompatible(candidatePath, idA, idB))
                    {
                        return candidatePath;
                    }
                }
                return QString();
            };

            // 在 matches 目录下查找已有 .match 文件（两种命名顺序，兼容 A__B_lightglue.match）
            QString foundPath = findExistingMatchCache(baseA, baseB);
            if (foundPath.isEmpty())
            {
                foundPath = findExistingMatchCache(baseB, baseA);
            }

            if (foundPath.isEmpty())
            {
                const QString pairKey = canonicalPairKey(idToPath.value(idA), idToPath.value(idB));
                const QString metaPath = metaMatchPathByPair.value(pairKey);
                if (!metaPath.isEmpty() && QFile::exists(metaPath))
                {
                    foundPath = metaPath;
                    ++metaFoundPathCount;
                }
                else
                {
                    const QString basePairKey = canonicalNamePairKey(baseA, baseB);
                    const QString metaBasePath = metaMatchPathByBasePair.value(basePairKey);
                    if (!metaBasePath.isEmpty() && QFile::exists(metaBasePath))
                    {
                        foundPath = metaBasePath;
                        ++metaFoundPathCount;
                        ++metaBaseHitCount;
                    }
                }
            }

            if (!foundPath.isEmpty() && !existingMatchCompatible(foundPath, idA, idB))
            {
                LOG_INFO(QStringLiteral("  匹配缓存与当前 %1 + %2 链路不兼容，重新生成: %3")
                    .arg(featureAlgorithm.toUpper(), matchAlgorithm, foundPath));
                foundPath.clear();
            }

            // 检查 .match 文件是否过期：仅当特征文件比 .match 更新时才视为过期
            // （即特征点重新提取后，旧匹配结果失效）
            // 注意：不再检查 match_threshold，避免因参数变化引发全量重匹配
            if (!foundPath.isEmpty()) 
            {
                const QDateTime matchTime = QFileInfo(foundPath).lastModified();
                bool stale = false;
                if (featureFilePaths.contains(idA) &&
                    QFileInfo(featureFilePaths[idA]).lastModified() > matchTime)
                    stale = true;
                if (featureFilePaths.contains(idB) &&
                    QFileInfo(featureFilePaths[idB]).lastModified() > matchTime)
                    stale = true;

                if (stale)
                {
                    if (opts.autoGenerateMissingMatches)
                    {
                        LOG_INFO(QStringLiteral("  匹配缓存过期(特征已更新), 重新生成: %1")
                                                     .arg(foundPath));
                        QFile::remove(foundPath);
                        QFile::remove(foundPath + QStringLiteral(".json"));
                    }
                    else
                    {
                        LOG_WARN(QStringLiteral("  匹配缓存过期(特征已更新), 自动补匹配已禁用，跳过该对: %1")
                                                     .arg(foundPath));
                    }
                    foundPath.clear();
                }
            }

            if (!foundPath.isEmpty()) 
            {
                const QJsonObject sidecar = readJsonObjectFile(foundPath + QStringLiteral(".json"));
                auto itA = featureCache.constFind(idA);
                auto itB = featureCache.constFind(idB);
                bool direct = true;
                if (itA != featureCache.constEnd() &&
                    itB != featureCache.constEnd() &&
                    resolveIndexedSidecarOrder(sidecar,
                                                idToPath.value(idA),
                                                idToPath.value(idB),
                                                featureFilePaths.value(idA),
                                                featureFilePaths.value(idB),
                                                &direct) &&
                    appendIndexedMatchesFromSidecar(sidecar, direct, *itA, *itB, &pd.matches))
                {
                    pd.loaded = true;
                    ++metaReadOkCount;
                }
                else if (metaMatchPathByPair.contains(canonicalPairKey(idToPath.value(idA), idToPath.value(idB))))
                {
                    LOG_WARN(QStringLiteral("  元数据命中但读取匹配失败或为空: %1").arg(foundPath));
                }
            }

            // ── 无匹配记录检查：若已知该对无匹配，且 sp 文件在记录后未更新，跳过 ──
            if (!pd.loaded) 
            {
                const QString pA = normalizePath(idToPath[idA]);
                const QString pB = normalizePath(idToPath[idB]);
                const QString nmKey = pA + QStringLiteral("|") + pB;
                if (noMatchSet.contains(nmKey) && noMatchFileMtime.isValid()) 
                {
                    // 以 no_match_pairs.json 文件的修改时间为基准
                    // 若两张影像的特征文件均早于该时间 → 特征未变 → 直接跳过
                    bool featureNewer = false;
                    if (featureFilePaths.contains(idA) &&
                        QFileInfo(featureFilePaths[idA]).lastModified() > noMatchFileMtime)
                        featureNewer = true;
                    if (featureFilePaths.contains(idB) &&
                        QFileInfo(featureFilePaths[idB]).lastModified() > noMatchFileMtime)
                        featureNewer = true;
                    if (!featureNewer)
                    {
                        pd.loaded = true;   // 已知无匹配且特征未更新，跳过
                        pd.skippedByNoMatchCache = true;
                    }
                }
            }

            const int pairIdx = allPairs.size();
            allPairs.append(pd);
            if (!pd.loaded) 
            {
                missingPairIndices.append(pairIdx);
            }
    };

    if (pairPlan.restrictPairs)
    {
        QMap<QString, ImageId> validIdByPath;
        for (const ImageId id : validIds)
        {
            const QString normalizedPath = normalizePath(idToPath.value(id));
            if (!normalizedPath.isEmpty())
            {
                validIdByPath.insert(normalizedPath, id);
            }
        }

        QSet<QString> emittedPairKeys;
        for (const QString &pairKey : pairPlan.allowedPairKeys)
        {
            const QString trimmedPairKey = pairKey.trimmed();
            if (trimmedPairKey.isEmpty() ||
                !allowedPairSet.contains(trimmedPairKey) ||
                emittedPairKeys.contains(trimmedPairKey))
            {
                continue;
            }

            const QStringList pairPaths = pairKey.split(QStringLiteral("\n"));
            if (pairPaths.size() != 2)
            {
                continue;
            }

            auto itA = validIdByPath.constFind(normalizePath(pairPaths.at(0).trimmed()));
            auto itB = validIdByPath.constFind(normalizePath(pairPaths.at(1).trimmed()));
            if (itA == validIdByPath.constEnd() ||
                itB == validIdByPath.constEnd() ||
                itA.value() == itB.value())
            {
                continue;
            }

            emittedPairKeys.insert(trimmedPairKey);
            appendCandidatePair(itA.value(), itB.value());
        }
    }
    else
    {
        for (int i = 0; i < validIds.size(); ++i)
        {
            for (int j = i + 1; j < validIds.size(); ++j)
            {
                appendCandidatePair(validIds[i], validIds[j]);
            }
        }
    }

    if (allPairs.isEmpty())
    {
        result.errorMessage = pairPlan.restrictPairs
            ? QStringLiteral("所选影像中没有可用的已生成匹配对，请先创建连接点或检查 .lis 配对范围")
            : QStringLiteral("未找到可用影像对");
        result.summary = result.errorMessage;
        return result;
    }

    reportProgress(QStringLiteral("匹配候选对: %1 对，需生成 %2 对")
        .arg(allPairs.size())
        .arg(missingPairIndices.size()), 35);
    LOG_INFO(QStringLiteral("  总配对: %1, 已处理(含有效匹配/负缓存): %2, 需生成: %3")
        .arg(allPairs.size())
        .arg(allPairs.size() - missingPairIndices.size())
        .arg(missingPairIndices.size()));
    logSfmMatchDiagnostics(QStringLiteral("预检查"), validIds, allPairs);
    if (metaFoundPathCount > 0)
    {
        LOG_INFO(QStringLiteral("  元数据命中: %1 对(含基名兜底 %2), 读取成功: %3 对")
            .arg(metaFoundPathCount)
            .arg(metaBaseHitCount)
            .arg(metaReadOkCount));
    }

    // 2a. 对缺失配对执行特征匹配（可按选项禁用）
    if (!missingPairIndices.isEmpty() && opts.autoGenerateMissingMatches)
    {
        LOG_INFO(QStringLiteral("  启动 %1 特征匹配...").arg(matchAlgorithm.toUpper()));

        QString lgModelName;
        QString lgModelPath = useLightGlueSfmMatch
            ? findFirstModelFile(lightGlueModelCandidates(featureAlgorithm, useCuda), &lgModelName)
            : QString();
        const bool canUsePythonLightGlue = featureAlgorithm == QStringLiteral("disk")
                                        || featureAlgorithm == QStringLiteral("aliked")
                                        || featureAlgorithm == QStringLiteral("sift");
        QString lightGlueExportError;
        if (useLightGlueSfmMatch && lgModelPath.isEmpty() && canUsePythonLightGlue)
        {
            lgModelPath = ensureLightGlueTorchScriptModel(
                featureAlgorithm, useCuda, &lgModelName, &lightGlueExportError);
            if (!lgModelPath.isEmpty())
            {
                LOG_INFO(QStringLiteral("  自动导出后使用 LightGlue 模型: %1").arg(lgModelName));
            }
        }
        const bool usePythonLightGlue = useLightGlueSfmMatch
                                     && lgModelPath.isEmpty()
                                     && canUsePythonLightGlue
                                     && allowPythonLightGlueFallback();
        const float activeMatchThreshold = usePythonLightGlue
            ? pythonLightGlueFallbackThreshold(featureAlgorithm, presets.matchThreshold)
            : presets.matchThreshold;
        QString pythonLightGlueScript;
        if (useTraditionalSiftSfmMatch)
        {
            LOG_INFO(QStringLiteral("  SIFT 传统匹配器: %1").arg(matchAlgorithm));
        }
        else if (usePythonLightGlue)
        {
            pythonLightGlueScript = findScriptFile(QStringLiteral("run_lightglue.py"));
            if (pythonLightGlueScript.isEmpty())
            {
                result.errorMessage =
                    QStringLiteral("未找到 %1 专用 LightGlue TorchScript 模型(%2)，自动导出失败: %3；且未找到 scripts/run_lightglue.py")
                        .arg(featureAlgorithm.toUpper(),
                             lightGlueModelCandidates(featureAlgorithm, useCuda).join(QStringLiteral(", ")),
                             lightGlueExportError);
                result.summary = result.errorMessage;
                return result;
            }
            LOG_WARN(QStringLiteral("  LightGlue TorchScript 自动导出失败: %1").arg(lightGlueExportError));
            LOG_INFO(QStringLiteral("  使用 Python LightGlue: %1").arg(pythonLightGlueScript));
            LOG_INFO(QStringLiteral("  Python LightGlue 阈值: %1").arg(activeMatchThreshold, 0, 'g', 6));
        }
        else if (lgModelPath.isEmpty())
        {
            result.errorMessage = QStringLiteral("未找到 LightGlue TorchScript 模型文件: %1")
                .arg(lightGlueModelCandidates(featureAlgorithm, useCuda).join(QStringLiteral(", ")));
            if (canUsePythonLightGlue)
            {
                result.errorMessage += QStringLiteral(
                    "。自动导出失败: %1；请检查 PLASCAN_PYTHON_EXECUTABLE/PLASCAN_PYTHON 指向的环境是否包含 torch 和 lightglue。"
                    "如需临时使用 Python 逐对匹配，可设置 PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK=1")
                    .arg(lightGlueExportError);
            }
            result.summary      = result.errorMessage;
            return result;
        }
        else
        {
            LOG_INFO(QStringLiteral("  LightGlue 模型: %1").arg(lgModelName));
        }

        // ── 粗差剔除配置（USAC_MAGSAC，最优粗差剔除）──────────────────────────────────
        superglue::OutlierFilterConfig outlierCfg;
        outlierCfg.method          = superglue::OutlierMethod::FundamentalUsacMagsac;
        outlierCfg.reprojThreshold = presets.outlierReprojThresh;
        outlierCfg.confidence      = 0.9999;
        outlierCfg.maxIters        = 10000;
        outlierCfg.minInliers      = presets.minInliers;

        // ── 预加载所有影像尺寸，消除并行阶段的写竞争 ──────────────────────
        {
            QSet<ImageId> needDim;
            for (const int pi : missingPairIndices) 
            {
                needDim.insert(allPairs[pi].idA);
                needDim.insert(allPairs[pi].idB);
            }
            for (const ImageId id : needDim) 
            {
                if (!featureCache.contains(id)) continue;
                ImageFeatureCache &fc = featureCache[id];
                if (fc.imgH == 0 || fc.imgW == 0) 
                {
                    cv::Mat img = cv::imread(idToPath[id].toStdString(), cv::IMREAD_GRAYSCALE);
                    if (!img.empty()) { fc.imgH = img.rows; fc.imgW = img.cols; }
                    else              { fc.imgH = 1000;     fc.imgW = 1000;     }
                }
            }
        }

        // CUDA 模式：由用户指定并行对数（每个线程独立持有一个 Matcher 实例占用显存）
        // CPU 模式：按线程数并行
        const int numMatchThreads = useCuda
            ? std::max(1, opts.cudaParallelPairs)
            : std::max(1, opts.threads);
        LOG_INFO(QStringLiteral("  匹配线程数: %1 (%2 模式)")
            .arg(numMatchThreads)
            .arg(useCuda ? QStringLiteral("CUDA") : QStringLiteral("CPU-并行")));

        // 每个工作线程独立持有一个 LightGlueMatcher，避免模型权重并发写入问题
        xjw::feature_match::LightGlueConfig lgCfg;
        lgCfg.matcherModelPath = lgModelPath.toStdString();
        lgCfg.useCuda = useCuda;
        lgCfg.scoreThreshold = activeMatchThreshold;

        const bool use_skeleton_feature_budget =
            opts.enableTwoStageMatching &&
            opts.enableGuidedRematching &&
            useLightGlueSfmMatch &&
            opts.skeletonFeatureMaxKeypoints > 0;
        const int two_stage_skeleton_keypoint_limit =
            use_skeleton_feature_budget && presets.featureMaxKeypoints > 0
                ? std::min(presets.featureMaxKeypoints, opts.skeletonFeatureMaxKeypoints)
                : (use_skeleton_feature_budget
                       ? opts.skeletonFeatureMaxKeypoints
                       : presets.featureMaxKeypoints);
        if (use_skeleton_feature_budget)
        {
            LOG_INFO(QStringLiteral("  两阶段匹配: 特征全量提取，第一阶段只使用前 %1 个 keypoints 建骨架")
                .arg(two_stage_skeleton_keypoint_limit));
        }

        // ── 验证模型可加载（用单个测试实例） ─────────────────────────────
        if (useLightGlueSfmMatch && !usePythonLightGlue)
        {
            xjw::feature_match::LightGlueMatcher testMatcher(lgCfg);
            if (!testMatcher.isLoaded()) 
            {
                result.errorMessage = QStringLiteral("LightGlue 模型加载失败");
                result.summary      = result.errorMessage;
                return result;
            }
        }

        // ── 共享错误状态（线程间）────────────────────────────────────────
        std::atomic<bool>         matchErrorFlag{false};
        std::string               matchErrorMsg;
        std::mutex                writeMutex;       // 保护对 result 的追加操作
        std::atomic<int>          pairCursor{0};    // 原子索引，各线程竞争取下一个待处理 pair
        std::atomic<int>          lowInlierPairCount{0};
        std::atomic<int>          lowInlierPairSamples{0};
        const int                 totalMissing = static_cast<int>(missingPairIndices.size());

        // 省去重复读取：featureCache/idToPath/featureFilePaths 在预加载后为纯只读
        // 使用 constFind 进行线程安全只读访问（无写入时并发读取安全）
        auto matchWorker = [&]() 
        {
            try 
            {
                // 每线程自有 matcher 实例
                std::unique_ptr<xjw::feature_match::IFeatureMatcher> localMatcher;
                if (useLightGlueSfmMatch && !usePythonLightGlue)
                {
                    auto lgMatcher = std::make_unique<xjw::feature_match::LightGlueMatcher>(lgCfg);
                    if (!lgMatcher->isLoaded()) return;
                    localMatcher = std::move(lgMatcher);
                }

                while (!matchErrorFlag.load()) 
                {
                    // 取消检查
                    if (opts.cancelFlag && opts.cancelFlag->load()) break;

                    const int localIdx = pairCursor.fetch_add(1);
                    if (localIdx >= totalMissing) break;

                    // 报告匹配进度（35%~70% 区间映射）
                    if (shouldReportIndexedProgress(localIdx + 1, totalMissing))
                    {
                        const int doneCount = localIdx + 1;
                        int pct = 35 + (doneCount * 35) / std::max(1, totalMissing);
                        pct = std::clamp(pct, 35, 70);
                        reportProgress(QStringLiteral("正在匹配特征点... %1/%2")
                            .arg(doneCount).arg(totalMissing), pct);
                    }

                    const int pi      = missingPairIndices[localIdx];
                    PairMatchData &pd = allPairs[pi];   // pi 唯一，无写竞争
                    const ImageId idA = pd.idA;
                    const ImageId idB = pd.idB;

                    auto itA = featureCache.constFind(idA);
                    auto itB = featureCache.constFind(idB);
                    if (itA == featureCache.constEnd() || itB == featureCache.constEnd()) continue;

                    const ImageFeatureCache &fcA = *itA;
                    const ImageFeatureCache &fcB = *itB;

                    const QString baseA    = QFileInfo(idToPath.value(idA)).completeBaseName();
                    const QString baseB    = QFileInfo(idToPath.value(idB)).completeBaseName();
                    const QString pairName = QStringLiteral("%1__%2").arg(baseA, baseB);
                    const QString matchPath = QDir(matchDir).filePath(pairName + QStringLiteral(".match"));

                    auto fdA = xjw::feature_extractors::FeatureData::fromFeatureOutput(
                        fcA.featureOutput, featureAlgorithm.toStdString());
                    auto fdB = xjw::feature_extractors::FeatureData::fromFeatureOutput(
                        fcB.featureOutput, featureAlgorithm.toStdString());
                    fdA.imageWidth = fcA.imgW;
                    fdA.imageHeight = fcA.imgH;
                    fdB.imageWidth = fcB.imgW;
                    fdB.imageHeight = fcB.imgH;

                    const xjw::feature_extractors::FeatureData matchFdA =
                        use_skeleton_feature_budget
                            ? limitedFeatureData(fdA, two_stage_skeleton_keypoint_limit)
                            : fdA;
                    const xjw::feature_extractors::FeatureData matchFdB =
                        use_skeleton_feature_budget
                            ? limitedFeatureData(fdB, two_stage_skeleton_keypoint_limit)
                            : fdB;

                    // 执行匹配
                    xjw::feature_match::MatchResult mr;
                    bool usedLightGlueHalfTurnRetry = false;
                    bool usedTraditionalSiftFallback = false;
                    int primaryLightGlueInliers = -1;
                    int traditionalSiftFallbackRawMatchCount = 0;
                    if (useTraditionalSiftSfmMatch)
                    {
                        xjw::feature_match::tradition::TraditionalMatchConfig traditionalCfg;
                        traditionalCfg.algorithmName = matchAlgorithm.toStdString();
                        traditionalCfg.ratioTestThreshold = 0.75f;
                        traditionalCfg.requireMutualConsistency = true;
                        traditionalCfg.useCuda = useCuda && matchAlgorithm == QStringLiteral("sift_bf_l2");
                        traditionalCfg.cudaDevice = 0;
                        mr = xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
                            matchFdA.toCvDescriptors(matchAlgorithm.toStdString()),
                            matchFdB.toCvDescriptors(matchAlgorithm.toStdString()),
                            matchFdA.size(),
                            matchFdB.size(),
                            traditionalCfg);
                    }
                    else if (usePythonLightGlue)
                    {
                        QString pythonError;
                        if (!runPythonLightGlue(pythonLightGlueScript,
                                                featureFilePaths.value(idA),
                                                featureFilePaths.value(idB),
                                                matchPath,
                                                useCuda,
                                                activeMatchThreshold,
                                                featureAlgorithm,
                                                matchAlgorithm,
                                                &mr,
                                                &pythonError))
                        {
                            throw std::runtime_error(pythonError.toStdString());
                        }
                    }
                    else
                    {
                        mr = localMatcher->match(matchFdA, matchFdB);
                    }

                    // 粗差剔除
                    int inlierCount = mr.numMatches;
                    mr = superglue::MatchOutlierRejector::filter(
                        mr, matchFdA.keypoints, matchFdB.keypoints,
                        outlierCfg, &inlierCount);
                    primaryLightGlueInliers = mr.numMatches;

                    if (!usePythonLightGlue
                        && shouldRunLightGlueHalfTurnRetry(featureAlgorithm, matchAlgorithm)
                        && localMatcher
                        && mr.numMatches < presets.minInliers
                        && matchFdA.imageWidth > 0
                        && matchFdB.imageWidth > 0)
                    {
                        xjw::feature_match::MatchResult retryResult =
                            localMatcher->match(matchFdA, withHalfTurnRotatedKeypoints(matchFdB));
                        int retryInlierCount = retryResult.numMatches;
                        retryResult = superglue::MatchOutlierRejector::filter(
                            retryResult, matchFdA.keypoints, matchFdB.keypoints,
                            outlierCfg, &retryInlierCount);

                        if (retryResult.numMatches > mr.numMatches)
                        {
                            LOG_INFO(QStringLiteral("  %1 LightGlue 180°重试改善内点: %2 → %3")
                                .arg(pairName)
                                .arg(mr.numMatches)
                                .arg(retryResult.numMatches));
                            mr = std::move(retryResult);
                            usedLightGlueHalfTurnRetry = true;
                        }
                    }

                    if (!usePythonLightGlue
                        && useLightGlueSfmMatch
                        && featureAlgorithm == QStringLiteral("sift")
                        && mr.numMatches < presets.minInliers)
                    {
                        xjw::feature_match::MatchResult fallbackResult = runTraditionalSiftFallback(
                            matchFdA,
                            matchFdB,
                            matchFdA.keypoints,
                            matchFdB.keypoints,
                            outlierCfg,
                            useCuda,
                            0,
                            &traditionalSiftFallbackRawMatchCount);

                        if (fallbackResult.numMatches > mr.numMatches)
                        {
                            LOG_INFO(QStringLiteral("  %1 SIFT+BF-L2 fallback 改善内点: %2 → %3 (raw=%4, cuda=%5)")
                                .arg(pairName)
                                .arg(mr.numMatches)
                                .arg(fallbackResult.numMatches)
                                .arg(traditionalSiftFallbackRawMatchCount)
                                .arg(useCuda ? QStringLiteral("on") : QStringLiteral("off")));
                            mr = std::move(fallbackResult);
                            usedTraditionalSiftFallback = true;
                        }
                    }

                    // ── 最小内点数检测：内点不足时记录到 failedPairs（不写文件）──
                    if (mr.numMatches < presets.minInliers)
                    {
                        const int sampleIndex = lowInlierPairSamples.fetch_add(1);
                        if (sampleIndex < 8)
                        {
                            LOG_INFO(
                                QStringLiteral("  跳过 %1: 内点数 %2 < 阈值 %3（已记录为无匹配对）")
                                .arg(pairName).arg(mr.numMatches).arg(presets.minInliers));
                        }
                        lowInlierPairCount.fetch_add(1);
                        {
                            std::lock_guard<std::mutex> lk(writeMutex);
                            FailedPairRecord fpr;
                            fpr.imagePath0 = idToPath.value(idA);
                            fpr.imagePath1 = idToPath.value(idB);
                            result.failedPairs.append(fpr);
                        }
                        pd.loaded = true;
                        pd.skippedByNoMatchCache = true;
                        continue;
                    }

                    LOG_INFO(QStringLiteral("  匹配 %1: %2 对匹配点")
                        .arg(pairName).arg(mr.numMatches));

                    // ── 保存 .match 文件 ────────────────────────────────────
                    SuperGlueMatchIO::write(matchPath, baseA, baseB, mr);

                    // ── 保存 sidecar JSON ───────────────────────────────────
                    QJsonObject sidecar;
                    sidecar[QStringLiteral("match_file")]      = matchPath;
                    sidecar[QStringLiteral("image0_name")]     = baseA;
                    sidecar[QStringLiteral("image1_name")]     = baseB;
                    sidecar[QStringLiteral("image0_path")]     = idToPath.value(idA);
                    sidecar[QStringLiteral("image1_path")]     = idToPath.value(idB);
                    sidecar[QStringLiteral("feature0_path")]   = featureFilePaths.value(idA);
                    sidecar[QStringLiteral("feature1_path")]   = featureFilePaths.value(idB);
                    sidecar[QStringLiteral("sp0_path")]        = featureFilePaths.value(idA);
                    sidecar[QStringLiteral("sp1_path")]        = featureFilePaths.value(idB);
                    sidecar[QStringLiteral("feature_algorithm")] = featureAlgorithm;
                    sidecar[QStringLiteral("match_algorithm")] = matchAlgorithm;
                    sidecar[QStringLiteral("feature_format_version")] = 2;
                    sidecar[QStringLiteral("num_matches")]     = mr.numMatches;
                    sidecar[QStringLiteral("match_threshold")] = static_cast<double>(activeMatchThreshold);
                    if (usedLightGlueHalfTurnRetry)
                    {
                        sidecar[QStringLiteral("lightglue_rotation_retry")] = QStringLiteral("half_turn_image1");
                        sidecar[QStringLiteral("rotation_retry_degrees")] = 180;
                        sidecar[QStringLiteral("primary_inlier_count")] = primaryLightGlueInliers;
                    }
                    if (usedTraditionalSiftFallback)
                    {
                        sidecar[QStringLiteral("traditional_sift_fallback")] = true;
                        sidecar[QStringLiteral("fallback_algorithm")] = QStringLiteral("sift_bf_l2");
                        sidecar[QStringLiteral("fallback_raw_match_count")] = traditionalSiftFallbackRawMatchCount;
                        sidecar[QStringLiteral("primary_inlier_count")] = primaryLightGlueInliers;
                    }

                    QJsonArray pts0, pts1, indices0, indices1, scores;
                    for (const auto &dm : mr.cvMatches) 
                    {
                        const int qi = dm.queryIdx, ti = dm.trainIdx;
                        if (qi >= 0 && qi < static_cast<int>(fcA.featureOutput.keypoints.size()) &&
                            ti >= 0 && ti < static_cast<int>(fcB.featureOutput.keypoints.size()))
                        {
                            const auto &kp0 = fcA.featureOutput.keypoints[qi];
                            const auto &kp1 = fcB.featureOutput.keypoints[ti];
                            QJsonArray p0; p0.append(kp0.pt.x); p0.append(kp0.pt.y);
                            QJsonArray p1; p1.append(kp1.pt.x); p1.append(kp1.pt.y);
                            pts0.append(p0);
                            pts1.append(p1);
                            indices0.append(qi);
                            indices1.append(ti);
                            scores.append(static_cast<double>(matchScoreForDMatch(mr, dm)));
                        }
                    }
                    sidecar[QStringLiteral("matched_points0")] = pts0;
                    sidecar[QStringLiteral("matched_points1")] = pts1;
                    sidecar[QStringLiteral("matched_indices0")] = indices0;
                    sidecar[QStringLiteral("matched_indices1")] = indices1;
                    sidecar[QStringLiteral("matched_scores")] = scores;

                    const QString sidecarPath = matchPath + QStringLiteral(".json");
                    QFile sf(sidecarPath);
                    if (sf.open(QIODevice::WriteOnly | QIODevice::Truncate)) 
                    {
                        sf.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
                        sf.close();
                    }

                    // ── 构建匹配记录 ─────────────────────────────────────────
                    MatchFileRecord mfr;
                    mfr.pairName    = pairName;
                    mfr.matchPath   = matchPath;
                    mfr.sidecarPath = sidecarPath;

                    QJsonObject pairSettings;
                    QJsonArray imageFiles;
                    imageFiles.append(idToPath.value(idA));
                    imageFiles.append(idToPath.value(idB));
                    pairSettings[QStringLiteral("image_files")]  = imageFiles;
                    pairSettings[QStringLiteral("pair_name")]    = pairName;
                    pairSettings[QStringLiteral("sidecar_json")] = sidecarPath;
                    pairSettings[QStringLiteral("feature0_path")] = featureFilePaths.value(idA);
                    pairSettings[QStringLiteral("feature1_path")] = featureFilePaths.value(idB);
                    pairSettings[QStringLiteral("sp0_path")]     = featureFilePaths.value(idA);
                    pairSettings[QStringLiteral("sp1_path")]     = featureFilePaths.value(idB);
                    pairSettings[QStringLiteral("feature_algorithm")] = featureAlgorithm;
                    pairSettings[QStringLiteral("match_algorithm")] = matchAlgorithm;
                    if (usedLightGlueHalfTurnRetry)
                    {
                        pairSettings[QStringLiteral("lightglue_rotation_retry")] =
                            QStringLiteral("half_turn_image1");
                        pairSettings[QStringLiteral("rotation_retry_degrees")] = 180;
                    }
                    mfr.settings = pairSettings;

                    // ── 转换为 SFM 需要的 FeatureMatch ──────────────────────
                    pd.matches.reserve(mr.cvMatches.size());
                    for (const auto &dm : mr.cvMatches) {
                        FeatureMatch fm;
                        fm.idx1 = static_cast<FeatureIdx>(dm.queryIdx);
                        fm.idx2 = static_cast<FeatureIdx>(dm.trainIdx);
                        fm.score = matchScoreForDMatch(mr, dm);
                        pd.matches.push_back(fm);
                    }
                    pd.loaded = true;

                    // ── 追加到 result（需加锁）─────────────────────────────
                    {
                        std::lock_guard<std::mutex> lk(writeMutex);
                        result.newMatchFiles.append(mfr);
                    }

                    // ── 实时回调：通知 UI 本对已完成（无需加锁，调用方负责线程转发）
                    if (opts.pairMatchedFn) 
                    {
                        opts.pairMatchedFn(idToPath.value(idA), idToPath.value(idB),
                                           matchPath, mr.numMatches);
                    }
                }
            } 
            catch (const std::exception &ex) 
            {
                std::lock_guard<std::mutex> lk(writeMutex);
                if (!matchErrorFlag.exchange(true)) 
                {
                    matchErrorMsg = ex.what();
                }
            }
        };

        // ── 启动工作线程 ──────────────────────────────────────────────────
        std::vector<std::thread> workerThreads;
        workerThreads.reserve(numMatchThreads);
        for (int t = 0; t < numMatchThreads; ++t)
            workerThreads.emplace_back(matchWorker);
        for (auto &th : workerThreads) th.join();

        if (matchErrorFlag.load()) 
        {
            result.errorMessage = QStringLiteral("LightGlue 匹配失败: %1")
                .arg(QString::fromStdString(matchErrorMsg));
            result.summary = result.errorMessage;
            return result;
        }

        if (lowInlierPairCount.load() > 8)
        {
            LOG_INFO(QStringLiteral("  内点不足跳过 %1 对，日志仅显示前 8 对样例")
                .arg(lowInlierPairCount.load()));
        }

        // ── 将无匹配的影像对追加写入 no_match_pairs.json ─────────────────────
        if (!result.failedPairs.isEmpty()) 
        {
            QJsonArray existing;
            {
                QFile nmf(noMatchFilePath);
                if (nmf.open(QIODevice::ReadOnly))
                    existing = QJsonDocument::fromJson(nmf.readAll()).array();
            }
            // 构建已存在条目的集合（避免重复追加）
            QSet<QString> existingKeys;
            for (const QJsonValue &v : existing) 
            {
                const QJsonObject o = v.toObject();
                const QString recFeatureAlgorithm =
                    o.value(QStringLiteral("feature_algorithm")).toString().trimmed().toLower();
                const QString recMatchAlgorithm =
                    o.value(QStringLiteral("match_algorithm")).toString().trimmed().toLower();
                if (recFeatureAlgorithm != featureAlgorithm || recMatchAlgorithm != matchAlgorithm)
                {
                    continue;
                }
                existingKeys.insert(o[QStringLiteral("image0")].toString() +
                                    QStringLiteral("|") +
                                    o[QStringLiteral("image1")].toString());
            }
            const QString ts = QDateTime::currentDateTime().toString(Qt::ISODate);
            for (const FailedPairRecord &fpr : result.failedPairs) {
                const QString key  = fpr.imagePath0 + QStringLiteral("|") + fpr.imagePath1;
                const QString keyR = fpr.imagePath1 + QStringLiteral("|") + fpr.imagePath0;
                if (!existingKeys.contains(key) && !existingKeys.contains(keyR)) 
                {
                    QJsonObject o;
                    o[QStringLiteral("image0")]      = fpr.imagePath0;
                    o[QStringLiteral("image1")]      = fpr.imagePath1;
                    o[QStringLiteral("checked_at")]  = ts;
                    o[QStringLiteral("feature_algorithm")] = featureAlgorithm;
                    o[QStringLiteral("match_algorithm")] = matchAlgorithm;
                    existing.append(o);
                    existingKeys.insert(key);
                }
            }
            QFile nmfOut(noMatchFilePath);
            if (nmfOut.open(QIODevice::WriteOnly | QIODevice::Truncate)) 
            {
                nmfOut.write(QJsonDocument(existing).toJson(QJsonDocument::Compact));
                nmfOut.close();  // 显式关闭以确保 mtime 立即生效
            }
            LOG_INFO(QStringLiteral("  无匹配对 %1 条已记录到 %2")
                .arg(result.failedPairs.size()).arg(noMatchFilePath));
        }
    }
    else if (!missingPairIndices.isEmpty())
    {
        LOG_INFO(QStringLiteral("  自动补匹配已禁用，跳过 %1 对缺失配对").arg(missingPairIndices.size()));
    }

    logSfmMatchDiagnostics(QStringLiteral("匹配完成"), validIds, allPairs);
    result.sfmDiagnostics = buildSfmPairDiagnosticsJson(QStringLiteral("匹配完成"),
                                                        validIds,
                                                        allPairs,
                                                        pairPlan,
                                                        idToPath,
                                                        result.failedPairs);
    const QJsonObject matchingQualityReportFiles =
        writeSfmMatchingQualityReports(assetsDir,
                                       result.sfmDiagnostics,
                                       allPairs,
                                       pairPlan,
                                       idToPath,
                                       result.failedPairs);
    if (!matchingQualityReportFiles.isEmpty())
    {
        QJsonObject diagnostics = result.sfmDiagnostics;
        diagnostics[QStringLiteral("matching_quality_report")] = matchingQualityReportFiles;
        result.sfmDiagnostics = diagnostics;
        LOG_INFO(QStringLiteral("  匹配质量报告: JSON=%1 CSV=%2")
            .arg(matchingQualityReportFiles.value(QStringLiteral("json_path")).toString(),
                 matchingQualityReportFiles.value(QStringLiteral("csv_path")).toString()));
    }

    // 2b. 统计有效匹配
    int loadedMatches = 0;
    for (const auto &pd : allPairs) 
    {
        if (pd.loaded && !pd.matches.empty()) ++loadedMatches;
    }

    LOG_INFO(QStringLiteral("  有效匹配对: %1").arg(loadedMatches));

    if (loadedMatches < 1) 
    {
        result.errorMessage = QStringLiteral("未找到有效匹配对，无法执行重建");
        result.summary      = result.errorMessage;
        return result;
    }

    // 取消检查
    if (isCancelled())
    {
        result.errorMessage = QStringLiteral("用户取消");
        result.summary = result.errorMessage;
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // baOnly 模式：Phase 1+2 已完成，直接返回，由外部执行 BA
    // ══════════════════════════════════════════════════════════════════════════
    if (opts.baOnly) 
    {
        result.featureMatchesReady = true;
        result.summary = QStringLiteral("特征检测与匹配完成，可执行光束法平差");
        reportProgress(QStringLiteral("特征/匹配准备完毕"), 100);
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3: 执行增量式 SFM 重建
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("执行 SFM 重建..."), 70);
    LOG_INFO(QStringLiteral("SFM Phase 3: 增量式重建..."));

    IncrementalSfmOptions sfmOpts;
    sfmOpts.initMinNumMatches = presets.initMinNumMatches;
    sfmOpts.initMinNumInliers = presets.initMinNumInliers;
    sfmOpts.localBAInterval   = presets.localBAInterval;
    sfmOpts.globalBAInterval  = presets.globalBAInterval;
    sfmOpts.baOptions.cancelFlag = opts.cancelFlag;

    // 根据精度等级调整过滤参数：高精度/最高精度更严格
    if (opts.quality >= 2) 
    {
        sfmOpts.filterMaxReprojError = 1.5;      // 更严格的重投影误差阈值
        sfmOpts.filterMinTriAngle    = 2.0;
        sfmOpts.iterativeBARounds    = 4;         // 更多迭代精化轮数
    }

    // 3a. 添加影像（使用缓存中的特征点 + 内参）
    // 内参来源优先级：cameraPaths(.tsai文件) > projectMeta(项目元数据) > userFu/Fv > 自动估算
    const bool hasUserIntrinsics = (opts.userFu > 0 && opts.userFv > 0);
    const bool hasUserPitch = (opts.userPitch > 0);
    const bool hasCameraPaths = (!opts.cameraPaths.isEmpty()
                                 && opts.cameraPaths.size() == opts.images.size());

    // 从 projectMeta 中预构建 imagePath → cameraJson 映射。项目元数据里的相机通常来自 EXIF/GPS
    // 或前置估计，只作为增量 SfM 的相机初值/内参；只有显式 .tsai 列表才触发固定外参模式。
    QMap<QString, QJsonObject> projectCameraMap;
    if (!opts.projectMeta.isEmpty())
    {
        const QMap<QString, QJsonObject> projectImageMetaByPath =
            xjw::gui::project::projectImageMetaByPath(opts.projectMeta, true);
        for (auto it = projectImageMetaByPath.constBegin(); it != projectImageMetaByPath.constEnd(); ++it)
        {
            Camera camera;
            if (!xjw::gui::project::imageCameraFromEntry(it.value(), &camera) || !camera.isValid())
            {
                continue;
            }

            const QJsonObject camObj = it.value().value(QStringLiteral("camera")).toObject();
            if (!camObj.isEmpty())
            {
                projectCameraMap.insert(it.key(), camObj);
            }
        }
    }

    int projectMetaKnownCameraCount = 0;
    bool hasCompleteProjectMetaCameras = !opts.images.isEmpty() && !projectCameraMap.isEmpty();
    if (hasCompleteProjectMetaCameras)
    {
        for (const QString &imagePath : opts.images)
        {
            const QString normImgPath = normalizePath(imagePath);
            if (!projectCameraMap.contains(normImgPath))
            {
                hasCompleteProjectMetaCameras = false;
                break;
            }
            ++projectMetaKnownCameraCount;
        }
    }

    bool hasCompleteCameraFiles = hasCameraPaths;
    if (hasCompleteCameraFiles)
    {
        for (const QString &cameraPath : opts.cameraPaths)
        {
            if (cameraPath.isEmpty() || !QFileInfo::exists(cameraPath))
            {
                hasCompleteCameraFiles = false;
                break;
            }
        }
    }
    sfmOpts.useKnownCameraPoses = hasCompleteCameraFiles || hasCompleteProjectMetaCameras;
    sfmOpts.maxKnownPoseTracksPerImage = opts.maxTiePointsPerImage;
    sfmOpts.maxKnownPoseTracksPerGridCell = opts.maxTiePointsPerGridCell;
    sfmOpts.trackThinningGridColumns = opts.tiePointGridColumns;
    sfmOpts.trackThinningGridRows = opts.tiePointGridRows;
    if (sfmOpts.useKnownCameraPoses)
    {
        sfmOpts.baOptions.progressCallback =
            [&reportProgress, &isCancelled](int currentIteration,
                                            int maxIterations,
                                            double avgRms,
                                            int validPoints) -> bool
            {
                if (isCancelled())
                {
                    return false;
                }

                const int safeMaxIterations = std::max(1, maxIterations);
                const int clampedIteration = std::max(0, std::min(currentIteration, safeMaxIterations));
                const int pct = 80 + static_cast<int>(10.0 * clampedIteration / safeMaxIterations);
                reportProgress(QStringLiteral("正在进行光束法平差... %1/%2 RMS=%3 有效点=%4")
                                   .arg(currentIteration)
                                   .arg(maxIterations)
                                   .arg(avgRms, 0, 'f', 3)
                                   .arg(validPoints),
                               pct);
                return true;
            };
    }

    IncrementalSfm sfm(sfmOpts);

    if (hasCameraPaths) 
    {
        LOG_INFO(QStringLiteral("  使用相机文件路径列表 (%1 个)")
            .arg(opts.cameraPaths.size()));
        if (hasCompleteCameraFiles)
        {
            LOG_INFO(QStringLiteral("  使用 .tsai 已知外参初值模式：相机位姿参与全局 BA 微调"));
        }
        else
        {
            LOG_WARN(hasCompleteProjectMetaCameras
                ? QStringLiteral("  相机文件列表不完整，将使用项目元数据已知外参模式")
                : QStringLiteral("  相机文件列表不完整，回退到增量 SfM 估计位姿"));
        }
    } 
    if (hasCompleteProjectMetaCameras)
    {
        LOG_INFO(QStringLiteral("  使用项目元数据已知外参初值模式：相机位姿参与全局 BA 微调 (%1/%2)")
            .arg(projectMetaKnownCameraCount)
            .arg(opts.images.size()));
    }
    else if (!projectCameraMap.isEmpty())
    {
        LOG_INFO(QStringLiteral("  从项目元数据加载相机参数 (%1 个)")
            .arg(projectCameraMap.size()));
    }
    else if (hasUserIntrinsics) {
        LOG_INFO(QStringLiteral("  使用用户提供的内参: fu=%1 fv=%2 cu=%3 cv=%4")
            .arg(opts.userFu, 0, 'f', 2).arg(opts.userFv, 0, 'f', 2)
            .arg(opts.userCu, 0, 'f', 2).arg(opts.userCv, 0, 'f', 2));
        if (hasUserPitch) {
            LOG_INFO(QStringLiteral("  像元大小 pitch=%1 mm")
                .arg(opts.userPitch, 0, 'f', 6));
        }
    } else {
        LOG_INFO(QStringLiteral("  使用自动估算内参 (focal ≈ max(w,h)×1.2)"));
    }

    int addedImages = 0;
    int cameraFromFile = 0;
    int cameraFromMeta = 0;
    int cameraFromUser = 0;
    int cameraEstimated = 0;

    for (auto it = featureCache.constBegin(); it != featureCache.constEnd(); ++it)
    {
        const ImageId id = it.key();
        const ImageFeatureCache &fc = it.value();

        std::vector<FeatureKeypoint> kpts;
        kpts.reserve(fc.featureOutput.keypoints.size());
        for (const auto &kp : fc.featureOutput.keypoints)
        {
            FeatureKeypoint fkp;
            fkp.x = kp.pt.x;
            fkp.y = kp.pt.y;
            kpts.push_back(fkp);
        }

        // 获取图像尺寸（优先用缓存，否则读取）
        int imgW = fc.imgW;
        int imgH = fc.imgH;
        if (imgW == 0 || imgH == 0)
        {
            cv::Mat img = cv::imread(idToPath[id].toStdString(), cv::IMREAD_GRAYSCALE);
            if (!img.empty())
            {
                imgW = img.cols;
                imgH = img.rows;
            }
            else
            {
                imgW = 1920;
                imgH = 1080;
            }
        }

        bool added = false;

        // 方式 1: 从 cameraPaths 列表加载 .tsai 文件
        if (!added && hasCameraPaths && id < static_cast<ImageId>(opts.cameraPaths.size()))
        {
            const QString &camPath = opts.cameraPaths[static_cast<int>(id)];
            if (!camPath.isEmpty() && QFileInfo::exists(camPath))
            {
                sfm.addImage(id, idToPath[id].toStdString(),
                             camPath.toStdString(), kpts);
                added = true;
                ++cameraFromFile;
            }
        }

        // 方式 2: 从项目元数据中加载相机参数
        if (!added)
        {
            const QString normImgPath = normalizePath(idToPath[id]);
            auto metaIt = projectCameraMap.find(normImgPath);
            if (metaIt != projectCameraMap.end())
            {
                Camera cam;
                if (xjw::gui::project::cameraFromJson(metaIt.value(), &cam) && cam.isValid())
                {
                    sfm.addImageWithCamera(id, idToPath[id].toStdString(), cam, kpts);
                    added = true;
                    ++cameraFromMeta;
                }
            }
        }

        // 方式 3: 使用用户提供的内参
        if (!added && hasUserIntrinsics)
        {
            Camera estimatedCam;
            double fuPx = opts.userFu;
            double fvPx = opts.userFv;
            double cuPx = opts.userCu;
            double cvPx = opts.userCv;
            if (opts.userPitch > 0)
            {
                fuPx = opts.userFu / opts.userPitch;
                fvPx = opts.userFv / opts.userPitch;
                if (cuPx > 0)
                {
                    cuPx = opts.userCu / opts.userPitch;
                }
                if (cvPx > 0)
                {
                    cvPx = opts.userCv / opts.userPitch;
                }
            }
            if (cuPx <= 0)
            {
                cuPx = imgW * 0.5;
            }
            if (cvPx <= 0)
            {
                cvPx = imgH * 0.5;
            }
            estimatedCam.setIntrinsics(fuPx, fvPx, cuPx, cvPx);

            if (addedImages == 0)
            {
                LOG_INFO(QStringLiteral("  像素内参: fu=%1 fv=%2 cu=%3 cv=%4")
                    .arg(fuPx, 0, 'f', 2).arg(fvPx, 0, 'f', 2)
                    .arg(cuPx, 0, 'f', 2).arg(cvPx, 0, 'f', 2));
            }
            sfm.addImageWithCamera(id, idToPath[id].toStdString(), estimatedCam, kpts);
            added = true;
            ++cameraFromUser;
        }

        // 方式 4: 自动估算焦距
        if (!added)
        {
            Camera estimatedCam;
            const double focalPx = std::max(imgW, imgH) * 1.2;
            estimatedCam.setIntrinsics(focalPx, focalPx,
                                       imgW * 0.5, imgH * 0.5);
            sfm.addImageWithCamera(id, idToPath[id].toStdString(), estimatedCam, kpts);
            ++cameraEstimated;
        }

        ++addedImages;
    }

    LOG_INFO(QStringLiteral("  相机来源: file=%1, meta=%2, user=%3, estimated=%4")
        .arg(cameraFromFile).arg(cameraFromMeta).arg(cameraFromUser).arg(cameraEstimated));

    // 3b. 添加匹配
    for (const auto &pd : allPairs) {
        if (!pd.loaded || pd.matches.empty()) continue;
        sfm.addMatches(pd.idA, pd.idB, pd.matches);
    }

    // 保存关键点位置（轻量，不含描述子），供导出步骤做颜色采样
    QMap<ImageId, std::vector<cv::KeyPoint>> kptPositions;
    for (auto it = featureCache.constBegin(); it != featureCache.constEnd(); ++it)
        kptPositions[it.key()] = it.value().featureOutput.keypoints;

    // 默认及时释放描述子张量；guided rematching 需要在初始 SfM 后复用描述子做第二轮补匹配。
    if (!opts.enableGuidedRematching)
    {
        featureCache.clear();
    }

    LOG_INFO(QStringLiteral("  影像: %1, 匹配对: %2, 开始重建...")
        .arg(addedImages).arg(loadedMatches));

    // 3c. 运行 SFM
    auto sfmResult = sfm.run([&reportProgress, &isCancelled](int registered, int total, const std::string &msg) -> bool {
        LOG_INFO(QStringLiteral("  SFM 进度: [%1/%2] %3")
            .arg(registered).arg(total).arg(QString::fromStdString(msg)));
        if (isCancelled())
            return false;
        int pct = 70;
        if (total > 0)
            pct = 70 + static_cast<int>(20.0 * registered / total);
        reportProgress(QStringLiteral("正在重建... %1/%2").arg(registered).arg(total), pct);
        return true;
    });

    // 3d. 取消检查
    if (isCancelled())
    {
        result.success = false;
        result.errorMessage = QStringLiteral("用户取消");
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 4: 处理结果
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("整理结果..."), 90);
    auto applySfmResultStats = [&]()
    {
        result.numRegisteredImages = sfmResult.numRegisteredImages;
        result.numPoints3D         = sfmResult.numPoints3D;
        result.meanReprojError     = sfmResult.meanReprojError;
        result.baRmsBefore         = sfmResult.baRmsBefore;
        result.baRmsAfter          = sfmResult.baRmsAfter;
        result.baTracksTotal       = sfmResult.baTracksTotal;
        result.baTracksOptimized   = sfmResult.baTracksOptimized;
        result.baTracksFiltered    = sfmResult.baTracksFiltered;
    };
    applySfmResultStats();
    result.durationSeconds   = elapsedTimer.elapsed() / 1000.0;

    if (sfmResult.success && sfmResult.reconstruction) {
        result.success = true;

        LOG_INFO(QStringLiteral("SFM 完成: 注册 %1 张影像, %2 个三维点, 平均重投影误差 %3 px")
            .arg(sfmResult.numRegisteredImages)
            .arg(sfmResult.numPoints3D)
            .arg(sfmResult.meanReprojError, 0, 'f', 2));

        const auto *recon = sfmResult.reconstruction.get();

        // 4a. 收集相机参数更新
        LOG_INFO(QStringLiteral("SFM: 开始收集相机更新和质量统计"));
        for (auto it = imageIdMap.constBegin(); it != imageIdMap.constEnd(); ++it) {
            const ImageId id = it.value();
            if (!recon->isRegistered(id)) continue;
            const Camera &cam = recon->camera(id);
            result.pendingCamUpdates.insert(it.key(), cameraToJson(cam));
        }
        LOG_INFO(QStringLiteral("SFM: 相机更新收集完成 %1 项").arg(result.pendingCamUpdates.size()));

        GuidedRematchExecutionStats guidedStats;
        {
            SfmGuidedMatchPlan guidedPlan;
            if (opts.enableGuidedRematching)
            {
                SfmGuidedMatchPlannerOptions guidedOptions;
                guidedOptions.minSeedMatches = presets.initMinNumMatches;
                guidedOptions.maxHealthyMatches = presets.minInliers;
                guidedOptions.maxCandidates = 2000;
                for (const ImageId id : validIds)
                {
                    if (recon->isRegistered(id))
                    {
                        guidedOptions.registeredImageIds.insert(static_cast<int>(id));
                    }
                }

                guidedPlan = planSfmGuidedMatching(makeSfmDiagnosticImageIds(validIds),
                                                   makeSfmDiagnosticPairs(allPairs),
                                                   guidedOptions);
                LOG_INFO(QStringLiteral("SFM: Guided matching 候选 %1 对，种子匹配 %2 对，未注册跳过 %3 对")
                    .arg(guidedPlan.candidates.size())
                    .arg(guidedPlan.seedPairCount)
                    .arg(guidedPlan.skippedUnregisteredPairs));
            }
            else
            {
                LOG_INFO(QStringLiteral("SFM: Guided matching 未启用，跳过第二轮候选规划"));
            }

            if (opts.enableGuidedRematching)
            {
                guidedStats = appendGuidedRematchCandidatesToPairs(guidedPlan,
                                                                    &allPairs,
                                                                    featureCache,
                                                                    *recon,
                                                                    featureAlgorithm,
                                                                    presets.minInliers);
                LOG_INFO(QStringLiteral(
                    "SFM: Guided matching 尝试 %1/%2 对，生成候选 %3 个，追加 %4 个，不替换已有匹配")
                    .arg(guidedStats.attemptedPairCount)
                    .arg(guidedStats.plannedPairCount)
                    .arg(guidedStats.generatedMatchCount)
                    .arg(guidedStats.addedMatchCount));

                if (guidedStats.addedMatchCount > 0 && !isCancelled())
                {
                    guidedStats.secondPassAttempted = true;
                    LOG_INFO(QStringLiteral("SFM Guided matching: second pass with %1 appended matches")
                        .arg(guidedStats.addedMatchCount));
                    reportProgress(QStringLiteral("Guided matching 第二轮 SFM..."), 88);

                    IncrementalSfm guidedSfm(sfmOpts);
                    for (auto it = featureCache.constBegin(); it != featureCache.constEnd(); ++it)
                    {
                        const ImageId id = it.key();
                        const ImageFeatureCache &fc = it.value();

                        std::vector<FeatureKeypoint> kpts;
                        kpts.reserve(fc.featureOutput.keypoints.size());
                        for (const cv::KeyPoint &kp : fc.featureOutput.keypoints)
                        {
                            FeatureKeypoint fkp;
                            fkp.x = kp.pt.x;
                            fkp.y = kp.pt.y;
                            kpts.push_back(fkp);
                        }

                        int imgW = fc.imgW;
                        int imgH = fc.imgH;
                        if (imgW == 0 || imgH == 0)
                        {
                            cv::Mat img = cv::imread(idToPath[id].toStdString(), cv::IMREAD_GRAYSCALE);
                            if (!img.empty())
                            {
                                imgW = img.cols;
                                imgH = img.rows;
                            }
                            else
                            {
                                imgW = 1920;
                                imgH = 1080;
                            }
                        }

                        bool added = false;
                        if (!added && hasCameraPaths && id < static_cast<ImageId>(opts.cameraPaths.size()))
                        {
                            const QString &camPath = opts.cameraPaths[static_cast<int>(id)];
                            if (!camPath.isEmpty() && QFileInfo::exists(camPath))
                            {
                                guidedSfm.addImage(id, idToPath[id].toStdString(), camPath.toStdString(), kpts);
                                added = true;
                            }
                        }

                        if (!added)
                        {
                            const QString normImgPath = normalizePath(idToPath[id]);
                            auto metaIt = projectCameraMap.find(normImgPath);
                            if (metaIt != projectCameraMap.end())
                            {
                                Camera cam;
                                if (xjw::gui::project::cameraFromJson(metaIt.value(), &cam) && cam.isValid())
                                {
                                    guidedSfm.addImageWithCamera(id, idToPath[id].toStdString(), cam, kpts);
                                    added = true;
                                }
                            }
                        }

                        if (!added && hasUserIntrinsics)
                        {
                            Camera estimatedCam;
                            double fuPx = opts.userFu;
                            double fvPx = opts.userFv;
                            double cuPx = opts.userCu;
                            double cvPx = opts.userCv;
                            if (opts.userPitch > 0)
                            {
                                fuPx = opts.userFu / opts.userPitch;
                                fvPx = opts.userFv / opts.userPitch;
                                if (cuPx > 0)
                                {
                                    cuPx = opts.userCu / opts.userPitch;
                                }
                                if (cvPx > 0)
                                {
                                    cvPx = opts.userCv / opts.userPitch;
                                }
                            }
                            if (cuPx <= 0)
                            {
                                cuPx = imgW * 0.5;
                            }
                            if (cvPx <= 0)
                            {
                                cvPx = imgH * 0.5;
                            }
                            estimatedCam.setIntrinsics(fuPx, fvPx, cuPx, cvPx);
                            guidedSfm.addImageWithCamera(id, idToPath[id].toStdString(), estimatedCam, kpts);
                            added = true;
                        }

                        if (!added)
                        {
                            Camera estimatedCam;
                            const double focalPx = std::max(imgW, imgH) * 1.2;
                            estimatedCam.setIntrinsics(focalPx, focalPx, imgW * 0.5, imgH * 0.5);
                            guidedSfm.addImageWithCamera(id, idToPath[id].toStdString(), estimatedCam, kpts);
                        }
                    }

                    int guidedLoadedPairs = 0;
                    for (const PairMatchData &pair : allPairs)
                    {
                        if (!pair.loaded || pair.matches.empty())
                        {
                            continue;
                        }
                        guidedSfm.addMatches(pair.idA, pair.idB, pair.matches);
                        ++guidedLoadedPairs;
                    }

                    IncrementalSfmResult guidedSfmResult =
                        guidedSfm.run([&reportProgress, &isCancelled](int registered,
                                                                       int total,
                                                                       const std::string &msg) -> bool
                    {
                        LOG_INFO(QStringLiteral("  Guided SFM 进度: [%1/%2] %3")
                            .arg(registered)
                            .arg(total)
                            .arg(QString::fromStdString(msg)));
                        if (isCancelled())
                        {
                            return false;
                        }
                        const int pct = total > 0 ? 88 + static_cast<int>(7.0 * registered / total) : 88;
                        reportProgress(QStringLiteral("Guided matching 二次重建... %1/%2")
                                           .arg(registered)
                                           .arg(total),
                                       pct);
                        return true;
                    });

                    const int guided_point_gain = guidedSfmResult.numPoints3D - sfmResult.numPoints3D;
                    const int guided_min_point_gain =
                        std::max(50, static_cast<int>(std::ceil(
                            sfmResult.numPoints3D *
                            std::max(0.0, opts.guidedFillMinPointGainRatio))));
                    const double initial_rms =
                        sfmResult.baRmsAfter > 0.0 ? sfmResult.baRmsAfter : sfmResult.meanReprojError;
                    const double guided_rms =
                        guidedSfmResult.baRmsAfter > 0.0
                            ? guidedSfmResult.baRmsAfter
                            : guidedSfmResult.meanReprojError;
                    const bool guided_rms_acceptable =
                        initial_rms <= 0.0 ||
                        guided_rms <= initial_rms * std::max(1.0, opts.guidedFillMaxRmsRegressionRatio);
                    const bool guided_improved =
                        guidedSfmResult.numRegisteredImages > sfmResult.numRegisteredImages ||
                        guided_point_gain >= guided_min_point_gain;

                    if (guidedSfmResult.success &&
                        guidedSfmResult.numRegisteredImages >= sfmResult.numRegisteredImages &&
                        guidedSfmResult.numPoints3D > 0 &&
                        guided_improved &&
                        guided_rms_acceptable)
                    {
                        guidedStats.secondPassAccepted = true;
                        LOG_INFO(QStringLiteral("SFM Guided matching: second pass accepted, pairs=%1, registered=%2, "
                                                "points=%3, gain=%4/%5, rms=%6 -> %7")
                            .arg(guidedLoadedPairs)
                            .arg(guidedSfmResult.numRegisteredImages)
                            .arg(guidedSfmResult.numPoints3D)
                            .arg(guided_point_gain)
                            .arg(guided_min_point_gain)
                            .arg(initial_rms, 0, 'f', 4)
                            .arg(guided_rms, 0, 'f', 4));
                        sfmResult = std::move(guidedSfmResult);
                        applySfmResultStats();
                        recon = sfmResult.reconstruction.get();

                        result.pendingCamUpdates.clear();
                        for (auto it = imageIdMap.constBegin(); it != imageIdMap.constEnd(); ++it)
                        {
                            const ImageId id = it.value();
                            if (!recon->isRegistered(id))
                            {
                                continue;
                            }
                            const Camera &cam = recon->camera(id);
                            result.pendingCamUpdates.insert(it.key(), cameraToJson(cam));
                        }
                        LOG_INFO(QStringLiteral("SFM: Guided second pass 相机更新收集完成 %1 项")
                            .arg(result.pendingCamUpdates.size()));
                    }
                    else
                    {
                        LOG_WARN(QStringLiteral("SFM Guided matching: second pass rejected, keeping initial reconstruction "
                                                "(insufficient gain or worse RMS; success=%1, registered=%2/%3, "
                                                "points=%4, gain=%5/%6, rms=%7 -> %8)")
                            .arg(guidedSfmResult.success ? 1 : 0)
                            .arg(guidedSfmResult.numRegisteredImages)
                            .arg(sfmResult.numRegisteredImages)
                            .arg(guidedSfmResult.numPoints3D)
                            .arg(guided_point_gain)
                            .arg(guided_min_point_gain)
                            .arg(initial_rms, 0, 'f', 4)
                            .arg(guided_rms, 0, 'f', 4));
                    }
                }
            }

            QJsonObject guidedDiagnostics =
                sfmGuidedMatchPlanToJson(guidedPlan, idToPath, opts.enableGuidedRematching);
            const QJsonObject guidedStatsJson = guidedStats.toJson();
            for (auto it = guidedStatsJson.constBegin(); it != guidedStatsJson.constEnd(); ++it)
            {
                guidedDiagnostics.insert(it.key(), it.value());
            }

            QJsonObject diagnostics = result.sfmDiagnostics;
            diagnostics[QStringLiteral("guided_matching")] = guidedDiagnostics;
            result.sfmDiagnostics = diagnostics;
        }

        if (opts.enableGuidedRematching)
        {
            featureCache.clear();
        }

        // 4a-2. 收集逐相机残差（供报告使用，一次遍历所有三维点）
        {
            LOG_INFO(QStringLiteral("SFM: 开始统计逐相机残差"));
            // 先统计每相机的误差累计
            std::unordered_map<ImageId, std::pair<double,int>> camErr;  // id → (sumErr, count)
            for (Point3DId pid : recon->allPoint3DIds()) {
                if (!recon->hasPoint3D(pid)) continue;
                const auto &pt = recon->point3D(pid);
                for (const auto &elem : pt.track.elements) {
                    camErr[elem.imageId].first  += pt.error;
                    camErr[elem.imageId].second += 1;
                }
            }

            for (auto it = imageIdMap.constBegin(); it != imageIdMap.constEnd(); ++it) {
                const ImageId id  = it.value();
                const bool    reg = recon->isRegistered(id);
                double        res = 0.0;
                if (reg) {
                    auto eit = camErr.find(id);
                    if (eit != camErr.end() && eit->second.second > 0)
                        res = eit->second.first / eit->second.second;
                    else
                        res = sfmResult.meanReprojError;
                }
                QJsonObject camObj;
                camObj["path"]        = it.key();
                camObj["registered"]  = reg;
                camObj["residual_px"] = res;
                result.perCameraResiduals.append(camObj);
            }
            LOG_INFO(QStringLiteral("SFM: 逐相机残差统计完成 %1 项").arg(result.perCameraResiduals.size()));
        }

        QMap<Point3DId, double> triangulationAngleByPoint;
        {
            LOG_INFO(QStringLiteral("SFM: 开始统计稀疏质量指标"));
            std::vector<SfmQualityPoint> qualityPoints;
            qualityPoints.reserve(recon->allPoint3DIds().size());

            for (Point3DId pid : recon->allPoint3DIds())
            {
                if (!recon->hasPoint3D(pid))
                {
                    continue;
                }
                const auto &pt = recon->point3D(pid);
                const int trackLen = static_cast<int>(pt.track.length());
                const double triAngle = computeTrackMaxTriangulationAngleDeg(*recon, pt);
                triangulationAngleByPoint.insert(pid, triAngle);

                SfmQualityPoint qualityPoint;
                qualityPoint.trackLength = trackLen;
                qualityPoint.reprojectionErrorPx = pt.error;
                qualityPoint.triangulationAngleDeg = triAngle;

                for (const TrackElement &element : pt.track.elements)
                {
                    if (!recon->hasImage(element.imageId))
                    {
                        continue;
                    }

                    const ImageData &image = recon->image(element.imageId);
                    if (element.featureIdx >= static_cast<FeatureIdx>(image.keypoints.size()))
                    {
                        continue;
                    }

                    const FeatureKeypoint &keypoint = image.keypoints[element.featureIdx];
                    qualityPoint.observations.push_back({
                        static_cast<int>(element.imageId),
                        static_cast<double>(keypoint.x),
                        static_cast<double>(keypoint.y),
                    });
                }

                qualityPoints.push_back(std::move(qualityPoint));
            }

            const QSize qualityImageSize = inferSfmQualityImageSize(*recon, opts.images);
            SfmQualityReportOptions qualityOptions;
            qualityOptions.totalImageCount = opts.images.size();
            qualityOptions.registeredImageCount = result.numRegisteredImages;
            qualityOptions.imageWidth = qualityImageSize.width();
            qualityOptions.imageHeight = qualityImageSize.height();
            qualityOptions.coverageGridColumns = 4;
            qualityOptions.coverageGridRows = 4;
            qualityOptions.minTrackLength = sfmOpts.filterMinTrackLen;
            qualityOptions.minTriangulationAngleDeg = sfmOpts.filterMinTriAngle;
            qualityOptions.maxReprojectionErrorPx = sfmOpts.filterMaxReprojError;
            qualityOptions.minObservationGridCoverageMeanForMvs = 0.08;

            const SfmQualityReport qualityReport = analyzeSfmQuality(qualityPoints, qualityOptions);
            QJsonObject sparseQuality = qualityReport.toJson();
            sparseQuality[QStringLiteral("mean_reprojection_error_px")] = result.meanReprojError;

            QJsonObject baSummary;
            baSummary[QStringLiteral("rms_before_px")] = result.baRmsBefore;
            baSummary[QStringLiteral("rms_after_px")] = result.baRmsAfter;
            baSummary[QStringLiteral("tracks_total")] = result.baTracksTotal;
            baSummary[QStringLiteral("tracks_optimized")] = result.baTracksOptimized;
            baSummary[QStringLiteral("tracks_filtered")] = result.baTracksFiltered;

            QJsonObject diagnostics = result.sfmDiagnostics;
            diagnostics[QStringLiteral("sparse_quality")] = sparseQuality;
            diagnostics[QStringLiteral("ba_summary")] = baSummary;
            result.sfmDiagnostics = diagnostics;
            LOG_INFO(QStringLiteral("SFM: 稀疏质量指标统计完成 %1 个点").arg(triangulationAngleByPoint.size()));
        }

        // 4b. 导出稀疏点云（使用 plapoint::PointCloud 写 PLY 二进制）
        if (!outDir.isEmpty()) {
            LOG_INFO(QStringLiteral("SFM: 开始导出稀疏点云和质量 sidecar"));
            const QString plyPath = QDir(outDir).filePath(QStringLiteral("sfm_sparse.ply"));
            const auto ptIds = recon->allPoint3DIds();

            std::vector<float> ptsData;
            std::vector<uint8_t> colorsData;
            QJsonArray pointsForQuality;
            ptsData.reserve(ptIds.size() * 3);
            colorsData.reserve(ptIds.size() * 3);

            QMap<ImageId, std::vector<SparseExportColorRequest>> colorRequestsByImage;

            for (Point3DId pid : ptIds) {
                if (!recon->hasPoint3D(pid)) continue;
                const auto &pt = recon->point3D(pid);
                // 过滤离群点：重投影误差 > 4px 或 track 长度 < 2
                if (pt.error > 4.0) continue;
                if (pt.track.length() < 2) continue;

                const std::size_t outputPointIndex = ptsData.size() / 3;
                for (const auto &elem : pt.track.elements) {
                    auto kptIt = kptPositions.find(elem.imageId);
                    if (kptIt == kptPositions.end()) continue;
                    const auto &kpts = kptIt.value();
                    if (elem.featureIdx >= static_cast<FeatureIdx>(kpts.size())) continue;
                    colorRequestsByImage[elem.imageId].push_back({outputPointIndex, elem.featureIdx});
                }

                ptsData.push_back(static_cast<float>(pt.xyz[0]));
                ptsData.push_back(static_cast<float>(pt.xyz[1]));
                ptsData.push_back(static_cast<float>(pt.xyz[2]));
                colorsData.push_back(128);
                colorsData.push_back(128);
                colorsData.push_back(128);

                QJsonObject pointObject;
                pointObject[QStringLiteral("track_len")] = static_cast<int>(pt.track.length());
                pointObject[QStringLiteral("rms_reproj_px")] = pt.error;
                pointObject[QStringLiteral("triangulation_angle_deg")] = triangulationAngleByPoint.value(pid, 0.0);
                pointObject[QStringLiteral("min_tri_angle_deg")] = triangulationAngleByPoint.value(pid, 0.0);
                pointObject[QStringLiteral("point_xyz")] = QJsonArray{pt.xyz[0], pt.xyz[1], pt.xyz[2]};
                pointsForQuality.append(pointObject);
            }

            sampleSparseExportColorsByImage(colorRequestsByImage, kptPositions, idToPath, &colorsData);

            const size_t N = ptsData.size() / 3;
            LOG_INFO(QStringLiteral("SFM: 稀疏点云颜色采样完成 %1/%2 个点")
                .arg(static_cast<int>(N)).arg(static_cast<int>(ptIds.size())));
            plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(N, 3);
            plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(N, 3);
            for (size_t i = 0; i < N; ++i)
            {
                for (int c = 0; c < 3; ++c)
                {
                    pts(i, c) = ptsData[i * 3 + c];
                    colors(i, c) = colorsData[i * 3 + c];
                }
            }
            LOG_INFO(QStringLiteral("SFM: 稀疏点云矩阵填充完成 %1 个点").arg(static_cast<int>(N)));

            plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(pts));
            cloud.setColors(std::move(colors));

            try
            {
                LOG_INFO(QStringLiteral("SFM: 开始写出稀疏 PLY → %1").arg(plyPath));
                plapoint::io::writePly<float>(plyPath.toStdString(), cloud, plapoint::io::PlyFormat::BinaryLE);
                result.sparseCloudPath = plyPath;
                const bool baApplied = sfmResult.baTracksTotal > 0 || sfmResult.baTracksOptimized > 0;
                result.qualityMetadata = xjw::common::project::buildSparseQualityMetadata(
                    pointsForQuality,
                    sfmResult.numRegisteredImages,
                    baApplied,
                    xjw::common::project::kSparseResultKindSfmSparseReconstruction,
                    QString(),
                    QString(),
                    opts.images.size());

                const QString sidecarPath = QDir(outDir).filePath(QStringLiteral("sfm_sparse_points.json"));
                QJsonObject sidecarRoot = xjw::common::project::mergeSparseQualityIntoRecord(
                    QJsonObject{{QStringLiteral("points"), pointsForQuality},
                                {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")}},
                    result.qualityMetadata);
                sidecarRoot[QStringLiteral("sfm_diagnostics")] = result.sfmDiagnostics;
                QFile sidecarFile(sidecarPath);
                if (sidecarFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
                {
                    sidecarFile.write(QJsonDocument(sidecarRoot).toJson(QJsonDocument::Indented));
                    sidecarFile.close();
                }
                else
                {
                    LOG_WARN(QStringLiteral("SFM: 稀疏点云 sidecar 写出失败 → %1").arg(sidecarPath));
                }

                QJsonObject files;
                files[QStringLiteral("sparse_cloud_points_json")] = sidecarPath;
                result.resultRecordExtra = xjw::common::project::mergeSparseQualityIntoRecord(
                    QJsonObject{{QStringLiteral("files"), files},
                                {QStringLiteral("source"), QStringLiteral("workflow_aerial_triangulation")},
                                {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")}},
                    result.qualityMetadata);
                result.resultRecordExtra[QStringLiteral("sfm_diagnostics")] = result.sfmDiagnostics;
                LOG_INFO(QStringLiteral("SFM: 稀疏点云已保存 %1 个点（原始 %2 个）→ %3")
                    .arg(static_cast<int>(N)).arg(static_cast<int>(ptIds.size())).arg(plyPath));
            }
            catch (const std::exception &e)
            {
                LOG_INFO(QStringLiteral("SFM: 稀疏点云写出失败 → %1: %2")
                    .arg(plyPath).arg(QString::fromStdString(e.what())));
            }
        }

        result.summary = QStringLiteral("SFM 重建成功：注册 %1/%2 张影像，%3 个三维点")
            .arg(sfmResult.numRegisteredImages)
            .arg(opts.images.size())
            .arg(sfmResult.numPoints3D);
        result.durationSeconds = elapsedTimer.elapsed() / 1000.0;
        reportProgress(QStringLiteral("完成"), 100);

    } else {
        result.errorMessage = QStringLiteral("SFM 重建失败: %1")
            .arg(QString::fromStdString(sfmResult.summary));
        result.summary = result.errorMessage;
        LOG_WARN(result.errorMessage);
    }

    return result;
}

int minimumUsableSparsePointCountForSiftFallback(const SFMServiceOptions &opts,
                                                 const SFMServiceResult &result)
{
    const int imageCount = static_cast<int>(opts.images.size());
    const int registeredImages = std::max(0, result.numRegisteredImages);
    const int registeredTarget = imageCount > 0
        ? std::max(1, static_cast<int>(std::ceil(static_cast<double>(imageCount) * 0.95)))
        : std::max(1, registeredImages);
    return std::max(300, registeredTarget * 100);
}

bool primarySfmResultHasProductionSparseCloud(const SFMServiceOptions &opts,
                                              const SFMServiceResult &result)
{
    if (!result.success)
    {
        return false;
    }

    const int imageCount = static_cast<int>(opts.images.size());
    if (imageCount > 0)
    {
        const int minRegisteredImages =
            std::max(1, static_cast<int>(std::ceil(static_cast<double>(imageCount) * 0.95)));
        if (result.numRegisteredImages < minRegisteredImages)
        {
            return false;
        }
    }

    return result.numPoints3D >= minimumUsableSparsePointCountForSiftFallback(opts, result);
}

bool shouldRetrySfmWithSiftFallback(const SFMServiceOptions &opts,
                                    const SFMServiceResult &result)
{
    const QString featureAlgorithm = normalizedAlgorithm(opts.featureAlgorithm, QStringLiteral("disk"));
    const QString matchAlgorithm = normalizedAlgorithm(opts.matchAlgorithm, QStringLiteral("lightglue"));
    if (!opts.autoGenerateMissingMatches ||
        opts.restrictPairs ||
        !isLightGlueSfmMatch(featureAlgorithm, matchAlgorithm) ||
        featureAlgorithm != QStringLiteral("disk"))
    {
        return false;
    }

    if (opts.cancelFlag && opts.cancelFlag->load())
    {
        return false;
    }

    if (primarySfmResultHasProductionSparseCloud(opts, result))
    {
        const QString message = QStringLiteral(
            "SFM fallback skipped: DISK+LightGlue 已生成可用点云 "
            "(registered=%1/%2, points=%3, minPoints=%4)")
            .arg(result.numRegisteredImages)
            .arg(opts.images.size())
            .arg(result.numPoints3D)
            .arg(minimumUsableSparsePointCountForSiftFallback(opts, result));
        LOG_INFO(message);
        return false;
    }

    const QJsonObject actualGraph =
        result.sfmDiagnostics.value(QStringLiteral("actual_match_graph")).toObject();
    const int nodeCount = actualGraph.value(QStringLiteral("node_count")).toInt();
    const int componentCount = actualGraph.value(QStringLiteral("component_count")).toInt();
    const int largestComponentSize = actualGraph.value(QStringLiteral("largest_component_size")).toInt();
    if (nodeCount <= 1 || componentCount <= 1)
    {
        return false;
    }

    const double largestRatio = static_cast<double>(largestComponentSize) /
        static_cast<double>(std::max(1, nodeCount));
    return largestRatio < 0.95;
}

SFMServiceResult SFMService::run(const SFMServiceOptions &opts)
{
    SFMServiceResult firstResult = runSingleSfmAttempt(opts);
    if (!shouldRetrySfmWithSiftFallback(opts, firstResult))
    {
        return firstResult;
    }

    LOG_WARN(QStringLiteral(
        "SFM 默认 DISK+LightGlue 匹配图不连通，自动切换 SIFT+BF-L2 重跑一次空三"));
    SFMServiceOptions fallbackOptions = opts;
    fallbackOptions.featureAlgorithm = QStringLiteral("sift");
    fallbackOptions.matchAlgorithm = QStringLiteral("sift_bf_l2");
    fallbackOptions.cudaParallelPairs = 1;

    SFMServiceResult fallbackResult = runSingleSfmAttempt(fallbackOptions);
    if (fallbackResult.success || !firstResult.success)
    {
        QJsonObject diagnostics = fallbackResult.sfmDiagnostics;
        diagnostics[QStringLiteral("fallback_from_feature_algorithm")] =
            normalizedAlgorithm(opts.featureAlgorithm, QStringLiteral("disk"));
        diagnostics[QStringLiteral("fallback_from_match_algorithm")] =
            normalizedAlgorithm(opts.matchAlgorithm, QStringLiteral("lightglue"));
        diagnostics[QStringLiteral("fallback_reason")] =
            QStringLiteral("disconnected_default_match_graph");
        fallbackResult.sfmDiagnostics = diagnostics;
        return fallbackResult;
    }

    LOG_WARN(QStringLiteral("SFM SIFT+BF-L2 fallback 失败，保留首轮 DISK+LightGlue 结果: %1")
        .arg(fallbackResult.errorMessage));
    return firstResult;
}

} // namespace gui
} // namespace xjw

#ifdef _MSC_VER
#pragma warning(pop)
#endif

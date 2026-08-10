// =============================================================================
// 文件: DepthMapGenerator.cpp
// 模块: MVS - Qt 封装的深度图生成 + COLMAP BFS 融合
// =============================================================================

#include "DepthMapGenerator.h"

#include "concurrency/SafeWorkerGroup.h"
#include "DepthComputeScheduler.h"
#include "DenseCloudBuilder.h"
#include "DepthConsistencyCache.h"
#include "DepthConsistencyEvidencePolicy.h"
#include "DepthCrossViewHoleRepair.h"
#include "DepthGeometryConsistency.h"
#include "DepthFrameUtils.h"
#include "GpuDeviceLease.h"
#include "DepthMemoryPolicy.h"
#include "DepthPyramidPolicy.h"
#include "DepthProvenance.h"
#include "EpipolarRectifier.h"
#include "CameraBaseline.h"
#include "MvsImagePreprocessor.h"
#include "MvsImageMetadataProbe.h"
#include "MvsQualityReport.h"
#include "MvsSourcePlanner.h"
#include "MvsVisibilityGraphBuilder.h"
#include "MvsViewSelection.h"
#include "Logger.h"
#include "io/PathIO.h"
#include "string_utils/StringTransform.h"
#include <QtConcurrent/QtConcurrent>
#include <QDir>
#include <QCryptographicHash>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <sstream>
#include <functional>
#include <exception>
#include <memory>
#include <limits>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw
{
namespace mvs
{

using common::string_utils::asciiLowerCopy;

namespace
{

QJsonArray doubleArrayToJson(const double *values, int count)
{
    QJsonArray array;
    for (int index = 0; index < count; ++index)
    {
        array.append(values[index]);
    }
    return array;
}

Camera mvsPinholeCamera(const Camera &camera)
{
    Camera result = camera.normalizedForPositiveDepth();
    result.setDistortion(Camera::Distortion{});
    return result;
}

cv::Mat restoreNativePyramidArtifact(const cv::Mat &artifact, const cv::Size &working_size)
{
    if (artifact.empty() || artifact.size() == working_size)
    {
        return artifact;
    }

    cv::Mat restored;
    cv::resize(artifact, restored, working_size, 0.0, 0.0, cv::INTER_NEAREST);
    return restored;
}

QJsonObject cameraModelToJson(const Camera &camera)
{
    const Camera::Intrinsics intrinsics = camera.intrinsics();
    const std::array<double, 9> rotation = camera.worldToCameraRotation();
    const std::array<double, 3> translation = camera.worldToCameraTranslation();
    const std::array<double, 3> center = camera.cameraCenter();
    return QJsonObject{
        {QStringLiteral("fx"), intrinsics.focalX},
        {QStringLiteral("fy"), intrinsics.focalY},
        {QStringLiteral("cx"), intrinsics.principalX},
        {QStringLiteral("cy"), intrinsics.principalY},
        {QStringLiteral("rotation_world_to_camera"), doubleArrayToJson(rotation.data(), 9)},
        {QStringLiteral("translation_world_to_camera"), doubleArrayToJson(translation.data(), 3)},
        {QStringLiteral("camera_center"), doubleArrayToJson(center.data(), 3)}
    };
}

QJsonObject depthPoseRefinementCandidateToJson(
    const DepthPoseRefinementCandidate &candidate,
    const DepthPoseRefinementStageResult &stage)
{
    QJsonArray pivot;
    QJsonArray translation;
    QJsonArray rotation;
    for (int index = 0; index < 3; ++index)
    {
        pivot.append(candidate.correction.pivotWorld[index]);
        translation.append(candidate.correction.translation[index]);
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            rotation.append(candidate.correction.rotation(row, column));
        }
    }
    return QJsonObject{
        {QStringLiteral("enabled"), stage.enabled},
        {QStringLiteral("candidate_only"), stage.candidateOnly},
        {QStringLiteral("application_status"),
         QStringLiteral("not_applied_candidate_only")},
        {QStringLiteral("scale_locked"), true},
        {QStringLiteral("anchor_camera_index"), stage.anchorCameraIndex},
        {QStringLiteral("camera_index"), candidate.cameraIndex},
        {QStringLiteral("evidence_complete"), candidate.evidenceComplete},
        {QStringLiteral("accepted"), candidate.accepted},
        {QStringLiteral("reason"), QString::fromStdString(candidate.reason)},
        {QStringLiteral("evidence_pixel_count"), candidate.evidencePixelCount},
        {QStringLiteral("correspondence_count"),
         candidate.generatedCorrespondenceCount},
        {QStringLiteral("occluded_candidate_count"),
         candidate.occludedCandidateCount},
        {QStringLiteral("depth_conflict_candidate_count"),
         candidate.depthConflictCandidateCount},
        {QStringLiteral("evidence_sample_coverage"),
         candidate.evidenceSampleCoverage},
        {QStringLiteral("projection_retention_ratio"),
         candidate.projectionRetentionRatio},
        {QStringLiteral("translation_norm"),
         candidate.correctionTranslation},
        {QStringLiteral("rotation_degrees"),
         candidate.correctionRotationDegrees},
        {QStringLiteral("residual_median_before"),
         candidate.correction.residualMedianBefore},
        {QStringLiteral("residual_median_after"),
         candidate.correction.residualMedianAfter},
        {QStringLiteral("residual_p90_before"),
         candidate.correction.residualP90Before},
        {QStringLiteral("residual_p90_after"),
         candidate.correction.residualP90After},
        {QStringLiteral("correction_pivot_world"), pivot},
        {QStringLiteral("correction_translation_world"), translation},
        {QStringLiteral("correction_rotation_world"), rotation},
        {QStringLiteral("rotation_mapping"),
         QStringLiteral("R_wc'=R_wc*Q^T; stored R_cw'=Q*R_cw")}
    };
}

QJsonArray depthPyramidLevelsToJson(const std::vector<DepthLevelSummary> &summaries)
{
    QJsonArray array;
    for (const DepthLevelSummary &summary : summaries)
    {
        QJsonObject object;
        object.insert(QStringLiteral("level"), summary.level);
        object.insert(QStringLiteral("downsample_factor"), summary.downsampleFactor);
        object.insert(QStringLiteral("valid_pixel_count"), summary.validPixelCount);
        object.insert(QStringLiteral("valid_coverage"), summary.validCoverage);
        object.insert(QStringLiteral("mean_confidence"), summary.meanConfidence);
        object.insert(QStringLiteral("mean_support_views"), summary.meanSupportViews);
        object.insert(QStringLiteral("depth_discontinuity_ratio"),
                      summary.depthDiscontinuityRatio);
        object.insert(QStringLiteral("elapsed_ms"), summary.elapsedMs);
        object.insert(QStringLiteral("success"), summary.success);
        object.insert(QStringLiteral("error"), QString::fromStdString(summary.errorMessage));
        array.append(object);
    }
    return array;
}

QString sceneProfileId(MvsSceneProfile profile)
{
    switch (profile)
    {
    case MvsSceneProfile::AerialTerrain:
        return QStringLiteral("aerial_terrain");
    case MvsSceneProfile::OrbitalObject:
        return QStringLiteral("orbital_object");
    case MvsSceneProfile::Custom:
        return QStringLiteral("custom");
    case MvsSceneProfile::Auto:
    default:
        return QStringLiteral("auto");
    }
}

QString depthFilterModeId(DepthFilterMode mode)
{
    switch (mode)
    {
    case DepthFilterMode::Mild:
        return QStringLiteral("mild");
    case DepthFilterMode::Aggressive:
        return QStringLiteral("aggressive");
    case DepthFilterMode::Moderate:
    default:
        return QStringLiteral("moderate");
    }
}

constexpr float kSkipContentMaskCoverage = 0.985f;
constexpr std::size_t kMaxInlineDenseFilterPoints = 500000;
constexpr std::size_t kMaxProjectedDepthQuantileSamples = 8192;
constexpr uint64_t kBytesPerGiB = 1024ull * 1024ull * 1024ull;

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct FrameTiming
{
    double sourceMs = 0.0;
    double rangeMs = 0.0;
    double hintMs = 0.0;
    double rectifyMs = 0.0;
    double patchmatchMs = 0.0;
    double filterMs = 0.0;
    double totalMs = 0.0;
};

struct SystemMemorySnapshot
{
    uint64_t totalPhysicalBytes = 0;
    uint64_t availablePhysicalBytes = 0;
    bool valid = false;
};

QString normalizedMvsPathKey(const std::string &path)
{
    const QString rawPath = QString::fromStdString(path).trimmed();
    if (rawPath.isEmpty())
    {
        return QString();
    }

    const QFileInfo info(rawPath);
    QString absolutePath = info.exists() ? info.canonicalFilePath() : info.absoluteFilePath();
    if (absolutePath.isEmpty())
    {
        absolutePath = rawPath;
    }

    return QDir::cleanPath(absolutePath).replace(QLatin1Char('\\'), QLatin1Char('/')).toCaseFolded();
}

template <typename T>
void addHashValue(QCryptographicHash *hash, const T &value)
{
    hash->addData(reinterpret_cast<const char *>(&value),
                  static_cast<qsizetype>(sizeof(value)));
}

QString makeMvsDepthInputHash(const DepthGenConfig &config,
                              const std::vector<CameraView> &views,
                              const SparseCloud &sparse)
{
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(makeMvsDepthConfigHash(config, static_cast<int>(views.size())).toUtf8());
    for (const CameraView &view : views)
    {
        const QByteArray image_path = QByteArray::fromStdString(view.imagePath);
        hash.addData(image_path);
        const char separator = '\0';
        addHashValue(&hash, separator);
        addHashValue(&hash, view.imageWidth);
        addHashValue(&hash, view.imageHeight);
        const QFileInfo image_info(QString::fromStdString(view.imagePath));
        const qint64 image_size = image_info.exists() ? image_info.size() : -1;
        const qint64 image_modified = image_info.exists()
            ? image_info.lastModified().toMSecsSinceEpoch()
            : -1;
        addHashValue(&hash, image_size);
        addHashValue(&hash, image_modified);

        const xjw::Camera::Intrinsics intrinsics = view.camera.intrinsics();
        addHashValue(&hash, intrinsics.focalX);
        addHashValue(&hash, intrinsics.focalY);
        addHashValue(&hash, intrinsics.principalX);
        addHashValue(&hash, intrinsics.principalY);
        addHashValue(&hash, intrinsics.pixelPitch);
        addHashValue(&hash, intrinsics.uAxisSign);
        addHashValue(&hash, intrinsics.vAxisSign);

        const xjw::Camera::Distortion distortion = view.camera.distortion();
        addHashValue(&hash, distortion.radialK1);
        addHashValue(&hash, distortion.radialK2);
        addHashValue(&hash, distortion.radialK3);
        addHashValue(&hash, distortion.tangentialP1);
        addHashValue(&hash, distortion.tangentialP2);

        const auto rotation = view.camera.cameraToWorldRotation();
        const auto center = view.camera.cameraCenter();
        for (const double value : rotation)
        {
            addHashValue(&hash, value);
        }
        for (const double value : center)
        {
            addHashValue(&hash, value);
        }
        const bool depth_axis_flipped = view.camera.depthAxisFlipped();
        addHashValue(&hash, depth_axis_flipped);
    }

    addHashValue(&hash, config.requireVerifiedSourcePairs);
    addHashValue(&hash, config.minSourcePairGeometricInliers);
    const std::size_t pair_quality_count = config.sourcePairQualities.size();
    addHashValue(&hash, pair_quality_count);
    for (const MvsSourcePairQuality &quality : config.sourcePairQualities)
    {
        hash.addData(QByteArray::fromStdString(quality.imageA));
        hash.addData(QByteArray(1, '\0'));
        hash.addData(QByteArray::fromStdString(quality.imageB));
        hash.addData(QByteArray(1, '\0'));
        addHashValue(&hash, quality.totalMatches);
        addHashValue(&hash, quality.geometricInliers);
        addHashValue(&hash, quality.verified);
        addHashValue(&hash, quality.hasVerificationStatistics);
        addHashValue(&hash, quality.geometricCoverage);
        hash.addData(QByteArray::fromStdString(quality.verificationReason));
        hash.addData(QByteArray(1, '\0'));
    }

    const std::size_t sparse_point_count = sparse.points.size();
    addHashValue(&hash, sparse_point_count);
    for (const float value : sparse.minPt)
    {
        addHashValue(&hash, value);
    }
    for (const float value : sparse.maxPt)
    {
        addHashValue(&hash, value);
    }
    for (const auto &point : sparse.points)
    {
        for (const float coordinate : point)
        {
            addHashValue(&hash, coordinate);
        }
    }
    return QString::fromLatin1(hash.result().toHex());
}

std::string mvsSourcePairKey(const QString &keyA, const QString &keyB)
{
    if (keyA.isEmpty() || keyB.isEmpty() || keyA == keyB)
    {
        return std::string();
    }

    const QString pairKey = keyA < keyB
        ? keyA + QLatin1Char('\n') + keyB
        : keyB + QLatin1Char('\n') + keyA;
    return pairKey.toStdString();
}

std::string mvsSourcePairKey(const std::string &imageA, const std::string &imageB)
{
    return mvsSourcePairKey(
        normalizedMvsPathKey(imageA), normalizedMvsPathKey(imageB));
}

struct MvsSourcePairQualityLookup
{
    std::unordered_map<std::string, MvsSourcePairQuality> qualitiesByPairKey;

    const MvsSourcePairQuality *findNormalized(const QString &imageAKey,
                                               const QString &imageBKey) const
    {
        const std::string key = mvsSourcePairKey(imageAKey, imageBKey);
        if (key.empty())
        {
            return nullptr;
        }

        const auto it = qualitiesByPairKey.find(key);
        return it == qualitiesByPairKey.end() ? nullptr : &it->second;
    }
};

MvsSourcePairQualityLookup buildMvsSourcePairQualityLookup(
    const std::vector<MvsSourcePairQuality> &qualities)
{
    MvsSourcePairQualityLookup lookup;
    lookup.qualitiesByPairKey.reserve(qualities.size());
    for (const MvsSourcePairQuality &quality : qualities)
    {
        const std::string key = mvsSourcePairKey(quality.imageA, quality.imageB);
        if (key.empty())
        {
            continue;
        }

        auto it = lookup.qualitiesByPairKey.find(key);
        if (it == lookup.qualitiesByPairKey.end()
            || quality.geometricInliers > it->second.geometricInliers)
        {
            lookup.qualitiesByPairKey[key] = quality;
        }
    }
    return lookup;
}

struct SourceQualitySummary
{
    int sourceViewCount = 0;
    int verifiedSourceViewCount = 0;
    int backfillSourceViewCount = 0;
    int sequenceFallbackSourceViewCount = 0;
    double meanQuality = 0.0;
    double minQuality = 0.0;
};

struct DepthConfidenceSummary
{
    int validPixelCount = 0;
    double meanConfidence = 0.0;
};

QJsonObject depthPostProcessStatsToJson(const DepthPostProcessStats &stats)
{
    QJsonObject object;
    object.insert(QStringLiteral("valid_before"), stats.validBeforePostprocess);
    object.insert(QStringLiteral("valid_after_confidence_filter"), stats.validAfterConfidenceFilter);
    object.insert(QStringLiteral("low_confidence_candidate_count"),
                  stats.lowConfidenceCandidateCount);
    object.insert(QStringLiteral("geometry_supported_low_confidence_retained"),
                  stats.geometrySupportedLowConfidenceRetained);
    object.insert(QStringLiteral("confidence_removed"), stats.confidenceRemoved);
    object.insert(QStringLiteral("local_depth_outlier_removed"), stats.localDepthOutlierRemoved);
    object.insert(QStringLiteral("small_component_removed"), stats.smallComponentRemoved);
    object.insert(QStringLiteral("speckle_removed"), stats.speckleRemoved);
    object.insert(QStringLiteral("edge_confidence_removed"), stats.edgeConfidenceRemoved);
    object.insert(QStringLiteral("geom_consistency_removed"), stats.geomConsistencyRemoved);
    object.insert(QStringLiteral("valid_after"), stats.validAfterPostprocess);
    object.insert(QStringLiteral("effective_confidence_threshold"), stats.effectiveConfidenceThreshold);
    return object;
}

DepthPostProcessEvidence depthPostProcessEvidence(const DepthFrameResult &frame)
{
    DepthPostProcessEvidence evidence;
    if (frame.geometrySupportCount)
    {
        evidence.geometrySupportCount = *frame.geometrySupportCount;
    }
    if (frame.inverseDepthRelativeSpread)
    {
        evidence.inverseDepthRelativeSpread = *frame.inverseDepthRelativeSpread;
    }
    if (frame.adaptiveGeometrySupportWeight)
    {
        evidence.adaptiveSupportWeight = *frame.adaptiveGeometrySupportWeight;
    }
    if (frame.adaptiveGeometryEffectiveViewCount)
    {
        evidence.adaptiveEffectiveViewCount = *frame.adaptiveGeometryEffectiveViewCount;
    }
    if (frame.adaptiveGeometryConflictRatio)
    {
        evidence.adaptiveConflictRatio = *frame.adaptiveGeometryConflictRatio;
    }
    return evidence;
}

double bytesToGiB(uint64_t bytes)
{
    return static_cast<double>(bytes) / static_cast<double>(kBytesPerGiB);
}

SystemMemorySnapshot querySystemMemorySnapshot()
{
    SystemMemorySnapshot snapshot;
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
    {
        snapshot.totalPhysicalBytes = static_cast<uint64_t>(status.ullTotalPhys);
        snapshot.availablePhysicalBytes = static_cast<uint64_t>(status.ullAvailPhys);
        snapshot.valid = snapshot.totalPhysicalBytes > 0 && snapshot.availablePhysicalBytes > 0;
    }
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t valueKb = 0;
    std::string unit;
    uint64_t totalKb = 0;
    uint64_t availableKb = 0;
    while (meminfo >> key >> valueKb >> unit)
    {
        if (key == "MemTotal:")
        {
            totalKb = valueKb;
        }
        else if (key == "MemAvailable:")
        {
            availableKb = valueKb;
        }
    }
    if (totalKb > 0 && availableKb > 0)
    {
        snapshot.totalPhysicalBytes = totalKb * 1024ull;
        snapshot.availablePhysicalBytes = availableKb * 1024ull;
        snapshot.valid = true;
    }
#endif
    return snapshot;
}

uint64_t depthFramePixelStorageBytes(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        return 0;
    }

    const uint64_t pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    return pixels * sizeof(float) * 2ull; // depth + confidence
}

SourceQualitySummary summarizeSourceQuality(const QJsonArray &sourcePlan,
                                             int fallbackSourceViewCount)
{
    SourceQualitySummary summary;
    summary.sourceViewCount = std::max(0, fallbackSourceViewCount);

    double qualitySum = 0.0;
    double minQuality = std::numeric_limits<double>::max();
    int qualityCount = 0;
    for (const QJsonValue &value : sourcePlan)
    {
        if (!value.isObject())
        {
            continue;
        }

        const QJsonObject entry = value.toObject();
        const QString sourceTier = entry.value(QStringLiteral("source_tier")).toString();
        if (sourceTier == QStringLiteral("verified_pair"))
        {
            ++summary.verifiedSourceViewCount;
        }
        else if (sourceTier == QStringLiteral("track_geometry_backfill"))
        {
            ++summary.backfillSourceViewCount;
        }
        else if (sourceTier == QStringLiteral("sequence_fallback"))
        {
            ++summary.sequenceFallbackSourceViewCount;
        }

        const QJsonValue qualityValue = entry.value(QStringLiteral("source_quality_score"));
        if (!qualityValue.isDouble())
        {
            continue;
        }

        const double quality = std::clamp(qualityValue.toDouble(), 0.0, 1.0);
        qualitySum += quality;
        minQuality = std::min(minQuality, quality);
        ++qualityCount;
    }

    if (qualityCount > 0)
    {
        summary.meanQuality = qualitySum / static_cast<double>(qualityCount);
        summary.minQuality = minQuality;
    }
    return summary;
}

DepthConfidenceSummary summarizeDepthConfidence(const cv::Mat &depthMap,
                                                const cv::Mat *confidenceMap)
{
    DepthConfidenceSummary summary;
    if (depthMap.empty() || depthMap.type() != CV_32F)
    {
        return summary;
    }

    const cv::Mat validMask = depthMap > 0.0f;
    summary.validPixelCount = cv::countNonZero(validMask);
    if (summary.validPixelCount <= 0 ||
        !confidenceMap ||
        confidenceMap->empty() ||
        confidenceMap->size() != depthMap.size() ||
        confidenceMap->type() != CV_32F)
    {
        return summary;
    }

    summary.meanConfidence = std::clamp(cv::mean(*confidenceMap, validMask)[0], 0.0, 1.0);
    return summary;
}

bool usesAdaptiveGeometryEvidence(const DepthGenConfig &config,
                                  MvsSceneProfile sceneProfile)
{
    return config.enableAdaptiveGeometryEvidence &&
           sceneProfile == MvsSceneProfile::OrbitalObject;
}

std::vector<DepthMemoryFrameSize> depthMemoryFrameSizes(
    const std::vector<CameraView> &views)
{
    std::vector<DepthMemoryFrameSize> frameSizes;
    frameSizes.reserve(views.size());
    for (const CameraView &view : views)
    {
        frameSizes.push_back({view.imageWidth, view.imageHeight});
    }
    return frameSizes;
}

std::vector<MvsImageMemoryFrame> imageMemoryFrames(
    const std::vector<CameraView> &views)
{
    std::vector<MvsImageMemoryFrame> frames;
    frames.reserve(views.size());
    for (const CameraView &view : views)
    {
        frames.push_back({
            view.imageWidth,
            view.imageHeight,
            !mvsImagePreparationRequiresDistinctPixels(view.camera)});
    }
    return frames;
}

int maximumConsistencySourceViews(const DepthGenConfig &config)
{
    return std::max({config.numSourceViews,
                     config.patchMatch.numSourceViews,
                     config.crossViewHoleRepairSourceCount,
                     config.postConsistencyResidualSourceCount});
}

DepthMemoryPolicyDecision evaluateDepthMemoryPolicy(
    const std::vector<CameraView> &views,
    const DepthGenConfig &config,
    MvsSceneProfile sceneProfile,
    const SystemMemorySnapshot &snapshot)
{
    const std::vector<DepthMemoryFrameSize> frameSizes = depthMemoryFrameSizes(views);
    return decideDepthMemoryPolicy(
        frameSizes,
        maximumConsistencySourceViews(config),
        usesAdaptiveGeometryEvidence(config, sceneProfile),
        config.saveIntermediatePyramidLevels,
        snapshot.valid ? snapshot.totalPhysicalBytes : 0,
        snapshot.valid ? snapshot.availablePhysicalBytes : 0,
        config.maxDepthCacheRamFraction,
        config.minFreeRamBytes);
}

uint64_t largestDepthFrameBytes(const std::vector<CameraView> &views)
{
    uint64_t largest = 0;
    for (const CameraView &view : views)
    {
        largest = std::max(largest, depthFramePixelStorageBytes(view.imageWidth, view.imageHeight));
    }
    return largest;
}

uint64_t saturatingMultiplyBytes(uint64_t value, uint64_t factor)
{
    if (value != 0 && factor > std::numeric_limits<uint64_t>::max() / value)
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return value * factor;
}

uint64_t retainedDepthMemoryBudgetBytes(const SystemMemorySnapshot &snapshot,
                                        const DepthGenConfig &config,
                                        uint64_t largestFrameBytes,
                                        uint64_t transientFrameBytes,
                                        size_t concurrentFrameWorkers)
{
    if (!snapshot.valid)
    {
        return 0;
    }

    const uint64_t producerWorkingSetReserve = saturatingMultiplyBytes(largestFrameBytes, 8);
    return calculateDepthSaveQueueBudgetBytes(
        snapshot.totalPhysicalBytes,
        snapshot.availablePhysicalBytes,
        config.maxDepthCacheRamFraction,
        config.minFreeRamBytes,
        transientFrameBytes,
        concurrentFrameWorkers,
        producerWorkingSetReserve);
}

QString depthMemoryPolicyReason(const DepthMemoryPolicyDecision &decision,
                                const SystemMemorySnapshot &snapshot)
{
    if (decision.estimate.totalPixels == 0)
    {
        return QStringLiteral("无有效影像尺寸，采用保守流式模式");
    }
    if (!snapshot.valid)
    {
        return QStringLiteral("无法读取系统内存，采用保守流式模式");
    }
    return QStringLiteral(
               "预计峰值 %1 GiB %2 内存预算 %3 GiB（常驻=%4，快照=%5，证据=%6，"
               "金字塔=%7，单帧临时=%8，动态保留=%9 GiB）")
        .arg(bytesToGiB(decision.estimate.peakBytes), 0, 'f', 2)
        .arg(decision.retainAllFrames ? QStringLiteral("<=") : QStringLiteral(">"))
        .arg(bytesToGiB(decision.budgetBytes), 0, 'f', 2)
        .arg(bytesToGiB(decision.estimate.residentFrameBytes), 0, 'f', 2)
        .arg(bytesToGiB(decision.estimate.consistencySnapshotBytes), 0, 'f', 2)
        .arg(bytesToGiB(decision.estimate.retainedEvidenceBytes), 0, 'f', 2)
        .arg(bytesToGiB(decision.estimate.intermediatePyramidBytes), 0, 'f', 2)
        .arg(bytesToGiB(decision.estimate.transientFrameBytes), 0, 'f', 2)
        .arg(bytesToGiB(decision.reserveBytes), 0, 'f', 2);
}

bool memoryPressureRequiresStreaming(const DepthGenConfig &config,
                                     const SystemMemorySnapshot &snapshot,
                                     const DepthMemoryPolicyDecision &decision)
{
    if (!config.adaptiveDepthCacheMemory || !snapshot.valid)
    {
        return false;
    }

    const uint64_t runtimeReserve = std::max(
        decision.reserveBytes,
        saturatingMultiplyBytes(decision.estimate.transientFrameBytes, 2));
    if (snapshot.availablePhysicalBytes <= runtimeReserve)
    {
        return true;
    }

    const uint64_t availableBudget = snapshot.availablePhysicalBytes - runtimeReserve;
    const uint64_t totalBudget = static_cast<uint64_t>(
        static_cast<double>(snapshot.totalPhysicalBytes) *
        std::clamp(static_cast<double>(config.maxDepthCacheRamFraction), 0.10, 0.90));
    return decision.estimate.peakBytes > std::min(totalBudget, availableBudget);
}

void releaseStoredDepthFramePixelStorage(std::vector<DepthFrameResult> &frames)
{
    for (DepthFrameResult &frame : frames)
    {
        frame.releasePixelStorage();
    }
}

size_t adaptiveSaveQueueCapacity(const SystemMemorySnapshot &snapshot,
                                 const DepthGenConfig &config,
                                 uint64_t largestFrameBytes,
                                 uint64_t transientFrameBytes,
                                 size_t concurrentFrameWorkers)
{
    if (!config.adaptiveDepthCacheMemory || !snapshot.valid || largestFrameBytes == 0)
    {
        return 2;
    }

    const uint64_t budgetBytes = retainedDepthMemoryBudgetBytes(
        snapshot,
        config,
        largestFrameBytes,
        transientFrameBytes,
        concurrentFrameWorkers);
    if (budgetBytes >= saturatingMultiplyBytes(largestFrameBytes, 8))
    {
        return 4;
    }
    return 2;
}

uint64_t estimatedSaveQueueProducerBytes(uint64_t largestFrameBytes)
{
    constexpr uint64_t kFallbackResidentBytes = 512ull * 1024ull * 1024ull;
    constexpr uint64_t kEstimatedResultToDepthConfidenceRatio = 8;

    if (largestFrameBytes == 0)
    {
        return kFallbackResidentBytes;
    }
    return saturatingMultiplyBytes(
        largestFrameBytes, kEstimatedResultToDepthConfidenceRatio);
}

uint64_t adaptiveSaveQueueResidentByteCapacity(
    const SystemMemorySnapshot &snapshot,
    const DepthGenConfig &config,
    uint64_t largestFrameBytes,
    size_t maxResidentTasks,
    uint64_t transientFrameBytes,
    size_t concurrentFrameWorkers)
{
    constexpr uint64_t kFallbackMaximumResidentBytes =
        4ull * 1024ull * 1024ull * 1024ull;

    const uint64_t estimatedTaskBytes =
        estimatedSaveQueueProducerBytes(largestFrameBytes);

    const uint64_t residentTaskCount = static_cast<uint64_t>(
        std::max<size_t>(1, maxResidentTasks));
    uint64_t byteCapacity = saturatingMultiplyBytes(estimatedTaskBytes, residentTaskCount);

    const uint64_t memoryBudget = retainedDepthMemoryBudgetBytes(
        snapshot,
        config,
        largestFrameBytes,
        transientFrameBytes,
        concurrentFrameWorkers);
    if (snapshot.valid)
    {
        byteCapacity = std::min(byteCapacity, memoryBudget);
    }
    else
    {
        // Keep the historical conservative cap only when no trustworthy
        // system-memory snapshot is available. On known-memory systems the
        // dynamic budget already reserves both OS headroom and transient MVS
        // working sets; a second fixed 4 GiB cap can otherwise collapse a
        // multi-GPU preparation pipeline to one frame per device.
        byteCapacity = std::min(byteCapacity, kFallbackMaximumResidentBytes);
    }
    return byteCapacity;
}

int preloadImagesWorkerCount(int viewCount, int requestedThreads)
{
    if (viewCount <= 1)
    {
        return std::max(0, viewCount);
    }

    const int hwThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int requested = std::max(1, requestedThreads);
    return std::clamp(std::min(requested, hwThreads), 1, std::min(viewCount, 8));
}

class DepthFrameArtifactSaveQueue
{
public:
    using SaveFn = std::function<bool(int, const DepthFrameResult &, const QString &)>;

    class ProducerReservation
    {
    public:
        ProducerReservation() = default;

        ProducerReservation(const ProducerReservation &) = delete;
        ProducerReservation &operator=(const ProducerReservation &) = delete;

        ProducerReservation(ProducerReservation &&other) noexcept
            : _owner(other._owner)
            , _residentBytes(other._residentBytes)
        {
            other.disarm();
        }

        ProducerReservation &operator=(ProducerReservation &&other) noexcept
        {
            if (this != &other)
            {
                reset();
                _owner = other._owner;
                _residentBytes = other._residentBytes;
                other.disarm();
            }
            return *this;
        }

        ~ProducerReservation()
        {
            reset();
        }

        explicit operator bool() const
        {
            return _owner != nullptr;
        }

        void reset()
        {
            if (_owner == nullptr)
            {
                return;
            }

            DepthFrameArtifactSaveQueue *owner = _owner;
            const uint64_t resident_bytes = _residentBytes;
            disarm();
            owner->releaseProducerReservation(resident_bytes);
        }

    private:
        friend class DepthFrameArtifactSaveQueue;

        ProducerReservation(DepthFrameArtifactSaveQueue *owner,
                            uint64_t residentBytes)
            : _owner(owner)
            , _residentBytes(residentBytes)
        {
        }

        void disarm()
        {
            _owner = nullptr;
            _residentBytes = 0;
        }

        DepthFrameArtifactSaveQueue *_owner = nullptr;
        uint64_t _residentBytes = 0;
    };

    explicit DepthFrameArtifactSaveQueue(SaveFn saveFn,
                                         size_t workerCount,
                                         size_t maxResidentTasks,
                                         uint64_t maxResidentBytes,
                                         uint64_t producerReservationBytes)
        : _saveFn(std::move(saveFn))
        , _workerCount(std::clamp<size_t>(workerCount, 1, 2))
        , _maxResidentTasks(std::max<size_t>(1, maxResidentTasks))
        , _maxResidentBytes(std::max<uint64_t>(1, maxResidentBytes))
        , _producerReservationBytes(std::max<uint64_t>(1, producerReservationBytes))
    {
        _workers.reserve(_workerCount);
        try
        {
            for (size_t worker_index = 0; worker_index < _workerCount; ++worker_index)
            {
                _workers.emplace_back(&DepthFrameArtifactSaveQueue::run, this);
            }
        }
        catch (...)
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _stopping = true;
            }
            _cv.notify_all();
            for (std::thread &worker : _workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
            throw;
        }
    }

    ~DepthFrameArtifactSaveQueue()
    {
        stop();
    }

    ProducerReservation reserveProducer(
        const std::atomic<bool> *cancelFlag = nullptr)
    {
        const auto wait_start = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lock(_mutex);
            while (!_stopping &&
                   !(cancelFlag && cancelFlag->load()) &&
                   !canAcceptLocked(_producerReservationBytes))
            {
                _capacityCv.wait_for(lock, std::chrono::milliseconds(25));
            }
            const auto reservation_time = std::chrono::steady_clock::now();
            const auto reservation_wait = reservation_time - wait_start;
            _enqueueWait += reservation_wait;
            _maxEnqueueWait = std::max(_maxEnqueueWait, reservation_wait);
            if (_stopping || (cancelFlag && cancelFlag->load()))
            {
                return {};
            }

            if (!_wallStarted)
            {
                _wallStarted = true;
                _wallStart = reservation_time;
            }
            _lastActivity = reservation_time;
            ++_producerReservations;
            ++_residentTasks;
            _residentBytes += _producerReservationBytes;
            _peakResidentTasks = std::max(_peakResidentTasks, _residentTasks);
            _peakResidentBytes = std::max(_peakResidentBytes, _residentBytes);
        }
        return ProducerReservation(this, _producerReservationBytes);
    }

    void enqueue(ProducerReservation &&reservation,
                 int frameIndex,
                 const DepthFrameResult &result,
                 QString stageLabel)
    {
        if (reservation._owner != this)
        {
            _failed = true;
            LOG_ERROR(QStringLiteral(
                "[MVS] 深度产物保存队列收到无效的生产者内存预约"));
            return;
        }

        uint64_t reserved_bytes_for_log = 0;
        uint64_t resident_bytes = 0;
        bool queued = false;
        bool reservation_too_small = false;
        QString enqueue_exception;
        try
        {
            resident_bytes = depthFrameResultResidentBytes(result);
            SaveTask task{
                frameIndex, result, std::move(stageLabel), resident_bytes};

            std::lock_guard<std::mutex> lock(_mutex);
            const uint64_t reserved_bytes = reservation._residentBytes;
            reserved_bytes_for_log = reserved_bytes;
            if (_stopping)
            {
                releaseProducerReservationLocked(reserved_bytes);
                reservation.disarm();
            }
            else if (resident_bytes > reserved_bytes)
            {
                // The reservation is a conservative upper bound for all matrices
                // currently retained by DepthFrameResult. Failing closed here keeps
                // the configured/4 GiB limit a real upper bound if that structure is
                // extended without updating the estimate.
                releaseProducerReservationLocked(reserved_bytes);
                reservation.disarm();
                _failed = true;
                ++_failedTasks;
                reservation_too_small = true;
            }
            else
            {
                // Commit the queue insertion before transferring reservation
                // accounting. std::deque::push_back has a strong exception
                // guarantee, so an allocation failure leaves the armed RAII
                // reservation to release the producer slot and bytes.
                _tasks.push_back(std::move(task));
                --_producerReservations;
                _residentBytes -= std::min(_residentBytes, reserved_bytes);
                _residentBytes += resident_bytes;
                reservation.disarm();
                _lastActivity = std::chrono::steady_clock::now();
                queued = true;
            }
        }
        catch (const std::exception &exception)
        {
            enqueue_exception = QString::fromUtf8(exception.what());
        }
        catch (...)
        {
            enqueue_exception = QStringLiteral("未知异常");
        }
        if (!enqueue_exception.isEmpty())
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _failed = true;
                ++_failedTasks;
            }
            LOG_ERROR(QStringLiteral(
                          "[MVS] 深度帧 %1 保存任务入队异常：%2")
                          .arg(frameIndex)
                          .arg(enqueue_exception));
        }
        _capacityCv.notify_all();
        if (reservation_too_small)
        {
            LOG_ERROR(QStringLiteral(
                          "[MVS] 深度帧 %1 保存预约不足: reserved=%2 MiB actual=%3 MiB；"
                          "为保持保存队列内存硬上限，本帧未入队")
                          .arg(frameIndex)
                          .arg(static_cast<double>(reserved_bytes_for_log) /
                                   (1024.0 * 1024.0),
                               0,
                               'f',
                               1)
                          .arg(static_cast<double>(resident_bytes) /
                                   (1024.0 * 1024.0),
                               0,
                               'f',
                               1));
        }
        if (queued)
        {
            _cv.notify_one();
        }
    }

    bool waitUntilIdle(const std::atomic<bool> *cancelFlag = nullptr)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        const auto idle = [this]()
        {
            return _tasks.empty() && _activeTasks == 0 &&
                _producerReservations == 0;
        };
        while (!idle())
        {
            if (cancelFlag && cancelFlag->load())
            {
                return false;
            }
            if (cancelFlag)
            {
                _idleCv.wait_for(lock, std::chrono::milliseconds(25));
            }
            else
            {
                _idleCv.wait(lock);
            }
        }
        return true;
    }

    void cancel()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _dropPendingTasks = true;
            _stopping = true;
            dropPendingTasksLocked();
            if (_wallStarted)
            {
                _lastActivity = std::chrono::steady_clock::now();
            }
            if (_activeTasks == 0 && _producerReservations == 0)
            {
                _idleCv.notify_all();
            }
        }
        _capacityCv.notify_all();
        _cv.notify_all();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopping = true;
        }
        _capacityCv.notify_all();
        _cv.notify_all();
        for (std::thread &worker : _workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        logSummaryOnce();
    }

    bool failed() const
    {
        return _failed.load();
    }

private:
    struct MatAllocationSpan
    {
        std::uintptr_t begin = 0;
        std::uintptr_t end = 0;
    };

    struct SaveTask
    {
        int frameIndex = -1;
        DepthFrameResult result;
        QString stageLabel;
        uint64_t residentBytes = 0;
    };

    static void appendMatAllocationSpan(
        const cv::Mat &matrix,
        std::vector<MatAllocationSpan> &spans)
    {
        if (matrix.empty() || matrix.datastart == nullptr || matrix.dataend == nullptr)
        {
            return;
        }

        const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(matrix.datastart);
        const std::uintptr_t end = reinterpret_cast<std::uintptr_t>(matrix.dataend);
        if (end > begin)
        {
            spans.push_back({begin, end});
        }
    }

    static void appendSharedMatAllocationSpan(
        const QSharedPointer<cv::Mat> &matrix,
        std::vector<MatAllocationSpan> &spans)
    {
        if (matrix)
        {
            appendMatAllocationSpan(*matrix, spans);
        }
    }

    static uint64_t depthFrameResultResidentBytes(const DepthFrameResult &result)
    {
        std::vector<MatAllocationSpan> spans;
        spans.reserve(32 + result.intermediatePyramidLevels.size() * 6);
        appendSharedMatAllocationSpan(result.depthMap, spans);
        appendSharedMatAllocationSpan(result.confidence, spans);
        appendSharedMatAllocationSpan(result.normalMap, spans);
        appendSharedMatAllocationSpan(result.supportCount, spans);
        appendSharedMatAllocationSpan(result.geometrySupportCount, spans);
        appendSharedMatAllocationSpan(result.geometrySourceMask, spans);
        appendSharedMatAllocationSpan(result.inverseDepthMean, spans);
        appendSharedMatAllocationSpan(result.inverseDepthRelativeSpread, spans);
        appendSharedMatAllocationSpan(result.adaptiveGeometrySupportWeight, spans);
        appendSharedMatAllocationSpan(result.adaptiveGeometryEffectiveViewCount, spans);
        appendSharedMatAllocationSpan(result.adaptiveGeometryConflictRatio, spans);
        appendSharedMatAllocationSpan(result.crossViewRepairedMask, spans);
        appendSharedMatAllocationSpan(result.targetedGapRecoveredMask, spans);
        appendSharedMatAllocationSpan(result.residualReestimatedMask, spans);
        appendSharedMatAllocationSpan(result.depthProvenance, spans);
        appendSharedMatAllocationSpan(result.missingReasonMap, spans);
        appendSharedMatAllocationSpan(result.validMask, spans);
        appendSharedMatAllocationSpan(result.supportRegionMask, spans);
        for (const DepthLevelResult &level : result.intermediatePyramidLevels)
        {
            appendMatAllocationSpan(level.depth, spans);
            appendMatAllocationSpan(level.normalMap, spans);
            appendMatAllocationSpan(level.confidence, spans);
            appendMatAllocationSpan(level.supportCount, spans);
            appendMatAllocationSpan(level.uncertainty, spans);
            appendMatAllocationSpan(level.validMask, spans);
        }

        if (spans.empty())
        {
            return 0;
        }
        std::sort(spans.begin(), spans.end(), [](const MatAllocationSpan &left,
                                                  const MatAllocationSpan &right)
        {
            return left.begin < right.begin ||
                (left.begin == right.begin && left.end < right.end);
        });

        uint64_t resident_bytes = 0;
        std::uintptr_t allocation_begin = spans.front().begin;
        std::uintptr_t allocation_end = spans.front().end;
        for (size_t index = 1; index < spans.size(); ++index)
        {
            const MatAllocationSpan &span = spans[index];
            if (span.begin <= allocation_end)
            {
                allocation_end = std::max(allocation_end, span.end);
                continue;
            }
            resident_bytes += static_cast<uint64_t>(allocation_end - allocation_begin);
            allocation_begin = span.begin;
            allocation_end = span.end;
        }
        resident_bytes += static_cast<uint64_t>(allocation_end - allocation_begin);
        return resident_bytes;
    }

    bool canAcceptLocked(uint64_t taskResidentBytes) const
    {
        if (taskResidentBytes > _maxResidentBytes ||
            _residentTasks >= _maxResidentTasks ||
            _residentBytes >= _maxResidentBytes)
        {
            return false;
        }
        return taskResidentBytes <= _maxResidentBytes - _residentBytes;
    }

    void releaseProducerReservationLocked(uint64_t residentBytes)
    {
        if (_producerReservations == 0 || _residentTasks == 0)
        {
            return;
        }
        --_producerReservations;
        --_residentTasks;
        _residentBytes -= std::min(_residentBytes, residentBytes);
        _lastActivity = std::chrono::steady_clock::now();
        if (_tasks.empty() && _activeTasks == 0 &&
            _producerReservations == 0)
        {
            _idleCv.notify_all();
        }
    }

    void releaseProducerReservation(uint64_t residentBytes)
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            releaseProducerReservationLocked(residentBytes);
        }
        _capacityCv.notify_all();
    }

    void dropPendingTasksLocked()
    {
        for (const SaveTask &task : _tasks)
        {
            _residentBytes -= std::min(_residentBytes, task.residentBytes);
        }
        _residentTasks -= std::min(_residentTasks, _tasks.size());
        _tasks.clear();
    }

    void logSummaryOnce()
    {
        std::chrono::steady_clock::duration wall;
        std::chrono::steady_clock::duration enqueue_wait;
        std::chrono::steady_clock::duration max_enqueue_wait;
        std::chrono::steady_clock::duration busy;
        uint64_t peak_resident_bytes = 0;
        size_t peak_resident_tasks = 0;
        size_t saved_tasks = 0;
        size_t failed_tasks = 0;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_summaryLogged)
            {
                return;
            }
            _summaryLogged = true;
            if (_wallStarted && _lastActivity >= _wallStart)
            {
                wall = _lastActivity - _wallStart;
            }
            enqueue_wait = _enqueueWait;
            max_enqueue_wait = _maxEnqueueWait;
            busy = _busy;
            peak_resident_bytes = _peakResidentBytes;
            peak_resident_tasks = _peakResidentTasks;
            saved_tasks = _savedTasks;
            failed_tasks = _failedTasks;
        }

        const double wall_ms = std::chrono::duration<double, std::milli>(wall).count();
        const double enqueue_wait_ms =
            std::chrono::duration<double, std::milli>(enqueue_wait).count();
        const double max_enqueue_wait_ms =
            std::chrono::duration<double, std::milli>(max_enqueue_wait).count();
        const double busy_ms = std::chrono::duration<double, std::milli>(busy).count();
        LOG_INFO(QStringLiteral(
                     "[MVS] 深度产物保存队列统计: workers=%1 task_limit=%2 "
                     "byte_limit=%3 MiB wall=%4 ms enqueue_wait_total=%5 ms "
                     "enqueue_wait_max=%6 ms busy=%7 ms peak_resident_tasks=%8 "
                     "peak_resident=%9 MiB saved=%10 failed=%11")
                     .arg(_workerCount)
                     .arg(_maxResidentTasks)
                     .arg(static_cast<double>(_maxResidentBytes) / (1024.0 * 1024.0), 0, 'f', 1)
                     .arg(wall_ms, 0, 'f', 1)
                     .arg(enqueue_wait_ms, 0, 'f', 1)
                     .arg(max_enqueue_wait_ms, 0, 'f', 1)
                     .arg(busy_ms, 0, 'f', 1)
                     .arg(peak_resident_tasks)
                     .arg(static_cast<double>(peak_resident_bytes) / (1024.0 * 1024.0), 0, 'f', 1)
                     .arg(saved_tasks)
                     .arg(failed_tasks));
    }

    void run()
    {
        for (;;)
        {
            SaveTask task;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [this]() {
                    return _stopping || !_tasks.empty();
                });

                if (_tasks.empty())
                {
                    if (_stopping)
                    {
                        break;
                    }
                    continue;
                }

                if (_dropPendingTasks)
                {
                    dropPendingTasksLocked();
                    if (_activeTasks == 0 && _producerReservations == 0)
                    {
                        _idleCv.notify_all();
                    }
                    if (_stopping)
                    {
                        break;
                    }
                    continue;
                }

                task = std::move(_tasks.front());
                _tasks.pop_front();
                ++_activeTasks;
            }

            const auto busy_start = std::chrono::steady_clock::now();
            bool saved = false;
            bool save_exception_occurred = false;
            std::array<char, 512> save_exception{};
            try
            {
                saved = _saveFn(task.frameIndex, task.result, task.stageLabel);
            }
            catch (const std::exception &exception)
            {
                save_exception_occurred = true;
                std::snprintf(save_exception.data(),
                              save_exception.size(),
                              "%s",
                              exception.what());
            }
            catch (...)
            {
                save_exception_occurred = true;
                std::snprintf(save_exception.data(),
                              save_exception.size(),
                              "%s",
                              "unknown exception");
            }
            const auto busy_end = std::chrono::steady_clock::now();
            const auto busy_elapsed = busy_end - busy_start;
            if (!saved)
            {
                _failed = true;
            }

            {
                std::lock_guard<std::mutex> lock(_mutex);
                _busy += busy_elapsed;
                if (saved)
                {
                    ++_savedTasks;
                }
                else
                {
                    ++_failedTasks;
                }
                --_activeTasks;
                --_residentTasks;
                _residentBytes -= std::min(_residentBytes, task.residentBytes);
                _lastActivity = std::max(_lastActivity, busy_end);
                if (_tasks.empty() && _activeTasks == 0 &&
                    _producerReservations == 0)
                {
                    _idleCv.notify_all();
                }
            }
            _capacityCv.notify_all();
            if (save_exception_occurred)
            {
                // Accounting is already complete. Keep diagnostic formatting
                // behind a second exception boundary so low-memory logging can
                // never escape this std::thread.
                try
                {
                    LOG_ERROR(QStringLiteral(
                                  "[MVS] 深度帧 %1 保存线程异常：%2")
                                  .arg(task.frameIndex)
                                  .arg(QString::fromUtf8(save_exception.data())));
                }
                catch (...)
                {
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_tasks.empty() && _activeTasks == 0 &&
                _producerReservations == 0)
            {
                _idleCv.notify_all();
            }
        }
        _capacityCv.notify_all();
    }

    SaveFn _saveFn;
    std::deque<SaveTask> _tasks;
    mutable std::mutex _mutex;
    std::condition_variable _cv;
    std::condition_variable _capacityCv;
    std::condition_variable _idleCv;
    std::vector<std::thread> _workers;
    std::atomic<bool> _failed{false};
    size_t _workerCount = 1;
    size_t _maxResidentTasks = 1;
    uint64_t _maxResidentBytes = 1;
    uint64_t _producerReservationBytes = 1;
    uint64_t _residentBytes = 0;
    uint64_t _peakResidentBytes = 0;
    size_t _residentTasks = 0;
    size_t _producerReservations = 0;
    size_t _peakResidentTasks = 0;
    size_t _savedTasks = 0;
    size_t _failedTasks = 0;
    std::chrono::steady_clock::duration _enqueueWait{};
    std::chrono::steady_clock::duration _maxEnqueueWait{};
    std::chrono::steady_clock::duration _busy{};
    std::chrono::steady_clock::time_point _wallStart{};
    std::chrono::steady_clock::time_point _lastActivity{};
    bool _stopping = false;
    bool _dropPendingTasks = false;
    bool _summaryLogged = false;
    bool _wallStarted = false;
    int _activeTasks = 0;
};

bool writeFastDepthMatStorage(const std::string &path, const cv::Mat &matrix, std::string *errorMsg)
{
    const xjw::common::OperationResult result =
        xjw::core::project::writeDepthMatStorage(QString::fromStdString(path), matrix);
    if (!result.ok)
    {
        if (errorMsg)
        {
            *errorMsg = result.errorMessage.toStdString();
        }
        return false;
    }

    return true;
}

bool saveDepthPreviewPng(const std::string &path, const cv::Mat &depthMap, std::string *errorMsg)
{
    if (path.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "深度预览 PNG 路径为空";
        }
        return false;
    }

    if (depthMap.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "深度图为空，无法写入预览 PNG: " + path;
        }
        return false;
    }

    const int maxPreviewDimension = 2048;
    cv::Mat previewDepth = depthMap;
    const int maxDim = std::max(depthMap.cols, depthMap.rows);
    if (maxDim > maxPreviewDimension)
    {
        const double scale = static_cast<double>(maxPreviewDimension) / static_cast<double>(maxDim);
        cv::resize(depthMap, previewDepth, cv::Size(), scale, scale, cv::INTER_NEAREST);
    }

    const cv::Mat validMask = previewDepth > 0;
    cv::Mat vis = cv::Mat::zeros(previewDepth.size(), CV_8U);
    if (cv::countNonZero(validMask) > 0)
    {
        double dMin = 0.0;
        double dMax = 0.0;
        cv::minMaxLoc(previewDepth, &dMin, &dMax, nullptr, nullptr, validMask);
        if (dMax > dMin)
        {
            previewDepth.convertTo(vis, CV_8U, 255.0 / (dMax - dMin), -255.0 * dMin / (dMax - dMin));
        }
    }
    vis.setTo(0, previewDepth <= 0);

    cv::Mat colorVis;
    cv::applyColorMap(vis, colorVis, cv::COLORMAP_TURBO);
    colorVis.setTo(cv::Scalar(0, 0, 0), previewDepth <= 0);

    if (!xjw::common::io::writeImage(path, colorVis))
    {
        if (errorMsg)
        {
            *errorMsg = "无法写入深度预览 PNG: " + path;
        }
        return false;
    }

    return true;
}

QString manifestPathForOutput(const DepthGenConfig &config, const std::string &outputDir)
{
    QString dir;
    if (!config.intermediateDir.empty())
    {
        dir = QString::fromStdString(config.intermediateDir);
    }
    else if (!outputDir.empty())
    {
        dir = QString::fromStdString(outputDir);
    }

    if (dir.trimmed().isEmpty())
    {
        return {};
    }
    return QDir(dir).filePath(QStringLiteral("mvs_manifest.json"));
}

template <typename Fn>
void parallelForRows(int rowCount, int workerCount, Fn &&fn)
{
    if (rowCount <= 0)
    {
        return;
    }
    const int workers = std::clamp(std::max(1, workerCount), 1, rowCount);
    if (workers == 1)
    {
        for (int row = 0; row < rowCount; ++row)
        {
            fn(row);
        }
        return;
    }

    std::atomic<int> nextRow{0};
    xjw::common::concurrency::runWorkerGroup(
        static_cast<std::size_t>(workers),
        [&](std::stop_token stopToken)
    {
        while (!stopToken.stop_requested())
        {
            const int row = nextRow.fetch_add(1);
            if (row >= rowCount)
            {
                break;
            }
            fn(row);
        }
    });
}

int resolvedTotalCpuThreadBudget(const DepthGenConfig &config)
{
    const int hardware_threads = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency()));
    const int active_frame_workers = std::max(
        1, config.gpuFrameWorkerCount + config.cpuFrameWorkerCount);
    const int derived_budget = std::max(
        1, config.cpuWorkerCount * active_frame_workers);
    const int requested_budget = config.totalCpuThreadBudget > 0
        ? config.totalCpuThreadBudget : derived_budget;
    return std::clamp(requested_budget, 1, hardware_threads);
}

cv::Size patchMatchWorkSize(const cv::Mat &image, const PatchMatchConfig &config)
{
    const int ds = std::max(1, config.downsampleFactor);
    return cv::Size(std::max(1, image.cols / ds), std::max(1, image.rows / ds));
}

CrossViewHoleRepairOptions orbitalCrossViewHoleRepairOptions(
    const DepthGenConfig &config)
{
    CrossViewHoleRepairOptions options;
    options.minimumDistinctSourceCount = 3;
    options.maximumRelativeDepthSpread = 0.015f;
    options.maximumProjectionDistancePixels = 0.8f;
    options.maximumLocalRelativeDepthDifference = 0.035f;
    options.repairedConfidence = 0.70f;
    options.enableTwoSourceGrowth = config.enableTwoSourceCrossViewGrowth;
    options.maximumGrowthDistancePixels = config.twoSourceGrowthDistancePixels;
    options.maximumGrowthInverseDepthSpread =
        config.twoSourceGrowthInverseDepthSpread;
    options.maximumGrowthNormalAngleDegrees =
        config.twoSourceGrowthNormalAngleDegrees;
    options.maximumGrowthComponentArea =
        config.twoSourceGrowthMaximumComponentArea;
    options.includeValidNativeInterpolationAnchors = true;
    options.anchoredInterpolation.enabled = true;
    options.anchoredInterpolation.maximumComponentArea = 32000;
    options.anchoredInterpolation.maximumComponentAreaRatio = 0.25f;
    options.anchoredInterpolation.allowSilhouetteConnectedInterior = true;
    options.anchoredInterpolation.silhouetteProtectionRadiusPixels = 4;
    return options;
}

std::vector<int> consistencySourceIndicesForFrame(const std::vector<DepthFrameResult> &frames,
                                                  int refIdx,
                                                  int viewCount)
{
    std::vector<int> sources;
    if (refIdx >= 0 && refIdx < static_cast<int>(frames.size()))
    {
        for (int sourceIdx : frames[refIdx].sourceViewIndices)
        {
            if (sourceIdx < 0 || sourceIdx >= viewCount || sourceIdx == refIdx)
            {
                continue;
            }
            if (std::find(sources.begin(), sources.end(), sourceIdx) == sources.end())
            {
                sources.push_back(sourceIdx);
            }
        }
    }

    if (!sources.empty())
    {
        return sources;
    }

    sources.reserve(static_cast<size_t>(std::max(0, viewCount - 1)));
    for (int idx = 0; idx < viewCount; ++idx)
    {
        if (idx != refIdx)
        {
            sources.push_back(idx);
        }
    }
    return sources;
}

std::vector<int> orbitalHoleRepairSourceIndices(
    const std::vector<DepthFrameResult> &frames,
    const std::vector<int> &consistencySources,
    int refIdx,
    int viewCount,
    int requestedSourceCount)
{
    std::vector<bool> source_eligibility(
        static_cast<std::size_t>(std::max(0, viewCount)), false);
    const int available_frame_count = std::min(
        viewCount, static_cast<int>(frames.size()));
    for (int frame_index = 0; frame_index < available_frame_count; ++frame_index)
    {
        source_eligibility[static_cast<std::size_t>(frame_index)] =
            frames[static_cast<std::size_t>(frame_index)].eligibleAsConsistencySource();
    }
    return planMvsRepairSourceViews(
        consistencySources,
        source_eligibility,
        refIdx,
        requestedSourceCount);
}

bool isCudaMemoryFailure(const std::string &message)
{
    const std::string lower = asciiLowerCopy(message);

    return lower.find("out of memory") != std::string::npos
        || lower.find("cuda_error_memory") != std::string::npos
        || (lower.find("cuda") != std::string::npos && lower.find("memory") != std::string::npos);
}

PatchMatchConfig patchMatchConfigForRecordedWorker(
    PatchMatchConfig config,
    std::string_view workerId)
{
    std::optional<DepthComputeWorker> worker = depthComputeWorkerFromId(workerId);
    if (!worker && asciiLowerCopy(workerId) == "gpu")
    {
        worker = DepthComputeWorker{DepthComputeBackend::Cuda, -1};
    }
    if (!worker)
    {
        return config;
    }

    config.backend = worker->backend == DepthComputeBackend::Cuda
        ? PatchMatchBackend::Cuda
        : worker->backend == DepthComputeBackend::OpenCl
            ? PatchMatchBackend::OpenCl
            : PatchMatchBackend::Cpu;
    config.useCuda = worker->backend == DepthComputeBackend::Cuda;
    config.cudaDeviceIndex = worker->backend == DepthComputeBackend::Cuda
        ? worker->deviceIndex
        : -1;
    config.openClDeviceIndex = worker->backend == DepthComputeBackend::OpenCl
        ? worker->deviceIndex
        : -1;
    config.cudaFallbackToCpu = false;
    config.openClFallbackToCpu = false;
    return config;
}

bool estimatePatchMatchWithAdaptiveCuda(
    const char *stageLabel,
    int refIdx,
    const cv::Mat &refGray,
    const std::vector<cv::Mat> &srcGrays,
    const Camera &refCam,
    const std::vector<Camera> &srcCams,
    float zNear,
    float zFar,
    const PatchMatchConfig &config,
    cv::Mat &depthOut,
    cv::Mat *confOut,
    std::string *errorMsg,
    const cv::Mat *hintDepth,
    const cv::Mat *hintRadius,
    const cv::Mat *referenceValidMask,
    const std::vector<cv::Mat> *sourceValidMasks)
{
    const bool tryCuda =
        (config.backend == PatchMatchBackend::Cuda ||
         config.backend == PatchMatchBackend::Auto) &&
        PatchMatchDepthEstimator::isCudaAvailable();
    if (!tryCuda)
    {
        return PatchMatchDepthEstimator::estimate(refGray,
                                                  srcGrays,
                                                  refCam,
                                                  srcCams,
                                                  zNear,
                                                  zFar,
                                                  config,
                                                  depthOut,
                                                  confOut,
                                                  errorMsg,
                                                  hintDepth,
                                                  hintRadius,
                                                  referenceValidMask,
                                                  sourceValidMasks);
    }

    constexpr int kMaxCudaAttempts = 4;
    PatchMatchConfig attemptConfig = config;
    attemptConfig.cudaFallbackToCpu = false;
    std::string lastCudaError;

    for (int attempt = 0; attempt < kMaxCudaAttempts; ++attempt)
    {
        std::string attemptError;
        if (PatchMatchDepthEstimator::estimate(refGray,
                                               srcGrays,
                                               refCam,
                                               srcCams,
                                               zNear,
                                               zFar,
                                               attemptConfig,
                                               depthOut,
                                               confOut,
                                               &attemptError,
                                               hintDepth,
                                               hintRadius,
                                               referenceValidMask,
                                               sourceValidMasks))
        {
            if (attemptConfig.downsampleFactor != config.downsampleFactor)
            {
                LOG_INFO("[MVS][帧 %d][PatchMatch] %s CUDA 重试成功: ds=%d iterations=%d patch=%d",
                         refIdx,
                         stageLabel,
                         attemptConfig.downsampleFactor,
                         attemptConfig.numIterations,
                         attemptConfig.patchHalf * 2 + 1);
            }
            if (errorMsg)
            {
                errorMsg->clear();
            }
            return true;
        }

        lastCudaError = attemptError;
        if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
        {
            if (errorMsg)
            {
                *errorMsg = attemptError.empty() ? std::string("PatchMatch cancelled") : attemptError;
            }
            return false;
        }

        PatchMatchDepthEstimator::cleanupGpuImageCache();

        if (!isCudaMemoryFailure(attemptError) || attemptConfig.downsampleFactor >= 12)
        {
            break;
        }

        PatchMatchConfig nextConfig = DepthMapGenerator::nextCudaRetryPatchMatchConfig(attemptConfig,
                                                                                       refGray.cols,
                                                                                       refGray.rows);
        if (nextConfig.downsampleFactor <= attemptConfig.downsampleFactor)
        {
            break;
        }

        LOG_WARN("[MVS][帧 %d][PatchMatch] %s CUDA 显存不足，ds=%d -> %d 后重试: %s",
                 refIdx,
                 stageLabel,
                 attemptConfig.downsampleFactor,
                 nextConfig.downsampleFactor,
                 attemptError.c_str());

        attemptConfig = nextConfig;
        attemptConfig.cudaFallbackToCpu = false;
    }

    if (!config.cudaFallbackToCpu)
    {
        LOG_ERROR("[MVS][帧 %d][PatchMatch] %s CUDA 重试失败，配置禁止回退 CPU: %s",
                  refIdx,
                  stageLabel,
                  lastCudaError.empty() ? "未知 CUDA 错误" : lastCudaError.c_str());
        if (errorMsg)
        {
            *errorMsg = lastCudaError.empty()
                ? std::string("CUDA PatchMatch retries exhausted")
                : lastCudaError;
        }
        return false;
    }

    LOG_WARN("[MVS][帧 %d][PatchMatch] %s CUDA 重试失败，按配置回退 CPU: %s",
             refIdx,
             stageLabel,
             lastCudaError.empty() ? "未知 CUDA 错误" : lastCudaError.c_str());

    PatchMatchConfig cpuConfig = config;
    cpuConfig.useCuda = false;
    cpuConfig.backend = PatchMatchBackend::Cpu;
    cpuConfig.cudaFallbackToCpu = false;
    const bool cpuOk = PatchMatchDepthEstimator::estimate(refGray,
                                                          srcGrays,
                                                          refCam,
                                                          srcCams,
                                                          zNear,
                                                          zFar,
                                                          cpuConfig,
                                                          depthOut,
                                                          confOut,
                                                          errorMsg,
                                                          hintDepth,
                                                          hintRadius,
                                                          referenceValidMask,
                                                          sourceValidMasks);
    if (!cpuOk && errorMsg && errorMsg->empty())
    {
        *errorMsg = lastCudaError;
    }
    return cpuOk;
}

float sourceGeometryReliabilityWeight(const DepthFrameResult &reference_frame,
                                      int source_view_index)
{
    const auto entry = std::find_if(
        reference_frame.sourceViewPlan.cbegin(),
        reference_frame.sourceViewPlan.cend(),
        [source_view_index](const MvsSourcePlanEntry &candidate)
        {
            return candidate.viewIndex == source_view_index;
        });
    if (entry == reference_frame.sourceViewPlan.cend() ||
        !std::isfinite(entry->sourceQualityScore) ||
        entry->sourceQualityScore <= 0.0f)
    {
        return 1.0f;
    }
    return std::clamp(entry->sourceQualityScore, 0.05f, 1.0f);
}

int cameraBaselineSector(const Camera &reference_camera,
                         const Camera &source_camera)
{
    const std::array<double, 3> reference_center =
        reference_camera.cameraCenter();
    const std::array<double, 3> source_center = source_camera.cameraCenter();
    std::array<double, 3> delta{
        source_center[0] - reference_center[0],
        source_center[1] - reference_center[1],
        source_center[2] - reference_center[2]};
    int dominant_axis = 0;
    for (int axis = 1; axis < 3; ++axis)
    {
        if (std::fabs(delta[static_cast<std::size_t>(axis)]) >
            std::fabs(delta[static_cast<std::size_t>(dominant_axis)]))
        {
            dominant_axis = axis;
        }
    }
    return dominant_axis * 2 +
        (delta[static_cast<std::size_t>(dominant_axis)] >= 0.0 ? 1 : 0);
}

struct DepthConsistencySourceInput
{
    cv::Mat depth;
    Camera camera;
    cv::Mat confidence;
    float reliabilityWeight = 1.0f;
    int sourceOrdinal = -1;
};

void accumulateDepthConsistency(const cv::Mat &referenceDepth,
                                const Camera &referenceCamera,
                                const std::vector<DepthConsistencySourceInput> &sources,
                                float relativeThreshold,
                                int rowWorkers,
                                const std::atomic<bool> &cancelled,
                                cv::Mat &consistentVotes,
                                cv::Mat &occludedVotes,
                                cv::Mat &contradictedVotes,
                                cv::Mat &unverifiableVotes,
                                cv::Mat &geometrySourceMask,
                                cv::Mat &sourceInverseDepthSum,
                                cv::Mat &sourceInverseDepthSquaredSum,
                                AdaptiveGeometryEvidenceAccumulatorMaps *adaptiveEvidence)
{
    if (sources.empty())
    {
        return;
    }
    const bool accumulate_adaptive_evidence =
        adaptiveEvidence &&
        adaptiveEvidence->positiveSupport.type() == CV_32FC1 &&
        adaptiveEvidence->squaredPositiveSupport.type() == CV_32FC1 &&
        adaptiveEvidence->conflict.type() == CV_32FC1 &&
        adaptiveEvidence->observable.type() == CV_32FC1 &&
        adaptiveEvidence->positiveSupport.size() == referenceDepth.size() &&
        adaptiveEvidence->squaredPositiveSupport.size() == referenceDepth.size() &&
        adaptiveEvidence->conflict.size() == referenceDepth.size() &&
        adaptiveEvidence->observable.size() == referenceDepth.size();
    parallelForRows(referenceDepth.rows, rowWorkers, [&](int row)
    {
        if (cancelled.load())
        {
            return;
        }
        const float *reference_row = referenceDepth.ptr<float>(row);
        uint16_t *consistent_row = consistentVotes.ptr<uint16_t>(row);
        uint16_t *occluded_row = occludedVotes.ptr<uint16_t>(row);
        uint16_t *contradicted_row = contradictedVotes.ptr<uint16_t>(row);
        uint16_t *unverifiable_row = unverifiableVotes.ptr<uint16_t>(row);
        uint16_t *source_mask_row = geometrySourceMask.ptr<uint16_t>(row);
        float *inverse_sum_row = sourceInverseDepthSum.ptr<float>(row);
        float *inverse_squared_sum_row = sourceInverseDepthSquaredSum.ptr<float>(row);
        float *adaptive_positive_row = accumulate_adaptive_evidence
            ? adaptiveEvidence->positiveSupport.ptr<float>(row) : nullptr;
        float *adaptive_squared_row = accumulate_adaptive_evidence
            ? adaptiveEvidence->squaredPositiveSupport.ptr<float>(row) : nullptr;
        float *adaptive_conflict_row = accumulate_adaptive_evidence
            ? adaptiveEvidence->conflict.ptr<float>(row) : nullptr;
        float *adaptive_observable_row = accumulate_adaptive_evidence
            ? adaptiveEvidence->observable.ptr<float>(row) : nullptr;
        for (int column = 0; column < referenceDepth.cols; ++column)
        {
            if (cancelled.load())
            {
                return;
            }
            const float reference_depth = reference_row[column];
            if (reference_depth <= 0.0f)
            {
                continue;
            }
            const cv::Point2f reference_pixel(
                static_cast<float>(column), static_cast<float>(row));
            const double pixel[2] = {
                static_cast<double>(reference_pixel.x),
                static_cast<double>(reference_pixel.y)};
            double world[3] = {0.0, 0.0, 0.0};
            if (!referenceCamera.unprojectPixel(pixel, reference_depth, world))
            {
                // Preserve the former per-source evaluation semantics on the
                // exceptional unprojection failure path: every source would
                // have contributed one unverifiable observation.
                unverifiable_row[column] = static_cast<std::uint16_t>(
                    unverifiable_row[column] + sources.size());
                continue;
            }
            const std::array<double, 3> reference_world = {
                world[0], world[1], world[2]};

            // Preserve the source-plan order for each pixel.  This keeps the
            // exact floating accumulation order while amortizing reference
            // unprojection and row-worker dispatch across all source views.
            for (const DepthConsistencySourceInput &source : sources)
            {
                const ProjectedDepthConsistencyResult result =
                    evaluateProjectedDepthConsistencyFromReferenceWorld(
                        referenceCamera,
                        reference_pixel,
                        reference_depth,
                        reference_world,
                        source.camera,
                        source.depth,
                        relativeThreshold,
                        1,
                        3.0f,
                        accumulate_adaptive_evidence);
                if (accumulate_adaptive_evidence)
                {
                    AdaptiveGeometryEvidenceObservation observation;
                    observation.worldResidual = result.worldSurfaceResidual;
                    observation.worldPixelFootprint = result.jointWorldPixelFootprint;
                    observation.roundTripResidualPixels = result.roundTripErrorPixels;
                    observation.reliabilityWeight = std::clamp(
                        source.reliabilityWeight, 0.0f, 1.0f);
                    if (!source.confidence.empty() &&
                        source.confidence.type() == CV_32FC1 &&
                        source.confidence.size() == source.depth.size() &&
                        result.sourcePixel.x >= 0 &&
                        result.sourcePixel.x < source.confidence.cols &&
                        result.sourcePixel.y >= 0 &&
                        result.sourcePixel.y < source.confidence.rows)
                    {
                        const float source_confidence =
                            source.confidence.at<float>(
                                result.sourcePixel.y, result.sourcePixel.x);
                        observation.reliabilityWeight *=
                            std::isfinite(source_confidence)
                            ? std::clamp(source_confidence, 0.0f, 1.0f)
                            : 0.0f;
                    }
                    observation.evidenceClass =
                        adaptiveGeometryEvidenceClass(result);
                    AdaptiveGeometryEvidenceAccumulator accumulator;
                    accumulator.positiveSupport = adaptive_positive_row[column];
                    accumulator.squaredPositiveSupport =
                        adaptive_squared_row[column];
                    accumulator.conflict = adaptive_conflict_row[column];
                    accumulator.observable = adaptive_observable_row[column];
                    accumulator.add(observation);
                    adaptive_positive_row[column] = accumulator.positiveSupport;
                    adaptive_squared_row[column] =
                        accumulator.squaredPositiveSupport;
                    adaptive_conflict_row[column] = accumulator.conflict;
                    adaptive_observable_row[column] = accumulator.observable;
                }
                switch (result.evidence)
                {
                case DepthConsistencyEvidence::Consistent:
                    ++consistent_row[column];
                    if (source.sourceOrdinal >= 0 &&
                        source.sourceOrdinal < 16)
                    {
                        source_mask_row[column] = static_cast<std::uint16_t>(
                            source_mask_row[column] |
                            (static_cast<std::uint16_t>(1U) <<
                             source.sourceOrdinal));
                    }
                    if (result.consistentReferenceDepth > 0.0f &&
                        std::isfinite(result.consistentReferenceDepth))
                    {
                        const float inverse_depth =
                            1.0f / result.consistentReferenceDepth;
                        inverse_sum_row[column] += inverse_depth;
                        inverse_squared_sum_row[column] +=
                            inverse_depth * inverse_depth;
                    }
                    break;
                case DepthConsistencyEvidence::Occluded:
                    ++occluded_row[column];
                    break;
                case DepthConsistencyEvidence::Contradicted:
                    ++contradicted_row[column];
                    break;
                case DepthConsistencyEvidence::Unverifiable:
                default:
                    ++unverifiable_row[column];
                    break;
                }
            }
        }
    });
}

cv::Mat makeDepthConsistencyMask(const cv::Mat &referenceDepth,
                                 int sourceViewCount,
                                 int minimumSourceConfirmations,
                                 const cv::Mat &consistentVotes,
                                 const cv::Mat &occludedVotes,
                                 const cv::Mat &contradictedVotes,
                                 int rowWorkerCount,
                                 const std::atomic<bool> *cancelled)
{
    cv::Mat mask(referenceDepth.size(), CV_8U, cv::Scalar(0));
    parallelForRows(referenceDepth.rows, rowWorkerCount, [&](int row)
    {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
        {
            return;
        }
        const float *depth_row = referenceDepth.ptr<float>(row);
        const uint16_t *consistent_row = consistentVotes.ptr<uint16_t>(row);
        const uint16_t *occluded_row = occludedVotes.ptr<uint16_t>(row);
        const uint16_t *contradicted_row = contradictedVotes.ptr<uint16_t>(row);
        uint8_t *mask_row = mask.ptr<uint8_t>(row);
        for (int column = 0; column < referenceDepth.cols; ++column)
        {
            if ((column & 63) == 0 && cancelled &&
                cancelled->load(std::memory_order_relaxed))
            {
                break;
            }
            if (depth_row[column] <= 0.0f)
            {
                continue;
            }
            if (shouldRetainDepthFromConsistencyVotes(
                    sourceViewCount,
                    consistent_row[column],
                    occluded_row[column],
                    contradicted_row[column],
                    minimumSourceConfirmations))
            {
                mask_row[column] = 255;
            }
        }
    });
    return mask;
}

struct DepthConsistencyVoteTotals
{
    std::uint64_t consistent = 0;
    std::uint64_t occluded = 0;
    std::uint64_t contradicted = 0;
    std::uint64_t unverifiable = 0;
};

DepthConsistencyVoteTotals summarizeDepthConsistencyVotes(
    const cv::Mat &consistentVotes,
    const cv::Mat &occludedVotes,
    const cv::Mat &contradictedVotes,
    const cv::Mat &unverifiableVotes,
    int rowWorkerCount)
{
    std::array<std::atomic<std::uint64_t>, 4> totals{};
    parallelForRows(consistentVotes.rows, rowWorkerCount, [&](int row)
    {
        const std::uint16_t *consistent_row =
            consistentVotes.ptr<std::uint16_t>(row);
        const std::uint16_t *occluded_row =
            occludedVotes.ptr<std::uint16_t>(row);
        const std::uint16_t *contradicted_row =
            contradictedVotes.ptr<std::uint16_t>(row);
        const std::uint16_t *unverifiable_row =
            unverifiableVotes.ptr<std::uint16_t>(row);
        std::array<std::uint64_t, 4> row_totals{};
        for (int column = 0; column < consistentVotes.cols; ++column)
        {
            row_totals[0] += consistent_row[column];
            row_totals[1] += occluded_row[column];
            row_totals[2] += contradicted_row[column];
            row_totals[3] += unverifiable_row[column];
        }
        for (std::size_t index = 0; index < totals.size(); ++index)
        {
            totals[index].fetch_add(
                row_totals[index], std::memory_order_relaxed);
        }
    });
    return {
        totals[0].load(std::memory_order_relaxed),
        totals[1].load(std::memory_order_relaxed),
        totals[2].load(std::memory_order_relaxed),
        totals[3].load(std::memory_order_relaxed)};
}

void updateDepthCompletenessAfterPostprocess(DepthFrameResult &result,
                                             const cv::Mat &depth,
                                             const DepthPostProcessStats &stats)
{
    result.depthCompleteness.preFusionPostprocessValidCount =
        stats.validBeforePostprocess;
    result.depthCompleteness.postConfidenceFilterValidCount =
        stats.validAfterConfidenceFilter;
    result.depthCompleteness.postFusionPostprocessValidCount =
        stats.validAfterPostprocess;
    if (result.depthCompleteness.fusionPostprocessRetentionRatio < 0.0f &&
        stats.validBeforePostprocess > 0)
    {
        result.depthCompleteness.fusionPostprocessRetentionRatio =
            static_cast<float>(stats.validAfterPostprocess) /
            static_cast<float>(stats.validBeforePostprocess);
    }
    if (!result.supportRegionMask || result.supportRegionMask->empty())
    {
        return;
    }
    cv::Mat effective_mask = *result.supportRegionMask;
    if (effective_mask.size() != depth.size())
    {
        cv::resize(effective_mask,
                   effective_mask,
                   depth.size(),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);
    }
    result.depthCompleteness.finalMetrics = analyzeDepthCompleteness(
        depth, effective_mask);
}

DepthAnchoredHoleInterpolationStats repairPostprocessedInternalDepthHoles(
    DepthFrameResult &result,
    cv::Mat &depth,
    cv::Mat &confidence,
    MvsSceneProfile sceneProfile,
    cv::Mat *anchoredInterpolationMask = nullptr)
{
    if (sceneProfile != MvsSceneProfile::OrbitalObject ||
        !result.supportRegionMask || result.supportRegionMask->empty() ||
        !result.crossViewRepairedMask || result.crossViewRepairedMask->empty())
    {
        return {};
    }
    cv::Mat support_mask = *result.supportRegionMask;
    if (support_mask.size() != depth.size())
    {
        cv::resize(
            support_mask, support_mask, depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::Mat anchor_mask = depth > 0.0f;
    if (result.crossViewRepairedMask->size() != depth.size())
    {
        cv::resize(*result.crossViewRepairedMask,
                   *result.crossViewRepairedMask,
                   depth.size(),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);
    }
    DepthAnchoredHoleInterpolationOptions options;
    options.enabled = true;
    options.maximumComponentArea = 32000;
    options.maximumComponentAreaRatio = 0.25f;
    options.allowSilhouetteConnectedInterior = true;
    options.silhouetteProtectionRadiusPixels = 4;
    cv::Mat local_interpolation_mask;
    cv::Mat *interpolation_mask = anchoredInterpolationMask
        ? anchoredInterpolationMask : &local_interpolation_mask;
    *interpolation_mask = cv::Mat(
        depth.size(), CV_8UC1, cv::Scalar(0));
    const DepthAnchoredHoleInterpolationStats stats =
        interpolateAnchoredInternalDepthHoles(
        depth,
        support_mask,
        anchor_mask,
        nullptr,
        options,
        confidence.empty() ? nullptr : &confidence,
        interpolation_mask);
    cv::bitwise_or(*result.crossViewRepairedMask,
                   *interpolation_mask,
                   *result.crossViewRepairedMask);
    return stats;
}

void updateDepthFrameQualityAfterConsistency(DepthFrameResult &result,
                                             const cv::Mat &depth,
                                             const cv::Mat &confidence,
                                             MvsSceneProfile scene_profile,
                                             DepthFilterMode filter_mode,
                                             float consistency_keep_rate)
{
    // Frame acceptance describes the final depth product that enters fusion.
    // Repaired pixels retain their per-pixel provenance and are guarded again by
    // TSDF geometry support, so removing them here would penalize the same
    // evidence twice and could downgrade an otherwise complete orbital sequence.
    const cv::Mat &quality_depth = depth;
    const cv::Mat &quality_confidence = confidence;
    const float quality_consistency_keep_rate = consistency_keep_rate;
    const float search_boundary_ratio =
        result.qualityMetrics.depthAtSearchBoundaryRatio;
    result.qualityMetrics = analyzeDepthMapQuality(
        quality_depth,
        quality_confidence,
        static_cast<int>(result.sourceViewIndices.size()));
    result.qualityMetrics.depthAtSearchBoundaryRatio = search_boundary_ratio;

    DepthFrameQualityInput quality_input;
    quality_input.sceneProfile = scene_profile;
    quality_input.filterMode = filter_mode;
    quality_input.sourceViewCount = static_cast<int>(result.sourceViewIndices.size());
    quality_input.validCoverage = result.qualityMetrics.validCoverage;
    quality_input.largestComponentRatio = result.qualityMetrics.largestComponentRatio;
    quality_input.meanConfidence = result.qualityMetrics.meanConfidence;
    quality_input.multiViewConsistency = std::clamp(
        quality_consistency_keep_rate, 0.0f, 1.0f);
    quality_input.depthAtSearchBoundaryRatio = search_boundary_ratio;
    quality_input.hasConstrainedSupportMask =
        result.maskSource != "full_image" && result.maskCoverage < 0.999f;
    if (result.supportRegionMask && !result.supportRegionMask->empty())
    {
        cv::Mat effective_mask = *result.supportRegionMask;
        if (effective_mask.size() != quality_depth.size())
        {
            cv::resize(effective_mask,
                       effective_mask,
                       quality_depth.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        result.depthCompleteness.finalMetrics = analyzeDepthCompleteness(
            quality_depth, effective_mask);
        quality_input.validWithinMaskRatio =
            result.depthCompleteness.finalMetrics.validInputs
            ? result.depthCompleteness.finalMetrics.validWithinMaskRatio
            : -1.0f;
    }
    quality_input.outputFilterRetentionRatio =
        result.depthCompleteness.outputFilterRetentionRatio;
    quality_input.consistencyRetentionRatio = quality_consistency_keep_rate;
    quality_input.fusionPostprocessRetentionRatio =
        result.depthCompleteness.fusionPostprocessRetentionRatio;
    result.qualityDecision = evaluateDepthFrame(quality_input);
}

class AdaptivePatchMatchBackend final : public IPatchMatchBackend
{
public:
    explicit AdaptivePatchMatchBackend(int ref_idx)
        : _refIdx(ref_idx)
    {
    }

    bool estimate(const PatchMatchBackendRequest &request,
                  DepthLevelResult &result,
                  std::string *error_message) override
    {
        const std::string stage_label = "Level " + std::to_string(request.levelConfig.level) + " PatchMatch";
        cv::Mat confidence;
        const cv::Mat *hint = request.prior && !request.prior->center.empty()
            ? &request.prior->center
            : nullptr;
        const cv::Mat *radius = request.prior && !request.prior->radius.empty()
            ? &request.prior->radius
            : nullptr;
        LOG_DEBUG("[MVS][帧 %d][PatchMatch] %s ds=%d iterations=%d patch=%d",
                  _refIdx,
                  stage_label.c_str(),
                  request.levelConfig.patchMatch.downsampleFactor,
                  request.levelConfig.patchMatch.numIterations,
                  request.levelConfig.patchMatch.patchHalf * 2 + 1);

        if (!estimatePatchMatchWithAdaptiveCuda(stage_label.c_str(),
                                                _refIdx,
                                                request.referenceImage,
                                                request.sourceImages,
                                                request.referenceCamera,
                                                request.sourceCameras,
                                                request.zNear,
                                                request.zFar,
                                                request.levelConfig.patchMatch,
                                                result.depth,
                                                &confidence,
                                                error_message,
                                                hint,
                                                radius,
                                                request.referenceValidMask.empty()
                                                    ? nullptr
                                                    : &request.referenceValidMask,
                                                request.sourceValidMasks.empty()
                                                    ? nullptr
                                                    : &request.sourceValidMasks))
        {
            return false;
        }

        result.level = request.levelConfig.level;
        result.downsampleFactor = request.levelConfig.patchMatch.downsampleFactor;
        result.confidence = confidence;
        result.validMask = result.depth > 0.0f;
        result.supportCount = cv::Mat(result.depth.size(), CV_16U, cv::Scalar(0));
        result.supportCount.setTo(
            cv::Scalar(static_cast<int>(request.sourceImages.size())), result.validMask);
        result.uncertainty = cv::Mat(result.depth.size(), CV_32F, cv::Scalar(0.0f));
        for (int row = 0; row < result.depth.rows; ++row)
        {
            for (int column = 0; column < result.depth.cols; ++column)
            {
                const float depth = result.depth.at<float>(row, column);
                if (depth <= 0.0f || !std::isfinite(depth))
                {
                    continue;
                }
                const float confidence_value = confidence.empty()
                    ? 0.5f
                    : std::clamp(confidence.at<float>(row, column), 0.0f, 1.0f);
                result.uncertainty.at<float>(row, column) =
                    std::max(0.001f, depth * 0.05f * (1.0f - confidence_value));
            }
        }
        return true;
    }

private:
    int _refIdx = -1;
};

} // namespace

cv::Mat DepthMapGenerator::buildContentMask(const cv::Mat &gray,
                                            float *coverage,
                                            double *otsuThreshold,
                                            int *adaptiveThreshold)
{
    if (gray.empty())
    {
        if (coverage)
        {
            *coverage = 0.f;
        }
        return cv::Mat();
    }

    cv::Mat blurOrig;
    cv::GaussianBlur(gray, blurOrig, cv::Size(15, 15), 0);

    cv::Mat otsuBin;
    const double otsuThresh = cv::threshold(blurOrig, otsuBin, 0, 255,
                                            cv::THRESH_BINARY | cv::THRESH_OTSU);
    const int adaptiveThresh = std::max(8, static_cast<int>(otsuThresh * 0.3));

    cv::Mat mask = (blurOrig > adaptiveThresh);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    const int totalPx = gray.rows * gray.cols;
    const float maskCoverage = totalPx > 0
        ? static_cast<float>(cv::countNonZero(mask)) / static_cast<float>(totalPx)
        : 0.f;

    if (coverage)
    {
        *coverage = maskCoverage;
    }
    if (otsuThreshold)
    {
        *otsuThreshold = otsuThresh;
    }
    if (adaptiveThreshold)
    {
        *adaptiveThreshold = adaptiveThresh;
    }

    if (maskCoverage >= kSkipContentMaskCoverage)
    {
        return cv::Mat();
    }

    return mask;
}

cv::Mat DepthMapGenerator::projectMaskToValidMask(const cv::Mat &projectMask,
                                                  cv::Size targetSize)
{
    if (projectMask.empty() || targetSize.width <= 0 || targetSize.height <= 0)
    {
        return cv::Mat();
    }

    cv::Mat gray_mask;
    if (projectMask.channels() == 1)
    {
        gray_mask = projectMask;
    }
    else
    {
        cv::cvtColor(projectMask,
                     gray_mask,
                     projectMask.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    }
    if (gray_mask.depth() != CV_8U)
    {
        gray_mask.convertTo(gray_mask, CV_8U);
    }
    if (gray_mask.size() != targetSize)
    {
        cv::resize(gray_mask, gray_mask, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
    }

    cv::Mat valid_mask;
    cv::compare(gray_mask, 0, valid_mask, cv::CMP_EQ);
    return valid_mask;
}

cv::Mat DepthMapGenerator::refineOrbitalProjectValidMask(const cv::Mat &gray,
                                                         const cv::Mat &projectValidMask,
                                                         bool *refined,
                                                         float *retainedRatio)
{
    if (refined)
    {
        *refined = false;
    }
    if (retainedRatio)
    {
        *retainedRatio = 1.0f;
    }
    if (gray.empty() || projectValidMask.empty() ||
        gray.type() != CV_8UC1 || projectValidMask.type() != CV_8UC1 ||
        gray.size() != projectValidMask.size())
    {
        return projectValidMask.clone();
    }

    const int valid_pixels = cv::countNonZero(projectValidMask);
    if (valid_pixels <= 0)
    {
        return projectValidMask.clone();
    }

    cv::Mat excluded_mask;
    cv::compare(projectValidMask, 0, excluded_mask, cv::CMP_EQ);
    const int excluded_pixels = cv::countNonZero(excluded_mask);
    const double excluded_mean = excluded_pixels > 0
        ? cv::mean(gray, excluded_mask)[0]
        : 255.0;
    if (excluded_mean > 45.0)
    {
        return projectValidMask.clone();
    }

    cv::Mat content_mask = buildContentMask(gray);
    if (content_mask.empty())
    {
        return projectValidMask.clone();
    }

    cv::Mat protected_interior;
    const cv::Mat boundary_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::erode(projectValidMask, protected_interior, boundary_kernel);

    cv::Mat content_excluded;
    cv::compare(content_mask, 0, content_excluded, cv::CMP_EQ);
    cv::Mat removable;
    cv::bitwise_and(protected_interior, content_excluded, removable);
    if (cv::countNonZero(removable) == 0)
    {
        return projectValidMask.clone();
    }

    cv::Mat candidate = projectValidMask.clone();
    candidate.setTo(0, removable);
    const int retained_pixels = cv::countNonZero(candidate);
    const float retained_ratio = static_cast<float>(retained_pixels) /
                                 static_cast<float>(valid_pixels);
    if (retainedRatio)
    {
        *retainedRatio = retained_ratio;
    }
    if (retained_ratio < 0.75f)
    {
        return projectValidMask.clone();
    }

    if (refined)
    {
        *refined = retained_pixels < valid_pixels;
    }
    return candidate;
}

PatchMatchConfig DepthMapGenerator::nextCudaRetryPatchMatchConfig(const PatchMatchConfig &config,
                                                                  int imageWidth,
                                                                  int imageHeight)
{
    PatchMatchConfig retryConfig = config;
    const int current = std::max(1, retryConfig.downsampleFactor);
    int next = current < 4 ? current + 1 : current + 2;

    const int maxDim = std::max(imageWidth, imageHeight);
    if (maxDim >= 5000 && current <= 2)
    {
        next = std::max(next, 3);
    }

    retryConfig.downsampleFactor = std::min(next, 12);
    retryConfig.numIterations = std::max(1, retryConfig.numIterations - 1);
    retryConfig.patchHalf = std::max(3, retryConfig.patchHalf - 1);
    return retryConfig;
}

// =============================================================================
// 辅助：旋转矩阵行列式
// =============================================================================
static double det3(const double *R)
{
    return R[0] * (R[4] * R[8] - R[5] * R[7])
           - R[1] * (R[3] * R[8] - R[5] * R[6])
           + R[2] * (R[3] * R[7] - R[4] * R[6]);
}

// =============================================================================
/**
 * @brief 默认构造深度图生成器。
 *
 * 仅初始化 Qt 元类型注册；实际数据通过 `setViews()`、`setSparseCloud()`
 * 与 `setConfig()` 注入。
 */
DepthMapGenerator::DepthMapGenerator(QObject *parent)
    : QObject(parent)
{
    qRegisterMetaType<DepthFrameResult>("DepthFrameResult");
    qRegisterMetaType<QSharedPointer<cv::Mat>>("QSharedPointer<cv::Mat>");
    qRegisterMetaType<std::vector<DensePoint>>("std::vector<DensePoint>");
}

DepthMapGenerator::~DepthMapGenerator()
{
    requestCancel();
    if (_backgroundFuture.isRunning())
    {
        _backgroundFuture.waitForFinished();
    }
    PatchMatchDepthEstimator::cleanupGpuImageCache();
    PatchMatchDepthEstimator::cleanupOpenClResources();
}

// =============================================================================
void DepthMapGenerator::setViews(const std::vector<CameraView> &views)
{
    _views = views;
    _skipFrameMask.assign(_views.size(), 0);
    clearFrameCaches();
}

void DepthMapGenerator::setSparseCloud(const SparseCloud &sparse)
{
    _sparse = sparse;
    clearFrameCaches();
}

void DepthMapGenerator::setConfig(const DepthGenConfig &config)
{
    _config = config;
}

void DepthMapGenerator::setSkippedFrameIndices(const std::vector<int> &indices)
{
    _skipFrameMask.assign(_views.size(), 0);
    for (int index : indices)
    {
        if (index >= 0 && index < static_cast<int>(_skipFrameMask.size()))
        {
            _skipFrameMask[static_cast<size_t>(index)] = 1;
        }
    }
}

void DepthMapGenerator::initializeWorkspaceManifest()
{
    std::vector<QJsonObject> reusableArtifacts;
    int manifestSkipCount = 0;
    QString manifestPathForLog;
    QString depthConfigHashForLog;

    {
        std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
        _workspaceManifestPath = manifestPathForOutput(_config, _outputDir);
        _depthConfigHash = makeMvsDepthInputHash(_config, _views, _sparse);
        manifestPathForLog = _workspaceManifestPath;
        depthConfigHashForLog = _depthConfigHash;
        _workspaceManifest.clear();
        _workspaceManifest.setConfigHash(_depthConfigHash);

        if (_workspaceManifestPath.isEmpty())
        {
            return;
        }

        QString error;
        if (QFile::exists(_workspaceManifestPath))
        {
            MvsWorkspaceManifest loaded;
            if (loaded.load(_workspaceManifestPath, &error))
            {
                if (loaded.configHash() == _depthConfigHash)
                {
                    _workspaceManifest = loaded;
                }
                else
                {
                    LOG_INFO(QStringLiteral("[MVS] 深度图 manifest 参数已变化，旧记录不复用: %1")
                                 .arg(QDir::toNativeSeparators(_workspaceManifestPath)));
                }
            }
            else
            {
                LOG_WARN(QStringLiteral("[MVS] 读取深度图 manifest 失败，将重建: %1 error=%2")
                             .arg(QDir::toNativeSeparators(_workspaceManifestPath), error));
            }
        }

        _workspaceManifest.setConfigHash(_depthConfigHash);
        if (_skipFrameMask.size() != _views.size())
        {
            _skipFrameMask.assign(_views.size(), 0);
        }
        for (int i = 0; i < static_cast<int>(_views.size()); ++i)
        {
            if (_workspaceManifest.hasReusableCompletedFrame(i, _depthConfigHash))
            {
                _skipFrameMask[static_cast<size_t>(i)] = 1;
                ++manifestSkipCount;

                for (const MvsDepthFrameRecord &record : _workspaceManifest.frames())
                {
                    if (record.refIndex == i)
                    {
                        QJsonObject artifact = record.toJson();
                        artifact[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
                        artifact[QStringLiteral("manifest_path")] = _workspaceManifestPath;
                        reusableArtifacts.push_back(artifact);
                        break;
                    }
                }
            }
        }

        LOG_INFO(QStringLiteral("[MVS] 深度图 manifest: path=%1 config=%2 reusable=%3")
                     .arg(QDir::toNativeSeparators(_workspaceManifestPath))
                     .arg(_depthConfigHash.left(12))
                     .arg(manifestSkipCount));

        if (manifestSkipCount > 0 && _config.runFusion)
        {
            LOG_WARN(QStringLiteral(
                "[MVS] 检测到 %1 个可复用深度帧，本次自动切换为深度图续跑模式；"
                "完成后请使用已保存深度图运行融合，避免只融合本次新计算的子集。")
                         .arg(manifestSkipCount));
            _config.runFusion = false;
        }
    }

    for (const QJsonObject &artifact : reusableArtifacts)
    {
        emit depthMapArtifactSaved(artifact);
    }
    if (!reusableArtifacts.empty())
    {
        LOG_INFO(QStringLiteral("[MVS] 已从 manifest 回灌 %1 个深度图记录到项目元数据: %2 config=%3")
                     .arg(static_cast<int>(reusableArtifacts.size()))
                     .arg(QDir::toNativeSeparators(manifestPathForLog))
                     .arg(depthConfigHashForLog.left(12)));
    }
}

bool DepthMapGenerator::persistWorkspaceManifest(QString *errorMsg)
{
    if (_workspaceManifestPath.isEmpty())
    {
        return true;
    }
    return _workspaceManifest.saveAtomic(_workspaceManifestPath, errorMsg);
}

void DepthMapGenerator::markManifestFrameRunning(int frameIndex)
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
    if (_workspaceManifestPath.isEmpty())
    {
        return;
    }

    _workspaceManifest.markRunning(frameIndex,
                                    QString::fromStdString(_views[frameIndex].imagePath),
                                    _depthConfigHash);
    QString error;
    if (!persistWorkspaceManifest(&error))
    {
        LOG_WARN(QStringLiteral("[MVS] 写入运行中 manifest 失败: %1").arg(error));
    }
}

void DepthMapGenerator::markManifestFrameFailed(int frameIndex, const QString &error)
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
    if (_workspaceManifestPath.isEmpty())
    {
        return;
    }

    _workspaceManifest.markFailed(frameIndex, error);
    QString saveError;
    if (!persistWorkspaceManifest(&saveError))
    {
        LOG_WARN(QStringLiteral("[MVS] 写入失败 manifest 失败: %1").arg(saveError));
    }
}

void DepthMapGenerator::clearFrameCaches()
{
    _frameCaches.clear();
    _visibilityBits.clear();
    _visibilityAdjacency.clear();
    _visibilityWordCount = 0;
    _frameCachesReady = false;
}

bool DepthMapGenerator::isSparsePointVisibleInFrame(int viewIdx, size_t pointIndex) const
{
    if (!_frameCachesReady
        || viewIdx < 0
        || viewIdx >= static_cast<int>(_frameCaches.size())
        || _visibilityWordCount == 0
        || pointIndex >= _sparse.points.size())
    {
        return false;
    }

    const size_t word = pointIndex / 64;
    const size_t bit = pointIndex % 64;
    const size_t offset = static_cast<size_t>(viewIdx) * _visibilityWordCount + word;
    if (offset >= _visibilityBits.size())
    {
        return false;
    }
    return (_visibilityBits[offset] & (uint64_t{1} << bit)) != 0;
}

void DepthMapGenerator::prepareFrameCaches()
{
    if (_cancelled.load(std::memory_order_relaxed))
    {
        clearFrameCaches();
        return;
    }
    if (_frameCachesReady)
    {
        return;
    }

    const int NV = static_cast<int>(_views.size());
    const size_t pointCount = _sparse.points.size();
    _frameCaches.assign(static_cast<size_t>(std::max(0, NV)), FrameMvsCache{});
    _visibilityWordCount = 0;
    _visibilityBits.clear();
    _visibilityAdjacency.assign(static_cast<size_t>(std::max(0, NV)), {});

    if (NV <= 0 || pointCount == 0)
    {
        _frameCachesReady = true;
        return;
    }

    const auto start = Clock::now();
    constexpr size_t kParallelVisibilityPointThreshold = 20000;
    std::vector<std::string> activeImagePaths;
    std::vector<QString> activeImagePathKeys;
    activeImagePaths.reserve(_views.size());
    activeImagePathKeys.reserve(_views.size());
    for (const CameraView &view : _views)
    {
        activeImagePaths.push_back(view.imagePath);
        activeImagePathKeys.push_back(normalizedMvsPathKey(view.imagePath));
    }
    const std::vector<MvsSourcePairQuality> activePairQualities =
        filterMvsSourcePairQualitiesForImages(_config.sourcePairQualities, activeImagePaths);
    const MvsSourcePairQualityLookup pairQualityLookup =
        buildMvsSourcePairQualityLookup(activePairQualities);

    MvsVisibilityGraphBuildOptions visibilityOptions;
    const int hardwareThreadCount = static_cast<int>(
        std::max(1U, std::thread::hardware_concurrency()));
    const int requestedThreadCount = resolvedTotalCpuThreadBudget(_config);
    visibilityOptions.workerCount = pointCount >= kParallelVisibilityPointThreshold && NV > 1
        ? std::clamp(std::min(requestedThreadCount, hardwareThreadCount), 1, 16)
        : 1;
    visibilityOptions.cancelFlag = &_cancelled;

    std::unordered_map<std::string, int> viewIndexByPathKey;
    viewIndexByPathKey.reserve(activeImagePathKeys.size());
    for (int viewIndex = 0; viewIndex < NV; ++viewIndex)
    {
        viewIndexByPathKey.emplace(
            activeImagePathKeys[static_cast<std::size_t>(viewIndex)].toStdString(),
            viewIndex);
    }
    for (const MvsSourcePairQuality &quality : activePairQualities)
    {
        if (!quality.verified)
        {
            continue;
        }
        const auto first = viewIndexByPathKey.find(
            normalizedMvsPathKey(quality.imageA).toStdString());
        const auto second = viewIndexByPathKey.find(
            normalizedMvsPathKey(quality.imageB).toStdString());
        if (first != viewIndexByPathKey.end() && second != viewIndexByPathKey.end())
        {
            visibilityOptions.requiredPairs.push_back({first->second, second->second});
        }
    }

    MvsVisibilityGraph visibilityGraph = MvsVisibilityGraphBuilder::build(
        _views, _sparse, visibilityOptions);
    if (visibilityGraph.cancelled || _cancelled.load(std::memory_order_relaxed))
    {
        clearFrameCaches();
        return;
    }
    _visibilityWordCount = visibilityGraph.statistics.visibilityWordCount;
    _visibilityBits = std::move(visibilityGraph.visibilityBits);
    _visibilityAdjacency = std::move(visibilityGraph.neighborsByView);
    for (int viewIndex = 0; viewIndex < NV; ++viewIndex)
    {
        _frameCaches[static_cast<std::size_t>(viewIndex)].visiblePointIndices =
            std::move(visibilityGraph.visiblePointIndicesByView[static_cast<std::size_t>(viewIndex)]);
    }
    _frameCachesReady = true;

    const bool hasSourcePairQuality = !pairQualityLookup.qualitiesByPairKey.empty();
    const bool requireVerifiedSourcePairs = _config.requireVerifiedSourcePairs && hasSourcePairQuality;
    const int minVerifiedPairInliers = std::max(1, _config.minSourcePairGeometricInliers);
    int referenceTrackFallbackCount = 0;
    if (_config.requireVerifiedSourcePairs
        && !_config.sourcePairQualities.empty()
        && !hasSourcePairQuality)
    {
        LOG_WARN(QStringLiteral(
                     "[MVS] 配置中的 %1 个几何验证对均不属于当前 %2 张影像，"
                     "已忽略旧引用并回退到当前空三稀疏轨迹")
                     .arg(_config.sourcePairQualities.size())
                     .arg(NV));
    }

    auto sampledMedianAngle = [this, NV](int refIdx, int sourceIdx) -> float
    {
        if (refIdx < 0 || refIdx >= NV || sourceIdx < 0 || sourceIdx >= NV)
        {
            return 0.f;
        }

        constexpr size_t kMaxAngleSamples = 2048;
        std::vector<float> angles;
        angles.reserve(kMaxAngleSamples);
        for (size_t pointIndex : _frameCaches[static_cast<size_t>(refIdx)].visiblePointIndices)
        {
            if (!isSparsePointVisibleInFrame(sourceIdx, pointIndex))
            {
                continue;
            }
            const float angle = mvsTriangulationAngleDeg(_views[refIdx],
                                                          _views[sourceIdx],
                                                          _sparse.points[pointIndex]);
            if (angle <= 0.0f)
            {
                continue;
            }
            angles.push_back(angle);
            if (angles.size() >= kMaxAngleSamples)
            {
                break;
            }
        }

        if (angles.empty())
        {
            return 0.f;
        }

        const auto mid = angles.begin() + static_cast<long>(angles.size() / 2);
        std::nth_element(angles.begin(), mid, angles.end());
        return *mid;
    };

    for (int refIdx = 0; refIdx < NV; ++refIdx)
    {
        const int desiredSourceCount = std::max(1, _config.numSourceViews);
        const size_t refVisibleCount =
            _frameCaches[static_cast<size_t>(refIdx)].visiblePointIndices.size();

        MvsSourcePlannerOptions plannerOptions;
        plannerOptions.refIndex = refIdx;
        plannerOptions.viewCount = NV;
        plannerOptions.maxSources = desiredSourceCount;
        plannerOptions.rejectAngleOutliers = true;
        plannerOptions.minTriangulationAngleDeg = 0.2f;
        plannerOptions.maxTriangulationAngleDeg =
            recommendedMvsSourceMaximumAngleDeg(
                _effectiveSceneProfile, desiredSourceCount);
        struct RankedSourceCandidate
        {
            int viewIndex = -1;
            int commonVisiblePoints = 0;
            int sequenceDistance = 0;
            bool verifiedPair = false;
        };

        std::vector<RankedSourceCandidate> rankedSourceCandidates;
        const auto &visibilityNeighbors =
            _visibilityAdjacency[static_cast<std::size_t>(refIdx)];
        rankedSourceCandidates.reserve(visibilityNeighbors.size());
        for (const MvsVisibilityNeighbor &neighbor : visibilityNeighbors)
        {
            const MvsSourcePairQuality *pairQuality =
                pairQualityLookup.findNormalized(
                    activeImagePathKeys[static_cast<std::size_t>(refIdx)],
                    activeImagePathKeys[static_cast<std::size_t>(neighbor.viewIndex)]);
            const bool verifiedPair = pairQuality && pairQuality->verified;
            if (neighbor.sharedTrackCount <= 0 && !verifiedPair)
            {
                continue;
            }

            rankedSourceCandidates.push_back({
                neighbor.viewIndex,
                neighbor.sharedTrackCount,
                std::abs(neighbor.viewIndex - refIdx),
                verifiedPair
            });
        }

        std::sort(rankedSourceCandidates.begin(),
                  rankedSourceCandidates.end(),
                  [](const RankedSourceCandidate &a, const RankedSourceCandidate &b)
                  {
                      if (a.verifiedPair != b.verifiedPair)
                      {
                          return a.verifiedPair;
                      }
                      if (a.commonVisiblePoints != b.commonVisiblePoints)
                      {
                          return a.commonVisiblePoints > b.commonVisiblePoints;
                      }
                      if (a.sequenceDistance != b.sequenceDistance)
                      {
                          return a.sequenceDistance < b.sequenceDistance;
                      }
                      return a.viewIndex < b.viewIndex;
                  });

        bool referenceHasPairQuality = false;
        for (const RankedSourceCandidate &candidate : rankedSourceCandidates)
        {
            if (pairQualityLookup.findNormalized(
                    activeImagePathKeys[static_cast<std::size_t>(refIdx)],
                    activeImagePathKeys[
                        static_cast<std::size_t>(candidate.viewIndex)]))
            {
                referenceHasPairQuality = true;
                break;
            }
        }
        const bool requireVerifiedSourcePairsForReference =
            requireVerifiedSourcePairs && referenceHasPairQuality;
        if (requireVerifiedSourcePairsForReference)
        {
            plannerOptions.minGeometricInliers = minVerifiedPairInliers;
            plannerOptions.allowWeakKnownOverlap = false;
            plannerOptions.requireVerifiedPairGeometry = true;
            plannerOptions.allowSequenceFallback = false;
        }
        else if (_config.numSourceViews >= 5 && _config.fusion.minConsistentViews >= 3)
        {
            plannerOptions.minSharedTracks = 20;
            plannerOptions.minGeometricInliers = 20;
            plannerOptions.minSourceQualityScore = 0.35f;
            plannerOptions.allowWeakKnownOverlap = false;
        }
        if (requireVerifiedSourcePairs && !referenceHasPairQuality)
        {
            ++referenceTrackFallbackCount;
        }

        std::vector<MvsSourceCandidate> candidates;
        candidates.reserve(rankedSourceCandidates.size());
        int provenSourceCount = 0;
        int currentSourceScoreCutoff = -1;
        for (const RankedSourceCandidate &candidate : rankedSourceCandidates)
        {
            // remaining candidates are sorted by common count; once enough stronger sources are proven,
            // avoid spending more time sampling triangulation angles for weaker candidates.
            if (provenSourceCount >= desiredSourceCount
                && candidate.commonVisiblePoints <= currentSourceScoreCutoff)
            {
                break;
            }

            const float medianAngle = sampledMedianAngle(refIdx, candidate.viewIndex);
            MvsSourceCandidate sourceCandidate;
            sourceCandidate.viewIndex = candidate.viewIndex;
            sourceCandidate.sharedTracks = candidate.commonVisiblePoints;
            const MvsSourcePairQuality *pairQuality =
                pairQualityLookup.findNormalized(
                    activeImagePathKeys[static_cast<std::size_t>(refIdx)],
                    activeImagePathKeys[
                        static_cast<std::size_t>(candidate.viewIndex)]);
            const bool pairVerificationFailed = pairQuality
                && pairQuality->hasVerificationStatistics
                && !pairQuality->verified;
            const bool pairVerificationMissing = pairQuality
                && !pairQuality->hasVerificationStatistics;
            sourceCandidate.geometricInliers =
                pairQuality && pairQuality->hasVerificationStatistics
                ? std::max(0, pairQuality->geometricInliers)
                : candidate.commonVisiblePoints;
            sourceCandidate.verifiedPairGeometry = pairQuality
                && pairQuality->verified
                && pairQuality->geometricInliers > 0;
            sourceCandidate.verificationStatus =
                sourceCandidate.verifiedPairGeometry
                ? MvsSourceVerificationStatus::Verified
                : (pairVerificationFailed
                       ? MvsSourceVerificationStatus::Failed
                       : (pairVerificationMissing || referenceHasPairQuality
                              ? MvsSourceVerificationStatus::MissingStatistics
                              : MvsSourceVerificationStatus::NotRequested));
            sourceCandidate.pairTotalMatches =
                pairQuality ? std::max(0, pairQuality->totalMatches) : 0;
            sourceCandidate.pairCoverageScore =
                pairQuality ? pairQuality->geometricCoverage : 0.0f;
            sourceCandidate.verificationReason = pairQuality
                ? pairQuality->verificationReason
                : (referenceHasPairQuality
                       ? "pair_not_present_in_current_catalog"
                       : "pair_verification_not_requested");
            sourceCandidate.medianTriangulationAngleDeg = medianAngle;
            sourceCandidate.coverageScore = refVisibleCount > 0
                ? std::clamp(static_cast<float>(candidate.commonVisiblePoints) / static_cast<float>(refVisibleCount), 0.0f, 1.0f)
                : 0.0f;
            sourceCandidate.baselineScore = std::clamp(medianAngle / 20.0f, 0.0f, 1.0f);
            sourceCandidate.sequenceDistance = candidate.sequenceDistance;
            sourceCandidate.knownOverlap =
                !pairVerificationFailed
                && (candidate.commonVisiblePoints > 0
                    || sourceCandidate.verifiedPairGeometry);
            candidates.push_back(sourceCandidate);
            const bool hasRequiredPairQuality = !plannerOptions.requireVerifiedPairGeometry
                || (sourceCandidate.verifiedPairGeometry
                    && sourceCandidate.geometricInliers >= plannerOptions.minGeometricInliers);
            if (hasRequiredPairQuality
                && medianAngle >= plannerOptions.minTriangulationAngleDeg
                && medianAngle <= plannerOptions.maxTriangulationAngleDeg)
            {
                ++provenSourceCount;
                if (provenSourceCount >= desiredSourceCount)
                {
                    currentSourceScoreCutoff = candidate.commonVisiblePoints;
                }
            }
        }
        if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            std::vector<float> candidate_angles;
            candidate_angles.reserve(candidates.size());
            for (const MvsSourceCandidate &candidate : candidates)
            {
                candidate_angles.push_back(
                    candidate.medianTriangulationAngleDeg);
            }
            plannerOptions.maxTriangulationAngleDeg =
                adaptiveMvsSourceMaximumAngleDeg(
                    _effectiveSceneProfile,
                    desiredSourceCount,
                    candidate_angles);
            // Failed SfM pairs are never promoted back into SfM. For a short
            // orbital sequence, a narrowly qualified failed pair may still be
            // useful as PatchMatch source-only evidence. PatchMatch now uses
            // majority support (3 of 4 sources), so a fourth independently
            // qualified direction can reject a mutually consistent wrong
            // layer without letting that single weaker view dominate NCC.
            plannerOptions.allowFailedPairBackfill = true;
            plannerOptions.failedPairBackfillMaximumTotalSources = 4;
            plannerOptions.failedPairBackfillMinimumInliers = 12;
            plannerOptions.failedPairBackfillMinimumMatches = 14;
            plannerOptions.failedPairBackfillMinimumSharedTracks = 20;
            plannerOptions.failedPairBackfillMinimumCoverage = 0.1875f;
            plannerOptions.failedPairBackfillMinimumWilsonLowerBound = 0.50f;
            plannerOptions.failedPairBackfillMaximumAngleDeg = 65.0f;
        }

        const MvsSourcePlan sourcePlan = requireVerifiedSourcePairsForReference
            ? planMvsSourceViewsVerifiedFirst(candidates, plannerOptions)
            : planMvsSourceViews(candidates, plannerOptions);

        auto &sources = _frameCaches[static_cast<size_t>(refIdx)].sourceViewIndices;
        _frameCaches[static_cast<size_t>(refIdx)].sourceViewScores = sourcePlan.selected;
        _frameCaches[static_cast<size_t>(refIdx)].requestedSourceViewCount = desiredSourceCount;
        _frameCaches[static_cast<size_t>(refIdx)].sourceViewShortfall = sourcePlan.sourceViewShortfall;
        if (sourcePlan.sourceViewShortfall > 0)
        {
            bool verification_failed = false;
            bool verification_missing = false;
            bool angle_rejected = false;
            bool evidence_missing = false;
            for (const MvsSourceRejectedCandidate &rejected : sourcePlan.rejected)
            {
                verification_failed = verification_failed ||
                    rejected.candidate.verificationStatus ==
                        MvsSourceVerificationStatus::Failed;
                verification_missing = verification_missing ||
                    rejected.candidate.verificationStatus ==
                        MvsSourceVerificationStatus::MissingStatistics;
                angle_rejected = angle_rejected ||
                    rejected.reason == MvsSourceRejectReason::TriangulationAngle;
                evidence_missing = evidence_missing ||
                    rejected.reason == MvsSourceRejectReason::NoEvidence;
            }
            std::string reason = "insufficient_qualified_sources";
            if (verification_failed)
            {
                reason = "pair_geometry_verification_failed";
            }
            else if (verification_missing)
            {
                reason = "missing_pair_verification_statistics";
            }
            else if (angle_rejected)
            {
                reason = "safe_baseline_source_shortfall";
            }
            else if (evidence_missing)
            {
                reason = "insufficient_overlap_evidence";
            }
            _frameCaches[static_cast<size_t>(refIdx)]
                .sourceViewShortfallReason = std::move(reason);
        }
        sources.reserve(static_cast<size_t>(std::min(NV - 1, desiredSourceCount)));
        for (const auto &score : sourcePlan.selected)
        {
            if (score.score <= 0.f)
            {
                continue;
            }
            sources.push_back(score.viewIndex);
            if (static_cast<int>(sources.size()) >= desiredSourceCount)
            {
                break;
            }
        }

        if (sources.empty())
        {
            if (plannerOptions.allowSequenceFallback)
            {
                const MvsSourcePlan fallbackPlan = planMvsSourceViews({}, plannerOptions);
                _frameCaches[static_cast<size_t>(refIdx)].sourceViewScores = fallbackPlan.selected;
                for (const auto &entry : fallbackPlan.selected)
                {
                    sources.push_back(entry.viewIndex);
                }
            }
        }

        auto &cache = _frameCaches[static_cast<size_t>(refIdx)];
        cache.sourceSharedPointIndices.reserve(cache.visiblePointIndices.size());
        for (size_t pointIndex : cache.visiblePointIndices)
        {
            for (int sourceIdx : cache.sourceViewIndices)
            {
                if (sourceIdx < 0 || sourceIdx >= NV || sourceIdx == refIdx)
                {
                    continue;
                }
                if (isSparsePointVisibleInFrame(sourceIdx, pointIndex))
                {
                    cache.sourceSharedPointIndices.push_back(pointIndex);
                    break;
                }
            }
        }
    }

    if (referenceTrackFallbackCount > 0)
    {
        LOG_WARN(QStringLiteral(
                     "[MVS] %1/%2 个参考帧在当前影像集合中没有几何验证对，"
                     "已逐帧回退到当前空三稀疏轨迹，避免源视图被旧引用清空")
                     .arg(referenceTrackFallbackCount)
                     .arg(NV));
    }
    LOG_INFO(QStringLiteral("[MVS] MVS 可见性缓存完成: views=%1 points=%2 elapsed=%3 ms")
                 .arg(NV)
                 .arg(static_cast<qulonglong>(pointCount))
                 .arg(elapsedMs(start, Clock::now()), 0, 'f', 1));
}

std::vector<int> DepthMapGenerator::sourceViewIndicesForFrame(int refIdx, int maxSources) const
{
    if (_frameCachesReady
        && refIdx >= 0
        && refIdx < static_cast<int>(_frameCaches.size())
        && maxSources > 0)
    {
        const auto &cached = _frameCaches[static_cast<size_t>(refIdx)].sourceViewIndices;
        const int count = std::min(maxSources, static_cast<int>(cached.size()));
        return std::vector<int>(cached.begin(), cached.begin() + count);
    }

    return selectMvsSourceViewIndices(_views, _sparse, refIdx, maxSources);
}

std::vector<size_t> DepthMapGenerator::visibleSparsePointIndicesForFrame(
    int refIdx,
    const std::vector<int> &sourceIndices,
    int minSourceViews) const
{
    if (!_frameCachesReady
        || refIdx < 0
        || refIdx >= static_cast<int>(_frameCaches.size()))
    {
        return collectMvsVisibleSparsePointIndices(_views, _sparse, refIdx, sourceIndices, minSourceViews);
    }

    const auto &cache = _frameCaches[static_cast<size_t>(refIdx)];
    const auto &refVisible = cache.visiblePointIndices;
    if (sourceIndices.empty() || minSourceViews <= 0)
    {
        return refVisible;
    }

    auto sourceIndicesMatchCachedPrefix = [&cache, &sourceIndices]()
    {
        if (sourceIndices.size() != cache.sourceViewIndices.size())
        {
            return false;
        }
        return std::equal(sourceIndices.begin(),
                          sourceIndices.end(),
                          cache.sourceViewIndices.begin());
    };
    if (minSourceViews <= 1 && sourceIndicesMatchCachedPrefix())
    {
        return cache.sourceSharedPointIndices;
    }

    std::vector<size_t> filtered;
    filtered.reserve(refVisible.size());
    for (size_t pointIndex : refVisible)
    {
        int sourceVisible = 0;
        for (int sourceIdx : sourceIndices)
        {
            if (sourceIdx < 0 || sourceIdx >= static_cast<int>(_views.size()) || sourceIdx == refIdx)
            {
                continue;
            }
            if (isSparsePointVisibleInFrame(sourceIdx, pointIndex))
            {
                ++sourceVisible;
                if (sourceVisible >= minSourceViews)
                {
                    filtered.push_back(pointIndex);
                    break;
                }
            }
        }
    }
    return filtered;
}

// =============================================================================
// 预加载所有图像到灰度缓存，避免逐帧重复从磁盘读取
// =============================================================================
bool DepthMapGenerator::probeImageMetadata(QString *errorMessage)
{
    for (int index = 0; index < static_cast<int>(_views.size()); ++index)
    {
        if (_cancelled.load(std::memory_order_relaxed))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("影像头部探测已取消");
            }
            return false;
        }

        MvsImageMetadata metadata;
        std::string probeError;
        if (!probeMvsImageMetadata(
                _views[static_cast<std::size_t>(index)].imagePath,
                &metadata,
                &probeError))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取影像 %1 的头部尺寸：%2")
                                    .arg(index)
                                    .arg(QString::fromStdString(probeError));
            }
            return false;
        }

        CameraView &view = _views[static_cast<std::size_t>(index)];
        if ((view.imageWidth > 0 && view.imageWidth != metadata.width) ||
            (view.imageHeight > 0 && view.imageHeight != metadata.height))
        {
            LOG_WARN(QStringLiteral(
                         "[MVS][内存规划] 视图 %1 声明尺寸 %2x%3 与影像头 %4x%5 不一致，"
                         "以内嵌元数据为准")
                         .arg(index)
                         .arg(view.imageWidth)
                         .arg(view.imageHeight)
                         .arg(metadata.width)
                         .arg(metadata.height));
        }
        view.imageWidth = metadata.width;
        view.imageHeight = metadata.height;
    }

    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

bool DepthMapGenerator::loadMvsImageFrame(
    int frameIndex,
    const std::atomic_bool *cancelFlag,
    MvsImageFrame *frame,
    std::string *errorMessage)
{
    if (!frame || frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()))
    {
        if (errorMessage)
        {
            *errorMessage = "MVS image loader received an invalid frame";
        }
        return false;
    }

    const auto cancelled = [cancelFlag]()
    {
        return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
    };
    const auto stopIfCancelled = [&](const char *message)
    {
        if (!cancelled())
        {
            return false;
        }
        if (errorMessage)
        {
            *errorMessage = message;
        }
        return true;
    };
    if (stopIfCancelled("MVS image load cancelled"))
    {
        return false;
    }

    const CameraView &view = _views[static_cast<std::size_t>(frameIndex)];
    frame->gray = xjw::common::io::readImage(
        view.imagePath, cv::IMREAD_GRAYSCALE);
    if (frame->gray.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "无法解码影像: " + view.imagePath;
        }
        return false;
    }
    if (stopIfCancelled("MVS image load cancelled"))
    {
        return false;
    }

    if (!view.validRegionMaskPath.empty())
    {
        const cv::Mat projectMask = xjw::common::io::readImage(
            view.validRegionMaskPath, cv::IMREAD_GRAYSCALE);
        if (stopIfCancelled("MVS project mask load cancelled"))
        {
            return false;
        }
        if (!projectMask.empty())
        {
            frame->validMask = projectMaskToValidMask(
                projectMask, frame->gray.size());
            if (stopIfCancelled("MVS project mask preparation cancelled"))
            {
                return false;
            }
            bool contentRefined = false;
            float retainedRatio = 1.0f;
            if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                frame->validMask = refineOrbitalProjectValidMask(
                    frame->gray,
                    frame->validMask,
                    &contentRefined,
                    &retainedRatio);
            }
            if (stopIfCancelled("MVS project mask refinement cancelled"))
            {
                return false;
            }
            frame->projectMaskLoaded = true;
            if (contentRefined)
            {
                LOG_DEBUG(QStringLiteral(
                              "[MVS][图像缓存] 视图 %1 暗背景蒙版细化后保留 %2%")
                              .arg(frameIndex)
                              .arg(100.0 * static_cast<double>(retainedRatio),
                                   0,
                                   'f',
                                   1));
            }
        }
        else
        {
            LOG_WARN(QStringLiteral(
                         "[MVS][图像缓存] 视图 %1 项目蒙版读取失败，"
                         "回退内容区域检测: %2")
                         .arg(frameIndex)
                         .arg(QString::fromStdString(view.validRegionMaskPath)));
        }
    }

    if (!frame->projectMaskLoaded)
    {
        float coverage = 0.0f;
        double otsuThreshold = 0.0;
        int adaptiveThreshold = 0;
        frame->validMask = buildContentMask(
            frame->gray,
            &coverage,
            &otsuThreshold,
            &adaptiveThreshold);
        if (stopIfCancelled("MVS content mask preparation cancelled"))
        {
            return false;
        }
        LOG_DEBUG(
            "[MVS][图像缓存] 视图 %d 内容掩码 coverage=%.1f%% "
            "Otsu=%.0f adaptive_threshold=%d",
            frameIndex,
            coverage * 100.0f,
            otsuThreshold,
            adaptiveThreshold);
    }

    const double imageMean = cv::mean(frame->gray)[0];
    if (stopIfCancelled("MVS image statistics cancelled"))
    {
        return false;
    }
    if (imageMean < 80.0)
    {
        cv::Mat enhanced;
        if (imageMean < 30.0)
        {
            cv::Mat floatImage;
            frame->gray.convertTo(floatImage, CV_32F, 1.0 / 255.0);
            cv::pow(floatImage, 0.4, floatImage);
            floatImage.convertTo(enhanced, CV_8U, 255.0);
            if (stopIfCancelled("MVS image enhancement cancelled"))
            {
                return false;
            }
            cv::createCLAHE(8.0, cv::Size(8, 8))->apply(
                enhanced, enhanced);
        }
        else
        {
            cv::createCLAHE(4.0, cv::Size(8, 8))->apply(
                frame->gray, enhanced);
        }
        frame->gray = std::move(enhanced);
    }
    if (stopIfCancelled("MVS image preprocessing cancelled"))
    {
        return false;
    }

    std::string preparationError;
    if (!prepareMvsImage(frame->gray,
                         view.camera,
                         &frame->preparedGray,
                         &frame->preparedCamera,
                         &preparationError))
    {
        if (errorMessage)
        {
            *errorMessage = "影像 MVS 预处理失败: " + preparationError;
        }
        return false;
    }
    if (stopIfCancelled("MVS image preparation cancelled"))
    {
        frame->preparedGray.release();
        return false;
    }

    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

bool DepthMapGenerator::initializeImageProvider(
    const MvsPipelineMemoryPolicyDecision &decision,
    QString *errorMessage)
{
    if (decision.imageStrategy == MvsImageCacheStrategy::Insufficient ||
        decision.imageCacheCapacity == 0)
    {
        if (errorMessage)
        {
            const std::string_view strategy = mvsImageCacheStrategyName(
                decision.imageStrategy);
            *errorMessage = QStringLiteral(
                "MVS 内存不足: required=%1 GiB available=%2 GiB strategy=%3。"
                "visibility=%4 GiB (bitset=%5 GiB visible_index=%6 GiB "
                "pairs=%7 GiB nominated=%8 GiB adjacency=%9 GiB "
                "pair_bound=%10 saturated=%11)。"
                "请降低 gpuFrameWorkerCount/cpuFrameWorkerCount、numSourceViews、"
                "输入分辨率或质量档位，或减少输入视图/稀疏点规模后重试")
                                .arg(bytesToGiB(decision.requiredBytes), 0, 'f', 2)
                                .arg(bytesToGiB(decision.availableBytes), 0, 'f', 2)
                                .arg(QString::fromLatin1(
                                    strategy.data(),
                                    static_cast<int>(strategy.size())))
                                .arg(bytesToGiB(
                                    decision.estimate.visibility.totalBytes), 0, 'f', 2)
                                .arg(bytesToGiB(
                                    decision.estimate.visibility.visibilityBitsetBytes), 0, 'f', 2)
                                .arg(bytesToGiB(
                                    decision.estimate.visibility.visibleIndexBytes), 0, 'f', 2)
                                .arg(bytesToGiB(
                                    decision.estimate.visibility.pairBytes), 0, 'f', 2)
                                .arg(bytesToGiB(
                                    decision.estimate.visibility.nominatedPeerBytes), 0, 'f', 2)
                                .arg(bytesToGiB(
                                    decision.estimate.visibility.adjacencyBytes), 0, 'f', 2)
                                .arg(decision.estimate.visibility.candidatePairUpperBound)
                                .arg(decision.estimate.visibility.saturated
                                         ? QStringLiteral("true")
                                         : QStringLiteral("false"));
        }
        return false;
    }

    _pipelineMemoryDecision = decision;
    _imageCache = std::make_unique<MvsImageCache>(
        _views.size(),
        decision.imageCacheCapacity,
        [this](int frameIndex,
               const std::atomic_bool *cancelFlag,
               MvsImageFrame *frame,
               std::string *loaderError)
        {
            return loadMvsImageFrame(
                frameIndex, cancelFlag, frame, loaderError);
        });
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

bool DepthMapGenerator::preloadImages(QString *errorMessage)
{
    if (!_imageCache)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS 图像 provider 尚未初始化");
        }
        return false;
    }
    if (_pipelineMemoryDecision.imageStrategy != MvsImageCacheStrategy::Eager)
    {
        return true;
    }

    const auto preloadStart = Clock::now();
    std::string preloadError;
    const int workerCount = preloadImagesWorkerCount(
        static_cast<int>(_views.size()), resolvedTotalCpuThreadBudget(_config));
    if (!_imageCache->preloadAll(
            workerCount, &_cancelled, &preloadError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS 影像 eager 预加载失败：%1")
                                .arg(QString::fromStdString(preloadError));
        }
        return false;
    }

    LOG_INFO(QStringLiteral(
                 "[MVS][图像缓存] eager 预加载完成: resident=%1/%2 "
                 "bytes=%3 GiB elapsed=%4 ms")
                 .arg(_imageCache->residentCount())
                 .arg(_imageCache->frameCount())
                 .arg(bytesToGiB(_imageCache->residentBytes()), 0, 'f', 2)
                 .arg(elapsedMs(preloadStart, Clock::now()), 0, 'f', 1));
    return true;
}

MvsImageCache::ImageLease DepthMapGenerator::acquireImageFrame(
    int frameIndex,
    std::string *errorMessage)
{
    if (!_imageCache)
    {
        if (errorMessage)
        {
            *errorMessage = "MVS image provider is not initialized";
        }
        return {};
    }
    return _imageCache->acquire(frameIndex, &_cancelled, errorMessage);
}

// =============================================================================
void DepthMapGenerator::start()
{
    if (_backgroundFuture.isRunning())
    {
        emit errorOccurred(QStringLiteral("深度图生成任务已经在运行，忽略重复启动请求"));
        return;
    }

    _cancelled = false;
    _finishedEmitted = false;
    _depthFrames.clear();
    _backgroundFuture = QtConcurrent::run([this]()
    {
        runInBackground();
    });
}

// =============================================================================
bool DepthMapGenerator::estimateDepthRange(int refIdx,
                                           float &zNear,
                                           float &zFar,
                                           const std::vector<int> &sourceIndices) const
{
    const int minSourceViews = sourceIndices.empty() ? 0 : 1;
    std::vector<size_t> visiblePointIndices =
        visibleSparsePointIndicesForFrame(refIdx, sourceIndices, minSourceViews);
    if (visiblePointIndices.size() < 5 && minSourceViews > 0)
    {
        visiblePointIndices = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
        LOG_DEBUG(
            "[MVS][帧 %d][深度范围] 共视稀疏点不足，回退参考帧可见点 (%zu)",
            refIdx, visiblePointIndices.size());
    }
    return estimateDepthRangeFromVisiblePoints(refIdx, visiblePointIndices, zNear, zFar);
}

bool DepthMapGenerator::estimateDepthRangeFromVisiblePoints(
    int refIdx,
    const std::vector<size_t> &visiblePointIndices,
    float &zNear,
    float &zFar) const
{
    const CameraView &ref = _views[refIdx];
    const Camera cam = mvsPinholeCamera(ref.camera);

    std::vector<float> depths;
    depths.reserve(visiblePointIndices.size());

    for (size_t pointIndex : visiblePointIndices)
    {
        if (pointIndex >= _sparse.points.size())
        {
            continue;
        }
        const auto &pt = _sparse.points[pointIndex];
        const double world[3] = {pt[0], pt[1], pt[2]};
        double camera_point[3] = {0.0, 0.0, 0.0};
        cam.worldToCamera(world, camera_point);
        if (camera_point[2] > 0.0)
        {
            depths.push_back(static_cast<float>(camera_point[2]));
        }
    }

    if (depths.size() < 5)
    {
        // 没有足够的稀疏点——使用全局最大相机基线估算深度范围。
        // 关键：使用「全局最大基线」（所有相机对之间的最大距离），
        // 而非 per-camera 最大基线，以保证所有帧使用一致的 zNear/zFar，
        // 防止深度图均值差异过大导致融合一致性检查全部失败。
        float maxBaseline = 0.f;
        const int NVall = static_cast<int>(_views.size());
        for (int ia = 0; ia < NVall; ++ia)
        {
            for (int ib = ia + 1; ib < NVall; ++ib)
            {
                const CameraBaseline baseline = CameraBaseline::evaluate(
                    _views[ia].camera,
                    _views[ib].camera);
                if (baseline.isValid())
                {
                    maxBaseline = std::max(maxBaseline, static_cast<float>(baseline.length()));
                }
            }
        }
        if (maxBaseline > 1e-3f)
        {
            // 航空摄影测量：典型场景深度 ≈ 基线的 0.5× ~ 100×
            zNear = maxBaseline * 0.1f;
            zFar  = maxBaseline * 100.f;
            LOG_DEBUG(
                "[MVS][深度范围] 回退全局基线: baseline=%.4f zNear=%.4f zFar=%.4f",
                maxBaseline, zNear, zFar);
        } else {
            // 实在无法估计
            float dx = _sparse.maxPt[0] - _sparse.minPt[0];
            float dy = _sparse.maxPt[1] - _sparse.minPt[1];
            float dz = _sparse.maxPt[2] - _sparse.minPt[2];
            float diag = std::sqrt(dx*dx + dy*dy + dz*dz);
            zNear = 0.1f;
            zFar  = diag > 0 ? diag * 3.f : 100.f;
            LOG_DEBUG(
                "[MVS][深度范围] 回退 AABB: diag=%.4f zNear=%.4f zFar=%.4f",
                diag, zNear, zFar);
        }
        return true;
    }

    std::sort(depths.begin(), depths.end());
    size_t n = depths.size();

    // 使用 IQR (四分位距) 剔除离群深度值，比固定百分位更鲁棒
    float Q1 = depths[n / 4];
    float Q3 = depths[n * 3 / 4];
    float IQR = Q3 - Q1;
    float lowerFence = Q1 - 1.5f * IQR;
    float upperFence = Q3 + 1.5f * IQR;

    // 在 fence 范围内重新取 5%/95% 分位
    std::vector<float> inlierDepths;
    inlierDepths.reserve(n);
    for (float d : depths) {
        if (d >= lowerFence && d <= upperFence)
            inlierDepths.push_back(d);
    }
    if (inlierDepths.size() < 3) inlierDepths = depths; // fallback

    size_t ni = inlierDepths.size();
    zNear = inlierDepths[static_cast<size_t>(ni * 0.02f)] * _config.zNearScale;
    zFar  = inlierDepths[static_cast<size_t>(ni * 0.98f)] * _config.zFarScale;
    zNear = std::max(zNear, 0.01f);
    zFar  = std::max(zFar,  zNear + 0.1f);

    LOG_DEBUG(
        "[MVS][帧 %d][深度范围] Q1=%.4f Q3=%.4f IQR=%.4f "
        "fence=[%.4f,%.4f] inliers=%zu/%zu visible=%zu",
        refIdx, Q1, Q3, IQR, lowerFence, upperFence,
        inlierDepths.size(), n, visiblePointIndices.size());
    return true;
}

// =============================================================================
cv::Mat DepthMapGenerator::buildHintDepth(int refIdx,
                                          int W,
                                          int H,
                                          const std::vector<int> &sourceIndices) const
{
    const int minSourceViews = sourceIndices.empty() ? 0 : 1;
    std::vector<size_t> visiblePointIndices =
        visibleSparsePointIndicesForFrame(refIdx, sourceIndices, minSourceViews);
    if (visiblePointIndices.empty() && minSourceViews > 0)
    {
        visiblePointIndices = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
        LOG_DEBUG(
            "[MVS][帧 %d][稀疏引导] 共视点为空，回退参考帧可见点 (%zu)",
            refIdx, visiblePointIndices.size());
    }

    return buildHintDepthFromVisiblePoints(refIdx, W, H, visiblePointIndices);
}

cv::Mat DepthMapGenerator::buildHintDepthFromVisiblePoints(
    int refIdx,
    int W,
    int H,
    const std::vector<size_t> &visiblePointIndices) const
{
    if (refIdx < 0 || refIdx >= static_cast<int>(_views.size()))
    {
        return cv::Mat();
    }

    return buildHintDepthForCamera(refIdx,
                                   mvsPinholeCamera(_views[refIdx].camera),
                                   W,
                                   H,
                                   visiblePointIndices);
}

cv::Mat DepthMapGenerator::buildHintDepthForCamera(
    int refIdx,
    const Camera &camera,
    int W,
    int H,
    const std::vector<size_t> &visiblePointIndices) const
{
    const std::vector<ProjectedSparseDepthSample> samples =
        collectProjectedSparseDepthSamples(_sparse, camera, W, H, visiblePointIndices);
    return buildHintDepthFromProjectedSamples(refIdx, W, H, samples);
}

std::vector<ProjectedSparseDepthSample> DepthMapGenerator::collectProjectedSparseDepthSamples(
    const SparseCloud &sparse,
    const Camera &camera,
    int imageWidth,
    int imageHeight,
    const std::vector<size_t> &visiblePointIndices)
{
    std::vector<ProjectedSparseDepthSample> samples;
    if (sparse.points.empty() || imageWidth <= 0 || imageHeight <= 0 || !camera.isValid())
    {
        return samples;
    }

    std::vector<ProjectedSparseDepthSample> projectedCandidates;
    projectedCandidates.reserve(visiblePointIndices.size());
    std::vector<float> depthQuantileSamples;
    depthQuantileSamples.reserve(std::min(visiblePointIndices.size(), kMaxProjectedDepthQuantileSamples));
    const size_t quantileSampleStride = visiblePointIndices.size() > kMaxProjectedDepthQuantileSamples
        ? (visiblePointIndices.size() + kMaxProjectedDepthQuantileSamples - 1)
              / kMaxProjectedDepthQuantileSamples
        : 1;
    size_t validDepthOrdinal = 0;
    for (size_t pointIndex : visiblePointIndices)
    {
        if (pointIndex >= sparse.points.size())
        {
            continue;
        }
        const auto &pt = sparse.points[pointIndex];
        const double world[3] = {pt[0], pt[1], pt[2]};
        double pixel[2] = {0.0, 0.0};
        double depth = 0.0;
        if (!camera.projectWorldPointWithDepth(world, pixel, depth)
            || !std::isfinite(depth)
            || pixel[0] < 0.0
            || pixel[0] >= static_cast<double>(imageWidth)
            || pixel[1] < 0.0
            || pixel[1] >= static_cast<double>(imageHeight))
        {
            continue;
        }

        ProjectedSparseDepthSample candidate;
        candidate.uNorm = static_cast<float>(pixel[0] / static_cast<double>(imageWidth));
        candidate.vNorm = static_cast<float>(pixel[1] / static_cast<double>(imageHeight));
        candidate.depth = static_cast<float>(depth);
        projectedCandidates.push_back(candidate);

        if ((validDepthOrdinal % quantileSampleStride) == 0
            && depthQuantileSamples.size() < kMaxProjectedDepthQuantileSamples)
        {
            depthQuantileSamples.push_back(candidate.depth);
        }
        ++validDepthOrdinal;
    }

    float depthLo = 0.f, depthHi = 1e30f;
    if (depthQuantileSamples.size() >= 4)
    {
        auto q1It = depthQuantileSamples.begin()
            + static_cast<std::ptrdiff_t>(depthQuantileSamples.size() / 4);
        std::nth_element(depthQuantileSamples.begin(), q1It, depthQuantileSamples.end());
        const float Q1 = *q1It;

        auto q3It = depthQuantileSamples.begin()
            + static_cast<std::ptrdiff_t>(depthQuantileSamples.size() * 3 / 4);
        std::nth_element(depthQuantileSamples.begin(), q3It, depthQuantileSamples.end());
        const float Q3 = *q3It;

        float IQR = Q3 - Q1;
        depthLo = Q1 - 1.5f * IQR;
        depthHi = Q3 + 1.5f * IQR;
    }

    samples.reserve(projectedCandidates.size());
    for (const ProjectedSparseDepthSample &candidate : projectedCandidates)
    {
        if (candidate.depth < depthLo || candidate.depth > depthHi || !std::isfinite(candidate.depth))
        {
            continue;
        }

        samples.push_back(candidate);
    }

    return samples;
}

cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples(
    int refIdx,
    int W,
    int H,
    const std::vector<ProjectedSparseDepthSample> &samples)
{
    if (samples.empty() || W <= 0 || H <= 0)
    {
        return cv::Mat();
    }

    cv::Mat hint = buildSparseSeedDepthFromProjectedSamples(refIdx, W, H, samples);
    if (hint.empty())
    {
        return cv::Mat();
    }

    // 第二步：限距离膨胀——仅将稀疏种子传播到 maxHintRadius 像素范围内
    // 不做全图扫线传播，以免把远离稀疏点的像素也强制初始化为"错误 hint"：
    //   GPU 初始化对有 hint 的像素使用 hint±30% 的窄范围，
    //   若 hint 覆盖了距离真实深度很远的区域，PatchMatch 将无法逃脱。
    // 超出 maxHintRadius 的像素保持 hint=0 → GPU 用全范围随机初始化。
    const int seedHintCnt = cv::countNonZero(hint > 0);
    if (seedHintCnt <= 0)
    {
        LOG_DEBUG(
            "[MVS][帧 %d][稀疏引导] 可见点=%zu，无有效种子，跳过传播",
            refIdx, samples.size());
        return cv::Mat();
    }

    const int adaptiveHintRadius = seedHintCnt > 0
        ? std::clamp(static_cast<int>(std::sqrt(static_cast<float>(W * H) / seedHintCnt) * 0.5f), 16, 48)
        : 0;
    const int maxHintRadius = adaptiveHintRadius;
    // 距离变换限距离膨胀：用 OpenCV 的优化扫描求每个像素最近的稀疏 seed，
    // 避免手写多轮 at<> 全图扫描。只在 maxHintRadius 内传播，远处仍保留 0。
    {
        cv::Mat seedDistanceMask(H, W, CV_8U, cv::Scalar(255));
        seedDistanceMask.setTo(0, hint > 0);

        cv::Mat distanceMap;
        cv::Mat nearestSeedLabels;
        cv::distanceTransform(seedDistanceMask,
                              distanceMap,
                              nearestSeedLabels,
                              cv::DIST_L1,
                              3,
                              cv::DIST_LABEL_PIXEL);

        double maxLabelValue = 0.0;
        cv::minMaxLoc(nearestSeedLabels, nullptr, &maxLabelValue);
        std::vector<float> labelDepths(static_cast<size_t>(std::max(0.0, maxLabelValue)) + 1, 0.0f);
        for (int row = 0; row < H; ++row)
        {
            const float *hintRow = hint.ptr<float>(row);
            const int *labelRow = nearestSeedLabels.ptr<int>(row);
            for (int col = 0; col < W; ++col)
            {
                const float depth = hintRow[col];
                const int label = labelRow[col];
                if (depth <= 0.0f || label <= 0)
                {
                    continue;
                }

                float &labelDepth = labelDepths[static_cast<size_t>(label)];
                if (labelDepth == 0.0f || depth < labelDepth)
                {
                    labelDepth = depth;
                }
            }
        }

        for (int row = 0; row < H; ++row)
        {
            float *hintRow = hint.ptr<float>(row);
            const float *distanceRow = distanceMap.ptr<float>(row);
            const int *labelRow = nearestSeedLabels.ptr<int>(row);
            for (int col = 0; col < W; ++col)
            {
                if (hintRow[col] > 0.0f || distanceRow[col] > static_cast<float>(maxHintRadius))
                {
                    continue;
                }

                const int label = labelRow[col];
                if (label <= 0 || static_cast<size_t>(label) >= labelDepths.size())
                {
                    continue;
                }

                const float depth = labelDepths[static_cast<size_t>(label)];
                if (depth > 0.0f)
                {
                    hintRow[col] = depth;
                }
            }
        }
    }

    int hintCnt = cv::countNonZero(hint > 0);
    LOG_DEBUG(
        "[MVS][帧 %d][稀疏引导] visible=%zu seeds=%d radius=%d coverage=%d/%d (%.1f%%)",
        refIdx, samples.size(), seedHintCnt, maxHintRadius,
        hintCnt, W * H, 100.0f * hintCnt / (W * H));
    return hint;
}

cv::Mat DepthMapGenerator::buildSparseSeedDepthFromProjectedSamples(
    int refIdx,
    int W,
    int H,
    const std::vector<ProjectedSparseDepthSample> &samples,
    int seedRadius)
{
    (void)refIdx;
    if (samples.empty() || W <= 0 || H <= 0)
    {
        return cv::Mat();
    }

    const int radius = std::clamp(seedRadius, 0, 8);
    cv::Mat hint(H, W, CV_32F, cv::Scalar(0.f));

    for (const ProjectedSparseDepthSample &sample : samples)
    {
        const int iu = static_cast<int>(std::round(sample.uNorm * static_cast<float>(W)));
        const int iv = static_cast<int>(std::round(sample.vNorm * static_cast<float>(H)));
        if (iu < 0 || iu >= W || iv < 0 || iv >= H || sample.depth <= 0.0f)
        {
            continue;
        }

        for (int dv = -radius; dv <= radius; ++dv)
        {
            for (int du = -radius; du <= radius; ++du)
            {
                int nu = iu+du, nv = iv+dv;
                if (nu<0||nu>=W||nv<0||nv>=H)
                {
                    continue;
                }
                float &h = hint.at<float>(nv, nu);
                if (h == 0.f || sample.depth < h)
                {
                    h = sample.depth;
                }
            }
        }
    }

    return cv::countNonZero(hint > 0) > 0 ? hint : cv::Mat();
}

// =============================================================================
cv::Mat DepthMapGenerator::buildSparseSupportMask(const std::vector<CameraView> &views,
                                                  const SparseCloud &sparse,
                                                  int refIdx,
                                                  int W,
                                                  int H,
                                                  const std::vector<int> &sourceIndices)
{
    if (W <= 0 || H <= 0 || refIdx < 0 || refIdx >= static_cast<int>(views.size()) || sparse.points.size() < 20)
    {
        return cv::Mat();
    }

    const Camera cam = mvsPinholeCamera(views[refIdx].camera);
    if (!cam.isValid())
    {
        return cv::Mat();
    }

    const int minSourceViews = sourceIndices.empty() ? 0 : 1;
    std::vector<size_t> visiblePointIndices =
        collectMvsVisibleSparsePointIndices(views, sparse, refIdx, sourceIndices, minSourceViews);
    if (visiblePointIndices.size() < 20 && minSourceViews > 0)
    {
        visiblePointIndices = collectMvsVisibleSparsePointIndices(views, sparse, refIdx, {}, 0);
    }
    if (visiblePointIndices.size() < 20)
    {
        return cv::Mat();
    }

    const std::vector<ProjectedSparseDepthSample> samples =
        collectProjectedSparseDepthSamples(sparse, cam, W, H, visiblePointIndices);
    return buildSparseSupportMaskFromProjectedSamples(refIdx, W, H, samples);
}

cv::Mat DepthMapGenerator::buildSparseSupportMaskFromProjectedSamples(
    int refIdx,
    int W,
    int H,
    const std::vector<ProjectedSparseDepthSample> &samples)
{
    if (W <= 0 || H <= 0 || samples.size() < 20)
    {
        return cv::Mat();
    }

    std::vector<cv::Point> seed_points;
    seed_points.reserve(samples.size());
    int projectedSeeds = 0;
    for (const ProjectedSparseDepthSample &sample : samples)
    {
        const int iu = static_cast<int>(std::round(sample.uNorm * static_cast<float>(W)));
        const int iv = static_cast<int>(std::round(sample.vNorm * static_cast<float>(H)));
        if (iu < 0 || iu >= W || iv < 0 || iv >= H || sample.depth <= 0.0f)
        {
            continue;
        }

        seed_points.emplace_back(iu, iv);
        ++projectedSeeds;
    }

    std::sort(seed_points.begin(), seed_points.end(), [](const cv::Point &left,
                                                          const cv::Point &right)
    {
        return left.y < right.y || (left.y == right.y && left.x < right.x);
    });
    seed_points.erase(
        std::unique(seed_points.begin(), seed_points.end()), seed_points.end());
    const int seedPixels = static_cast<int>(seed_points.size());
    if (seedPixels < 10)
    {
        return cv::Mat();
    }

    const int maxDim = std::max(W, H);
    const int radius = maxDim < 512
        ? std::clamp(maxDim / 8, 8, 48)
        : (maxDim < 1200
            ? std::clamp(maxDim / 16, 32, 64)
            : std::clamp(maxDim / 48, 48, 128));
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(radius * 2 + 1, radius * 2 + 1));
    std::vector<std::pair<int, int>> kernel_spans(
        static_cast<std::size_t>(kernel.rows), {0, -1});
    for (int kernel_row = 0; kernel_row < kernel.rows; ++kernel_row)
    {
        const uint8_t *row = kernel.ptr<uint8_t>(kernel_row);
        int first = 0;
        while (first < kernel.cols && row[first] == 0)
        {
            ++first;
        }
        int last = kernel.cols - 1;
        while (last >= first && row[last] == 0)
        {
            --last;
        }
        kernel_spans[static_cast<std::size_t>(kernel_row)] = {first, last};
    }

    cv::Mat support(H, W, CV_8U, cv::Scalar(0));
    for (const cv::Point &seed_point : seed_points)
    {
        // Binary dilation of point seeds is exactly the union of translated
        // structuring elements. Stamp the ellipse as contiguous row spans so
        // the mask stays bit-identical to cv::dilate without scanning every
        // full-resolution pixel with a 257x257 kernel between GPU frames.
        for (int kernel_row = 0; kernel_row < kernel.rows; ++kernel_row)
        {
            const int support_row = seed_point.y + kernel_row - radius;
            if (support_row < 0 || support_row >= H)
            {
                continue;
            }
            const auto [span_first, span_last] =
                kernel_spans[static_cast<std::size_t>(kernel_row)];
            const int first = std::max(0, seed_point.x + span_first - radius);
            const int last = std::min(W - 1, seed_point.x + span_last - radius);
            if (first <= last)
            {
                uint8_t *row = support.ptr<uint8_t>(support_row);
                std::fill(row + first, row + last + 1, static_cast<uint8_t>(255));
            }
        }
    }

    const int supportPixels = cv::countNonZero(support);
    const float coverage = static_cast<float>(supportPixels) / static_cast<float>(W * H);
    if (coverage < 0.03f || coverage > 0.95f)
    {
        return cv::Mat();
    }

    LOG_DEBUG(
        "[MVS][帧 %d][稀疏支撑] seeds=%d/%d radius=%d coverage=%d/%d (%.1f%%)",
        refIdx, seedPixels, projectedSeeds, radius, supportPixels, W * H,
        coverage * 100.0f);
    return support;
}

cv::Mat DepthMapGenerator::buildSparseSupportMaskFromVisiblePoints(
    int refIdx,
    int W,
    int H,
    const std::vector<size_t> &visiblePointIndices) const
{
    if (refIdx < 0 || refIdx >= static_cast<int>(_views.size()))
    {
        return cv::Mat();
    }

    return buildSparseSupportMaskForCamera(refIdx,
                                           mvsPinholeCamera(_views[refIdx].camera),
                                           W,
                                           H,
                                           visiblePointIndices);
}

cv::Mat DepthMapGenerator::buildSparseSupportMaskForCamera(
    int refIdx,
    const Camera &camera,
    int W,
    int H,
    const std::vector<size_t> &visiblePointIndices) const
{
    if (W <= 0 || H <= 0 || refIdx < 0 || refIdx >= static_cast<int>(_views.size()) || _sparse.points.size() < 20)
    {
        return cv::Mat();
    }

    if (!camera.isValid() || visiblePointIndices.size() < 20)
    {
        return cv::Mat();
    }

    const std::vector<ProjectedSparseDepthSample> samples =
        collectProjectedSparseDepthSamples(_sparse, camera, W, H, visiblePointIndices);
    return buildSparseSupportMaskFromProjectedSamples(refIdx, W, H, samples);
}

void DepthMapGenerator::applySparseSupportPrior(cv::Mat &depthMap,
                                                cv::Mat &confidenceMap,
                                                const cv::Mat &supportMask,
                                                int refIdx)
{
    if (depthMap.empty() || supportMask.empty())
    {
        return;
    }

    cv::Mat support;
    if (supportMask.type() == CV_8U)
    {
        support = supportMask;
    }
    else
    {
        supportMask.convertTo(support, CV_8U);
    }

    if (support.size() != depthMap.size())
    {
        cv::resize(support, support, depthMap.size(), 0, 0, cv::INTER_NEAREST);
    }

    const cv::Mat validDepth = depthMap > 0;
    const int beforeValid = cv::countNonZero(validDepth);
    if (beforeValid <= 0)
    {
        return;
    }

    cv::Mat unsupportedMask;
    cv::bitwise_and(validDepth, support == 0, unsupportedMask);
    const int unsupportedValid = cv::countNonZero(unsupportedMask);
    if (unsupportedValid <= 0)
    {
        return;
    }

    if (confidenceMap.empty() ||
        confidenceMap.size() != depthMap.size() ||
        confidenceMap.type() != CV_32F)
    {
        LOG_DEBUG("[MVS][帧 %d][稀疏支撑] support 外=%d/%d，置信图不可用，深度保持不变",
                  refIdx,
                  unsupportedValid,
                  beforeValid);
        return;
    }

    constexpr float kUnsupportedConfidenceScale = 0.75f;
    for (int y = 0; y < confidenceMap.rows; ++y)
    {
        float *confRow = confidenceMap.ptr<float>(y);
        const uint8_t *maskRow = unsupportedMask.ptr<uint8_t>(y);
        for (int x = 0; x < confidenceMap.cols; ++x)
        {
            if (maskRow[x] != 0)
            {
                confRow[x] *= kUnsupportedConfidenceScale;
            }
        }
    }

    LOG_DEBUG("[MVS][帧 %d][稀疏支撑] support 外=%d/%d，置信度缩放=%.2f，深度保持不变",
              refIdx,
              unsupportedValid,
              beforeValid,
              kUnsupportedConfidenceScale);
}

int DepthMapGenerator::removeLocalDepthOutliers(cv::Mat &depthMap,
                                                cv::Mat &confidenceMap,
                                                int kernelSize,
                                                float relDepthThreshold,
                                                float maxRemovalRatio,
                                                int refIdx)
{
    if (depthMap.empty() || depthMap.type() != CV_32F)
    {
        return 0;
    }
    if (kernelSize < 3 || relDepthThreshold <= 0.0f || maxRemovalRatio <= 0.0f)
    {
        return 0;
    }

    const cv::Mat validMask = depthMap > 0.0f;
    const int validBefore = cv::countNonZero(validMask);
    if (validBefore <= 0)
    {
        return 0;
    }

    const int medianKernel = std::clamp(kernelSize | 1, 3, 5);
    cv::Mat localMedian;
    cv::medianBlur(depthMap, localMedian, medianKernel);

    cv::Mat outlierMask = cv::Mat::zeros(depthMap.size(), CV_8U);
#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < depthMap.rows; ++y)
    {
        const float *depthRow = depthMap.ptr<float>(y);
        const float *medianRow = localMedian.ptr<float>(y);
        uint8_t *maskRow = outlierMask.ptr<uint8_t>(y);
        for (int x = 0; x < depthMap.cols; ++x)
        {
            const float depth = depthRow[x];
            const float medianDepth = medianRow[x];
            if (depth <= 0.0f || medianDepth <= 0.0f)
            {
                continue;
            }

            const float relDiff = std::fabs(depth - medianDepth) / std::max(medianDepth, 1e-6f);
            if (relDiff > relDepthThreshold)
            {
                maskRow[x] = 255;
            }
        }
    }

    const int candidateCount = cv::countNonZero(outlierMask);
    if (candidateCount <= 0)
    {
        return 0;
    }

    const float removalRatio = static_cast<float>(candidateCount) / static_cast<float>(validBefore);
    if (removalRatio > maxRemovalRatio)
    {
        LOG_DEBUG("[MVS][帧 %d][后处理] 局部离群候选过多 %d/%d (%.1f%% > %.1f%%)，跳过",
                  refIdx,
                  candidateCount,
                  validBefore,
                  removalRatio * 100.0f,
                  maxRemovalRatio * 100.0f);
        return 0;
    }

    depthMap.setTo(0.0f, outlierMask);
    if (!confidenceMap.empty() &&
        confidenceMap.size() == depthMap.size() &&
        confidenceMap.type() == CV_32F)
    {
        confidenceMap.setTo(0.0f, outlierMask);
    }

    LOG_DEBUG("[MVS][帧 %d][后处理] 局部离群移除=%d/%d kernel=%d relative_threshold=%.2f",
              refIdx,
              candidateCount,
              validBefore,
              medianKernel,
              relDepthThreshold);
    return candidateCount;
}

int DepthMapGenerator::removeSmallDepthComponents(cv::Mat &depthMap,
                                                  cv::Mat &confidenceMap,
                                                  int minComponentArea,
                                                  float maxRemovalRatio,
                                                  int refIdx)
{
    if (depthMap.empty() || depthMap.type() != CV_32F)
    {
        return 0;
    }
    if (minComponentArea <= 1 || maxRemovalRatio <= 0.0f)
    {
        return 0;
    }

    const cv::Mat validMask = depthMap > 0.0f;
    const int validBefore = cv::countNonZero(validMask);
    if (validBefore <= 0)
    {
        return 0;
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int componentCount = cv::connectedComponentsWithStats(
        validMask,
        labels,
        stats,
        centroids,
        8,
        CV_32S);

    std::vector<unsigned char> remove_labels(
        static_cast<std::size_t>(componentCount), 0);
    int candidateCount = 0;
    for (int label = 1; label < componentCount; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area <= 0 || area >= minComponentArea)
        {
            continue;
        }

        candidateCount += area;
        remove_labels[static_cast<std::size_t>(label)] = 1;
    }

    if (candidateCount <= 0)
    {
        return 0;
    }

    const float removalRatio = static_cast<float>(candidateCount) / static_cast<float>(validBefore);
    if (removalRatio > maxRemovalRatio)
    {
        LOG_DEBUG("[MVS][帧 %d][后处理] 小连通域候选过多 %d/%d (%.1f%% > %.1f%%)，跳过",
                  refIdx,
                  candidateCount,
                  validBefore,
                  removalRatio * 100.0f,
                  maxRemovalRatio * 100.0f);
        return 0;
    }

    cv::Mat removeMask = cv::Mat::zeros(depthMap.size(), CV_8U);
    cv::parallel_for_(cv::Range(0, labels.rows), [&](const cv::Range &range)
    {
        for (int y = range.start; y < range.end; ++y)
        {
            const int *label_row = labels.ptr<int>(y);
            unsigned char *mask_row = removeMask.ptr<unsigned char>(y);
            for (int x = 0; x < labels.cols; ++x)
            {
                const int label = label_row[x];
                if (label > 0 &&
                    remove_labels[static_cast<std::size_t>(label)] != 0)
                {
                    mask_row[x] = 255;
                }
            }
        }
    });

    depthMap.setTo(0.0f, removeMask);
    if (!confidenceMap.empty() &&
        confidenceMap.size() == depthMap.size() &&
        confidenceMap.type() == CV_32F)
    {
        confidenceMap.setTo(0.0f, removeMask);
    }

    LOG_DEBUG("[MVS][帧 %d][后处理] 小连通域移除=%d/%d min_area=%d",
              refIdx,
              candidateCount,
              validBefore,
              minComponentArea);
    return candidateCount;
}

DepthPostProcessStats DepthMapGenerator::postprocessFusionDepthMap(cv::Mat &depthMap,
                                                                    cv::Mat &confidenceMap,
                                                                    const FusionConfig &config,
                                                                    int refIdx,
                                                                    int viewCount,
                                                                    cv::Mat *missingReasonMap,
                                                                    const DepthPostProcessEvidence *evidence)
{
    DepthPostProcessStats stats;
    if (depthMap.empty() || depthMap.type() != CV_32F)
    {
        return stats;
    }

    stats.validBeforePostprocess = cv::countNonZero(depthMap > 0.0f);
    stats.validAfterConfidenceFilter = stats.validBeforePostprocess;
    stats.validAfterPostprocess = stats.validBeforePostprocess;
    if (stats.validBeforePostprocess <= 0)
    {
        return stats;
    }

    float confThresh = config.confidenceThresh;
    if (viewCount <= 2)
    {
        confThresh = 0.0f;
    }

    const bool hasConfidence = !confidenceMap.empty()
        && confidenceMap.size() == depthMap.size()
        && confidenceMap.type() == CV_32F;
    bool adaptiveConfidenceRaised = false;
    if (hasConfidence)
    {
        double cMin = 0.0;
        double cMax = 0.0;
        cv::minMaxLoc(confidenceMap, &cMin, &cMax);
        const cv::Mat validMask = depthMap > 0.0f;
        const cv::Scalar cMean = cv::mean(confidenceMap, validMask);
        LOG_DEBUG("[MVS][帧 %d][后处理] confidence min=%.4f max=%.4f mean=%.4f threshold=%.4f",
                  refIdx,
                  cMin,
                  cMax,
                  cMean[0],
                  confThresh);

        if (viewCount > 2 && config.enableAdaptiveConfidenceFilter)
        {
            const DepthMapQualityMetrics quality =
                analyzeDepthMapQuality(depthMap, confidenceMap, viewCount);
            const bool suspiciousFullCoverage =
                quality.validCoverage >= config.adaptiveFullCoverageThreshold &&
                quality.meanConfidence > 0.0f &&
                quality.meanConfidence < config.adaptiveLowMeanConfidenceThreshold;
            if (suspiciousFullCoverage)
            {
                const float strictThreshold = std::max(
                    config.adaptiveStrictConfidenceThreshold,
                    quality.recommendedFusionConfidence);
                if (strictThreshold > confThresh)
                {
                    LOG_DEBUG("[MVS][帧 %d][后处理] 低置信满幅深度: coverage=%.3f mean_confidence=%.3f "
                              "threshold=%.3f->%.3f",
                              refIdx,
                              quality.validCoverage,
                              quality.meanConfidence,
                              confThresh,
                              strictThreshold);
                    confThresh = strictThreshold;
                    adaptiveConfidenceRaised = true;
                }
            }
        }

        if (confThresh > 0.0f)
        {
            cv::Mat beforeConfidence = depthMap.clone();
            const bool hasGeometryEvidence = evidence &&
                evidence->geometrySupportCount.type() == CV_16UC1 &&
                evidence->geometrySupportCount.size() == depthMap.size() &&
                evidence->inverseDepthRelativeSpread.type() == CV_32FC1 &&
                evidence->inverseDepthRelativeSpread.size() == depthMap.size();
            const bool hasAdaptiveGeometryEvidence = hasGeometryEvidence &&
                evidence->adaptiveSupportWeight.type() == CV_32FC1 &&
                evidence->adaptiveSupportWeight.size() == depthMap.size() &&
                evidence->adaptiveEffectiveViewCount.type() == CV_32FC1 &&
                evidence->adaptiveEffectiveViewCount.size() == depthMap.size() &&
                evidence->adaptiveConflictRatio.type() == CV_32FC1 &&
                evidence->adaptiveConflictRatio.size() == depthMap.size();
            int lowConfidenceCandidateCount = 0;
            int geometrySupportedRetainedCount = 0;
#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:lowConfidenceCandidateCount, geometrySupportedRetainedCount)
#endif
            for (int v = 0; v < depthMap.rows; ++v)
            {
                float *depthRow = depthMap.ptr<float>(v);
                const float *confRow = confidenceMap.ptr<float>(v);
                const std::uint16_t *geometrySupportRow = hasGeometryEvidence
                    ? evidence->geometrySupportCount.ptr<std::uint16_t>(v) : nullptr;
                const float *inverseDepthSpreadRow = hasGeometryEvidence
                    ? evidence->inverseDepthRelativeSpread.ptr<float>(v) : nullptr;
                const float *adaptiveSupportRow = hasAdaptiveGeometryEvidence
                    ? evidence->adaptiveSupportWeight.ptr<float>(v) : nullptr;
                const float *adaptiveEffectiveViewRow = hasAdaptiveGeometryEvidence
                    ? evidence->adaptiveEffectiveViewCount.ptr<float>(v) : nullptr;
                const float *adaptiveConflictRow = hasAdaptiveGeometryEvidence
                    ? evidence->adaptiveConflictRatio.ptr<float>(v) : nullptr;
                for (int u = 0; u < depthMap.cols; ++u)
                {
                    if (depthRow[u] > 0.0f && confRow[u] < confThresh)
                    {
                        ++lowConfidenceCandidateCount;
                        bool retainForGeometry =
                            config.enableGeometrySupportedLowConfidenceRetention &&
                            confRow[u] >= config.geometrySupportedMinimumConfidence &&
                            hasGeometryEvidence &&
                            geometrySupportRow[u] >=
                                config.geometrySupportedMinimumObservationCount &&
                            std::isfinite(inverseDepthSpreadRow[u]) &&
                            inverseDepthSpreadRow[u] >= 0.0f &&
                            inverseDepthSpreadRow[u] <=
                                config.geometrySupportedMaximumInverseDepthSpread;
                        if (retainForGeometry && hasAdaptiveGeometryEvidence)
                        {
                            retainForGeometry =
                                std::isfinite(adaptiveSupportRow[u]) &&
                                adaptiveSupportRow[u] >=
                                    config.geometrySupportedMinimumAdaptiveSupportWeight &&
                                std::isfinite(adaptiveEffectiveViewRow[u]) &&
                                adaptiveEffectiveViewRow[u] >=
                                    config.geometrySupportedMinimumAdaptiveEffectiveViews &&
                                std::isfinite(adaptiveConflictRow[u]) &&
                                adaptiveConflictRow[u] <=
                                    config.geometrySupportedMaximumAdaptiveConflictRatio;
                        }
                        if (retainForGeometry)
                        {
                            ++geometrySupportedRetainedCount;
                        }
                        else
                        {
                            depthRow[u] = 0.0f;
                        }
                    }
                }
            }

            stats.lowConfidenceCandidateCount = lowConfidenceCandidateCount;
            stats.geometrySupportedLowConfidenceRetained =
                geometrySupportedRetainedCount;

            int validAfterConfidence = cv::countNonZero(depthMap > 0.0f);
            LOG_DEBUG("[MVS][帧 %d][后处理] 置信度过滤 %d->%d threshold=%.4f "
                      "candidates=%d geometry_retained=%d",
                      refIdx,
                      stats.validBeforePostprocess,
                      validAfterConfidence,
                      confThresh,
                      lowConfidenceCandidateCount,
                      geometrySupportedRetainedCount);

            if (!adaptiveConfidenceRaised &&
                validAfterConfidence < stats.validBeforePostprocess / 20)
            {
                LOG_WARN("[MVS][帧 %d][后处理] 置信度过滤后像素过少，已回退为不过滤", refIdx);
                depthMap = std::move(beforeConfidence);
                validAfterConfidence = stats.validBeforePostprocess;
            }

            stats.validAfterConfidenceFilter = validAfterConfidence;
            stats.confidenceRemoved = std::max(0, stats.validBeforePostprocess - validAfterConfidence);
            if (missingReasonMap)
            {
                markDepthLossReason(*missingReasonMap,
                                    beforeConfidence,
                                    depthMap,
                                    DepthMissingReason::LowConfidence);
            }
        }
    }
    stats.effectiveConfidenceThreshold = confThresh;

    if (config.enableLocalDepthOutlierFilter)
    {
        const cv::Mat before_local_filter = depthMap.clone();
        stats.localDepthOutlierRemoved = removeLocalDepthOutliers(
            depthMap,
            confidenceMap,
            config.localDepthOutlierKernelSize,
            config.localDepthOutlierRelThresh,
            config.maxLocalDepthOutlierRemovalRatio,
            refIdx);
        if (missingReasonMap)
        {
            markDepthLossReason(*missingReasonMap,
                                before_local_filter,
                                depthMap,
                                DepthMissingReason::LocalDepthOutlier);
        }
    }

    if (config.enableSpeckleFilter)
    {
        const cv::Mat before_speckle_filter = depthMap.clone();
        stats.smallComponentRemoved = removeSmallDepthComponents(
            depthMap,
            confidenceMap,
            config.minSpeckleComponentArea,
            config.maxSpeckleRemovalRatio,
            refIdx);
        stats.speckleRemoved = stats.smallComponentRemoved;
        if (missingReasonMap)
        {
            markDepthLossReason(*missingReasonMap,
                                before_speckle_filter,
                                depthMap,
                                DepthMissingReason::SmallComponent);
        }
    }

    stats.validAfterPostprocess = cv::countNonZero(depthMap > 0.0f);
    if (stats.confidenceRemoved > 0
        || stats.localDepthOutlierRemoved > 0
        || stats.smallComponentRemoved > 0)
    {
        LOG_DEBUG("[MVS][帧 %d][后处理] before=%d after_confidence=%d confidence_removed=%d "
                  "geometry_retained=%d local_removed=%d speckle_removed=%d after=%d",
                  refIdx,
                  stats.validBeforePostprocess,
                  stats.validAfterConfidenceFilter,
                  stats.confidenceRemoved,
                  stats.geometrySupportedLowConfidenceRetained,
                  stats.localDepthOutlierRemoved,
                  stats.speckleRemoved,
                  stats.validAfterPostprocess);
    }
    return stats;
}

// =============================================================================
DepthFrameResult DepthMapGenerator::computeDepthForView(
    int refIdx,
    const DepthGenConfig *configOverride,
    const std::function<bool(const DepthLevelSummary &, std::string *)>
        &firstLevelCompletionGate)
{
    DepthFrameResult result;
    result.refViewIdx = refIdx;
    result.success = false;

    const DepthGenConfig &config = configOverride ? *configOverride : _config;
    FrameTiming timing;
    const auto totalStart = Clock::now();
    auto stageStart = totalStart;
    const auto cancelled = [this, &result](const char *stage) -> bool
    {
        if (!_cancelled.load())
        {
            return false;
        }
        result.errorMsg = std::string("深度估计已取消: ") + stage;
        return true;
    };

    const CameraView &refView = _views[refIdx];
    if (cancelled("读取参考影像前"))
    {
        return result;
    }

    std::string referenceLoadError;
    MvsImageCache::ImageLease referenceLease = acquireImageFrame(
        refIdx, &referenceLoadError);
    if (!referenceLease)
    {
        result.errorMsg = "无法获取参考帧图像: " + refView.imagePath +
            " (" + referenceLoadError + ")";
        return result;
    }
    if (cancelled("读取参考影像后"))
    {
        return result;
    }
    cv::Mat refImg = referenceLease->preparedGray;
    const Camera refCam = referenceLease->preparedCamera;
    const int W = refImg.cols;
    const int H = refImg.rows;

    // 选择源帧（从缓存中取，省去重复加载）
    const int NV = static_cast<int>(_views.size());
    int numSrc = std::min(config.numSourceViews, NV - 1);
    std::vector<cv::Mat> srcGrays;
    std::vector<Camera> srcCams;
    std::vector<int> sourceIndices;
    std::vector<MvsImageCache::ImageLease> sourceLeases;
    sourceLeases.reserve(static_cast<std::size_t>(std::max(0, numSrc)));

    const std::vector<int> selectedSources = sourceViewIndicesForFrame(refIdx, numSrc);
    for (int si : selectedSources)
    {
        if (si < 0 || si >= NV || si == refIdx)
        {
            continue;
        }
        std::string sourceLoadError;
        MvsImageCache::ImageLease sourceLease = acquireImageFrame(
            si, &sourceLoadError);
        if (!sourceLease)
        {
            LOG_WARN(QStringLiteral("[MVS] 跳过源帧 %1：图像 provider 获取失败：%2")
                         .arg(si)
                         .arg(QString::fromStdString(sourceLoadError)));
            continue;
        }
        cv::Mat srcImg = sourceLease->preparedGray;
        Camera sourceCamera = sourceLease->preparedCamera;
        if (srcImg.cols != W || srcImg.rows != H)
        {
            const double scaleX = static_cast<double>(W) / std::max(1, srcImg.cols);
            const double scaleY = static_cast<double>(H) / std::max(1, srcImg.rows);
            cv::resize(srcImg, srcImg, cv::Size(W, H));
            sourceCamera = sourceCamera.scaledIntrinsics(scaleX, scaleY);
        }
        srcGrays.push_back(srcImg);
        srcCams.push_back(sourceCamera);
        sourceIndices.push_back(si);
        sourceLeases.push_back(std::move(sourceLease));
        if (static_cast<int>(srcGrays.size()) >= numSrc) break;
    }

    if (srcGrays.empty()) {
        timing.sourceMs = elapsedMs(stageStart, Clock::now());
        result.errorMsg = "没有可用的源帧";
        return result;
    }
    if (cancelled("选择源视图后"))
    {
        return result;
    }
    result.sourceViewIndices = sourceIndices;
    if (_frameCachesReady &&
        refIdx >= 0 &&
        refIdx < static_cast<int>(_frameCaches.size()))
    {
        const auto &cache = _frameCaches[static_cast<size_t>(refIdx)];
        result.requestedSourceViewCount = cache.requestedSourceViewCount;
        result.sourceViewShortfall = std::max(
            0,
            result.requestedSourceViewCount -
                static_cast<int>(result.sourceViewIndices.size()));
        result.sourceViewShortfallReason = cache.sourceViewShortfallReason;
        result.sourceViewPlan.reserve(result.sourceViewIndices.size());
        for (const int sourceIndex : result.sourceViewIndices)
        {
            const auto score = std::find_if(
                cache.sourceViewScores.cbegin(),
                cache.sourceViewScores.cend(),
                [sourceIndex](const MvsSourcePlanEntry &entry)
                {
                    return entry.viewIndex == sourceIndex;
                });
            if (score != cache.sourceViewScores.cend())
            {
                result.sourceViewPlan.push_back(*score);
            }
        }
    }
    if (result.requestedSourceViewCount <= 0)
    {
        result.requestedSourceViewCount =
            static_cast<int>(result.sourceViewIndices.size());
        result.sourceViewShortfall = 0;
    }

    {
        std::ostringstream oss;
        for (size_t k = 0; k < sourceIndices.size(); ++k)
        {
            if (k > 0)
            {
                oss << ",";
            }
            oss << sourceIndices[k];
        }
        LOG_DEBUG("[MVS][帧 %d][源视图] 共视评分选择 [%s]",
                  refIdx, oss.str().c_str());

        if (_frameCachesReady &&
            refIdx >= 0 &&
            refIdx < static_cast<int>(_frameCaches.size()) &&
            !_frameCaches[static_cast<size_t>(refIdx)].sourceViewScores.empty())
        {
            std::ostringstream details;
            int detailCount = 0;
            const auto &scores = _frameCaches[static_cast<size_t>(refIdx)].sourceViewScores;
            for (int sourceIndex : sourceIndices)
            {
                const auto it = std::find_if(scores.begin(), scores.end(),
                                             [sourceIndex](const MvsSourcePlanEntry &score)
                                             {
                                                 return score.viewIndex == sourceIndex;
                                             });
                if (it == scores.end())
                {
                    continue;
                }
                if (detailCount > 0)
                {
                    details << "; ";
                }
                details << it->viewIndex
                        << "(tracks=" << it->sharedTracks
                        << ",inliers=" << it->geometricInliers
                        << ",verified_pair=" << (it->verifiedPairGeometry ? 1 : 0)
                        << ",angle=" << it->medianTriangulationAngleDeg
                        << ",coverage=" << it->coverageScore
                        << ",score=" << it->score << ")";
                ++detailCount;
            }
            if (detailCount > 0)
            {
                LOG_DEBUG(QStringLiteral("[MVS][帧 %1][源视图] 诊断: %2")
                             .arg(refIdx)
                             .arg(QString::fromStdString(details.str())));
            }
        }
    }
    timing.sourceMs = elapsedMs(stageStart, Clock::now());

    const int minSourceViews = sourceIndices.empty() ? 0 : 1;
    const std::vector<size_t> visibleSparsePointIndices = [this, refIdx, &sourceIndices, minSourceViews]()
    {
        std::vector<size_t> points = visibleSparsePointIndicesForFrame(refIdx, sourceIndices, minSourceViews);
        if (points.empty() && minSourceViews > 0)
        {
            points = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
            LOG_DEBUG("[MVS][帧 %d][稀疏引导] 共视点为空，回退参考帧可见点=%zu",
                      refIdx, points.size());
        }
        return points;
    }();

    // 自适应置信度阈值：源视图越少，NCC 方差越大，需适当降低阈值
    PatchMatchConfig pmCfg = config.patchMatch;
    pmCfg.cpuThreadCount = std::max(1, config.cpuWorkerCount);
    if ((int)srcGrays.size() == 1) {
        // 单源视图：使用较低但非零的置信度阈值。
        // 完全归零会保留所有随机初始化深度（90% 以上暗像素 NCC=0 → conf=0），
        // 导致可视化和 crossCheck 被海量噪声淹没。
        // 阈值 0.10 ≈ NCC > 0.1，可过滤纯随机噪声但保留弱匹配。
        pmCfg.confidenceThresh = 0.10f;
        LOG_DEBUG("[MVS][帧 %d][配置] 单源视图模式，GPU confidence_threshold=0.10",
                  refIdx);
    } else if ((int)srcGrays.size() <= 2) {
        // 2 源视图：适当降低阈值但不可过低，0.10 会保留大量低质量匹配→噪声
        pmCfg.confidenceThresh = std::min(pmCfg.confidenceThresh, 0.20f);
    }
    const DepthConfidenceThresholds confidence_thresholds =
        depthConfidenceThresholds(
            _effectiveSceneProfile,
            _effectiveDepthFilterMode,
            static_cast<int>(srcGrays.size()),
            pmCfg.confidenceThresh,
            config.fusion.confidenceThresh);
    pmCfg.confidenceThresh = confidence_thresholds.patchMatch;
    result.effectivePatchMatchConfidenceThreshold = pmCfg.confidenceThresh;

    // 详细相机诊断仅写入 Debug 日志，避免默认终端被逐帧矩阵信息淹没。
    const std::array<double, 9> refRotation = refCam.worldToCameraRotation();
    const std::array<double, 3> refCenter = refCam.cameraCenter();
    LOG_DEBUG("[MVS][帧 %d][配置] image=%dx%d sources=%d det_rotation=%.4f center=[%.3f,%.3f,%.3f]",
              refIdx, W, H, static_cast<int>(srcCams.size()), det3(refRotation.data()),
              refCenter[0], refCenter[1], refCenter[2]);

    // 深度范围
    stageStart = Clock::now();
    float zNear, zFar;
    std::vector<size_t> depthRangeVisiblePoints = visibleSparsePointIndices;
    if (depthRangeVisiblePoints.size() < 5 && minSourceViews > 0)
    {
        depthRangeVisiblePoints = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
        LOG_DEBUG("[MVS][帧 %d][深度范围] 共视点不足，回退参考帧可见点=%zu",
                  refIdx, depthRangeVisiblePoints.size());
    }
    estimateDepthRangeFromVisiblePoints(refIdx, depthRangeVisiblePoints, zNear, zFar);
    LOG_DEBUG("[MVS][帧 %d][深度范围] near=%.4f far=%.4f", refIdx, zNear, zFar);
    timing.rangeMs = elapsedMs(stageStart, Clock::now());
    if (cancelled("深度范围估计后"))
    {
        return result;
    }

    // =========================================================================
    // ★ 极线校正（仅双目立体对时启用）
    //   将两张图像校正到极线对齐状态，使 PatchMatch 的搜索从 2D 降为近似 1D，
    //   显著降低匹配噪声。
    //   始终以较小索引为 left 进行校正，避免不同帧顺序产生不同校正几何。
    // =========================================================================
    mvs::EpipolarRectifier::RectifiedPair rectPair;
    bool useRectified = false;
    cv::Mat workRefImg = refImg;
    std::vector<cv::Mat> workSrcGrays = srcGrays;
    Camera workRefCam = refCam;
    std::vector<Camera> workSrcCams = srcCams;
    cv::Mat referenceValidMask = referenceLease->validMask;
    std::vector<cv::Mat> source_valid_masks;
    source_valid_masks.reserve(sourceIndices.size());
    bool has_source_valid_mask = false;
    for (std::size_t sourceOrdinal = 0;
         sourceOrdinal < sourceIndices.size();
         ++sourceOrdinal)
    {
        cv::Mat source_mask = sourceLeases[sourceOrdinal]->validMask;
        if (!source_mask.empty() && source_mask.size() != cv::Size(W, H))
        {
            cv::resize(source_mask,
                       source_mask,
                       cv::Size(W, H),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        has_source_valid_mask = has_source_valid_mask || !source_mask.empty();
        source_valid_masks.push_back(std::move(source_mask));
    }
    if (!has_source_valid_mask)
    {
        source_valid_masks.clear();
    }
    const bool has_project_mask = referenceLease->projectMaskLoaded;
    if (has_project_mask)
    {
        result.maskSource = "project";
    }
    else if (!referenceValidMask.empty())
    {
        result.maskSource = "content";
    }
    else
    {
        result.maskSource = "full_image";
    }
    result.maskCoverage = referenceValidMask.empty()
        ? 1.0f
        : static_cast<float>(cv::countNonZero(referenceValidMask)) /
              static_cast<float>(std::max<std::size_t>(1, referenceValidMask.total()));
    cv::Mat workReferenceValidMask = referenceValidMask;
    std::vector<cv::Mat> work_source_valid_masks = source_valid_masks;

    stageStart = Clock::now();
    if (srcGrays.size() == 1)
    {
        int srcIdx = sourceIndices.empty() ? -1 : sourceIndices.front();

        bool refIsCanonicalLeft = (srcIdx < 0 || refIdx < srcIdx);

        cv::Mat canonLeft  = refIsCanonicalLeft ? refImg      : srcGrays[0];
        cv::Mat canonRight = refIsCanonicalLeft ? srcGrays[0] : refImg;
        auto    camL       = refIsCanonicalLeft ? refCam      : srcCams[0];
        auto    camR       = refIsCanonicalLeft ? srcCams[0]  : refCam;

        std::string rectErr;
        if (mvs::EpipolarRectifier::rectify(
                canonLeft, canonRight, camL, camR, rectPair, &rectErr))
        {
            if (refIsCanonicalLeft) {
                workRefImg = rectPair.rectLeft;
                workSrcGrays = { rectPair.rectRight };
                workRefCam = rectPair.rectCamLeft;
                workSrcCams = { rectPair.rectCamRight };
                rectPair.refIsRight = false;
            } else {
                workRefImg = rectPair.rectRight;
                workSrcGrays = { rectPair.rectLeft };
                workRefCam = rectPair.rectCamRight;
                workSrcCams = { rectPair.rectCamLeft };
                rectPair.refIsRight = true;
            }
            useRectified = true;
            if (!referenceValidMask.empty())
            {
                const cv::Mat &reference_homography = refIsCanonicalLeft
                    ? rectPair.H1
                    : rectPair.H2;
                cv::warpPerspective(referenceValidMask,
                                    workReferenceValidMask,
                                    reference_homography,
                                    workRefImg.size(),
                                    cv::INTER_NEAREST,
                                    cv::BORDER_CONSTANT,
                                    cv::Scalar(0));
            }
            if (!source_valid_masks.empty() && !source_valid_masks.front().empty())
            {
                const cv::Mat &source_homography = refIsCanonicalLeft
                    ? rectPair.H2
                    : rectPair.H1;
                cv::warpPerspective(source_valid_masks.front(),
                                    work_source_valid_masks.front(),
                                    source_homography,
                                    workSrcGrays.front().size(),
                                    cv::INTER_NEAREST,
                                    cv::BORDER_CONSTANT,
                                    cv::Scalar(0));
            }
            LOG_DEBUG("[MVS][帧 %d][极线校正] 成功 reference=%s",
                      refIdx, refIsCanonicalLeft ? "left" : "right");
        }
        else
        {
            LOG_WARN("[MVS][帧 %d][极线校正] 失败，使用原始图像: %s",
                     refIdx, rectErr.c_str());
        }
    }
    timing.rectifyMs = elapsedMs(stageStart, Clock::now());
    if (cancelled("极线校正后"))
    {
        return result;
    }

    // 三级深度金字塔：Level 3 全局结构、Level 2 几何稳定、Level 1 细节恢复。
    cv::Mat depthMap;
    cv::Mat confMap;
    cv::Mat normalMap;
    cv::Mat supportCount;
    DepthPyramidConfig pyramid_config = makeDepthPyramidConfig(pmCfg,
                                                               workRefImg.cols,
                                                               workRefImg.rows);
    result.pyramidRequestedLevelCount = 3;
    result.pyramidActiveLevelCount = pyramid_config.activeLevelCount;
    result.pyramidMinimumShortSide = depthPyramidMinimumLevelShortSide();
    result.pyramidDegradedReason = pyramid_config.degradedReason;
    pyramid_config.sceneProfile = _effectiveSceneProfile;
    pyramid_config.filterMode = _effectiveDepthFilterMode;
    pyramid_config.saveIntermediateLevels = _config.saveIntermediatePyramidLevels;
    for (int level_index = 0; level_index < pyramid_config.activeLevelCount; ++level_index)
    {
        PatchMatchConfig &level_config = pyramid_config.levels[level_index].patchMatch;
        level_config.epipolarRectified = useRectified;
        if (pyramid_config.levels[level_index].level == 3)
        {
            level_config.confidenceThresh = std::min(level_config.confidenceThresh, 0.08f);
        }
        else if (pyramid_config.levels[level_index].level == 2)
        {
            level_config.confidenceThresh = std::min(level_config.confidenceThresh, 0.25f);
        }
    }

    stageStart = Clock::now();
    const std::vector<ProjectedSparseDepthSample> workRefSparseSamples =
        collectProjectedSparseDepthSamples(_sparse,
                                           workRefCam,
                                           workRefImg.cols,
                                           workRefImg.rows,
                                           visibleSparsePointIndices);
    std::array<cv::Mat, 3> pyramid_sparse_hints;
    for (int level_index = 0; level_index < pyramid_config.activeLevelCount; ++level_index)
    {
        const cv::Size hint_size = patchMatchWorkSize(
            workRefImg, pyramid_config.levels[level_index].patchMatch);
        pyramid_sparse_hints[level_index] = level_index == 0
            ? buildHintDepthFromProjectedSamples(refIdx,
                                                 hint_size.width,
                                                 hint_size.height,
                                                 workRefSparseSamples)
            : buildSparseSeedDepthFromProjectedSamples(refIdx,
                                                       hint_size.width,
                                                       hint_size.height,
                                                       workRefSparseSamples);
    }

    const PatchMatchConfig &support_mask_config =
        pyramid_config.levels[pyramid_config.activeLevelCount - 1].patchMatch;
    const cv::Size supportMaskSize = patchMatchWorkSize(refImg, support_mask_config);
    std::vector<ProjectedSparseDepthSample> rectifiedSupportSamples;
    const std::vector<ProjectedSparseDepthSample> *supportSamples = &workRefSparseSamples;
    if (useRectified)
    {
        rectifiedSupportSamples =
            collectProjectedSparseDepthSamples(_sparse, refCam, W, H, visibleSparsePointIndices);
        supportSamples = &rectifiedSupportSamples;
    }
    cv::Mat sparseSupportMask = buildSparseSupportMaskFromProjectedSamples(refIdx,
                                                                           supportMaskSize.width,
                                                                           supportMaskSize.height,
                                                                           *supportSamples);
    timing.hintMs = elapsedMs(stageStart, Clock::now());
    if (cancelled("构建三级深度先验后"))
    {
        return result;
    }

    stageStart = Clock::now();
    AdaptivePatchMatchBackend pyramid_backend(refIdx);
    DepthPyramidEstimator pyramid_estimator(&pyramid_backend);
    DepthPyramidRequest pyramid_request;
    pyramid_request.referenceImage = workRefImg;
    pyramid_request.referenceValidMask = workReferenceValidMask;
    pyramid_request.sourceImages = workSrcGrays;
    pyramid_request.sourceValidMasks = work_source_valid_masks;
    pyramid_request.guideImage = workRefImg;
    pyramid_request.referenceCamera = workRefCam;
    pyramid_request.sourceCameras = workSrcCams;
    pyramid_request.zNear = zNear;
    pyramid_request.zFar = zFar;
    pyramid_request.pyramidConfig = pyramid_config;
    pyramid_request.sparseDepthHints = pyramid_sparse_hints;
    pyramid_request.cancelFlag = pmCfg.cancelFlag;
    if (pyramid_config.activeLevelCount > 1)
    {
        pyramid_request.firstLevelCompletionGate = firstLevelCompletionGate;
    }

    DepthPyramidResult pyramid_result = pyramid_estimator.estimate(pyramid_request);
    if (cancelled("三级 PatchMatch 后"))
    {
        return result;
    }
    if (!pyramid_result.success)
    {
        result.errorMsg = pyramid_result.errorMessage;
        return result;
    }

    depthMap = std::move(pyramid_result.finalLevel.depth);
    confMap = std::move(pyramid_result.finalLevel.confidence);
    normalMap = std::move(pyramid_result.finalLevel.normalMap);
    supportCount = std::move(pyramid_result.finalLevel.supportCount);
    result.selectedLevel = pyramid_result.finalLevel.level;
    {
        std::vector<std::string> fallback_reasons;
        if (!pyramid_config.degradedReason.empty())
        {
            fallback_reasons.push_back(pyramid_config.degradedReason);
        }
        if (!pyramid_result.errorMessage.empty())
        {
            fallback_reasons.push_back(pyramid_result.errorMessage);
        }
        std::ostringstream reason_stream;
        for (std::size_t index = 0; index < fallback_reasons.size(); ++index)
        {
            if (index > 0)
            {
                reason_stream << "; ";
            }
            reason_stream << fallback_reasons[index];
        }
        result.fallbackReason = reason_stream.str();
    }
    result.pyramidLevels = std::move(pyramid_result.levelSummaries);
    result.intermediatePyramidLevels = std::move(pyramid_result.intermediateLevels);
    for (const DepthLevelSummary &summary : result.pyramidLevels)
    {
        LOG_DEBUG("[MVS][帧 %d][金字塔] level=%d ds=%d valid=%d coverage=%.1f%% "
                  "confidence=%.3f elapsed=%.1f ms status=%s",
                  refIdx,
                  summary.level,
                  summary.downsampleFactor,
                  summary.validPixelCount,
                  summary.validCoverage * 100.0f,
                  summary.meanConfidence,
                  summary.elapsedMs,
                  summary.success ? "ok" : "failed");
    }
    if (!pyramid_result.errorMessage.empty())
    {
        LOG_WARN("[MVS][帧 %d][金字塔] 降级使用 level=%d: %s",
                 refIdx,
                 pyramid_result.finalLevel.level,
                 pyramid_result.errorMessage.c_str());
    }

    // ── 极线校正反变换：将校正空间的深度图映射回原始图像空间 ──────────────
    if (useRectified && !depthMap.empty())
    {
        depthMap = mvs::EpipolarRectifier::unrectifyDepth(
            depthMap, rectPair, W, H);
        if (!confMap.empty())
            confMap = mvs::EpipolarRectifier::unrectifyDepth(
                confMap, rectPair, W, H);
        if (!supportCount.empty())
        {
            supportCount = mvs::EpipolarRectifier::unrectifyNearest(
                supportCount, rectPair, W, H);
        }
        // Rectified normals are expressed in the rectified camera frame. Re-estimate them from
        // fused depth later instead of attaching vectors in the wrong coordinate frame.
        normalMap.release();
        LOG_DEBUG("[MVS][帧 %d][极线校正] 深度图已映射回原始空间", refIdx);
    }
    result.depthCompleteness.pyramidValidCount = cv::countNonZero(depthMap > 0.0f);
    timing.patchmatchMs = elapsedMs(stageStart, Clock::now());
    if (cancelled("深度图反变换后"))
    {
        return result;
    }

    // 在原始图像空间再次应用有效区域，消除极线反变换插值带来的边界泄漏。
    // 项目蒙版优先；未提供项目蒙版时使用 CLAHE 前计算的内容掩码。
    stageStart = Clock::now();
    cv::Mat effectiveReferenceMask;
    {
        if (!referenceLease->validMask.empty())
        {
            effectiveReferenceMask = referenceLease->validMask;
        }
        else
        {
            LOG_DEBUG("[MVS][帧 %d][掩码] 内容掩码已自动跳过", refIdx);
        }

        if (!effectiveReferenceMask.empty())
        {
            // 适配深度图尺寸（PatchMatch 可能有上采样）
            if (effectiveReferenceMask.size() != depthMap.size())
            {
                cv::resize(effectiveReferenceMask,
                           effectiveReferenceMask,
                           depthMap.size(),
                           0,
                           0,
                           cv::INTER_NEAREST);
            }
            cv::compare(effectiveReferenceMask,
                        0,
                        effectiveReferenceMask,
                        cv::CMP_GT);

            int beforeMask = cv::countNonZero(depthMap > 0);
            depthMap.setTo(0, ~effectiveReferenceMask);
            if (!confMap.empty())
            {
                confMap.setTo(0, ~effectiveReferenceMask);
            }
            int afterMask = cv::countNonZero(depthMap > 0);

            if (afterMask < beforeMask)
            {
                LOG_DEBUG("[MVS][帧 %d][掩码] 有效区域过滤 %d->%d", refIdx, beforeMask, afterMask);
            }
        }
    }

    if (effectiveReferenceMask.empty())
    {
        effectiveReferenceMask = cv::Mat(depthMap.size(), CV_8UC1, cv::Scalar(255));
    }
    result.depthCompleteness.afterMaskValidCount = cv::countNonZero(depthMap > 0.0f);

    cv::Mat targeted_gap_recovered_mask;
    DepthGapTargetedRecoveryStats targeted_gap_stats;
    if (config.enableTargetedGapRecovery &&
        _effectiveSceneProfile == MvsSceneProfile::OrbitalObject &&
        !useRectified && srcGrays.size() >= 2)
    {
        DepthGapTargetedRecoveryOptions recovery_options;
        recovery_options.minimumCandidateConfidence = std::clamp(
            config.targetedGapRecoveryConfidence, 0.0f, 1.0f);
        recovery_options.maximumCandidatePriorRelativeDifference = std::max(
            0.0f, config.targetedGapRecoveryPriorRelativeDifference);
        recovery_options.maximumConsensusInverseDepthRelativeSpread = std::max(
            0.0f, config.targetedGapRecoveryConsensusInverseDepthSpread);
        recovery_options.maximumConsensusPriorRelativeDifference = std::max(
            recovery_options.maximumCandidatePriorRelativeDifference,
            config.targetedGapRecoveryConsensusPriorRelativeDifference);
        if (srcGrays.size() >= 4 &&
            config.targetedGapRecoveryHypothesisCount >= 2)
        {
            recovery_options.enableSurfaceAwarePrior =
                config.enableTargetedGapSurfacePrior;
            recovery_options.maximumSurfaceAnchorInverseDepthRelativeSpread =
                std::max(0.0f,
                         config.targetedGapSurfacePriorMaximumAnchorSpread);
            recovery_options.maximumSurfacePriorFitRelativeResidual =
                std::max(0.0f,
                         config.targetedGapSurfacePriorMaximumFitResidual);
            recovery_options.missingPriorRadiusRatio = std::max(
                recovery_options.missingPriorRadiusRatio,
                recovery_options.maximumConsensusPriorRelativeDifference);
        }
        recovery_options.maximumPriorDistancePixels = std::max(
            1, config.targetedGapRecoveryMaximumPriorDistancePixels);
        const DepthGapTarget recovery_target = buildDepthGapTarget(
            depthMap, effectiveReferenceMask, recovery_options);
        targeted_gap_stats.supportPixelCount = recovery_target.supportPixelCount;
        targeted_gap_stats.requestedGapPixelCount =
            recovery_target.requestedGapPixelCount;
        targeted_gap_stats.priorCoveredGapPixelCount =
            recovery_target.priorCoveredGapPixelCount;
        targeted_gap_stats.skippedReason = recovery_target.skippedReason;
        if (recovery_target.valid)
        {
            const int recovery_source_count = std::clamp(
                config.targetedGapRecoverySourceCount,
                1,
                static_cast<int>(srcGrays.size()));
            const int requested_hypothesis_count = std::clamp(
                config.targetedGapRecoveryHypothesisCount,
                1,
                recovery_source_count);
            const int hypothesis_count = recovery_source_count >= 4
                ? std::min(requested_hypothesis_count, recovery_source_count / 2)
                : 1;
            std::vector<std::vector<int>> source_groups(
                static_cast<std::size_t>(hypothesis_count));
            for (int source_ordinal = 0;
                 source_ordinal < recovery_source_count;
                 ++source_ordinal)
            {
                source_groups[static_cast<std::size_t>(
                    source_ordinal % hypothesis_count)].push_back(source_ordinal);
            }

            std::vector<cv::Mat> candidate_depths;
            std::vector<cv::Mat> candidate_confidences;
            int successful_source_count = 0;
            QStringList hypothesis_errors;
            for (int hypothesis_index = 0;
                 hypothesis_index < hypothesis_count;
                 ++hypothesis_index)
            {
                const std::vector<int> &source_group =
                    source_groups[static_cast<std::size_t>(hypothesis_index)];
                std::vector<cv::Mat> recovery_images;
                std::vector<Camera> recovery_cameras;
                std::vector<cv::Mat> recovery_source_masks;
                recovery_images.reserve(source_group.size());
                recovery_cameras.reserve(source_group.size());
                recovery_source_masks.reserve(source_group.size());
                for (const int source_ordinal : source_group)
                {
                    recovery_images.push_back(
                        srcGrays[static_cast<std::size_t>(source_ordinal)]);
                    recovery_cameras.push_back(
                        srcCams[static_cast<std::size_t>(source_ordinal)]);
                    if (!source_valid_masks.empty())
                    {
                        recovery_source_masks.push_back(
                            source_valid_masks[static_cast<std::size_t>(source_ordinal)]);
                    }
                }

                PatchMatchConfig recovery_config = pmCfg;
                recovery_config.downsampleFactor = std::max(
                    1, pyramid_result.finalLevel.downsampleFactor);
                recovery_config.numIterations = std::clamp(
                    std::max(6, pmCfg.numIterations / 2), 6, 10);
                recovery_config.patchHalf = std::max(3, pmCfg.patchHalf - 1);
                recovery_config.confidenceThresh = std::min(
                    pmCfg.confidenceThresh, 0.18f);
                recovery_config.minimumMaskedPatchSupportRatio = std::min(
                    recovery_config.minimumMaskedPatchSupportRatio, 0.25f);
                recovery_config.geomConsistency = false;

                cv::Mat candidate_depth;
                cv::Mat candidate_confidence;
                std::string recovery_error;
                const bool recovery_ok = estimatePatchMatchWithAdaptiveCuda(
                    "targeted gap PatchMatch",
                    refIdx,
                    refImg,
                    recovery_images,
                    refCam,
                    recovery_cameras,
                    zNear,
                    zFar,
                    recovery_config,
                    candidate_depth,
                    &candidate_confidence,
                    &recovery_error,
                    &recovery_target.hintDepth,
                    &recovery_target.hintRadius,
                    &recovery_target.estimationMask,
                    recovery_source_masks.empty()
                        ? nullptr : &recovery_source_masks);
                if (recovery_ok)
                {
                    candidate_depths.push_back(std::move(candidate_depth));
                    candidate_confidences.push_back(
                        std::move(candidate_confidence));
                    successful_source_count +=
                        static_cast<int>(source_group.size());
                }
                else
                {
                    hypothesis_errors.push_back(
                        QStringLiteral("group_%1:%2")
                            .arg(hypothesis_index)
                            .arg(QString::fromStdString(recovery_error)));
                }
            }
            if (!candidate_depths.empty())
            {
                targeted_gap_stats =
                    mergeMultiHypothesisTargetedDepthGapCandidates(
                    depthMap,
                    confMap,
                    candidate_depths,
                    candidate_confidences,
                    recovery_target,
                    &targeted_gap_recovered_mask,
                    recovery_options);
                targeted_gap_stats.sourceCount = successful_source_count;
                targeted_gap_stats.attemptedHypothesisCount = hypothesis_count;
                targeted_gap_stats.failedHypothesisCount = hypothesis_count -
                    static_cast<int>(candidate_depths.size());
                if (!supportCount.empty() &&
                    supportCount.size() == targeted_gap_recovered_mask.size())
                {
                    supportCount.setTo(
                        cv::Scalar(successful_source_count),
                        targeted_gap_recovered_mask);
                }
                LOG_INFO(QStringLiteral(
                             "[MVS] 帧 %1 缺口定向 PatchMatch: target=%2 "
                             "candidate=%3 consensus=%4 recovered=%5 (%6%) "
                             "sources=%7 hypotheses=%8/%9")
                             .arg(refIdx)
                             .arg(targeted_gap_stats.priorCoveredGapPixelCount)
                             .arg(targeted_gap_stats.candidatePixelCount)
                             .arg(targeted_gap_stats.consensusCandidatePixelCount)
                             .arg(targeted_gap_stats.recoveredPixelCount)
                             .arg(targeted_gap_stats.recoveryRatio * 100.0f,
                                  0, 'f', 1)
                             .arg(successful_source_count)
                             .arg(static_cast<int>(candidate_depths.size()))
                             .arg(hypothesis_count));
                if (!hypothesis_errors.isEmpty())
                {
                    targeted_gap_stats.skippedReason =
                        QStringLiteral("partial_hypothesis_failure:%1")
                            .arg(hypothesis_errors.join(QStringLiteral(";")));
                }
            }
            else
            {
                targeted_gap_stats.attempted = true;
                targeted_gap_stats.attemptedHypothesisCount = hypothesis_count;
                targeted_gap_stats.failedHypothesisCount = hypothesis_count;
                targeted_gap_stats.skippedReason = QStringLiteral(
                    "all_hypotheses_failed:%1")
                    .arg(hypothesis_errors.join(QStringLiteral(";")));
                LOG_WARN(QStringLiteral(
                             "[MVS] 帧 %1 缺口定向 PatchMatch 失败: %2")
                             .arg(refIdx)
                             .arg(hypothesis_errors.join(QStringLiteral(";"))));
            }
        }
    }
    else
    {
        targeted_gap_stats.skippedReason = !config.enableTargetedGapRecovery
            ? QStringLiteral("disabled")
            : (_effectiveSceneProfile != MvsSceneProfile::OrbitalObject
                   ? QStringLiteral("non_orbital_scene")
                   : QStringLiteral("insufficient_sources_or_rectified_pair"));
    }
    result.targetedGapRecoveryDiagnostics =
        depthGapTargetedRecoveryStatsToJson(targeted_gap_stats);

    if (!sparseSupportMask.empty())
    {
        cv::Mat supportMask = sparseSupportMask;
        if (supportMask.size() != depthMap.size())
        {
            cv::resize(supportMask, supportMask, depthMap.size(), 0, 0, cv::INTER_NEAREST);
        }

        applySparseSupportPrior(depthMap, confMap, supportMask, refIdx);
    }
    result.depthCompleteness.afterSparseSupportValidCount =
        cv::countNonZero(depthMap > 0.0f);

    const DepthFilterSettings quality_filter_settings = depthFilterSettings(
        _effectiveDepthFilterMode,
        static_cast<int>(sourceIndices.size()));
    cv::Mat missing_reason_map = initializeDepthMissingReasonMap(
        depthMap, effectiveReferenceMask);
    const cv::Mat before_output_filter = depthMap.clone();
    result.depthCompleteness.preOutputFilterValidCount =
        cv::countNonZero(depthMap > 0.0f);
    if (config.fusion.enableLocalDepthOutlierFilter)
    {
        const int previewOutliers = removeLocalDepthOutliers(
            depthMap,
            confMap,
            config.fusion.localDepthOutlierKernelSize,
            quality_filter_settings.localDepthOutlierRelThreshold,
            config.fusion.maxLocalDepthOutlierRemovalRatio,
            refIdx);
        if (previewOutliers > 0)
        {
            LOG_DEBUG("[MVS][帧 %d][后处理] 输出前局部突刺移除=%d", refIdx, previewOutliers);
        }
        result.depthCompleteness.outputFilterRemovedCount = previewOutliers;
    }
    markDepthLossReason(missing_reason_map,
                        before_output_filter,
                        depthMap,
                        DepthMissingReason::LocalDepthOutlier);
    result.depthCompleteness.postOutputFilterValidCount =
        cv::countNonZero(depthMap > 0.0f);
    if (result.depthCompleteness.preOutputFilterValidCount > 0)
    {
        result.depthCompleteness.outputFilterRetentionRatio =
            static_cast<float>(result.depthCompleteness.postOutputFilterValidCount)
            / static_cast<float>(result.depthCompleteness.preOutputFilterValidCount);
    }
    if (cancelled("深度后处理后"))
    {
        return result;
    }

    cv::Mat finalValidMask = depthMap > 0.0f;
    if (!supportCount.empty())
    {
        if (supportCount.size() != depthMap.size())
        {
            cv::resize(supportCount,
                       supportCount,
                       depthMap.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        supportCount.setTo(cv::Scalar(0), ~finalValidMask);
    }
    if (!normalMap.empty())
    {
        if (normalMap.size() != depthMap.size())
        {
            cv::resize(normalMap,
                       normalMap,
                       depthMap.size(),
                       0.0,
                       0.0,
                       cv::INTER_LINEAR);
        }
        normalMap.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), ~finalValidMask);
    }

    result.qualityMetrics = analyzeDepthMapQuality(depthMap,
                                                   confMap,
                                                   static_cast<int>(sourceIndices.size()),
                                                   zNear,
                                                   zFar);
    result.depthCompleteness.finalMetrics = analyzeDepthCompleteness(
        depthMap, effectiveReferenceMask);
    DepthFrameQualityInput quality_input;
    quality_input.sceneProfile = _effectiveSceneProfile;
    quality_input.filterMode = _effectiveDepthFilterMode;
    quality_input.sourceViewCount = static_cast<int>(sourceIndices.size());
    quality_input.validCoverage = result.qualityMetrics.validCoverage;
    quality_input.largestComponentRatio = result.qualityMetrics.largestComponentRatio;
    quality_input.meanConfidence = result.qualityMetrics.meanConfidence;
    quality_input.multiViewConsistency = std::clamp(
        result.qualityMetrics.meanConfidence * 1.15f,
        0.0f,
        1.0f);
    quality_input.depthAtSearchBoundaryRatio =
        result.qualityMetrics.depthAtSearchBoundaryRatio;
    quality_input.hasConstrainedSupportMask =
        result.maskSource != "full_image" && result.maskCoverage < 0.999f;
    quality_input.validWithinMaskRatio =
        result.depthCompleteness.finalMetrics.validInputs
        ? result.depthCompleteness.finalMetrics.validWithinMaskRatio
        : -1.0f;
    quality_input.outputFilterRetentionRatio =
        result.depthCompleteness.outputFilterRetentionRatio;
    result.qualityDecision = evaluateDepthFrame(quality_input);

    const QString quality_reasons = [&result]()
    {
        QStringList values;
        for (const std::string &reason : result.qualityDecision.reasons)
        {
            values.append(QString::fromStdString(reason));
        }
        return values.join(QStringLiteral(","));
    }();
    LOG_DEBUG(QStringLiteral(
                 "[MVS] 帧 %1 质量门: acceptance=%2 coverage=%3 largest_component=%4 "
                 "boundary=%5 confidence=%6 reasons=%7")
                 .arg(refIdx)
                 .arg(QString::fromLatin1(depthFrameAcceptanceId(
                     result.qualityDecision.acceptance)))
                 .arg(result.qualityMetrics.validCoverage, 0, 'f', 3)
                 .arg(result.qualityMetrics.largestComponentRatio, 0, 'f', 3)
                 .arg(result.qualityMetrics.depthAtSearchBoundaryRatio, 0, 'f', 3)
                 .arg(result.qualityMetrics.meanConfidence, 0, 'f', 3)
                 .arg(quality_reasons));

    timing.filterMs = elapsedMs(stageStart, Clock::now());
    timing.totalMs = elapsedMs(totalStart, Clock::now());
    LOG_INFO(QStringLiteral("[MVS] 帧 %1 耗时统计: source=%2 ms range=%3 ms hint=%4 ms rectify=%5 ms patchmatch=%6 ms filter=%7 ms total=%8 ms")
                 .arg(refIdx)
                 .arg(timing.sourceMs, 0, 'f', 1)
                 .arg(timing.rangeMs, 0, 'f', 1)
                 .arg(timing.hintMs, 0, 'f', 1)
                 .arg(timing.rectifyMs, 0, 'f', 1)
                 .arg(timing.patchmatchMs, 0, 'f', 1)
                 .arg(timing.filterMs, 0, 'f', 1)
                 .arg(timing.totalMs, 0, 'f', 1));

    result.depthMap   = QSharedPointer<cv::Mat>::create(depthMap);
    result.confidence = QSharedPointer<cv::Mat>::create(confMap);
    result.normalMap = normalMap.empty()
        ? QSharedPointer<cv::Mat>()
        : QSharedPointer<cv::Mat>::create(normalMap);
    result.supportCount = supportCount.empty()
        ? QSharedPointer<cv::Mat>()
        : QSharedPointer<cv::Mat>::create(supportCount);
    result.validMask = QSharedPointer<cv::Mat>::create(finalValidMask);
    result.supportRegionMask = QSharedPointer<cv::Mat>::create(effectiveReferenceMask);
    result.targetedGapRecoveredMask = targeted_gap_recovered_mask.empty()
        ? QSharedPointer<cv::Mat>()
        : QSharedPointer<cv::Mat>::create(targeted_gap_recovered_mask);
    result.depthProvenance = QSharedPointer<cv::Mat>::create(
        initializeDepthProvenance(depthMap, targeted_gap_recovered_mask));
    result.missingReasonMap = QSharedPointer<cv::Mat>::create(missing_reason_map);
    result.cameraModel = refCam;
    if (!depthMap.empty() && (depthMap.cols != W || depthMap.rows != H))
    {
        result.cameraModel = refCam.scaledIntrinsics(
            static_cast<double>(depthMap.cols) / std::max(1, W),
            static_cast<double>(depthMap.rows) / std::max(1, H));
    }
    result.success    = true;
    return result;
}

// =============================================================================
FusionFrameInput DepthMapGenerator::buildFusionFrame(const DepthFrameResult &res) const
{
    FusionFrameInput frame;
    if (!res.eligibleForFusion())
    {
        return frame;
    }
    frame.cameraModel = res.cameraModel.isValid()
        ? res.cameraModel
        : mvsPinholeCamera(_views[res.refViewIdx].camera);
    frame.sourceCamera = _views[res.refViewIdx].camera;
    frame.viewIndex = res.refViewIdx;
    frame.sourceImageIndices = res.sourceViewIndices;
    frame.imgW = res.depthMap ? res.depthMap->cols : 0;
    frame.imgH = res.depthMap ? res.depthMap->rows : 0;
    frame.imagePath = _views[res.refViewIdx].imagePath;

    if (!res.depthMap || res.depthMap->empty()) return frame;

    const cv::Mat &rawDepth = *res.depthMap;
    cv::Mat filteredDepth = rawDepth.clone();
    cv::Mat filteredConfidence = res.confidence ? res.confidence->clone() : cv::Mat();

    if (res.depthPostprocessApplied)
    {
        frame.depthPostprocess = res.depthPostprocess;
    }
    else
    {
        FusionConfig fusion_config = _config.fusion;
        const DepthFilterSettings filter_settings = depthFilterSettings(
            _effectiveDepthFilterMode,
            static_cast<int>(res.sourceViewIndices.size()));
        fusion_config.localDepthOutlierRelThresh =
            filter_settings.localDepthOutlierRelThreshold;
        fusion_config.minSpeckleComponentArea = filter_settings.minComponentArea;
        fusion_config.minConsistentViews = filter_settings.minConsistentViews;
        fusion_config.confidenceThresh = depthConfidenceThresholds(
            _effectiveSceneProfile,
            _effectiveDepthFilterMode,
            static_cast<int>(res.sourceViewIndices.size()),
            _config.patchMatch.confidenceThresh,
            fusion_config.confidenceThresh).fusion;
        const DepthPostProcessEvidence postprocess_evidence =
            depthPostProcessEvidence(res);
        frame.depthPostprocess = postprocessFusionDepthMap(filteredDepth,
                                                           filteredConfidence,
                                                           fusion_config,
                                                           res.refViewIdx,
                                                           static_cast<int>(_views.size()),
                                                           nullptr,
                                                           &postprocess_evidence);
    }

    frame.depthMap   = filteredDepth;
    frame.confidence = filteredConfidence;
    frame.normalMap = res.normalMap ? res.normalMap->clone() : cv::Mat();
    frame.supportCount = res.supportCount ? res.supportCount->clone() : cv::Mat();
    frame.validMask = res.validMask ? res.validMask->clone() : (filteredDepth > 0.0f);
    if (!frame.validMask.empty())
    {
        frame.validMask.setTo(cv::Scalar(0), filteredDepth <= 0.0f);
    }
    if (!frame.supportCount.empty())
    {
        frame.supportCount.setTo(cv::Scalar(0), filteredDepth <= 0.0f);
    }
    if (!frame.normalMap.empty())
    {
        frame.normalMap.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), filteredDepth <= 0.0f);
    }
    frame.confidence.release();
    return frame;
}

// =============================================================================
// 双视图深度图左右一致性检查
// 对每个深度像素：反投影到 3D → 投影到另一视图 → 比较另一视图的深度值
//
// 策略:
//   多视图 (≥3): "需要确认" — 像素必须得到至少一个其他视图的深度一致性确认
//   单源视图 (1): "仅移除矛盾" — 仅移除被其他视图明确否定的像素；
//                 对方深度为 0 或超出投影范围时，保留原像素（疑罪从无）
//   双源视图 (2): "需要确认" — 至少由一个相邻源视图在 10% 相对误差内确认；
//                 避免环拍稀疏环每帧只有两个相邻来源时保留互相冲突的完整深度。
// =============================================================================
void DepthMapGenerator::crossCheckDepthConsistency()
{
    const int NV = static_cast<int>(_views.size());
    if (NV < 2) return;
    if (_cancelled.load())
    {
        return;
    }

    // ── 先保存所有帧的原始深度图拷贝，避免顺序处理的级联清除问题 ──────
    const bool capture_adaptive_reliability =
        _config.enableAdaptiveGeometryEvidence &&
        _effectiveSceneProfile == MvsSceneProfile::OrbitalObject;
    std::vector<cv::Mat> origDepths(NV);
    std::vector<cv::Mat> origConfidences(
        capture_adaptive_reliability ? static_cast<std::size_t>(NV) : 0U);
    for (int i = 0; i < NV; ++i)
    {
        if (_cancelled.load())
        {
            return;
        }
        if (_depthFrames[i].eligibleAsConsistencySource() && _depthFrames[i].depthMap)
        {
            origDepths[i] = _depthFrames[i].depthMap->clone();
            if (capture_adaptive_reliability && _depthFrames[i].confidence &&
                !_depthFrames[i].confidence->empty())
            {
                origConfidences[static_cast<std::size_t>(i)] =
                    _depthFrames[i].confidence->clone();
            }
        }
    }

    for (int i = 0; i < NV; ++i)
    {
        if (_cancelled.load())
        {
            return;
        }
        if (!_depthFrames[i].eligibleForConsistencyCheck() ||
            !_depthFrames[i].depthMap)
        {
            continue;
        }
        cv::Mat &depthI = *_depthFrames[i].depthMap;
        const Camera camI = _depthFrames[i].cameraModel.isValid()
            ? _depthFrames[i].cameraModel
            : mvsPinholeCamera(_views[i].camera);

        // origDepths 本就是为防止顺序处理级联修改而保留的快照，
        // 同时作为本帧回退源，避免再复制一张全分辨率深度图。
        const cv::Mat &depthBackup = origDepths[static_cast<std::size_t>(i)];

        const auto frame_start = Clock::now();
        const int rowWorkers = resolvedTotalCpuThreadBudget(_config);
        const std::vector<int> consistencySources =
            consistencySourceIndicesForFrame(_depthFrames, i, NV);
        const std::vector<int> repairSources =
            _effectiveSceneProfile == MvsSceneProfile::OrbitalObject
            ? orbitalHoleRepairSourceIndices(
                  _depthFrames,
                  consistencySources,
                  i,
                  NV,
                  _config.crossViewHoleRepairSourceCount)
            : consistencySources;
        const bool fewViews = useContradictionOnlyDepthConsistency(
            static_cast<int>(consistencySources.size()));
        const float relThresh = depthConsistencyRelativeThreshold(
            _effectiveSceneProfile,
            static_cast<int>(consistencySources.size()),
            _effectiveDepthFilterMode);
        const int minimum_source_confirmations =
            minimumDepthConsistencySourceConfirmations(
                _effectiveSceneProfile,
                _effectiveDepthFilterMode,
                static_cast<int>(consistencySources.size()));
        const int beforeValid = cv::countNonZero(depthI > 0.0f);
        const CrossViewHoleRepairOptions repair_options =
            orbitalCrossViewHoleRepairOptions(_config);

        cv::Mat consistent_votes(depthI.size(), CV_16U, cv::Scalar(0));
        cv::Mat occluded_votes(depthI.size(), CV_16U, cv::Scalar(0));
        cv::Mat contradicted_votes(depthI.size(), CV_16U, cv::Scalar(0));
        cv::Mat unverifiable_votes(depthI.size(), CV_16U, cv::Scalar(0));
        cv::Mat geometry_source_mask(depthI.size(), CV_16U, cv::Scalar(0));
        cv::Mat source_inverse_depth_sum(depthI.size(), CV_32F, cv::Scalar(0.0f));
        cv::Mat source_inverse_depth_squared_sum(
            depthI.size(), CV_32F, cv::Scalar(0.0f));
        const bool generate_adaptive_evidence =
            _config.enableAdaptiveGeometryEvidence &&
            _effectiveSceneProfile == MvsSceneProfile::OrbitalObject;
        AdaptiveGeometryEvidenceAccumulatorMaps adaptive_evidence_accumulator =
            generate_adaptive_evidence
            ? makeAdaptiveGeometryEvidenceAccumulatorMaps(depthI.size())
            : AdaptiveGeometryEvidenceAccumulatorMaps{};
        std::vector<cv::Mat> projected_sources;
        if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            projected_sources.resize(repairSources.size());
        }
        std::vector<DepthConsistencySourceInput> consistency_inputs;
        consistency_inputs.reserve(consistencySources.size());

        for (int source_ordinal = 0;
             source_ordinal < static_cast<int>(repairSources.size());
             ++source_ordinal)
        {
            const float source_progress =
                (static_cast<float>(i) +
                 static_cast<float>(source_ordinal) /
                     static_cast<float>(std::max<std::size_t>(1, repairSources.size()))) /
                static_cast<float>(NV);
            emit progressChanged(
                QStringLiteral("多视一致性：帧 %1/%2，源视图 %3/%4")
                    .arg(i + 1)
                    .arg(NV)
                    .arg(source_ordinal + 1)
                    .arg(repairSources.size()),
                source_progress);
            const int j = repairSources[static_cast<std::size_t>(source_ordinal)];
            if (_cancelled.load())
            {
                return;
            }
            if (j == i)
            {
                continue;
            }
            if (origDepths[j].empty())
            {
                continue;
            }
            const cv::Mat &depthJ = origDepths[j];
            const Camera camJ = _depthFrames[j].cameraModel.isValid()
                ? _depthFrames[j].cameraModel
                : mvsPinholeCamera(_views[j].camera);

            if (std::find(
                    consistencySources.begin(), consistencySources.end(), j) !=
                consistencySources.end())
            {
                DepthConsistencySourceInput input;
                input.depth = depthJ;
                input.camera = camJ;
                if (generate_adaptive_evidence &&
                    !origConfidences[static_cast<std::size_t>(j)].empty())
                {
                    input.confidence =
                        origConfidences[static_cast<std::size_t>(j)];
                }
                input.reliabilityWeight = sourceGeometryReliabilityWeight(
                    _depthFrames[i], j);
                input.sourceOrdinal = source_ordinal;
                consistency_inputs.push_back(std::move(input));
            }
            if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                projected_sources[static_cast<std::size_t>(source_ordinal)] =
                    projectSourceDepthToReference(
                    depthJ,
                    camJ,
                    camI,
                    depthI.size(),
                    repair_options.maximumProjectionDistancePixels,
                    nullptr,
                    rowWorkers,
                    &_cancelled);
            }
        }

        accumulateDepthConsistency(
            depthI,
            camI,
            consistency_inputs,
            relThresh,
            rowWorkers,
            _cancelled,
            consistent_votes,
            occluded_votes,
            contradicted_votes,
            unverifiable_votes,
            geometry_source_mask,
            source_inverse_depth_sum,
            source_inverse_depth_squared_sum,
            generate_adaptive_evidence
                ? &adaptive_evidence_accumulator
                : nullptr);

        cv::Mat repair_mask = _depthFrames[i].supportRegionMask &&
                !_depthFrames[i].supportRegionMask->empty()
            ? *_depthFrames[i].supportRegionMask
            : cv::Mat(depthI.size(), CV_8UC1, cv::Scalar(255));
        if (repair_mask.size() != depthI.size())
        {
            cv::resize(repair_mask,
                       repair_mask,
                       depthI.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        const cv::Mat consistentMask = makeDepthConsistencyMask(
            depthI,
            static_cast<int>(consistencySources.size()),
            minimum_source_confirmations,
            consistent_votes,
            occluded_votes,
            contradicted_votes,
            rowWorkers,
            &_cancelled);

        if (_cancelled.load())
        {
            return;
        }
        emit progressChanged(
            QStringLiteral("多视一致性：帧 %1/%2，选择主深度层")
                .arg(i + 1)
                .arg(NV),
            static_cast<float>(i) / static_cast<float>(NV));

        _depthFrames[i].crossViewRepairedMask =
            QSharedPointer<cv::Mat>::create(
                depthI.size(), CV_8UC1, cv::Scalar(0));
        DominantDepthLayerSelectionStats layer_selection_stats;
        if (generate_adaptive_evidence)
        {
            // Revision 15 chooses one observable depth layer before TSDF.
            // A stable projected-source cluster may refine/switch the native
            // hypothesis or transfer measured depth into a missing pixel.
            // Ambiguous native samples remain available at reduced confidence
            // so consistency filtering cannot create an entire missing sector.
            depthBackup.copyTo(depthI);
            depthI.setTo(0.0f, repair_mask == 0);
            layer_selection_stats = selectDominantProjectedDepthLayer(
                depthI,
                repair_mask,
                projected_sources,
                consistent_votes,
                contradicted_votes,
                {},
                _depthFrames[i].confidence
                    ? _depthFrames[i].confidence.data() : nullptr,
                _depthFrames[i].crossViewRepairedMask.data(),
                &geometry_source_mask,
                &source_inverse_depth_sum,
                &source_inverse_depth_squared_sum,
                &consistent_votes,
                rowWorkers,
                &_cancelled);
        }
        else
        {
            depthI.setTo(0, consistentMask == 0);
        }
        WeakNativeDepthRetentionStats weak_native_retention;
        if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            const cv::Mat empty_confidence;
            const cv::Mat &original_confidence = _depthFrames[i].confidence
                ? *_depthFrames[i].confidence
                : empty_confidence;
            weak_native_retention = retainWeaklyVerifiedNativeDepth(
                depthBackup,
                original_confidence,
                repair_mask,
                consistent_votes,
                contradicted_votes,
                {},
                &depthI,
                _depthFrames[i].confidence
                    ? _depthFrames[i].confidence.data()
                    : nullptr);
        }
        cv::Mat anchored_interpolation_mask;
        std::string referenceImageError;
        MvsImageCache::ImageLease referenceImageLease = acquireImageFrame(
            i, &referenceImageError);
        const cv::Mat *referenceGray = referenceImageLease
            ? &referenceImageLease->preparedGray
            : nullptr;
        if (!referenceImageLease)
        {
            LOG_WARN(QStringLiteral(
                         "[MVS][帧 %1][一致性] 参考影像不可用，跳过影像引导修复：%2")
                         .arg(i)
                         .arg(QString::fromStdString(referenceImageError)));
        }
        emit progressChanged(
            QStringLiteral("多视一致性：帧 %1/%2，修复内部缺口")
                .arg(i + 1)
                .arg(NV),
            static_cast<float>(i) / static_cast<float>(NV));
        const CrossViewHoleRepairStats repair_stats =
            repairDepthHolesFromProjectedSources(
                depthI,
                repair_mask,
                projected_sources,
                repair_options,
                _depthFrames[i].confidence ? _depthFrames[i].confidence.data() : nullptr,
                &consistent_votes,
                _depthFrames[i].crossViewRepairedMask.data(),
                &geometry_source_mask,
                &source_inverse_depth_sum,
                &source_inverse_depth_squared_sum,
                &camI,
                referenceGray,
                &anchored_interpolation_mask,
                rowWorkers,
                &_cancelled);
        if (_cancelled.load())
        {
            return;
        }
        WeakNativeDepthRetentionOptions unconfirmed_backfill_options;
        unconfirmed_backfill_options.minimumConfirmationCount =
            std::numeric_limits<int>::max();
        unconfirmed_backfill_options.retainUnconfirmedWithoutContradiction = true;
        const cv::Mat empty_backfill_confidence;
        const cv::Mat &original_backfill_confidence = _depthFrames[i].confidence
            ? *_depthFrames[i].confidence
            : empty_backfill_confidence;
        const WeakNativeDepthRetentionStats unconfirmed_native_backfill =
            retainWeaklyVerifiedNativeDepth(
                depthBackup,
                original_backfill_confidence,
                repair_mask,
                consistent_votes,
                contradicted_votes,
                unconfirmed_backfill_options,
                &depthI,
                _depthFrames[i].confidence
                    ? _depthFrames[i].confidence.data()
                    : nullptr);
        _depthFrames[i].crossViewRepairDiagnostics =
            crossViewHoleRepairStatsToJson(repair_stats);
        _depthFrames[i].crossViewRepairDiagnostics.insert(
            QStringLiteral("dominant_depth_layer_selection"),
            dominantDepthLayerSelectionStatsToJson(layer_selection_stats));
        _depthFrames[i].crossViewRepairDiagnostics.insert(
            QStringLiteral("weak_native_retention"),
            QJsonObject{
                {QStringLiteral("considered_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.consideredPixelCount)},
                {QStringLiteral("retained_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.retainedPixelCount)},
                {QStringLiteral("retained_unconfirmed_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.retainedUnconfirmedPixelCount)},
                {QStringLiteral("rejected_contradiction_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.rejectedContradictionPixelCount)},
                {QStringLiteral("rejected_no_confirmation_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.rejectedNoConfirmationPixelCount)},
                {QStringLiteral("confidence_multiplier"), 0.55},
                {QStringLiteral("minimum_retained_confidence"), 0.80}});
        _depthFrames[i].crossViewRepairDiagnostics.insert(
            QStringLiteral("unconfirmed_native_backfill"),
            QJsonObject{
                {QStringLiteral("considered_pixel_count"),
                 static_cast<double>(
                     unconfirmed_native_backfill.consideredPixelCount)},
                {QStringLiteral("retained_pixel_count"),
                 static_cast<double>(
                     unconfirmed_native_backfill.retainedPixelCount)},
                {QStringLiteral("rejected_contradiction_pixel_count"),
                 static_cast<double>(
                     unconfirmed_native_backfill.rejectedContradictionPixelCount)}});
        _depthFrames[i].depthCompleteness.crossViewRepairedCount +=
            static_cast<int>(
                repair_stats.repairedPixelCount +
                layer_selection_stats.switchedNativePixelCount +
                layer_selection_stats.transferredMissingPixelCount);
        cv::Mat restoration_mask = _depthFrames[i].supportRegionMask &&
                !_depthFrames[i].supportRegionMask->empty()
            ? *_depthFrames[i].supportRegionMask
            : cv::Mat(depthI.size(), CV_8UC1, cv::Scalar(255));
        if (restoration_mask.size() != depthI.size())
        {
            cv::resize(restoration_mask,
                       restoration_mask,
                       depthI.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        const cv::Mat empty_confidence;
        const cv::Mat &restoration_confidence = _depthFrames[i].confidence
            ? *_depthFrames[i].confidence
            : empty_confidence;
        const int restored_count = restoreSmallInteriorDepthHoles(
            depthI,
            depthBackup,
            restoration_confidence,
            restoration_mask);
        _depthFrames[i].depthCompleteness.restoredFromPrefilterCount += restored_count;
        int afterValid = cv::countNonZero(depthI > 0);
        float keepRate = beforeValid > 0 ? 100.f * afterValid / beforeValid : 0.f;
        const double frame_elapsed_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - frame_start).count();
        LOG_INFO("[MVS][帧 %d][一致性] mode=%s workers=%d sources=%zu/%zu confirmations=%d "
                 "cross_view=%llu anchored=%llu two_source=%llu/%llu valid=%d->%d "
                 "retention=%.1f%% elapsed=%.1f ms",
                  i, fewViews ? "conflict_only" : "confirmed",
                  rowWorkers,
                  consistencySources.size(), repairSources.size(),
                  minimum_source_confirmations,
                  static_cast<unsigned long long>(repair_stats.repairedPixelCount),
                  static_cast<unsigned long long>(
                      repair_stats.anchoredInterpolation.interpolatedPixelCount),
                  static_cast<unsigned long long>(repair_stats.twoSourceGrownPixelCount),
                  static_cast<unsigned long long>(repair_stats.twoSourceCandidatePixelCount),
                  beforeValid, afterValid, keepRate, frame_elapsed_ms);

        // 安全回退：如果保留率过低（< 10%），回退使用原始深度图
        if (afterValid < beforeValid / 10 && beforeValid > 100) 
        {
            LOG_WARN("[MVS][帧 %d][一致性] 保留率过低 %.1f%%，回退原始深度图", i, keepRate);
            depthBackup.copyTo(depthI);
            _depthFrames[i].crossViewRepairedMask->setTo(cv::Scalar(0));
            anchored_interpolation_mask.setTo(cv::Scalar(0));
            afterValid = beforeValid;
        }
        if (!_depthFrames[i].depthProvenance ||
            _depthFrames[i].depthProvenance->empty())
        {
            _depthFrames[i].depthProvenance = QSharedPointer<cv::Mat>::create(
                initializeDepthProvenance(
                    depthBackup,
                    _depthFrames[i].targetedGapRecoveredMask
                        ? *_depthFrames[i].targetedGapRecoveredMask : cv::Mat()));
        }
        updateDepthProvenance(
            *_depthFrames[i].depthProvenance,
            depthI,
            _depthFrames[i].targetedGapRecoveredMask
                ? *_depthFrames[i].targetedGapRecoveredMask : cv::Mat(),
            *_depthFrames[i].crossViewRepairedMask,
            anchored_interpolation_mask);
        const GeometryEvidenceMaps geometry_evidence = makeGeometryEvidenceMaps(
            depthI,
            consistent_votes,
            geometry_source_mask,
            source_inverse_depth_sum,
            source_inverse_depth_squared_sum);
        _depthFrames[i].geometrySupportCount = QSharedPointer<cv::Mat>::create(
            geometry_evidence.supportCount);
        _depthFrames[i].geometrySourceMask = QSharedPointer<cv::Mat>::create(
            geometry_evidence.sourceMask);
        _depthFrames[i].inverseDepthMean = QSharedPointer<cv::Mat>::create(
            geometry_evidence.inverseDepthMean);
        _depthFrames[i].inverseDepthRelativeSpread = QSharedPointer<cv::Mat>::create(
            geometry_evidence.inverseDepthRelativeSpread);
        if (!_depthFrames[i].missingReasonMap ||
            _depthFrames[i].missingReasonMap->empty())
        {
            _depthFrames[i].missingReasonMap = QSharedPointer<cv::Mat>::create(
                initializeDepthMissingReasonMap(depthBackup, repair_mask));
        }
        markDepthLossReason(*_depthFrames[i].missingReasonMap,
                            depthBackup,
                            depthI,
                            DepthMissingReason::InsufficientGeometrySupport);
        finalizeDepthMissingReasonMap(
            *_depthFrames[i].missingReasonMap,
            depthI,
            repair_mask,
            geometry_evidence.supportCount,
            contradicted_votes);
        if (generate_adaptive_evidence)
        {
            const AdaptiveGeometryEvidenceMaps adaptive_evidence =
                makeAdaptiveGeometryEvidenceMaps(
                    depthBackup, adaptive_evidence_accumulator);
            _depthFrames[i].adaptiveGeometrySupportWeight =
                QSharedPointer<cv::Mat>::create(adaptive_evidence.supportWeight);
            _depthFrames[i].adaptiveGeometryEffectiveViewCount =
                QSharedPointer<cv::Mat>::create(adaptive_evidence.effectiveViewCount);
            _depthFrames[i].adaptiveGeometryConflictRatio =
                QSharedPointer<cv::Mat>::create(adaptive_evidence.conflictRatio);
        }
        const float consistency_keep_rate = beforeValid > 0
            ? static_cast<float>(afterValid) / static_cast<float>(beforeValid)
            : 0.0f;
        _depthFrames[i].depthCompleteness.preConsistencyValidCount = beforeValid;
        _depthFrames[i].depthCompleteness.postConsistencyValidCount = afterValid;
        _depthFrames[i].depthCompleteness.consistencyRetentionRatio =
            consistency_keep_rate;
        const DepthConsistencyVoteTotals vote_totals =
            summarizeDepthConsistencyVotes(
                consistent_votes,
                occluded_votes,
                contradicted_votes,
                unverifiable_votes,
                rowWorkers);
        _depthFrames[i].depthCompleteness.consistencyConfirmedObservationCount =
            static_cast<int>(vote_totals.consistent);
        _depthFrames[i].depthCompleteness.consistencyOccludedObservationCount =
            static_cast<int>(vote_totals.occluded);
        _depthFrames[i].depthCompleteness.consistencyContradictedObservationCount =
            static_cast<int>(vote_totals.contradicted);
        _depthFrames[i].depthCompleteness.consistencyUnverifiableObservationCount =
            static_cast<int>(vote_totals.unverifiable);
        _depthFrames[i].depthCompleteness.consistencyRejectedPixelCount =
            std::max(0, beforeValid - afterValid);
        updateDepthFrameQualityAfterConsistency(_depthFrames[i],
                                                depthI,
                                                restoration_confidence,
                                                _effectiveSceneProfile,
                                                _effectiveDepthFilterMode,
                                                consistency_keep_rate);
        emit progressChanged(
            QStringLiteral("多视一致性：已处理 %1/%2，单帧耗时 %3 秒")
                .arg(i + 1)
                .arg(NV)
                .arg(frame_elapsed_ms / 1000.0, 0, 'f', 1),
            static_cast<float>(i + 1) / static_cast<float>(NV));
    }
}

void DepthMapGenerator::recoverResidualDepthAfterConsistency()
{
    const int view_count = static_cast<int>(_views.size());
    if (!_config.enablePostConsistencyResidualReestimation ||
        _effectiveSceneProfile != MvsSceneProfile::OrbitalObject ||
        view_count < 4 || static_cast<int>(_depthFrames.size()) != view_count)
    {
        return;
    }
    if (!_imageCache)
    {
        LOG_WARN("[MVS][残余重估] 图像缓存不可用，跳过一致性后局部重估");
        return;
    }

    std::vector<cv::Mat> frozen_depths(static_cast<std::size_t>(view_count));
    for (int frame_index = 0; frame_index < view_count; ++frame_index)
    {
        if (_depthFrames[frame_index].eligibleAsConsistencySource() &&
            _depthFrames[frame_index].depthMap &&
            !_depthFrames[frame_index].depthMap->empty())
        {
            // Recovery writes only to pending results until all frame workers
            // have joined, so cv::Mat's shared immutable view is a sufficient
            // snapshot here; cloning every full-resolution frame is redundant.
            frozen_depths[static_cast<std::size_t>(frame_index)] =
                *_depthFrames[frame_index].depthMap;
        }
    }

    struct PendingResidualRecovery
    {
        cv::Mat depth;
        cv::Mat confidence;
        cv::Mat recoveredMask;
        cv::Mat geometrySupportCount;
        DepthResidualReestimationStats stats;
    };
    std::vector<PendingResidualRecovery> pending(
        static_cast<std::size_t>(view_count));
    const DepthResidualReestimationOptions base_options{
        64,
        0.001f,
        4,
        2,
        2,
        std::max(0.0f,
                 _config.postConsistencyResidualMaximumLayerSpread),
        std::clamp(_config.postConsistencyResidualMaximumPriorRadius,
                   0.005f,
                   0.25f),
        std::clamp(_config.postConsistencyResidualConfidence, 0.0f, 1.0f)};
    const int residual_frame_workers = std::clamp(
        _config.gpuFrameWorkerCount,
        1,
        std::min(2, view_count));
    const int row_workers = std::max(
        1,
        resolvedTotalCpuThreadBudget(_config) / residual_frame_workers);
    parallelForRows(view_count, residual_frame_workers, [&](int frame_index)
    {
        if (_cancelled.load())
        {
            return;
        }
        DepthFrameResult &frame = _depthFrames[frame_index];
        PendingResidualRecovery &recovery =
            pending[static_cast<std::size_t>(frame_index)];
        if (!frame.eligibleForConsistencyCheck() || !frame.depthMap ||
            frame.depthMap->empty())
        {
            recovery.stats.skippedReason = QStringLiteral("frame_unavailable");
            return;
        }
        std::string referenceImageError;
        MvsImageCache::ImageLease referenceImageLease = acquireImageFrame(
            frame_index, &referenceImageError);
        if (!referenceImageLease)
        {
            recovery.stats.skippedReason = QStringLiteral("reference_image_unavailable");
            LOG_WARN(QStringLiteral(
                         "[MVS][帧 %1][残余重估] 参考影像不可用：%2")
                         .arg(frame_index)
                         .arg(QString::fromStdString(referenceImageError)));
            return;
        }
        const Camera reference_camera = frame.cameraModel.isValid()
            ? frame.cameraModel
            : mvsPinholeCamera(_views[frame_index].camera);
        const std::vector<int> source_indices = orbitalHoleRepairSourceIndices(
            _depthFrames,
            consistencySourceIndicesForFrame(
                _depthFrames, frame_index, view_count),
            frame_index,
            view_count,
            _config.postConsistencyResidualSourceCount);
        cv::Mat support_mask = frame.supportRegionMask &&
                !frame.supportRegionMask->empty()
            ? *frame.supportRegionMask
            : cv::Mat(frame.depthMap->size(), CV_8UC1, cv::Scalar(255));
        DepthResidualReestimationPreflight preflight =
            inspectDepthResidualReestimationNeed(
                *frame.depthMap,
                support_mask,
                base_options);
        recovery.stats.supportPixelCount = preflight.supportPixelCount;
        recovery.stats.requestedResidualPixelCount =
            preflight.requestedResidualPixelCount;
        recovery.stats.skippedReason = preflight.skippedReason;
        if (!preflight.shouldProjectSources)
        {
            return;
        }
        std::vector<cv::Mat> projected_sources;
        std::vector<int> source_sector_ids;
        std::vector<cv::Mat> source_images;
        std::vector<Camera> source_cameras;
        std::vector<cv::Mat> source_masks;
        std::vector<MvsImageCache::ImageLease> sourceImageLeases;
        sourceImageLeases.reserve(source_indices.size());
        for (const int source_index : source_indices)
        {
            if (source_index < 0 || source_index >= view_count ||
                frozen_depths[static_cast<std::size_t>(source_index)].empty())
            {
                continue;
            }
            std::string sourceImageError;
            MvsImageCache::ImageLease sourceImageLease = acquireImageFrame(
                source_index, &sourceImageError);
            if (!sourceImageLease)
            {
                LOG_WARN(QStringLiteral(
                             "[MVS][帧 %1][残余重估] 源影像 %2 不可用：%3")
                             .arg(frame_index)
                             .arg(source_index)
                             .arg(QString::fromStdString(sourceImageError)));
                continue;
            }
            const Camera source_camera =
                _depthFrames[source_index].cameraModel.isValid()
                ? _depthFrames[source_index].cameraModel
                : mvsPinholeCamera(_views[source_index].camera);
            projected_sources.push_back(projectSourceDepthToReference(
                frozen_depths[static_cast<std::size_t>(source_index)],
                source_camera,
                reference_camera,
                frame.depthMap->size(),
                0.8f,
                nullptr,
                row_workers,
                &_cancelled));
            source_sector_ids.push_back(cameraBaselineSector(
                reference_camera, source_camera));
            source_images.push_back(sourceImageLease->preparedGray);
            source_cameras.push_back(source_camera);
            source_masks.push_back(
                _depthFrames[source_index].supportRegionMask &&
                    !_depthFrames[source_index].supportRegionMask->empty()
                ? *_depthFrames[source_index].supportRegionMask
                : cv::Mat(source_images.back().size(), CV_8UC1, cv::Scalar(255)));
            sourceImageLeases.push_back(std::move(sourceImageLease));
        }
        const DepthResidualReestimationTarget target =
            buildDepthResidualReestimationTarget(
                *frame.depthMap,
                support_mask,
                projected_sources,
                source_sector_ids,
                base_options,
                std::move(preflight));
        recovery.stats.supportPixelCount = target.supportPixelCount;
        recovery.stats.requestedResidualPixelCount =
            target.requestedResidualPixelCount;
        recovery.stats.layerCoveredPixelCount = target.layerCoveredPixelCount;
        recovery.stats.insufficientSourcePixelCount =
            target.insufficientSourcePixelCount;
        recovery.stats.insufficientSectorPixelCount =
            target.insufficientSectorPixelCount;
        recovery.stats.layerSpreadRejectedPixelCount =
            target.layerSpreadRejectedPixelCount;
        recovery.stats.sourceCount = static_cast<int>(source_images.size());
        recovery.stats.skippedReason = target.skippedReason;
        if (!target.valid || source_images.size() < 4)
        {
            if (recovery.stats.skippedReason.isEmpty())
            {
                recovery.stats.skippedReason =
                    QStringLiteral("insufficient_patchmatch_sources");
            }
            return;
        }

        std::vector<std::vector<int>> source_groups(2);
        for (int source_ordinal = 0;
             source_ordinal < static_cast<int>(source_images.size());
             ++source_ordinal)
        {
            source_groups[static_cast<std::size_t>(source_ordinal % 2)]
                .push_back(source_ordinal);
        }
        std::vector<cv::Mat> candidate_depths;
        std::vector<cv::Mat> candidate_confidences;
        recovery.stats.attemptedHypothesisCount = 2;
        double minimum_hint = 0.0;
        double maximum_hint = 0.0;
        cv::minMaxLoc(
            target.hintDepth,
            &minimum_hint,
            &maximum_hint,
            nullptr,
            nullptr,
            target.residualMask);
        const float z_near = std::max(
            1.0e-4f, static_cast<float>(minimum_hint * 0.90));
        const float z_far = std::max(
            z_near * 1.01f, static_cast<float>(maximum_hint * 1.10));
        for (const std::vector<int> &source_group : source_groups)
        {
            std::vector<cv::Mat> group_images;
            std::vector<Camera> group_cameras;
            std::vector<cv::Mat> group_masks;
            for (const int source_ordinal : source_group)
            {
                group_images.push_back(
                    source_images[static_cast<std::size_t>(source_ordinal)]);
                group_cameras.push_back(
                    source_cameras[static_cast<std::size_t>(source_ordinal)]);
                group_masks.push_back(
                    source_masks[static_cast<std::size_t>(source_ordinal)]);
            }
            PatchMatchConfig patch_match = patchMatchConfigForRecordedWorker(
                _config.patchMatch, frame.device);
            patch_match.numIterations = std::clamp(
                std::max(6, patch_match.numIterations / 2), 6, 10);
            patch_match.patchHalf = std::max(3, patch_match.patchHalf - 1);
            patch_match.confidenceThresh = std::min(
                patch_match.confidenceThresh, 0.18f);
            patch_match.minimumMaskedPatchSupportRatio = std::min(
                patch_match.minimumMaskedPatchSupportRatio, 0.25f);
            patch_match.geomConsistency = false;
            patch_match.cancelFlag = &_cancelled;
            cv::Mat candidate_depth;
            cv::Mat candidate_confidence;
            std::string error_message;
            if (estimatePatchMatchWithAdaptiveCuda(
                    "post-consistency residual PatchMatch",
                    frame_index,
                    referenceImageLease->preparedGray,
                    group_images,
                    reference_camera,
                    group_cameras,
                    z_near,
                    z_far,
                    patch_match,
                    candidate_depth,
                    &candidate_confidence,
                    &error_message,
                    &target.hintDepth,
                    &target.hintRadius,
                    &target.estimationMask,
                    &group_masks))
            {
                candidate_depths.push_back(std::move(candidate_depth));
                candidate_confidences.push_back(
                    std::move(candidate_confidence));
            }
            else
            {
                LOG_WARN("[MVS][帧 %d][残余重估] PatchMatch 组失败: %s",
                         frame_index,
                         error_message.c_str());
            }
        }
        recovery.stats.successfulHypothesisCount =
            static_cast<int>(candidate_depths.size());
        recovery.stats.failedHypothesisCount = 2 -
            recovery.stats.successfulHypothesisCount;
        if (candidate_depths.size() != 2)
        {
            recovery.stats.attempted = true;
            recovery.stats.skippedReason =
                QStringLiteral("incomplete_hypothesis_pair");
            return;
        }
        recovery.depth = frame.depthMap->clone();
        recovery.confidence = frame.confidence && !frame.confidence->empty()
            ? frame.confidence->clone()
            : cv::Mat(frame.depthMap->size(), CV_32FC1, cv::Scalar(0.0f));
        recovery.stats = mergeDepthResidualReestimationCandidates(
            recovery.depth,
            recovery.confidence,
            candidate_depths,
            candidate_confidences,
            target,
            projected_sources,
            source_sector_ids,
            &recovery.recoveredMask,
            base_options);
        recovery.stats.attemptedHypothesisCount = 2;
        recovery.stats.successfulHypothesisCount = 2;
        recovery.stats.sourceCount = static_cast<int>(source_images.size());
        recovery.geometrySupportCount = target.layerSourceCount;
        LOG_INFO("[MVS][帧 %d][残余重估] target=%d layer=%d candidate=%d "
                 "consensus=%d recovered=%d",
                 frame_index,
                 recovery.stats.requestedResidualPixelCount,
                 recovery.stats.layerCoveredPixelCount,
                 recovery.stats.candidatePixelCount,
                 recovery.stats.consensusCandidatePixelCount,
                 recovery.stats.recoveredPixelCount);
    });

    if (_cancelled.load())
    {
        return;
    }
    for (int frame_index = 0; frame_index < view_count; ++frame_index)
    {
        DepthFrameResult &frame = _depthFrames[frame_index];
        PendingResidualRecovery &recovery =
            pending[static_cast<std::size_t>(frame_index)];
        frame.residualReestimationDiagnostics =
            depthResidualReestimationStatsToJson(recovery.stats);
        if (recovery.stats.recoveredPixelCount <= 0 ||
            recovery.depth.empty())
        {
            continue;
        }
        *frame.depthMap = std::move(recovery.depth);
        if (!frame.confidence)
        {
            frame.confidence = QSharedPointer<cv::Mat>::create();
        }
        *frame.confidence = std::move(recovery.confidence);
        frame.residualReestimatedMask = QSharedPointer<cv::Mat>::create(
            std::move(recovery.recoveredMask));
        if (!frame.depthProvenance || frame.depthProvenance->empty())
        {
            frame.depthProvenance = QSharedPointer<cv::Mat>::create(
                initializeDepthProvenance(
                    *frame.depthMap,
                    frame.targetedGapRecoveredMask
                        ? *frame.targetedGapRecoveredMask : cv::Mat()));
        }
        updateDepthProvenance(
            *frame.depthProvenance,
            *frame.depthMap,
            frame.targetedGapRecoveredMask
                ? *frame.targetedGapRecoveredMask : cv::Mat(),
            frame.crossViewRepairedMask
                ? *frame.crossViewRepairedMask : cv::Mat(),
            cv::Mat(),
            *frame.residualReestimatedMask);
        if (frame.geometrySupportCount &&
            frame.geometrySupportCount->size() ==
                frame.residualReestimatedMask->size())
        {
            cv::Mat retained_support;
            recovery.geometrySupportCount.convertTo(
                retained_support, CV_16UC1, 1.0, 1.0);
            cv::max(retained_support, 3, retained_support);
            retained_support.copyTo(
                *frame.geometrySupportCount,
                *frame.residualReestimatedMask);
        }
        if (frame.inverseDepthMean &&
            frame.inverseDepthMean->size() == frame.depthMap->size())
        {
            cv::Mat recovered_inverse_depth;
            cv::divide(1.0f, *frame.depthMap, recovered_inverse_depth);
            recovered_inverse_depth.copyTo(
                *frame.inverseDepthMean,
                *frame.residualReestimatedMask);
        }
        if (frame.inverseDepthRelativeSpread &&
            frame.inverseDepthRelativeSpread->size() == frame.depthMap->size())
        {
            frame.inverseDepthRelativeSpread->setTo(
                cv::Scalar(
                    _config.postConsistencyResidualMaximumLayerSpread),
                *frame.residualReestimatedMask);
        }
    }
}

bool DepthMapGenerator::crossCheckDepthConsistencyStreaming()
{
    const int view_count = static_cast<int>(_views.size());
    if (view_count < 2)
    {
        return true;
    }
    if (_consistencyDepthDirectory.empty())
    {
        emit errorOccurred(QStringLiteral("流式深度一致性检查缺少缓存目录"));
        return false;
    }

    const uint64_t largest_frame_bytes = std::max<uint64_t>(
        1,
        largestDepthFrameBytes(_views) / 2ull);
    const SystemMemorySnapshot memory = querySystemMemorySnapshot();
    const uint64_t available_budget = memory.valid
        ? memory.availablePhysicalBytes / 4ull
        : largest_frame_bytes * 3ull;
    const uint64_t cache_budget = std::max<uint64_t>(
        largest_frame_bytes * 2ull,
        std::min<uint64_t>(available_budget, 2ull * kBytesPerGiB));
    const QDir storage_dir(QString::fromStdString(_consistencyDepthDirectory));

    DepthConsistencyCache cache(
        [storage_dir](int frame_index,
                      DepthConsistencyFrame &frame,
                      std::string *error_message)
        {
            const QString depth_path = storage_dir.filePath(
                QStringLiteral("depth_%1.bin").arg(frame_index));
            const xjw::common::OperationResult load_result =
                xjw::core::project::loadDepthMatStorage(depth_path, &frame.depth);
            if (!load_result.ok)
            {
                if (error_message)
                {
                    *error_message = load_result.errorMessage.toStdString();
                }
                return false;
            }
            const QString confidence_path = storage_dir.filePath(
                QStringLiteral("depth_%1_conf.bin").arg(frame_index));
            if (QFileInfo::exists(confidence_path))
            {
                const xjw::common::OperationResult confidence_result =
                    xjw::core::project::loadDepthMatStorage(confidence_path,
                                                           &frame.confidence);
                if (!confidence_result.ok)
                {
                    frame.confidence.release();
                }
            }
            frame.frameIndex = frame_index;
            return true;
        },
        static_cast<std::size_t>(cache_budget));

    struct PendingDepthReplacement
    {
        QString originalPath;
        QString filteredPath;
        QString originalConfidencePath;
        QString filteredConfidencePath;
        QString filteredGeometrySupportPath;
        QString filteredGeometrySourceMaskPath;
        QString filteredInverseDepthMeanPath;
        QString filteredInverseDepthSpreadPath;
        QString filteredAdaptiveSupportWeightPath;
        QString filteredAdaptiveEffectiveViewCountPath;
        QString filteredAdaptiveConflictRatioPath;
        int frameIndex = -1;
    };
    std::vector<PendingDepthReplacement> pending_replacements;
    pending_replacements.reserve(static_cast<std::size_t>(view_count));

    auto remove_pending_files = [&pending_replacements]()
    {
        for (const PendingDepthReplacement &replacement : pending_replacements)
        {
            QFile::remove(replacement.filteredPath);
            QFile::remove(replacement.filteredConfidencePath);
            QFile::remove(replacement.filteredGeometrySupportPath);
            QFile::remove(replacement.filteredGeometrySourceMaskPath);
            QFile::remove(replacement.filteredInverseDepthMeanPath);
            QFile::remove(replacement.filteredInverseDepthSpreadPath);
            if (!replacement.filteredAdaptiveSupportWeightPath.isEmpty())
            {
                QFile::remove(replacement.filteredAdaptiveSupportWeightPath);
                QFile::remove(replacement.filteredAdaptiveEffectiveViewCountPath);
                QFile::remove(replacement.filteredAdaptiveConflictRatioPath);
            }
        }
    };

    const int row_workers = resolvedTotalCpuThreadBudget(_config);
    int completed_frames = 0;

    for (int frame_index = 0; frame_index < view_count; ++frame_index)
    {
        if (_cancelled.load())
        {
            remove_pending_files();
            return false;
        }
        if (!_depthFrames[frame_index].eligibleForConsistencyCheck())
        {
            continue;
        }

        std::string load_error;
        const DepthConsistencyCache::FrameHandle reference =
            cache.acquire(frame_index, &load_error);
        if (!reference || reference->depth.empty())
        {
            remove_pending_files();
            const QString message = QStringLiteral(
                "流式一致性检查读取参考帧 %1 失败：%2")
                                        .arg(frame_index)
                                        .arg(QString::fromStdString(load_error));
            LOG_WARN(QStringLiteral("[MVS] %1").arg(message));
            emit errorOccurred(message);
            return false;
        }

        std::string referenceImageError;
        MvsImageCache::ImageLease referenceImageLease = acquireImageFrame(
            frame_index, &referenceImageError);
        const cv::Mat *referenceGray = referenceImageLease
            ? &referenceImageLease->preparedGray
            : nullptr;
        if (!referenceImageLease)
        {
            LOG_WARN(QStringLiteral(
                         "[MVS][帧 %1][流式一致性] 参考影像不可用，"
                         "跳过影像引导修复：%2")
                         .arg(frame_index)
                         .arg(QString::fromStdString(referenceImageError)));
        }

        cv::Mat filtered_depth = reference->depth.clone();
        cv::Mat filtered_confidence = reference->confidence.empty()
            ? cv::Mat()
            : reference->confidence.clone();
        const Camera reference_camera = _depthFrames[frame_index].cameraModel.isValid()
            ? _depthFrames[frame_index].cameraModel
            : mvsPinholeCamera(_views[frame_index].camera);
        const std::vector<int> source_indices = consistencySourceIndicesForFrame(
            _depthFrames,
            frame_index,
            view_count);
        const std::vector<int> repair_source_indices =
            _effectiveSceneProfile == MvsSceneProfile::OrbitalObject
            ? orbitalHoleRepairSourceIndices(
                  _depthFrames,
                  source_indices,
                  frame_index,
                  view_count,
                  _config.crossViewHoleRepairSourceCount)
            : source_indices;
        const float relative_threshold = depthConsistencyRelativeThreshold(
            _effectiveSceneProfile,
            static_cast<int>(source_indices.size()),
            _effectiveDepthFilterMode);
        const int minimum_source_confirmations =
            minimumDepthConsistencySourceConfirmations(
                _effectiveSceneProfile,
                _effectiveDepthFilterMode,
                static_cast<int>(source_indices.size()));
        const CrossViewHoleRepairOptions repair_options =
            orbitalCrossViewHoleRepairOptions(_config);
        cv::Mat consistent_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
        cv::Mat occluded_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
        cv::Mat contradicted_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
        cv::Mat unverifiable_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
        cv::Mat geometry_source_mask(reference->depth.size(), CV_16U, cv::Scalar(0));
        cv::Mat source_inverse_depth_sum(
            reference->depth.size(), CV_32F, cv::Scalar(0.0f));
        cv::Mat source_inverse_depth_squared_sum(
            reference->depth.size(), CV_32F, cv::Scalar(0.0f));
        const bool generate_adaptive_evidence =
            _config.enableAdaptiveGeometryEvidence &&
            _effectiveSceneProfile == MvsSceneProfile::OrbitalObject;
        AdaptiveGeometryEvidenceAccumulatorMaps adaptive_evidence_accumulator =
            generate_adaptive_evidence
            ? makeAdaptiveGeometryEvidenceAccumulatorMaps(reference->depth.size())
            : AdaptiveGeometryEvidenceAccumulatorMaps{};
        std::vector<cv::Mat> projected_sources;
        if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            projected_sources.resize(repair_source_indices.size());
        }
        std::vector<DepthConsistencyCache::FrameHandle>
            consistency_source_handles;
        std::vector<DepthConsistencySourceInput> consistency_inputs;
        consistency_source_handles.reserve(source_indices.size());
        consistency_inputs.reserve(source_indices.size());
        uint64_t consistency_batch_bytes = 0;
        const uint64_t estimated_source_bytes = std::max<uint64_t>(
            largest_frame_bytes,
            static_cast<uint64_t>(reference->byteSize()));
        const uint64_t consistency_batch_budget = std::max<uint64_t>(
            estimated_source_bytes,
            cache_budget > reference->byteSize()
                ? cache_budget - static_cast<uint64_t>(reference->byteSize())
                : estimated_source_bytes);
        auto flush_consistency_batch = [&]()
        {
            accumulateDepthConsistency(
                reference->depth,
                reference_camera,
                consistency_inputs,
                relative_threshold,
                row_workers,
                _cancelled,
                consistent_votes,
                occluded_votes,
                contradicted_votes,
                unverifiable_votes,
                geometry_source_mask,
                source_inverse_depth_sum,
                source_inverse_depth_squared_sum,
                generate_adaptive_evidence
                    ? &adaptive_evidence_accumulator
                    : nullptr);
            consistency_inputs.clear();
            consistency_source_handles.clear();
            consistency_batch_bytes = 0;
        };

        for (int source_ordinal = 0;
             source_ordinal < static_cast<int>(repair_source_indices.size());
             ++source_ordinal)
        {
            const float source_progress =
                (static_cast<float>(completed_frames) +
                 static_cast<float>(source_ordinal) /
                     static_cast<float>(std::max<std::size_t>(
                         1, repair_source_indices.size()))) /
                static_cast<float>(std::max(1, view_count));
            emit progressChanged(
                QStringLiteral("流式多视一致性：帧 %1/%2，源视图 %3/%4")
                    .arg(frame_index + 1)
                    .arg(view_count)
                    .arg(source_ordinal + 1)
                    .arg(repair_source_indices.size()),
                source_progress);
            const int source_index = repair_source_indices[
                static_cast<std::size_t>(source_ordinal)];
            if (_cancelled.load())
            {
                remove_pending_files();
                return false;
            }
            if (source_index == frame_index ||
                !_depthFrames[source_index].eligibleAsConsistencySource())
            {
                continue;
            }

            const bool is_consistency_source = std::find(
                source_indices.begin(), source_indices.end(), source_index) !=
                source_indices.end();
            if (!consistency_inputs.empty() &&
                (!is_consistency_source ||
                 consistency_batch_bytes + estimated_source_bytes >
                    consistency_batch_budget))
            {
                // Release pinned consistency handles before acquiring a
                // repair-only source, leaving one transient-frame slot in the
                // bounded cache instead of exceeding the budget by one frame.
                flush_consistency_batch();
            }

            const DepthConsistencyCache::FrameHandle source =
                cache.acquire(source_index, &load_error);
            if (!source || source->depth.empty())
            {
                LOG_WARN(QStringLiteral(
                             "[MVS] 流式一致性检查跳过无法读取的源帧 %1：%2")
                             .arg(source_index)
                             .arg(QString::fromStdString(load_error)));
                continue;
            }
            if (is_consistency_source)
            {
                const uint64_t source_bytes = static_cast<uint64_t>(
                    source->byteSize());
                if (!consistency_inputs.empty() &&
                    consistency_batch_bytes + source_bytes >
                        consistency_batch_budget)
                {
                    flush_consistency_batch();
                }
                DepthConsistencySourceInput input;
                input.depth = source->depth;
                input.camera =
                    _depthFrames[source_index].cameraModel.isValid()
                    ? _depthFrames[source_index].cameraModel
                    : mvsPinholeCamera(_views[source_index].camera);
                input.confidence = source->confidence;
                input.reliabilityWeight = sourceGeometryReliabilityWeight(
                    _depthFrames[frame_index], source_index);
                input.sourceOrdinal = source_ordinal;
                consistency_inputs.push_back(std::move(input));
                consistency_source_handles.push_back(source);
                consistency_batch_bytes += source_bytes;
            }
            if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                projected_sources[static_cast<std::size_t>(source_ordinal)] =
                    projectSourceDepthToReference(
                    source->depth,
                    _depthFrames[source_index].cameraModel.isValid()
                        ? _depthFrames[source_index].cameraModel
                        : mvsPinholeCamera(_views[source_index].camera),
                    reference_camera,
                    filtered_depth.size(),
                    repair_options.maximumProjectionDistancePixels,
                    nullptr,
                    row_workers,
                    &_cancelled);
            }
        }
        flush_consistency_batch();

        cv::Mat repair_mask = _depthFrames[frame_index].supportRegionMask &&
                !_depthFrames[frame_index].supportRegionMask->empty()
            ? *_depthFrames[frame_index].supportRegionMask
            : cv::Mat(filtered_depth.size(), CV_8UC1, cv::Scalar(255));
        if (repair_mask.size() != filtered_depth.size())
        {
            cv::resize(repair_mask,
                       repair_mask,
                       filtered_depth.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        if (!_depthFrames[frame_index].missingReasonMap ||
            _depthFrames[frame_index].missingReasonMap->empty())
        {
            const QString reason_path = storage_dir.filePath(
                QStringLiteral("depth_%1_missing_reason.png").arg(frame_index));
            cv::Mat reason_map = xjw::common::io::readImage(
                xjw::common::io::toUtf8Path(reason_path),
                cv::IMREAD_GRAYSCALE);
            if (reason_map.empty() || reason_map.type() != CV_8UC1 ||
                reason_map.size() != filtered_depth.size())
            {
                reason_map = initializeDepthMissingReasonMap(
                    reference->depth, repair_mask);
            }
            _depthFrames[frame_index].missingReasonMap =
                QSharedPointer<cv::Mat>::create(reason_map);
        }
        const cv::Mat consistent_mask = makeDepthConsistencyMask(
            filtered_depth,
            static_cast<int>(source_indices.size()),
            minimum_source_confirmations,
            consistent_votes,
            occluded_votes,
            contradicted_votes,
            row_workers,
            &_cancelled);

        if (_cancelled.load())
        {
            remove_pending_files();
            return false;
        }
        emit progressChanged(
            QStringLiteral("流式多视一致性：帧 %1/%2，选择主深度层")
                .arg(frame_index + 1)
                .arg(view_count),
            static_cast<float>(completed_frames) /
                static_cast<float>(std::max(1, view_count)));

        const int valid_before = cv::countNonZero(reference->depth > 0.0f);
        _depthFrames[frame_index].crossViewRepairedMask =
            QSharedPointer<cv::Mat>::create(
                filtered_depth.size(), CV_8UC1, cv::Scalar(0));
        DominantDepthLayerSelectionStats layer_selection_stats;
        if (generate_adaptive_evidence)
        {
            reference->depth.copyTo(filtered_depth);
            filtered_depth.setTo(0.0f, repair_mask == 0);
            layer_selection_stats = selectDominantProjectedDepthLayer(
                filtered_depth,
                repair_mask,
                projected_sources,
                consistent_votes,
                contradicted_votes,
                {},
                filtered_confidence.empty() ? nullptr : &filtered_confidence,
                _depthFrames[frame_index].crossViewRepairedMask.data(),
                &geometry_source_mask,
                &source_inverse_depth_sum,
                &source_inverse_depth_squared_sum,
                &consistent_votes,
                row_workers,
                &_cancelled);
        }
        else
        {
            filtered_depth.setTo(0.0f, consistent_mask == 0);
        }
        WeakNativeDepthRetentionStats weak_native_retention;
        if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            weak_native_retention = retainWeaklyVerifiedNativeDepth(
                reference->depth,
                reference->confidence,
                repair_mask,
                consistent_votes,
                contradicted_votes,
                {},
                &filtered_depth,
                filtered_confidence.empty() ? nullptr : &filtered_confidence);
        }
        cv::Mat anchored_interpolation_mask;
        emit progressChanged(
            QStringLiteral("流式多视一致性：帧 %1/%2，修复内部缺口")
                .arg(frame_index + 1)
                .arg(view_count),
            static_cast<float>(completed_frames) /
                static_cast<float>(std::max(1, view_count)));
        const CrossViewHoleRepairStats repair_stats =
            repairDepthHolesFromProjectedSources(
                filtered_depth,
                repair_mask,
                projected_sources,
                repair_options,
                filtered_confidence.empty() ? nullptr : &filtered_confidence,
                &consistent_votes,
                _depthFrames[frame_index].crossViewRepairedMask.data(),
                &geometry_source_mask,
                &source_inverse_depth_sum,
                &source_inverse_depth_squared_sum,
                &reference_camera,
                referenceGray,
                &anchored_interpolation_mask,
                row_workers,
                &_cancelled);
        if (_cancelled.load())
        {
            remove_pending_files();
            return false;
        }
        WeakNativeDepthRetentionOptions unconfirmed_backfill_options;
        unconfirmed_backfill_options.minimumConfirmationCount =
            std::numeric_limits<int>::max();
        unconfirmed_backfill_options.retainUnconfirmedWithoutContradiction = true;
        const WeakNativeDepthRetentionStats unconfirmed_native_backfill =
            retainWeaklyVerifiedNativeDepth(
                reference->depth,
                reference->confidence,
                repair_mask,
                consistent_votes,
                contradicted_votes,
                unconfirmed_backfill_options,
                &filtered_depth,
                filtered_confidence.empty() ? nullptr : &filtered_confidence);
        _depthFrames[frame_index].crossViewRepairDiagnostics =
            crossViewHoleRepairStatsToJson(repair_stats);
        _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
            QStringLiteral("dominant_depth_layer_selection"),
            dominantDepthLayerSelectionStatsToJson(layer_selection_stats));
        _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
            QStringLiteral("weak_native_retention"),
            QJsonObject{
                {QStringLiteral("considered_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.consideredPixelCount)},
                {QStringLiteral("retained_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.retainedPixelCount)},
                {QStringLiteral("retained_unconfirmed_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.retainedUnconfirmedPixelCount)},
                {QStringLiteral("rejected_contradiction_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.rejectedContradictionPixelCount)},
                {QStringLiteral("rejected_no_confirmation_pixel_count"),
                 static_cast<double>(
                     weak_native_retention.rejectedNoConfirmationPixelCount)},
                {QStringLiteral("confidence_multiplier"), 0.55},
                {QStringLiteral("minimum_retained_confidence"), 0.80}});
        _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
            QStringLiteral("unconfirmed_native_backfill"),
            QJsonObject{
                {QStringLiteral("considered_pixel_count"),
                 static_cast<double>(
                     unconfirmed_native_backfill.consideredPixelCount)},
                {QStringLiteral("retained_pixel_count"),
                 static_cast<double>(
                     unconfirmed_native_backfill.retainedPixelCount)},
                {QStringLiteral("rejected_contradiction_pixel_count"),
                 static_cast<double>(
                     unconfirmed_native_backfill.rejectedContradictionPixelCount)}});
        _depthFrames[frame_index].depthCompleteness.crossViewRepairedCount +=
            static_cast<int>(
                repair_stats.repairedPixelCount +
                layer_selection_stats.switchedNativePixelCount +
                layer_selection_stats.transferredMissingPixelCount);
        cv::Mat restoration_mask = _depthFrames[frame_index].supportRegionMask &&
                !_depthFrames[frame_index].supportRegionMask->empty()
            ? *_depthFrames[frame_index].supportRegionMask
            : cv::Mat(filtered_depth.size(), CV_8UC1, cv::Scalar(255));
        if (restoration_mask.size() != filtered_depth.size())
        {
            cv::resize(restoration_mask,
                       restoration_mask,
                       filtered_depth.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        const int restored_count = restoreSmallInteriorDepthHoles(
            filtered_depth,
            reference->depth,
            reference->confidence,
            restoration_mask);
        _depthFrames[frame_index].depthCompleteness.restoredFromPrefilterCount +=
            restored_count;
        DepthResidualReestimationStats residual_stats;
        cv::Mat residual_reestimated_mask;
        if (_config.enablePostConsistencyResidualReestimation &&
            _effectiveSceneProfile == MvsSceneProfile::OrbitalObject &&
            referenceImageLease)
        {
            std::vector<cv::Mat> residual_projected_sources;
            std::vector<int> residual_sector_ids;
            std::vector<cv::Mat> residual_source_images;
            std::vector<Camera> residual_source_cameras;
            std::vector<cv::Mat> residual_source_masks;
            std::vector<MvsImageCache::ImageLease> residualSourceImageLeases;
            residualSourceImageLeases.reserve(repair_source_indices.size());
            for (int source_ordinal = 0;
                 source_ordinal < static_cast<int>(projected_sources.size());
                 ++source_ordinal)
            {
                const int source_index = repair_source_indices[
                    static_cast<std::size_t>(source_ordinal)];
                const cv::Mat &projected = projected_sources[
                    static_cast<std::size_t>(source_ordinal)];
                if (projected.empty() || source_index < 0 ||
                    source_index >= view_count)
                {
                    continue;
                }
                std::string sourceImageError;
                MvsImageCache::ImageLease sourceImageLease = acquireImageFrame(
                    source_index, &sourceImageError);
                if (!sourceImageLease)
                {
                    LOG_WARN(QStringLiteral(
                                 "[MVS][帧 %1][流式残余重估] 源影像 %2 不可用：%3")
                                 .arg(frame_index)
                                 .arg(source_index)
                                 .arg(QString::fromStdString(sourceImageError)));
                    continue;
                }
                const Camera source_camera =
                    _depthFrames[source_index].cameraModel.isValid()
                    ? _depthFrames[source_index].cameraModel
                    : mvsPinholeCamera(_views[source_index].camera);
                residual_projected_sources.push_back(projected);
                residual_sector_ids.push_back(cameraBaselineSector(
                    reference_camera, source_camera));
                residual_source_images.push_back(sourceImageLease->preparedGray);
                residual_source_cameras.push_back(source_camera);
                residual_source_masks.push_back(
                    _depthFrames[source_index].supportRegionMask &&
                        !_depthFrames[source_index].supportRegionMask->empty()
                    ? *_depthFrames[source_index].supportRegionMask
                    : cv::Mat(
                          residual_source_images.back().size(),
                          CV_8UC1,
                          cv::Scalar(255)));
                residualSourceImageLeases.push_back(std::move(sourceImageLease));
            }
            DepthResidualReestimationOptions residual_options;
            residual_options.maximumLayerInverseDepthRelativeSpread =
                std::max(
                    0.0f,
                    _config.postConsistencyResidualMaximumLayerSpread);
            residual_options.maximumPriorRadiusRatio = std::clamp(
                _config.postConsistencyResidualMaximumPriorRadius,
                0.005f,
                0.25f);
            residual_options.minimumCandidateConfidence = std::clamp(
                _config.postConsistencyResidualConfidence, 0.0f, 1.0f);
            const DepthResidualReestimationTarget residual_target =
                buildDepthResidualReestimationTarget(
                    filtered_depth,
                    repair_mask,
                    residual_projected_sources,
                    residual_sector_ids,
                    residual_options);
            residual_stats.supportPixelCount =
                residual_target.supportPixelCount;
            residual_stats.requestedResidualPixelCount =
                residual_target.requestedResidualPixelCount;
            residual_stats.layerCoveredPixelCount =
                residual_target.layerCoveredPixelCount;
            residual_stats.insufficientSourcePixelCount =
                residual_target.insufficientSourcePixelCount;
            residual_stats.insufficientSectorPixelCount =
                residual_target.insufficientSectorPixelCount;
            residual_stats.layerSpreadRejectedPixelCount =
                residual_target.layerSpreadRejectedPixelCount;
            residual_stats.sourceCount = static_cast<int>(
                residual_source_images.size());
            residual_stats.skippedReason = residual_target.skippedReason;
            if (residual_target.valid && residual_source_images.size() >= 4)
            {
                std::vector<std::vector<int>> source_groups(2);
                for (int source_ordinal = 0;
                     source_ordinal <
                         static_cast<int>(residual_source_images.size());
                     ++source_ordinal)
                {
                    source_groups[static_cast<std::size_t>(
                        source_ordinal % 2)].push_back(source_ordinal);
                }
                double minimum_hint = 0.0;
                double maximum_hint = 0.0;
                cv::minMaxLoc(
                    residual_target.hintDepth,
                    &minimum_hint,
                    &maximum_hint,
                    nullptr,
                    nullptr,
                    residual_target.residualMask);
                const float z_near = std::max(
                    1.0e-4f, static_cast<float>(minimum_hint * 0.90));
                const float z_far = std::max(
                    z_near * 1.01f,
                    static_cast<float>(maximum_hint * 1.10));
                std::vector<cv::Mat> candidate_depths;
                std::vector<cv::Mat> candidate_confidences;
                residual_stats.attemptedHypothesisCount = 2;
                for (const std::vector<int> &source_group : source_groups)
                {
                    std::vector<cv::Mat> group_images;
                    std::vector<Camera> group_cameras;
                    std::vector<cv::Mat> group_masks;
                    for (const int source_ordinal : source_group)
                    {
                        group_images.push_back(residual_source_images[
                            static_cast<std::size_t>(source_ordinal)]);
                        group_cameras.push_back(residual_source_cameras[
                            static_cast<std::size_t>(source_ordinal)]);
                        group_masks.push_back(residual_source_masks[
                            static_cast<std::size_t>(source_ordinal)]);
                    }
                    PatchMatchConfig patch_match = patchMatchConfigForRecordedWorker(
                        _config.patchMatch,
                        _depthFrames[static_cast<std::size_t>(frame_index)].device);
                    patch_match.numIterations = std::clamp(
                        std::max(6, patch_match.numIterations / 2), 6, 10);
                    patch_match.patchHalf = std::max(
                        3, patch_match.patchHalf - 1);
                    patch_match.confidenceThresh = std::min(
                        patch_match.confidenceThresh, 0.18f);
                    patch_match.minimumMaskedPatchSupportRatio = std::min(
                        patch_match.minimumMaskedPatchSupportRatio, 0.25f);
                    patch_match.geomConsistency = false;
                    patch_match.cancelFlag = &_cancelled;
                    cv::Mat candidate_depth;
                    cv::Mat candidate_confidence;
                    std::string residual_error;
                    if (estimatePatchMatchWithAdaptiveCuda(
                            "streaming post-consistency residual PatchMatch",
                            frame_index,
                            referenceImageLease->preparedGray,
                            group_images,
                            reference_camera,
                            group_cameras,
                            z_near,
                            z_far,
                            patch_match,
                            candidate_depth,
                            &candidate_confidence,
                            &residual_error,
                            &residual_target.hintDepth,
                            &residual_target.hintRadius,
                            &residual_target.estimationMask,
                            &group_masks))
                    {
                        candidate_depths.push_back(
                            std::move(candidate_depth));
                        candidate_confidences.push_back(
                            std::move(candidate_confidence));
                    }
                    else
                    {
                        LOG_WARN(
                            "[MVS][帧 %d][流式残余重估] PatchMatch 组失败: %s",
                            frame_index,
                            residual_error.c_str());
                    }
                }
                residual_stats.successfulHypothesisCount =
                    static_cast<int>(candidate_depths.size());
                residual_stats.failedHypothesisCount = 2 -
                    residual_stats.successfulHypothesisCount;
                if (candidate_depths.size() == 2)
                {
                    residual_stats =
                        mergeDepthResidualReestimationCandidates(
                            filtered_depth,
                            filtered_confidence,
                            candidate_depths,
                            candidate_confidences,
                            residual_target,
                            residual_projected_sources,
                            residual_sector_ids,
                            &residual_reestimated_mask,
                            residual_options);
                    residual_stats.attemptedHypothesisCount = 2;
                    residual_stats.successfulHypothesisCount = 2;
                    residual_stats.sourceCount = static_cast<int>(
                        residual_source_images.size());
                    if (residual_stats.recoveredPixelCount > 0)
                    {
                        consistent_votes.setTo(
                            cv::Scalar(2), residual_reestimated_mask);
                        geometry_source_mask.setTo(
                            cv::Scalar(3), residual_reestimated_mask);
                        cv::Mat recovered_inverse_depth;
                        cv::divide(
                            1.0f, filtered_depth, recovered_inverse_depth);
                        cv::Mat recovered_inverse_depth_sum =
                            recovered_inverse_depth * 2.0f;
                        recovered_inverse_depth_sum.copyTo(
                            source_inverse_depth_sum,
                            residual_reestimated_mask);
                        cv::Mat recovered_inverse_depth_squared_sum =
                            recovered_inverse_depth.mul(
                                recovered_inverse_depth) * 2.0f;
                        recovered_inverse_depth_squared_sum.copyTo(
                            source_inverse_depth_squared_sum,
                            residual_reestimated_mask);
                    }
                }
                else
                {
                    residual_stats.attempted = true;
                    residual_stats.skippedReason =
                        QStringLiteral("incomplete_hypothesis_pair");
                }
            }
            _depthFrames[frame_index].residualReestimationDiagnostics =
                depthResidualReestimationStatsToJson(residual_stats);
            if (!residual_reestimated_mask.empty())
            {
                _depthFrames[frame_index].residualReestimatedMask =
                    QSharedPointer<cv::Mat>::create(
                        residual_reestimated_mask);
            }
        }
        if (!filtered_confidence.empty())
        {
            filtered_confidence.setTo(0.0f, filtered_depth <= 0.0f);
        }
        int valid_after = cv::countNonZero(filtered_depth > 0.0f);
        if (valid_before > 100 && valid_after < valid_before / 10)
        {
            reference->depth.copyTo(filtered_depth);
            if (!filtered_confidence.empty())
            {
                reference->confidence.copyTo(filtered_confidence);
            }
            _depthFrames[frame_index].crossViewRepairedMask->setTo(
                cv::Scalar(0));
            anchored_interpolation_mask.setTo(cv::Scalar(0));
            valid_after = valid_before;
        }
        if (!_depthFrames[frame_index].depthProvenance ||
            _depthFrames[frame_index].depthProvenance->empty())
        {
            _depthFrames[frame_index].depthProvenance =
                QSharedPointer<cv::Mat>::create(initializeDepthProvenance(
                    reference->depth,
                    _depthFrames[frame_index].targetedGapRecoveredMask
                        ? *_depthFrames[frame_index].targetedGapRecoveredMask
                        : cv::Mat()));
        }
        updateDepthProvenance(
            *_depthFrames[frame_index].depthProvenance,
            filtered_depth,
            _depthFrames[frame_index].targetedGapRecoveredMask
                ? *_depthFrames[frame_index].targetedGapRecoveredMask
                : cv::Mat(),
            *_depthFrames[frame_index].crossViewRepairedMask,
            anchored_interpolation_mask,
            residual_reestimated_mask);
        const float consistency_keep_rate = valid_before > 0
            ? static_cast<float>(valid_after) / static_cast<float>(valid_before)
            : 0.0f;
        _depthFrames[frame_index].depthCompleteness.preConsistencyValidCount =
            valid_before;
        _depthFrames[frame_index].depthCompleteness.postConsistencyValidCount =
            valid_after;
        _depthFrames[frame_index].depthCompleteness.consistencyRetentionRatio =
            consistency_keep_rate;
        const DepthConsistencyVoteTotals vote_totals =
            summarizeDepthConsistencyVotes(
                consistent_votes,
                occluded_votes,
                contradicted_votes,
                unverifiable_votes,
                row_workers);
        _depthFrames[frame_index].depthCompleteness.consistencyConfirmedObservationCount =
            static_cast<int>(vote_totals.consistent);
        _depthFrames[frame_index].depthCompleteness.consistencyOccludedObservationCount =
            static_cast<int>(vote_totals.occluded);
        _depthFrames[frame_index].depthCompleteness.consistencyContradictedObservationCount =
            static_cast<int>(vote_totals.contradicted);
        _depthFrames[frame_index].depthCompleteness.consistencyUnverifiableObservationCount =
            static_cast<int>(vote_totals.unverifiable);
        _depthFrames[frame_index].depthCompleteness.consistencyRejectedPixelCount =
            std::max(0, valid_before - valid_after);

        FusionConfig fusion_config = _config.fusion;
        const DepthFilterSettings filter_settings = depthFilterSettings(
            _effectiveDepthFilterMode,
            static_cast<int>(source_indices.size()));
        fusion_config.localDepthOutlierRelThresh =
            filter_settings.localDepthOutlierRelThreshold;
        fusion_config.minSpeckleComponentArea = filter_settings.minComponentArea;
        fusion_config.minConsistentViews = filter_settings.minConsistentViews;
        fusion_config.confidenceThresh = depthConfidenceThresholds(
            _effectiveSceneProfile,
            _effectiveDepthFilterMode,
            static_cast<int>(source_indices.size()),
            _config.patchMatch.confidenceThresh,
            fusion_config.confidenceThresh).fusion;
        const GeometryEvidenceMaps postprocess_input_geometry_evidence =
            makeGeometryEvidenceMaps(
                filtered_depth,
                consistent_votes,
                geometry_source_mask,
                source_inverse_depth_sum,
                source_inverse_depth_squared_sum);
        const AdaptiveGeometryEvidenceMaps adaptive_evidence =
            generate_adaptive_evidence
            ? makeAdaptiveGeometryEvidenceMaps(
                  reference->depth, adaptive_evidence_accumulator)
            : AdaptiveGeometryEvidenceMaps{};
        DepthPostProcessEvidence postprocess_evidence;
        postprocess_evidence.geometrySupportCount =
            postprocess_input_geometry_evidence.supportCount;
        postprocess_evidence.inverseDepthRelativeSpread =
            postprocess_input_geometry_evidence.inverseDepthRelativeSpread;
        postprocess_evidence.adaptiveSupportWeight = adaptive_evidence.supportWeight;
        postprocess_evidence.adaptiveEffectiveViewCount =
            adaptive_evidence.effectiveViewCount;
        postprocess_evidence.adaptiveConflictRatio = adaptive_evidence.conflictRatio;
        markDepthLossReason(*_depthFrames[frame_index].missingReasonMap,
                            reference->depth,
                            filtered_depth,
                            DepthMissingReason::InsufficientGeometrySupport);
        _depthFrames[frame_index].depthPostprocess = postprocessFusionDepthMap(
            filtered_depth,
            filtered_confidence,
            fusion_config,
            frame_index,
            view_count,
            _depthFrames[frame_index].missingReasonMap.data(),
            &postprocess_evidence);
        _depthFrames[frame_index].depthPostprocessApplied = true;
        cv::Mat final_interpolation_mask;
        const DepthAnchoredHoleInterpolationStats final_repair =
            repairPostprocessedInternalDepthHoles(
                _depthFrames[frame_index],
                filtered_depth,
                filtered_confidence,
                _effectiveSceneProfile,
                &final_interpolation_mask);
        updateDepthProvenance(
            *_depthFrames[frame_index].depthProvenance,
            filtered_depth,
            _depthFrames[frame_index].targetedGapRecoveredMask
                ? *_depthFrames[frame_index].targetedGapRecoveredMask
                : cv::Mat(),
            *_depthFrames[frame_index].crossViewRepairedMask,
            final_interpolation_mask,
            residual_reestimated_mask);
        _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
            QStringLiteral("postprocess_anchored_interpolation"),
            depthAnchoredHoleInterpolationStatsToJson(final_repair));
        _depthFrames[frame_index].depthCompleteness.crossViewRepairedCount +=
            static_cast<int>(final_repair.interpolatedPixelCount);
        if (final_repair.interpolatedPixelCount > 0)
        {
            LOG_DEBUG("[MVS][帧 %d][后处理] 锚定修复 pixels=%llu components=%llu",
                      frame_index,
                      static_cast<unsigned long long>(final_repair.interpolatedPixelCount),
                      static_cast<unsigned long long>(final_repair.acceptedComponentCount));
        }
        updateDepthCompletenessAfterPostprocess(
            _depthFrames[frame_index],
            filtered_depth,
            _depthFrames[frame_index].depthPostprocess);
        valid_after = cv::countNonZero(filtered_depth > 0.0f);
        updateDepthFrameQualityAfterConsistency(_depthFrames[frame_index],
                                                filtered_depth,
                                                filtered_confidence,
                                                _effectiveSceneProfile,
                                                _effectiveDepthFilterMode,
                                                consistency_keep_rate);

        const GeometryEvidenceMaps geometry_evidence = makeGeometryEvidenceMaps(
            filtered_depth,
            consistent_votes,
            geometry_source_mask,
            source_inverse_depth_sum,
            source_inverse_depth_squared_sum);
        finalizeDepthMissingReasonMap(
            *_depthFrames[frame_index].missingReasonMap,
            filtered_depth,
            repair_mask,
            geometry_evidence.supportCount,
            contradicted_votes);
        const cv::Mat &geometry_support = geometry_evidence.supportCount;

        const QString original_path = storage_dir.filePath(
            QStringLiteral("depth_%1.bin").arg(frame_index));
        const QString filtered_path = storage_dir.filePath(
            QStringLiteral("depth_%1_consistency.bin").arg(frame_index));
        const QString original_confidence_path = storage_dir.filePath(
            QStringLiteral("depth_%1_conf.bin").arg(frame_index));
        const QString filtered_confidence_path = storage_dir.filePath(
            QStringLiteral("depth_%1_conf_consistency.bin").arg(frame_index));
        const QString filtered_geometry_support_path = storage_dir.filePath(
            QStringLiteral("depth_%1_geometry_support_consistency.bin").arg(frame_index));
        const QString filtered_geometry_source_mask_path = storage_dir.filePath(
            QStringLiteral("depth_%1_geometry_source_mask_consistency.bin").arg(frame_index));
        const QString filtered_inverse_depth_mean_path = storage_dir.filePath(
            QStringLiteral("depth_%1_inverse_depth_mean_consistency.bin").arg(frame_index));
        const QString filtered_inverse_depth_spread_path = storage_dir.filePath(
            QStringLiteral("depth_%1_inverse_depth_spread_consistency.bin").arg(frame_index));
        const QString filtered_adaptive_support_weight_path = generate_adaptive_evidence
            ? storage_dir.filePath(QStringLiteral(
                  "depth_%1_adaptive_geometry_support_weight_consistency.bin")
                  .arg(frame_index))
            : QString();
        const QString filtered_adaptive_effective_view_count_path =
            generate_adaptive_evidence
            ? storage_dir.filePath(QStringLiteral(
                  "depth_%1_adaptive_geometry_effective_view_count_consistency.bin")
                  .arg(frame_index))
            : QString();
        const QString filtered_adaptive_conflict_ratio_path = generate_adaptive_evidence
            ? storage_dir.filePath(QStringLiteral(
                  "depth_%1_adaptive_geometry_conflict_ratio_consistency.bin")
                  .arg(frame_index))
            : QString();
        const xjw::common::OperationResult write_result =
            xjw::core::project::writeDepthMatStorage(filtered_path, filtered_depth);
        if (!write_result.ok)
        {
            remove_pending_files();
            LOG_WARN(QStringLiteral("[MVS] %1").arg(write_result.errorMessage));
            emit errorOccurred(write_result.errorMessage);
            return false;
        }
        if (!filtered_confidence.empty())
        {
            const xjw::common::OperationResult confidence_write_result =
                xjw::core::project::writeDepthMatStorage(filtered_confidence_path,
                                                         filtered_confidence);
            if (!confidence_write_result.ok)
            {
                QFile::remove(filtered_path);
                remove_pending_files();
                emit errorOccurred(confidence_write_result.errorMessage);
                return false;
            }
        }
        const xjw::common::OperationResult support_write_result =
            xjw::core::project::writeDepthMatStorage(
                filtered_geometry_support_path, geometry_support);
        if (!support_write_result.ok)
        {
            QFile::remove(filtered_path);
            QFile::remove(filtered_confidence_path);
            remove_pending_files();
            emit errorOccurred(support_write_result.errorMessage);
            return false;
        }
        const xjw::common::OperationResult source_mask_write_result =
            xjw::core::project::writeDepthMatStorage(
                filtered_geometry_source_mask_path, geometry_evidence.sourceMask);
        const xjw::common::OperationResult inverse_mean_write_result =
            xjw::core::project::writeDepthMatStorage(
                filtered_inverse_depth_mean_path, geometry_evidence.inverseDepthMean);
        const xjw::common::OperationResult inverse_spread_write_result =
            xjw::core::project::writeDepthMatStorage(
                filtered_inverse_depth_spread_path,
                geometry_evidence.inverseDepthRelativeSpread);
        if (!source_mask_write_result.ok || !inverse_mean_write_result.ok ||
            !inverse_spread_write_result.ok)
        {
            QFile::remove(filtered_path);
            QFile::remove(filtered_confidence_path);
            QFile::remove(filtered_geometry_support_path);
            QFile::remove(filtered_geometry_source_mask_path);
            QFile::remove(filtered_inverse_depth_mean_path);
            QFile::remove(filtered_inverse_depth_spread_path);
            remove_pending_files();
            const QString message = !source_mask_write_result.ok
                ? source_mask_write_result.errorMessage
                : (!inverse_mean_write_result.ok
                       ? inverse_mean_write_result.errorMessage
                       : inverse_spread_write_result.errorMessage);
            emit errorOccurred(message);
            return false;
        }
        if (generate_adaptive_evidence)
        {
            const xjw::common::OperationResult support_weight_result =
                xjw::core::project::writeDepthMatStorage(
                    filtered_adaptive_support_weight_path,
                    adaptive_evidence.supportWeight);
            const xjw::common::OperationResult effective_view_result =
                xjw::core::project::writeDepthMatStorage(
                    filtered_adaptive_effective_view_count_path,
                    adaptive_evidence.effectiveViewCount);
            const xjw::common::OperationResult conflict_ratio_result =
                xjw::core::project::writeDepthMatStorage(
                    filtered_adaptive_conflict_ratio_path,
                    adaptive_evidence.conflictRatio);
            if (!support_weight_result.ok || !effective_view_result.ok ||
                !conflict_ratio_result.ok)
            {
                QFile::remove(filtered_path);
                QFile::remove(filtered_confidence_path);
                QFile::remove(filtered_geometry_support_path);
                QFile::remove(filtered_geometry_source_mask_path);
                QFile::remove(filtered_inverse_depth_mean_path);
                QFile::remove(filtered_inverse_depth_spread_path);
                QFile::remove(filtered_adaptive_support_weight_path);
                QFile::remove(filtered_adaptive_effective_view_count_path);
                QFile::remove(filtered_adaptive_conflict_ratio_path);
                remove_pending_files();
                const QString message = !support_weight_result.ok
                    ? support_weight_result.errorMessage
                    : (!effective_view_result.ok
                           ? effective_view_result.errorMessage
                           : conflict_ratio_result.errorMessage);
                emit errorOccurred(message);
                return false;
            }
        }
        pending_replacements.push_back({original_path,
                                        filtered_path,
                                        original_confidence_path,
                                        filtered_confidence.empty()
                                            ? QString()
                                            : filtered_confidence_path,
                                        filtered_geometry_support_path,
                                        filtered_geometry_source_mask_path,
                                        filtered_inverse_depth_mean_path,
                                        filtered_inverse_depth_spread_path,
                                        filtered_adaptive_support_weight_path,
                                        filtered_adaptive_effective_view_count_path,
                                        filtered_adaptive_conflict_ratio_path,
                                        frame_index});

        ++completed_frames;
        const float ratio = static_cast<float>(completed_frames) /
                            static_cast<float>(std::max(1, view_count));
        emit progressChanged(
            QStringLiteral("多视一致性：已处理 %1/%2").arg(completed_frames).arg(view_count),
            ratio);
        LOG_INFO(QStringLiteral(
                     "[MVS] 流式一致性 frame=%1 workers=%2 valid=%3->%4 sources=%5 "
                     "twoSource=%6/%7 cache=%8/%9 MiB")
                     .arg(frame_index)
                     .arg(row_workers)
                     .arg(valid_before)
                     .arg(valid_after)
                     .arg(source_indices.size())
                     .arg(repair_stats.twoSourceGrownPixelCount)
                     .arg(repair_stats.twoSourceCandidatePixelCount)
                     .arg(cache.currentBytes() / (1024 * 1024))
                     .arg(cache.memoryBudgetBytes() / (1024 * 1024)));
    }

    for (const PendingDepthReplacement &replacement : pending_replacements)
    {
        const QString backup_path = replacement.originalPath + QStringLiteral(".original.bin");
        QFile::remove(backup_path);

        if (!replacement.filteredConfidencePath.isEmpty())
        {
            const QString confidence_backup =
                replacement.originalConfidencePath + QStringLiteral(".original.bin");
            QFile::remove(confidence_backup);
            if (QFileInfo::exists(replacement.originalConfidencePath) &&
                !QFile::rename(replacement.originalConfidencePath, confidence_backup))
            {
                remove_pending_files();
                emit errorOccurred(QStringLiteral("无法备份原始置信度缓存：%1")
                                       .arg(replacement.originalConfidencePath));
                return false;
            }
            if (!QFile::rename(replacement.filteredConfidencePath,
                               replacement.originalConfidencePath))
            {
                if (QFileInfo::exists(confidence_backup))
                {
                    QFile::rename(confidence_backup,
                                  replacement.originalConfidencePath);
                }
                remove_pending_files();
                emit errorOccurred(QStringLiteral("无法提交一致性过滤置信度缓存：%1")
                                       .arg(replacement.originalConfidencePath));
                return false;
            }
            QFile::remove(confidence_backup);
        }
        if (!QFile::rename(replacement.originalPath, backup_path))
        {
            remove_pending_files();
            emit errorOccurred(QStringLiteral("无法备份原始深度缓存：%1")
                                   .arg(replacement.originalPath));
            return false;
        }
        if (!QFile::rename(replacement.filteredPath, replacement.originalPath))
        {
            QFile::rename(backup_path, replacement.originalPath);
            remove_pending_files();
            emit errorOccurred(QStringLiteral("无法提交一致性过滤深度缓存：%1")
                                   .arg(replacement.originalPath));
            return false;
        }
        QFile::remove(backup_path);

        cv::Mat filtered_depth;
        const xjw::common::OperationResult load_result =
            xjw::core::project::loadDepthMatStorage(replacement.originalPath,
                                                    &filtered_depth);
        if (load_result.ok && !filtered_depth.empty())
        {
            cv::Mat filtered_confidence;
            if (QFileInfo::exists(replacement.originalConfidencePath))
            {
                (void)xjw::core::project::loadDepthMatStorage(
                    replacement.originalConfidencePath,
                    &filtered_confidence);
            }
            DepthFrameResult artifact_result = _depthFrames[replacement.frameIndex];
            artifact_result.depthMap = QSharedPointer<cv::Mat>::create(filtered_depth);
            artifact_result.confidence = QSharedPointer<cv::Mat>::create(filtered_confidence);
            const QString targeted_recovered_path = storage_dir.filePath(
                QStringLiteral("depth_%1_targeted_gap_recovered_mask.png")
                    .arg(replacement.frameIndex));
            cv::Mat targeted_recovered = xjw::common::io::readImage(
                xjw::common::io::toUtf8Path(targeted_recovered_path),
                cv::IMREAD_GRAYSCALE);
            if (!targeted_recovered.empty() &&
                targeted_recovered.type() == CV_8UC1 &&
                targeted_recovered.size() == filtered_depth.size())
            {
                artifact_result.targetedGapRecoveredMask =
                    QSharedPointer<cv::Mat>::create(targeted_recovered);
            }
            const QString provenance_path = storage_dir.filePath(
                QStringLiteral("depth_%1_provenance.png")
                    .arg(replacement.frameIndex));
            cv::Mat provenance = xjw::common::io::readImage(
                xjw::common::io::toUtf8Path(provenance_path),
                cv::IMREAD_GRAYSCALE);
            if (!provenance.empty() && provenance.type() == CV_8UC1 &&
                provenance.size() == filtered_depth.size())
            {
                artifact_result.depthProvenance =
                    QSharedPointer<cv::Mat>::create(provenance);
            }
            cv::Mat geometry_support;
            const xjw::common::OperationResult geometry_support_result =
                xjw::core::project::loadDepthMatStorage(
                    replacement.filteredGeometrySupportPath, &geometry_support);
            if (!geometry_support_result.ok || geometry_support.empty())
            {
                remove_pending_files();
                emit errorOccurred(geometry_support_result.errorMessage);
                return false;
            }
            artifact_result.geometrySupportCount =
                QSharedPointer<cv::Mat>::create(geometry_support);
            cv::Mat geometry_source_mask;
            cv::Mat inverse_depth_mean;
            cv::Mat inverse_depth_spread;
            const xjw::common::OperationResult source_mask_result =
                xjw::core::project::loadDepthMatStorage(
                    replacement.filteredGeometrySourceMaskPath,
                    &geometry_source_mask);
            const xjw::common::OperationResult inverse_mean_result =
                xjw::core::project::loadDepthMatStorage(
                    replacement.filteredInverseDepthMeanPath,
                    &inverse_depth_mean);
            const xjw::common::OperationResult inverse_spread_result =
                xjw::core::project::loadDepthMatStorage(
                    replacement.filteredInverseDepthSpreadPath,
                    &inverse_depth_spread);
            if (!source_mask_result.ok || !inverse_mean_result.ok ||
                !inverse_spread_result.ok)
            {
                remove_pending_files();
                const QString message = !source_mask_result.ok
                    ? source_mask_result.errorMessage
                    : (!inverse_mean_result.ok
                           ? inverse_mean_result.errorMessage
                           : inverse_spread_result.errorMessage);
                emit errorOccurred(message);
                return false;
            }
            artifact_result.geometrySourceMask =
                QSharedPointer<cv::Mat>::create(geometry_source_mask);
            artifact_result.inverseDepthMean =
                QSharedPointer<cv::Mat>::create(inverse_depth_mean);
            artifact_result.inverseDepthRelativeSpread =
                QSharedPointer<cv::Mat>::create(inverse_depth_spread);
            if (!replacement.filteredAdaptiveSupportWeightPath.isEmpty())
            {
                cv::Mat adaptive_support_weight;
                cv::Mat adaptive_effective_view_count;
                cv::Mat adaptive_conflict_ratio;
                const xjw::common::OperationResult adaptive_support_result =
                    xjw::core::project::loadDepthMatStorage(
                        replacement.filteredAdaptiveSupportWeightPath,
                        &adaptive_support_weight);
                const xjw::common::OperationResult adaptive_view_result =
                    xjw::core::project::loadDepthMatStorage(
                        replacement.filteredAdaptiveEffectiveViewCountPath,
                        &adaptive_effective_view_count);
                const xjw::common::OperationResult adaptive_conflict_result =
                    xjw::core::project::loadDepthMatStorage(
                        replacement.filteredAdaptiveConflictRatioPath,
                        &adaptive_conflict_ratio);
                if (!adaptive_support_result.ok || !adaptive_view_result.ok ||
                    !adaptive_conflict_result.ok)
                {
                    remove_pending_files();
                    const QString message = !adaptive_support_result.ok
                        ? adaptive_support_result.errorMessage
                        : (!adaptive_view_result.ok
                               ? adaptive_view_result.errorMessage
                               : adaptive_conflict_result.errorMessage);
                    emit errorOccurred(message);
                    return false;
                }
                artifact_result.adaptiveGeometrySupportWeight =
                    QSharedPointer<cv::Mat>::create(adaptive_support_weight);
                artifact_result.adaptiveGeometryEffectiveViewCount =
                    QSharedPointer<cv::Mat>::create(adaptive_effective_view_count);
                artifact_result.adaptiveGeometryConflictRatio =
                    QSharedPointer<cv::Mat>::create(adaptive_conflict_ratio);
            }
            if (!saveDepthFrameArtifacts(replacement.frameIndex,
                                         artifact_result,
                                         QStringLiteral("一致性过滤后")))
            {
                return false;
            }
            QFile::remove(replacement.filteredGeometrySupportPath);
            QFile::remove(replacement.filteredGeometrySourceMaskPath);
            QFile::remove(replacement.filteredInverseDepthMeanPath);
            QFile::remove(replacement.filteredInverseDepthSpreadPath);
            QFile::remove(replacement.filteredAdaptiveSupportWeightPath);
            QFile::remove(replacement.filteredAdaptiveEffectiveViewCountPath);
            QFile::remove(replacement.filteredAdaptiveConflictRatioPath);
        }
    }
    LOG_INFO(QStringLiteral(
                 "[MVS] 流式一致性检查完成：frames=%1 cachePeak=%2 MiB budget=%3 MiB")
                 .arg(pending_replacements.size())
                 .arg(cache.peakBytes() / (1024 * 1024))
                 .arg(cache.memoryBudgetBytes() / (1024 * 1024)));
    return true;
}

bool DepthMapGenerator::saveDepthFrameArtifacts(int frameIndex,
                                                const DepthFrameResult &result,
                                                const QString &stageLabel)
{
    if (frameIndex < 0 ||
        frameIndex >= static_cast<int>(_views.size()) ||
        !result.success ||
        !result.depthMap ||
        result.depthMap->empty())
    {
        return true;
    }

    const bool savePreviewPng = !_outputDir.empty();
    const bool has_raw_directory = !_config.intermediateDir.empty() ||
                                   !_outputDir.empty() ||
                                   _streamConsistencyStorageEnabled;
    const bool saveRawDepth =
        ((_config.saveIntermediateDepthMaps || _config.saveIntermediatePyramidLevels) &&
         has_raw_directory) ||
        _streamConsistencyStorageEnabled;
    if (!savePreviewPng && !saveRawDepth)
    {
        return true;
    }

    std::string saveErr;
    const auto saveStart = Clock::now();
    const std::string pngPath = _outputDir + "/depth_" + std::to_string(frameIndex) + ".png";
    const std::string raw_directory = _streamConsistencyStorageEnabled
        ? _consistencyDepthDirectory
        : (!_config.intermediateDir.empty() ? _config.intermediateDir : _outputDir);
    const std::string rawDepthPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + ".bin";
    const std::string rawConfidencePath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_conf.bin";
    const std::string rawGeometrySupportPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_geometry_support.bin";
    const std::string rawGeometrySourceMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_geometry_source_mask.bin";
    const std::string rawInverseDepthMeanPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_inverse_depth_mean.bin";
    const std::string rawInverseDepthSpreadPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_inverse_depth_spread.bin";
    const std::string rawAdaptiveGeometrySupportWeightPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_adaptive_geometry_support_weight.bin";
    const std::string rawAdaptiveGeometryEffectiveViewCountPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_adaptive_geometry_effective_view_count.bin";
    const std::string rawAdaptiveGeometryConflictRatioPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_adaptive_geometry_conflict_ratio.bin";
    const std::string crossViewRepairedMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_cross_view_repaired_mask.png";
    const std::string targetedGapRecoveredMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_targeted_gap_recovered_mask.png";
    const std::string residualReestimatedMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_residual_reestimated_mask.png";
    const std::string depthProvenancePath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_provenance.png";
    const std::string validMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_mask.png";
    const std::string supportMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_support_mask.png";
    const std::string missingReasonPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_missing_reason.png";
    const std::string missingReasonPreviewPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_missing_reason_preview.png";
    bool previewSaved = !savePreviewPng;
    double previewMs = 0.0;
    if (savePreviewPng)
    {
        const auto previewStart = Clock::now();
        if (!saveDepthPreviewPng(pngPath, *result.depthMap, &saveErr))
        {
            LOG_WARN(QStringLiteral("[MVS] 保存%1深度预览失败: %2")
                         .arg(stageLabel, QString::fromStdString(saveErr)));
            emit errorOccurred(QString::fromStdString(saveErr));
            markManifestFrameFailed(frameIndex, QString::fromStdString(saveErr));
            return false;
        }

        previewMs = elapsedMs(previewStart, Clock::now());
        previewSaved = true;
        LOG_DEBUG(QStringLiteral("[MVS] 帧 %1 %2深度预览已保存: %3 (%4x%5)")
                     .arg(frameIndex)
                     .arg(stageLabel)
                     .arg(QString::fromStdString(pngPath))
                     .arg(result.depthMap->cols)
                     .arg(result.depthMap->rows));
    }

    bool rawSaved = true;
    double rawMs = 0.0;
    if (saveRawDepth)
    {
        const auto rawStart = Clock::now();
        if (!writeFastDepthMatStorage(rawDepthPath, *result.depthMap, &saveErr))
        {
            rawSaved = false;
            LOG_WARN(QStringLiteral("[MVS] 保存%1原始深度失败: %2")
                             .arg(stageLabel, QString::fromStdString(saveErr)));
            emit errorOccurred(QString::fromStdString(saveErr));
            markManifestFrameFailed(frameIndex, QString::fromStdString(saveErr));
        }
        rawMs = elapsedMs(rawStart, Clock::now());
    }

    double confidenceMs = 0.0;
    bool confidenceSaved = false;
    if (saveRawDepth && result.confidence && !result.confidence->empty())
    {
        const auto confidenceStart = Clock::now();
        if (!writeFastDepthMatStorage(rawConfidencePath, *result.confidence, &saveErr))
        {
            LOG_WARN(QStringLiteral("[MVS] 保存%1置信图失败: %2")
                         .arg(stageLabel, QString::fromStdString(saveErr)));
        }
        else
        {
            confidenceSaved = true;
        }
        confidenceMs = elapsedMs(confidenceStart, Clock::now());
    }

    bool maskSaved = false;
    bool supportMaskSaved = false;
    cv::Mat supportMask;
    double maskMs = 0.0;
    if (saveRawDepth)
    {
        const auto maskStart = Clock::now();
        cv::Mat validMask = (*result.depthMap > 0.0f);
        if (!validMask.empty())
        {
            maskSaved = xjw::common::io::writeImage(validMaskPath, validMask);
            if (!maskSaved)
            {
                LOG_WARN(QStringLiteral("[MVS] 保存%1有效掩码失败: %2")
                             .arg(stageLabel, QString::fromStdString(validMaskPath)));
            }
        }
        if (!result.supportRegionMask || result.supportRegionMask->empty())
        {
            supportMask = cv::Mat(result.depthMap->size(), CV_8UC1, cv::Scalar(255));
        }
        else
        {
            if (result.supportRegionMask->size() == result.depthMap->size())
            {
                supportMask = result.supportRegionMask->clone();
            }
            else
            {
                cv::resize(*result.supportRegionMask,
                           supportMask,
                           result.depthMap->size(),
                           0.0,
                           0.0,
                           cv::INTER_NEAREST);
            }
            if (supportMask.type() != CV_8UC1)
            {
                supportMask.convertTo(supportMask, CV_8UC1);
            }
            cv::threshold(supportMask, supportMask, 0.0, 255.0, cv::THRESH_BINARY);
        }
        supportMaskSaved = xjw::common::io::writeImage(supportMaskPath, supportMask);
        if (!supportMaskSaved)
        {
            LOG_WARN(QStringLiteral("[MVS] 保存%1支持掩码失败: %2")
                         .arg(stageLabel, QString::fromStdString(supportMaskPath)));
        }
        maskMs = elapsedMs(maskStart, Clock::now());
    }

    bool missingReasonSaved = false;
    bool missingReasonPreviewSaved = false;
    QJsonObject missingReasonSummaryJson;
    if (saveRawDepth && !supportMask.empty())
    {
        cv::Mat missing_reason = result.missingReasonMap &&
                !result.missingReasonMap->empty()
            ? result.missingReasonMap->clone()
            : initializeDepthMissingReasonMap(*result.depthMap, supportMask);
        finalizeDepthMissingReasonMap(
            missing_reason,
            *result.depthMap,
            supportMask,
            result.geometrySupportCount && !result.geometrySupportCount->empty()
                ? *result.geometrySupportCount
                : cv::Mat());
        const DepthMissingReasonSummary missing_summary =
            summarizeDepthMissingReasons(missing_reason, supportMask);
        missingReasonSummaryJson = depthMissingReasonSummaryToJson(
            missing_summary);
        missingReasonSaved = xjw::common::io::writeImage(
            missingReasonPath, missing_reason);
        const cv::Mat missing_preview = makeDepthMissingReasonPreview(
            missing_reason);
        missingReasonPreviewSaved = !missing_preview.empty() &&
            xjw::common::io::writeImage(
                missingReasonPreviewPath, missing_preview);
        if (!missingReasonSaved || !missingReasonPreviewSaved)
        {
            LOG_WARN(QStringLiteral(
                         "[MVS] 保存%1深度缺失原因诊断失败: frame=%2")
                         .arg(stageLabel)
                         .arg(frameIndex));
        }
    }

    bool geometrySupportSaved = false;
    if (saveRawDepth && result.geometrySupportCount &&
        !result.geometrySupportCount->empty())
    {
        cv::Mat geometry_support = result.geometrySupportCount->clone();
        if (geometry_support.type() == CV_16UC1 &&
            geometry_support.size() == result.depthMap->size())
        {
            geometry_support.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
            geometrySupportSaved = writeFastDepthMatStorage(
                rawGeometrySupportPath, geometry_support, &saveErr);
        }
        if (!geometrySupportSaved)
        {
            LOG_WARN(QStringLiteral("[MVS] 保存%1跨视几何支持图失败: %2")
                         .arg(stageLabel, QString::fromStdString(saveErr)));
        }
    }

    auto save_geometry_evidence = [&](const QSharedPointer<cv::Mat> &evidence,
                                      int expected_type,
                                      const std::string &path,
                                      const QString &label,
                                      bool mask_invalid_depth)
    {
        if (!saveRawDepth || !evidence || evidence->empty())
        {
            return false;
        }
        cv::Mat stored = evidence->clone();
        if (stored.type() != expected_type || stored.size() != result.depthMap->size())
        {
            LOG_WARN(QStringLiteral("[MVS] 跳过帧 %1 %2：类型或尺寸不匹配")
                         .arg(frameIndex)
                         .arg(label));
            return false;
        }
        if (mask_invalid_depth)
        {
            stored.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
        }
        if (!writeFastDepthMatStorage(path, stored, &saveErr))
        {
            LOG_WARN(QStringLiteral("[MVS] 保存帧 %1 %2失败: %3")
                         .arg(frameIndex)
                         .arg(label, QString::fromStdString(saveErr)));
            return false;
        }
        return true;
    };
    const bool geometrySourceMaskSaved = save_geometry_evidence(
        result.geometrySourceMask,
        CV_16UC1,
        rawGeometrySourceMaskPath,
        QStringLiteral("跨视来源掩码"),
        true);
    const bool inverseDepthMeanSaved = save_geometry_evidence(
        result.inverseDepthMean,
        CV_32FC1,
        rawInverseDepthMeanPath,
        QStringLiteral("逆深度均值图"),
        true);
    const bool inverseDepthSpreadSaved = save_geometry_evidence(
        result.inverseDepthRelativeSpread,
        CV_32FC1,
        rawInverseDepthSpreadPath,
        QStringLiteral("逆深度离散度图"),
        true);
    const bool adaptiveGeometrySupportWeightSaved = save_geometry_evidence(
        result.adaptiveGeometrySupportWeight,
        CV_32FC1,
        rawAdaptiveGeometrySupportWeightPath,
        QStringLiteral("连续几何支持权重"),
        false);
    const bool adaptiveGeometryEffectiveViewCountSaved = save_geometry_evidence(
        result.adaptiveGeometryEffectiveViewCount,
        CV_32FC1,
        rawAdaptiveGeometryEffectiveViewCountPath,
        QStringLiteral("连续几何有效视图数"),
        false);
    const bool adaptiveGeometryConflictRatioSaved = save_geometry_evidence(
        result.adaptiveGeometryConflictRatio,
        CV_32FC1,
        rawAdaptiveGeometryConflictRatioPath,
        QStringLiteral("连续几何冲突比例"),
        false);
    // 连续几何证据由所有原始深度完成后的跨视一致性阶段生成。这里保存的是
    // 单帧初始产物，此时证据图尚不存在，不能据此把有效深度帧标记为失败。
    // 一致性阶段会原子写入三张 revision 14 证据图，并在任一写入失败时终止。
    bool crossViewRepairedMaskSaved = false;
    if (saveRawDepth && result.crossViewRepairedMask &&
        !result.crossViewRepairedMask->empty())
    {
        cv::Mat repaired_mask = result.crossViewRepairedMask->clone();
        if (repaired_mask.size() == result.depthMap->size())
        {
            if (repaired_mask.type() != CV_8UC1)
            {
                repaired_mask.convertTo(repaired_mask, CV_8UC1);
            }
            cv::threshold(repaired_mask,
                          repaired_mask,
                          0.0,
                          255.0,
                          cv::THRESH_BINARY);
            repaired_mask.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
            crossViewRepairedMaskSaved = xjw::common::io::writeImage(
                crossViewRepairedMaskPath, repaired_mask);
        }
    }
    bool targetedGapRecoveredMaskSaved = false;
    if (saveRawDepth && result.targetedGapRecoveredMask &&
        !result.targetedGapRecoveredMask->empty())
    {
        cv::Mat recovered_mask = result.targetedGapRecoveredMask->clone();
        if (recovered_mask.size() == result.depthMap->size())
        {
            if (recovered_mask.type() != CV_8UC1)
            {
                recovered_mask.convertTo(recovered_mask, CV_8UC1);
            }
            cv::threshold(recovered_mask,
                          recovered_mask,
                          0.0,
                          255.0,
                          cv::THRESH_BINARY);
            recovered_mask.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
            targetedGapRecoveredMaskSaved = xjw::common::io::writeImage(
                targetedGapRecoveredMaskPath, recovered_mask);
        }
    }
    bool residualReestimatedMaskSaved = false;
    if (saveRawDepth && result.residualReestimatedMask &&
        !result.residualReestimatedMask->empty())
    {
        cv::Mat recovered_mask = result.residualReestimatedMask->clone();
        if (recovered_mask.size() == result.depthMap->size())
        {
            if (recovered_mask.type() != CV_8UC1)
            {
                recovered_mask.convertTo(recovered_mask, CV_8UC1);
            }
            cv::threshold(recovered_mask,
                          recovered_mask,
                          0.0,
                          255.0,
                          cv::THRESH_BINARY);
            recovered_mask.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
            residualReestimatedMaskSaved = xjw::common::io::writeImage(
                residualReestimatedMaskPath, recovered_mask);
        }
    }
    bool depthProvenanceSaved = false;
    QJsonObject depthProvenanceSummaryJson;
    if (saveRawDepth)
    {
        cv::Mat provenance = result.depthProvenance &&
                !result.depthProvenance->empty()
            ? result.depthProvenance->clone()
            : initializeDepthProvenance(
                  *result.depthMap,
                  result.targetedGapRecoveredMask
                      ? *result.targetedGapRecoveredMask : cv::Mat());
        updateDepthProvenance(
            provenance,
            *result.depthMap,
            result.targetedGapRecoveredMask
                ? *result.targetedGapRecoveredMask : cv::Mat(),
            result.crossViewRepairedMask
                ? *result.crossViewRepairedMask : cv::Mat(),
            cv::Mat(),
            result.residualReestimatedMask
                ? *result.residualReestimatedMask : cv::Mat());
        depthProvenanceSummaryJson = depthProvenanceSummaryToJson(
            summarizeDepthProvenance(provenance, *result.depthMap));
        depthProvenanceSaved = xjw::common::io::writeImage(
            depthProvenancePath, provenance);
        if (!depthProvenanceSaved)
        {
            LOG_WARN(QStringLiteral(
                         "[MVS] 保存%1深度来源图失败: frame=%2")
                         .arg(stageLabel)
                         .arg(frameIndex));
        }
    }

    std::unordered_map<int, QJsonObject> pyramid_level_paths;
    if (_config.saveIntermediatePyramidLevels && saveRawDepth)
    {
        for (const DepthLevelResult &level : result.intermediatePyramidLevels)
        {
            if (level.depth.empty())
            {
                continue;
            }


            const cv::Size working_size = depthPyramidWorkingSize(
                result.depthMap->cols,
                result.depthMap->rows,
                level.downsampleFactor);
            const cv::Mat native_depth = restoreNativePyramidArtifact(level.depth, working_size);
            const cv::Mat native_confidence = restoreNativePyramidArtifact(
                level.confidence,
                working_size);
            const cv::Mat native_support = restoreNativePyramidArtifact(
                level.supportCount,
                working_size);
            const cv::Mat native_uncertainty = restoreNativePyramidArtifact(
                level.uncertainty,
                working_size);
            const cv::Mat native_mask = restoreNativePyramidArtifact(
                level.validMask,
                working_size);

            const std::string level_prefix = raw_directory + "/depth_" +
                                             std::to_string(frameIndex) + "_level_" +
                                             std::to_string(level.level);
            const std::string level_depth_path = level_prefix + ".bin";
            const std::string level_confidence_path = level_prefix + "_conf.bin";
            const std::string level_support_path = level_prefix + "_support.bin";
            const std::string level_uncertainty_path = level_prefix + "_uncertainty.bin";
            const std::string level_mask_path = level_prefix + "_mask.png";
            const std::string level_preview_path = _outputDir + "/depth_" +
                                                   std::to_string(frameIndex) + "_level_" +
                                                   std::to_string(level.level) + ".png";
            const std::string level_confidence_preview_path = _outputDir + "/depth_" +
                                                              std::to_string(frameIndex) + "_level_" +
                                                              std::to_string(level.level) + "_conf.png";
            if (!writeFastDepthMatStorage(level_depth_path, native_depth, &saveErr))
            {
                LOG_WARN(QStringLiteral("[MVS] 保存帧 %1 Level %2 深度失败: %3")
                             .arg(frameIndex)
                             .arg(level.level)
                             .arg(QString::fromStdString(saveErr)));
                continue;
            }

            QJsonObject paths;
            paths.insert(QStringLiteral("raw_depth_path"), QString::fromStdString(level_depth_path));
            paths.insert(QStringLiteral("artifact_width"), working_size.width);
            paths.insert(QStringLiteral("artifact_height"), working_size.height);
            if (!native_confidence.empty())
            {
                if (writeFastDepthMatStorage(level_confidence_path, native_confidence, &saveErr))
                {
                    paths.insert(QStringLiteral("raw_confidence_path"),
                                 QString::fromStdString(level_confidence_path));
                }
                else
                {
                    LOG_WARN(QStringLiteral("[MVS] 保存帧 %1 Level %2 置信图失败: %3")
                                 .arg(frameIndex)
                                 .arg(level.level)
                                 .arg(QString::fromStdString(saveErr)));
                }
            }

            if (!native_support.empty() &&
                writeFastDepthMatStorage(level_support_path, native_support, &saveErr))
            {
                paths.insert(QStringLiteral("raw_support_count_path"),
                             QString::fromStdString(level_support_path));
            }
            if (!native_uncertainty.empty() &&
                writeFastDepthMatStorage(level_uncertainty_path, native_uncertainty, &saveErr))
            {
                paths.insert(QStringLiteral("raw_uncertainty_path"),
                             QString::fromStdString(level_uncertainty_path));
            }
            if (!native_mask.empty() && xjw::common::io::writeImage(level_mask_path, native_mask))
            {
                paths.insert(QStringLiteral("valid_mask_path"), QString::fromStdString(level_mask_path));
            }

            if (savePreviewPng)
            {
                if (saveDepthPreviewPng(level_preview_path, native_depth, &saveErr))
                {
                    paths.insert(QStringLiteral("preview_path"),
                                 QString::fromStdString(level_preview_path));
                }
                if (!native_confidence.empty())
                {
                    cv::Mat confidence_preview;
                    native_confidence.convertTo(confidence_preview, CV_8U, 255.0);
                    if (xjw::common::io::writeImage(level_confidence_preview_path,
                                                   confidence_preview))
                    {
                        paths.insert(QStringLiteral("confidence_preview_path"),
                                     QString::fromStdString(level_confidence_preview_path));
                    }
                }
            }
            pyramid_level_paths.emplace(level.level, std::move(paths));
        }
    }

    LOG_DEBUG(QStringLiteral("[MVS] 保存%1深度产物耗时: frame=%2 preview=%3 ms raw=%4 ms confidence=%5 ms mask=%6 ms total=%7 ms")
                 .arg(stageLabel)
                 .arg(frameIndex)
                 .arg(previewMs, 0, 'f', 1)
                 .arg(rawMs, 0, 'f', 1)
                 .arg(confidenceMs, 0, 'f', 1)
                 .arg(maskMs, 0, 'f', 1)
                 .arg(elapsedMs(saveStart, Clock::now()), 0, 'f', 1));

    if (previewSaved && rawSaved && savePreviewPng)
    {
        emit depthMapSaved(
            QString::fromStdString(pngPath),
            result.depthMap->cols,
            result.depthMap->rows,
            QString::fromStdString(_views[frameIndex].imagePath));

        QJsonArray sourceImages;
        QJsonArray sourceIndices;
        QJsonArray sourcePlan;
        QStringList sourceImageList;
        const bool hasResultSourcePlan = !result.sourceViewPlan.empty();
        const bool hasSourceScoreCache =
            frameIndex >= 0
            && frameIndex < static_cast<int>(_frameCaches.size())
            && !_frameCaches[static_cast<size_t>(frameIndex)].sourceViewScores.empty();
        for (const int sourceIndex : result.sourceViewIndices)
        {
            if (sourceIndex < 0 || sourceIndex >= static_cast<int>(_views.size()))
            {
                continue;
            }
            sourceIndices.append(sourceIndex);
            const QString sourceImage = QString::fromStdString(_views[sourceIndex].imagePath);
            sourceImages.append(sourceImage);
            sourceImageList.append(sourceImage);
            const std::vector<MvsSourcePlanEntry> *scores = nullptr;
            if (hasResultSourcePlan)
            {
                scores = &result.sourceViewPlan;
            }
            else if (hasSourceScoreCache)
            {
                scores = &_frameCaches[static_cast<size_t>(frameIndex)].sourceViewScores;
            }
            if (scores)
            {
                const auto it = std::find_if(scores->cbegin(),
                                             scores->cend(),
                                             [sourceIndex](const MvsSourcePlanEntry &entry)
                                             {
                                                 return entry.viewIndex == sourceIndex;
                                             });
                if (it != scores->cend())
                {
                    QJsonObject sourcePlanEntry = mvsSourcePlanEntryToJson(*it);
                    sourcePlanEntry.insert(QStringLiteral("source_image"), sourceImage);
                    sourcePlan.append(sourcePlanEntry);
                }
            }
        }

        const SourceQualitySummary sourceQualitySummary =
            summarizeSourceQuality(sourcePlan, sourceImageList.size());
        const cv::Mat *confidenceMap = (result.confidence && !result.confidence->empty())
            ? result.confidence.data()
            : nullptr;
        const DepthConfidenceSummary depthConfidenceSummary =
            summarizeDepthConfidence(*result.depthMap, confidenceMap);
        DepthCompletenessDiagnostics depthCompleteness = result.depthCompleteness;
        cv::Mat completenessMask;
        if (result.supportRegionMask && !result.supportRegionMask->empty())
        {
            completenessMask = *result.supportRegionMask;
            if (completenessMask.size() != result.depthMap->size())
            {
                cv::resize(completenessMask,
                           completenessMask,
                           result.depthMap->size(),
                           0.0,
                           0.0,
                           cv::INTER_NEAREST);
            }
        }
        else
        {
            completenessMask = cv::Mat(
                result.depthMap->size(), CV_8UC1, cv::Scalar(255));
        }
        depthCompleteness.finalMetrics = analyzeDepthCompleteness(
            *result.depthMap, completenessMask);
        if (result.depthPostprocessApplied)
        {
            depthCompleteness.preFusionPostprocessValidCount =
                result.depthPostprocess.validBeforePostprocess;
            depthCompleteness.postConfidenceFilterValidCount =
                result.depthPostprocess.validAfterConfidenceFilter;
            depthCompleteness.postFusionPostprocessValidCount =
                result.depthPostprocess.validAfterPostprocess;
        }
        const QJsonObject depthCompletenessJson =
            depthCompletenessDiagnosticsToJson(depthCompleteness);
        const cv::Mat empty_geometry_evidence;
        QJsonObject geometryEvidenceDiagnostics =
            geometryEvidenceDiagnosticsToJson(
                *result.depthMap,
                result.geometrySupportCount &&
                        !result.geometrySupportCount->empty()
                    ? *result.geometrySupportCount
                    : empty_geometry_evidence,
                result.inverseDepthRelativeSpread &&
                        !result.inverseDepthRelativeSpread->empty()
                    ? *result.inverseDepthRelativeSpread
                    : empty_geometry_evidence,
                result.crossViewRepairedMask &&
                        !result.crossViewRepairedMask->empty()
                    ? *result.crossViewRepairedMask
                    : empty_geometry_evidence,
                result.supportRegionMask &&
                        !result.supportRegionMask->empty()
                    ? *result.supportRegionMask
                    : empty_geometry_evidence);
        const bool adaptive_evidence_available =
            result.adaptiveGeometrySupportWeight &&
            !result.adaptiveGeometrySupportWeight->empty() &&
            result.adaptiveGeometryEffectiveViewCount &&
            !result.adaptiveGeometryEffectiveViewCount->empty() &&
            result.adaptiveGeometryConflictRatio &&
            !result.adaptiveGeometryConflictRatio->empty() &&
            result.adaptiveGeometrySupportWeight->size() == result.depthMap->size() &&
            result.adaptiveGeometryEffectiveViewCount->size() == result.depthMap->size() &&
            result.adaptiveGeometryConflictRatio->size() == result.depthMap->size();
        geometryEvidenceDiagnostics.insert(
            QStringLiteral("adaptive_evidence_available"),
            adaptive_evidence_available);
        geometryEvidenceDiagnostics.insert(
            QStringLiteral("adaptive_reliability_model"),
            QStringLiteral("source_confidence_x_source_quality"));
        geometryEvidenceDiagnostics.insert(
            QStringLiteral("adaptive_hypothesis_domain"),
            QStringLiteral("pre_consistency_depth"));
        if (adaptive_evidence_available)
        {
            const cv::Mat candidate_mask =
                *result.adaptiveGeometryEffectiveViewCount > 0.0f;
            const cv::Mat retained_candidate_mask =
                candidate_mask & (*result.depthMap > 0.0f);
            const int candidate_count = cv::countNonZero(candidate_mask);
            const int retained_candidate_count =
                cv::countNonZero(retained_candidate_mask);
            geometryEvidenceDiagnostics.insert(
                QStringLiteral("adaptive_candidate_pixel_count"),
                candidate_count);
            geometryEvidenceDiagnostics.insert(
                QStringLiteral("adaptive_candidate_removed_by_hard_gate"),
                std::max(0, candidate_count - retained_candidate_count));
            geometryEvidenceDiagnostics.insert(
                QStringLiteral("adaptive_support_weight_mean"),
                cv::mean(
                    *result.adaptiveGeometrySupportWeight,
                    candidate_mask)[0]);
            geometryEvidenceDiagnostics.insert(
                QStringLiteral("adaptive_effective_view_count_mean"),
                cv::mean(
                    *result.adaptiveGeometryEffectiveViewCount,
                    candidate_mask)[0]);
            geometryEvidenceDiagnostics.insert(
                QStringLiteral("adaptive_conflict_ratio_mean"),
                cv::mean(
                    *result.adaptiveGeometryConflictRatio,
                    candidate_mask)[0]);
        }
        const cv::Mat emptyConfidence;
        const DepthMapQualityMetrics depthQualityMetrics = result.qualityMetrics.width > 0
            ? result.qualityMetrics
            : analyzeDepthMapQuality(*result.depthMap,
                                     confidenceMap ? *confidenceMap : emptyConfidence,
                                     sourceQualitySummary.sourceViewCount);
        QJsonObject depthQualityJson = depthMapQualityMetricsToJson(depthQualityMetrics);
        for (auto it = depthCompletenessJson.constBegin();
             it != depthCompletenessJson.constEnd();
             ++it)
        {
            depthQualityJson.insert(it.key(), it.value());
        }
        const int requestedSourceViewCount =
            result.requestedSourceViewCount > 0
            ? result.requestedSourceViewCount
            : (hasSourceScoreCache
                   ? _frameCaches[static_cast<size_t>(frameIndex)]
                         .requestedSourceViewCount
                   : sourceQualitySummary.sourceViewCount);
        const int sourceViewShortfall = std::max(
            0,
            requestedSourceViewCount - sourceQualitySummary.sourceViewCount);
        const QString sourceViewShortfallReason =
            !result.sourceViewShortfallReason.empty()
            ? QString::fromStdString(result.sourceViewShortfallReason)
            : (hasSourceScoreCache
                   ? QString::fromStdString(
                         _frameCaches[static_cast<size_t>(frameIndex)]
                             .sourceViewShortfallReason)
                   : QString());
        depthQualityJson[QStringLiteral("requested_source_view_count")] = requestedSourceViewCount;
        depthQualityJson[QStringLiteral("source_view_shortfall")] = sourceViewShortfall;
        depthQualityJson[QStringLiteral("source_view_shortfall_reason")] =
            sourceViewShortfallReason;
        depthQualityJson[QStringLiteral("verified_source_view_count")] =
            sourceQualitySummary.verifiedSourceViewCount;
        depthQualityJson[QStringLiteral("backfill_source_view_count")] =
            sourceQualitySummary.backfillSourceViewCount;
        depthQualityJson[QStringLiteral("sequence_fallback_source_view_count")] =
            sourceQualitySummary.sequenceFallbackSourceViewCount;
        const QJsonObject qualityDecisionJson =
            depthFrameQualityDecisionToJson(result.qualityDecision);
        QJsonArray pyramidLevelsJson = depthPyramidLevelsToJson(result.pyramidLevels);
        for (qsizetype index = 0; index < pyramidLevelsJson.size(); ++index)
        {
            QJsonObject level_object = pyramidLevelsJson.at(index).toObject();
            const int level = level_object.value(QStringLiteral("level")).toInt();
            if (level == 1)
            {
                level_object.insert(QStringLiteral("preview_path"), QString::fromStdString(pngPath));
                level_object.insert(QStringLiteral("raw_depth_path"),
                                    saveRawDepth ? QString::fromStdString(rawDepthPath) : QString());
                level_object.insert(QStringLiteral("raw_confidence_path"),
                                    confidenceSaved ? QString::fromStdString(rawConfidencePath) : QString());
                level_object.insert(QStringLiteral("raw_geometry_support_path"),
                                    geometrySupportSaved
                                        ? QString::fromStdString(rawGeometrySupportPath)
                                        : QString());
                level_object.insert(QStringLiteral("raw_geometry_source_mask_path"),
                                    geometrySourceMaskSaved
                                        ? QString::fromStdString(rawGeometrySourceMaskPath)
                                        : QString());
                level_object.insert(QStringLiteral("raw_inverse_depth_mean_path"),
                                    inverseDepthMeanSaved
                                        ? QString::fromStdString(rawInverseDepthMeanPath)
                                        : QString());
                level_object.insert(QStringLiteral("raw_inverse_depth_spread_path"),
                                    inverseDepthSpreadSaved
                                        ? QString::fromStdString(rawInverseDepthSpreadPath)
                                        : QString());
                level_object.insert(
                    QStringLiteral("raw_adaptive_geometry_support_weight_path"),
                    adaptiveGeometrySupportWeightSaved
                        ? QString::fromStdString(rawAdaptiveGeometrySupportWeightPath)
                        : QString());
                level_object.insert(
                    QStringLiteral("raw_adaptive_geometry_effective_view_count_path"),
                    adaptiveGeometryEffectiveViewCountSaved
                        ? QString::fromStdString(
                              rawAdaptiveGeometryEffectiveViewCountPath)
                        : QString());
                level_object.insert(
                    QStringLiteral("raw_adaptive_geometry_conflict_ratio_path"),
                    adaptiveGeometryConflictRatioSaved
                        ? QString::fromStdString(rawAdaptiveGeometryConflictRatioPath)
                        : QString());
                level_object.insert(QStringLiteral("cross_view_repaired_mask_path"),
                                    crossViewRepairedMaskSaved
                                        ? QString::fromStdString(crossViewRepairedMaskPath)
                                        : QString());
                level_object.insert(QStringLiteral("valid_mask_path"),
                                    maskSaved ? QString::fromStdString(validMaskPath) : QString());
                level_object.insert(QStringLiteral("missing_reason_path"),
                                    missingReasonSaved
                                        ? QString::fromStdString(missingReasonPath)
                                        : QString());
                level_object.insert(
                    QStringLiteral("targeted_gap_recovered_mask_path"),
                    targetedGapRecoveredMaskSaved
                        ? QString::fromStdString(targetedGapRecoveredMaskPath)
                        : QString());
                level_object.insert(
                    QStringLiteral("residual_reestimated_mask_path"),
                    residualReestimatedMaskSaved
                        ? QString::fromStdString(residualReestimatedMaskPath)
                        : QString());
                level_object.insert(
                    QStringLiteral("depth_provenance_path"),
                    depthProvenanceSaved
                        ? QString::fromStdString(depthProvenancePath)
                        : QString());
                level_object.insert(
                    QStringLiteral("missing_reason_preview_path"),
                    missingReasonPreviewSaved
                        ? QString::fromStdString(missingReasonPreviewPath)
                        : QString());
            }
            const auto path_it = pyramid_level_paths.find(level);
            if (path_it != pyramid_level_paths.end())
            {
                for (auto value_it = path_it->second.constBegin();
                     value_it != path_it->second.constEnd();
                     ++value_it)
                {
                    level_object.insert(value_it.key(), value_it.value());
                }
            }
            pyramidLevelsJson.replace(index, level_object);
        }
        const QString sceneProfile = sceneProfileId(_effectiveSceneProfile);
        const QString filterMode = depthFilterModeId(_effectiveDepthFilterMode);
        const QString acceptance = QString::fromLatin1(
            depthFrameAcceptanceId(result.qualityDecision.acceptance));
        const QJsonObject depthPostprocessJson =
            depthPostProcessStatsToJson(result.depthPostprocess);

        QJsonObject artifact;
        artifact[QStringLiteral("ref_index")] = frameIndex;
        artifact[QStringLiteral("depth_png")] = QString::fromStdString(pngPath);
        artifact[QStringLiteral("raw_depth_path")] = saveRawDepth ? QString::fromStdString(rawDepthPath) : QString();
        artifact[QStringLiteral("raw_confidence_path")] =
            confidenceSaved ? QString::fromStdString(rawConfidencePath) : QString();
        artifact[QStringLiteral("raw_geometry_support_path")] =
            geometrySupportSaved ? QString::fromStdString(rawGeometrySupportPath) : QString();
        artifact[QStringLiteral("raw_geometry_source_mask_path")] =
            geometrySourceMaskSaved
                ? QString::fromStdString(rawGeometrySourceMaskPath)
                : QString();
        artifact[QStringLiteral("raw_inverse_depth_mean_path")] =
            inverseDepthMeanSaved ? QString::fromStdString(rawInverseDepthMeanPath) : QString();
        artifact[QStringLiteral("raw_inverse_depth_spread_path")] =
            inverseDepthSpreadSaved
                ? QString::fromStdString(rawInverseDepthSpreadPath)
                : QString();
        artifact[QStringLiteral("raw_adaptive_geometry_support_weight_path")] =
            adaptiveGeometrySupportWeightSaved
                ? QString::fromStdString(rawAdaptiveGeometrySupportWeightPath)
                : QString();
        artifact[QStringLiteral("raw_adaptive_geometry_effective_view_count_path")] =
            adaptiveGeometryEffectiveViewCountSaved
                ? QString::fromStdString(rawAdaptiveGeometryEffectiveViewCountPath)
                : QString();
        artifact[QStringLiteral("raw_adaptive_geometry_conflict_ratio_path")] =
            adaptiveGeometryConflictRatioSaved
                ? QString::fromStdString(rawAdaptiveGeometryConflictRatioPath)
                : QString();
        artifact[QStringLiteral("cross_view_repaired_mask_path")] =
            crossViewRepairedMaskSaved
                ? QString::fromStdString(crossViewRepairedMaskPath)
                : QString();
        artifact[QStringLiteral("targeted_gap_recovered_mask_path")] =
            targetedGapRecoveredMaskSaved
                ? QString::fromStdString(targetedGapRecoveredMaskPath)
                : QString();
        artifact[QStringLiteral("residual_reestimated_mask_path")] =
            residualReestimatedMaskSaved
                ? QString::fromStdString(residualReestimatedMaskPath)
                : QString();
        artifact[QStringLiteral("depth_provenance_path")] =
            depthProvenanceSaved
                ? QString::fromStdString(depthProvenancePath)
                : QString();
        artifact[QStringLiteral("valid_mask_path")] = maskSaved ? QString::fromStdString(validMaskPath) : QString();
        artifact[QStringLiteral("support_mask_path")] =
            supportMaskSaved ? QString::fromStdString(supportMaskPath) : QString();
        artifact[QStringLiteral("missing_reason_path")] =
            missingReasonSaved ? QString::fromStdString(missingReasonPath) : QString();
        artifact[QStringLiteral("missing_reason_preview_path")] =
            missingReasonPreviewSaved
                ? QString::fromStdString(missingReasonPreviewPath)
                : QString();
        artifact[QStringLiteral("ref_image")] = QString::fromStdString(_views[frameIndex].imagePath);
        artifact[QStringLiteral("source_images")] = sourceImages;
        artifact[QStringLiteral("source_indices")] = sourceIndices;
        artifact[QStringLiteral("source_plan")] = sourcePlan;
        artifact[QStringLiteral("quality_profile")] =
            QString::fromStdString(_config.qualityProfile);
        artifact[QStringLiteral("configured_source_view_count")] =
            _configuredSourceViewCount;
        artifact[QStringLiteral("source_view_count")] = sourceQualitySummary.sourceViewCount;
        artifact[QStringLiteral("requested_source_view_count")] = requestedSourceViewCount;
        artifact[QStringLiteral("source_view_shortfall")] = sourceViewShortfall;
        artifact[QStringLiteral("source_view_shortfall_reason")] =
            sourceViewShortfallReason;
        artifact[QStringLiteral("verified_source_view_count")] =
            sourceQualitySummary.verifiedSourceViewCount;
        artifact[QStringLiteral("backfill_source_view_count")] =
            sourceQualitySummary.backfillSourceViewCount;
        artifact[QStringLiteral("source_quality_mean")] = sourceQualitySummary.meanQuality;
        artifact[QStringLiteral("source_quality_min")] = sourceQualitySummary.minQuality;
        artifact[QStringLiteral("depth_confidence_mean")] = depthConfidenceSummary.meanConfidence;
        artifact[QStringLiteral("effective_patch_match_confidence_threshold")] =
            result.effectivePatchMatchConfidenceThreshold;
        artifact[QStringLiteral("valid_pixel_count")] = depthConfidenceSummary.validPixelCount;
        artifact[QStringLiteral("valid_coverage")] =
            static_cast<double>(result.qualityMetrics.validCoverage);
        artifact[QStringLiteral("depth_quality")] = depthQualityJson;
        artifact[QStringLiteral("depth_completeness")] = depthCompletenessJson;
        artifact[QStringLiteral("missing_reason_summary")] =
            missingReasonSummaryJson;
        artifact[QStringLiteral("cross_view_repair_diagnostics")] =
            result.crossViewRepairDiagnostics;
        artifact[QStringLiteral("targeted_gap_recovery_diagnostics")] =
            result.targetedGapRecoveryDiagnostics;
        artifact[QStringLiteral("residual_reestimation_diagnostics")] =
            result.residualReestimationDiagnostics;
        artifact[QStringLiteral("depth_provenance_summary")] =
            depthProvenanceSummaryJson;
        artifact[QStringLiteral("geometry_evidence_diagnostics")] =
            geometryEvidenceDiagnostics;
        if (!result.poseRefinementDiagnostics.isEmpty())
        {
            artifact[QStringLiteral("pose_refinement_diagnostics")] =
                result.poseRefinementDiagnostics;
        }
        if (result.derivedCameraModel.isValid())
        {
            artifact[QStringLiteral("derived_camera_model")] =
                cameraModelToJson(result.derivedCameraModel);
        }
        if (depthCompleteness.finalMetrics.validInputs)
        {
            artifact[QStringLiteral("mask_pixel_count")] =
                depthCompleteness.finalMetrics.maskPixelCount;
            artifact[QStringLiteral("valid_within_mask_count")] =
                depthCompleteness.finalMetrics.validWithinMaskCount;
            artifact[QStringLiteral("valid_within_mask_ratio")] =
                depthCompleteness.finalMetrics.validWithinMaskRatio;
        }
        artifact[QStringLiteral("quality_decision")] = qualityDecisionJson;
        artifact[QStringLiteral("pyramid_levels")] = pyramidLevelsJson;
        artifact[QStringLiteral("mask_source")] = QString::fromStdString(result.maskSource);
        artifact[QStringLiteral("mask_coverage")] = result.maskCoverage;
        artifact[QStringLiteral("selected_level")] = result.selectedLevel;
        artifact[QStringLiteral("fallback_reason")] = QString::fromStdString(result.fallbackReason);
        artifact[QStringLiteral("pyramid_requested_level_count")] =
            result.pyramidRequestedLevelCount;
        artifact[QStringLiteral("pyramid_active_level_count")] =
            result.pyramidActiveLevelCount;
        artifact[QStringLiteral("pyramid_minimum_short_side")] =
            result.pyramidMinimumShortSide;
        artifact[QStringLiteral("pyramid_degraded_reason")] =
            QString::fromStdString(result.pyramidDegradedReason);
        artifact[QStringLiteral("scene_profile")] = sceneProfile;
        artifact[QStringLiteral("filter_mode")] = filterMode;
        artifact[QStringLiteral("acceptance")] = acceptance;
        artifact[QStringLiteral("fusion_eligible")] = result.eligibleForFusion();
        artifact[QStringLiteral("depth_postprocess")] = depthPostprocessJson;
        artifact[QStringLiteral("camera_model")] =
            cameraModelToJson(result.cameraModel.isValid()
                                  ? result.cameraModel
                                  : mvsPinholeCamera(_views[frameIndex].camera));
        artifact[QStringLiteral("status")] = QStringLiteral("completed");
        artifact[QStringLiteral("stage")] = stageLabel;
        artifact[QStringLiteral("device")] = QString::fromStdString(result.device.empty() ? "unknown" : result.device);
        artifact[QStringLiteral("elapsed_ms")] = result.elapsedMs;
        artifact[QStringLiteral("grid_width")] = result.depthMap->cols;
        artifact[QStringLiteral("grid_height")] = result.depthMap->rows;
        artifact[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
        artifact[QStringLiteral("config_hash")] = _depthConfigHash;
        artifact[QStringLiteral("algorithm_revision")] = kMvsDepthAlgorithmRevision;
        artifact[QStringLiteral("manifest_path")] = _workspaceManifestPath;

        MvsDepthFrameRecord record;
        record.refIndex = frameIndex;
        record.refImage = QString::fromStdString(_views[frameIndex].imagePath);
        record.sourceImages = sourceImageList;
        for (const QJsonValue &source_value : sourceIndices)
        {
            record.sourceIndices.push_back(source_value.toInt(-1));
        }
        record.sourcePlan = sourcePlan;
        record.qualityProfile = QString::fromStdString(_config.qualityProfile);
        record.configuredSourceViewCount = _configuredSourceViewCount;
        record.sourceViewCount = sourceQualitySummary.sourceViewCount;
        record.requestedSourceViewCount = requestedSourceViewCount;
        record.sourceViewShortfall = sourceViewShortfall;
        record.sourceViewShortfallReason = sourceViewShortfallReason;
        record.meanSourceQualityScore = sourceQualitySummary.meanQuality;
        record.minSourceQualityScore = sourceQualitySummary.minQuality;
        record.meanDepthConfidence = depthConfidenceSummary.meanConfidence;
        record.effectivePatchMatchConfidenceThreshold =
            result.effectivePatchMatchConfidenceThreshold;
        record.validPixelCount = depthConfidenceSummary.validPixelCount;
        record.validCoverage = static_cast<double>(result.qualityMetrics.validCoverage);
        record.depthQuality = depthQualityJson;
        record.depthCompleteness = depthCompletenessJson;
        record.missingReasonSummary = missingReasonSummaryJson;
        record.crossViewRepairDiagnostics = result.crossViewRepairDiagnostics;
        record.targetedGapRecoveryDiagnostics =
            result.targetedGapRecoveryDiagnostics;
        record.residualReestimationDiagnostics =
            result.residualReestimationDiagnostics;
        record.depthProvenanceSummary = depthProvenanceSummaryJson;
        record.geometryEvidenceDiagnostics = geometryEvidenceDiagnostics;
        record.poseRefinementDiagnostics = result.poseRefinementDiagnostics;
        if (result.derivedCameraModel.isValid())
        {
            record.derivedCameraModel =
                cameraModelToJson(result.derivedCameraModel);
        }
        record.qualityDecision = qualityDecisionJson;
        record.pyramidLevels = pyramidLevelsJson;
        record.maskSource = QString::fromStdString(result.maskSource);
        record.maskCoverage = result.maskCoverage;
        record.selectedLevel = result.selectedLevel;
        record.fallbackReason = QString::fromStdString(result.fallbackReason);
        record.pyramidRequestedLevelCount = result.pyramidRequestedLevelCount;
        record.pyramidActiveLevelCount = result.pyramidActiveLevelCount;
        record.pyramidMinimumShortSide = result.pyramidMinimumShortSide;
        record.pyramidDegradedReason = QString::fromStdString(
            result.pyramidDegradedReason);
        record.sceneProfile = sceneProfile;
        record.filterMode = filterMode;
        record.acceptance = acceptance;
        record.depthPostprocess = depthPostprocessJson;
        record.cameraModel = cameraModelToJson(result.cameraModel.isValid()
                                                   ? result.cameraModel
                                                   : mvsPinholeCamera(_views[frameIndex].camera));
        record.status = QStringLiteral("completed");
        record.device = QString::fromStdString(result.device.empty() ? "unknown" : result.device);
        record.depthPng = QString::fromStdString(pngPath);
        record.rawDepthPath = saveRawDepth ? QString::fromStdString(rawDepthPath) : QString();
        record.rawConfidencePath = confidenceSaved ? QString::fromStdString(rawConfidencePath) : QString();
        record.rawGeometrySupportPath = geometrySupportSaved
            ? QString::fromStdString(rawGeometrySupportPath)
            : QString();
        record.rawGeometrySourceMaskPath = geometrySourceMaskSaved
            ? QString::fromStdString(rawGeometrySourceMaskPath)
            : QString();
        record.rawInverseDepthMeanPath = inverseDepthMeanSaved
            ? QString::fromStdString(rawInverseDepthMeanPath)
            : QString();
        record.rawInverseDepthSpreadPath = inverseDepthSpreadSaved
            ? QString::fromStdString(rawInverseDepthSpreadPath)
            : QString();
        record.rawAdaptiveGeometrySupportWeightPath =
            adaptiveGeometrySupportWeightSaved
            ? QString::fromStdString(rawAdaptiveGeometrySupportWeightPath)
            : QString();
        record.rawAdaptiveGeometryEffectiveViewCountPath =
            adaptiveGeometryEffectiveViewCountSaved
            ? QString::fromStdString(rawAdaptiveGeometryEffectiveViewCountPath)
            : QString();
        record.rawAdaptiveGeometryConflictRatioPath =
            adaptiveGeometryConflictRatioSaved
            ? QString::fromStdString(rawAdaptiveGeometryConflictRatioPath)
            : QString();
        record.crossViewRepairedMaskPath = crossViewRepairedMaskSaved
            ? QString::fromStdString(crossViewRepairedMaskPath)
            : QString();
        record.targetedGapRecoveredMaskPath = targetedGapRecoveredMaskSaved
            ? QString::fromStdString(targetedGapRecoveredMaskPath)
            : QString();
        record.residualReestimatedMaskPath = residualReestimatedMaskSaved
            ? QString::fromStdString(residualReestimatedMaskPath)
            : QString();
        record.depthProvenancePath = depthProvenanceSaved
            ? QString::fromStdString(depthProvenancePath)
            : QString();
        record.validMaskPath = maskSaved ? QString::fromStdString(validMaskPath) : QString();
        record.supportMaskPath = supportMaskSaved ? QString::fromStdString(supportMaskPath) : QString();
        record.missingReasonPath = missingReasonSaved
            ? QString::fromStdString(missingReasonPath)
            : QString();
        record.missingReasonPreviewPath = missingReasonPreviewSaved
            ? QString::fromStdString(missingReasonPreviewPath)
            : QString();
        record.gridWidth = result.depthMap->cols;
        record.gridHeight = result.depthMap->rows;
        record.elapsedMs = static_cast<qint64>(std::llround(result.elapsedMs));
        record.configHash = _depthConfigHash;
        record.algorithmRevision = kMvsDepthAlgorithmRevision;

        {
            std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
            _workspaceManifest.markCompleted(record);
            QString manifestError;
            if (!persistWorkspaceManifest(&manifestError))
            {
                LOG_WARN(QStringLiteral("[MVS] 写入完成 manifest 失败: %1").arg(manifestError));
                emit errorOccurred(manifestError);
                return false;
            }
        }
        emit depthMapArtifactSaved(artifact);
    }

    return previewSaved && rawSaved;
}

void DepthMapGenerator::runDepthPoseRefinementCandidateStage(
    bool residentDepthFrames)
{
    if (!_config.depthPoseRefinement.enabled)
    {
        return;
    }

    DepthPoseRefinementStageResult stage;
    stage.enabled = true;
    stage.candidateOnly = true;
    stage.anchorCameraIndex =
        _config.depthPoseRefinement.optimizer.anchorCameraIndex;
    if (residentDepthFrames)
    {
        std::vector<DepthPoseRefinementFrame> frames;
        frames.reserve(_depthFrames.size());
        for (int frame_index = 0;
             frame_index < static_cast<int>(_depthFrames.size());
             ++frame_index)
        {
            const DepthFrameResult &depth_frame =
                _depthFrames[static_cast<std::size_t>(frame_index)];
            DepthPoseRefinementFrame frame;
            frame.cameraIndex = frame_index;
            frame.camera = depth_frame.cameraModel.isValid()
                ? depth_frame.cameraModel
                : mvsPinholeCamera(
                    _views[static_cast<std::size_t>(frame_index)].camera);
            frame.depthMap = depth_frame.depthMap
                ? *depth_frame.depthMap : cv::Mat();
            frame.normalMap = depth_frame.normalMap
                ? *depth_frame.normalMap : cv::Mat();
            frame.confidence = depth_frame.confidence
                ? *depth_frame.confidence : cv::Mat();
            frame.adaptiveSupportWeight =
                depth_frame.adaptiveGeometrySupportWeight
                ? *depth_frame.adaptiveGeometrySupportWeight : cv::Mat();
            frame.adaptiveEffectiveViewCount =
                depth_frame.adaptiveGeometryEffectiveViewCount
                ? *depth_frame.adaptiveGeometryEffectiveViewCount : cv::Mat();
            frame.adaptiveConflictRatio =
                depth_frame.adaptiveGeometryConflictRatio
                ? *depth_frame.adaptiveGeometryConflictRatio : cv::Mat();
            frame.sourceCameraIndices = depth_frame.sourceViewIndices;
            frames.push_back(std::move(frame));
        }
        stage = DepthPoseRefinementStage::buildCandidates(
            frames,
            _config.depthPoseRefinement);
    }
    else
    {
        stage.candidates.reserve(_depthFrames.size());
        for (int frame_index = 0;
             frame_index < static_cast<int>(_depthFrames.size());
             ++frame_index)
        {
            DepthPoseRefinementCandidate candidate;
            candidate.cameraIndex = frame_index;
            candidate.reason = "streaming_depth_not_resident";
            stage.candidates.push_back(std::move(candidate));
        }
    }

    {
        std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
        for (const DepthPoseRefinementCandidate &candidate : stage.candidates)
        {
            if (candidate.cameraIndex < 0 ||
                candidate.cameraIndex >=
                    static_cast<int>(_depthFrames.size()))
            {
                continue;
            }
            DepthFrameResult &frame = _depthFrames[
                static_cast<std::size_t>(candidate.cameraIndex)];
            frame.poseRefinementDiagnostics =
                depthPoseRefinementCandidateToJson(candidate, stage);
            if (candidate.accepted && candidate.derivedCamera.isValid())
            {
                frame.derivedCameraModel = candidate.derivedCamera;
            }
            const QJsonObject derived_camera =
                frame.derivedCameraModel.isValid()
                ? cameraModelToJson(frame.derivedCameraModel)
                : QJsonObject{};
            _workspaceManifest.updatePoseRefinement(
                candidate.cameraIndex,
                frame.poseRefinementDiagnostics,
                derived_camera);
        }
        QString manifest_error;
        if (!_workspaceManifestPath.isEmpty() &&
            !persistWorkspaceManifest(&manifest_error))
        {
            LOG_WARN(QStringLiteral(
                "[MVS] 写入位姿细化候选诊断失败: %1")
                         .arg(manifest_error));
        }
    }

    LOG_INFO(QStringLiteral(
        "[MVS] 深度约束位姿细化候选完成: accepted=%1/%2 mode=candidate_only; "
        "项目相机与本轮深度均未修改")
                 .arg(std::count_if(
                     stage.candidates.begin(),
                     stage.candidates.end(),
                     [](const DepthPoseRefinementCandidate &candidate)
                     {
                         return candidate.accepted;
                     }))
                 .arg(stage.candidates.size()));
}

// =============================================================================
void DepthMapGenerator::emitFinishedOnce(bool success)
{
    bool expected = false;
    if (_finishedEmitted.compare_exchange_strong(expected, true))
    {
        emit finished(success);
    }
}

void DepthMapGenerator::clearRuntimeCachesAfterFailure()
{
    _imageCache.reset();
    clearFrameCaches();
    releaseStoredDepthFramePixelStorage(_depthFrames);

    std::lock_guard<std::mutex> lock(_filteredDepthsMutex);
    _filteredDepths.clear();
}

void detail::runDepthMapBackgroundTaskWithExceptionBoundary(
    const std::function<void()> &task,
    const std::function<void(const QString &)> &failureHandler)
{
    try
    {
        task();
    }
    catch (const std::exception &error)
    {
        failureHandler(QStringLiteral("MVS 后台任务异常终止：%1")
                           .arg(QString::fromUtf8(error.what())));
    }
    catch (...)
    {
        failureHandler(QStringLiteral("MVS 后台任务异常终止：未知异常"));
    }
}

void DepthMapGenerator::runInBackground()
{
    detail::runDepthMapBackgroundTaskWithExceptionBoundary(
        [this]()
        {
            runInBackgroundImpl();
        },
        [this](const QString &message)
        {
            clearRuntimeCachesAfterFailure();
            if (_cancelled.load(std::memory_order_relaxed))
            {
                emitFinishedOnce(false);
                return;
            }
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            emit errorOccurred(message);
            emitFinishedOnce(false);
        });
}

void DepthMapGenerator::runInBackgroundImpl()
{
    const auto runStart = Clock::now();
    bool allOk = true;
    const int NV = static_cast<int>(_views.size());
    const int runCpuThreadBudget = resolvedTotalCpuThreadBudget(_config);
#ifdef _OPENMP
    // This background thread owns the serial preparation, consistency, and
    // fusion stages. Keep their implicit OpenMP teams inside the same budget
    // used by frame workers instead of falling back to all logical CPUs.
    omp_set_num_threads(runCpuThreadBudget);
#endif

    if (!_config.runDepthEstimation)
    {
        emit errorOccurred(QStringLiteral("当前生成器配置未启用深度估计阶段"));
        emitFinishedOnce(false);
        return;
    }

    _sceneClassification = classifyMvsScene(_views, _sparse);
    _effectiveSceneProfile = _config.sceneProfile == MvsSceneProfile::Auto
        ? _sceneClassification.profile
        : _config.sceneProfile;
    if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject &&
        kMvsDepthAlgorithmRevision >= kMvsAdaptiveGeometryEvidenceRevision &&
        !_config.enableAdaptiveGeometryEvidence)
    {
        const QString error = QStringLiteral(
            "MVS revision %1 的环拍深度必须生成连续几何证据；"
            "请启用 adaptive geometry evidence 后重试")
                                  .arg(kMvsDepthAlgorithmRevision);
        LOG_ERROR(QStringLiteral("[MVS] %1").arg(error));
        emit errorOccurred(error);
        emitFinishedOnce(false);
        return;
    }
    _effectiveDepthFilterMode = _config.depthFilterMode;
    if (_config.adaptiveDepthFilterMode)
    {
        _effectiveDepthFilterMode = _effectiveSceneProfile == MvsSceneProfile::AerialTerrain
            ? DepthFilterMode::Moderate
            : DepthFilterMode::Mild;
    }
    const int configured_source_count = _config.configuredSourceViewCount > 0
        ? _config.configuredSourceViewCount
        : _config.numSourceViews;
    _configuredSourceViewCount = configured_source_count;
    _config.numSourceViews = recommendedMvsSourceViewCount(
        _effectiveSceneProfile,
        _config.patchMatch.downsampleFactor,
        configured_source_count,
        NV);
    _config.patchMatch.numSourceViews = _config.numSourceViews;
    LOG_INFO(QStringLiteral(
                 "[MVS] 场景分类: profile=%1 plane_thickness=%2 down_looking=%3 "
                 "axis_convergence=%4 filter=%5 source_pool=%6 (configured=%7) reason=%8")
                 .arg(_effectiveSceneProfile == MvsSceneProfile::AerialTerrain
                          ? QStringLiteral("aerial_terrain")
                          : QStringLiteral("orbital_object"))
                 .arg(_sceneClassification.planeThicknessRatio, 0, 'f', 3)
                 .arg(_sceneClassification.downLookingConsistency, 0, 'f', 3)
                 .arg(_sceneClassification.opticalAxisConvergence, 0, 'f', 3)
                 .arg(_effectiveDepthFilterMode == DepthFilterMode::Mild
                          ? QStringLiteral("mild")
                          : (_effectiveDepthFilterMode == DepthFilterMode::Aggressive
                                 ? QStringLiteral("aggressive")
                                 : QStringLiteral("moderate")))
                 .arg(_config.numSourceViews)
                 .arg(configured_source_count)
                 .arg(QString::fromStdString(_sceneClassification.reason)));

    const PatchMatchBackend configuredBackend = _config.patchMatch.backend;
    const bool automaticAcceleration =
        configuredBackend == PatchMatchBackend::Auto && _config.patchMatch.useCuda;
    const bool probeCuda = configuredBackend == PatchMatchBackend::Cuda ||
        automaticAcceleration;
    const int cudaDeviceCount = probeCuda
        ? PatchMatchDepthEstimator::cudaDeviceCount()
        : 0;

    std::vector<DepthComputeWorker> physicalAcceleratorWorkers;
    std::vector<std::unique_ptr<GpuDeviceLeaseSet>> acceleratorDeviceLeases;
    std::unordered_set<std::string> selectedPhysicalDeviceIdentities;
    QStringList acceleratorPreparationFailures;
    auto acquire_device_lease = [&acceleratorPreparationFailures](
                                    const GpuDeviceDescriptor &descriptor)
        -> std::unique_ptr<GpuDeviceLeaseSet>
    {
        auto lease = std::make_unique<GpuDeviceLeaseSet>();
        QString lease_error;
        if (!lease->acquire({descriptor}, &lease_error))
        {
            acceleratorPreparationFailures.push_back(lease_error);
            LOG_WARN(QStringLiteral("[MVS] %1").arg(lease_error));
            return nullptr;
        }
        return lease;
    };

    auto try_add_cuda_device = [&](int device_index)
    {
        const std::string name = PatchMatchDepthEstimator::cudaDeviceName(device_index);
        std::string identity = PatchMatchDepthEstimator::cudaDeviceIdentity(device_index);
        if (identity.empty())
        {
            identity = fallbackGpuPhysicalIdentity("NVIDIA", name, device_index);
        }
        const GpuDeviceDescriptor descriptor{
            identity,
            name.empty() ? "CUDA:" + std::to_string(device_index) : name};
        std::unique_ptr<GpuDeviceLeaseSet> lease = acquire_device_lease(descriptor);
        if (!lease)
        {
            return;
        }
        physicalAcceleratorWorkers.push_back(
            {DepthComputeBackend::Cuda, device_index});
        selectedPhysicalDeviceIdentities.insert(descriptor.physicalIdentity);
        acceleratorDeviceLeases.push_back(std::move(lease));
    };

    if (_config.patchMatch.cudaDeviceIndex >= 0 &&
        _config.patchMatch.cudaDeviceIndex < cudaDeviceCount)
    {
        try_add_cuda_device(_config.patchMatch.cudaDeviceIndex);
    }
    else if (_config.patchMatch.cudaDeviceIndex < 0)
    {
        for (int device_index = 0; device_index < cudaDeviceCount; ++device_index)
        {
            try_add_cuda_device(device_index);
        }
    }
    const bool cudaAvailable = !physicalAcceleratorWorkers.empty();
    const bool probeOpenCl = configuredBackend == PatchMatchBackend::OpenCl ||
        automaticAcceleration;
    const std::vector<OpenClDeviceInfo> detectedOpenClDevices = probeOpenCl
        ? PatchMatchDepthEstimator::openClDevices()
        : std::vector<OpenClDeviceInfo>{};
    std::vector<OpenClDeviceInfo> selectedOpenClDevices;
    for (const OpenClDeviceInfo &device : detectedOpenClDevices)
    {
        if (_config.patchMatch.openClDeviceIndex >= 0 &&
            device.index != _config.patchMatch.openClDeviceIndex)
        {
            continue;
        }

        const GpuDeviceDescriptor descriptor{
            device.physicalDeviceIdentity,
            device.vendor + " " + device.name};
        if (selectedPhysicalDeviceIdentities.contains(descriptor.physicalIdentity))
        {
            LOG_DEBUG(QStringLiteral(
                          "[MVS] 跳过与已选 CUDA 设备重复的 OpenCL 接口: index=%1 device=%2 identity=%3")
                          .arg(device.index)
                          .arg(QString::fromStdString(device.name))
                          .arg(QString::fromStdString(descriptor.physicalIdentity)));
            continue;
        }
        if (shouldSkipUnstableOpenClCudaAlias(
                device.vendor, descriptor.physicalIdentity, cudaAvailable))
        {
            LOG_WARN(QStringLiteral(
                         "[MVS] 跳过无法取得稳定 PCI 身份的 NVIDIA OpenCL 接口："
                         "index=%1 device=%2；CUDA 已接管该厂商设备，避免重复执行通道")
                         .arg(device.index)
                         .arg(QString::fromStdString(device.name)));
            continue;
        }
        std::unique_ptr<GpuDeviceLeaseSet> lease = acquire_device_lease(descriptor);
        if (!lease)
        {
            continue;
        }

        std::string preparation_error;
        if (!PatchMatchDepthEstimator::prepareOpenClDevice(
                device.index, &preparation_error))
        {
            QString detail = QStringLiteral("OpenCL GPU %1 (%2) 预检失败：%3")
                                 .arg(device.index)
                                 .arg(QString::fromStdString(device.name))
                                 .arg(QString::fromStdString(preparation_error));
            if (detail.size() > 1024)
            {
                detail = detail.left(1021) + QStringLiteral("...");
            }
            acceleratorPreparationFailures.push_back(detail);
            LOG_WARN(QStringLiteral("[MVS] %1").arg(detail));
            continue;
        }
        selectedOpenClDevices.push_back(device);
        physicalAcceleratorWorkers.push_back(
            {DepthComputeBackend::OpenCl, device.index});
        selectedPhysicalDeviceIdentities.insert(descriptor.physicalIdentity);
        acceleratorDeviceLeases.push_back(std::move(lease));
    }
    const std::optional<DepthComputeBackend> requestedBackend =
        configuredBackend == PatchMatchBackend::Auto
            ? std::nullopt
            : std::make_optional(
                  configuredBackend == PatchMatchBackend::Cuda
                      ? DepthComputeBackend::Cuda
                      : configuredBackend == PatchMatchBackend::OpenCl
                          ? DepthComputeBackend::OpenCl
                          : DepthComputeBackend::Cpu);
    const DepthComputeBackend effectiveBackend = resolveDepthComputeBackend(
        requestedBackend,
        cudaAvailable,
        !selectedOpenClDevices.empty(),
        _config.patchMatch.useCuda);
    const bool openClAvailable = !selectedOpenClDevices.empty();
    const bool heterogeneousAuto = configuredBackend == PatchMatchBackend::Auto &&
        automaticAcceleration && cudaAvailable && openClAvailable;
    const bool requestedBackendUnavailable =
        (effectiveBackend == DepthComputeBackend::Cuda && !cudaAvailable) ||
        (effectiveBackend == DepthComputeBackend::OpenCl && !openClAvailable);
    if (requestedBackendUnavailable)
    {
        QString message = QStringLiteral(
            "请求的 %1 深度估计后端不可用或设备编号无效；显式后端不会自动切换到其他设备")
                                    .arg(QString::fromLatin1(
                                        depthComputeBackendName(effectiveBackend)));
        if (!acceleratorPreparationFailures.isEmpty())
        {
            message += QStringLiteral("。%1")
                           .arg(acceleratorPreparationFailures.join(
                               QStringLiteral("；")));
        }
        LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
        emit errorOccurred(message);
        emitFinishedOnce(false);
        return;
    }
    // A heterogeneous Auto batch retains the Auto token in the workspace hash;
    // single-family and explicit batches keep the resolved strict backend. This
    // prevents a CUDA-only resume from silently reusing a CUDA+OpenCL workset.
    _config.patchMatch.backend = heterogeneousAuto
        ? PatchMatchBackend::Auto
        : effectiveBackend == DepthComputeBackend::Cuda
            ? PatchMatchBackend::Cuda
            : effectiveBackend == DepthComputeBackend::OpenCl
                ? PatchMatchBackend::OpenCl
                : PatchMatchBackend::Cpu;
    _config.patchMatch.useCuda = heterogeneousAuto ||
        effectiveBackend == DepthComputeBackend::Cuda;
    _config.patchMatch.cudaFallbackToCpu = false;
    _config.patchMatch.openClFallbackToCpu = false;

    const QString effective_backend_name = QString::fromLatin1(
        depthComputeBackendName(effectiveBackend));
    QString backend_message;
    if (configuredBackend == PatchMatchBackend::Auto && !automaticAcceleration)
    {
        backend_message = QStringLiteral("深度估计后端：兼容 useCuda=false，强制使用 CPU");
    }
    else if (configuredBackend == PatchMatchBackend::Auto)
    {
        backend_message = heterogeneousAuto
            ? QStringLiteral(
                  "深度估计后端：Auto 异构调度已启用 CUDA + OpenCL（逐帧收益调度）")
            : QStringLiteral("深度估计后端：Auto 已选择 %1")
                  .arg(effective_backend_name);
    }
    else
    {
        backend_message = QStringLiteral("深度估计后端：请求并使用 %1")
                              .arg(effective_backend_name);
    }
    if (configuredBackend == PatchMatchBackend::Auto &&
        effectiveBackend == DepthComputeBackend::Cpu &&
        !acceleratorPreparationFailures.isEmpty())
    {
        backend_message += QStringLiteral(
            "；CUDA 租约或 OpenCL 运行时预检不可用，已继续使用 CPU");
    }
    LOG_INFO(QStringLiteral("[MVS] %1").arg(backend_message));
    emit progressChanged(backend_message, 0.0f);

    emit progressChanged(
        QStringLiteral("读取 %1 张影像头部并规划内存...").arg(NV),
        0.0f);
    QString imagePlanningError;
    if (!probeImageMetadata(&imagePlanningError))
    {
        if (!_cancelled.load())
        {
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(imagePlanningError));
            emit errorOccurred(imagePlanningError);
        }
        emitFinishedOnce(false);
        return;
    }

    const SystemMemorySnapshot initialMemory = querySystemMemorySnapshot();
    const DepthMemoryPolicyDecision initialMemoryDecision = evaluateDepthMemoryPolicy(
        _views, _config, _effectiveSceneProfile, initialMemory);
    const uint64_t largestFrameBytes = largestDepthFrameBytes(_views);
    const int plannedMaximumFrameWorkers = std::min(
        std::max(4, static_cast<int>(physicalAcceleratorWorkers.size())),
        std::max(1, NV));
    const int plannedPhysicalCudaWorkers = static_cast<int>(std::count_if(
        physicalAcceleratorWorkers.begin(),
        physicalAcceleratorWorkers.end(),
        [](const DepthComputeWorker &worker)
        {
            return worker.backend == DepthComputeBackend::Cuda;
        }));
    const int plannedPhysicalOpenClWorkers = static_cast<int>(std::count_if(
        physicalAcceleratorWorkers.begin(),
        physicalAcceleratorWorkers.end(),
        [](const DepthComputeWorker &worker)
        {
            return worker.backend == DepthComputeBackend::OpenCl;
        }));
    const int plannedAcceleratorHostSlots =
        (plannedPhysicalCudaWorkers > 0
             ? std::max(plannedPhysicalCudaWorkers, _config.gpuFrameWorkerCount)
             : 0) +
        (plannedPhysicalOpenClWorkers > 0
             ? std::max(plannedPhysicalOpenClWorkers, _config.gpuFrameWorkerCount)
             : 0);
    const int plannedFrameWorkerCount = effectiveBackend == DepthComputeBackend::Cpu
        ? std::clamp(
              std::max(1, _config.cpuFrameWorkerCount),
              1,
              plannedMaximumFrameWorkers)
        : std::clamp(
              plannedAcceleratorHostSlots,
              1,
              plannedMaximumFrameWorkers);
    const std::size_t plannedConcurrentWorkers = static_cast<std::size_t>(
        std::max(plannedFrameWorkerCount,
                 _config.enablePostConsistencyResidualReestimation
                     ? std::min(2, std::max(1, NV))
                     : 1));
    constexpr uint64_t kPlannedMaximumResidentSaveTasks = 6;
    const uint64_t plannedSaveQueueBytes = saturatingMultiplyBytes(
        estimatedSaveQueueProducerBytes(largestFrameBytes),
        kPlannedMaximumResidentSaveTasks);
    const uint64_t plannedBackendStagingBytes = saturatingMultiplyBytes(
        initialMemoryDecision.estimate.transientFrameBytes,
        static_cast<uint64_t>(std::max<std::size_t>(
            2, plannedConcurrentWorkers)));
    const std::vector<MvsImageMemoryFrame> plannedImageFrames =
        imageMemoryFrames(_views);
    const std::size_t plannedRequiredVisibilityPairCount = static_cast<std::size_t>(
        std::count_if(
            _config.sourcePairQualities.begin(),
            _config.sourcePairQualities.end(),
            [](const MvsSourcePairQuality &quality)
            {
                return quality.verified;
            }));
    const MvsVisibilityMemoryEstimate plannedVisibilityMemory =
        estimateMvsVisibilityGraphMemory(
            _views.size(),
            _sparse.points.size(),
            kMvsVisibilityFullPairViewLimit,
            kMvsVisibilityMaximumSampledPeersPerView,
            plannedRequiredVisibilityPairCount);
    const MvsPipelineMemoryPolicyDecision pipelineMemoryDecision =
        decideMvsPipelineMemoryPolicy(
            plannedImageFrames,
            initialMemoryDecision.estimate,
            _config.adaptiveDepthCacheMemory,
            maximumConsistencySourceViews(_config),
            plannedConcurrentWorkers,
            plannedSaveQueueBytes,
            plannedBackendStagingBytes,
            initialMemory.valid ? initialMemory.totalPhysicalBytes : 0,
            initialMemory.valid ? initialMemory.availablePhysicalBytes : 0,
            _config.maxDepthCacheRamFraction,
            _config.minFreeRamBytes,
            plannedVisibilityMemory);
    const std::string_view imageStrategy = mvsImageCacheStrategyName(
        pipelineMemoryDecision.imageStrategy);
    _config.resolvedImageCacheStrategy.assign(
        imageStrategy.data(), imageStrategy.size());
    _config.resolvedImageCacheCapacity = static_cast<int>(
        pipelineMemoryDecision.imageCacheCapacity);

    if (!initializeImageProvider(pipelineMemoryDecision, &imagePlanningError))
    {
        if (_cancelled.load(std::memory_order_relaxed))
        {
            _imageCache.reset();
            clearFrameCaches();
            emitFinishedOnce(false);
            return;
        }
        LOG_ERROR(QStringLiteral("[MVS] %1").arg(imagePlanningError));
        emit errorOccurred(imagePlanningError);
        emitFinishedOnce(false);
        return;
    }
    LOG_INFO(QStringLiteral(
                 "[MVS][内存规划] strategy=%1 capacity=%2/%3 required=%4 GiB "
                 "available=%5 GiB gray=%6 GiB prepared=%7 GiB mask=%8 GiB "
                 "depth=%9 GiB save_queue=%10 GiB backend_staging=%11 GiB "
                 "visibility=%12 GiB visibility_bitset=%13 GiB "
                 "visibility_indices=%14 GiB visibility_pairs=%15 GiB "
                 "visibility_nominated=%16 GiB visibility_adjacency=%17 GiB "
                 "visibility_pair_bound=%18 saturated=%19")
                 .arg(QString::fromLatin1(
                     imageStrategy.data(), static_cast<int>(imageStrategy.size())))
                 .arg(pipelineMemoryDecision.imageCacheCapacity)
                 .arg(NV)
                 .arg(bytesToGiB(pipelineMemoryDecision.requiredBytes), 0, 'f', 2)
                 .arg(bytesToGiB(pipelineMemoryDecision.availableBytes), 0, 'f', 2)
                 .arg(bytesToGiB(pipelineMemoryDecision.estimate.grayBytes), 0, 'f', 2)
                 .arg(bytesToGiB(pipelineMemoryDecision.estimate.preparedBytes), 0, 'f', 2)
                 .arg(bytesToGiB(pipelineMemoryDecision.estimate.maskBytes), 0, 'f', 2)
                 .arg(bytesToGiB(
                     pipelineMemoryDecision.retainAllDepthFrames
                         ? pipelineMemoryDecision.estimate.depthResidentBytes
                         : pipelineMemoryDecision.estimate.depthStreamingBytes),
                      0,
                      'f',
                      2)
                 .arg(bytesToGiB(pipelineMemoryDecision.estimate.saveQueueBytes), 0, 'f', 2)
                 .arg(bytesToGiB(pipelineMemoryDecision.estimate.backendStagingBytes), 0, 'f', 2)
                 .arg(bytesToGiB(
                     pipelineMemoryDecision.estimate.visibility.totalBytes), 0, 'f', 2)
                 .arg(bytesToGiB(
                     pipelineMemoryDecision.estimate.visibility.visibilityBitsetBytes), 0, 'f', 2)
                 .arg(bytesToGiB(
                     pipelineMemoryDecision.estimate.visibility.visibleIndexBytes), 0, 'f', 2)
                 .arg(bytesToGiB(
                     pipelineMemoryDecision.estimate.visibility.pairBytes), 0, 'f', 2)
                 .arg(bytesToGiB(
                     pipelineMemoryDecision.estimate.visibility.nominatedPeerBytes), 0, 'f', 2)
                 .arg(bytesToGiB(
                     pipelineMemoryDecision.estimate.visibility.adjacencyBytes), 0, 'f', 2)
                 .arg(pipelineMemoryDecision.estimate.visibility.candidatePairUpperBound)
                 .arg(pipelineMemoryDecision.estimate.visibility.saturated
                          ? QStringLiteral("true")
                          : QStringLiteral("false")));

    initializeWorkspaceManifest();

    if (pipelineMemoryDecision.imageStrategy == MvsImageCacheStrategy::Eager)
    {
        emit progressChanged(QString("预加载 %1 张图像...").arg(NV), 0.f);
    }
    else
    {
        emit progressChanged(
            QStringLiteral("使用有界影像缓存（容量 %1/%2）")
                .arg(pipelineMemoryDecision.imageCacheCapacity)
                .arg(NV),
            0.0f);
    }
    if (!preloadImages(&imagePlanningError))
    {
        if (!_cancelled.load())
        {
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(imagePlanningError));
            emit errorOccurred(imagePlanningError);
        }
        _imageCache.reset();
        emitFinishedOnce(false);
        return;
    }
    if (_cancelled.load())
    {
        _imageCache.reset();
        clearFrameCaches();
        emitFinishedOnce(false);
        return;
    }

    emit progressChanged(QStringLiteral("预计算 MVS 可见性..."), 0.02f);
    prepareFrameCaches();
    if (_cancelled.load())
    {
        _imageCache.reset();
        clearFrameCaches();
        emitFinishedOnce(false);
        return;
    }

    _depthFrames.resize(NV);
    const bool savePreviewPng = !_outputDir.empty();
    const bool retainDepthFrames =
        pipelineMemoryDecision.retainAllDepthFrames;
    const QString memoryPolicyReason = _config.adaptiveDepthCacheMemory
        ? depthMemoryPolicyReason(initialMemoryDecision, initialMemory)
        : QStringLiteral("adaptiveDepthCacheMemory=false，按配置保留全部深度帧");

    _streamConsistencyStorageEnabled = NV >= 2 && _config.adaptiveDepthCacheMemory;
    if (_streamConsistencyStorageEnabled)
    {
        if (!_config.intermediateDir.empty())
        {
            _consistencyDepthDirectory = _config.intermediateDir;
        }
        else if (!_outputDir.empty())
        {
            _consistencyDepthDirectory = _outputDir;
        }
        else
        {
            const QString cache_name = QStringLiteral("plascan-mvs-%1")
                .arg(_depthConfigHash.left(16));
            _consistencyDepthDirectory = QDir(QDir::tempPath())
                .filePath(cache_name)
                .toStdString();
        }
        if (!QDir().mkpath(QString::fromStdString(_consistencyDepthDirectory)))
        {
            emit errorOccurred(QStringLiteral("无法创建深度一致性缓存目录：%1")
                                   .arg(QString::fromStdString(_consistencyDepthDirectory)));
            _imageCache.reset();
            clearFrameCaches();
            emitFinishedOnce(false);
            return;
        }
    }
    else
    {
        _consistencyDepthDirectory.clear();
    }
    const bool saveRawDepth =
        (_config.saveIntermediateDepthMaps && !_config.intermediateDir.empty()) ||
        _streamConsistencyStorageEnabled;

    if (_config.runFusion && !retainDepthFrames)
    {
        const QString msg = QStringLiteral(
            "系统内存不足以在单次任务中保留全部 full-res 深度图用于内存融合，"
            "本次自动切换为“仅生成/续跑深度图”；完成后请运行“深度图融合”。原因：%1")
                                .arg(memoryPolicyReason);
        LOG_WARN(QStringLiteral("[MVS] %1").arg(msg));
        _config.runFusion = false;
    }

    std::atomic<bool> keepDepthFramesInMemory{retainDepthFrames};
    std::mutex depthFramesMutex;

    LOG_INFO(QStringLiteral(
                 "[MVS] 深度图内存策略: mode=%1 estimatedPeak=%2 GiB total=%3 GiB "
                 "available=%4 GiB reserve=%5 GiB maxFraction=%6 reason=%7")
                 .arg(retainDepthFrames ? QStringLiteral("cache") : QStringLiteral("stream"))
                 .arg(bytesToGiB(initialMemoryDecision.estimate.peakBytes), 0, 'f', 2)
                 .arg(initialMemory.valid ? bytesToGiB(initialMemory.totalPhysicalBytes) : 0.0, 0, 'f', 2)
                 .arg(initialMemory.valid ? bytesToGiB(initialMemory.availablePhysicalBytes) : 0.0, 0, 'f', 2)
                 .arg(bytesToGiB(initialMemoryDecision.reserveBytes), 0, 'f', 2)
                 .arg(static_cast<double>(_config.maxDepthCacheRamFraction), 0, 'f', 2)
                 .arg(memoryPolicyReason));
    if (!keepDepthFramesInMemory.load())
    {
        LOG_INFO(QStringLiteral("[MVS] 深度图估计采用流式保存模式：保存后释放全分辨率深度/置信图"));
    }

    // ── 阶段一：有界三段流水线 ──────────────────────────────────────────────
    // 两个 GPU 主机帧槽并行执行 CPU 准备/后处理；统一内存 OpenCL GPU
    // 最多使用两个执行槽交错驱动空洞，其他物理 GPU 保持单执行槽。
    // 产物由 saveQueue 在第三段异步落盘。
    int skippedFrames = 0;
    for (size_t i = 0; i < _skipFrameMask.size(); ++i)
    {
        if (_skipFrameMask[i] != 0)
        {
            ++skippedFrames;
        }
    }

    std::atomic<int> completedTasks{skippedFrames};
    std::atomic<int> activeCudaTasks{0};
    std::atomic<int> activeOpenClTasks{0};
    std::atomic<int> activeCpuTasks{0};
    std::atomic<bool> anyFailure{false};
    struct FramePriority
    {
        int viewIndex = -1;
        float score = 0.f;
    };

    std::vector<FramePriority> framePriorities;
    framePriorities.reserve(static_cast<size_t>(NV));
    for (int i = 0; i < NV; ++i)
    {
        if (i >= 0 && i < static_cast<int>(_skipFrameMask.size()) && _skipFrameMask[static_cast<size_t>(i)] != 0)
        {
            continue;
        }

        const CameraView &priorityView = _views[static_cast<std::size_t>(i)];
        const float resolutionScore = static_cast<float>(
            static_cast<double>(priorityView.imageWidth) *
            static_cast<double>(priorityView.imageHeight));
        constexpr float contentRatio = 0.5f;

        const int leftNeighbors = i;
        const int rightNeighbors = (NV - 1) - i;
        const int localSupport = std::min(leftNeighbors, rightNeighbors);

        FramePriority priority;
        priority.viewIndex = i;
        priority.score = resolutionScore * (0.70f + 0.30f * contentRatio)
                       + static_cast<float>(localSupport) * 100000.0f;
        framePriorities.push_back(priority);
    }

    std::sort(framePriorities.begin(), framePriorities.end(), [](const FramePriority &a, const FramePriority &b)
    {
        return a.score > b.score;
    });

    std::vector<DepthFrameTask> frameTasks;
    frameTasks.reserve(framePriorities.size());
    for (const FramePriority &priority : framePriorities)
    {
        frameTasks.push_back({priority.viewIndex, priority.score});
    }
    const int pendingFrameCount = static_cast<int>(framePriorities.size());
    const int maxWorkersByPendingFrames = std::max(1, pendingFrameCount);
    const int maxFrameWorkers = std::min(
        std::max(4, static_cast<int>(physicalAcceleratorWorkers.size())),
        maxWorkersByPendingFrames);
    if (effectiveBackend != DepthComputeBackend::Cpu &&
        physicalAcceleratorWorkers.empty() && pendingFrameCount > 0)
    {
        const QString backendName = QString::fromLatin1(
            depthComputeBackendName(effectiveBackend));
        const QString message = QStringLiteral(
            "请求的 %1 深度估计后端不可用或设备编号无效；显式后端不会自动切换到其他设备")
                                    .arg(backendName);
        LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
        emit errorOccurred(message);
        _imageCache.reset();
        clearFrameCaches();
        emitFinishedOnce(false);
        return;
    }

    const int physicalCudaWorkers = static_cast<int>(std::count_if(
        physicalAcceleratorWorkers.begin(),
        physicalAcceleratorWorkers.end(),
        [](const DepthComputeWorker &worker)
        {
            return worker.backend == DepthComputeBackend::Cuda;
        }));
    const int physicalOpenClWorkers = static_cast<int>(std::count_if(
        physicalAcceleratorWorkers.begin(),
        physicalAcceleratorWorkers.end(),
        [](const DepthComputeWorker &worker)
        {
            return worker.backend == DepthComputeBackend::OpenCl;
        }));
    // Every leased physical accelerator is represented first. Additional host
    // slots overlap frame preparation, while the per-device execution locks
    // retain one kernel lane for each physical GPU.
    const int requestedCudaHostSlots = physicalCudaWorkers > 0
        ? std::max(physicalCudaWorkers, _config.gpuFrameWorkerCount)
        : 0;
    const int requestedOpenClHostSlots = physicalOpenClWorkers > 0
        ? std::max(physicalOpenClWorkers, _config.gpuFrameWorkerCount)
        : 0;
    const std::vector<DepthComputeWorker> acceleratorWorkers =
        buildDepthComputeWorkerPool(physicalAcceleratorWorkers,
                                    requestedCudaHostSlots,
                                    requestedOpenClHostSlots,
                                    static_cast<std::size_t>(maxFrameWorkers));
    const bool benefitAwareScheduling = heterogeneousAuto &&
        frameTasks.size() >= physicalAcceleratorWorkers.size();
    // Retain one bounded full-frame OpenCL contribution for large hybrid
    // batches, then return to measured tail-benefit scheduling. Smaller
    // batches keep the cheap coarse-level profitability gate so an asymmetric
    // accelerator cannot create a long tail.
    DepthComputeSchedulingPolicy schedulingPolicy;
    schedulingPolicy.guaranteedOpenClFullFramesPerDevice =
        recommendedOpenClFullFrameFloorPerDevice(
            benefitAwareScheduling,
            frameTasks.size(),
            physicalCudaWorkers,
            physicalOpenClWorkers);
    schedulingPolicy.maximumOpenClInFlightTasksPerDevice =
        benefitAwareScheduling && physicalOpenClWorkers > 0 ? 2 : 0;
    DepthComputeScheduler computeScheduler(
        std::move(frameTasks),
        benefitAwareScheduling,
        acceleratorWorkers,
        schedulingPolicy);
    const int guaranteedOpenClFullFrameTarget =
        physicalOpenClWorkers *
        schedulingPolicy.guaranteedOpenClFullFramesPerDevice;
    std::atomic<int> completedGuaranteedOpenClFullFrames{0};

    const int gpuFrameWorkers = static_cast<int>(acceleratorWorkers.size());
    const int cpuFrameWorkers = effectiveBackend == DepthComputeBackend::Cpu
        ? std::clamp(std::max(1, _config.cpuFrameWorkerCount), 1, maxFrameWorkers)
        : 0;
    const int activeFrameWorkerCount = std::max(
        1, gpuFrameWorkers + cpuFrameWorkers);
    const int totalCpuThreadBudget = runCpuThreadBudget;
    const int minimumCpuThreadsPerWorker = std::max(
        1, totalCpuThreadBudget / activeFrameWorkerCount);
    const int cpuThreadRemainder = std::max(
        0, totalCpuThreadBudget -
            minimumCpuThreadsPerWorker * activeFrameWorkerCount);
    const int maximumCpuThreadsPerWorker = minimumCpuThreadsPerWorker +
        (cpuThreadRemainder > 0 ? 1 : 0);

    const QString scheduling_backend_name = heterogeneousAuto
        ? QStringLiteral("Hybrid(CUDA+OpenCL)")
        : QString::fromLatin1(depthComputeBackendName(effectiveBackend));
    LOG_INFO(QStringLiteral("[MVS] 深度估计调度: backend=%1 cuda_devices=%2 opencl_devices=%3 "
                            "physical_gpu_workers=%4 gpu_host_slots=%5 "
                            "cuda_host_slots=%6 opencl_host_slots=%7 cpu_frame_workers=%8 "
                            "cpu_thread_budget=%9 cpu_pixel_threads=%10..%11 views=%12 "
                            "pending=%13 benefit_aware=%14 opencl_full_frame_floor=%15 "
                            "opencl_inflight_limit=%16")
                 .arg(scheduling_backend_name)
                 .arg(cudaDeviceCount)
                 .arg(selectedOpenClDevices.size())
                 .arg(physicalAcceleratorWorkers.size())
                 .arg(gpuFrameWorkers)
                 .arg(requestedCudaHostSlots)
                 .arg(requestedOpenClHostSlots)
                 .arg(cpuFrameWorkers)
                 .arg(totalCpuThreadBudget)
                 .arg(minimumCpuThreadsPerWorker)
                 .arg(maximumCpuThreadsPerWorker)
                 .arg(NV)
                 .arg(static_cast<qulonglong>(computeScheduler.pendingTaskCount()))
                 .arg(benefitAwareScheduling ? 1 : 0)
                 .arg(guaranteedOpenClFullFrameTarget)
                 .arg(schedulingPolicy.maximumOpenClInFlightTasksPerDevice));
    if (skippedFrames > 0)
    {
        LOG_INFO(QStringLiteral("[MVS] 续跑模式：跳过已存在深度图 %1 帧").arg(skippedFrames));
    }
    if (cudaAvailable)
    {
        LOG_DEBUG(QStringLiteral("[MVS] CUDA 已启用，设备=%1，CUDA 主机帧槽=%2；"
                                 "每个 CUDA 设备使用独立工作区和执行槽")
                      .arg(physicalCudaWorkers)
                      .arg(requestedCudaHostSlots));
    }
    for (const OpenClDeviceInfo &device : selectedOpenClDevices)
    {
        LOG_INFO(QStringLiteral("[MVS] OpenCL GPU 已启用: index=%1 vendor=%2 device=%3 "
                                "compute_units=%4 memory=%5 MiB")
                     .arg(device.index)
                     .arg(QString::fromStdString(device.vendor))
                     .arg(QString::fromStdString(device.name))
                     .arg(device.computeUnits)
                     .arg(device.globalMemoryBytes / (1024ULL * 1024ULL)));
    }
    if (physicalOpenClWorkers > 0)
    {
        PatchMatchDepthEstimator::resetOpenClExecutionStats();
    }

    const int physicalGpuCount = static_cast<int>(physicalAcceleratorWorkers.size());
    const size_t saveWorkerCount = !retainDepthFrames && physicalGpuCount >= 2
        ? 2
        : 1;
    const size_t transientReserveFrameWorkers = std::max<size_t>(
        2, static_cast<size_t>(activeFrameWorkerCount));
    const size_t maxBufferedSaveTasks =
        adaptiveSaveQueueCapacity(
            initialMemory,
            _config,
            largestFrameBytes,
            initialMemoryDecision.estimate.transientFrameBytes,
            transientReserveFrameWorkers);
    const size_t maxResidentSaveTasks = maxBufferedSaveTasks + saveWorkerCount;
    const uint64_t maxResidentSaveBytes = adaptiveSaveQueueResidentByteCapacity(
        initialMemory,
        _config,
        largestFrameBytes,
        maxResidentSaveTasks,
        initialMemoryDecision.estimate.transientFrameBytes,
        transientReserveFrameWorkers);
    const uint64_t producerSaveReservationBytes =
        estimatedSaveQueueProducerBytes(largestFrameBytes);
    const uint64_t concurrentTransientReserveBytes = saturatingMultiplyBytes(
        initialMemoryDecision.estimate.transientFrameBytes,
        static_cast<uint64_t>(transientReserveFrameWorkers));
    if (maxResidentSaveBytes < producerSaveReservationBytes)
    {
        const QString message = QStringLiteral(
            "系统可用内存不足以安全启动深度估计：保存队列预算=%1 GiB，"
            "单帧最低预约=%2 GiB，并发临时内存预留=%3 GiB（%4 个主机帧槽）。"
            "请关闭其他占用内存的程序或降低输入分辨率/质量后重试")
                                    .arg(bytesToGiB(maxResidentSaveBytes), 0, 'f', 2)
                                    .arg(bytesToGiB(producerSaveReservationBytes), 0, 'f', 2)
                                    .arg(bytesToGiB(concurrentTransientReserveBytes), 0, 'f', 2)
                                    .arg(transientReserveFrameWorkers);
        LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
        emit errorOccurred(message);
        _imageCache.reset();
        clearFrameCaches();
        emitFinishedOnce(false);
        return;
    }
    LOG_INFO(QStringLiteral(
                 "[MVS] 深度产物保存队列配置: workers=%1 task_limit=%2 "
                 "resident_limit=%3 GiB producer_reservation=%4 GiB "
                 "transient_reserve=%5 GiB transient_slots=%6")
                 .arg(saveWorkerCount)
                 .arg(maxResidentSaveTasks)
                 .arg(bytesToGiB(maxResidentSaveBytes), 0, 'f', 2)
                 .arg(bytesToGiB(producerSaveReservationBytes), 0, 'f', 2)
                 .arg(bytesToGiB(concurrentTransientReserveBytes), 0, 'f', 2)
                 .arg(transientReserveFrameWorkers));
    DepthFrameArtifactSaveQueue saveQueue(
        [this](int frameIndex, const DepthFrameResult &result, const QString &stageLabel)
        {
            return saveDepthFrameArtifacts(frameIndex, result, stageLabel);
        },
        saveWorkerCount,
        maxResidentSaveTasks,
        maxResidentSaveBytes,
        producerSaveReservationBytes);

    auto emitDepthProgress =
        [this,
         NV,
         physicalGpuCount,
         &completedTasks,
         &activeCudaTasks,
         &activeOpenClTasks,
         &activeCpuTasks,
         guaranteedOpenClFullFrameTarget,
         &completedGuaranteedOpenClFullFrames,
         &computeScheduler](
            const QString& workerTag, int frameIndex, bool pickedTask)
    {
        const int done = completedTasks.load();
        const int cudaActive = activeCudaTasks.load();
        const int openClActive = activeOpenClTasks.load();
        const int gpuActive = cudaActive + openClActive;
        const int cpuActive = activeCpuTasks.load();
        const int pending = static_cast<int>(computeScheduler.pendingTaskCount());
        const float ratio = static_cast<float>(done) / (NV + 2);

        QString stage = QStringLiteral(
                            "深度估计: 已完成 %1/%2, 物理 GPU %3, GPU主机帧 %4 "
                            "(CUDA %5/OpenCL %6), CPU %7, 待处理 %8")
                            .arg(done)
                            .arg(NV)
                            .arg(physicalGpuCount)
                            .arg(gpuActive)
                            .arg(cudaActive)
                            .arg(openClActive)
                            .arg(cpuActive)
                            .arg(pending);
        if (guaranteedOpenClFullFrameTarget > 0)
        {
            stage += QStringLiteral(", OpenCL完整帧 %1/%2")
                         .arg(completedGuaranteedOpenClFullFrames.load())
                         .arg(guaranteedOpenClFullFrameTarget);
        }

        if (pickedTask && frameIndex >= 0)
        {
            stage += QStringLiteral(", 当前启动帧 %1 [%2]")
                         .arg(frameIndex + 1)
                         .arg(workerTag);
        }
        else if (frameIndex >= 0)
        {
            stage += QStringLiteral(", 最新完成帧 %1 [%2]")
                         .arg(frameIndex + 1)
                         .arg(workerTag);
        }

        emit progressChanged(stage, ratio);
    };

    std::atomic<bool> workerExceptionReported{false};
    auto workerFunc = [this,
                       NV,
                       benefitAwareScheduling,
                       initialMemoryDecision,
                       &keepDepthFramesInMemory,
                       &depthFramesMutex,
                       &completedTasks,
                       &activeCudaTasks,
                       &activeOpenClTasks,
                       &activeCpuTasks,
                       guaranteedOpenClFullFrameTarget,
                       &completedGuaranteedOpenClFullFrames,
                       &anyFailure,
                       &computeScheduler,
                       &emitDepthProgress,
                       &saveQueue,
                       &workerExceptionReported](DepthComputeWorker worker, int assignedCpuThreadCount)
    {
#ifdef _OPENMP
        // Each host slot is a separate std::thread. Set its OpenMP ICV so
        // implicit post-processing regions do not each expand to all CPUs.
        omp_set_num_threads(std::max(1, assignedCpuThreadCount));
#endif
        const bool useGpu = worker.backend != DepthComputeBackend::Cpu;
        DepthGenConfig workerConfig = _config;
        workerConfig.cpuWorkerCount = std::max(1, assignedCpuThreadCount);
        workerConfig.patchMatch.backend = worker.backend == DepthComputeBackend::Cuda     ? PatchMatchBackend::Cuda
                                          : worker.backend == DepthComputeBackend::OpenCl ? PatchMatchBackend::OpenCl
                                                                                          : PatchMatchBackend::Cpu;
        workerConfig.patchMatch.useCuda = worker.backend == DepthComputeBackend::Cuda;
        workerConfig.patchMatch.cudaDeviceIndex = worker.backend == DepthComputeBackend::Cuda ? worker.deviceIndex : -1;
        workerConfig.patchMatch.openClDeviceIndex =
            worker.backend == DepthComputeBackend::OpenCl ? worker.deviceIndex : -1;
        workerConfig.patchMatch.cudaFallbackToCpu = false;
        workerConfig.patchMatch.openClFallbackToCpu = false;
        workerConfig.patchMatch.cancelFlag = &_cancelled;
        const QString workerTag = QString::fromStdString(worker.id());
        int claimedViewIndex = -1;
        bool completionReported = false;
        bool activeTask = false;
        auto claimedFrameStart = std::chrono::steady_clock::now();
        const auto adjust_active_task_count = [&](int delta)
        {
            if (!useGpu)
            {
                activeCpuTasks.fetch_add(delta);
                return;
            }
            if (worker.backend == DepthComputeBackend::Cuda)
            {
                activeCudaTasks.fetch_add(delta);
            }
            else if (worker.backend == DepthComputeBackend::OpenCl)
            {
                activeOpenClTasks.fetch_add(delta);
            }
        };

        const auto run_worker = [&]()
        {
            while (!_cancelled)
            {
                const DepthTaskClaim claim = computeScheduler.claimNext(worker);
                if (claim.status == DepthTaskClaimStatus::Retry)
                {
                    computeScheduler.waitForStateChange(claim.revision, std::chrono::milliseconds(25));
                    continue;
                }
                if (claim.status != DepthTaskClaimStatus::Task)
                {
                    break;
                }
                const int i = claim.viewIndex;
                if (i < 0 || i >= NV)
                {
                    const DepthTaskCompletionResult completion =
                        computeScheduler.complete(worker, i, std::chrono::milliseconds(0), false);
                    if (!completion.retryScheduled)
                    {
                        anyFailure = true;
                    }
                    continue;
                }

                claimedViewIndex = i;
                completionReported = false;

                DepthFrameArtifactSaveQueue::ProducerReservation saveReservation =
                    saveQueue.reserveProducer(&_cancelled);
                if (!saveReservation || _cancelled.load())
                {
                    break;
                }
                claimedFrameStart = std::chrono::steady_clock::now();

                adjust_active_task_count(1);
                activeTask = true;
                emitDepthProgress(workerTag, i, true);

                const auto frameStart = claimedFrameStart;
                LOG_DEBUG(QStringLiteral("[MVS][深度估计] 帧 %1/%2 开始: id=%3 device=%4")
                              .arg(i + 1)
                              .arg(NV)
                              .arg(i)
                              .arg(workerTag));
                if (claim.requiresFullFrame)
                {
                    LOG_INFO(QStringLiteral(
                                 "[MVS][深度估计] %1 已领取保留的 OpenCL 完整帧 %2/%3；"
                                 "该帧不会在粗层校准后回迁")
                                 .arg(workerTag)
                                 .arg(i + 1)
                                 .arg(NV));
                }
                markManifestFrameRunning(i);

                std::function<bool(const DepthLevelSummary &, std::string *)>
                    first_level_completion_gate;
                bool calibration_probe_requeued = false;
                if (benefitAwareScheduling && claim.calibrationProbe &&
                    !claim.requiresFullFrame)
                {
                    first_level_completion_gate =
                        [&computeScheduler,
                         worker,
                         i,
                         frameStart,
                         &completionReported,
                         &calibration_probe_requeued](
                            const DepthLevelSummary &summary,
                            std::string *error_message)
                    {
                        const std::optional<double> alternative_elapsed =
                            computeScheduler.tryRejectUnprofitableCalibrationProbe(
                                worker,
                                i,
                                summary.elapsedMs,
                                std::chrono::steady_clock::now() - frameStart);
                        if (!alternative_elapsed.has_value())
                        {
                            return true;
                        }

                        // Scheduler ownership has already moved atomically to
                        // the retry queue. Mark it before formatting diagnostics
                        // so an exception cannot report this completion twice.
                        calibration_probe_requeued = true;
                        completionReported = true;

                        std::ostringstream message;
                        message << "calibration_probe_unprofitable: "
                                << depthComputeBackendName(worker.backend)
                                << " coarse level " << summary.elapsedMs
                                << " ms >= alternative full frame "
                                << *alternative_elapsed << " ms";
                        if (error_message)
                        {
                            *error_message = message.str();
                        }
                        LOG_WARN(QStringLiteral(
                                     "[MVS][深度估计] %1 粗层校准耗时 %2 ms，"
                                     "已不低于其他后端完整帧 %3 ms；停止更高分辨率层并回迁该帧")
                                     .arg(QString::fromStdString(worker.id()))
                                     .arg(summary.elapsedMs, 0, 'f', 1)
                                     .arg(*alternative_elapsed, 0, 'f', 1));
                        return false;
                    };
                }
                DepthFrameResult res = computeDepthForView(
                    i, &workerConfig, first_level_completion_gate);
                if (_cancelled.load())
                {
                    LOG_INFO(QStringLiteral("[MVS] 帧 %1 收到取消请求，跳过结果保存").arg(i));
                    adjust_active_task_count(-1);
                    activeTask = false;
                    emitDepthProgress(workerTag, i, false);
                    break;
                }

                const auto frameEnd = std::chrono::steady_clock::now();
                const double elapsedMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
                if (calibration_probe_requeued)
                {
                    res.device = worker.id();
                    res.elapsedMs = elapsedMs;
                    adjust_active_task_count(-1);
                    activeTask = false;
                    claimedViewIndex = -1;
                    emitDepthProgress(workerTag, i, false);
                    continue;
                }
                const DepthTaskCompletionResult completion =
                    computeScheduler.complete(worker, i, frameEnd - frameStart, res.success);
                completionReported = completion.accepted;
                res.device = worker.id();
                res.elapsedMs = elapsedMs;

                if (!completion.accepted)
                {
                    res.success = false;
                    res.errorMsg = "depth scheduler rejected frame completion ownership";
                }
                else if (res.success && claim.requiresFullFrame &&
                         worker.backend == DepthComputeBackend::OpenCl)
                {
                    const int completed_opencl_full_frames =
                        completedGuaranteedOpenClFullFrames.fetch_add(1) + 1;
                    LOG_INFO(QStringLiteral(
                                 "[MVS][深度估计] OpenCL 保留完整帧已完成: %1/%2 "
                                 "device=%3 frame=%4 elapsed=%5 ms")
                                 .arg(completed_opencl_full_frames)
                                 .arg(guaranteedOpenClFullFrameTarget)
                                 .arg(workerTag)
                                 .arg(i + 1)
                                 .arg(elapsedMs, 0, 'f', 1));
                }

                if (!res.success && completion.retryScheduled)
                {
                    LOG_WARN(QStringLiteral("[MVS][深度估计] 帧 %1/%2 在 %3 失败，将由其他后端限次重试: "
                                            "id=%4 elapsed=%5 ms error=%6")
                                 .arg(i + 1)
                                 .arg(NV)
                                 .arg(workerTag)
                                 .arg(i)
                                 .arg(elapsedMs, 0, 'f', 1)
                                 .arg(QString::fromStdString(res.errorMsg)));
                    adjust_active_task_count(-1);
                    activeTask = false;
                    claimedViewIndex = -1;
                    emitDepthProgress(workerTag, i, false);
                    continue;
                }

                DepthFrameResult storedResult = res;
                if (!keepDepthFramesInMemory.load())
                {
                    storedResult.releasePixelStorage();
                }
                {
                    std::lock_guard<std::mutex> lock(depthFramesMutex);
                    _depthFrames[i] = storedResult;
                }

                if (!res.success)
                {
                    LOG_WARN(QStringLiteral("[MVS][深度估计] 帧 %1/%2 失败: id=%3 device=%4 elapsed=%5 ms error=%6")
                                 .arg(i + 1)
                                 .arg(NV)
                                 .arg(i)
                                 .arg(workerTag)
                                 .arg(elapsedMs, 0, 'f', 1)
                                 .arg(QString::fromStdString(res.errorMsg)));
                    markManifestFrameFailed(i, QString::fromStdString(res.errorMsg));
                    anyFailure = true;
                }
                else
                {
                    const int depthWidth = (res.depthMap && !res.depthMap->empty()) ? res.depthMap->cols : 0;
                    const int depthHeight = (res.depthMap && !res.depthMap->empty()) ? res.depthMap->rows : 0;
                    LOG_INFO(QStringLiteral("[MVS][深度估计] 帧 %1/%2 完成: id=%3 device=%4 size=%5x%6 "
                                            "coverage=%7 confidence=%8 acceptance=%9 elapsed=%10 ms")
                                 .arg(i + 1)
                                 .arg(NV)
                                 .arg(i)
                                 .arg(workerTag)
                                 .arg(depthWidth)
                                 .arg(depthHeight)
                                 .arg(res.qualityMetrics.validCoverage, 0, 'f', 3)
                                 .arg(res.qualityMetrics.meanConfidence, 0, 'f', 3)
                                 .arg(QString::fromLatin1(depthFrameAcceptanceId(res.qualityDecision.acceptance)))
                                 .arg(elapsedMs, 0, 'f', 1));
                    saveQueue.enqueue(std::move(saveReservation), i, res, QStringLiteral("初始"));

                    if (keepDepthFramesInMemory.load() &&
                        memoryPressureRequiresStreaming(_config, querySystemMemorySnapshot(), initialMemoryDecision))
                    {
                        bool expected = true;
                        if (keepDepthFramesInMemory.compare_exchange_strong(expected, false))
                        {
                            LOG_WARN(QStringLiteral("[MVS] 内存压力升高，切换为流式保存并释放已缓存深度图；"
                                                    "后续使用有界 LRU 继续多视一致性检查"));
                            std::lock_guard<std::mutex> lock(depthFramesMutex);
                            releaseStoredDepthFramePixelStorage(_depthFrames);
                        }
                    }
                }

                emit depthMapReady(res);
                if (!keepDepthFramesInMemory.load())
                {
                    res.releasePixelStorage();
                }
                const int done = completedTasks.fetch_add(1) + 1;
                adjust_active_task_count(-1);
                activeTask = false;
                claimedViewIndex = -1;
                Q_UNUSED(done);
                emitDepthProgress(workerTag, i, false);
            }
        };

        try
        {
            run_worker();
        }
        catch (const std::exception &exception)
        {
            if (activeTask)
            {
                adjust_active_task_count(-1);
                activeTask = false;
            }
            if (claimedViewIndex >= 0 && !completionReported)
            {
                try
                {
                    computeScheduler.complete(
                        worker, claimedViewIndex, std::chrono::steady_clock::now() - claimedFrameStart, false);
                }
                catch (...)
                {
                    // The whole batch is cancelled below; never let an error
                    // while repairing scheduler state escape a std::thread.
                }
            }
            anyFailure = true;
            _cancelled = true;
            saveQueue.cancel();
            if (!workerExceptionReported.exchange(true))
            {
                const QString message = QStringLiteral("MVS 深度计算线程异常终止（%1，帧 %2）：%3")
                                            .arg(workerTag)
                                            .arg(claimedViewIndex)
                                            .arg(QString::fromUtf8(exception.what()));
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                emit errorOccurred(message);
            }
        }
        catch (...)
        {
            if (activeTask)
            {
                adjust_active_task_count(-1);
                activeTask = false;
            }
            if (claimedViewIndex >= 0 && !completionReported)
            {
                try
                {
                    computeScheduler.complete(
                        worker, claimedViewIndex, std::chrono::steady_clock::now() - claimedFrameStart, false);
                }
                catch (...)
                {
                }
            }
            anyFailure = true;
            _cancelled = true;
            saveQueue.cancel();
            if (!workerExceptionReported.exchange(true))
            {
                const QString message = QStringLiteral("MVS 深度计算线程异常终止（%1，帧 %2）：未知异常")
                                            .arg(workerTag)
                                            .arg(claimedViewIndex);
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                emit errorOccurred(message);
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(cpuFrameWorkers + gpuFrameWorkers));

    int workerThreadIndex = 0;
    const auto assigned_cpu_threads = [&](int workerIndex)
    { return minimumCpuThreadsPerWorker + (workerIndex < cpuThreadRemainder ? 1 : 0); };
    try
    {
        for (const DepthComputeWorker &worker : acceleratorWorkers)
        {
            workers.emplace_back(workerFunc, worker, assigned_cpu_threads(workerThreadIndex++));
        }
        for (int workerIndex = 0; workerIndex < cpuFrameWorkers; ++workerIndex)
        {
            workers.emplace_back(workerFunc,
                                 DepthComputeWorker{DepthComputeBackend::Cpu, workerIndex},
                                 assigned_cpu_threads(workerThreadIndex++));
        }
    }
    catch (...)
    {
        _cancelled = true;
        saveQueue.cancel();
        for (std::thread &worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        throw;
    }

    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    for (const auto &[worker_id, stats] : computeScheduler.workerStats())
    {
        LOG_INFO(QStringLiteral(
                     "[MVS] 深度调度统计: worker=%1 completed=%2 successful=%3 "
                     "failed=%4 elapsed=%5 ms ema=%6 ms")
                     .arg(QString::fromStdString(worker_id))
                     .arg(stats.completedTasks)
                     .arg(stats.successfulTasks)
                     .arg(stats.failedTasks)
                     .arg(stats.elapsedMilliseconds, 0, 'f', 1)
                     .arg(stats.emaElapsedMilliseconds, 0, 'f', 1));
    }
    if (guaranteedOpenClFullFrameTarget > 0)
    {
        LOG_INFO(QStringLiteral(
                     "[MVS] OpenCL 完整帧贡献: completed=%1 target=%2")
                     .arg(completedGuaranteedOpenClFullFrames.load())
                     .arg(guaranteedOpenClFullFrameTarget));
    }

    if (_cancelled.load())
    {
        saveQueue.cancel();
        saveQueue.stop();
        _imageCache.reset();
        clearFrameCaches();
        emitFinishedOnce(false);
        return;
    }

    if (!saveQueue.waitUntilIdle(&_cancelled))
    {
        saveQueue.cancel();
        saveQueue.stop();
        _imageCache.reset();
        clearFrameCaches();
        emitFinishedOnce(false);
        return;
    }
    if (saveQueue.failed())
    {
        anyFailure = true;
    }

    if (keepDepthFramesInMemory.load() && _config.adaptiveDepthCacheMemory)
    {
        const SystemMemorySnapshot consistencyMemory = querySystemMemorySnapshot();
        const DepthMemoryPolicyDecision consistencyDecision = evaluateDepthMemoryPolicy(
            _views, _config, _effectiveSceneProfile, consistencyMemory);
        if (!consistencyDecision.retainAllFrames ||
            memoryPressureRequiresStreaming(
                _config, consistencyMemory, consistencyDecision))
        {
            LOG_WARN(QStringLiteral(
                         "[MVS] 进入多视一致性检查前检测到内存预算不足，"
                         "切换为有界流式检查并释放常驻深度帧：%1")
                         .arg(depthMemoryPolicyReason(
                             consistencyDecision, consistencyMemory)));
            keepDepthFramesInMemory = false;
            std::lock_guard<std::mutex> lock(depthFramesMutex);
            releaseStoredDepthFramePixelStorage(_depthFrames);
        }
    }

    clearFrameCaches();

    // ── 阶段 1.5：双视图深度图左右一致性检查 ────────────────────────────────
    // 在融合前剔除 PatchMatch 产生的幽灵深度（两视图深度互不一致的像素）
    if (keepDepthFramesInMemory.load() && NV >= 2)
    {
        emit progressChanged("深度一致性检查...", static_cast<float>(NV) / (NV + 3));
        crossCheckDepthConsistency();
        if (_cancelled.load())
        {
            saveQueue.cancel();
            saveQueue.stop();
            _imageCache.reset();
            emitFinishedOnce(false);
            return;
        }
    }

    else if (!keepDepthFramesInMemory.load() && NV >= 2)
    {
        emit progressChanged(QStringLiteral("准备流式深度一致性检查..."),
                             static_cast<float>(NV) / (NV + 3));
        if (!crossCheckDepthConsistencyStreaming())
        {
            saveQueue.cancel();
            saveQueue.stop();
            _imageCache.reset();
            emitFinishedOnce(false);
            return;
        }
    }

    // Experimental, default-off stage. It only emits derived camera candidates
    // and diagnostics; neither project cameras nor this run's depth maps change.
    runDepthPoseRefinementCandidateStage(keepDepthFramesInMemory.load());

    if (keepDepthFramesInMemory.load() && (savePreviewPng || saveRawDepth))
    {
        for (int i = 0; i < NV; ++i)
        {
            if (_cancelled.load())
            {
                saveQueue.cancel();
                saveQueue.stop();
                _imageCache.reset();
                emitFinishedOnce(false);
                return;
            }
            DepthFrameResult &res = _depthFrames[i];
            if (res.eligibleForFusion() &&
                res.depthMap &&
                !res.depthMap->empty() &&
                !res.depthPostprocessApplied)
            {
                cv::Mat emptyConfidence;
                cv::Mat &confidence =
                    (res.confidence && !res.confidence->empty()) ? *res.confidence : emptyConfidence;
                FusionConfig fusion_config = _config.fusion;
                const DepthFilterSettings filter_settings = depthFilterSettings(
                    _effectiveDepthFilterMode,
                    static_cast<int>(res.sourceViewIndices.size()));
                fusion_config.localDepthOutlierRelThresh =
                    filter_settings.localDepthOutlierRelThreshold;
                fusion_config.minSpeckleComponentArea = filter_settings.minComponentArea;
                fusion_config.minConsistentViews = filter_settings.minConsistentViews;
                fusion_config.confidenceThresh = depthConfidenceThresholds(
                    _effectiveSceneProfile,
                    _effectiveDepthFilterMode,
                    static_cast<int>(res.sourceViewIndices.size()),
                    _config.patchMatch.confidenceThresh,
                    fusion_config.confidenceThresh).fusion;
                const DepthPostProcessEvidence postprocess_evidence =
                    depthPostProcessEvidence(res);
                res.depthPostprocess = postprocessFusionDepthMap(*res.depthMap,
                                                                  confidence,
                                                                  fusion_config,
                                                                  res.refViewIdx,
                                                                  static_cast<int>(_views.size()),
                                                                  res.missingReasonMap.data(),
                                                                  &postprocess_evidence);
                res.depthPostprocessApplied = true;
                cv::Mat final_interpolation_mask;
                const DepthAnchoredHoleInterpolationStats final_repair =
                    repairPostprocessedInternalDepthHoles(
                        res,
                        *res.depthMap,
                        confidence,
                        _effectiveSceneProfile,
                        &final_interpolation_mask);
                if (!res.depthProvenance || res.depthProvenance->empty())
                {
                    res.depthProvenance = QSharedPointer<cv::Mat>::create(
                        initializeDepthProvenance(
                            *res.depthMap,
                            res.targetedGapRecoveredMask
                                ? *res.targetedGapRecoveredMask : cv::Mat()));
                }
                updateDepthProvenance(
                    *res.depthProvenance,
                    *res.depthMap,
                    res.targetedGapRecoveredMask
                        ? *res.targetedGapRecoveredMask : cv::Mat(),
                    res.crossViewRepairedMask
                        ? *res.crossViewRepairedMask : cv::Mat(),
                    final_interpolation_mask);
                res.crossViewRepairDiagnostics.insert(
                    QStringLiteral("postprocess_anchored_interpolation"),
                    depthAnchoredHoleInterpolationStatsToJson(final_repair));
                res.depthCompleteness.crossViewRepairedCount +=
                    static_cast<int>(final_repair.interpolatedPixelCount);
                updateDepthCompletenessAfterPostprocess(
                    res, *res.depthMap, res.depthPostprocess);
                const float consistency_keep_rate =
                    res.depthCompleteness.consistencyRetentionRatio;
                updateDepthFrameQualityAfterConsistency(
                    res,
                    *res.depthMap,
                    confidence,
                    _effectiveSceneProfile,
                    _effectiveDepthFilterMode,
                    consistency_keep_rate);
                if (res.missingReasonMap && res.supportRegionMask)
                {
                    finalizeDepthMissingReasonMap(
                        *res.missingReasonMap,
                        *res.depthMap,
                        *res.supportRegionMask,
                        res.geometrySupportCount ? *res.geometrySupportCount : cv::Mat());
                }
            }
            if (_cancelled.load())
            {
                saveQueue.cancel();
                saveQueue.stop();
                _imageCache.reset();
                emitFinishedOnce(false);
                return;
            }
        }

        if (_config.enablePostConsistencyResidualReestimation &&
            _effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            emit progressChanged(
                QStringLiteral("最终残余空洞局部重估..."),
                static_cast<float>(NV + 1) / (NV + 3));
            recoverResidualDepthAfterConsistency();
            if (_cancelled.load())
            {
                saveQueue.cancel();
                saveQueue.stop();
                _imageCache.reset();
                emitFinishedOnce(false);
                return;
            }
        }

        for (int i = 0; i < NV; ++i)
        {
            if (_cancelled.load())
            {
                break;
            }
            DepthFrameArtifactSaveQueue::ProducerReservation saveReservation =
                saveQueue.reserveProducer(&_cancelled);
            if (!saveReservation)
            {
                if (!_cancelled.load())
                {
                    anyFailure = true;
                }
                break;
            }
            DepthFrameResult &res = _depthFrames[i];
            if (res.success && res.depthMap &&
                !res.depthMap->empty())
            {
                const int valid_after_recovery = cv::countNonZero(
                    *res.depthMap > 0.0f);
                res.depthPostprocess.validAfterPostprocess =
                    valid_after_recovery;
                res.validMask = QSharedPointer<cv::Mat>::create(
                    *res.depthMap > 0.0f);
                updateDepthCompletenessAfterPostprocess(
                    res, *res.depthMap, res.depthPostprocess);
                const float consistency_keep_rate =
                    res.depthCompleteness.consistencyRetentionRatio;
                const cv::Mat empty_confidence;
                const cv::Mat &confidence = res.confidence
                    ? *res.confidence
                    : empty_confidence;
                const DepthFrameQualityDecision frozen_quality_decision =
                    res.qualityDecision;
                updateDepthFrameQualityAfterConsistency(
                    res,
                    *res.depthMap,
                    confidence,
                    _effectiveSceneProfile,
                    _effectiveDepthFilterMode,
                    consistency_keep_rate);
                res.qualityDecision = frozen_quality_decision;
                if (res.missingReasonMap && res.supportRegionMask)
                {
                    finalizeDepthMissingReasonMap(
                        *res.missingReasonMap,
                        *res.depthMap,
                        *res.supportRegionMask,
                        res.geometrySupportCount
                            ? *res.geometrySupportCount : cv::Mat());
                }
            }
            if (_cancelled.load())
            {
                break;
            }
            saveQueue.enqueue(
                std::move(saveReservation), i, res, QStringLiteral("过滤后"));
        }
        if (_cancelled.load() || !saveQueue.waitUntilIdle(&_cancelled))
        {
            saveQueue.cancel();
            saveQueue.stop();
            _imageCache.reset();
            emitFinishedOnce(false);
            return;
        }
        if (saveQueue.failed())
        {
            anyFailure = true;
        }
    }
    _imageCache.reset();
    saveQueue.stop();

    if (_cancelled.load())
    {
        emitFinishedOnce(false);
        return;
    }

    allOk = !anyFailure.load() && !_cancelled.load();
    int successfulFrames = 0;
    int failedFrames = 0;
    int fusionEligibleFrames = 0;
    double coverageSum = 0.0;
    int coverageCount = 0;
    for (int frameIndex = 0; frameIndex < NV; ++frameIndex)
    {
        const bool skipped = frameIndex < static_cast<int>(_skipFrameMask.size()) &&
                             _skipFrameMask[static_cast<size_t>(frameIndex)] != 0;
        if (skipped)
        {
            continue;
        }

        const DepthFrameResult &frame = _depthFrames[static_cast<size_t>(frameIndex)];
        if (frame.success)
        {
            ++successfulFrames;
            coverageSum += frame.qualityMetrics.validCoverage;
            ++coverageCount;
            if (frame.eligibleForFusion())
            {
                ++fusionEligibleFrames;
            }
        }
        else
        {
            ++failedFrames;
        }
    }
    for (const OpenClExecutionStats &stats :
         PatchMatchDepthEstimator::openClExecutionStats())
    {
        LOG_INFO(QStringLiteral(
                     "[MVS][OpenCL][批次利用率] device=%1 calls=%2 wall=%3 ms "
                     "queue=%4 ms inter_call_idle=%5 ms queue_non_kernel=%6 ms "
                     "queue_occupancy=%7% kernel_active=%8 ms "
                     "end_to_end_kernel_duty=%9%")
                     .arg(stats.deviceIndex)
                     .arg(static_cast<qulonglong>(stats.callCount))
                     .arg(stats.wallSpanMilliseconds, 0, 'f', 1)
                     .arg(stats.queueMilliseconds, 0, 'f', 1)
                     .arg(stats.interCallIdleMilliseconds, 0, 'f', 1)
                     .arg(stats.queueNonKernelMilliseconds, 0, 'f', 1)
                     .arg(stats.queueOccupancyRatio * 100.0, 0, 'f', 1)
                     .arg(stats.kernelActiveMilliseconds, 0, 'f', 1)
                     .arg(stats.kernelDutyRatio * 100.0, 0, 'f', 1));
    }
    LOG_INFO(QStringLiteral(
                 "[MVS][深度估计] 批次完成: success=%1 failed=%2 skipped=%3 fusion_eligible=%4/%5 "
                 "mean_coverage=%6 elapsed=%7 ms")
                 .arg(successfulFrames)
                 .arg(failedFrames)
                 .arg(skippedFrames)
                 .arg(fusionEligibleFrames)
                 .arg(successfulFrames)
                 .arg(coverageCount > 0 ? coverageSum / static_cast<double>(coverageCount) : 0.0,
                      0,
                      'f',
                      3)
                 .arg(elapsedMs(runStart, Clock::now()), 0, 'f', 1));

    if (_config.runFusion && !keepDepthFramesInMemory.load())
    {
        const QString msg = QStringLiteral(
            "内存压力已触发流式深度图保存，无法继续本次内存融合，自动跳过内存融合；"
            "请使用已保存的深度图运行“深度图融合”阶段。");
        LOG_WARN(QStringLiteral("[MVS] %1").arg(msg));
        _config.runFusion = false;
    }

    if (!_config.runFusion)
    {
        if (!_cancelled.load())
        {
            emit progressChanged("完成", 1.f);
        }
        emitFinishedOnce(allOk && !_cancelled.load());
        return;
    }

    // ── 阶段二：COLMAP BFS 深度图融合 → 直接输出 3D 点 ──────────────────────
    emit progressChanged("深度图融合...", static_cast<float>(NV) / (NV + 2));

    std::vector<FusionFrameInput> frames;
    for (const auto &fr : _depthFrames)
    {
        if (_cancelled.load())
        {
            emitFinishedOnce(false);
            return;
        }
        if (fr.eligibleForFusion() && fr.depthMap && !fr.depthMap->empty())
        {
            frames.push_back(buildFusionFrame(fr));
        }
    }

    if (frames.empty()) {
        emit errorOccurred("没有有效的深度帧，融合失败");
        emitFinishedOnce(false);
        return;
    }

    for (auto &frame : frames)
    {
        if (_cancelled.load())
        {
            emitFinishedOnce(false);
            return;
        }
        if (frame.sourceImageIndices.empty())
        {
            continue;
        }

        std::vector<int> remappedSources;
        remappedSources.reserve(frame.sourceImageIndices.size());
        for (int sourceViewIndex : frame.sourceImageIndices)
        {
            for (int frameIndex = 0; frameIndex < static_cast<int>(frames.size()); ++frameIndex)
            {
                if (frames[static_cast<size_t>(frameIndex)].viewIndex == sourceViewIndex)
                {
                    remappedSources.push_back(frameIndex);
                    break;
                }
            }
        }
        frame.sourceImageIndices = std::move(remappedSources);
    }

    LOG_INFO("[MVS][深度融合] 开始: frames=%d", static_cast<int>(frames.size()));

    // 使用新的 StereoFusionConfig
    StereoFusionConfig fusionCfg;
    fusionCfg.minNumPixels   = _config.fusion.minConsistentViews;
    fusionCfg.maxReprojError = _config.fusion.pixelThresh;
    fusionCfg.maxDepthError  = _config.fusion.relDepthThresh;
    fusionCfg.checkNumImages = std::min(50, NV);
    fusionCfg.workerCount    = resolvedTotalCpuThreadBudget(_config);
    fusionCfg.requireValidMask = true;
    fusionCfg.minSupportViews = std::max(1, _config.fusion.minConsistentViews - 1);
    // StereoFusionConfig historically owns a shared cancellation token. This
    // non-owning view is safe because the fusion object cannot outlive this
    // generator method, and it lets the existing inner row/BFS checks observe
    // GUI cancellation without copying an atomic value.
    fusionCfg.cancelFlag = std::shared_ptr<std::atomic_bool>(
        &_cancelled, [](std::atomic_bool *) {});
    fusionCfg.maxLocalDepthGradient = _config.fusion.enableLocalDepthOutlierFilter
        ? _config.fusion.localDepthOutlierRelThresh
        : 0.0f;

    // 少视图场景（≤2张）：crossCheck 已保证深度一致性，
    // 融合只需 1 个观测即可通过（避免 BFS 因级联过滤找不到第二观测而全部拒绝）
    if (NV <= 2) {
        fusionCfg.minNumPixels   = 1;   // 单观测即可，crossCheck 已保证质量
        fusionCfg.minSupportViews = 1;
        fusionCfg.maxDepthError  = std::max(fusionCfg.maxDepthError, 0.08f); // 放宽至 8%
        fusionCfg.maxReprojError = std::max(fusionCfg.maxReprojError, 3.0f); // 3 像素重投影误差
        LOG_WARN("[MVS][深度融合] 少视图模式: frames=%d min_pixels=1，已放宽融合阈值", NV);
    }

    // 如果有稀疏云 AABB，设置包围盒过滤离群点
    // 但必须检查稀疏云与相机是否在同一坐标系（防止 GPS 稀疏云 vs SFM 相机失配）
    if (_sparse.points.size() > 10) {
        // 计算相机中心的最大分量，作为坐标系尺度参考
        float camExtent = 1.0f;
        for (const auto &v : _views) {
            const std::array<double, 3> center = v.camera.cameraCenter();
            for (int k = 0; k < 3; ++k)
                camExtent = std::max(camExtent, static_cast<float>(std::fabs(center[k])));
        }
        // 稀疏云 AABB 最大分量
        float cloudExtent = 0.0f;
        for (int k = 0; k < 3; ++k)
            cloudExtent = std::max(cloudExtent,
                std::max(std::fabs(_sparse.maxPt[k]), std::fabs(_sparse.minPt[k])));
        // 坐标系兼容判断：尺度差不超过 50 倍才启用包围盒
        const float scaleRatio = cloudExtent / std::max(camExtent, 1.0f);
        if (scaleRatio < 50.0f)
        {
            fusionCfg.useBoundingBox = true;
            float pad = (NV <= 2) ? 0.5f : 0.2f;
            for (int k = 0; k < 3; ++k)
            {
                float range = _sparse.maxPt[k] - _sparse.minPt[k];
                fusionCfg.bboxMin[k] = _sparse.minPt[k] - range * pad;
                fusionCfg.bboxMax[k] = _sparse.maxPt[k] + range * pad;
            }
            LOG_DEBUG("[MVS][深度融合] bbox=[%.2f,%.2f,%.2f]-[%.2f,%.2f,%.2f]",
                      fusionCfg.bboxMin[0], fusionCfg.bboxMin[1], fusionCfg.bboxMin[2],
                      fusionCfg.bboxMax[0], fusionCfg.bboxMax[1], fusionCfg.bboxMax[2]);
        }
        else
        {
            LOG_WARN("[MVS][深度融合] 稀疏云与相机坐标系不匹配: cloud_extent=%.1f "
                     "camera_extent=%.1f ratio=%.1f，禁用包围盒",
                     cloudExtent, camExtent, scaleRatio);
            // 坐标系不兼容时深度初始化无先验，放宽一致性阈值至 10%
            fusionCfg.maxDepthError = std::max(fusionCfg.maxDepthError, 0.10f);
            LOG_WARN("[MVS][深度融合] 坐标系不兼容，max_depth_error 放宽至 %.2f",
                     fusionCfg.maxDepthError);
        }
    }

    DepthMapFusion fusion(fusionCfg);
    std::vector<FusedPoint> fusedPoints;
    std::string fuseErr;

    bool fuseOk = fusion.fuse(frames, fusedPoints,
        [this, NV](const std::string &msg, float ratio)
        {
            emit progressChanged(
                QString::fromStdString(msg),
                static_cast<float>(NV + ratio) / (NV + 2));
        },
        &fuseErr);

    if (!fuseOk)
    {
        if (_cancelled.load())
        {
            LOG_INFO("[MVS][深度融合] 收到取消请求");
            emitFinishedOnce(false);
            return;
        }
        LOG_ERROR("[MVS][深度融合] 失败: %s", fuseErr.c_str());
        emit errorOccurred(QString::fromStdString(fuseErr));
        emitFinishedOnce(false);
        return;
    }

    std::vector<DensePoint> cloud;
    cloud.reserve(fusedPoints.size());
    for (const FusedPoint &point : fusedPoints)
    {
        if (_cancelled.load())
        {
            emitFinishedOnce(false);
            return;
        }
        cloud.push_back(DensePoint{
            point.x, point.y, point.z,
            point.r, point.g, point.b});
    }

    // 保存每帧一致性过滤深度图（加锁，防止 GUI 线程并发读取）
    {
        std::lock_guard<std::mutex> lock(_filteredDepthsMutex);
        _filteredDepths = fusion.filteredDepths();
    }

    LOG_INFO("[MVS][深度融合] 完成: points=%d", static_cast<int>(cloud.size()));

    if (cloud.empty())
    {
        LOG_WARN("[MVS][深度融合] 产出 0 个点: frames=%d；可能是有效深度不足、视图过少或一致性阈值过严",
                 NV);
        emit errorOccurred(QStringLiteral("深度图融合产出 0 个点，请尝试增加影像数量或降低融合阈值"));
    }

    // ── 阶段三：离群点过滤 ──────────────────────────────────────────────
    if (!cloud.empty())
    {
        emit progressChanged("离群点过滤...", 0.90f);

        const std::size_t initialCount = cloud.size();
        if (initialCount <= kMaxInlineDenseFilterPoints)
        {
            const float maxStageRemovalRatio = 0.45f;
            const float minOverallRetentionRatio = 0.40f;

            auto applyFilterWithGuard = [&](const char *stageName, auto &&filterOp)
            {
                if (_cancelled.load() || cloud.size() < 100)
                {
                    return;
                }

                const std::size_t beforeCount = cloud.size();
                std::vector<DensePoint> filtered = filterOp(cloud);
                if (_cancelled.load())
                {
                    return;
                }
                if (filtered.empty())
                {
                    LOG_WARN("[MVS][点云过滤] %s 结果为空，跳过该阶段以保护点云", stageName);
                    return;
                }

                const float stageRemovedRatio = static_cast<float>(beforeCount - filtered.size())
                                                / static_cast<float>(beforeCount);
                const float overallRetentionRatio = static_cast<float>(filtered.size())
                                                  / static_cast<float>(initialCount);

                if (stageRemovedRatio > maxStageRemovalRatio || overallRetentionRatio < minOverallRetentionRatio)
                {
                    LOG_WARN("[MVS][点云过滤] %s 触发护栏: removed=%.1f%% limit=%.1f%% "
                             "overall_retention=%.1f%% minimum=%.1f%%，保留上一阶段结果",
                             stageName,
                             stageRemovedRatio * 100.0f,
                             maxStageRemovalRatio * 100.0f,
                             overallRetentionRatio * 100.0f,
                             minOverallRetentionRatio * 100.0f);
                    return;
                }

                cloud.swap(filtered);
            };

            const plapoint::ProcessingDevice pointFilterDevice =
                _config.pointCloudProcessingDevice;

            // 第 1 遍：统计离群点过滤（SOR）— 移除 kNN 距离异常的点
            applyFilterWithGuard("SOR-1", [pointFilterDevice](const std::vector<DensePoint> &inputCloud) {
                return DenseCloudBuilder::statisticalOutlierRemoval(inputCloud, 30, 1.2f, pointFilterDevice);
            });

            // 第 2 遍：半径过滤 — 基于点云密度估算搜索半径
            if (cloud.size() > 100)
            {
                float minX = cloud[0].x, maxX = cloud[0].x;
                float minY = cloud[0].y, maxY = cloud[0].y;
                float minZ = cloud[0].z, maxZ = cloud[0].z;
                for (const auto &p : cloud)
                {
                    minX = std::min(minX, p.x);
                    maxX = std::max(maxX, p.x);
                    minY = std::min(minY, p.y);
                    maxY = std::max(maxY, p.y);
                    minZ = std::min(minZ, p.z);
                    maxZ = std::max(maxZ, p.z);
                }
                float volume = (maxX - minX) * (maxY - minY) * (maxZ - minZ);
                volume = std::max(volume, 1e-12f);
                const float avgSpacing = std::cbrt(volume / static_cast<float>(cloud.size()));
                const float searchRadius = std::max(avgSpacing * 4.0f, 1e-4f);
                applyFilterWithGuard("RadiusOR", [searchRadius, pointFilterDevice](
                                         const std::vector<DensePoint> &inputCloud) {
                    return DenseCloudBuilder::radiusOutlierRemoval(inputCloud, searchRadius, 5, pointFilterDevice);
                });
            }

            LOG_INFO("[MVS][点云过滤] 完成: points=%d", static_cast<int>(cloud.size()));
        }
        else
        {
            LOG_INFO("[MVS][点云过滤] 跳过内联过滤: points=%zu limit=%zu；保留完整融合点云",
                     initialCount,
                     kMaxInlineDenseFilterPoints);
        }
    }

    if (_cancelled.load())
    {
        emitFinishedOnce(false);
        return;
    }

    emit pointCloudReady(cloud);
    emit progressChanged("完成", 1.f);
    // 只要最终生成了有效点云就算成功（部分帧深度估计失败不影响最终结果）
    emitFinishedOnce(!cloud.empty() && !_cancelled.load());
}

} // namespace mvs
} // namespace xjw

// =============================================================================
// 文件: DepthMapGenerator.cpp
// 模块: MVS - Qt 封装的深度图生成 + COLMAP BFS 融合
// =============================================================================

#include "DepthMapGenerator.h"
#include "DenseCloudBuilder.h"
#include "DepthConsistencyCache.h"
#include "DepthCrossViewHoleRepair.h"
#include "DepthGeometryConsistency.h"
#include "DepthFrameUtils.h"
#include "DepthPyramidPolicy.h"
#include "EpipolarRectifier.h"
#include "CameraBaseline.h"
#include "MvsImagePreprocessor.h"
#include "MvsQualityReport.h"
#include "MvsSourcePlanner.h"
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
#include <limits>
#include <fstream>
#include <unordered_map>
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

std::string mvsSourcePairKey(const std::string &imageA, const std::string &imageB)
{
    const QString keyA = normalizedMvsPathKey(imageA);
    const QString keyB = normalizedMvsPathKey(imageB);
    if (keyA.isEmpty() || keyB.isEmpty() || keyA == keyB)
    {
        return std::string();
    }

    const QString pairKey = keyA < keyB
        ? keyA + QLatin1Char('\n') + keyB
        : keyB + QLatin1Char('\n') + keyA;
    return pairKey.toStdString();
}

struct MvsSourcePairQualityLookup
{
    std::unordered_map<std::string, MvsSourcePairQuality> qualitiesByPairKey;

    const MvsSourcePairQuality *find(const std::string &imageA,
                                     const std::string &imageB) const
    {
        const std::string key = mvsSourcePairKey(imageA, imageB);
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

uint64_t estimateDepthFrameCacheBytes(const std::vector<CameraView> &views)
{
    uint64_t total = 0;
    for (const CameraView &view : views)
    {
        const uint64_t frameBytes = depthFramePixelStorageBytes(view.imageWidth, view.imageHeight);
        if (std::numeric_limits<uint64_t>::max() - total < frameBytes)
        {
            return std::numeric_limits<uint64_t>::max();
        }
        total += frameBytes;
    }
    return total;
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

uint64_t retainedDepthMemoryBudgetBytes(const SystemMemorySnapshot &snapshot, const DepthGenConfig &config)
{
    if (!snapshot.valid)
    {
        return 0;
    }

    const double fraction = std::clamp(static_cast<double>(config.maxDepthCacheRamFraction), 0.10, 0.90);
    const uint64_t totalBudget = static_cast<uint64_t>(
        static_cast<double>(snapshot.totalPhysicalBytes) * fraction);
    const uint64_t availableBudget = snapshot.availablePhysicalBytes > config.minFreeRamBytes
        ? snapshot.availablePhysicalBytes - config.minFreeRamBytes
        : 0ull;
    return std::min(totalBudget, availableBudget);
}

bool shouldRetainAllDepthFramesInMemory(const std::vector<CameraView> &views,
                                        const DepthGenConfig &config,
                                        const SystemMemorySnapshot &snapshot,
                                        QString *reason)
{
    if (!config.adaptiveDepthCacheMemory)
    {
        if (reason)
        {
            *reason = QStringLiteral("adaptiveDepthCacheMemory=false");
        }
        return true;
    }

    const uint64_t requiredBytes = estimateDepthFrameCacheBytes(views);
    if (requiredBytes == 0)
    {
        if (reason)
        {
            *reason = QStringLiteral("无有效影像尺寸，采用保守流式模式");
        }
        return false;
    }
    if (!snapshot.valid)
    {
        if (reason)
        {
            *reason = QStringLiteral("无法读取系统内存，采用保守流式模式");
        }
        return false;
    }

    const uint64_t budgetBytes = retainedDepthMemoryBudgetBytes(snapshot, config);
    if (requiredBytes <= budgetBytes)
    {
        if (reason)
        {
            *reason = QStringLiteral("预计缓存 %1 GiB <= 内存预算 %2 GiB")
                          .arg(bytesToGiB(requiredBytes), 0, 'f', 2)
                          .arg(bytesToGiB(budgetBytes), 0, 'f', 2);
        }
        return true;
    }

    if (reason)
    {
        *reason = QStringLiteral("预计缓存 %1 GiB > 内存预算 %2 GiB")
                      .arg(bytesToGiB(requiredBytes), 0, 'f', 2)
                      .arg(bytesToGiB(budgetBytes), 0, 'f', 2);
    }
    return false;
}

bool memoryPressureRequiresStreaming(const DepthGenConfig &config,
                                     const SystemMemorySnapshot &snapshot,
                                     uint64_t largestFrameBytes)
{
    if (!config.adaptiveDepthCacheMemory || !snapshot.valid)
    {
        return false;
    }

    const uint64_t transientReserve =
        std::max<uint64_t>(config.minFreeRamBytes, largestFrameBytes * 4ull);
    return snapshot.availablePhysicalBytes < transientReserve;
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
                                 uint64_t largestFrameBytes)
{
    if (!config.adaptiveDepthCacheMemory || !snapshot.valid || largestFrameBytes == 0)
    {
        return 2;
    }

    const uint64_t budgetBytes = retainedDepthMemoryBudgetBytes(snapshot, config);
    if (budgetBytes >= largestFrameBytes * 8ull)
    {
        return 4;
    }
    return 2;
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

    explicit DepthFrameArtifactSaveQueue(SaveFn saveFn, size_t maxBufferedTasks = 2)
        : m_saveFn(std::move(saveFn))
        , m_maxBufferedTasks(std::max<size_t>(1, maxBufferedTasks))
        , m_worker(&DepthFrameArtifactSaveQueue::run, this)
    {
    }

    ~DepthFrameArtifactSaveQueue()
    {
        stop();
    }

    void enqueue(int frameIndex, const DepthFrameResult &result, const QString &stageLabel)
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_capacityCv.wait(lock, [this]() {
                return m_stopping || m_tasks.size() < m_maxBufferedTasks;
            });
            if (m_stopping)
            {
                return;
            }
            m_tasks.push_back(SaveTask{frameIndex, result, stageLabel});
        }
        m_cv.notify_one();
    }

    void waitUntilIdle()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_idleCv.wait(lock, [this]() {
            return m_tasks.empty() && m_activeTasks == 0;
        });
    }

    void cancel()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_dropPendingTasks = true;
            m_stopping = true;
            m_tasks.clear();
            if (m_activeTasks == 0)
            {
                m_idleCv.notify_all();
            }
        }
        m_capacityCv.notify_all();
        m_cv.notify_all();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
        }
        m_capacityCv.notify_all();
        m_cv.notify_all();
        if (m_worker.joinable())
        {
            m_worker.join();
        }
    }

    bool failed() const
    {
        return m_failed.load();
    }

private:
    struct SaveTask
    {
        int frameIndex = -1;
        DepthFrameResult result;
        QString stageLabel;
    };

    void run()
    {
        for (;;)
        {
            SaveTask task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this]() {
                    return m_stopping || !m_tasks.empty();
                });

                if (m_tasks.empty())
                {
                    if (m_stopping)
                    {
                        break;
                    }
                    continue;
                }

                if (m_dropPendingTasks)
                {
                    m_tasks.clear();
                    if (m_activeTasks == 0)
                    {
                        m_idleCv.notify_all();
                    }
                    if (m_stopping)
                    {
                        break;
                    }
                    continue;
                }

                task = m_tasks.front();
                m_tasks.pop_front();
                ++m_activeTasks;
                m_capacityCv.notify_one();
            }

            if (!m_saveFn(task.frameIndex, task.result, task.stageLabel))
            {
                m_failed = true;
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                --m_activeTasks;
                if (m_tasks.empty() && m_activeTasks == 0)
                {
                    m_idleCv.notify_all();
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_tasks.empty() && m_activeTasks == 0)
            {
                m_idleCv.notify_all();
            }
            m_capacityCv.notify_all();
        }
    }

    SaveFn m_saveFn;
    std::deque<SaveTask> m_tasks;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::condition_variable m_capacityCv;
    std::condition_variable m_idleCv;
    std::thread m_worker;
    std::atomic<bool> m_failed{false};
    size_t m_maxBufferedTasks = 1;
    bool m_stopping = false;
    bool m_dropPendingTasks = false;
    int m_activeTasks = 0;
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
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(workers));
    for (int worker = 0; worker < workers; ++worker)
    {
        threads.emplace_back([&]() {
            for (;;)
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
    for (std::thread &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
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
    options.anchoredInterpolation.enabled = true;
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
    std::vector<int> sources;
    const int target_count = std::clamp(
        requestedSourceCount, 2, std::min(16, std::max(2, viewCount - 1)));
    auto append_source = [&](int source_index)
    {
        if (source_index < 0 || source_index >= viewCount || source_index == refIdx ||
            source_index >= static_cast<int>(frames.size()) ||
            !frames[source_index].eligibleAsConsistencySource() ||
            std::find(sources.begin(), sources.end(), source_index) != sources.end())
        {
            return;
        }
        sources.push_back(source_index);
    };
    for (int source_index : consistencySources)
    {
        append_source(source_index);
    }
    for (int distance = 1;
         distance < viewCount && static_cast<int>(sources.size()) < target_count;
         ++distance)
    {
        append_source((refIdx - distance + viewCount) % viewCount);
        if (static_cast<int>(sources.size()) < target_count)
        {
            append_source((refIdx + distance) % viewCount);
        }
    }
    return sources;
}

bool isCudaMemoryFailure(const std::string &message)
{
    const std::string lower = asciiLowerCopy(message);

    return lower.find("out of memory") != std::string::npos
        || lower.find("cuda_error_memory") != std::string::npos
        || (lower.find("cuda") != std::string::npos && lower.find("memory") != std::string::npos);
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
    const cv::Mat *hintRadius)
{
    const bool tryCuda = config.useCuda && PatchMatchDepthEstimator::isCudaAvailable();
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
                                                  hintRadius);
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
                                               hintRadius))
        {
            if (attemptConfig.downsampleFactor != config.downsampleFactor)
            {
                fprintf(stderr,
                        "[MVS] 帧 %d: %s CUDA 自适应重试成功，最终 ds=%d iters=%d patch=%d\n",
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

        fprintf(stderr,
                "[MVS] 帧 %d: %s CUDA 显存不足，ds=%d -> ds=%d 自动重试 (%s)\n",
                refIdx,
                stageLabel,
                attemptConfig.downsampleFactor,
                nextConfig.downsampleFactor,
                attemptError.c_str());

        attemptConfig = nextConfig;
        attemptConfig.cudaFallbackToCpu = false;
    }

    fprintf(stderr,
            "[MVS] 帧 %d: %s CUDA 自适应重试未成功，回退 CPU (%s)\n",
            refIdx,
            stageLabel,
            lastCudaError.empty() ? "未知 CUDA 错误" : lastCudaError.c_str());

    PatchMatchConfig cpuConfig = config;
    cpuConfig.useCuda = false;
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
                                                          hintRadius);
    if (!cpuOk && errorMsg && errorMsg->empty())
    {
        *errorMsg = lastCudaError;
    }
    return cpuOk;
}

void accumulateDepthConsistency(const cv::Mat &referenceDepth,
                                const Camera &referenceCamera,
                                const cv::Mat &sourceDepth,
                                const Camera &sourceCamera,
                                int sourceOrdinal,
                                float relativeThreshold,
                                int rowWorkers,
                                const std::atomic<bool> &cancelled,
                                cv::Mat &consistentVotes,
                                cv::Mat &occludedVotes,
                                cv::Mat &contradictedVotes,
                                cv::Mat &unverifiableVotes,
                                cv::Mat &geometrySourceMask,
                                cv::Mat &sourceInverseDepthSum,
                                cv::Mat &sourceInverseDepthSquaredSum)
{
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
            const ProjectedDepthConsistencyResult result =
                evaluateProjectedDepthConsistency(
                    referenceCamera,
                    cv::Point2f(static_cast<float>(column), static_cast<float>(row)),
                    reference_depth,
                    sourceCamera,
                    sourceDepth,
                    relativeThreshold);
            switch (result.evidence)
            {
            case DepthConsistencyEvidence::Consistent:
                ++consistent_row[column];
                if (sourceOrdinal >= 0 && sourceOrdinal < 16)
                {
                    source_mask_row[column] = static_cast<std::uint16_t>(
                        source_mask_row[column] |
                        (static_cast<std::uint16_t>(1U) << sourceOrdinal));
                }
                if (result.consistentReferenceDepth > 0.0f &&
                    std::isfinite(result.consistentReferenceDepth))
                {
                    const float inverse_depth = 1.0f / result.consistentReferenceDepth;
                    inverse_sum_row[column] += inverse_depth;
                    inverse_squared_sum_row[column] += inverse_depth * inverse_depth;
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
    });
}

cv::Mat makeDepthConsistencyMask(const cv::Mat &referenceDepth,
                                 int sourceViewCount,
                                 int minimumSourceConfirmations,
                                 const cv::Mat &consistentVotes,
                                 const cv::Mat &occludedVotes,
                                 const cv::Mat &contradictedVotes)
{
    cv::Mat mask(referenceDepth.size(), CV_8U, cv::Scalar(0));
    for (int row = 0; row < referenceDepth.rows; ++row)
    {
        const float *depth_row = referenceDepth.ptr<float>(row);
        const uint16_t *consistent_row = consistentVotes.ptr<uint16_t>(row);
        const uint16_t *occluded_row = occludedVotes.ptr<uint16_t>(row);
        const uint16_t *contradicted_row = contradictedVotes.ptr<uint16_t>(row);
        uint8_t *mask_row = mask.ptr<uint8_t>(row);
        for (int column = 0; column < referenceDepth.cols; ++column)
        {
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
    }
    return mask;
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
    MvsSceneProfile sceneProfile)
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
    cv::Mat anchor_mask = *result.crossViewRepairedMask;
    if (anchor_mask.size() != depth.size())
    {
        cv::resize(
            anchor_mask, anchor_mask, depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    if (result.crossViewRepairedMask->size() != depth.size())
    {
        *result.crossViewRepairedMask = anchor_mask.clone();
    }
    DepthAnchoredHoleInterpolationOptions options;
    options.enabled = true;
    return interpolateAnchoredInternalDepthHoles(
        depth,
        support_mask,
        anchor_mask,
        nullptr,
        options,
        confidence.empty() ? nullptr : &confidence,
        result.crossViewRepairedMask.data());
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
        fprintf(stderr,
                "[MVS] 帧 %d: %s ds=%d iters=%d patch=%d\n",
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
                                                radius))
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
    _pairCommonCounts.clear();
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
    if (_frameCachesReady)
    {
        return;
    }

    const int NV = static_cast<int>(_views.size());
    const size_t pointCount = _sparse.points.size();
    _frameCaches.assign(static_cast<size_t>(std::max(0, NV)), FrameMvsCache{});
    _visibilityWordCount = (pointCount + 63) / 64;
    _visibilityBits.assign(static_cast<size_t>(std::max(0, NV)) * _visibilityWordCount, 0);
    _pairCommonCounts.assign(static_cast<size_t>(std::max(0, NV)) * static_cast<size_t>(std::max(0, NV)), 0);

    if (NV <= 0 || pointCount == 0)
    {
        _frameCachesReady = true;
        return;
    }

    const auto start = Clock::now();
    constexpr int kMaxPairViewsPerPoint = 32;
    constexpr size_t kParallelVisibilityPointThreshold = 20000;

    struct VisibilityCacheShard
    {
        explicit VisibilityCacheShard(int viewCount)
            : visiblePointIndicesByView(static_cast<size_t>(std::max(0, viewCount)))
            , pairCommonCounts(static_cast<size_t>(std::max(0, viewCount))
                                   * static_cast<size_t>(std::max(0, viewCount)),
                               0)
        {
        }

        std::vector<std::vector<size_t>> visiblePointIndicesByView;
        std::vector<int> pairCommonCounts;
    };

    int visibilityWorkerCount = 1;
#ifdef _OPENMP
    {
        const int hwThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        const int requested = std::max(1, _config.cpuWorkerCount);
        visibilityWorkerCount = pointCount >= kParallelVisibilityPointThreshold && NV > 1
            ? std::clamp(std::min(requested, hwThreads), 1, 16)
            : 1;
    }
#endif

    std::vector<VisibilityCacheShard> visibilityShards;
    visibilityShards.reserve(static_cast<size_t>(visibilityWorkerCount));
    for (int workerIndex = 0; workerIndex < visibilityWorkerCount; ++workerIndex)
    {
        visibilityShards.emplace_back(NV);
    }

#pragma omp parallel num_threads(visibilityWorkerCount) if(visibilityWorkerCount > 1)
    {
        int workerIndex = 0;
#ifdef _OPENMP
        workerIndex = omp_get_thread_num();
#endif
        VisibilityCacheShard &shard = visibilityShards[static_cast<size_t>(workerIndex)];
        std::vector<int> visibleViews;
        visibleViews.reserve(std::min(NV, kMaxPairViewsPerPoint));
        std::vector<int> pairViews;
        pairViews.reserve(kMaxPairViewsPerPoint);

#pragma omp for schedule(static)
        for (long long pointIndexSigned = 0; pointIndexSigned < static_cast<long long>(pointCount); ++pointIndexSigned)
        {
            const size_t pointIndex = static_cast<size_t>(pointIndexSigned);
            visibleViews.clear();
            const auto &point = _sparse.points[pointIndex];
            for (int viewIdx = 0; viewIdx < NV; ++viewIdx)
            {
                if (!isMvsSparsePointVisibleInView(_views[viewIdx], point))
                {
                    continue;
                }

                shard.visiblePointIndicesByView[static_cast<size_t>(viewIdx)].push_back(pointIndex);
                visibleViews.push_back(viewIdx);
            }

            if (visibleViews.size() < 2)
            {
                continue;
            }

            pairViews.clear();
            if (static_cast<int>(visibleViews.size()) <= kMaxPairViewsPerPoint)
            {
                pairViews.assign(visibleViews.begin(), visibleViews.end());
            }
            else
            {
                for (int sample = 0; sample < kMaxPairViewsPerPoint; ++sample)
                {
                    const size_t idx = static_cast<size_t>(sample) * visibleViews.size()
                                       / static_cast<size_t>(kMaxPairViewsPerPoint);
                    pairViews.push_back(visibleViews[std::min(idx, visibleViews.size() - 1)]);
                }
            }

            for (size_t a = 0; a < pairViews.size(); ++a)
            {
                for (size_t b = a + 1; b < pairViews.size(); ++b)
                {
                    const int ia = pairViews[a];
                    const int ib = pairViews[b];
                    ++shard.pairCommonCounts[static_cast<size_t>(ia) * NV + ib];
                    ++shard.pairCommonCounts[static_cast<size_t>(ib) * NV + ia];
                }
            }
        }
    }

    auto mergeVisibilityCacheShards = [&]()
    {
        for (int viewIdx = 0; viewIdx < NV; ++viewIdx)
        {
            size_t visibleCount = 0;
            for (const VisibilityCacheShard &shard : visibilityShards)
            {
                visibleCount += shard.visiblePointIndicesByView[static_cast<size_t>(viewIdx)].size();
            }

            auto &visible = _frameCaches[static_cast<size_t>(viewIdx)].visiblePointIndices;
            visible.reserve(visibleCount);
            for (const VisibilityCacheShard &shard : visibilityShards)
            {
                const auto &localVisible = shard.visiblePointIndicesByView[static_cast<size_t>(viewIdx)];
                visible.insert(visible.end(), localVisible.begin(), localVisible.end());
            }
        }

        for (VisibilityCacheShard &shard : visibilityShards)
        {
            for (size_t idx = 0; idx < _pairCommonCounts.size(); ++idx)
            {
                _pairCommonCounts[idx] += shard.pairCommonCounts[idx];
            }
        }
    };
    mergeVisibilityCacheShards();

    auto buildVisibilityBitsFromFrameCaches = [&]()
    {
        const bool parallelBuildVisibilityBits = NV > 8 && pointCount >= kParallelVisibilityPointThreshold;
#pragma omp parallel for schedule(static) if(parallelBuildVisibilityBits)
        for (int viewIdx = 0; viewIdx < NV; ++viewIdx)
        {
            const auto &visible = _frameCaches[static_cast<size_t>(viewIdx)].visiblePointIndices;
            const size_t viewOffset = static_cast<size_t>(viewIdx) * _visibilityWordCount;
            for (size_t pointIndex : visible)
            {
                const size_t word = pointIndex / 64;
                const size_t bit = pointIndex % 64;
                _visibilityBits[viewOffset + word] |= (uint64_t{1} << bit);
            }
        }
    };
    buildVisibilityBitsFromFrameCaches();

    _frameCachesReady = true;
    std::vector<std::string> activeImagePaths;
    activeImagePaths.reserve(_views.size());
    for (const CameraView &view : _views)
    {
        activeImagePaths.push_back(view.imagePath);
    }
    const std::vector<MvsSourcePairQuality> activePairQualities =
        filterMvsSourcePairQualitiesForImages(_config.sourcePairQualities, activeImagePaths);
    const MvsSourcePairQualityLookup pairQualityLookup =
        buildMvsSourcePairQualityLookup(activePairQualities);
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
        };

        std::vector<RankedSourceCandidate> rankedSourceCandidates;
        rankedSourceCandidates.reserve(static_cast<size_t>(std::max(0, NV - 1)));
        for (int sourceIdx = 0; sourceIdx < NV; ++sourceIdx)
        {
            if (sourceIdx == refIdx)
            {
                continue;
            }

            const int common = _pairCommonCounts[static_cast<size_t>(refIdx) * NV + sourceIdx];
            if (common <= 0)
            {
                continue;
            }

            rankedSourceCandidates.push_back({sourceIdx, common, std::abs(sourceIdx - refIdx)});
        }

        std::sort(rankedSourceCandidates.begin(),
                  rankedSourceCandidates.end(),
                  [](const RankedSourceCandidate &a, const RankedSourceCandidate &b)
                  {
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
            if (pairQualityLookup.find(
                    _views[static_cast<size_t>(refIdx)].imagePath,
                    _views[static_cast<size_t>(candidate.viewIndex)].imagePath))
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
            const MvsSourcePairQuality *pairQuality = pairQualityLookup.find(
                _views[static_cast<size_t>(refIdx)].imagePath,
                _views[static_cast<size_t>(candidate.viewIndex)].imagePath);
            sourceCandidate.geometricInliers = referenceHasPairQuality
                ? (pairQuality ? std::max(0, pairQuality->geometricInliers) : 0)
                : candidate.commonVisiblePoints;
            sourceCandidate.verifiedPairGeometry = pairQuality
                && pairQuality->verified
                && pairQuality->geometricInliers > 0;
            sourceCandidate.medianTriangulationAngleDeg = medianAngle;
            sourceCandidate.coverageScore = refVisibleCount > 0
                ? std::clamp(static_cast<float>(candidate.commonVisiblePoints) / static_cast<float>(refVisibleCount), 0.0f, 1.0f)
                : 0.0f;
            sourceCandidate.baselineScore = std::clamp(medianAngle / 20.0f, 0.0f, 1.0f);
            sourceCandidate.sequenceDistance = candidate.sequenceDistance;
            sourceCandidate.knownOverlap = referenceHasPairQuality
                ? sourceCandidate.verifiedPairGeometry
                : true;
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

        const MvsSourcePlan sourcePlan = requireVerifiedSourcePairsForReference
            ? planMvsSourceViewsVerifiedFirst(candidates, plannerOptions)
            : planMvsSourceViews(candidates, plannerOptions);

        auto &sources = _frameCaches[static_cast<size_t>(refIdx)].sourceViewIndices;
        _frameCaches[static_cast<size_t>(refIdx)].sourceViewScores = sourcePlan.selected;
        _frameCaches[static_cast<size_t>(refIdx)].requestedSourceViewCount = desiredSourceCount;
        _frameCaches[static_cast<size_t>(refIdx)].sourceViewShortfall = sourcePlan.sourceViewShortfall;
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
void DepthMapGenerator::preloadImages()
{
    const int NV = static_cast<int>(_views.size());
    _grayCache.resize(NV);
    _validRegionMasks.resize(NV);
    _projectMaskLoaded.assign(static_cast<size_t>(NV), 0);

    const int workerCount = preloadImagesWorkerCount(NV, _config.cpuWorkerCount);
    fprintf(stderr, "[MVS] preloadImages(): workers=%d views=%d\n", workerCount, NV);
    if (workerCount <= 0)
    {
        return;
    }

    std::atomic<int> nextImage{0};
    auto preloadOneImage = [this, NV, &nextImage]()
    {
        for (;;)
        {
            if (_cancelled.load())
            {
                break;
            }

            const int i = nextImage.fetch_add(1);
            if (i >= NV)
            {
                break;
            }

            _grayCache[i] = xjw::common::io::readImage(_views[i].imagePath, cv::IMREAD_GRAYSCALE);
            if (_grayCache[i].empty())
            {
                LOG_WARN(QStringLiteral("[MVS] 警告: 无法读取图像 %1: %2").arg(i).arg(QString::fromStdString(_views[i].imagePath)));
                continue;
            }

            const std::string &project_mask_path = _views[static_cast<size_t>(i)].validRegionMaskPath;
            if (!project_mask_path.empty())
            {
                const cv::Mat project_mask = xjw::common::io::readImage(
                    project_mask_path,
                    cv::IMREAD_GRAYSCALE);
                if (!project_mask.empty())
                {
                    _validRegionMasks[static_cast<size_t>(i)] = projectMaskToValidMask(
                        project_mask,
                        _grayCache[static_cast<size_t>(i)].size());
                    bool content_refined = false;
                    float retained_ratio = 1.0f;
                    if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
                    {
                        _validRegionMasks[static_cast<size_t>(i)] =
                            refineOrbitalProjectValidMask(
                                _grayCache[static_cast<size_t>(i)],
                                _validRegionMasks[static_cast<size_t>(i)],
                                &content_refined,
                                &retained_ratio);
                    }
                    _projectMaskLoaded[static_cast<size_t>(i)] = 1;
                    const int valid_pixels = cv::countNonZero(
                        _validRegionMasks[static_cast<size_t>(i)]);
                    const int total_pixels = _validRegionMasks[static_cast<size_t>(i)].rows *
                                             _validRegionMasks[static_cast<size_t>(i)].cols;
                    LOG_INFO(QStringLiteral("[MVS] 图像 %1: 使用项目蒙版，有效区域 %2/%3 (%4%)")
                                 .arg(i)
                                 .arg(valid_pixels)
                                 .arg(total_pixels)
                                 .arg(total_pixels > 0
                                          ? 100.0 * static_cast<double>(valid_pixels) / total_pixels
                                          : 0.0,
                                      0,
                                      'f',
                                      1));
                    if (content_refined)
                    {
                        LOG_INFO(QStringLiteral(
                                     "[MVS] 图像 %1: 暗背景内部开口细化，保留项目有效区 %2%")
                                     .arg(i)
                                     .arg(100.0 * static_cast<double>(retained_ratio),
                                          0,
                                          'f',
                                          1));
                    }
                }
                else
                {
                    LOG_WARN(QStringLiteral("[MVS] 图像 %1: 项目蒙版读取失败，回退内容区域检测: %2")
                                 .arg(i)
                                 .arg(QString::fromStdString(project_mask_path)));
                }
            }

            // 无项目蒙版时，才在 CLAHE 增强前自动检测内容区域。
            // gamma/CLAHE 会抬高黑边亮度，因此检测必须基于原始灰度图。
            if (_projectMaskLoaded[static_cast<size_t>(i)] == 0)
            {
                float coverage = 0.f;
                double otsuThresh = 0.0;
                int adaptiveThresh = 0;
                _validRegionMasks[static_cast<size_t>(i)] = buildContentMask(
                    _grayCache[static_cast<size_t>(i)],
                    &coverage,
                    &otsuThresh,
                    &adaptiveThresh);

                const int totalPx = _grayCache[i].rows * _grayCache[i].cols;
                const int contentPx = static_cast<int>(std::round(coverage * totalPx));
                if (_validRegionMasks[static_cast<size_t>(i)].empty())
                {
                    fprintf(stderr,
                            "[MVS] 图像 %d: 内容掩码覆盖 %.1f%%，自动跳过内容掩码过滤 (Otsu=%.0f adaptiveThresh=%d)\n",
                            i,
                            coverage * 100.0f,
                            otsuThresh,
                            adaptiveThresh);
                }
                else
                {
                    fprintf(stderr, "[MVS] 图像 %d: 内容掩码 (Otsu=%.0f adaptiveThresh=%d): %d/%d (%.1f%%)\n",
                            i, otsuThresh, adaptiveThresh, contentPx, totalPx, coverage * 100.0f);
                }
            }

            // ── 自适应对比度增强 (CLAHE) ──────────────────────────────────
            // 暗场/低对比度图像（均值 < 40）的像素方差极小，NCC 分母接近 0，
            // 导致 PatchMatch 无法区分正确/错误深度假设 → 全图噪声。
            // CLAHE 局部均衡化可在不影响已有纹理的前提下大幅提升暗区对比度。
            double imgMean = cv::mean(_grayCache[i])[0];
            if (imgMean < 80.0)
            {
                cv::Mat enhanced;
                if (imgMean < 30.0)
                {
                    // 极暗图像（航空影像常见）：先做 gamma 校正提升暗区可见度，
                    // 再用高 clipLimit CLAHE 进一步增强局部对比度。
                    // gamma=0.4 将 [0,255] 非线性映射，使暗部细节显现。
                    cv::Mat floatImg;
                    _grayCache[i].convertTo(floatImg, CV_32F, 1.0 / 255.0);
                    cv::pow(floatImg, 0.4, floatImg);
                    floatImg.convertTo(enhanced, CV_8U, 255.0);
                    auto clahe = cv::createCLAHE(8.0, cv::Size(8, 8));
                    clahe->apply(enhanced, enhanced);
                }
                else
                {
                    // 中等暗度：标准 CLAHE
                    auto clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
                    clahe->apply(_grayCache[i], enhanced);
                }
                double newMean = cv::mean(enhanced)[0];
                fprintf(stderr, "[MVS] 图像 %d: 低对比度 (mean=%.1f)，应用 CLAHE → mean=%.1f\n",
                        i, imgMean, newMean);
                _grayCache[i] = enhanced;
            }
            else
            {
                fprintf(stderr, "[MVS] 预加载图像 %d/%d: %dx%d (mean=%.1f)\n",
                        i+1, NV, _grayCache[i].cols, _grayCache[i].rows, imgMean);
            }
        }
    };

    std::vector<std::thread> preloadWorkers;
    preloadWorkers.reserve(static_cast<size_t>(workerCount));
    for (int workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        preloadWorkers.emplace_back(preloadOneImage);
    }

    for (std::thread &worker : preloadWorkers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

void DepthMapGenerator::refreshViewImageDimensionsFromCache()
{
    int updatedCount = 0;
    const int count = std::min(static_cast<int>(_views.size()), static_cast<int>(_grayCache.size()));
    for (int i = 0; i < count; ++i)
    {
        CameraView &view = _views[static_cast<size_t>(i)];
        const cv::Mat &image = _grayCache[static_cast<size_t>(i)];
        if ((view.imageWidth <= 0 || view.imageHeight <= 0) && !image.empty())
        {
            view.imageWidth = image.cols;
            view.imageHeight = image.rows;
            ++updatedCount;
        }
    }

    if (updatedCount > 0)
    {
        LOG_INFO(QStringLiteral("[MVS] 已从预加载图像补齐 %1 个视图尺寸，用于深度图内存预算")
                     .arg(updatedCount));
    }
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
        fprintf(stderr,
                "[MVS] 帧 %d: 共视稀疏点不足，深度范围回退到参考帧可见点 (%zu)\n",
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
            fprintf(stderr, "[MVS] 深度范围回退(全局基线): global_baseline=%.4f → zNear=%.4f zFar=%.4f\n",
                    maxBaseline, zNear, zFar);
        } else {
            // 实在无法估计
            float dx = _sparse.maxPt[0] - _sparse.minPt[0];
            float dy = _sparse.maxPt[1] - _sparse.minPt[1];
            float dz = _sparse.maxPt[2] - _sparse.minPt[2];
            float diag = std::sqrt(dx*dx + dy*dy + dz*dz);
            zNear = 0.1f;
            zFar  = diag > 0 ? diag * 3.f : 100.f;
            fprintf(stderr, "[MVS] 深度范围回退(AABB): diag=%.4f → zNear=%.4f zFar=%.4f\n",
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

    fprintf(stderr,
            "[MVS] 帧 %d: 深度范围 IQR: Q1=%.4f Q3=%.4f IQR=%.4f fence=[%.4f, %.4f] inliers=%zu/%zu visiblePts=%zu\n",
            refIdx, Q1, Q3, IQR, lowerFence, upperFence, inlierDepths.size(), n, visiblePointIndices.size());
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
        fprintf(stderr,
                "[Hint] 帧 %d: 共视 hint 点为空，回退到参考帧可见点 (%zu)\n",
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
        fprintf(stderr,
                "[Hint] 帧 %d: 可见稀疏点=%zu 没有可用 hint seed，跳过 hint 传播\n",
                refIdx,
                samples.size());
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
    fprintf(stderr,
            "[Hint] 帧 %d: 可见稀疏点=%zu seedPixels=%d radius=%d hint覆盖=%d/%d (%.1f%%)\n",
            refIdx, samples.size(), seedHintCnt, maxHintRadius,
            hintCnt, W*H, 100.f*hintCnt/(W*H));
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

    cv::Mat seed(H, W, CV_8U, cv::Scalar(0));
    int projectedSeeds = 0;
    for (const ProjectedSparseDepthSample &sample : samples)
    {
        const int iu = static_cast<int>(std::round(sample.uNorm * static_cast<float>(W)));
        const int iv = static_cast<int>(std::round(sample.vNorm * static_cast<float>(H)));
        if (iu < 0 || iu >= W || iv < 0 || iv >= H || sample.depth <= 0.0f)
        {
            continue;
        }

        seed.at<uint8_t>(iv, iu) = 255;
        ++projectedSeeds;
    }

    const int seedPixels = cv::countNonZero(seed);
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
    cv::Mat support;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(radius * 2 + 1, radius * 2 + 1));
    cv::dilate(seed, support, kernel);

    const int supportPixels = cv::countNonZero(support);
    const float coverage = static_cast<float>(supportPixels) / static_cast<float>(W * H);
    if (coverage < 0.03f || coverage > 0.95f)
    {
        return cv::Mat();
    }

    fprintf(stderr,
            "[MVS] 帧 %d: 稀疏支撑掩码 seed=%d/%d radius=%d coverage=%d/%d (%.1f%%)\n",
            refIdx,
            seedPixels,
            projectedSeeds,
            radius,
            supportPixels,
            W * H,
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
        fprintf(stderr,
                "[MVS] 帧 %d: 稀疏支撑软约束 support外=%d/%d，置信图不可用，深度保持不变\n",
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

    fprintf(stderr,
            "[MVS] 帧 %d: 稀疏支撑软约束 support外=%d/%d，置信度缩放 %.2f，深度保持不变\n",
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
        fprintf(stderr,
                "[MVS] 帧 %d: 局部深度离群过滤候选过多 %d/%d (%.1f%% > %.1f%%)，已跳过\n",
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

    fprintf(stderr,
            "[MVS] 帧 %d: 局部深度离群过滤移除 %d/%d 像素 (kernel=%d rel=%.2f)\n",
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
        fprintf(stderr,
                "[MVS] 帧 %d: 小连通域 speckle 候选过多 %d/%d (%.1f%% > %.1f%%)，已跳过\n",
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

    fprintf(stderr,
            "[MVS] 帧 %d: 小连通域 speckle 过滤移除 %d/%d 像素 (min_area=%d)\n",
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
                                                                    int viewCount)
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
        fprintf(stderr,
                "[MVS] 帧%d 置信度统计: min=%.4f max=%.4f mean=%.4f (thresh=%.4f)\n",
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
                    fprintf(stderr,
                            "[MVS] 帧%d 低置信满幅深度图: coverage=%.3f mean_conf=%.3f, "
                            "fusion confidence %.3f→%.3f\n",
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
#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int v = 0; v < depthMap.rows; ++v)
            {
                float *depthRow = depthMap.ptr<float>(v);
                const float *confRow = confidenceMap.ptr<float>(v);
                for (int u = 0; u < depthMap.cols; ++u)
                {
                    if (depthRow[u] > 0.0f && confRow[u] < confThresh)
                    {
                        depthRow[u] = 0.0f;
                    }
                }
            }

            int validAfterConfidence = cv::countNonZero(depthMap > 0.0f);
            fprintf(stderr,
                    "[MVS] 帧%d 置信度过滤: %d→%d 有效像素 (thresh=%.4f)\n",
                    refIdx,
                    stats.validBeforePostprocess,
                    validAfterConfidence,
                    confThresh);

            if (!adaptiveConfidenceRaised &&
                validAfterConfidence < stats.validBeforePostprocess / 20)
            {
                fprintf(stderr, "[MVS] 帧%d 置信度过滤后像素太少，回退为不过滤\n", refIdx);
                depthMap = std::move(beforeConfidence);
                validAfterConfidence = stats.validBeforePostprocess;
            }

            stats.validAfterConfidenceFilter = validAfterConfidence;
            stats.confidenceRemoved = std::max(0, stats.validBeforePostprocess - validAfterConfidence);
        }
    }
    stats.effectiveConfidenceThreshold = confThresh;

    if (config.enableLocalDepthOutlierFilter)
    {
        stats.localDepthOutlierRemoved = removeLocalDepthOutliers(
            depthMap,
            confidenceMap,
            config.localDepthOutlierKernelSize,
            config.localDepthOutlierRelThresh,
            config.maxLocalDepthOutlierRemovalRatio,
            refIdx);
    }

    if (config.enableSpeckleFilter)
    {
        stats.smallComponentRemoved = removeSmallDepthComponents(
            depthMap,
            confidenceMap,
            config.minSpeckleComponentArea,
            config.maxSpeckleRemovalRatio,
            refIdx);
        stats.speckleRemoved = stats.smallComponentRemoved;
    }

    stats.validAfterPostprocess = cv::countNonZero(depthMap > 0.0f);
    if (stats.confidenceRemoved > 0
        || stats.localDepthOutlierRemoved > 0
        || stats.smallComponentRemoved > 0)
    {
        fprintf(stderr,
                "[MVS] 帧%d 深度后处理: before=%d after_conf=%d conf_removed=%d local_removed=%d speckle_removed=%d after=%d\n",
                refIdx,
                stats.validBeforePostprocess,
                stats.validAfterConfidenceFilter,
                stats.confidenceRemoved,
                stats.localDepthOutlierRemoved,
                stats.speckleRemoved,
                stats.validAfterPostprocess);
    }
    return stats;
}

// =============================================================================
DepthFrameResult DepthMapGenerator::computeDepthForView(int refIdx, const DepthGenConfig *configOverride)
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

    // 使用预加载的灰度图缓存（无磁盘 I/O）
    cv::Mat refImg;
    if (refIdx >= 0 && refIdx < (int)_grayCache.size() && !_grayCache[refIdx].empty()) {
        refImg = _grayCache[refIdx];  // 浅拷贝，零开销
    } else {
        refImg = xjw::common::io::readImage(refView.imagePath, cv::IMREAD_GRAYSCALE);
    }
    if (refImg.empty()) {
        result.errorMsg = "无法读取参考帧图像: " + refView.imagePath;
        return result;
    }
    if (cancelled("读取参考影像后"))
    {
        return result;
    }
    cv::Mat preparedRefImage;
    Camera refCam;
    std::string preprocessError;
    if (!prepareMvsImage(refImg,
                         refView.camera,
                         &preparedRefImage,
                         &refCam,
                         &preprocessError))
    {
        result.errorMsg = "参考帧预处理失败: " + preprocessError;
        return result;
    }
    refImg = std::move(preparedRefImage);
    const int W = refImg.cols;
    const int H = refImg.rows;

    // 选择源帧（从缓存中取，省去重复加载）
    const int NV = static_cast<int>(_views.size());
    int numSrc = std::min(config.numSourceViews, NV - 1);
    std::vector<cv::Mat> srcGrays;
    std::vector<Camera> srcCams;
    std::vector<int> sourceIndices;

    const std::vector<int> selectedSources = sourceViewIndicesForFrame(refIdx, numSrc);
    for (int si : selectedSources)
    {
        if (si < 0 || si >= NV || si == refIdx)
        {
            continue;
        }
        cv::Mat srcImg;
        if (si >= 0 && si < (int)_grayCache.size() && !_grayCache[si].empty()) {
            srcImg = _grayCache[si];
        } else {
            srcImg = xjw::common::io::readImage(_views[si].imagePath, cv::IMREAD_GRAYSCALE);
        }
        if (srcImg.empty())
        {
            continue;
        }

        cv::Mat preparedSourceImage;
        Camera sourceCamera;
        std::string sourcePreprocessError;
        if (!prepareMvsImage(srcImg,
                             _views[si].camera,
                             &preparedSourceImage,
                             &sourceCamera,
                             &sourcePreprocessError))
        {
            LOG_WARN(QStringLiteral("[MVS] 跳过源帧 %1：影像预处理失败：%2")
                         .arg(si)
                         .arg(QString::fromStdString(sourcePreprocessError)));
            continue;
        }
        srcImg = std::move(preparedSourceImage);
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
        fprintf(stderr, "[MVS] 帧 %d: 共视评分选择源帧 [%s]\n", refIdx, oss.str().c_str());

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
                LOG_INFO(QStringLiteral("[MVS] 帧 %1 source 诊断: %2")
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
            fprintf(stderr,
                    "[MVS] 帧 %d: 共视可见稀疏点为空，回退到参考帧可见点 (%zu)\n",
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
        fprintf(stderr, "[MVS] 帧 %d: 单源视图模式，GPU confidenceThresh=0.10\n",
                refIdx);
    } else if ((int)srcGrays.size() <= 2) {
        // 2 源视图：适当降低阈值但不可过低，0.10 会保留大量低质量匹配→噪声
        pmCfg.confidenceThresh = std::min(pmCfg.confidenceThresh, 0.20f);
    }

    // 诊断输出
    const std::array<double, 9> refRotation = refCam.worldToCameraRotation();
    const std::array<double, 3> refCenter = refCam.cameraCenter();
    fprintf(stderr, "\n[MVS] 帧 %d: 图像 %dx%d, 源帧 %d 个, det(R)=%.4f\n",
            refIdx, W, H, (int)srcCams.size(), det3(refRotation.data()));
    fprintf(stderr, "[MVS] 帧 %d: C=[%.3f, %.3f, %.3f]\n",
            refIdx, refCenter[0], refCenter[1], refCenter[2]);

    // 深度范围
    stageStart = Clock::now();
    float zNear, zFar;
    std::vector<size_t> depthRangeVisiblePoints = visibleSparsePointIndices;
    if (depthRangeVisiblePoints.size() < 5 && minSourceViews > 0)
    {
        depthRangeVisiblePoints = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
        fprintf(stderr,
                "[MVS] 帧 %d: 共视稀疏点不足，深度范围回退到参考帧可见点 (%zu)\n",
                refIdx, depthRangeVisiblePoints.size());
    }
    estimateDepthRangeFromVisiblePoints(refIdx, depthRangeVisiblePoints, zNear, zFar);
    fprintf(stderr, "[MVS] 帧 %d: zNear=%.4f  zFar=%.4f\n", refIdx, zNear, zFar);
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
    cv::Mat referenceValidMask;
    if (refIdx >= 0 && refIdx < static_cast<int>(_validRegionMasks.size()))
    {
        referenceValidMask = _validRegionMasks[static_cast<size_t>(refIdx)];
    }
    const bool has_project_mask = refIdx >= 0 &&
                                  refIdx < static_cast<int>(_projectMaskLoaded.size()) &&
                                  _projectMaskLoaded[static_cast<size_t>(refIdx)] != 0;
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
            fprintf(stderr, "[MVS] 帧 %d: 极线校正成功 (ref=%s)\n",
                    refIdx, refIsCanonicalLeft ? "left" : "right");
        }
        else
        {
            fprintf(stderr, "[MVS] 帧 %d: 极线校正失败(%s)，使用原始图像\n",
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
    pyramid_request.guideImage = workRefImg;
    pyramid_request.referenceCamera = workRefCam;
    pyramid_request.sourceCameras = workSrcCams;
    pyramid_request.zNear = zNear;
    pyramid_request.zFar = zFar;
    pyramid_request.pyramidConfig = pyramid_config;
    pyramid_request.sparseDepthHints = pyramid_sparse_hints;
    pyramid_request.cancelFlag = pmCfg.cancelFlag;

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
        fprintf(stderr,
                "[MVS] 帧 %d: Level %d ds=%d valid=%d coverage=%.1f%% confidence=%.3f elapsed=%.1f ms%s\n",
                refIdx,
                summary.level,
                summary.downsampleFactor,
                summary.validPixelCount,
                summary.validCoverage * 100.0f,
                summary.meanConfidence,
                summary.elapsedMs,
                summary.success ? "" : " failed");
    }
    if (!pyramid_result.errorMessage.empty())
    {
        fprintf(stderr,
                "[MVS] 帧 %d: 三级估计降级使用 Level %d (%s)\n",
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
        fprintf(stderr, "[MVS] 帧 %d: 深度图已从校正空间映射回原始空间\n", refIdx);
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
        const bool hasPreloadedMaskSlot = refIdx >= 0 && refIdx < static_cast<int>(_validRegionMasks.size());
        if (hasPreloadedMaskSlot && !_validRegionMasks[static_cast<size_t>(refIdx)].empty())
        {
            effectiveReferenceMask = _validRegionMasks[static_cast<size_t>(refIdx)];
        }
        else if (hasPreloadedMaskSlot)
        {
            fprintf(stderr, "[MVS] 帧 %d: 内容掩码已自动跳过\n", refIdx);
        }
        else
        {
            float coverage = 0.f;
            double otsuThresh = 0.0;
            int adaptiveThresh = 0;
            effectiveReferenceMask = buildContentMask(
                refImg, &coverage, &otsuThresh, &adaptiveThresh);
            if (effectiveReferenceMask.empty())
            {
                fprintf(stderr,
                        "[MVS] 帧 %d: 内容掩码覆盖 %.1f%%，自动跳过内容掩码过滤\n",
                        refIdx,
                        coverage * 100.0f);
            }
            else
            {
                fprintf(stderr,
                        "[MVS] 帧 %d: 警告 - 内容掩码不可用，现场生成 (覆盖 %.1f%%, Otsu=%.0f adaptiveThresh=%d)\n",
                        refIdx,
                        coverage * 100.0f,
                        otsuThresh,
                        adaptiveThresh);
            }
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
                fprintf(stderr, "[MVS] 帧 %d: 有效区域过滤 %d→%d 有效像素\n",
                        refIdx, beforeMask, afterMask);
            }
        }
    }

    if (effectiveReferenceMask.empty())
    {
        effectiveReferenceMask = cv::Mat(depthMap.size(), CV_8UC1, cv::Scalar(255));
    }
    result.depthCompleteness.afterMaskValidCount = cv::countNonZero(depthMap > 0.0f);

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
            fprintf(stderr,
                    "[MVS] 帧 %d: 输出前局部深度突刺过滤移除 %d 像素\n",
                    refIdx,
                    previewOutliers);
        }
        result.depthCompleteness.outputFilterRemovedCount = previewOutliers;
    }
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
    LOG_INFO(QStringLiteral(
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

    // 统计
    int validCnt = cv::countNonZero(depthMap > 0);
    fprintf(stderr, "[MVS] 帧 %d: PatchMatch 完成, 有效深度像素=%d/%d (%.1f%%)\n",
            refIdx, validCnt, W*H, 100.f * validCnt / (W*H));
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
        frame.depthPostprocess = postprocessFusionDepthMap(filteredDepth,
                                                           filteredConfidence,
                                                           fusion_config,
                                                           res.refViewIdx,
                                                           static_cast<int>(_views.size()));
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
    std::vector<cv::Mat> origDepths(NV);
    for (int i = 0; i < NV; ++i)
    {
        if (_cancelled.load())
        {
            return;
        }
        if (_depthFrames[i].eligibleAsConsistencySource() && _depthFrames[i].depthMap)
        {
            origDepths[i] = _depthFrames[i].depthMap->clone();
        }
    }

    for (int i = 0; i < NV; ++i)
    {
        if (_cancelled.load())
        {
            return;
        }
        if (!_depthFrames[i].eligibleForFusion() || !_depthFrames[i].depthMap)
        {
            continue;
        }
        cv::Mat &depthI = *_depthFrames[i].depthMap;
        const Camera camI = _depthFrames[i].cameraModel.isValid()
            ? _depthFrames[i].cameraModel
            : mvsPinholeCamera(_views[i].camera);

        // 备份，万一过滤太激进需要回退
        cv::Mat depthBackup = depthI.clone();

        const int rowWorkers = std::max(1, _config.cpuWorkerCount);
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
        std::vector<cv::Mat> projected_sources;
        if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            projected_sources.resize(repairSources.size());
        }

        for (int source_ordinal = 0;
             source_ordinal < static_cast<int>(repairSources.size());
             ++source_ordinal)
        {
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
                accumulateDepthConsistency(depthI,
                                           camI,
                                           depthJ,
                                           camJ,
                                           source_ordinal,
                                           relThresh,
                                           rowWorkers,
                                           _cancelled,
                                           consistent_votes,
                                           occluded_votes,
                                           contradicted_votes,
                                           unverifiable_votes,
                                           geometry_source_mask,
                                           source_inverse_depth_sum,
                                           source_inverse_depth_squared_sum);
            }
            if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                projected_sources[static_cast<std::size_t>(source_ordinal)] =
                    projectSourceDepthToReference(
                    depthJ,
                    camJ,
                    camI,
                    depthI.size(),
                    repair_options.maximumProjectionDistancePixels);
            }
        }

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
            contradicted_votes);

        // 剔除不一致像素
        depthI.setTo(0, consistentMask == 0);
        _depthFrames[i].crossViewRepairedMask = QSharedPointer<cv::Mat>::create();
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
                i >= 0 && i < static_cast<int>(_grayCache.size())
                    ? &_grayCache[static_cast<std::size_t>(i)] : nullptr);
        _depthFrames[i].depthCompleteness.crossViewRepairedCount +=
            static_cast<int>(repair_stats.repairedPixelCount);
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
        fprintf(stderr, "[MVS] 帧%d 一致性检查(%s, 检查/修复源 %zu/%zu, 最少确认 %d, 跨视补回 %llu, 锚定插值 %llu, 两源生长 %llu/%llu): %d→%d 有效像素 (保留 %.1f%%)\n",
                i, fewViews ? "仅移除矛盾" : "需要确认",
                consistencySources.size(), repairSources.size(),
                minimum_source_confirmations,
                static_cast<unsigned long long>(repair_stats.repairedPixelCount),
                static_cast<unsigned long long>(
                    repair_stats.anchoredInterpolation.interpolatedPixelCount),
                static_cast<unsigned long long>(repair_stats.twoSourceGrownPixelCount),
                static_cast<unsigned long long>(repair_stats.twoSourceCandidatePixelCount),
                beforeValid, afterValid, keepRate);

        // 安全回退：如果保留率过低（< 10%），回退使用原始深度图
        if (afterValid < beforeValid / 10 && beforeValid > 100) 
        {
            fprintf(stderr, "[MVS] 帧%d 一致性过滤后像素太少 (%.1f%%)，回退为原始深度图\n",
                    i, keepRate);
            depthBackup.copyTo(depthI);
            afterValid = beforeValid;
        }
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
        const float consistency_keep_rate = beforeValid > 0
            ? static_cast<float>(afterValid) / static_cast<float>(beforeValid)
            : 0.0f;
        _depthFrames[i].depthCompleteness.preConsistencyValidCount = beforeValid;
        _depthFrames[i].depthCompleteness.postConsistencyValidCount = afterValid;
        _depthFrames[i].depthCompleteness.consistencyRetentionRatio =
            consistency_keep_rate;
        _depthFrames[i].depthCompleteness.consistencyConfirmedObservationCount =
            static_cast<int>(cv::sum(consistent_votes)[0]);
        _depthFrames[i].depthCompleteness.consistencyOccludedObservationCount =
            static_cast<int>(cv::sum(occluded_votes)[0]);
        _depthFrames[i].depthCompleteness.consistencyContradictedObservationCount =
            static_cast<int>(cv::sum(contradicted_votes)[0]);
        _depthFrames[i].depthCompleteness.consistencyUnverifiableObservationCount =
            static_cast<int>(cv::sum(unverifiable_votes)[0]);
        _depthFrames[i].depthCompleteness.consistencyRejectedPixelCount =
            std::max(0, beforeValid - afterValid);
        updateDepthFrameQualityAfterConsistency(_depthFrames[i],
                                                depthI,
                                                restoration_confidence,
                                                _effectiveSceneProfile,
                                                _effectiveDepthFilterMode,
                                                consistency_keep_rate);
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
        }
    };

    const int row_workers = std::max(1, _config.cpuWorkerCount);
    int completed_frames = 0;

    for (int frame_index = 0; frame_index < view_count; ++frame_index)
    {
        if (_cancelled.load())
        {
            remove_pending_files();
            return false;
        }
        if (!_depthFrames[frame_index].eligibleForFusion())
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
        std::vector<cv::Mat> projected_sources;
        if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
        {
            projected_sources.resize(repair_source_indices.size());
        }

        for (int source_ordinal = 0;
             source_ordinal < static_cast<int>(repair_source_indices.size());
             ++source_ordinal)
        {
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
            if (std::find(
                    source_indices.begin(), source_indices.end(), source_index) !=
                source_indices.end())
            {
                accumulateDepthConsistency(
                    reference->depth,
                    reference_camera,
                    source->depth,
                    _depthFrames[source_index].cameraModel.isValid()
                        ? _depthFrames[source_index].cameraModel
                        : mvsPinholeCamera(_views[source_index].camera),
                    source_ordinal,
                    relative_threshold,
                    row_workers,
                    _cancelled,
                    consistent_votes,
                    occluded_votes,
                    contradicted_votes,
                    unverifiable_votes,
                    geometry_source_mask,
                    source_inverse_depth_sum,
                    source_inverse_depth_squared_sum);
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
                    repair_options.maximumProjectionDistancePixels);
            }
        }

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
        const cv::Mat consistent_mask = makeDepthConsistencyMask(
            filtered_depth,
            static_cast<int>(source_indices.size()),
            minimum_source_confirmations,
            consistent_votes,
            occluded_votes,
            contradicted_votes);

        const int valid_before = cv::countNonZero(reference->depth > 0.0f);
        filtered_depth.setTo(0.0f, consistent_mask == 0);
        _depthFrames[frame_index].crossViewRepairedMask =
            QSharedPointer<cv::Mat>::create();
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
                frame_index >= 0 && frame_index < static_cast<int>(_grayCache.size())
                    ? &_grayCache[static_cast<std::size_t>(frame_index)] : nullptr);
        _depthFrames[frame_index].depthCompleteness.crossViewRepairedCount +=
            static_cast<int>(repair_stats.repairedPixelCount);
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
            valid_after = valid_before;
        }
        const float consistency_keep_rate = valid_before > 0
            ? static_cast<float>(valid_after) / static_cast<float>(valid_before)
            : 0.0f;
        _depthFrames[frame_index].depthCompleteness.preConsistencyValidCount =
            valid_before;
        _depthFrames[frame_index].depthCompleteness.postConsistencyValidCount =
            valid_after;
        _depthFrames[frame_index].depthCompleteness.consistencyRetentionRatio =
            consistency_keep_rate;
        _depthFrames[frame_index].depthCompleteness.consistencyConfirmedObservationCount =
            static_cast<int>(cv::sum(consistent_votes)[0]);
        _depthFrames[frame_index].depthCompleteness.consistencyOccludedObservationCount =
            static_cast<int>(cv::sum(occluded_votes)[0]);
        _depthFrames[frame_index].depthCompleteness.consistencyContradictedObservationCount =
            static_cast<int>(cv::sum(contradicted_votes)[0]);
        _depthFrames[frame_index].depthCompleteness.consistencyUnverifiableObservationCount =
            static_cast<int>(cv::sum(unverifiable_votes)[0]);
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
        _depthFrames[frame_index].depthPostprocess = postprocessFusionDepthMap(
            filtered_depth,
            filtered_confidence,
            fusion_config,
            frame_index,
            view_count);
        _depthFrames[frame_index].depthPostprocessApplied = true;
        const DepthAnchoredHoleInterpolationStats final_repair =
            repairPostprocessedInternalDepthHoles(
                _depthFrames[frame_index],
                filtered_depth,
                filtered_confidence,
                _effectiveSceneProfile);
        _depthFrames[frame_index].depthCompleteness.crossViewRepairedCount +=
            static_cast<int>(final_repair.interpolatedPixelCount);
        if (final_repair.interpolatedPixelCount > 0)
        {
            fprintf(
                stderr,
                "[MVS] 帧%d 最终后处理后锚定修复 %llu 像素 (%llu 个内部连通域)\n",
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
                                        frame_index});

        ++completed_frames;
        const float ratio = static_cast<float>(completed_frames) /
                            static_cast<float>(std::max(1, view_count));
        emit progressChanged(
            QStringLiteral("多视一致性：已处理 %1/%2").arg(completed_frames).arg(view_count),
            ratio);
        LOG_INFO(QStringLiteral(
                     "[MVS] 流式一致性 frame=%1 valid=%2->%3 sources=%4 "
                     "twoSource=%5/%6 cache=%7/%8 MiB")
                     .arg(frame_index)
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
    const std::string crossViewRepairedMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) +
        "_cross_view_repaired_mask.png";
    const std::string validMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_mask.png";
    const std::string supportMaskPath =
        raw_directory + "/depth_" + std::to_string(frameIndex) + "_support_mask.png";
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
        LOG_INFO(QStringLiteral("[MVS] 帧 %1 %2深度预览已保存: %3 (%4x%5)")
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
        cv::Mat supportMask;
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
                                      const QString &label)
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
        stored.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
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
        QStringLiteral("跨视来源掩码"));
    const bool inverseDepthMeanSaved = save_geometry_evidence(
        result.inverseDepthMean,
        CV_32FC1,
        rawInverseDepthMeanPath,
        QStringLiteral("逆深度均值图"));
    const bool inverseDepthSpreadSaved = save_geometry_evidence(
        result.inverseDepthRelativeSpread,
        CV_32FC1,
        rawInverseDepthSpreadPath,
        QStringLiteral("逆深度离散度图"));
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

    LOG_INFO(QStringLiteral("[MVS] 保存%1深度产物耗时: frame=%2 preview=%3 ms raw=%4 ms confidence=%5 ms mask=%6 ms total=%7 ms")
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
            if (hasSourceScoreCache)
            {
                const auto &scores = _frameCaches[static_cast<size_t>(frameIndex)].sourceViewScores;
                const auto it = std::find_if(scores.begin(),
                                             scores.end(),
                                             [sourceIndex](const MvsSourcePlanEntry &entry)
                                             {
                                                 return entry.viewIndex == sourceIndex;
                                             });
                if (it != scores.end())
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
        const int requestedSourceViewCount = hasSourceScoreCache
            ? _frameCaches[static_cast<size_t>(frameIndex)].requestedSourceViewCount
            : sourceQualitySummary.sourceViewCount;
        const int sourceViewShortfall = std::max(
            0,
            requestedSourceViewCount - sourceQualitySummary.sourceViewCount);
        depthQualityJson[QStringLiteral("requested_source_view_count")] = requestedSourceViewCount;
        depthQualityJson[QStringLiteral("source_view_shortfall")] = sourceViewShortfall;
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
                level_object.insert(QStringLiteral("cross_view_repaired_mask_path"),
                                    crossViewRepairedMaskSaved
                                        ? QString::fromStdString(crossViewRepairedMaskPath)
                                        : QString());
                level_object.insert(QStringLiteral("valid_mask_path"),
                                    maskSaved ? QString::fromStdString(validMaskPath) : QString());
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
        artifact[QStringLiteral("cross_view_repaired_mask_path")] =
            crossViewRepairedMaskSaved
                ? QString::fromStdString(crossViewRepairedMaskPath)
                : QString();
        artifact[QStringLiteral("valid_mask_path")] = maskSaved ? QString::fromStdString(validMaskPath) : QString();
        artifact[QStringLiteral("support_mask_path")] =
            supportMaskSaved ? QString::fromStdString(supportMaskPath) : QString();
        artifact[QStringLiteral("ref_image")] = QString::fromStdString(_views[frameIndex].imagePath);
        artifact[QStringLiteral("source_images")] = sourceImages;
        artifact[QStringLiteral("source_indices")] = sourceIndices;
        artifact[QStringLiteral("source_plan")] = sourcePlan;
        artifact[QStringLiteral("source_view_count")] = sourceQualitySummary.sourceViewCount;
        artifact[QStringLiteral("requested_source_view_count")] = requestedSourceViewCount;
        artifact[QStringLiteral("source_view_shortfall")] = sourceViewShortfall;
        artifact[QStringLiteral("verified_source_view_count")] =
            sourceQualitySummary.verifiedSourceViewCount;
        artifact[QStringLiteral("backfill_source_view_count")] =
            sourceQualitySummary.backfillSourceViewCount;
        artifact[QStringLiteral("source_quality_mean")] = sourceQualitySummary.meanQuality;
        artifact[QStringLiteral("source_quality_min")] = sourceQualitySummary.minQuality;
        artifact[QStringLiteral("depth_confidence_mean")] = depthConfidenceSummary.meanConfidence;
        artifact[QStringLiteral("valid_pixel_count")] = depthConfidenceSummary.validPixelCount;
        artifact[QStringLiteral("valid_coverage")] =
            static_cast<double>(result.qualityMetrics.validCoverage);
        artifact[QStringLiteral("depth_quality")] = depthQualityJson;
        artifact[QStringLiteral("depth_completeness")] = depthCompletenessJson;
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
        record.sourceViewCount = sourceQualitySummary.sourceViewCount;
        record.meanSourceQualityScore = sourceQualitySummary.meanQuality;
        record.minSourceQualityScore = sourceQualitySummary.minQuality;
        record.meanDepthConfidence = depthConfidenceSummary.meanConfidence;
        record.validPixelCount = depthConfidenceSummary.validPixelCount;
        record.validCoverage = static_cast<double>(result.qualityMetrics.validCoverage);
        record.depthQuality = depthQualityJson;
        record.depthCompleteness = depthCompletenessJson;
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
        record.crossViewRepairedMaskPath = crossViewRepairedMaskSaved
            ? QString::fromStdString(crossViewRepairedMaskPath)
            : QString();
        record.validMaskPath = maskSaved ? QString::fromStdString(validMaskPath) : QString();
        record.supportMaskPath = supportMaskSaved ? QString::fromStdString(supportMaskPath) : QString();
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

// =============================================================================
void DepthMapGenerator::runInBackground()
{
    bool allOk = true;
    const int NV = static_cast<int>(_views.size());

    if (!_config.runDepthEstimation)
    {
        emit errorOccurred(QStringLiteral("当前生成器配置未启用深度估计阶段"));
        emit finished(false);
        return;
    }

    _sceneClassification = classifyMvsScene(_views, _sparse);
    _effectiveSceneProfile = _config.sceneProfile == MvsSceneProfile::Auto
        ? _sceneClassification.profile
        : _config.sceneProfile;
    _effectiveDepthFilterMode = _config.depthFilterMode;
    if (_config.adaptiveDepthFilterMode)
    {
        _effectiveDepthFilterMode = _effectiveSceneProfile == MvsSceneProfile::AerialTerrain
            ? DepthFilterMode::Moderate
            : DepthFilterMode::Mild;
    }
    const int configured_source_count = _config.numSourceViews;
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

    initializeWorkspaceManifest();

    // ── 预加载所有图像（一次性磁盘 I/O，后续全从内存读取）────────────────────
    emit progressChanged(QString("预加载 %1 张图像...").arg(NV), 0.f);
    preloadImages();
    refreshViewImageDimensionsFromCache();
    if (_cancelled.load())
    {
        clearFrameCaches();
        emit finished(false);
        return;
    }

    emit progressChanged(QStringLiteral("预计算 MVS 可见性..."), 0.02f);
    prepareFrameCaches();
    if (_cancelled.load())
    {
        clearFrameCaches();
        emit finished(false);
        return;
    }

    _depthFrames.resize(NV);
    const bool savePreviewPng = !_outputDir.empty();
    const SystemMemorySnapshot initialMemory = querySystemMemorySnapshot();
    const uint64_t estimatedDepthCacheBytes = estimateDepthFrameCacheBytes(_views);
    const uint64_t largestFrameBytes = largestDepthFrameBytes(_views);
    QString memoryPolicyReason;
    bool retainDepthFrames = shouldRetainAllDepthFramesInMemory(_views, _config, initialMemory, &memoryPolicyReason);

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
            clearFrameCaches();
            emit finished(false);
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

    LOG_INFO(QStringLiteral("[MVS] 深度图内存策略: mode=%1 estimatedCache=%2 GiB total=%3 GiB available=%4 GiB reserve=%5 GiB maxFraction=%6 reason=%7")
                 .arg(retainDepthFrames ? QStringLiteral("cache") : QStringLiteral("stream"))
                 .arg(bytesToGiB(estimatedDepthCacheBytes), 0, 'f', 2)
                 .arg(initialMemory.valid ? bytesToGiB(initialMemory.totalPhysicalBytes) : 0.0, 0, 'f', 2)
                 .arg(initialMemory.valid ? bytesToGiB(initialMemory.availablePhysicalBytes) : 0.0, 0, 'f', 2)
                 .arg(bytesToGiB(_config.minFreeRamBytes), 0, 'f', 2)
                 .arg(static_cast<double>(_config.maxDepthCacheRamFraction), 0, 'f', 2)
                 .arg(memoryPolicyReason));
    if (!keepDepthFramesInMemory.load())
    {
        LOG_INFO(QStringLiteral("[MVS] 深度图估计采用流式保存模式：保存后释放全分辨率深度/置信图"));
    }

    // ── 阶段一：优先级队列并行估计深度图（GPU 优先高价值帧，CPU 处理其余帧）────
    int skippedFrames = 0;
    for (size_t i = 0; i < _skipFrameMask.size(); ++i)
    {
        if (_skipFrameMask[i] != 0)
        {
            ++skippedFrames;
        }
    }

    std::atomic<int> completedTasks{skippedFrames};
    std::atomic<int> activeGpuTasks{0};
    std::atomic<int> activeCpuTasks{0};
    std::atomic<bool> anyFailure{false};
    std::mutex taskMutex;

    struct FramePriority
    {
        int viewIndex = -1;
        float score = 0.f;
    };

    const bool cudaAvailable = _config.patchMatch.useCuda && PatchMatchDepthEstimator::isCudaAvailable();
    const int cpuThreadCount = std::max(1, _config.cpuWorkerCount);

    std::vector<FramePriority> framePriorities;
    framePriorities.reserve(static_cast<size_t>(NV));
    for (int i = 0; i < NV; ++i)
    {
        if (i >= 0 && i < static_cast<int>(_skipFrameMask.size()) && _skipFrameMask[static_cast<size_t>(i)] != 0)
        {
            continue;
        }

        float resolutionScore = 0.f;
        if (i >= 0 && i < static_cast<int>(_grayCache.size()) && !_grayCache[i].empty())
        {
            resolutionScore = static_cast<float>(_grayCache[i].cols * _grayCache[i].rows);
        }

        float contentRatio = 0.5f;
        if (i >= 0 && i < static_cast<int>(_validRegionMasks.size()) &&
            !_validRegionMasks[static_cast<size_t>(i)].empty())
        {
            const cv::Mat &valid_region_mask = _validRegionMasks[static_cast<size_t>(i)];
            const int totalPx = valid_region_mask.rows * valid_region_mask.cols;
            if (totalPx > 0)
            {
                contentRatio = static_cast<float>(cv::countNonZero(valid_region_mask)) /
                               static_cast<float>(totalPx);
            }
        }

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

    std::deque<int> gpuQueue;
    std::deque<int> cpuQueue;
    if (cudaAvailable)
    {
        for (const FramePriority &priority : framePriorities)
        {
            gpuQueue.push_back(priority.viewIndex);
        }
    }
    else
    {
        for (const FramePriority &priority : framePriorities)
        {
            cpuQueue.push_back(priority.viewIndex);
        }
    }

    const int pendingFrameCount = static_cast<int>(framePriorities.size());
    const int maxWorkersByPendingFrames = std::max(1, pendingFrameCount);
    const int maxFrameWorkers = std::min(4, maxWorkersByPendingFrames);
    const int gpuFrameWorkers = cudaAvailable
        ? std::clamp(std::max(1, _config.gpuFrameWorkerCount), 1, maxFrameWorkers)
        : 0;
    const int cpuFrameWorkers = cudaAvailable
        ? std::clamp(std::max(0, _config.cpuFrameWorkerCount), 0, maxFrameWorkers)
        : std::clamp(std::max(1, _config.cpuFrameWorkerCount), 1, maxFrameWorkers);

    LOG_INFO(QStringLiteral("[MVS] 深度估计调度: cuda=%1 gpu_frame_workers=%2 cpu_frame_workers=%3 cpu_pixel_threads=%4 views=%5 gpuQueue=%6 cpuQueue=%7")
                 .arg(cudaAvailable ? QStringLiteral("on") : QStringLiteral("off"))
                 .arg(gpuFrameWorkers)
                 .arg(cpuFrameWorkers)
                 .arg(cpuThreadCount)
                 .arg(NV)
                 .arg(static_cast<qulonglong>(gpuQueue.size()))
                 .arg(static_cast<qulonglong>(cpuQueue.size())));
    if (skippedFrames > 0)
    {
        LOG_INFO(QStringLiteral("[MVS] 续跑模式：跳过已存在深度图 %1 帧").arg(skippedFrames));
    }
    if (cudaAvailable)
    {
        LOG_INFO(QStringLiteral("[MVS] CUDA 已启用，GPU 帧并发=%1；每帧内部使用 CUDA kernel，显存不足时可降低 gpu_frame_workers")
                     .arg(gpuFrameWorkers));
    }

    const size_t maxBufferedSaveTasks =
        adaptiveSaveQueueCapacity(initialMemory, _config, largestFrameBytes);
    DepthFrameArtifactSaveQueue saveQueue(
        [this](int frameIndex, const DepthFrameResult &result, const QString &stageLabel)
        {
            return saveDepthFrameArtifacts(frameIndex, result, stageLabel);
        },
        maxBufferedSaveTasks);

    auto emitDepthProgress = [this, NV, &completedTasks, &activeGpuTasks, &activeCpuTasks, &taskMutex, &gpuQueue, &cpuQueue](
                                 const QString &workerTag,
                                 int frameIndex,
                                 bool pickedTask)
    {
        int gpuPending = 0;
        int cpuPending = 0;
        {
            std::lock_guard<std::mutex> lock(taskMutex);
            gpuPending = static_cast<int>(gpuQueue.size());
            cpuPending = static_cast<int>(cpuQueue.size());
        }

        const int done = completedTasks.load();
        const int gpuActive = activeGpuTasks.load();
        const int cpuActive = activeCpuTasks.load();
        const float ratio = static_cast<float>(done) / (NV + 2);

        QString stage = QStringLiteral("深度估计: 已完成 %1/%2, 运行中 GPU %3 CPU %4, 待处理 GPU %5 CPU %6")
                            .arg(done)
                            .arg(NV)
                            .arg(gpuActive)
                            .arg(cpuActive)
                            .arg(gpuPending)
                            .arg(cpuPending);

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

    auto workerFunc = [this,
                       NV,
                       largestFrameBytes,
                       &keepDepthFramesInMemory,
                       &depthFramesMutex,
                       &completedTasks,
                       &activeGpuTasks,
                       &activeCpuTasks,
                       &anyFailure,
                       &taskMutex,
                       &gpuQueue,
                       &cpuQueue,
                       &emitDepthProgress,
                       &saveQueue](bool useGpu) {
        DepthGenConfig workerConfig = _config;
        workerConfig.patchMatch.useCuda = useGpu;
        workerConfig.patchMatch.cancelFlag = &_cancelled;
        const QString workerTag = useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU");

        while (!_cancelled)
        {
            int i = -1;
            {
                std::lock_guard<std::mutex> lock(taskMutex);
                if (useGpu)
                {
                    if (!gpuQueue.empty())
                    {
                        i = gpuQueue.front();
                        gpuQueue.pop_front();
                    }
                    else if (!cpuQueue.empty())
                    {
                        i = cpuQueue.front();
                        cpuQueue.pop_front();
                    }
                }
                else
                {
                    if (!cpuQueue.empty())
                    {
                        i = cpuQueue.front();
                        cpuQueue.pop_front();
                    }
                    else if (!gpuQueue.empty())
                    {
                        i = gpuQueue.front();
                        gpuQueue.pop_front();
                    }
                }
            }

            if (i < 0 || i >= NV)
            {
                break;
            }

            if (useGpu)
            {
                activeGpuTasks.fetch_add(1);
            }
            else
            {
                activeCpuTasks.fetch_add(1);
            }
            emitDepthProgress(workerTag, i, true);

            const auto frameStart = std::chrono::steady_clock::now();
            LOG_INFO(QStringLiteral("[MVS] 帧 %1 开始深度估计: device=%2")
                         .arg(i)
                         .arg(useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU")));
            markManifestFrameRunning(i);

            DepthFrameResult res = computeDepthForView(i, &workerConfig);
            if (_cancelled.load())
            {
                LOG_INFO(QStringLiteral("[MVS] 帧 %1 收到取消请求，跳过结果保存").arg(i));
                if (useGpu)
                {
                    activeGpuTasks.fetch_sub(1);
                }
                else
                {
                    activeCpuTasks.fetch_sub(1);
                }
                emitDepthProgress(workerTag, i, false);
                break;
            }

            const auto frameEnd = std::chrono::steady_clock::now();
            const double elapsedMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
            res.device = useGpu ? "GPU" : "CPU";
            res.elapsedMs = elapsedMs;

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
                LOG_WARN(QStringLiteral("[MVS] 帧 %1 深度估计失败: device=%2 elapsed=%3 ms error=%4")
                             .arg(i)
                             .arg(useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU"))
                             .arg(elapsedMs, 0, 'f', 1)
                             .arg(QString::fromStdString(res.errorMsg)));
                markManifestFrameFailed(i, QString::fromStdString(res.errorMsg));
                anyFailure = true;
            }
            else
            {
                const int depthWidth = (res.depthMap && !res.depthMap->empty()) ? res.depthMap->cols : 0;
                const int depthHeight = (res.depthMap && !res.depthMap->empty()) ? res.depthMap->rows : 0;
                LOG_INFO(QStringLiteral("[MVS] 帧 %1 深度估计完成: device=%2 elapsed=%3 ms size=%4x%5")
                             .arg(i)
                             .arg(useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU"))
                             .arg(elapsedMs, 0, 'f', 1)
                             .arg(depthWidth)
                             .arg(depthHeight));
                saveQueue.enqueue(i, res, QStringLiteral("初始"));

                if (keepDepthFramesInMemory.load() &&
                    memoryPressureRequiresStreaming(_config, querySystemMemorySnapshot(), largestFrameBytes))
                {
                    bool expected = true;
                    if (keepDepthFramesInMemory.compare_exchange_strong(expected, false))
                    {
                        LOG_WARN(QStringLiteral(
                            "[MVS] 内存压力升高，切换为流式保存并释放已缓存深度图；"
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
            if (useGpu)
            {
                activeGpuTasks.fetch_sub(1);
            }
            else
            {
                activeCpuTasks.fetch_sub(1);
            }
            Q_UNUSED(done);
            emitDepthProgress(workerTag, i, false);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(cpuFrameWorkers + gpuFrameWorkers));

    for (int workerIndex = 0; workerIndex < gpuFrameWorkers; ++workerIndex)
    {
        workers.emplace_back(workerFunc, true);
    }
    for (int workerIndex = 0; workerIndex < cpuFrameWorkers; ++workerIndex)
    {
        workers.emplace_back(workerFunc, false);
    }

    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    if (_cancelled.load())
    {
        saveQueue.cancel();
        saveQueue.stop();
        _grayCache.clear();
        _grayCache.shrink_to_fit();
        _validRegionMasks.clear();
        _validRegionMasks.shrink_to_fit();
        _projectMaskLoaded.clear();
        _projectMaskLoaded.shrink_to_fit();
        clearFrameCaches();
        emit finished(false);
        return;
    }

    saveQueue.waitUntilIdle();
    if (saveQueue.failed())
    {
        anyFailure = true;
    }

    // 释放图像缓存（深度估计完毕后不再需要）
    _grayCache.clear();
    _grayCache.shrink_to_fit();
    _validRegionMasks.clear();
    _validRegionMasks.shrink_to_fit();
    _projectMaskLoaded.clear();
    _projectMaskLoaded.shrink_to_fit();
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
            emit finished(false);
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
            emit finished(false);
            return;
        }
    }

    if (keepDepthFramesInMemory.load() && (savePreviewPng || saveRawDepth))
    {
        for (int i = 0; i < NV; ++i)
        {
            if (_cancelled.load())
            {
                saveQueue.cancel();
                saveQueue.stop();
                emit finished(false);
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
                res.depthPostprocess = postprocessFusionDepthMap(*res.depthMap,
                                                                  confidence,
                                                                  fusion_config,
                                                                  res.refViewIdx,
                                                                  static_cast<int>(_views.size()));
                res.depthPostprocessApplied = true;
                const DepthAnchoredHoleInterpolationStats final_repair =
                    repairPostprocessedInternalDepthHoles(
                        res,
                        *res.depthMap,
                        confidence,
                        _effectiveSceneProfile);
                res.depthCompleteness.crossViewRepairedCount +=
                    static_cast<int>(final_repair.interpolatedPixelCount);
                updateDepthCompletenessAfterPostprocess(
                    res, *res.depthMap, res.depthPostprocess);
                const int valid_before = res.depthPostprocess.validBeforePostprocess;
                const float keep_rate = valid_before > 0
                    ? static_cast<float>(res.depthPostprocess.validAfterPostprocess)
                        / static_cast<float>(valid_before)
                    : 0.0f;
                updateDepthFrameQualityAfterConsistency(
                    res,
                    *res.depthMap,
                    confidence,
                    _effectiveSceneProfile,
                    _effectiveDepthFilterMode,
                    keep_rate);
            }
            if (_cancelled.load())
            {
                saveQueue.cancel();
                saveQueue.stop();
                emit finished(false);
                return;
            }
            saveQueue.enqueue(i, res, QStringLiteral("过滤后"));
        }
        saveQueue.waitUntilIdle();
        if (saveQueue.failed())
        {
            anyFailure = true;
        }
    }
    saveQueue.stop();

    allOk = !anyFailure.load();

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
        emit progressChanged("完成", 1.f);
        emit finished(allOk);
        return;
    }

    // ── 阶段二：COLMAP BFS 深度图融合 → 直接输出 3D 点 ──────────────────────
    emit progressChanged("深度图融合...", static_cast<float>(NV) / (NV + 2));

    std::vector<FusionFrameInput> frames;
    for (const auto &fr : _depthFrames)
    {
        if (fr.eligibleForFusion() && fr.depthMap && !fr.depthMap->empty())
        {
            frames.push_back(buildFusionFrame(fr));
        }
    }

    if (frames.empty()) {
        emit errorOccurred("没有有效的深度帧，融合失败");
        emit finished(false);
        return;
    }

    for (auto &frame : frames)
    {
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

    fprintf(stderr, "[MVS] COLMAP BFS 融合: %d 帧参与\n", (int)frames.size());

    // 使用新的 StereoFusionConfig
    StereoFusionConfig fusionCfg;
    fusionCfg.minNumPixels   = _config.fusion.minConsistentViews;
    fusionCfg.maxReprojError = _config.fusion.pixelThresh;
    fusionCfg.maxDepthError  = _config.fusion.relDepthThresh;
    fusionCfg.checkNumImages = std::min(50, NV);
    fusionCfg.workerCount    = std::max(1, _config.cpuWorkerCount);
    fusionCfg.requireValidMask = true;
    fusionCfg.minSupportViews = std::max(1, _config.fusion.minConsistentViews - 1);
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
        fprintf(stderr, "[MVS] 少视图模式 (%d 帧)，minNumPixels=1, 放宽融合阈值\n", NV);
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
            fprintf(stderr, "[MVS] 包围盒: [%.2f,%.2f,%.2f] ~ [%.2f,%.2f,%.2f]\n",
                    fusionCfg.bboxMin[0], fusionCfg.bboxMin[1], fusionCfg.bboxMin[2],
                    fusionCfg.bboxMax[0], fusionCfg.bboxMax[1], fusionCfg.bboxMax[2]);
        }
        else
        {
            fprintf(stderr, "[MVS] 稀疏云与相机坐标系不匹配 (cloudExtent=%.1f camExtent=%.1f ratio=%.1f)"
                            "，禁用包围盒过滤\n", cloudExtent, camExtent, scaleRatio);
            // 坐标系不兼容时深度初始化无先验，放宽一致性阈值至 10%
            fusionCfg.maxDepthError = std::max(fusionCfg.maxDepthError, 0.10f);
            fprintf(stderr, "[MVS] 坐标系不兼容，自动放宽 maxDepthError → %.2f\n",
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
        fprintf(stderr, "[MVS] 融合失败: %s\n", fuseErr.c_str());
        emit errorOccurred(QString::fromStdString(fuseErr));
        emit finished(false);
        return;
    }

    std::vector<DensePoint> cloud;
    cloud.reserve(fusedPoints.size());
    for (const FusedPoint &point : fusedPoints)
    {
        cloud.push_back(DensePoint{
            point.x, point.y, point.z,
            point.r, point.g, point.b});
    }

    // 保存每帧一致性过滤深度图（加锁，防止 GUI 线程并发读取）
    {
        std::lock_guard<std::mutex> lock(_filteredDepthsMutex);
        _filteredDepths = fusion.filteredDepths();
    }

    fprintf(stderr, "[MVS] 融合完成: %d 个稠密点\n", (int)cloud.size());

    if (cloud.empty())
    {
        fprintf(stderr, "[MVS] 警告: 融合产出 0 个点，可能原因：\n"
                        "  - 深度图质量不足（过少有效像素）\n"
                        "  - 视图数量太少（当前 %d 帧，建议 >=3）\n"
                        "  - 深度一致性阈值过严\n",
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
                if (cloud.size() < 100)
                {
                    return;
                }

                const std::size_t beforeCount = cloud.size();
                std::vector<DensePoint> filtered = filterOp(cloud);
                if (filtered.empty())
                {
                    fprintf(stderr, "[MVS] %s 结果为空，跳过该阶段以保护点云\n", stageName);
                    return;
                }

                const float stageRemovedRatio = static_cast<float>(beforeCount - filtered.size())
                                                / static_cast<float>(beforeCount);
                const float overallRetentionRatio = static_cast<float>(filtered.size())
                                                  / static_cast<float>(initialCount);

                if (stageRemovedRatio > maxStageRemovalRatio || overallRetentionRatio < minOverallRetentionRatio)
                {
                    fprintf(stderr,
                            "[MVS] %s 触发过滤护栏: stageRemoved=%.1f%%(上限%.1f%%), overallRetention=%.1f%%(下限%.1f%%)，保留上一阶段结果\n",
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
                _config.patchMatch.useCuda ? plapoint::ProcessingDevice::Auto : plapoint::ProcessingDevice::CPU;

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

            fprintf(stderr, "[MVS] 过滤后剩余: %d 个稠密点\n", (int)cloud.size());
        }
        else
        {
            fprintf(stderr,
                    "[MVS] 跳过内联稠密点云过滤: points=%zu limit=%zu，保留完整融合点云；"
                    "如需清理请运行稠密点云后处理/精炼\n",
                    initialCount,
                    kMaxInlineDenseFilterPoints);
        }
    }

    emit pointCloudReady(cloud);
    emit progressChanged("完成", 1.f);
    // 只要最终生成了有效点云就算成功（部分帧深度估计失败不影响最终结果）
    emit finished(!cloud.empty());
}

} // namespace mvs
} // namespace xjw

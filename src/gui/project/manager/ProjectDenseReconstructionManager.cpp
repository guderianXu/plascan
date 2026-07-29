#include "ProjectDenseReconstructionManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "ProjectDenseWorkflowConfig.h"
#include "DepthFrameUtils.h"
#include "ProjectMetadataOperations.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ProjectResultRecords.h"
#include "project/ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "ProjectWorkflowUtils.h"
#include "project/SparseResultQuality.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "DenseCloudQualityFilter.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "MvsSourcePlanner.h"
#include "preparation/MatchResultCatalog.h"
#include "GeometryVerifyStage.h"
#include "StoredPairGeometryAudit.h"
#include "project/ProjectIO.h"
#include "SparseCloudPreprocessor.h"
#include "io/PathIO.h"
#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/features/normal_estimation.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMap>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QSet>
#include <QImageReader>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <iterator>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if defined(Q_OS_WIN)
#include <windows.h>
#elif defined(Q_OS_LINUX)
#include <sys/sysinfo.h>
#endif

using xjw::gui::project::buildDepthGenConfig;
using xjw::gui::project::denseGenerationSettingsFromJson;
using xjw::gui::project::denseRefineSettingsFromJson;
using xjw::gui::project::findLatestProductionAtResultIndex;
using xjw::gui::project::makeDenseResultRecord;
using xjw::gui::project::makeDepthResultRecord;
using xjw::gui::project::sparseResultBlockingReason;
using xjw::common::project::normalizePath;
using xjw::core::project::buildStoredFusionFrame;
using xjw::core::project::collectLatestStoredDepthFrames;
using xjw::core::project::collectStoredDepthFramesForDirectory;
using xjw::core::project::depthFrameArtifactsExist;
using xjw::core::project::FusionFrameBuildResult;
using xjw::core::project::estimateFusionFrameWorkingSetBytes;
using xjw::core::project::recommendedDepthFrameLoadWorkers;
using xjw::common::project::normalizePath;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::projectDepthInputSignature;
using xjw::core::project::rawConfidenceStoragePath;
using xjw::core::project::rawDepthStoragePath;
using xjw::gui::project::resolveLatestDenseCloudPath;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::core::project::StoredDepthFrameRecord;
using xjw::gui::project::upsertMetaArrayRecordByPath;
using xjw::gui::project::upsertProjectRecordByPath;

namespace
{

QString utcNowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}

std::vector<xjw::mvs::MvsSourcePairQuality> loadMvsSourcePairQualities(
    const QString &matchDir,
    const std::vector<std::string> &currentImagePaths)
{
    std::vector<xjw::mvs::MvsSourcePairQuality> qualities;
    if (matchDir.trimmed().isEmpty())
    {
        return qualities;
    }

    const QString matchingReportPath =
        QDir(QDir(matchDir).filePath(QStringLiteral("..")))
            .filePath(QStringLiteral("reports/matching_quality_report.json"));
    QFile reportFile(matchingReportPath);
    if (reportFile.open(QIODevice::ReadOnly))
    {
        const QJsonDocument document = QJsonDocument::fromJson(reportFile.readAll());
        const QJsonObject root = document.object();
        const QJsonArray samples = root.value(QStringLiteral("pair_samples")).toArray();
        qualities.reserve(static_cast<size_t>(samples.size()));
        for (const QJsonValue &value : samples)
        {
            const QJsonObject sample = value.toObject();
            if (sample.value(QStringLiteral("status")).toString() != QStringLiteral("matched"))
            {
                continue;
            }

            xjw::mvs::MvsSourcePairQuality quality;
            quality.imageA = xjw::common::io::toUtf8Path(sample.value(QStringLiteral("image_a")).toString());
            quality.imageB = xjw::common::io::toUtf8Path(sample.value(QStringLiteral("image_b")).toString());
            quality.totalMatches = std::max(0, sample.value(QStringLiteral("match_count")).toInt());
            quality.hasVerificationStatistics =
                sample.contains(QStringLiteral("geometric_inlier_count"));
            quality.geometricInliers = std::max(
                0, sample.value(QStringLiteral("geometric_inlier_count")).toInt());
            quality.verified =
                quality.hasVerificationStatistics && quality.geometricInliers > 0;
            quality.geometricCoverage = static_cast<float>(
                sample.value(QStringLiteral("geometric_coverage_ratio")).toDouble(
                    sample.value(QStringLiteral("coverage_ratio")).toDouble(0.0)));
            quality.verificationReason = quality.verified
                ? "verified"
                : (quality.hasVerificationStatistics
                       ? "zero_geometric_inliers"
                       : "missing_geometric_inlier_statistics");
            qualities.push_back(std::move(quality));
        }
    }

    xjw::aerial_triangulation::MatchResultCatalogConfig config;
    config.matchDirectory = matchDir;
    for (const std::string &imagePath : currentImagePaths)
    {
        config.targetImagePaths.append(xjw::common::io::fromUtf8Path(imagePath));
    }
    const xjw::aerial_triangulation::MatchResultCatalogSummary summary =
        xjw::aerial_triangulation::MatchResultCatalog(config).scan();
    qualities.reserve(static_cast<size_t>(summary.pairGroups.size()));
    for (const xjw::aerial_triangulation::MatchPairGroup &group : summary.pairGroups)
    {
        if (group.bestVariantIndex < 0 || group.bestVariantIndex >= group.variants.size())
        {
            continue;
        }

        const xjw::aerial_triangulation::MatchVariant &variant = group.variants.at(group.bestVariantIndex);
        if (!variant.compatible)
        {
            continue;
        }

        xjw::mvs::MvsSourcePairQuality quality;
        quality.imageA = xjw::common::io::toUtf8Path(variant.imageA);
        quality.imageB = xjw::common::io::toUtf8Path(variant.imageB);
        quality.totalMatches = std::max(0, variant.totalMatches);
        quality.geometricInliers = std::max(0, variant.geometricVerifiedInliers);
        quality.hasVerificationStatistics =
            variant.hasInlierStats && quality.totalMatches >= 20;
        quality.verified =
            quality.hasVerificationStatistics &&
            xjw::matchphotos::passesGeometryQualityGate(
                quality.totalMatches, quality.geometricInliers, 20);
        quality.verificationReason = quality.verified
            ? "verified"
            : (quality.hasVerificationStatistics
                   ? "sidecar_geometry_gate_failed"
                   : "missing_geometric_inlier_statistics");
        if (!quality.hasVerificationStatistics)
        {
            const xjw::matchphotos::StoredPairGeometryAuditResult audit =
                xjw::matchphotos::auditStoredPairGeometry(
                    variant.matchFilePath, variant.sidecarPath);
            if (audit.statisticsAvailable)
            {
                quality.totalMatches = audit.totalMatches;
                quality.geometricInliers = audit.geometricInliers;
                quality.hasVerificationStatistics = true;
                quality.verified = audit.verified;
                quality.geometricCoverage = static_cast<float>(
                    audit.coverageScore);
            }
            quality.verificationReason =
                xjw::common::io::toUtf8Path(audit.reason);
        }
        qualities.push_back(std::move(quality));
    }
    return qualities;
}

void attachMvsSourcePairQualities(xjw::mvs::DepthGenConfig *config,
                                  const QString &plascanPath,
                                  const std::vector<xjw::mvs::CameraView> &views)
{
    if (!config || plascanPath.trimmed().isEmpty())
    {
        return;
    }

    std::vector<std::string> currentImagePaths;
    currentImagePaths.reserve(views.size());
    for (const xjw::mvs::CameraView &view : views)
    {
        currentImagePaths.push_back(view.imagePath);
    }
    const std::vector<xjw::mvs::MvsSourcePairQuality> loadedQualities =
        loadMvsSourcePairQualities(
            xjw::common::project::ProjectIO::ipmatchOutputDir(plascanPath),
            currentImagePaths);
    config->sourcePairQualities = xjw::mvs::filterMvsSourcePairQualitiesForImages(
        loadedQualities, currentImagePaths);
    const int verified_pair_count = static_cast<int>(std::count_if(
        config->sourcePairQualities.cbegin(),
        config->sourcePairQualities.cend(),
        [](const xjw::mvs::MvsSourcePairQuality &quality)
        {
            return quality.verified && quality.geometricInliers > 0;
        }));
    const int missing_statistics_count = static_cast<int>(std::count_if(
        config->sourcePairQualities.cbegin(),
        config->sourcePairQualities.cend(),
        [](const xjw::mvs::MvsSourcePairQuality &quality)
        {
            return !quality.hasVerificationStatistics;
        }));
    const int failed_verification_count =
        static_cast<int>(config->sourcePairQualities.size()) -
        verified_pair_count - missing_statistics_count;
    config->requireVerifiedSourcePairs = verified_pair_count > 0;
    if (!config->sourcePairQualities.empty())
    {
        LOG_INFO(QStringLiteral(
                     "[MVS] 当前影像匹配对审计：已验证=%1，缺少统计=%2，"
                     "验证失败=%3；忽略旧引用或重复记录=%4")
                     .arg(verified_pair_count)
                     .arg(missing_statistics_count)
                     .arg(failed_verification_count)
                     .arg(loadedQualities.size() - config->sourcePairQualities.size()));
    }
    else if (!loadedQualities.empty())
    {
        LOG_WARN(QStringLiteral(
                     "[MVS] 匹配目录中的 %1 个几何验证对均不属于当前影像集合，"
                     "已忽略旧引用并回退到当前空三稀疏轨迹选择源视图")
                     .arg(loadedQualities.size()));
    }
}

bool cameraForImagePath(const QMap<QString, xjw::Camera> &camMap,
                        const QString &imagePath,
                        xjw::Camera *camera)
{
    if (!camera)
    {
        return false;
    }

    const QString normalizedPath = normalizePath(imagePath);
    const auto it = camMap.constFind(normalizedPath);
    if (it == camMap.constEnd())
    {
        return false;
    }

    *camera = it.value();
    return true;
}

void applyImageSizeToMvsView(const QString &imagePath, xjw::mvs::CameraView *view)
{
    if (!view)
    {
        return;
    }

    QImageReader reader(imagePath);
    const QSize size = reader.size();
    if (size.isValid() && size.width() > 0 && size.height() > 0)
    {
        view->imageWidth = size.width();
        view->imageHeight = size.height();
    }
}

int streamFusionWindowSize(const xjw::gui::project::DenseGenerationSettings &request,
                           int frameCount)
{
    if (frameCount <= 1)
    {
        return 0;
    }
    const int desiredNeighbors = std::clamp(std::max(8, request.minViews * 3), 8, 16);
    return std::min(frameCount - 1, desiredNeighbors);
}

int fusionWindowCacheCapacity(int neighborCount)
{
    return std::clamp(neighborCount * 2 + 2, 4, 34);
}

std::uint64_t availablePhysicalMemoryBytes()
{
#if defined(Q_OS_WIN)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<std::uint64_t>(status.ullAvailPhys) : 0ULL;
#elif defined(Q_OS_LINUX)
    struct sysinfo status{};
    return sysinfo(&status) == 0
        ? static_cast<std::uint64_t>(status.freeram) * status.mem_unit
        : 0ULL;
#else
    return 0ULL;
#endif
}

double cameraCenterDistance(const xjw::Camera &lhs, const xjw::Camera &rhs)
{
    const auto lhsCenter = lhs.cameraCenter();
    const auto rhsCenter = rhs.cameraCenter();
    const double dx = rhsCenter[0] - lhsCenter[0];
    const double dy = rhsCenter[1] - lhsCenter[1];
    const double dz = rhsCenter[2] - lhsCenter[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::vector<int> nearestFusionWindowIndices(const std::vector<StoredDepthFrameRecord> &frames,
                                            const QMap<QString, xjw::Camera> &camMap,
                                            int referenceIndex,
                                            int neighborCount)
{
    std::vector<int> indices;
    if (referenceIndex < 0 || referenceIndex >= static_cast<int>(frames.size()))
    {
        return indices;
    }

    indices.push_back(referenceIndex);
    if (neighborCount <= 0)
    {
        return indices;
    }

    xjw::Camera referenceCamera;
    const bool hasReferenceCamera =
        cameraForImagePath(camMap, frames[referenceIndex].refImage, &referenceCamera);

    struct Candidate
    {
        int index = -1;
        double distance = 0.0;
        int sequenceDistance = 0;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(frames.size() > 0 ? frames.size() - 1 : 0);
    for (int index = 0; index < static_cast<int>(frames.size()); ++index)
    {
        if (index == referenceIndex)
        {
            continue;
        }

        xjw::Camera camera;
        double distance = static_cast<double>(std::abs(index - referenceIndex));
        if (hasReferenceCamera && cameraForImagePath(camMap, frames[index].refImage, &camera))
        {
            distance = cameraCenterDistance(referenceCamera, camera);
        }
        candidates.push_back({index, distance, std::abs(index - referenceIndex)});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate &lhs, const Candidate &rhs) {
        if (lhs.distance != rhs.distance)
        {
            return lhs.distance < rhs.distance;
        }
        return lhs.sequenceDistance < rhs.sequenceDistance;
    });

    const int count = std::min(neighborCount, static_cast<int>(candidates.size()));
    for (int i = 0; i < count; ++i)
    {
        indices.push_back(candidates[i].index);
    }
    return indices;
}

class DepthFrameLruCache
{
public:
    struct LoadStats
    {
        int loadedCount = 0;
        double readMs = 0.0;
        double postprocessMs = 0.0;
        double resizeMs = 0.0;
        double totalMs = 0.0;
    };

    DepthFrameLruCache(const std::vector<StoredDepthFrameRecord> &records,
                       const QMap<QString, xjw::Camera> &camMap,
                       const xjw::mvs::FusionConfig &fusionConfig,
                       int viewCount,
                       int fusionMaxImageDim,
                       int capacity,
                       int workerCount,
                       std::function<void(const LoadStats &)> loadProgress)
        : _records(records)
        , _cameraMap(camMap)
        , _fusionConfig(fusionConfig)
        , _viewCount(viewCount)
        , _fusionMaxImageDim(std::max(0, fusionMaxImageDim))
        , _capacity(std::max(1, capacity))
        , _loadProgress(std::move(loadProgress))
    {
        const int workers = std::clamp(workerCount, 1, 4);
        _workers.reserve(static_cast<std::size_t>(workers));
        for (int index = 0; index < workers; ++index)
        {
            _workers.emplace_back([this]() { workerLoop(); });
        }
    }

    ~DepthFrameLruCache()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopping = true;
            _jobs.clear();
            _inFlight.clear();
        }
        _jobReady.notify_all();
        for (std::thread &worker : _workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    FusionFrameBuildResult get(int index)
    {
        if (index < 0 || index >= static_cast<int>(_records.size()))
        {
            FusionFrameBuildResult result;
            result.status = {false, QStringLiteral("深度图索引越界：%1").arg(index)};
            return result;
        }

        {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _cache.find(index);
            if (it != _cache.end())
            {
                touch(it);
                return it->second.result;
            }
        }

        return schedule(index).get();
    }

    void prefetch(const std::vector<int> &indices)
    {
        for (const int index : indices)
        {
            if (index >= 0 && index < static_cast<int>(_records.size()))
            {
                (void)schedule(index);
            }
        }
    }

    LoadStats stats() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _stats;
    }

private:
    struct CacheEntry
    {
        FusionFrameBuildResult result;
        std::list<int>::iterator lruIt;
    };

    struct LoadJob
    {
        int index = -1;
        std::shared_ptr<std::promise<FusionFrameBuildResult>> promise;
    };

    std::shared_future<FusionFrameBuildResult> schedule(int index)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        auto cached = _cache.find(index);
        if (cached != _cache.end())
        {
            touch(cached);
            std::promise<FusionFrameBuildResult> ready;
            ready.set_value(cached->second.result);
            return ready.get_future().share();
        }

        const auto in_flight = _inFlight.find(index);
        if (in_flight != _inFlight.end())
        {
            return in_flight->second;
        }

        auto promise = std::make_shared<std::promise<FusionFrameBuildResult>>();
        std::shared_future<FusionFrameBuildResult> future = promise->get_future().share();
        _inFlight.emplace(index, future);
        _jobs.push_back({index, std::move(promise)});
        _jobReady.notify_one();
        return future;
    }

    FusionFrameBuildResult load(int index) const
    {
        xjw::Camera camera;
        if (!cameraForImagePath(_cameraMap, _records[index].refImage, &camera))
        {
            FusionFrameBuildResult result;
            result.status = {false, QStringLiteral("深度图对应影像缺少相机参数：%1")
                                      .arg(QDir::toNativeSeparators(_records[index].refImage))};
            return result;
        }

        FusionFrameBuildResult loaded = buildStoredFusionFrame(_records[index],
                                                               camera,
                                                               _fusionConfig,
                                                               _viewCount,
                                                               _fusionMaxImageDim);
        if (loaded.status.ok)
        {
            loaded.frame.viewIndex = index;
            loaded.frame.sourceImageIndices = storedFusionSourceIndices(_records, index);
        }
        return loaded;
    }

    void workerLoop()
    {
        while (true)
        {
            LoadJob job;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _jobReady.wait(lock, [this]() { return _stopping || !_jobs.empty(); });
                if (_stopping && _jobs.empty())
                {
                    return;
                }
                job = std::move(_jobs.front());
                _jobs.pop_front();
            }

            FusionFrameBuildResult loaded;
            try
            {
                loaded = load(job.index);
            }
            catch (const std::bad_alloc &)
            {
                loaded.status = {
                    false,
                    QStringLiteral("加载深度帧时内存不足，已停止当前融合任务；请降低融合分辨率或并发数")};
            }
            catch (const std::exception &exception)
            {
                loaded.status = {
                    false,
                    QStringLiteral("加载深度帧异常：%1").arg(QString::fromUtf8(exception.what()))};
            }
            LoadStats progress_stats;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (loaded.status.ok)
                {
                    _lru.push_front(job.index);
                    CacheEntry entry;
                    entry.result = loaded;
                    entry.lruIt = _lru.begin();
                    _cache.insert_or_assign(job.index, std::move(entry));
                    evictIfNeeded();
                    if (_loadedEver.insert(job.index).second)
                    {
                        ++_stats.loadedCount;
                    }
                    _stats.readMs += loaded.readMs;
                    _stats.postprocessMs += loaded.postprocessMs;
                    _stats.resizeMs += loaded.resizeMs;
                    _stats.totalMs += loaded.totalMs;
                }
                _inFlight.erase(job.index);
                progress_stats = _stats;
            }
            job.promise->set_value(std::move(loaded));
            if (_loadProgress)
            {
                _loadProgress(progress_stats);
            }
        }
    }

    void touch(std::unordered_map<int, CacheEntry>::iterator it)
    {
        _lru.erase(it->second.lruIt);
        _lru.push_front(it->first);
        it->second.lruIt = _lru.begin();
    }

    void evictIfNeeded()
    {
        while (static_cast<int>(_cache.size()) > _capacity)
        {
            const int victim = _lru.back();
            _lru.pop_back();
            _cache.erase(victim);
        }
    }

    const std::vector<StoredDepthFrameRecord> &_records;
    const QMap<QString, xjw::Camera> &_cameraMap;
    xjw::mvs::FusionConfig _fusionConfig;
    int _viewCount = 0;
    int _fusionMaxImageDim = 0;
    int _capacity = 1;
    std::function<void(const LoadStats &)> _loadProgress;
    mutable std::mutex _mutex;
    std::condition_variable _jobReady;
    bool _stopping = false;
    std::deque<LoadJob> _jobs;
    std::vector<std::thread> _workers;
    std::list<int> _lru;
    std::unordered_map<int, CacheEntry> _cache;
    std::unordered_map<int, std::shared_future<FusionFrameBuildResult>> _inFlight;
    std::unordered_set<int> _loadedEver;
    LoadStats _stats;
};

enum class ExistingDepthAction
{
    Cancel,
    Overwrite,
    ContinueMissing
};

QSet<int> collectExistingDepthFrameIndices(const QString &outputDir, int expectedCount)
{
    QSet<int> indices;
    for (int index = 0; index < expectedCount; ++index)
    {
        const QString pngPath = QDir(outputDir).filePath(QStringLiteral("depth_%1.png").arg(index));
        if (depthFrameArtifactsExist(pngPath))
        {
            indices.insert(index);
        }
    }
    return indices;
}

QString validMaskStoragePath(const QString &pngPath)
{
    const QFileInfo info(pngPath);
    return QDir(info.absolutePath()).filePath(QStringLiteral("%1_mask.png").arg(info.completeBaseName()));
}

ExistingDepthAction askExistingDepthAction(QWidget *parent,
                                           int existingCount,
                                           int totalCount,
                                           const QString &outputDir)
{
    QMessageBox box(parent);
    box.setIcon(QMessageBox::Question);
    box.setWindowTitle(QStringLiteral("深度图估计"));
    box.setText(QStringLiteral("检测到已有深度图结果（%1/%2）。").arg(existingCount).arg(totalCount));
    box.setInformativeText(QStringLiteral("输出目录：%1\n请选择覆盖重算，或继续生成未完成的帧。")
                               .arg(QDir::toNativeSeparators(outputDir)));

    QPushButton *overwriteButton = box.addButton(QStringLiteral("覆盖重算"), QMessageBox::DestructiveRole);
    QPushButton *continueButton = box.addButton(QStringLiteral("继续生成未完成帧"), QMessageBox::AcceptRole);
    QPushButton *cancelButton = box.addButton(QStringLiteral("取消"), QMessageBox::RejectRole);
    box.setDefaultButton(continueButton);

    box.exec();
    if (box.clickedButton() == overwriteButton)
    {
        return ExistingDepthAction::Overwrite;
    }
    if (box.clickedButton() == continueButton)
    {
        return ExistingDepthAction::ContinueMissing;
    }
    if (box.clickedButton() == cancelButton)
    {
        return ExistingDepthAction::Cancel;
    }
    return ExistingDepthAction::Cancel;
}

void removeDepthArtifactsForIndices(const QString &outputDir, const QSet<int> &indices)
{
    for (const int index : indices)
    {
        const QString pngPath = QDir(outputDir).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString rawPath = rawDepthStoragePath(pngPath);
        const QString confPath = rawConfidenceStoragePath(pngPath);
        const QString maskPath = validMaskStoragePath(pngPath);
        QFile::remove(pngPath);
        QFile::remove(rawPath);
        QFile::remove(confPath);
        QFile::remove(maskPath);
    }
}

QJsonObject makeProjectDepthRecordFromArtifact(const QJsonObject &artifact,
                                               const QString &sparseXyz,
                                               const QString &mvsOutputDir,
                                               int batch_frame_count,
                                               const QString &project_input_signature,
                                               const QString &reconstruction_generation_id)
{
    const QString pngPath = artifact.value(QStringLiteral("depth_png")).toString();
    if (pngPath.trimmed().isEmpty())
    {
        return {};
    }

    QJsonObject depthResult = makeDepthResultRecord(utcNowIso(),
                                                    pngPath,
                                                    artifact.value(QStringLiteral("grid_width")).toInt(0),
                                                    artifact.value(QStringLiteral("grid_height")).toInt(0),
                                                    sparseXyz,
                                                    artifact.value(QStringLiteral("ref_image")).toString());
    for (auto it = artifact.constBegin(); it != artifact.constEnd(); ++it)
    {
        depthResult[it.key()] = it.value();
    }

    if (depthResult.value(QStringLiteral("raw_depth_path")).toString().isEmpty())
    {
        depthResult[QStringLiteral("raw_depth_path")] = rawDepthStoragePath(pngPath);
    }
    if (depthResult.value(QStringLiteral("raw_confidence_path")).toString().isEmpty())
    {
        depthResult[QStringLiteral("raw_confidence_path")] = rawConfidenceStoragePath(pngPath);
    }
    if (depthResult.value(QStringLiteral("valid_mask_path")).toString().isEmpty())
    {
        depthResult[QStringLiteral("valid_mask_path")] = validMaskStoragePath(pngPath);
    }
    if (depthResult.value(QStringLiteral("mvs_output_dir")).toString().isEmpty())
    {
        depthResult[QStringLiteral("mvs_output_dir")] = mvsOutputDir;
    }
    if (depthResult.value(QStringLiteral("status")).toString().isEmpty())
    {
        depthResult[QStringLiteral("status")] = QStringLiteral("completed");
    }
    depthResult[QStringLiteral("batch_frame_count")] = batch_frame_count;
    depthResult[QStringLiteral("project_input_signature")] = project_input_signature;
    depthResult[QStringLiteral("reconstruction_generation_id")] =
        reconstruction_generation_id;

    return depthResult;
}

using PlaPC = plapoint::PointCloud<float, plamatrix::Device::CPU>;

constexpr std::size_t kMaxDenseRefineFilterInputPoints = 250000;
constexpr int kMaxDenseRefinePreconditionPasses = 6;
constexpr std::uint64_t kStreamingDenseRefineMinPoints = 1000000;
constexpr int kStreamingDenseRefineChunkMb = 512;

struct StreamingDenseRefineResult
{
    bool cancelled = false;
    QString error;
    QString reportPath;
    QJsonObject reportJson;
    QJsonObject terrainReportJson;
    xjw::mvs::TerrainHeightSpikeFilterReport terrainReport;
    int pointCount = 0;
};

PlaPC densePointsToPointCloud(const std::vector<xjw::mvs::DensePoint> &cloud)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(cloud.size(), 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        pts(i, 0) = cloud[i].x;
        pts(i, 1) = cloud[i].y;
        pts(i, 2) = cloud[i].z;
        colors(i, 0) = cloud[i].r;
        colors(i, 1) = cloud[i].g;
        colors(i, 2) = cloud[i].b;
    }
    PlaPC pc(std::move(pts));
    pc.setColors(std::move(colors));
    return pc;
}

PlaPC fusedPointsToPointCloud(const std::vector<xjw::mvs::FusedPoint> &cloud,
                              bool keepColor,
                              bool keepNormals)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        pts(i, 0) = cloud[i].x;
        pts(i, 1) = cloud[i].y;
        pts(i, 2) = cloud[i].z;
    }
    PlaPC pc(std::move(pts));

    if (keepColor)
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            colors(i, 0) = cloud[i].r;
            colors(i, 1) = cloud[i].g;
            colors(i, 2) = cloud[i].b;
        }
        pc.setColors(std::move(colors));
    }

    if (keepNormals)
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> nrm(cloud.size(), 3);
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            nrm(i, 0) = cloud[i].nx;
            nrm(i, 1) = cloud[i].ny;
            nrm(i, 2) = cloud[i].nz;
        }
        pc.setNormals(std::move(nrm));
    }

    return pc;
}

bool readPointCloudPly(const QString &path, PlaPC *cloud, QString *errorMessage)
{
    try
    {
        auto loaded = plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(path));
        if (!loaded)
        {
            if (errorMessage) *errorMessage = QStringLiteral("读取PLY文件失败: %1").arg(path);
            return false;
        }
        *cloud = std::move(*loaded);
        return true;
    }
    catch (const std::exception &e)
    {
        if (errorMessage) *errorMessage = QString::fromStdString(e.what());
        return false;
    }
}

PlaPC cloneCloudValue(const PlaPC &cloud, bool includeNormals = true)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i)
        for (int d = 0; d < 3; ++d)
            pts(static_cast<plamatrix::Index>(i), d) = cloud.points()(static_cast<plamatrix::Index>(i), d);

    PlaPC copy(std::move(pts));
    if (cloud.hasColors()) copy.setColors(*cloud.colors());
    if (cloud.hasIntensities()) copy.setIntensities(*cloud.intensities());
    if (includeNormals && cloud.hasNormals()) copy.setNormals(*cloud.normals());
    if (cloud.hasScalarFields()) copy.setScalarFields(cloud.scalarFieldNames(), *cloud.scalarFields());
    if (cloud.hasFaces()) copy.setFaces(*cloud.faces());
    copy.setMaterialLibraryFile(cloud.materialLibraryFile());
    copy.setTextureImageFile(cloud.textureImageFile());
    return copy;
}

bool writePointCloudPly(const QString &path,
                        const PlaPC &pointCloud,
                        bool writeNormals,
                        QString *errorMessage)
{
    try
    {
        if (writeNormals || !pointCloud.hasNormals())
        {
            plapoint::io::writePly(
                xjw::common::io::toNativeNarrowPath(path), pointCloud, plapoint::io::PlyFormat::BinaryLE);
        }
        else
        {
            const auto withoutNormals = cloneCloudValue(pointCloud, false);
            plapoint::io::writePly(
                xjw::common::io::toNativeNarrowPath(path), withoutNormals, plapoint::io::PlyFormat::BinaryLE);
        }
        return true;
    }
    catch (const std::exception &e)
    {
        if (errorMessage) *errorMessage = QString::fromStdString(e.what());
        return false;
    }
}

xjw::mvs::TerrainHeightSpikeFilterOptions terrainSpikeOptionsFromRequest(
    const xjw::gui::project::DenseRefineSettings &request)
{
    xjw::mvs::TerrainHeightSpikeFilterOptions options;
    options.enabled = request.terrainSpikeFilterEnabled;
    options.gridResolution = request.terrainSpikeGridResolution;
    options.minCellPoints = request.terrainSpikeMinCellPoints;
    options.minHeightThreshold = static_cast<float>(request.terrainSpikeMinHeightThreshold);
    options.madMultiplier = static_cast<float>(request.terrainSpikeMadMultiplier);
    options.localPlaneFilterEnabled = request.terrainLocalPlaneFilterEnabled;
    options.localPlaneMinPoints = request.terrainLocalPlaneMinPoints;
    options.localPlaneMinResidualThreshold =
        static_cast<float>(request.terrainLocalPlaneMinResidualThreshold);
    options.localPlaneMadMultiplier = static_cast<float>(request.terrainLocalPlaneMadMultiplier);
    return options;
}

QJsonObject terrainSpikeReportToJson(const xjw::mvs::TerrainHeightSpikeFilterReport &report)
{
    return QJsonObject{
        {QStringLiteral("input_points"), static_cast<double>(report.inputPoints)},
        {QStringLiteral("output_points"), static_cast<double>(report.outputPoints)},
        {QStringLiteral("removed_points"), static_cast<double>(report.removedPoints)},
        {QStringLiteral("local_plane_removed_points"), static_cast<double>(report.localPlaneRemovedPoints)},
        {QStringLiteral("median_cell_z_range_before"), report.medianCellZRangeBefore},
        {QStringLiteral("p95_cell_z_range_before"), report.p95CellZRangeBefore},
        {QStringLiteral("median_cell_z_range_after"), report.medianCellZRangeAfter},
        {QStringLiteral("p95_cell_z_range_after"), report.p95CellZRangeAfter}
    };
}

xjw::mvs::TerrainHeightSpikeFilterReport terrainSpikeReportFromJson(const QJsonObject &object)
{
    xjw::mvs::TerrainHeightSpikeFilterReport report;
    report.inputPoints = static_cast<std::size_t>(object.value(QStringLiteral("input_points")).toDouble(0.0));
    report.outputPoints = static_cast<std::size_t>(object.value(QStringLiteral("output_points")).toDouble(0.0));
    report.removedPoints = static_cast<std::size_t>(object.value(QStringLiteral("removed_points")).toDouble(0.0));
    report.localPlaneRemovedPoints =
        static_cast<std::size_t>(object.value(QStringLiteral("local_plane_removed_points")).toDouble(0.0));
    report.medianCellZRangeBefore = object.value(QStringLiteral("median_cell_z_range_before")).toDouble(0.0);
    report.p95CellZRangeBefore = object.value(QStringLiteral("p95_cell_z_range_before")).toDouble(0.0);
    report.medianCellZRangeAfter = object.value(QStringLiteral("median_cell_z_range_after")).toDouble(0.0);
    report.p95CellZRangeAfter = object.value(QStringLiteral("p95_cell_z_range_after")).toDouble(0.0);
    return report;
}

QString denseCloudRefineCliExecutablePath()
{
    const QString exeName =
#ifdef Q_OS_WIN
        QStringLiteral("dense_cloud_refine_cli.exe");
#else
        QStringLiteral("dense_cloud_refine_cli");
#endif

    const QStringList candidates{
        QDir(QCoreApplication::applicationDirPath()).filePath(exeName),
        QDir(QDir::currentPath()).filePath(exeName),
        QDir(QDir::currentPath()).filePath(QStringLiteral("bin/%1").arg(exeName))
    };

    for (const QString &candidate : candidates)
    {
        if (QFileInfo::exists(candidate))
        {
            return candidate;
        }
    }
    return {};
}

bool readBinaryPlyVertexCount(const QString &plyPath, std::uint64_t *vertexCount)
{
    if (!vertexCount)
    {
        return false;
    }

    plapoint::io::PlyVertexStreamHeader header;
    std::string error;
    if (!plapoint::io::parseBinaryPlyVertexStreamHeader(
            xjw::common::io::toNativeNarrowPath(plyPath), &header, &error))
    {
        return false;
    }

    *vertexCount = header.vertexCount;
    return true;
}

bool shouldUseStreamingDenseRefine(const QString &inputPly,
                                   const xjw::gui::project::DenseRefineSettings &request,
                                   std::uint64_t *vertexCount)
{
    std::uint64_t count = 0;
    if (!request.terrainSpikeFilterEnabled || !readBinaryPlyVertexCount(inputPly, &count))
    {
        return false;
    }
    if (vertexCount)
    {
        *vertexCount = count;
    }
    return count >= kStreamingDenseRefineMinPoints;
}

QStringList streamingDenseRefineArguments(const QString &inputPly,
                                          const QString &outputPly,
                                          const QString &reportPath,
                                          const xjw::gui::project::DenseRefineSettings &request)
{
    QStringList args{
        QStringLiteral("--input"),
        inputPly,
        QStringLiteral("--output"),
        outputPly,
        QStringLiteral("--report-json"),
        reportPath,
        QStringLiteral("--streaming-chunk-mb"),
        QString::number(kStreamingDenseRefineChunkMb),
        QStringLiteral("--terrain-grid-cells"),
        QString::number(request.terrainSpikeGridResolution),
        QStringLiteral("--terrain-min-cell-points"),
        QString::number(request.terrainSpikeMinCellPoints),
        QStringLiteral("--terrain-min-height-threshold"),
        QString::number(request.terrainSpikeMinHeightThreshold, 'g', 8),
        QStringLiteral("--terrain-mad-multiplier"),
        QString::number(request.terrainSpikeMadMultiplier, 'g', 8),
        QStringLiteral("--terrain-local-plane-min-points"),
        QString::number(request.terrainLocalPlaneMinPoints),
        QStringLiteral("--terrain-local-plane-min-residual-threshold"),
        QString::number(request.terrainLocalPlaneMinResidualThreshold, 'g', 8),
        QStringLiteral("--terrain-local-plane-mad-multiplier"),
        QString::number(request.terrainLocalPlaneMadMultiplier, 'g', 8),
        QStringLiteral("--terrain-filter-passes"),
        QString::number(request.terrainFilterPasses)
    };

    args << (request.terrainLocalPlaneFilterEnabled
                 ? QStringLiteral("--terrain-local-plane-filter")
                 : QStringLiteral("--disable-terrain-local-plane-filter"));
    return args;
}

bool runStreamingDenseCloudRefineCli(const QString &inputPly,
                                     const QString &outputPly,
                                     const xjw::gui::project::DenseRefineSettings &request,
                                     const std::function<bool()> &isCancelled,
                                     const std::function<void(const QString &, int)> &postProgress,
                                     StreamingDenseRefineResult *result)
{
    if (!result)
    {
        return false;
    }

    const QString exePath = denseCloudRefineCliExecutablePath();
    if (exePath.isEmpty())
    {
        result->error = QStringLiteral("未找到 dense_cloud_refine_cli 可执行文件");
        return false;
    }

    const QString reportPath = outputPly + QStringLiteral(".report.json");
    QFile::remove(reportPath);
    QFile::remove(outputPly);

    if (postProgress)
    {
        postProgress(QStringLiteral("流式密集点云地表清理..."), 8);
    }

    QProcess process;
    process.setProgram(exePath);
    process.setArguments(streamingDenseRefineArguments(inputPly, outputPly, reportPath, request));
    process.start();
    if (!process.waitForStarted(10000))
    {
        result->error = QStringLiteral("启动 dense_cloud_refine_cli 失败: %1").arg(process.errorString());
        return false;
    }

    while (!process.waitForFinished(500))
    {
        if (isCancelled && isCancelled())
        {
            process.kill();
            process.waitForFinished(5000);
            QFile::remove(outputPly);
            QFile::remove(reportPath);
            result->cancelled = true;
            result->error = QStringLiteral("用户取消了流式密集点云后处理");
            return false;
        }
    }

    const QString stdOut = QString::fromUtf8(process.readAllStandardOutput());
    const QString stdErr = QString::fromUtf8(process.readAllStandardError());
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
    {
        result->error = QStringLiteral("dense_cloud_refine_cli 失败(exit=%1): %2 %3")
                            .arg(process.exitCode())
                            .arg(stdErr.trimmed(), stdOut.trimmed());
        return false;
    }

    QFile reportFile(reportPath);
    if (!reportFile.open(QIODevice::ReadOnly))
    {
        result->error = QStringLiteral("无法读取流式清理报告: %1").arg(reportPath);
        return false;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(reportFile.readAll());
    const QJsonObject reportObject = doc.object();
    const QJsonObject terrainObject = reportObject.value(QStringLiteral("terrain_spike_filter")).toObject();
    result->reportPath = reportPath;
    result->reportJson = reportObject;
    result->terrainReportJson = terrainObject;
    result->terrainReport = terrainSpikeReportFromJson(terrainObject);
    result->pointCount = static_cast<int>(
        std::min<double>(reportObject.value(QStringLiteral("output_points")).toDouble(0.0),
                         static_cast<double>(std::numeric_limits<int>::max())));

    LOG_INFO(QStringLiteral("[DenseRefine] 流式地表清理完成: %1")
                 .arg(stdOut.trimmed()));
    return QFileInfo::exists(outputPly);
}

// Helper: deep-copy PointCloud to shared_ptr
static std::shared_ptr<PlaPC> cloneCloud(const PlaPC &cloud)
{
    return std::make_shared<PlaPC>(cloneCloudValue(cloud));
}

QString processingDeviceLabel(plapoint::ProcessingDevice device)
{
    switch (device)
    {
    case plapoint::ProcessingDevice::CPU:
        return QStringLiteral("CPU");
    case plapoint::ProcessingDevice::GPU:
        return QStringLiteral("GPU");
    case plapoint::ProcessingDevice::Auto:
        return QStringLiteral("Auto");
    }
    return QStringLiteral("Unknown");
}

void logPlaPointReport(const QString &stage,
                       const plapoint::ProcessingReport &report,
                       std::size_t beforeCount,
                       std::size_t afterCount)
{
    QString message = QStringLiteral("[DenseRefine] %1: %2 → %3, requested=%4, usedDevice=%5")
        .arg(stage)
        .arg(beforeCount)
        .arg(afterCount)
        .arg(processingDeviceLabel(report.requestedDevice),
             processingDeviceLabel(report.usedDevice));
    if (report.usedFallback)
    {
        message += QStringLiteral(", fallback=%1")
            .arg(QString::fromStdString(report.fallbackReason));
    }
    LOG_INFO(message);
}

struct PointCloudBounds
{
    bool valid = false;
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double minZ = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();
};

PointCloudBounds computePointCloudBounds(const PlaPC &cloud)
{
    PointCloudBounds bounds;
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        const double x = static_cast<double>(cloud.points()(row, 0));
        const double y = static_cast<double>(cloud.points()(row, 1));
        const double z = static_cast<double>(cloud.points()(row, 2));
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            continue;
        }

        bounds.valid = true;
        bounds.minX = std::min(bounds.minX, x);
        bounds.minY = std::min(bounds.minY, y);
        bounds.minZ = std::min(bounds.minZ, z);
        bounds.maxX = std::max(bounds.maxX, x);
        bounds.maxY = std::max(bounds.maxY, y);
        bounds.maxZ = std::max(bounds.maxZ, z);
    }
    return bounds;
}

float estimateDenseRefinePreconditionLeafSize(const PlaPC &cloud, std::size_t targetPoints)
{
    const PointCloudBounds bounds = computePointCloudBounds(cloud);
    if (!bounds.valid || targetPoints == 0)
    {
        return 0.005f;
    }

    const double dx = std::max(0.0, bounds.maxX - bounds.minX);
    const double dy = std::max(0.0, bounds.maxY - bounds.minY);
    const double dz = std::max(0.0, bounds.maxZ - bounds.minZ);
    const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double largestArea = std::max(dx * dy, std::max(dx * dz, dy * dz));

    double leafSize = 0.005;
    if (largestArea > 1e-12)
    {
        leafSize = std::sqrt(largestArea / static_cast<double>(targetPoints)) * 0.85;
    }
    else if (diag > 1e-9)
    {
        leafSize = diag / std::cbrt(static_cast<double>(targetPoints)) * 0.5;
    }

    if (diag > 1e-9)
    {
        const double minLeaf = std::max(diag * 0.00002, 1e-5);
        const double maxLeaf = std::max(minLeaf * 2.0, diag * 0.05);
        leafSize = std::clamp(leafSize, minLeaf, maxLeaf);
    }

    if (!std::isfinite(leafSize) || leafSize <= 0.0)
    {
        return 0.005f;
    }
    return static_cast<float>(leafSize);
}

PlaPC sorFilter(const PlaPC &cloud,
                int k,
                float stdRatio,
                plapoint::ProcessingDevice processingDevice,
                plapoint::ProcessingReport *report = nullptr)
{
    if (cloud.size() < static_cast<size_t>(k + 1))
    {
        if (report)
        {
            report->requestedDevice = processingDevice;
            report->usedDevice = plapoint::ProcessingDevice::CPU;
            report->usedFallback = false;
            report->fallbackReason = "skipped: point count is smaller than k + 1";
        }
        return std::move(*cloneCloud(cloud));
    }

    return plapoint::statisticalOutlierRemoval(cloud, k, stdRatio, processingDevice, nullptr, report);
}

PlaPC radiusFilter(const PlaPC &cloud,
                   float radius,
                   int minNeighbors,
                   plapoint::ProcessingDevice processingDevice,
                   plapoint::ProcessingReport *report = nullptr)
{
    if (cloud.size() == 0)
    {
        if (report)
        {
            report->requestedDevice = processingDevice;
            report->usedDevice = plapoint::ProcessingDevice::CPU;
            report->usedFallback = false;
            report->fallbackReason = "skipped: empty cloud";
        }
        PlaPC emptyOut(0);
        return emptyOut;
    }

    return plapoint::radiusOutlierRemoval(cloud, radius, minNeighbors, processingDevice, nullptr, report);
}

PlaPC voxelDownsample(const PlaPC &cloud,
                      float leafSize,
                      plapoint::ProcessingDevice processingDevice,
                      plapoint::ProcessingReport *report = nullptr)
{
    if (cloud.size() == 0 || leafSize <= 0)
    {
        if (report)
        {
            report->requestedDevice = processingDevice;
            report->usedDevice = plapoint::ProcessingDevice::CPU;
            report->usedFallback = false;
            report->fallbackReason = "skipped: empty cloud or invalid leaf size";
        }
        return std::move(*cloneCloud(cloud));
    }

    return plapoint::voxelDownsample(cloud, leafSize, processingDevice, report);
}

struct DenseRefinePreconditionStats
{
    bool applied = false;
    bool consumedRequestedVoxel = false;
    float leafSize = 0.0f;
    int passes = 0;
};

DenseRefinePreconditionStats preconditionDenseRefineCloudForFilters(
    PlaPC *cloud,
    const xjw::gui::project::DenseRefineSettings &request,
    const std::function<void(const QString &, int)> &progress)
{
    DenseRefinePreconditionStats stats;
    if (!cloud || cloud->size() <= kMaxDenseRefineFilterInputPoints)
    {
        return stats;
    }
    if (!request.sorEnabled && !request.normalsEnabled && !request.terrainSpikeFilterEnabled)
    {
        return stats;
    }

    stats.consumedRequestedVoxel = request.voxelEnabled && request.voxelSize > 0.0;
    float leafSize = stats.consumedRequestedVoxel
        ? static_cast<float>(request.voxelSize)
        : estimateDenseRefinePreconditionLeafSize(*cloud, kMaxDenseRefineFilterInputPoints);
    if (!std::isfinite(static_cast<double>(leafSize)) || leafSize <= 0.0f)
    {
        leafSize = 0.005f;
    }

    if (progress)
    {
        progress(QStringLiteral("点云过大，先进行预降采样..."), 10);
    }
    LOG_INFO(QStringLiteral("[DenseRefine] 点云过大，先进行预降采样: input=%1 targetPoints=%2 initialLeaf=%3")
                 .arg(cloud->size())
                 .arg(kMaxDenseRefineFilterInputPoints)
                 .arg(leafSize, 0, 'g', 6));

    while (cloud->size() > kMaxDenseRefineFilterInputPoints
           && stats.passes < kMaxDenseRefinePreconditionPasses)
    {
        const auto beforeVoxel = cloud->size();
        plapoint::ProcessingReport voxelReport;
        *cloud = voxelDownsample(*cloud, leafSize, request.processingDevice, &voxelReport);
        ++stats.passes;
        stats.applied = true;
        stats.leafSize = leafSize;
        logPlaPointReport(QStringLiteral("大点云预降采样"), voxelReport, beforeVoxel, cloud->size());

        if (cloud->size() <= kMaxDenseRefineFilterInputPoints || cloud->size() == 0)
        {
            break;
        }

        const double ratio = static_cast<double>(cloud->size())
            / static_cast<double>(kMaxDenseRefineFilterInputPoints);
        float nextLeafSize = static_cast<float>(static_cast<double>(leafSize)
            * std::sqrt(std::max(1.0, ratio)) * 1.05);
        if (!(nextLeafSize > leafSize))
        {
            nextLeafSize = leafSize * 2.0f;
        }
        leafSize = nextLeafSize;

        if (progress)
        {
            progress(QStringLiteral("点云过大，继续预降采样..."),
                     std::min(18, 10 + stats.passes * 2));
        }
    }

    if (stats.applied)
    {
        LOG_INFO(QStringLiteral("[DenseRefine] 大点云预降采样完成: output=%1 leaf=%2 passes=%3 targetPoints=%4")
                     .arg(cloud->size())
                     .arg(stats.leafSize, 0, 'g', 6)
                     .arg(stats.passes)
                     .arg(kMaxDenseRefineFilterInputPoints));
    }
    return stats;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> estimateNormals(
    const PlaPC &cloud,
    int normalK,
    plapoint::ProcessingDevice processingDevice,
    plapoint::ProcessingReport *report = nullptr)
{
    return plapoint::estimateNormals(cloud, normalK, processingDevice, report);
}

} // namespace

ProjectDenseReconstructionManager::ProjectDenseReconstructionManager(ProjectManager *owner,
                                                                     ProjectData *projectData,
                                                                     QWidget *parentWidget,
                                                                     QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(projectData)
    , _parentWidget(parentWidget)
{
}

bool ProjectDenseReconstructionManager::ensureProjectOpen(const QString &message,
                                                          const QString &title) const
{
    if (_projectData && _projectData->hasProject())
    {
        return true;
    }
    QMessageBox::warning(_parentWidget, title, message);
    return false;
}

std::shared_ptr<std::atomic_bool> ProjectDenseReconstructionManager::createActiveMvsCancelFlag()
{
    auto cancelFlag = std::make_shared<std::atomic_bool>(false);
    _activeMvsCancelFlag = cancelFlag;
    return cancelFlag;
}

void ProjectDenseReconstructionManager::clearActiveMvsCancelFlag(
    const std::shared_ptr<std::atomic_bool> &cancelFlag)
{
    if (_activeMvsCancelFlag == cancelFlag)
    {
        _activeMvsCancelFlag.reset();
    }
}

bool ProjectDenseReconstructionManager::startEstimateDepthMapsAsync(const QJsonObject &settings)
{
    using namespace xjw::mvs;

    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("深度图估计")))
    {
        return false;
    }
    if (isMvsRunning())
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("深度图估计"),
                                 QStringLiteral("已有深度图或密集点云任务正在运行，请先等待或取消当前任务。"));
        return false;
    }

    const xjw::gui::project::DenseGenerationSettings request = denseGenerationSettingsFromJson(settings);
    const QJsonObject meta = _projectData->metadata();
    const QJsonArray atArr = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (atArr.isEmpty())
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("深度图估计"),
                             QStringLiteral("未找到空三结果，请先执行空中三角测量。"));
        return false;
    }

    const int realIdx = (request.atIndex >= 0 && request.atIndex < atArr.size())
        ? request.atIndex
        : findLatestProductionAtResultIndex(meta);
    if (realIdx < 0)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("深度图估计"),
                             QStringLiteral("未找到可用的正式 SfM/BA 稀疏点云结果。请先运行三维重建/空三。"));
        return false;
    }
    const QJsonObject atResult = atArr[realIdx].toObject();
    const QString project_input_signature = projectDepthInputSignature(meta, realIdx);
    const QString reconstruction_generation_id =
        atResult.value(QStringLiteral("reconstruction_generation_id")).toString();
    if (!xjw::gui::project::isProductionSparseResult(atResult))
    {
        const QString reason = sparseResultBlockingReason(atResult);
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("深度图估计"),
                             reason.isEmpty()
                                 ? QStringLiteral("所选稀疏点云不是正式 SfM/BA 结果。")
                                 : reason);
        return false;
    }
    const QJsonArray selImgArr = atResult.value(QStringLiteral("selected_images")).toArray();
    const QJsonObject files = atResult.value(QStringLiteral("files")).toObject();
    const QString sparseXyz = files.value(QStringLiteral("sparse_cloud_xyz")).toString();

    QStringList selectedImages;
    for (const auto &value : selImgArr)
    {
        selectedImages.append(value.toString());
    }

    if (selectedImages.size() < 2)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("深度图估计"),
                             QStringLiteral("空三结果中影像数量不足（至少需要2张）。"));
        return false;
    }

    const QString mvsOutDir = resolveProjectOutputDir(_owner->currentProjectPath(), request.outputDir, QStringLiteral("mvs_output"));
    const QSet<int> existingIndices = collectExistingDepthFrameIndices(mvsOutDir, selectedImages.size());
    if (!existingIndices.isEmpty())
    {
        const ExistingDepthAction action = askExistingDepthAction(_parentWidget,
                                                                  existingIndices.size(),
                                                                  selectedImages.size(),
                                                                  mvsOutDir);
        if (action == ExistingDepthAction::Cancel)
        {
            return false;
        }

        if (action == ExistingDepthAction::Overwrite)
        {
            removeDepthArtifactsForIndices(mvsOutDir, existingIndices);
            LOG_INFO(QStringLiteral("[MVS] 用户选择覆盖重算，已清理已有深度图 %1 帧").arg(existingIndices.size()));
        }
        else if (action == ExistingDepthAction::ContinueMissing)
        {
            LOG_INFO(QStringLiteral("[MVS] 用户选择续跑：发现 %1 个深度文件，"
                                    "将由 manifest 校验影像、相机与参数后决定可复用帧")
                         .arg(existingIndices.size()));
        }
    }

    bool allCams = false;
    const QMap<QString, xjw::Camera> camMap = _owner->getCamerasForImages(selectedImages, &allCams);
    if (!allCams)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("深度图估计"),
                             QStringLiteral("部分影像缺少相机参数，无法执行深度图估计。"));
        return false;
    }

    std::vector<CameraView> views;
    views.reserve(selectedImages.size());
    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    for (const QString &imgPath : selectedImages)
    {
        CameraView view;
        view.imagePath = xjw::common::io::toUtf8Path(imgPath);
        view.validRegionMaskPath = xjw::common::io::toUtf8Path(
            xjw::common::project::ProjectIO::findMaskForImage(projectPath, imgPath));
        if (!cameraForImagePath(camMap, imgPath, &view.camera))
        {
            QMessageBox::warning(_parentWidget,
                                 QStringLiteral("深度图估计"),
                                 QStringLiteral("影像缺少相机参数：%1").arg(QDir::toNativeSeparators(imgPath)));
            return false;
        }
        applyImageSizeToMvsView(imgPath, &view);
        views.push_back(std::move(view));
    }

    DepthGenConfig genCfg = buildDepthGenConfig(request, static_cast<int>(views.size()));
    genCfg.inputSignature = project_input_signature.toStdString();
    genCfg.runFusion = false;
    genCfg.saveIntermediateDepthMaps = true;
    genCfg.intermediateDir = xjw::common::io::toUtf8Path(mvsOutDir);
    attachMvsSourcePairQualities(&genCfg, _owner->currentProjectPath(), views);

    auto *gen = new DepthMapGenerator(this);
    gen->setViews(views);
    gen->setConfig(genCfg);
    gen->setOutputDir(xjw::common::io::toUtf8Path(mvsOutDir));

    QPointer<ProjectDenseReconstructionManager> self(this);
    QObject::connect(gen, &DepthMapGenerator::progressChanged, this,
                     [self, projectPath](const QString &stage, float ratio) {
        if (!self || !self->_owner || self->_owner->currentProjectPath() != projectPath)
        {
            return;
        }
        emit self->mvsProgressChanged(stage, static_cast<int>(ratio * 100));
    });
    connect(gen, &DepthMapGenerator::errorOccurred, this, [](const QString &msg) {
        qWarning() << "[MVS] 错误:" << msg;
    });
    connect(gen, &DepthMapGenerator::depthMapArtifactSaved, this,
            [self,
             sparseXyz,
             mvsOutDir,
             projectPath,
             project_input_signature,
             reconstruction_generation_id,
             quality_profile = request.qualityProfile,
             batch_frame_count = selectedImages.size()](const QJsonObject &artifact) {
        if (!self || !self->_owner || self->_owner->currentProjectPath() != projectPath)
        {
            return;
        }
        QJsonObject depthResult = makeProjectDepthRecordFromArtifact(
            artifact,
            sparseXyz,
            mvsOutDir,
            batch_frame_count,
            project_input_signature,
            reconstruction_generation_id);
        if (depthResult.isEmpty())
        {
            return;
        }
        depthResult[QStringLiteral("quality_profile")] = quality_profile;
        upsertProjectRecordByPath(self->_projectData,
                                  QStringLiteral("depth_map_results"),
                                  QStringLiteral("depth_png"),
                                  depthResult);
    });
    connect(gen,
            &DepthMapGenerator::finished,
            this,
            [self,
             projectPath,
             mvsOutDir,
             selectedImageCount = selectedImages.size()](bool success) {
        if (!self)
        {
            return;
        }
        const bool project_matches = self->_owner &&
            self->_owner->currentProjectPath() == projectPath;
        if (success && project_matches)
        {
            self->_owner->refreshReconstructionQualityReport();
            emit self->depthMapBatchReady(mvsOutDir, selectedImageCount);
        }
        emit self->mvsProgressFinished(success && project_matches);
        if (!success && project_matches)
        {
            QMessageBox::warning(self->_parentWidget,
                                 QStringLiteral("深度图估计"),
                                 QStringLiteral("深度图估计失败或被取消。"));
        }
        if (self->_activeMvsGenerator)
        {
            self->_activeMvsGenerator->deleteLater();
            self->_activeMvsGenerator = nullptr;
        }
    });

    _activeMvsGenerator = gen;
    emit mvsProgressChanged(QStringLiteral("正在加载稀疏点云..."), 0);
    QPointer<DepthMapGenerator> genSelf(gen);
    (void)QtConcurrent::run([self, genSelf, sparseXyz, views, request, projectPath]() {
        SparseCloud sparse;
        if (!sparseXyz.isEmpty() && QFile::exists(sparseXyz))
        {
            // Sparse-cloud cleanup is a small preparation step. Keep it on CPU
            // so PlaPoint CUDA filtering cannot overlap GPU initialization with
            // the following PatchMatch task in the interactive workflow.
            SparseCloudPreprocessor pp(plapoint::ProcessingDevice::CPU);
            PreprocessResult ppRes;
            std::string ppErr;
            if (pp.run(xjw::common::io::toUtf8Path(sparseXyz), views, ppRes, &ppErr))
            {
                sparse = ppRes.cloud;
            }
        }
        if (!self || !genSelf)
        {
            return;
        }
        auto sparseCloud = std::make_shared<SparseCloud>(std::move(sparse));
        QMetaObject::invokeMethod(genSelf.data(), [self, genSelf, sparseCloud, projectPath]() {
            if (!self || !genSelf)
            {
                return;
            }
            if (!self->_owner || self->_owner->currentProjectPath() != projectPath)
            {
                if (self->_activeMvsGenerator == genSelf.data())
                {
                    self->_activeMvsGenerator = nullptr;
                }
                genSelf->deleteLater();
                emit self->mvsProgressFinished(false);
                return;
            }
            if (genSelf->isCancelled())
            {
                QMetaObject::invokeMethod(genSelf.data(), "finished", Qt::QueuedConnection, Q_ARG(bool, false));
                return;
            }
            genSelf->setSparseCloud(*sparseCloud);
            genSelf->start();
        }, Qt::QueuedConnection);
    });
    return true;
}

bool ProjectDenseReconstructionManager::startFuseDepthMapsAsync(const QJsonObject &settings)
{
    using namespace xjw::mvs;

    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("深度图融合")))
    {
        return false;
    }
    if (isMvsRunning())
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("深度图融合"),
                                 QStringLiteral("已有深度图或密集点云任务正在运行，请先等待或取消当前任务。"));
        return false;
    }

    const xjw::gui::project::DenseGenerationSettings request = denseGenerationSettingsFromJson(settings);
    const bool transientForModel = settings.value(QStringLiteral("transient_for_model")).toBool(false);
    const bool keepColor = settings.value(QStringLiteral("keepColor")).toBool(true);
    const bool keepNormals = settings.value(QStringLiteral("keepNormals")).toBool(true);

    const QString requested_output_dir = request.outputDir.trimmed().isEmpty()
        ? QString()
        : resolveProjectOutputDir(_owner->currentProjectPath(),
                                  request.outputDir,
                                  QStringLiteral("mvs_output"));
    const auto storedFramesResult = requested_output_dir.isEmpty()
        ? collectLatestStoredDepthFrames(_projectData->metadata())
        : collectStoredDepthFramesForDirectory(_projectData->metadata(), requested_output_dir);
    const std::vector<StoredDepthFrameRecord> &storedFrames = storedFramesResult.frames;
    if (storedFrames.size() < 2)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("深度图融合"),
                             storedFrames.empty()
                                 ? storedFramesResult.status.errorMessage
                                 : QStringLiteral("可用深度图数量不足（至少需要2张）。"));
        return false;
    }

    QStringList imagePaths;
    for (const auto &frame : storedFrames)
    {
        imagePaths.append(frame.refImage);
    }

    bool allCams = false;
    const QMap<QString, xjw::Camera> camMap = _owner->getCamerasForImages(imagePaths, &allCams);
    if (!allCams)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("深度图融合"),
                             QStringLiteral("部分深度图对应影像缺少相机参数，无法执行融合。"));
        return false;
    }

    const QString outputDir = requested_output_dir.isEmpty()
        ? storedFramesResult.batchDir
        : requested_output_dir;
    const QString outputPly = QDir(outputDir).filePath(QStringLiteral("dense_cloud.ply"));
    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    const auto cancelFlag = createActiveMvsCancelFlag();

    emit mvsProgressChanged(QStringLiteral("正在加载深度图批次..."), 0);

    const bool pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
    QPointer<ProjectDenseReconstructionManager> self(this);
    auto fusionWork = [self,
                       storedFrames,
                       camMap,
                       request,
                       keepColor,
                       keepNormals,
                       outputPly,
                       source_depth_map_dir = storedFramesResult.batchDir,
                       transientForModel,
                       pipelineMode,
                       projectPath,
                       cancelFlag]()
    {
        if (!self)
        {
            return;
        }
        const auto reportException = [self, cancelFlag, projectPath](const QString &message) {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, cancelFlag, projectPath, message]() {
                if (!self)
                {
                    return;
                }
                self->clearActiveMvsCancelFlag(cancelFlag);
                emit self->mvsProgressFinished(false);
                if (self->_owner && self->_owner->currentProjectPath() == projectPath)
                {
                    QMessageBox::warning(self->_parentWidget,
                                         QStringLiteral("深度图融合"),
                                         QStringLiteral("深度图融合异常终止：%1").arg(message));
                }
            }, Qt::QueuedConnection);
        };

        try
        {
        const auto isCancelled = [cancelFlag]() {
            return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
        };
        const auto finishCancelled = [self, cancelFlag]() {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, cancelFlag]() {
                if (!self)
                {
                    return;
                }
                self->clearActiveMvsCancelFlag(cancelFlag);
                LOG_INFO(QStringLiteral("[MVS] 深度图融合已取消"));
                emit self->mvsProgressFinished(false);
            }, Qt::QueuedConnection);
        };

        const int totalFrames = static_cast<int>(storedFrames.size());
        const int neighborCount = streamFusionWindowSize(request, totalFrames);
        const xjw::mvs::FusionConfig fusionConfig = buildDepthGenConfig(request, totalFrames).fusion;
        std::uint64_t frame_working_set = 0;
        for (const StoredDepthFrameRecord &record : storedFrames)
        {
            std::uint64_t record_working_set = 0;
            if (record.gridWidth > 0 && record.gridHeight > 0)
            {
                record_working_set = estimateFusionFrameWorkingSetBytes(
                    record.gridWidth, record.gridHeight, request.fusionMaxImageDim);
            }
            else
            {
                const qint64 raw_bytes = QFileInfo(record.rawDepthPath).size();
                const std::uint64_t source_pixels = raw_bytes > 64
                    ? static_cast<std::uint64_t>(raw_bytes - 64) / sizeof(float)
                    : 0ULL;
                record_working_set = source_pixels > 0
                    ? source_pixels * 24ULL + 16ULL * 1024ULL * 1024ULL
                    : 512ULL * 1024ULL * 1024ULL;
            }
            frame_working_set = std::max(frame_working_set, record_working_set);
        }
        const std::uint64_t available_memory = availablePhysicalMemoryBytes();
        const int load_workers = recommendedDepthFrameLoadWorkers(
            request.threads, available_memory, frame_working_set);
        auto current_progress = std::make_shared<std::atomic_int>(1);
        auto last_reported_load = std::make_shared<std::atomic_int>(0);
        DepthFrameLruCache depthFrameCache(storedFrames,
                                           camMap,
                                           fusionConfig,
                                           totalFrames,
                                           request.fusionMaxImageDim,
                                           fusionWindowCacheCapacity(neighborCount),
                                           load_workers,
                                           [self,
                                            totalFrames,
                                            current_progress,
                                            last_reported_load](const DepthFrameLruCache::LoadStats &stats)
                                           {
                                               const int previous = last_reported_load->exchange(
                                                   stats.loadedCount, std::memory_order_relaxed);
                                               if (stats.loadedCount <= previous)
                                               {
                                                   return;
                                               }
                                               LOG_INFO(QStringLiteral(
                                                            "[MVS] 深度帧加载 %1/%2: read=%3 ms postprocess=%4 ms resize=%5 ms total=%6 ms")
                                                            .arg(stats.loadedCount)
                                                            .arg(totalFrames)
                                                            .arg(stats.readMs, 0, 'f', 1)
                                                            .arg(stats.postprocessMs, 0, 'f', 1)
                                                            .arg(stats.resizeMs, 0, 'f', 1)
                                                            .arg(stats.totalMs, 0, 'f', 1));
                                               if (!self)
                                               {
                                                   return;
                                               }
                                               QMetaObject::invokeMethod(
                                                   self.data(),
                                                   [self, stats, totalFrames, current_progress]()
                                                   {
                                                       if (self)
                                                       {
                                                           emit self->mvsProgressChanged(
                                                               QStringLiteral("准备深度图：已加载 %1/%2")
                                                                   .arg(stats.loadedCount)
                                                                   .arg(totalFrames),
                                                               current_progress->load(
                                                                   std::memory_order_relaxed));
                                                       }
                                                   },
                                                   Qt::QueuedConnection);
                                           });
        std::vector<FusedPoint> fusedPoints;
        fusedPoints.reserve(100000);

        LOG_INFO(QStringLiteral("[MVS] 流式深度图融合: frames=%1 neighborWindow=%2 fusionMaxImageDim=%3 "
                                "cacheCapacity=%4 loadWorkers=%5 availableMemory=%6 MiB frameWorkingSet=%7 MiB")
                     .arg(totalFrames)
                     .arg(neighborCount)
                     .arg(request.fusionMaxImageDim)
                     .arg(fusionWindowCacheCapacity(neighborCount))
                     .arg(load_workers)
                     .arg(available_memory / (1024ULL * 1024ULL))
                     .arg(frame_working_set / (1024ULL * 1024ULL)));

        for (int refIndex = 0; refIndex < totalFrames; ++refIndex)
        {
            if (isCancelled())
            {
                finishCancelled();
                return;
            }

            const auto batchStart = std::chrono::steady_clock::now();
            const std::vector<int> windowIndices =
                nearestFusionWindowIndices(storedFrames, camMap, refIndex, neighborCount);
            std::vector<FusionFrameInput> frames;
            frames.reserve(windowIndices.size());
            for (const int frameIndex : windowIndices)
            {
                FusionFrameBuildResult frameResult = depthFrameCache.get(frameIndex);
                if (!frameResult.status.ok)
                {
                    const QString loadError = frameResult.status.errorMessage;
                    if (!self)
                    {
                        return;
                    }
                    QMetaObject::invokeMethod(self.data(), [self, loadError, cancelFlag]() {
                        if (!self)
                        {
                            return;
                        }
                        self->clearActiveMvsCancelFlag(cancelFlag);
                        emit self->mvsProgressFinished(false);
                        QMessageBox::warning(self->_parentWidget, QStringLiteral("深度图融合"), loadError);
                    }, Qt::QueuedConnection);
                    return;
                }
                frames.push_back(std::move(frameResult.frame));
            }
            const auto loadDone = std::chrono::steady_clock::now();

            if (refIndex + 1 < totalFrames)
            {
                const std::vector<int> next_window = nearestFusionWindowIndices(
                    storedFrames, camMap, refIndex + 1, neighborCount);
                depthFrameCache.prefetch(next_window);
            }

            if (frames.size() < 2)
            {
                continue;
            }

            const int baseProgress = std::clamp((refIndex * 90) / std::max(1, totalFrames), 1, 90);
            current_progress->store(baseProgress, std::memory_order_relaxed);
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, refIndex, totalFrames, baseProgress, windowSize = frames.size()]() {
                if (!self)
                {
                    return;
                }
                emit self->mvsProgressChanged(QStringLiteral("正在加载深度图 %1/%2，窗口 %3 帧")
                                                  .arg(refIndex + 1)
                                                  .arg(totalFrames)
                                                  .arg(static_cast<int>(windowSize)),
                                              baseProgress);
            }, Qt::QueuedConnection);

            StereoFusionConfig fusionCfg;
            fusionCfg.minNumPixels = std::max(1, request.minConsistentViews);
            fusionCfg.maxReprojError = request.depthConsistency;
            fusionCfg.maxDepthError = request.fusionRelDepthThreshold;
            fusionCfg.checkNumImages = std::min(neighborCount, static_cast<int>(frames.size()) - 1);
            fusionCfg.workerCount = std::max(1, request.threads);
            fusionCfg.useColor = keepColor;
            fusionCfg.colorCacheCapacity = keepColor ? 2 : 0;
            fusionCfg.fuseOnlyFirstFrame = true;
            fusionCfg.enableLowYieldFallback = totalFrames <= 32;
            fusionCfg.cancelFlag = cancelFlag;
            if (static_cast<int>(frames.size()) <= 2)
            {
                fusionCfg.minNumPixels = 1;
                fusionCfg.maxDepthError = std::max(fusionCfg.maxDepthError, 0.08f);
                fusionCfg.maxReprojError = std::max(fusionCfg.maxReprojError, 3.0f);
            }

            DepthMapFusion fusion(fusionCfg);
            std::vector<FusedPoint> batchPoints;
            std::string fuseErr;
            const bool fuseOk = fusion.fuse(frames,
                                            batchPoints,
                                            [self, refIndex, totalFrames, current_progress](const std::string &stage, float ratio) {
                                                if (!self)
                                                {
                                                    return;
                                                }
                                                QMetaObject::invokeMethod(
                                                    self.data(),
                                                    [self, stage, ratio, refIndex, totalFrames, current_progress]() {
                                                    if (!self)
                                                    {
                                                        return;
                                                    }
                                                    const int progressValue = std::clamp(
                                                        static_cast<int>(((refIndex + ratio) * 90.0f) /
                                                                         static_cast<float>(std::max(1, totalFrames))),
                                                        1,
                                                        90);
                                                    current_progress->store(progressValue,
                                                                            std::memory_order_relaxed);
                                                    emit self->mvsProgressChanged(
                                                        QStringLiteral("流式深度图融合 %1/%2: %3")
                                                            .arg(refIndex + 1)
                                                            .arg(totalFrames)
                                                            .arg(QString::fromStdString(stage)),
                                                        progressValue);
                                                    },
                                                    Qt::QueuedConnection);
                                            },
                                            &fuseErr);
            const auto fuseDone = std::chrono::steady_clock::now();
            const double loadMs =
                std::chrono::duration<double, std::milli>(loadDone - batchStart).count();
            const double fuseMs =
                std::chrono::duration<double, std::milli>(fuseDone - loadDone).count();
            const DepthFrameLruCache::LoadStats load_stats = depthFrameCache.stats();
            LOG_INFO(QStringLiteral("[MVS] 流式深度图融合批次 %1/%2: ref=%3 window=%4 wait=%5 ms fuse=%6 ms "
                                    "points=%7 loaded=%8 readTotal=%9 ms postTotal=%10 ms resizeTotal=%11 ms")
                         .arg(refIndex + 1)
                         .arg(totalFrames)
                         .arg(QFileInfo(storedFrames[refIndex].refImage).fileName())
                         .arg(frames.size())
                         .arg(loadMs, 0, 'f', 1)
                         .arg(fuseMs, 0, 'f', 1)
                         .arg(batchPoints.size())
                         .arg(load_stats.loadedCount)
                         .arg(load_stats.readMs, 0, 'f', 1)
                         .arg(load_stats.postprocessMs, 0, 'f', 1)
                         .arg(load_stats.resizeMs, 0, 'f', 1));

            if (isCancelled())
            {
                finishCancelled();
                return;
            }
            if (!fuseOk)
            {
                const QString err = QString::fromStdString(fuseErr);
                if (!self)
                {
                    return;
                }
                QMetaObject::invokeMethod(self.data(), [self, err, cancelFlag]() {
                    if (!self)
                    {
                        return;
                    }
                    self->clearActiveMvsCancelFlag(cancelFlag);
                    emit self->mvsProgressFinished(false);
                    QMessageBox::warning(self->_parentWidget,
                                         QStringLiteral("深度图融合"),
                                         QStringLiteral("深度图融合失败：%1").arg(err));
                }, Qt::QueuedConnection);
                return;
            }

            fusedPoints.insert(fusedPoints.end(),
                               std::make_move_iterator(batchPoints.begin()),
                               std::make_move_iterator(batchPoints.end()));

            if ((refIndex + 1) == totalFrames || ((refIndex + 1) % std::max(1, totalFrames / 100)) == 0)
            {
                const int progressValue = std::clamp(((refIndex + 1) * 90) / std::max(1, totalFrames), 1, 90);
                if (!self)
                {
                    return;
                }
                QMetaObject::invokeMethod(self.data(), [self, refIndex, totalFrames, progressValue]() {
                    if (!self)
                    {
                        return;
                    }
                    emit self->mvsProgressChanged(QStringLiteral("流式深度图融合 %1/%2")
                                                      .arg(refIndex + 1)
                                                      .arg(totalFrames),
                                                  progressValue);
                }, Qt::QueuedConnection);
            }
        }

        const PlaPC pointCloud = fusedPointsToPointCloud(fusedPoints, keepColor, keepNormals);
        if (isCancelled())
        {
            finishCancelled();
            return;
        }
        QString saveError;
        if (!writePointCloudPly(outputPly, pointCloud, keepNormals, &saveError))
        {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, saveError, cancelFlag]() {
                if (!self)
                {
                    return;
                }
                self->clearActiveMvsCancelFlag(cancelFlag);
                emit self->mvsProgressFinished(false);
                QMessageBox::warning(self->_parentWidget,
                                     QStringLiteral("深度图融合"),
                                     QStringLiteral("保存密集点云失败：%1").arg(saveError));
            }, Qt::QueuedConnection);
            return;
        }
        if (isCancelled())
        {
            QFile::remove(outputPly);
            finishCancelled();
            return;
        }

        const int pointCount = static_cast<int>(pointCloud.size());
        if (!self)
        {
            return;
        }
        const int source_depth_map_count = static_cast<int>(storedFrames.size());
        QString source_depth_config_hash;
        QString source_project_input_signature;
        bool depth_config_consistent = true;
        bool project_input_consistent = true;
        for (const auto &frame : storedFrames)
        {
            if (source_depth_config_hash.isEmpty())
            {
                source_depth_config_hash = frame.configHash;
            }
            else if (source_depth_config_hash != frame.configHash)
            {
                depth_config_consistent = false;
            }
            if (source_project_input_signature.isEmpty())
            {
                source_project_input_signature = frame.projectInputSignature;
            }
            else if (source_project_input_signature != frame.projectInputSignature)
            {
                project_input_consistent = false;
            }
        }
        if (!depth_config_consistent)
        {
            source_depth_config_hash.clear();
        }
        if (!project_input_consistent)
        {
            source_project_input_signature.clear();
        }
        QMetaObject::invokeMethod(self.data(),
                                  [self,
                                   outputPly,
                                   pointCount,
                                    source_depth_map_dir,
                                    source_depth_map_count,
                                    source_depth_config_hash,
                                    source_project_input_signature,
                                    transientForModel,
                                    pipelineMode,
                                    projectPath,
                                    cancelFlag]() {
            if (!self)
            {
                return;
            }
            if (!self->_owner || self->_owner->currentProjectPath() != projectPath)
            {
                self->clearActiveMvsCancelFlag(cancelFlag);
                emit self->mvsProgressFinished(false);
                return;
            }
            self->clearActiveMvsCancelFlag(cancelFlag);
            QJsonObject denseResult = makeDenseResultRecord(utcNowIso(), outputPly, pointCount);
            denseResult[QStringLiteral("source_depth_map_dir")] = source_depth_map_dir;
            denseResult[QStringLiteral("source_depth_map_count")] = source_depth_map_count;
            denseResult[QStringLiteral("source_depth_config_hash")] = source_depth_config_hash;
            denseResult[QStringLiteral("source_project_input_signature")] =
                source_project_input_signature;
            denseResult[QStringLiteral("fusion_pipeline_version")] =
                xjw::gui::project::kDenseFusionPipelineVersion;
            if (!transientForModel)
            {
                upsertProjectRecordByPath(self->_projectData,
                                          QStringLiteral("dense_cloud_results"),
                                          QStringLiteral("dense_cloud_xyz"),
                                          denseResult);
            }
            self->_owner->refreshReconstructionQualityReport();
            emit self->denseCloudResultReady(outputPly, pointCount);
            emit self->mvsProgressFinished(true);
            if (!pipelineMode)
            {
                QMessageBox::information(self->_parentWidget,
                                         QStringLiteral("深度图融合"),
                                         QStringLiteral("密集点云生成完成，共 %1 个点。").arg(pointCount));
            }
        }, Qt::QueuedConnection);
        }
        catch (const std::exception &error)
        {
            reportException(QString::fromUtf8(error.what()));
        }
        catch (...)
        {
            reportException(QStringLiteral("未知后台异常"));
        }
    };

    xjw::gui::tasks::runGuarded(this,
                                std::move(fusionWork),
                                [](ProjectDenseReconstructionManager *) {});
    return true;
}

bool ProjectDenseReconstructionManager::startGenerateDenseCloudAsync(const QJsonObject &settings)
{
    using namespace xjw::mvs;

    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("稠密重建")))
    {
        return false;
    }
    if (isMvsRunning())
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("稠密重建"),
                                 QStringLiteral("已有深度图或密集点云任务正在运行，请先等待或取消当前任务。"));
        return false;
    }

    const xjw::gui::project::DenseGenerationSettings request = denseGenerationSettingsFromJson(settings);
    const bool transientForModel = settings.value(QStringLiteral("transient_for_model")).toBool(false);
    const QJsonObject meta = _projectData->metadata();
    const QJsonArray atArr = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (atArr.isEmpty())
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("稠密重建"),
                             QStringLiteral("未找到空三结果，请先执行空中三角测量。"));
        return false;
    }

    const int realIdx = (request.atIndex >= 0 && request.atIndex < atArr.size())
        ? request.atIndex
        : findLatestProductionAtResultIndex(meta);
    if (realIdx < 0)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("稠密重建"),
                             QStringLiteral("未找到可用的正式 SfM/BA 稀疏点云结果。请先运行三维重建/空三。"));
        return false;
    }
    const QJsonObject atResult = atArr[realIdx].toObject();
    const QString project_input_signature = projectDepthInputSignature(meta, realIdx);
    const QString reconstruction_generation_id =
        atResult.value(QStringLiteral("reconstruction_generation_id")).toString();
    if (!xjw::gui::project::isProductionSparseResult(atResult))
    {
        const QString reason = sparseResultBlockingReason(atResult);
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("稠密重建"),
                             reason.isEmpty()
                                 ? QStringLiteral("所选稀疏点云不是正式 SfM/BA 结果。")
                                 : reason);
        return false;
    }
    const QJsonArray selImgArr = atResult.value(QStringLiteral("selected_images")).toArray();
    const QJsonObject files = atResult.value(QStringLiteral("files")).toObject();
    const QString sparseXyz = files.value(QStringLiteral("sparse_cloud_xyz")).toString();

    QStringList selectedImages;
    for (const auto &value : selImgArr)
    {
        selectedImages.append(value.toString());
    }
    if (selectedImages.size() < 2)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("稠密重建"),
                             QStringLiteral("空三结果中影像数量不足（至少需要2张）。"));
        return false;
    }

    bool allCams = false;
    const QMap<QString, xjw::Camera> camMap = _owner->getCamerasForImages(selectedImages, &allCams);
    if (!allCams)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("稠密重建"),
                             QStringLiteral("部分影像缺少相机参数，无法执行稠密重建。"));
        return false;
    }

    std::vector<CameraView> views;
    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    for (const QString &imgPath : selectedImages)
    {
        CameraView view;
        view.imagePath = xjw::common::io::toUtf8Path(imgPath);
        view.validRegionMaskPath = xjw::common::io::toUtf8Path(
            xjw::common::project::ProjectIO::findMaskForImage(projectPath, imgPath));
        if (!cameraForImagePath(camMap, imgPath, &view.camera))
        {
            QMessageBox::warning(_parentWidget,
                                 QStringLiteral("稠密重建"),
                                 QStringLiteral("影像缺少相机参数：%1").arg(QDir::toNativeSeparators(imgPath)));
            return false;
        }
        applyImageSizeToMvsView(imgPath, &view);
        views.push_back(std::move(view));
    }

    DepthGenConfig genCfg = buildDepthGenConfig(request, static_cast<int>(views.size()));
    genCfg.inputSignature = project_input_signature.toStdString();
    genCfg.saveIntermediateDepthMaps = true;
    const QString mvsOutDir = resolveProjectOutputDir(_owner->currentProjectPath(), request.outputDir, QStringLiteral("mvs_output"));
    genCfg.intermediateDir = xjw::common::io::toUtf8Path(mvsOutDir);
    attachMvsSourcePairQualities(&genCfg, _owner->currentProjectPath(), views);
    if (request.pipelineMode)
    {
        // 一键流程复用“深度图融合生成密集点云”的同一入口，避免内部融合与手动融合产物不一致。
        genCfg.runFusion = false;
    }

    const QSet<int> existingIndices = collectExistingDepthFrameIndices(mvsOutDir, selectedImages.size());
    bool continueMissingMode = false;
    if (!existingIndices.isEmpty())
    {
        ExistingDepthAction action;
        if (request.pipelineMode)
        {
            const bool force_depth_recompute =
                settings.value(QStringLiteral("force_depth_recompute")).toBool(false);
            action = force_depth_recompute
                ? ExistingDepthAction::Overwrite
                : ExistingDepthAction::ContinueMissing;
            LOG_INFO(force_depth_recompute
                         ? QStringLiteral("[MVS][pipeline] 检测到已有深度图 %1 帧，按模型设置强制重算")
                               .arg(existingIndices.size())
                         : QStringLiteral("[MVS][pipeline] 检测到已有深度图 %1 帧，自动续跑")
                               .arg(existingIndices.size()));
        }
        else
        {
            action = askExistingDepthAction(_parentWidget,
                                            existingIndices.size(),
                                            selectedImages.size(),
                                            mvsOutDir);
        }
        if (action == ExistingDepthAction::Cancel)
        {
            return false;
        }

        if (action == ExistingDepthAction::Overwrite)
        {
            removeDepthArtifactsForIndices(mvsOutDir, existingIndices);
            LOG_INFO(QStringLiteral("[MVS] 用户选择覆盖重算，已清理已有深度图 %1 帧").arg(existingIndices.size()));
        }
        else if (action == ExistingDepthAction::ContinueMissing)
        {
            continueMissingMode = true;
            genCfg.runFusion = false;
            LOG_INFO(QStringLiteral("[MVS] 发现 %1 个深度文件，"
                                    "将由 manifest 校验影像、相机与参数；完成后自动执行融合")
                         .arg(existingIndices.size()));
        }
    }

    auto *gen = new DepthMapGenerator(this);
    gen->setViews(views);
    gen->setConfig(genCfg);
    gen->setOutputDir(xjw::common::io::toUtf8Path(mvsOutDir));

    QPointer<ProjectDenseReconstructionManager> self(this);
    QObject::connect(gen, &DepthMapGenerator::progressChanged, this,
                     [self, projectPath](const QString &stage, float ratio) {
        if (!self || !self->_owner || self->_owner->currentProjectPath() != projectPath)
        {
            return;
        }
        emit self->mvsProgressChanged(stage, static_cast<int>(ratio * 100));
    });
    connect(gen, &DepthMapGenerator::errorOccurred, this, [](const QString &msg) {
        qWarning() << "[MVS] 错误:" << msg;
    });
    connect(gen, &DepthMapGenerator::depthMapArtifactSaved, this,
            [self,
             sparseXyz,
             mvsOutDir,
             projectPath,
             project_input_signature,
             reconstruction_generation_id,
             quality_profile = request.qualityProfile,
             batch_frame_count = selectedImages.size()](const QJsonObject &artifact) {
        if (!self || !self->_owner || self->_owner->currentProjectPath() != projectPath)
        {
            return;
        }
        QJsonObject depthResult = makeProjectDepthRecordFromArtifact(
            artifact,
            sparseXyz,
            mvsOutDir,
            batch_frame_count,
            project_input_signature,
            reconstruction_generation_id);
        if (depthResult.isEmpty())
        {
            return;
        }
        depthResult[QStringLiteral("quality_profile")] = quality_profile;
        upsertProjectRecordByPath(self->_projectData,
                                  QStringLiteral("depth_map_results"),
                                  QStringLiteral("depth_png"),
                                  depthResult);
    });
    connect(gen, &DepthMapGenerator::pointCloudReady, this,
            [self,
             mvsOutDir,
             projectPath,
             project_input_signature,
             reconstruction_generation_id,
             transientForModel,
             source_depth_map_count = static_cast<int>(selectedImages.size())](const std::vector<DensePoint> &cloud) {
        if (!self || !self->_owner || self->_owner->currentProjectPath() != projectPath)
        {
            return;
        }
        if ((cloud.size() == 0))
        {
            return;
        }
        const QString plyPath = mvsOutDir + QStringLiteral("/dense_cloud.ply");
        const PlaPC pointCloud = densePointsToPointCloud(cloud);
        QString saveErr;
        if (writePointCloudPly(plyPath, pointCloud, false, &saveErr))
        {
            QJsonObject denseResult = makeDenseResultRecord(
                utcNowIso(),
                plyPath,
                static_cast<int>(cloud.size()));
            denseResult[QStringLiteral("source_depth_map_dir")] = mvsOutDir;
            const auto stored_frames = collectStoredDepthFramesForDirectory(
                self->_projectData->metadata(),
                mvsOutDir);
            const int persisted_depth_map_count = stored_frames.status.ok
                ? static_cast<int>(stored_frames.frames.size())
                : source_depth_map_count;
            denseResult[QStringLiteral("source_depth_map_count")] =
                persisted_depth_map_count;
            QString source_depth_config_hash;
            if (stored_frames.status.ok)
            {
                for (const auto &frame : stored_frames.frames)
                {
                    if (source_depth_config_hash.isEmpty())
                    {
                        source_depth_config_hash = frame.configHash;
                    }
                    else if (source_depth_config_hash != frame.configHash)
                    {
                        source_depth_config_hash.clear();
                        break;
                    }
                }
            }
            denseResult[QStringLiteral("source_depth_config_hash")] =
                source_depth_config_hash;
            denseResult[QStringLiteral("source_project_input_signature")] =
                project_input_signature;
            denseResult[QStringLiteral("fusion_pipeline_version")] =
                xjw::gui::project::kDenseFusionPipelineVersion;
            denseResult[QStringLiteral("source_reconstruction_generation_id")] =
                reconstruction_generation_id;
            if (!transientForModel)
            {
                upsertProjectRecordByPath(self->_projectData,
                                          QStringLiteral("dense_cloud_results"),
                                          QStringLiteral("dense_cloud_xyz"),
                                          denseResult);
            }
            emit self->denseCloudResultReady(plyPath, static_cast<int>(cloud.size()));
            if (self->_owner)
            {
                self->_owner->refreshReconstructionQualityReport();
            }
        }
        else
        {
            qWarning() << "[MVS] 保存失败:" << saveErr;
        }
    });
    connect(gen,
            &DepthMapGenerator::finished,
            this,
            [self, settings, continueMissingMode, projectPath](bool success)
    {
        if (!self)
        {
            return;
        }
        if (self->_activeMvsGenerator)
        {
            self->_activeMvsGenerator->deleteLater();
            self->_activeMvsGenerator = nullptr;
        }

        if (!self->_owner || self->_owner->currentProjectPath() != projectPath)
        {
            emit self->mvsProgressFinished(false);
            return;
        }

        if (success)
        {
            self->_owner->refreshReconstructionQualityReport();
        }
        const bool pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
        const bool shouldStartFusion = success && (continueMissingMode || pipelineMode);

        if (!shouldStartFusion)
        {
            emit self->mvsProgressFinished(success);
        }

        if (shouldStartFusion)
        {
            self->_mvsTransitionPending = true;
            if (!pipelineMode)
            {
                QMessageBox::information(self->_parentWidget,
                                         QStringLiteral("稠密重建"),
                                         QStringLiteral("缺失深度图补齐完成，正在开始融合。"));
            }
            QMetaObject::invokeMethod(self.data(),
                                      [self, settings, projectPath]() {
                                          if (!self || !self->_mvsTransitionPending)
                                          {
                                              return;
                                          }
                                          self->_mvsTransitionPending = false;
                                          if (!self->_owner ||
                                              self->_owner->currentProjectPath() != projectPath)
                                          {
                                              emit self->mvsProgressFinished(false);
                                              return;
                                          }
                                          if (!self->startFuseDepthMapsAsync(settings))
                                          {
                                              emit self->mvsProgressFinished(false);
                                          }
                                      },
                                      Qt::QueuedConnection);
        }
        else if (!pipelineMode)
        {
            QMessageBox::information(self->_parentWidget,
                                     QStringLiteral("稠密重建"),
                                     success ? QStringLiteral("稠密点云生成完成。")
                                             : QStringLiteral("稠密点云生成失败或被取消。"));
        }
    });

    _activeMvsGenerator = gen;
    emit mvsProgressChanged(QStringLiteral("正在加载稀疏点云..."), 0);
    QPointer<DepthMapGenerator> genSelf(gen);
    (void)QtConcurrent::run([self, genSelf, sparseXyz, views, request, projectPath]() {
        SparseCloud sparse;
        if (!sparseXyz.isEmpty() && QFile::exists(sparseXyz))
        {
            // Sparse-cloud cleanup is a small preparation step. Keep it on CPU
            // so PlaPoint CUDA filtering cannot overlap GPU initialization with
            // the following PatchMatch task in the interactive workflow.
            SparseCloudPreprocessor pp(plapoint::ProcessingDevice::CPU);
            PreprocessResult ppRes;
            std::string ppErr;
            if (pp.run(xjw::common::io::toUtf8Path(sparseXyz), views, ppRes, &ppErr))
            {
                sparse = ppRes.cloud;
            }
        }
        if (!self || !genSelf)
        {
            return;
        }
        auto sparseCloud = std::make_shared<SparseCloud>(std::move(sparse));
        QMetaObject::invokeMethod(genSelf.data(), [self, genSelf, sparseCloud, projectPath]() {
            if (!self || !genSelf)
            {
                return;
            }
            if (!self->_owner || self->_owner->currentProjectPath() != projectPath)
            {
                if (self->_activeMvsGenerator == genSelf.data())
                {
                    self->_activeMvsGenerator = nullptr;
                }
                genSelf->deleteLater();
                emit self->mvsProgressFinished(false);
                return;
            }
            if (genSelf->isCancelled())
            {
                QMetaObject::invokeMethod(genSelf.data(), "finished", Qt::QueuedConnection, Q_ARG(bool, false));
                return;
            }
            genSelf->setSparseCloud(*sparseCloud);
            genSelf->start();
        }, Qt::QueuedConnection);
    });
    return true;
}

void ProjectDenseReconstructionManager::startDenseCloudRefineAsync(const QJsonObject &settings)
{
    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("密集点云后处理")))
    {
        return;
    }
    if (isMvsRunning())
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("密集点云后处理"),
                                 QStringLiteral("已有深度图或密集点云任务正在运行，请先等待或取消当前任务。"));
        return;
    }

    QString inputPly;
    QString denseError;
    if (!resolveLatestDenseCloudPath(_projectData, &inputPly, &denseError))
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("密集点云后处理"),
                             QStringLiteral("%1，请先执行深度图融合生成密集点云。").arg(denseError));
        return;
    }

    const xjw::gui::project::DenseRefineSettings request = denseRefineSettingsFromJson(settings);
    const bool pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
    const QString outDir = QFileInfo(inputPly).absolutePath();
    const QString ts = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const QString outputPly = outDir + QStringLiteral("/dense_cloud_refined_%1.ply").arg(ts);
    const auto cancelFlag = createActiveMvsCancelFlag();

    emit mvsProgressChanged(QStringLiteral("正在加载密集点云..."), 0);
    QPointer<ProjectDenseReconstructionManager> self(this);
    auto refineWork = [self, inputPly, outputPly, request, pipelineMode, cancelFlag]()
    {
        if (!self)
        {
            return;
        }
        const auto isCancelled = [cancelFlag]() {
            return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
        };
        const auto finishCancelled = [self, cancelFlag]() {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, cancelFlag]() {
                if (!self)
                {
                    return;
                }
                self->clearActiveMvsCancelFlag(cancelFlag);
                LOG_INFO(QStringLiteral("[MVS] 密集点云后处理已取消"));
                emit self->mvsProgressFinished(false);
            }, Qt::QueuedConnection);
        };

        const auto postProgress = [self](const QString &message, int progressValue) {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, message, progressValue]() {
                if (!self)
                {
                    return;
                }
                emit self->mvsProgressChanged(message, progressValue);
            }, Qt::QueuedConnection);
        };

        std::uint64_t streamingVertexCount = 0;
        if (shouldUseStreamingDenseRefine(inputPly, request, &streamingVertexCount))
        {
            StreamingDenseRefineResult streamingResult;
            LOG_INFO(QStringLiteral("[DenseRefine] 大点云启用流式地表清理: inputPoints=%1 threshold=%2")
                         .arg(static_cast<double>(streamingVertexCount), 0, 'f', 0)
                         .arg(static_cast<double>(kStreamingDenseRefineMinPoints), 0, 'f', 0));
            if (runStreamingDenseCloudRefineCli(inputPly,
                                                outputPly,
                                                request,
                                                isCancelled,
                                                postProgress,
                                                &streamingResult))
            {
                if (!self)
                {
                    return;
                }
                QMetaObject::invokeMethod(self.data(), [self,
                                                        inputPly,
                                                        outputPly,
                                                        request,
                                                        streamingResult,
                                                        pipelineMode,
                                                        cancelFlag]() {
                    if (!self)
                    {
                        return;
                    }
                    self->clearActiveMvsCancelFlag(cancelFlag);
                    QJsonObject record = makeDenseResultRecord(utcNowIso(),
                                                               outputPly,
                                                               streamingResult.pointCount);
                    record[QStringLiteral("stage")] = QStringLiteral("production");
                    record[QStringLiteral("quality_stage")] = QStringLiteral("terrain");
                    record[QStringLiteral("operation")] = QStringLiteral("dense_cloud_surface_cleanup");
                    record[QStringLiteral("backend")] = QStringLiteral("streaming_cli");
                    record[QStringLiteral("source_dense_cloud")] = inputPly;
                    record[QStringLiteral("terrain_filter_passes")] = request.terrainFilterPasses;
                    record[QStringLiteral("terrain_spike_filter")] = streamingResult.terrainReportJson;
                    record[QStringLiteral("dense_refine_report")] = streamingResult.reportJson;
                    if (request.normalsEnabled)
                    {
                        record[QStringLiteral("normal_status")] =
                            QStringLiteral("skipped_for_streaming_large_cloud");
                    }
                    upsertProjectRecordByPath(self->_projectData,
                                              QStringLiteral("dense_cloud_results"),
                                              QStringLiteral("dense_cloud_xyz"),
                                              record);
                    if (self->_owner)
                    {
                        self->_owner->refreshReconstructionQualityReport();
                    }
                    emit self->denseCloudResultReady(outputPly, streamingResult.pointCount);
                    emit self->mvsProgressFinished(true);
                    if (!pipelineMode)
                    {
                        QMessageBox::information(
                            self->_parentWidget,
                            QStringLiteral("密集点云后处理"),
                            QStringLiteral("流式后处理完成，共 %1 个点。").arg(streamingResult.pointCount));
                    }
                }, Qt::QueuedConnection);
                return;
            }

            if (streamingResult.cancelled)
            {
                finishCancelled();
                return;
            }
            LOG_WARN(QStringLiteral("[DenseRefine] 流式地表清理不可用，回退内存路径: %1")
                         .arg(streamingResult.error));
        }

        PlaPC cloud;
        QString loadErr;
        if (!readPointCloudPly(inputPly, &cloud, &loadErr))
        {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, loadErr, cancelFlag]() {
                if (!self)
                {
                    return;
                }
                self->clearActiveMvsCancelFlag(cancelFlag);
                emit self->mvsProgressFinished(false);
                QMessageBox::warning(self->_parentWidget,
                                     QStringLiteral("密集点云后处理"),
                                     QStringLiteral("加载点云失败：%1").arg(loadErr));
            }, Qt::QueuedConnection);
            return;
        }
        if (isCancelled())
        {
            finishCancelled();
            return;
        }

        const DenseRefinePreconditionStats precondition =
            preconditionDenseRefineCloudForFilters(&cloud, request, postProgress);
        xjw::mvs::TerrainHeightSpikeFilterReport terrainSpikeReport;
        if (isCancelled())
        {
            finishCancelled();
            return;
        }

        if (request.sorEnabled)
        {
            if (isCancelled())
            {
                finishCancelled();
                return;
            }
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self]() {
                if (!self)
                {
                    return;
                }
                emit self->mvsProgressChanged(QStringLiteral("统计离群点移除 (SOR)..."), 20);
            }, Qt::QueuedConnection);

            const auto beforeSor = cloud.size();
            plapoint::ProcessingReport sorReport;
            cloud = sorFilter(cloud,
                              request.sorK,
                              static_cast<float>(request.sorStdDev),
                              request.processingDevice,
                              &sorReport);
            logPlaPointReport(QStringLiteral("统计离群点移除 (SOR)"), sorReport, beforeSor, cloud.size());
            if (isCancelled())
            {
                finishCancelled();
                return;
            }

            if (cloud.size() > 64)
            {
                // Compute bounds
                float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
                float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
                for (size_t i = 0; i < cloud.size(); ++i)
                {
                    float x = cloud.points()(static_cast<plamatrix::Index>(i), 0);
                    float y = cloud.points()(static_cast<plamatrix::Index>(i), 1);
                    float z = cloud.points()(static_cast<plamatrix::Index>(i), 2);
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (z < minZ) minZ = z;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                    if (z > maxZ) maxZ = z;
                }

                const double dx = static_cast<double>(maxX - minX);
                const double dy = static_cast<double>(maxY - minY);
                const double dz = static_cast<double>(maxZ - minZ);
                const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double volume = std::max(dx * dy * dz, 1e-12);
                const double density = static_cast<double>(cloud.size()) / volume;
                const int radiusMinNeighbors = std::clamp(request.sorK / 2, 6, 32);
                double adaptiveRadius = std::cbrt(std::max(1.0, static_cast<double>(radiusMinNeighbors))
                                                  / std::max(density, 1e-12));
                adaptiveRadius *= 1.2;
                if (diag > 1e-9)
                {
                    const double radiusMin = std::max(diag * 0.001, 1e-4);
                    const double radiusMax = std::max(radiusMin * 2.0, diag * 0.08);
                    adaptiveRadius = std::clamp(adaptiveRadius, radiusMin, radiusMax);
                }

                if (isCancelled())
                {
                    finishCancelled();
                    return;
                }
                if (!self)
                {
                    return;
                }
                QMetaObject::invokeMethod(self.data(), [self]() {
                    if (!self)
                    {
                        return;
                    }
                    emit self->mvsProgressChanged(QStringLiteral("半径离群点移除..."), 35);
                }, Qt::QueuedConnection);
                const auto beforeRadius = cloud.size();
                plapoint::ProcessingReport radiusReport;
                cloud = radiusFilter(cloud,
                                     static_cast<float>(adaptiveRadius),
                                     radiusMinNeighbors,
                                     request.processingDevice,
                                     &radiusReport);
                logPlaPointReport(QStringLiteral("半径离群点移除"), radiusReport, beforeRadius, cloud.size());
                if (isCancelled())
                {
                    finishCancelled();
                    return;
                }

            }
        }
        if (request.voxelEnabled && request.voxelSize > 0.0 && !precondition.consumedRequestedVoxel)
        {
            if (isCancelled())
            {
                finishCancelled();
                return;
            }
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self]() {
                if (!self)
                {
                    return;
                }
                emit self->mvsProgressChanged(QStringLiteral("体素下采样..."), 50);
            }, Qt::QueuedConnection);
            const auto beforeVoxel = cloud.size();
            plapoint::ProcessingReport voxelReport;
            cloud = voxelDownsample(cloud,
                                    static_cast<float>(request.voxelSize),
                                    request.processingDevice,
                                    &voxelReport);
            logPlaPointReport(QStringLiteral("体素下采样"), voxelReport, beforeVoxel, cloud.size());
            if (isCancelled())
            {
                finishCancelled();
                return;
            }
        }
        if (request.terrainSpikeFilterEnabled)
        {
            if (isCancelled())
            {
                finishCancelled();
                return;
            }
            postProgress(QStringLiteral("局部高度突刺过滤..."), 60);
            const auto beforeTerrainFilter = cloud.size();
            cloud = xjw::mvs::filterTerrainHeightSpikes(
                cloud,
                terrainSpikeOptionsFromRequest(request),
                &terrainSpikeReport);
            LOG_INFO(QStringLiteral("[DenseRefine] 局部高度突刺过滤完成: %1 -> %2 点，移除 %3 点，"
                                    "cellZRangeP95 %4 -> %5")
                         .arg(beforeTerrainFilter)
                         .arg(cloud.size())
                         .arg(terrainSpikeReport.removedPoints)
                         .arg(terrainSpikeReport.p95CellZRangeBefore, 0, 'f', 4)
                         .arg(terrainSpikeReport.p95CellZRangeAfter, 0, 'f', 4));
            if (isCancelled())
            {
                finishCancelled();
                return;
            }
        }
        if (request.normalsEnabled)
        {
            if (isCancelled())
            {
                finishCancelled();
                return;
            }
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self]() {
                if (!self)
                {
                    return;
                }
                emit self->mvsProgressChanged(QStringLiteral("估计法向量..."), 70);
            }, Qt::QueuedConnection);

            plapoint::ProcessingReport normalReport;
            auto normals = estimateNormals(cloud, request.normalK, request.processingDevice, &normalReport);
            cloud.setNormals(std::move(normals));
            logPlaPointReport(QStringLiteral("估计法向量"), normalReport, cloud.size(), cloud.size());
            if (isCancelled())
            {
                finishCancelled();
                return;
            }
        }

        if (isCancelled())
        {
            finishCancelled();
            return;
        }
        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(self.data(), [self]() {
            if (!self)
            {
                return;
            }
            emit self->mvsProgressChanged(QStringLiteral("保存后处理结果..."), 90);
        }, Qt::QueuedConnection);

        const bool writeNormalsOut = request.normalsEnabled && cloud.hasNormals();
        QString saveErr;
        if (!writePointCloudPly(outputPly, cloud, writeNormalsOut, &saveErr))
        {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, saveErr, cancelFlag]() {
                if (!self)
                {
                    return;
                }
                self->clearActiveMvsCancelFlag(cancelFlag);
                emit self->mvsProgressFinished(false);
                QMessageBox::warning(self->_parentWidget,
                                     QStringLiteral("密集点云后处理"),
                                     QStringLiteral("保存点云失败：%1").arg(saveErr));
            }, Qt::QueuedConnection);
            return;
        }
        if (isCancelled())
        {
            QFile::remove(outputPly);
            finishCancelled();
            return;
        }

        const int pointCount = static_cast<int>(cloud.size());
        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(self.data(), [self,
                                                inputPly,
                                                outputPly,
                                                pointCount,
                                                request,
                                                terrainSpikeReport,
                                                pipelineMode,
                                                cancelFlag]() {
            if (!self)
            {
                return;
            }
            self->clearActiveMvsCancelFlag(cancelFlag);
            QJsonObject record = makeDenseResultRecord(utcNowIso(), outputPly, pointCount);
            const bool terrainProductionCloud = request.terrainSpikeFilterEnabled;
            record[QStringLiteral("stage")] =
                terrainProductionCloud ? QStringLiteral("production") : QStringLiteral("refined");
            record[QStringLiteral("quality_stage")] =
                terrainProductionCloud ? QStringLiteral("terrain") : QStringLiteral("refined");
            record[QStringLiteral("operation")] =
                terrainProductionCloud ? QStringLiteral("dense_cloud_surface_cleanup") : QStringLiteral("dense_refine");
            record[QStringLiteral("source_dense_cloud")] = inputPly;
            record[QStringLiteral("terrain_spike_filter")] = terrainSpikeReportToJson(terrainSpikeReport);
            upsertProjectRecordByPath(self->_projectData,
                                      QStringLiteral("dense_cloud_results"),
                                      QStringLiteral("dense_cloud_xyz"),
                                      record);
            if (self->_owner)
            {
                self->_owner->refreshReconstructionQualityReport();
            }
            emit self->denseCloudResultReady(outputPly, pointCount);
            emit self->mvsProgressFinished(true);
            if (!pipelineMode)
            {
                QMessageBox::information(self->_parentWidget,
                                         QStringLiteral("密集点云后处理"),
                                         QStringLiteral("后处理完成，共 %1 个点。").arg(pointCount));
            }
        }, Qt::QueuedConnection);
    };

    xjw::gui::tasks::runGuarded(this,
                                std::move(refineWork),
                                [](ProjectDenseReconstructionManager *) {});
}

void ProjectDenseReconstructionManager::cancelMvs()
{
    if (_mvsTransitionPending)
    {
        _mvsTransitionPending = false;
        qDebug() << "[MVS] 已取消待启动的深度图融合";
        emit mvsProgressFinished(false);
        return;
    }

    bool requestedCancel = false;
    if (_activeMvsCancelFlag)
    {
        _activeMvsCancelFlag->store(true, std::memory_order_relaxed);
        requestedCancel = true;
    }

    if (_activeMvsGenerator)
    {
        auto *gen = qobject_cast<xjw::mvs::DepthMapGenerator *>(_activeMvsGenerator.data());
        if (gen)
        {
            gen->requestCancel();
            requestedCancel = true;
        }
    }

    if (!requestedCancel)
    {
        qDebug() << "[MVS] 没有正在运行的任务，取消请求直接收敛";
        emit mvsProgressFinished(false);
        return;
    }

    if (requestedCancel)
    {
        qDebug() << "[MVS] 已请求取消";
    }
}

bool ProjectDenseReconstructionManager::isMvsRunning() const
{
    return !_activeMvsGenerator.isNull() || static_cast<bool>(_activeMvsCancelFlag) ||
        _mvsTransitionPending;
}

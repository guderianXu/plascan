// =============================================================================
// 文件: DepthMapGenerator.cpp
// 模块: MVS - Qt 封装的深度图生成 + COLMAP BFS 融合
// =============================================================================

#include "DepthMapGenerator.h"
#include "DenseCloudBuilder.h"
#include "EpipolarRectifier.h"
#include "MvsViewSelection.h"
#include "Logger.h"
#include <QtConcurrent/QtConcurrent>
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
#include <cctype>
#include <functional>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw
{
namespace mvs
{

namespace
{

constexpr float kSkipContentMaskCoverage = 0.985f;
constexpr std::size_t kMaxInlineDenseFilterPoints = 500000;
constexpr std::size_t kMaxProjectedDepthQuantileSamples = 8192;

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

    explicit DepthFrameArtifactSaveQueue(SaveFn saveFn)
        : m_saveFn(std::move(saveFn))
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
            std::lock_guard<std::mutex> lock(m_mutex);
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

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
        }
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

                task = m_tasks.front();
                m_tasks.pop_front();
                ++m_activeTasks;
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
        }
    }

    SaveFn m_saveFn;
    std::deque<SaveTask> m_tasks;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::condition_variable m_idleCv;
    std::thread m_worker;
    std::atomic<bool> m_failed{false};
    bool m_stopping = false;
    int m_activeTasks = 0;
};

bool writeCvMatStorage(const std::string &path, const cv::Mat &matrix, std::string *errorMsg)
{
    cv::FileStorage storage(path, cv::FileStorage::WRITE);
    if (!storage.isOpened())
    {
        if (errorMsg)
        {
            *errorMsg = "无法写入矩阵文件: " + path;
        }
        return false;
    }

    storage << "mat" << matrix;
    storage.release();
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

    const cv::Mat validMask = depthMap > 0;
    cv::Mat vis = cv::Mat::zeros(depthMap.size(), CV_8U);
    if (cv::countNonZero(validMask) > 0)
    {
        double dMin = 0.0;
        double dMax = 0.0;
        cv::minMaxLoc(depthMap, &dMin, &dMax, nullptr, nullptr, validMask);
        if (dMax > dMin)
        {
            depthMap.convertTo(vis, CV_8U, 255.0 / (dMax - dMin), -255.0 * dMin / (dMax - dMin));
        }
    }
    vis.setTo(0, depthMap <= 0);

    cv::Mat colorVis;
    cv::applyColorMap(vis, colorVis, cv::COLORMAP_TURBO);
    colorVis.setTo(cv::Scalar(0, 0, 0), depthMap <= 0);

    if (!cv::imwrite(path, colorVis))
    {
        if (errorMsg)
        {
            *errorMsg = "无法写入深度预览 PNG: " + path;
        }
        return false;
    }

    return true;
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

int clampPatchMatchIterations(int requested)
{
    return std::clamp(requested, 1, 64);
}

PatchMatchConfig makeCoarsePatchMatchConfig(const PatchMatchConfig &baseConfig, bool useRectified)
{
    PatchMatchConfig coarseConfig = baseConfig;
    coarseConfig.downsampleFactor = std::max(baseConfig.downsampleFactor * 2, 4);
    coarseConfig.numIterations = std::max(1, std::min(clampPatchMatchIterations(baseConfig.numIterations),
                                                      std::max(1, baseConfig.numIterations / 2)));
    coarseConfig.confidenceThresh = std::min(baseConfig.confidenceThresh, 0.08f);
    coarseConfig.epipolarRectified = useRectified;
    return coarseConfig;
}

PatchMatchConfig makeFinePatchMatchConfig(const PatchMatchConfig &baseConfig,
                                          bool useRectified,
                                          float hintCoverage)
{
    PatchMatchConfig fineConfig = baseConfig;
    fineConfig.numIterations = clampPatchMatchIterations(baseConfig.numIterations);
    fineConfig.epipolarRectified = useRectified;

    if (hintCoverage > 0.35f)
    {
        fineConfig.numIterations = std::max(1, fineConfig.numIterations - 1);
    }
    if (hintCoverage > 0.55f)
    {
        fineConfig.numIterations = std::max(1, fineConfig.numIterations - 1);
    }

    return fineConfig;
}

cv::Size patchMatchWorkSize(const cv::Mat &image, const PatchMatchConfig &config)
{
    const int ds = std::max(1, config.downsampleFactor);
    return cv::Size(std::max(1, image.cols / ds), std::max(1, image.rows / ds));
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

bool isCudaMemoryFailure(const std::string &message)
{
    std::string lower;
    lower.reserve(message.size());
    for (unsigned char ch : message)
    {
        lower.push_back(static_cast<char>(std::tolower(ch)));
    }

    return lower.find("out of memory") != std::string::npos
        || lower.find("cuda_error_memory") != std::string::npos
        || (lower.find("cuda") != std::string::npos && lower.find("memory") != std::string::npos);
}

bool estimatePatchMatchWithAdaptiveCuda(
    const char *stageLabel,
    int refIdx,
    const cv::Mat &refGray,
    const std::vector<cv::Mat> &srcGrays,
    const PositiveDepthCameraModel &refCam,
    const std::vector<PositiveDepthCameraModel> &srcCams,
    float zNear,
    float zFar,
    const PatchMatchConfig &config,
    cv::Mat &depthOut,
    cv::Mat *confOut,
    std::string *errorMsg,
    const cv::Mat *hintDepth)
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
                                                  hintDepth);
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
                                               hintDepth))
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
                                                          hintDepth);
    if (!cpuOk && errorMsg && errorMsg->empty())
    {
        *errorMsg = lastCudaError;
    }
    return cpuOk;
}

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
static float det3(const float *R)
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

/**
 * @brief 使用输入视图、稀疏点云与配置构造深度图生成器。
 */
DepthMapGenerator::DepthMapGenerator(const std::vector<CameraView> &views,
                                     const PreprocessResult        &ppResult,
                                     const DepthGenConfig          &config,
                                     QObject                       *parent)
    : QObject(parent)
    , m_views(views)
    , m_sparse(ppResult.cloud)
    , m_config(config)
{
    qRegisterMetaType<DepthFrameResult>("DepthFrameResult");
    qRegisterMetaType<QSharedPointer<cv::Mat>>("QSharedPointer<cv::Mat>");
    qRegisterMetaType<std::vector<DensePoint>>("std::vector<DensePoint>");
}

DepthMapGenerator::~DepthMapGenerator()
{
}

// =============================================================================
void DepthMapGenerator::setViews(const std::vector<CameraView> &views)
{
    m_views = views;
    m_skipFrameMask.assign(m_views.size(), 0);
    clearFrameCaches();
}

void DepthMapGenerator::setSparseCloud(const SparseCloud &sparse)
{
    m_sparse = sparse;
    clearFrameCaches();
}

void DepthMapGenerator::setConfig(const DepthGenConfig &config)
{
    m_config = config;
}

void DepthMapGenerator::setSkippedFrameIndices(const std::vector<int> &indices)
{
    m_skipFrameMask.assign(m_views.size(), 0);
    for (int index : indices)
    {
        if (index >= 0 && index < static_cast<int>(m_skipFrameMask.size()))
        {
            m_skipFrameMask[static_cast<size_t>(index)] = 1;
        }
    }
}

void DepthMapGenerator::clearFrameCaches()
{
    m_frameCaches.clear();
    m_visibilityBits.clear();
    m_pairCommonCounts.clear();
    m_visibilityWordCount = 0;
    m_frameCachesReady = false;
}

bool DepthMapGenerator::isSparsePointVisibleInFrame(int viewIdx, size_t pointIndex) const
{
    if (!m_frameCachesReady
        || viewIdx < 0
        || viewIdx >= static_cast<int>(m_frameCaches.size())
        || m_visibilityWordCount == 0
        || pointIndex >= m_sparse.points.size())
    {
        return false;
    }

    const size_t word = pointIndex / 64;
    const size_t bit = pointIndex % 64;
    const size_t offset = static_cast<size_t>(viewIdx) * m_visibilityWordCount + word;
    if (offset >= m_visibilityBits.size())
    {
        return false;
    }
    return (m_visibilityBits[offset] & (uint64_t{1} << bit)) != 0;
}

void DepthMapGenerator::prepareFrameCaches()
{
    if (m_frameCachesReady)
    {
        return;
    }

    const int NV = static_cast<int>(m_views.size());
    const size_t pointCount = m_sparse.points.size();
    m_frameCaches.assign(static_cast<size_t>(std::max(0, NV)), FrameMvsCache{});
    m_visibilityWordCount = (pointCount + 63) / 64;
    m_visibilityBits.assign(static_cast<size_t>(std::max(0, NV)) * m_visibilityWordCount, 0);
    m_pairCommonCounts.assign(static_cast<size_t>(std::max(0, NV)) * static_cast<size_t>(std::max(0, NV)), 0);

    if (NV <= 0 || pointCount == 0)
    {
        m_frameCachesReady = true;
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
        const int requested = std::max(1, m_config.cpuWorkerCount);
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
            const auto &point = m_sparse.points[pointIndex];
            for (int viewIdx = 0; viewIdx < NV; ++viewIdx)
            {
                if (!isMvsSparsePointVisibleInView(m_views[viewIdx], point))
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

            auto &visible = m_frameCaches[static_cast<size_t>(viewIdx)].visiblePointIndices;
            visible.reserve(visibleCount);
            for (const VisibilityCacheShard &shard : visibilityShards)
            {
                const auto &localVisible = shard.visiblePointIndicesByView[static_cast<size_t>(viewIdx)];
                visible.insert(visible.end(), localVisible.begin(), localVisible.end());
            }
        }

        for (VisibilityCacheShard &shard : visibilityShards)
        {
            for (size_t idx = 0; idx < m_pairCommonCounts.size(); ++idx)
            {
                m_pairCommonCounts[idx] += shard.pairCommonCounts[idx];
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
            const auto &visible = m_frameCaches[static_cast<size_t>(viewIdx)].visiblePointIndices;
            const size_t viewOffset = static_cast<size_t>(viewIdx) * m_visibilityWordCount;
            for (size_t pointIndex : visible)
            {
                const size_t word = pointIndex / 64;
                const size_t bit = pointIndex % 64;
                m_visibilityBits[viewOffset + word] |= (uint64_t{1} << bit);
            }
        }
    };
    buildVisibilityBitsFromFrameCaches();

    m_frameCachesReady = true;

    auto sampledMedianAngle = [this, NV](int refIdx, int sourceIdx) -> float
    {
        if (refIdx < 0 || refIdx >= NV || sourceIdx < 0 || sourceIdx >= NV)
        {
            return 0.f;
        }

        constexpr size_t kMaxAngleSamples = 2048;
        std::vector<float> angles;
        angles.reserve(kMaxAngleSamples);
        for (size_t pointIndex : m_frameCaches[static_cast<size_t>(refIdx)].visiblePointIndices)
        {
            if (!isSparsePointVisibleInFrame(sourceIdx, pointIndex))
            {
                continue;
            }
            angles.push_back(mvsTriangulationAngleDeg(m_views[refIdx],
                                                       m_views[sourceIdx],
                                                       m_sparse.points[pointIndex]));
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
        const int desiredSourceCount = std::max(1, m_config.numSourceViews);
        std::vector<MvsSourceViewScore> rankedSourceCandidates;
        rankedSourceCandidates.reserve(static_cast<size_t>(std::max(0, NV - 1)));
        for (int sourceIdx = 0; sourceIdx < NV; ++sourceIdx)
        {
            if (sourceIdx == refIdx)
            {
                continue;
            }

            const int common = m_pairCommonCounts[static_cast<size_t>(refIdx) * NV + sourceIdx];
            if (common <= 0)
            {
                continue;
            }

            MvsSourceViewScore candidate;
            candidate.viewIndex = sourceIdx;
            candidate.commonVisiblePoints = common;
            candidate.medianTriangulationAngleDeg = 0.f;
            candidate.score = static_cast<float>(common);
            rankedSourceCandidates.push_back(candidate);
        }

        std::sort(rankedSourceCandidates.begin(),
                  rankedSourceCandidates.end(),
                  [](const MvsSourceViewScore &a, const MvsSourceViewScore &b)
        {
            if (a.commonVisiblePoints != b.commonVisiblePoints)
            {
                return a.commonVisiblePoints > b.commonVisiblePoints;
            }
            return a.viewIndex < b.viewIndex;
        });

        auto compareSourceScores = [](const MvsSourceViewScore &a, const MvsSourceViewScore &b)
        {
            if (a.score != b.score)
            {
                return a.score > b.score;
            }
            if (a.commonVisiblePoints != b.commonVisiblePoints)
            {
                return a.commonVisiblePoints > b.commonVisiblePoints;
            }
            return a.viewIndex < b.viewIndex;
        };

        std::vector<MvsSourceViewScore> scores;
        scores.reserve(rankedSourceCandidates.size());
        for (const MvsSourceViewScore &candidate : rankedSourceCandidates)
        {
            const float currentSourceScoreCutoff =
                scores.size() >= static_cast<size_t>(desiredSourceCount)
                    ? scores[static_cast<size_t>(desiredSourceCount - 1)].score
                    : -std::numeric_limits<float>::infinity();

            if (scores.size() >= static_cast<size_t>(desiredSourceCount)
                && candidate.commonVisiblePoints <= currentSourceScoreCutoff)
            {
                // remaining candidates are sorted by common count; angle weights never raise a score above common.
                break;
            }

            const float medianAngle = sampledMedianAngle(refIdx, candidate.viewIndex);
            const float angleWeight =
                medianAngle < 0.2f ? 0.25f :
                medianAngle > 35.0f ? 0.50f :
                1.0f;
            const float proximityPenalty = 0.001f * static_cast<float>(std::abs(candidate.viewIndex - refIdx));

            MvsSourceViewScore score;
            score.viewIndex = candidate.viewIndex;
            score.commonVisiblePoints = candidate.commonVisiblePoints;
            score.medianTriangulationAngleDeg = medianAngle;
            score.score = static_cast<float>(candidate.commonVisiblePoints) * angleWeight - proximityPenalty;
            scores.push_back(score);
            std::sort(scores.begin(), scores.end(), compareSourceScores);
        }

        std::sort(scores.begin(), scores.end(), compareSourceScores);

        auto &sources = m_frameCaches[static_cast<size_t>(refIdx)].sourceViewIndices;
        sources.reserve(static_cast<size_t>(std::min(NV - 1, desiredSourceCount)));
        for (const auto &score : scores)
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
            sources = nearestMvsSourceViewIndices(NV, refIdx, desiredSourceCount);
        }

        auto &cache = m_frameCaches[static_cast<size_t>(refIdx)];
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

    LOG_INFO(QStringLiteral("[MVS] MVS 可见性缓存完成: views=%1 points=%2 elapsed=%3 ms")
                 .arg(NV)
                 .arg(static_cast<qulonglong>(pointCount))
                 .arg(elapsedMs(start, Clock::now()), 0, 'f', 1));
}

std::vector<int> DepthMapGenerator::sourceViewIndicesForFrame(int refIdx, int maxSources) const
{
    if (m_frameCachesReady
        && refIdx >= 0
        && refIdx < static_cast<int>(m_frameCaches.size())
        && maxSources > 0)
    {
        const auto &cached = m_frameCaches[static_cast<size_t>(refIdx)].sourceViewIndices;
        if (!cached.empty())
        {
            const int count = std::min(maxSources, static_cast<int>(cached.size()));
            return std::vector<int>(cached.begin(), cached.begin() + count);
        }
    }

    return selectMvsSourceViewIndices(m_views, m_sparse, refIdx, maxSources);
}

std::vector<size_t> DepthMapGenerator::visibleSparsePointIndicesForFrame(
    int refIdx,
    const std::vector<int> &sourceIndices,
    int minSourceViews) const
{
    if (!m_frameCachesReady
        || refIdx < 0
        || refIdx >= static_cast<int>(m_frameCaches.size()))
    {
        return collectMvsVisibleSparsePointIndices(m_views, m_sparse, refIdx, sourceIndices, minSourceViews);
    }

    const auto &cache = m_frameCaches[static_cast<size_t>(refIdx)];
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
            if (sourceIdx < 0 || sourceIdx >= static_cast<int>(m_views.size()) || sourceIdx == refIdx)
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
    const int NV = static_cast<int>(m_views.size());
    m_grayCache.resize(NV);
    m_contentMasks.resize(NV);

    const int workerCount = preloadImagesWorkerCount(NV, m_config.cpuWorkerCount);
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
            const int i = nextImage.fetch_add(1);
            if (i >= NV)
            {
                break;
            }

            m_grayCache[i] = cv::imread(m_views[i].imagePath, cv::IMREAD_GRAYSCALE);
            if (m_grayCache[i].empty())
            {
                LOG_WARN(QStringLiteral("[MVS] 警告: 无法读取图像 %1: %2").arg(i).arg(QString::fromStdString(m_views[i].imagePath)));
                continue;
            }

            // ── 在 CLAHE 增强之前计算内容区域掩码 ─────────────────────────
            // 关键：必须在原始图像上计算暗区掩码！
            // gamma(0.4) 会将 gray=3 提升到 37，gray=5 提升到 46，
            // 使得本应被屏蔽的黑边像素通过暗区阈值，
            // 导致 PatchMatch 在无纹理黑边区域生成大量噪声深度。
            {
                float coverage = 0.f;
                double otsuThresh = 0.0;
                int adaptiveThresh = 0;
                m_contentMasks[i] = buildContentMask(m_grayCache[i], &coverage, &otsuThresh, &adaptiveThresh);

                const int totalPx = m_grayCache[i].rows * m_grayCache[i].cols;
                const int contentPx = static_cast<int>(std::round(coverage * totalPx));
                if (m_contentMasks[i].empty())
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
            double imgMean = cv::mean(m_grayCache[i])[0];
            if (imgMean < 80.0)
            {
                cv::Mat enhanced;
                if (imgMean < 30.0)
                {
                    // 极暗图像（航空影像常见）：先做 gamma 校正提升暗区可见度，
                    // 再用高 clipLimit CLAHE 进一步增强局部对比度。
                    // gamma=0.4 将 [0,255] 非线性映射，使暗部细节显现。
                    cv::Mat floatImg;
                    m_grayCache[i].convertTo(floatImg, CV_32F, 1.0 / 255.0);
                    cv::pow(floatImg, 0.4, floatImg);
                    floatImg.convertTo(enhanced, CV_8U, 255.0);
                    auto clahe = cv::createCLAHE(8.0, cv::Size(8, 8));
                    clahe->apply(enhanced, enhanced);
                }
                else
                {
                    // 中等暗度：标准 CLAHE
                    auto clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
                    clahe->apply(m_grayCache[i], enhanced);
                }
                double newMean = cv::mean(enhanced)[0];
                fprintf(stderr, "[MVS] 图像 %d: 低对比度 (mean=%.1f)，应用 CLAHE → mean=%.1f\n",
                        i, imgMean, newMean);
                m_grayCache[i] = enhanced;
            }
            else
            {
                fprintf(stderr, "[MVS] 预加载图像 %d/%d: %dx%d (mean=%.1f)\n",
                        i+1, NV, m_grayCache[i].cols, m_grayCache[i].rows, imgMean);
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

// =============================================================================
void DepthMapGenerator::start()
{
    m_cancelled = false;
    m_depthFrames.clear();
    (void)QtConcurrent::run([this]()
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
    const CameraView &ref = m_views[refIdx];
    PositiveDepthCameraModel cam = ref.positiveDepthModel();

    std::vector<float> depths;
    depths.reserve(visiblePointIndices.size());

    for (size_t pointIndex : visiblePointIndices)
    {
        if (pointIndex >= m_sparse.points.size())
        {
            continue;
        }
        const auto &pt = m_sparse.points[pointIndex];
        float Zc = cam.R_cw[6]*pt[0] + cam.R_cw[7]*pt[1] + cam.R_cw[8]*pt[2] + cam.T[2];
        if (Zc > 0.f)
        {
            depths.push_back(Zc);
        }
    }

    if (depths.size() < 5)
    {
        // 没有足够的稀疏点——使用全局最大相机基线估算深度范围。
        // 关键：使用「全局最大基线」（所有相机对之间的最大距离），
        // 而非 per-camera 最大基线，以保证所有帧使用一致的 zNear/zFar，
        // 防止深度图均值差异过大导致融合一致性检查全部失败。
        float maxBaseline = 0.f;
        const int NVall = static_cast<int>(m_views.size());
        for (int ia = 0; ia < NVall; ++ia)
        {
            PositiveDepthCameraModel ca = m_views[ia].positiveDepthModel();
            for (int ib = ia + 1; ib < NVall; ++ib)
            {
                PositiveDepthCameraModel cb = m_views[ib].positiveDepthModel();
                float dx = ca.C[0]-cb.C[0], dy = ca.C[1]-cb.C[1], dz = ca.C[2]-cb.C[2];
                maxBaseline = std::max(maxBaseline, std::sqrt(dx*dx + dy*dy + dz*dz));
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
            float dx = m_sparse.maxPt[0] - m_sparse.minPt[0];
            float dy = m_sparse.maxPt[1] - m_sparse.minPt[1];
            float dz = m_sparse.maxPt[2] - m_sparse.minPt[2];
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
    zNear = inlierDepths[static_cast<size_t>(ni * 0.02f)] * m_config.zNearScale;
    zFar  = inlierDepths[static_cast<size_t>(ni * 0.98f)] * m_config.zFarScale;
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
    if (refIdx < 0 || refIdx >= static_cast<int>(m_views.size()))
    {
        return cv::Mat();
    }

    return buildHintDepthForCamera(refIdx,
                                   m_views[refIdx].positiveDepthModel(),
                                   W,
                                   H,
                                   visiblePointIndices);
}

cv::Mat DepthMapGenerator::buildHintDepthForCamera(
    int refIdx,
    const PositiveDepthCameraModel &camera,
    int W,
    int H,
    const std::vector<size_t> &visiblePointIndices) const
{
    const std::vector<ProjectedSparseDepthSample> samples =
        collectProjectedSparseDepthSamples(m_sparse, camera, W, H, visiblePointIndices);
    return buildHintDepthFromProjectedSamples(refIdx, W, H, samples);
}

std::vector<ProjectedSparseDepthSample> DepthMapGenerator::collectProjectedSparseDepthSamples(
    const SparseCloud &sparse,
    const PositiveDepthCameraModel &camera,
    int imageWidth,
    int imageHeight,
    const std::vector<size_t> &visiblePointIndices)
{
    std::vector<ProjectedSparseDepthSample> samples;
    if (sparse.points.empty() || imageWidth <= 0 || imageHeight <= 0 || !camera.valid())
    {
        return samples;
    }

    const PositiveDepthCameraModel &cam = camera;
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
        float u = 0.0f;
        float v = 0.0f;
        float depth = 0.0f;
        if (!cam.projectWithDepth(pt[0], pt[1], pt[2], u, v, depth)
            || !std::isfinite(depth)
            || u < 0.0f
            || u >= static_cast<float>(imageWidth)
            || v < 0.0f
            || v >= static_cast<float>(imageHeight))
        {
            continue;
        }

        ProjectedSparseDepthSample candidate;
        candidate.uNorm = u / static_cast<float>(imageWidth);
        candidate.vNorm = v / static_cast<float>(imageHeight);
        candidate.depth = depth;
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

    PositiveDepthCameraModel cam = views[refIdx].positiveDepthModel();
    if (!cam.valid())
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
    if (refIdx < 0 || refIdx >= static_cast<int>(m_views.size()))
    {
        return cv::Mat();
    }

    return buildSparseSupportMaskForCamera(refIdx,
                                           m_views[refIdx].positiveDepthModel(),
                                           W,
                                           H,
                                           visiblePointIndices);
}

cv::Mat DepthMapGenerator::buildSparseSupportMaskForCamera(
    int refIdx,
    const PositiveDepthCameraModel &camera,
    int W,
    int H,
    const std::vector<size_t> &visiblePointIndices) const
{
    if (W <= 0 || H <= 0 || refIdx < 0 || refIdx >= static_cast<int>(m_views.size()) || m_sparse.points.size() < 20)
    {
        return cv::Mat();
    }

    PositiveDepthCameraModel cam = camera;
    if (!cam.valid() || visiblePointIndices.size() < 20)
    {
        return cv::Mat();
    }

    const std::vector<ProjectedSparseDepthSample> samples =
        collectProjectedSparseDepthSamples(m_sparse, cam, W, H, visiblePointIndices);
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
    stats.effectiveConfidenceThreshold = confThresh;

    const bool hasConfidence = !confidenceMap.empty()
        && confidenceMap.size() == depthMap.size()
        && confidenceMap.type() == CV_32F;
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

        if (confThresh > 0.0f)
        {
            cv::Mat beforeConfidence = depthMap.clone();
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

            if (validAfterConfidence < stats.validBeforePostprocess / 20)
            {
                fprintf(stderr, "[MVS] 帧%d 置信度过滤后像素太少，回退为不过滤\n", refIdx);
                depthMap = std::move(beforeConfidence);
                validAfterConfidence = stats.validBeforePostprocess;
            }

            stats.validAfterConfidenceFilter = validAfterConfidence;
            stats.confidenceRemoved = std::max(0, stats.validBeforePostprocess - validAfterConfidence);
        }
    }

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

    stats.validAfterPostprocess = cv::countNonZero(depthMap > 0.0f);
    if (stats.confidenceRemoved > 0 || stats.localDepthOutlierRemoved > 0)
    {
        fprintf(stderr,
                "[MVS] 帧%d 深度后处理: before=%d after_conf=%d conf_removed=%d local_removed=%d after=%d\n",
                refIdx,
                stats.validBeforePostprocess,
                stats.validAfterConfidenceFilter,
                stats.confidenceRemoved,
                stats.localDepthOutlierRemoved,
                stats.validAfterPostprocess);
    }
    return stats;
}

// =============================================================================
DepthFrameResult DepthMapGenerator::computeDepthForView(int refIdx, const DepthGenConfig *configOverride)
{
    DepthFrameResult result;
    result.refViewIdx = refIdx;
    result.imageIndex = refIdx;
    result.success = false;

    const DepthGenConfig &config = configOverride ? *configOverride : m_config;
    FrameTiming timing;
    const auto totalStart = Clock::now();
    auto stageStart = totalStart;

    const CameraView &refView = m_views[refIdx];

    // 使用预加载的灰度图缓存（无磁盘 I/O）
    cv::Mat refImg;
    if (refIdx >= 0 && refIdx < (int)m_grayCache.size() && !m_grayCache[refIdx].empty()) {
        refImg = m_grayCache[refIdx];  // 浅拷贝，零开销
    } else {
        refImg = cv::imread(refView.imagePath, cv::IMREAD_GRAYSCALE);
    }
    if (refImg.empty()) {
        result.errorMsg = "无法读取参考帧图像: " + refView.imagePath;
        return result;
    }
    const int W = refImg.cols, H = refImg.rows;
    PositiveDepthCameraModel refCam = refView.positiveDepthModel();

    // 选择源帧（从缓存中取，省去重复加载）
    const int NV = static_cast<int>(m_views.size());
    int numSrc = std::min(config.numSourceViews, NV - 1);
    std::vector<cv::Mat> srcGrays;
    std::vector<PositiveDepthCameraModel> srcCams;
    std::vector<int> sourceIndices;

    const std::vector<int> selectedSources = sourceViewIndicesForFrame(refIdx, numSrc);
    for (int si : selectedSources)
    {
        if (si < 0 || si >= NV || si == refIdx)
        {
            continue;
        }
        cv::Mat srcImg;
        if (si >= 0 && si < (int)m_grayCache.size() && !m_grayCache[si].empty()) {
            srcImg = m_grayCache[si];
        } else {
            srcImg = cv::imread(m_views[si].imagePath, cv::IMREAD_GRAYSCALE);
        }
        if (srcImg.empty()) continue;
        if (srcImg.cols != W || srcImg.rows != H)
            cv::resize(srcImg, srcImg, cv::Size(W, H));
        srcGrays.push_back(srcImg);
        srcCams.push_back(m_views[si].positiveDepthModel());
        sourceIndices.push_back(si);
        if (static_cast<int>(srcGrays.size()) >= numSrc) break;
    }

    if (srcGrays.empty()) {
        timing.sourceMs = elapsedMs(stageStart, Clock::now());
        result.errorMsg = "没有可用的源帧";
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
    fprintf(stderr, "\n[MVS] 帧 %d: 图像 %dx%d, 源帧 %d 个, det(R)=%.4f\n",
            refIdx, W, H, (int)srcCams.size(), det3(refCam.R_cw));
    fprintf(stderr, "[MVS] 帧 %d: C=[%.3f, %.3f, %.3f]\n",
            refIdx, refCam.C[0], refCam.C[1], refCam.C[2]);

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
    PositiveDepthCameraModel workRefCam = refCam;
    std::vector<PositiveDepthCameraModel> workSrcCams = srcCams;

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

    // =========================================================================
    // ★ 多尺度 PatchMatch（粗到精）
    //   Pass 1: ds=4 粗分辨率 (1006×759)，NCC patch 等效 44×44 原始像素，
    //           覆盖更大空间范围 → 在低纹理/暗场景下也能获得正确初始深度。
    //   Pass 2: ds=2 精细分辨率 (2012×1518)，以粗结果为 hint 精化。
    //           只需较少迭代，因为初始深度已经接近正确值。
    // =========================================================================
    cv::Mat depthMap, confMap;
    std::string errMsg;
    PatchMatchConfig coarseCfg = makeCoarsePatchMatchConfig(pmCfg, useRectified);

    // 提示深度/支撑掩码：按 PatchMatch 实际工作尺寸生成，避免先生成全分辨率再缩小。
    stageStart = Clock::now();
    const std::vector<ProjectedSparseDepthSample> workRefSparseSamples =
        collectProjectedSparseDepthSamples(m_sparse,
                                           workRefCam,
                                           workRefImg.cols,
                                           workRefImg.rows,
                                           visibleSparsePointIndices);
    const cv::Size coarseHintSize = patchMatchWorkSize(workRefImg, coarseCfg);
    cv::Mat coarseHint = buildHintDepthFromProjectedSamples(refIdx,
                                                            coarseHintSize.width,
                                                            coarseHintSize.height,
                                                            workRefSparseSamples);
    const cv::Size supportMaskSize = patchMatchWorkSize(refImg, pmCfg);
    std::vector<ProjectedSparseDepthSample> rectifiedSupportSamples;
    const std::vector<ProjectedSparseDepthSample> *supportSamples = &workRefSparseSamples;
    if (useRectified)
    {
        rectifiedSupportSamples =
            collectProjectedSparseDepthSamples(m_sparse, refCam, W, H, visibleSparsePointIndices);
        supportSamples = &rectifiedSupportSamples;
    }
    cv::Mat sparseSupportMask = buildSparseSupportMaskFromProjectedSamples(refIdx,
                                                                           supportMaskSize.width,
                                                                           supportMaskSize.height,
                                                                           *supportSamples);
    timing.hintMs = elapsedMs(stageStart, Clock::now());

    // ── Pass 1: 粗分辨率 ────────────────────────────────────────────────
    stageStart = Clock::now();
    fprintf(stderr,
            "[MVS] 帧 %d: 粗层 PatchMatch ds=%d iters=%d patch=%d parallelSweep=%d\n",
            refIdx,
            coarseCfg.downsampleFactor,
            coarseCfg.numIterations,
            coarseCfg.patchHalf * 2 + 1,
            coarseCfg.cudaUseParallelSweep ? 1 : 0);

    cv::Mat coarseDepth, coarseConf;
    bool coarseOk = estimatePatchMatchWithAdaptiveCuda(
        "粗层 PatchMatch",
        refIdx,
        workRefImg, workSrcGrays, workRefCam, workSrcCams,
        zNear, zFar, coarseCfg,
        coarseDepth, &coarseConf, &errMsg,
        coarseHint.empty() ? nullptr : &coarseHint);

    if (coarseOk) {
        int coarseValid = cv::countNonZero(coarseDepth > 0);
        fprintf(stderr, "[MVS] 帧 %d: 粗分辨率(ds=%d) 完成, 有效像素=%d/%d (%.1f%%)\n",
                refIdx, coarseCfg.downsampleFactor, coarseValid, W*H,
                100.f * coarseValid / (W*H));

        // 合并 hint：粗层结果 + 精层稀疏点（稀疏点更精确，优先保留）。
        // fineHint 直接保持在精层工作尺寸，PatchMatch 入口可直接复用。
        PatchMatchConfig fineCfg = makeFinePatchMatchConfig(pmCfg, useRectified, 0.0f);
        const cv::Size fineHintSize = patchMatchWorkSize(workRefImg, fineCfg);
        cv::Mat fineHint;
        cv::resize(coarseDepth, fineHint, fineHintSize, 0, 0, cv::INTER_NEAREST);
        cv::Mat fineSparseSeedHint = buildSparseSeedDepthFromProjectedSamples(refIdx,
                                                                              fineHintSize.width,
                                                                              fineHintSize.height,
                                                                              workRefSparseSamples);
        if (!fineSparseSeedHint.empty())
        {
            fineSparseSeedHint.copyTo(fineHint, fineSparseSeedHint > 0);
        }

        // ── Pass 2: 精细分辨率 ──────────────────────────────────────
        const int hintValid = cv::countNonZero(fineHint > 0);
        const float hintCoverage =
            static_cast<float>(hintValid) / std::max(1, fineHint.rows * fineHint.cols);
        fineCfg = makeFinePatchMatchConfig(pmCfg, useRectified, hintCoverage);
        fprintf(stderr,
                "[MVS] 帧 %d: 精细层 PatchMatch ds=%d iters=%d patch=%d hintCoverage=%.1f%% parallelSweep=%d\n",
                refIdx,
                fineCfg.downsampleFactor,
                fineCfg.numIterations,
                fineCfg.patchHalf * 2 + 1,
                hintCoverage * 100.0f,
                fineCfg.cudaUseParallelSweep ? 1 : 0);

        bool fineOk = estimatePatchMatchWithAdaptiveCuda(
            "精细层 PatchMatch",
            refIdx,
            workRefImg, workSrcGrays, workRefCam, workSrcCams,
            zNear, zFar, fineCfg,
            depthMap, &confMap, &errMsg,
            &fineHint);

        if (!fineOk) {
            depthMap = coarseDepth;
            confMap  = coarseConf;
            fprintf(stderr, "[MVS] 帧 %d: 精细分辨率失败，回退粗分辨率\n", refIdx);
        }
    } else {
        fprintf(stderr, "[MVS] 帧 %d: 粗分辨率失败，回退单尺度\n", refIdx);
        PatchMatchConfig fallbackCfg = pmCfg;
        fallbackCfg.epipolarRectified = useRectified;
        bool ok = estimatePatchMatchWithAdaptiveCuda(
            "单尺度 PatchMatch",
            refIdx,
            workRefImg, workSrcGrays, workRefCam, workSrcCams,
            zNear, zFar, fallbackCfg,
            depthMap, &confMap, &errMsg,
            coarseHint.empty() ? nullptr : &coarseHint);
        if (!ok) {
            result.errorMsg = errMsg;
            return result;
        }
    }

    // ── 极线校正反变换：将校正空间的深度图映射回原始图像空间 ──────────────
    if (useRectified && !depthMap.empty())
    {
        depthMap = mvs::EpipolarRectifier::unrectifyDepth(
            depthMap, rectPair, W, H);
        if (!confMap.empty())
            confMap = mvs::EpipolarRectifier::unrectifyDepth(
                confMap, rectPair, W, H);
        fprintf(stderr, "[MVS] 帧 %d: 深度图已从校正空间映射回原始空间\n", refIdx);
    }
    timing.patchmatchMs = elapsedMs(stageStart, Clock::now());

    // ── 暗区掩码：使用 preloadImages() 中基于原始图像计算的内容掩码 ──────
    // 必须使用 CLAHE 增强前的掩码！gamma(0.4) 将黑边 gray=3 提升到 37,
    // 使得基于增强后图像的暗区阈值失效，导致 PatchMatch 在无纹理黑边
    // 区域产生大量噪声深度（如 44% 有效像素 vs 实际仅 10% 内容区域）。
    stageStart = Clock::now();
    {
        cv::Mat brightMask;
        const bool hasPreloadedMaskSlot = refIdx >= 0 && refIdx < static_cast<int>(m_contentMasks.size());
        if (hasPreloadedMaskSlot && !m_contentMasks[refIdx].empty())
        {
            brightMask = m_contentMasks[refIdx];
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
            brightMask = buildContentMask(refImg, &coverage, &otsuThresh, &adaptiveThresh);
            if (brightMask.empty())
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

        if (!brightMask.empty())
        {
            // 适配深度图尺寸（PatchMatch 可能有上采样）
            if (brightMask.size() != depthMap.size())
            {
                cv::resize(brightMask, brightMask, depthMap.size(), 0, 0, cv::INTER_NEAREST);
            }

            int beforeMask = cv::countNonZero(depthMap > 0);
            depthMap.setTo(0, ~brightMask);
            if (!confMap.empty())
            {
                confMap.setTo(0, ~brightMask);
            }
            int afterMask = cv::countNonZero(depthMap > 0);

            if (afterMask < beforeMask)
            {
                fprintf(stderr, "[MVS] 帧 %d: 内容掩码过滤 %d→%d 有效像素\n",
                        refIdx, beforeMask, afterMask);
            }
        }
    }

    if (!sparseSupportMask.empty())
    {
        cv::Mat supportMask = sparseSupportMask;
        if (supportMask.size() != depthMap.size())
        {
            cv::resize(supportMask, supportMask, depthMap.size(), 0, 0, cv::INTER_NEAREST);
        }

        applySparseSupportPrior(depthMap, confMap, supportMask, refIdx);
    }

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
    result.success    = true;
    return result;
}

// =============================================================================
FusionFrameInput DepthMapGenerator::buildFusionFrame(const DepthFrameResult &res) const
{
    FusionFrameInput frame;
    frame.cameraModel = m_views[res.refViewIdx].positiveDepthModel();
    frame.imgW = res.depthMap ? res.depthMap->cols : 0;
    frame.imgH = res.depthMap ? res.depthMap->rows : 0;
    frame.imagePath = m_views[res.refViewIdx].imagePath;

    if (!res.depthMap || res.depthMap->empty()) return frame;

    const cv::Mat &rawDepth = *res.depthMap;
    cv::Mat filteredDepth = rawDepth.clone();
    cv::Mat filteredConfidence = res.confidence ? res.confidence->clone() : cv::Mat();

    frame.depthPostprocess = postprocessFusionDepthMap(filteredDepth,
                                                       filteredConfidence,
                                                       m_config.fusion,
                                                       res.imageIndex,
                                                       static_cast<int>(m_views.size()));

    frame.depthMap   = filteredDepth;
    frame.confidence = filteredConfidence;
    return frame;
}

// =============================================================================
// 双视图深度图左右一致性检查
// 对每个深度像素：反投影到 3D → 投影到另一视图 → 比较另一视图的深度值
//
// 策略:
//   多视图 (≥3): "需要确认" — 像素必须得到至少一个其他视图的深度一致性确认
//   少视图 (≤2): "仅移除矛盾" — 仅移除被其他视图明确否定的像素；
//                 对方深度为 0 或超出投影范围时，保留原像素（疑罪从无）
// =============================================================================
void DepthMapGenerator::crossCheckDepthConsistency()
{
    const int NV = static_cast<int>(m_views.size());
    if (NV < 2) return;

    // 相对深度误差阈值：少视图放宽至 25%，多视图 5%
    const float relThresh = (NV <= 2) ? 0.25f : 0.05f;
    const bool  fewViews  = (NV <= 2);

    // ── 先保存所有帧的原始深度图拷贝，避免顺序处理的级联清除问题 ──────
    std::vector<cv::Mat> origDepths(NV);
    for (int i = 0; i < NV; ++i)
    {
        if (m_depthFrames[i].success && m_depthFrames[i].depthMap)
        {
            origDepths[i] = m_depthFrames[i].depthMap->clone();
        }
    }

    for (int i = 0; i < NV; ++i)
    {
        if (!m_depthFrames[i].success || !m_depthFrames[i].depthMap)
        {
            continue;
        }
        cv::Mat &depthI = *m_depthFrames[i].depthMap;
        PositiveDepthCameraModel camI = m_views[i].positiveDepthModel();

        // 备份，万一过滤太激进需要回退
        cv::Mat depthBackup = depthI.clone();

        // ─── 少视图模式: 初始化为全部保留, 仅标记"被明确否定"的像素 ────
        // ─── 多视图模式: 初始化为全部移除, 需要"被确认"才保留 ──────────
        cv::Mat consistentMask(depthI.size(), CV_8U,
                               fewViews ? cv::Scalar(255) : cv::Scalar(0));
        const int rowWorkers = std::max(1, m_config.cpuWorkerCount);
        const std::vector<int> consistencySources =
            consistencySourceIndicesForFrame(m_depthFrames, i, NV);

        for (int j : consistencySources)
        {
            if (j == i)
            {
                continue;
            }
            if (origDepths[j].empty())
            {
                continue;
            }
            const cv::Mat &depthJ = origDepths[j];
            PositiveDepthCameraModel camJ = m_views[j].positiveDepthModel();

            parallelForRows(depthI.rows, rowWorkers, [&](int v)
            {
                const float *pdi = depthI.ptr<float>(v);
                uint8_t     *pMask = consistentMask.ptr<uint8_t>(v);
                for (int u = 0; u < depthI.cols; ++u)
                {
                    float di = pdi[u];
                    if (di <= 0.f)
                    {
                        continue;
                    }

                    if (fewViews)
                    {
                        // 少视图: 只在被明确否定时移除
                        if (pMask[u] == 0)
                        {
                            continue; // 已被其他视图否定
                        }
                    }
                    else
                    {
                        // 多视图: 已确认的跳过
                        if (pMask[u])
                        {
                            continue;
                        }
                    }

                    // 反投影到世界坐标
                    float Xw, Yw, Zw;
                    camI.unproject(static_cast<float>(u), static_cast<float>(v), di, Xw, Yw, Zw);

                    // 投影到帧 j
                    float uj, vj;
                    if (!camJ.project(Xw, Yw, Zw, uj, vj))
                    {
                        continue;
                    }
                    int iuj = static_cast<int>(std::round(uj));
                    int ivj = static_cast<int>(std::round(vj));
                    if (iuj < 0 || iuj >= depthJ.cols || ivj < 0 || ivj >= depthJ.rows)
                    {
                        continue;
                    }

                    float dj = depthJ.at<float>(ivj, iuj);
                    if (dj <= 0.f)
                    {
                        continue; // 对方无深度 → 少视图保留，多视图无变化
                    }

                    // 计算预期深度与实测深度的相对误差
                    float Zc_j = camJ.R_cw[6]*Xw + camJ.R_cw[7]*Yw + camJ.R_cw[8]*Zw + camJ.T[2];
                    if (Zc_j <= 0.f)
                    {
                        continue;
                    }

                    float relErr = std::fabs(dj - Zc_j) / Zc_j;

                    if (fewViews)
                    {
                        // 少视图: 明确矛盾 → 标记移除
                        if (relErr >= relThresh)
                        {
                            pMask[u] = 0;
                        }
                    }
                    else
                    {
                        // 多视图: 一致 → 标记保留
                        if (relErr < relThresh)
                        {
                            pMask[u] = 255;
                        }
                    }
                }
            });
        }

        // 剔除不一致像素
        int beforeValid = cv::countNonZero(depthI > 0);
        depthI.setTo(0, consistentMask == 0);
        int afterValid = cv::countNonZero(depthI > 0);
        float keepRate = beforeValid > 0 ? 100.f * afterValid / beforeValid : 0.f;
        fprintf(stderr, "[MVS] 帧%d 一致性检查(%s, 源视图 %zu/%d): %d→%d 有效像素 (保留 %.1f%%)\n",
                i, fewViews ? "仅移除矛盾" : "需要确认",
                consistencySources.size(), std::max(0, NV - 1),
                beforeValid, afterValid, keepRate);

        // 安全回退：如果保留率过低（< 10%），回退使用原始深度图
        if (afterValid < beforeValid / 10 && beforeValid > 100) 
        {
            fprintf(stderr, "[MVS] 帧%d 一致性过滤后像素太少 (%.1f%%)，回退为原始深度图\n",
                    i, keepRate);
            depthBackup.copyTo(depthI);
        }
    }
}

bool DepthMapGenerator::saveDepthFrameArtifacts(int frameIndex,
                                                const DepthFrameResult &result,
                                                const QString &stageLabel)
{
    if (frameIndex < 0 ||
        frameIndex >= static_cast<int>(m_views.size()) ||
        !result.success ||
        !result.depthMap ||
        result.depthMap->empty())
    {
        return true;
    }

    const bool savePreviewPng = !m_outputDir.empty();
    const bool saveRawDepth = m_config.saveIntermediateDepthMaps && !m_config.intermediateDir.empty();
    if (!savePreviewPng && !saveRawDepth)
    {
        return true;
    }

    std::string saveErr;
    bool previewSaved = !savePreviewPng;
    if (savePreviewPng)
    {
        const std::string pngPath = m_outputDir + "/depth_" + std::to_string(frameIndex) + ".png";
        if (!saveDepthPreviewPng(pngPath, *result.depthMap, &saveErr))
        {
            LOG_WARN(QStringLiteral("[MVS] 保存%1深度预览失败: %2")
                         .arg(stageLabel, QString::fromStdString(saveErr)));
            emit errorOccurred(QString::fromStdString(saveErr));
            return false;
        }

        previewSaved = true;
        LOG_INFO(QStringLiteral("[MVS] 帧 %1 %2深度预览已保存: %3 (%4x%5)")
                     .arg(frameIndex)
                     .arg(stageLabel)
                     .arg(QString::fromStdString(pngPath))
                     .arg(result.depthMap->cols)
                     .arg(result.depthMap->rows));
    }

    bool rawSaved = true;
    if (saveRawDepth)
    {
        const std::string depthPath =
            m_config.intermediateDir + "/depth_" + std::to_string(frameIndex) + ".yml.gz";
        if (!writeCvMatStorage(depthPath, *result.depthMap, &saveErr))
        {
            rawSaved = false;
            LOG_WARN(QStringLiteral("[MVS] 保存%1原始深度失败: %2")
                         .arg(stageLabel, QString::fromStdString(saveErr)));
            emit errorOccurred(QString::fromStdString(saveErr));
        }
    }

    if (saveRawDepth && result.confidence && !result.confidence->empty())
    {
        const std::string confPath =
            m_config.intermediateDir + "/depth_" + std::to_string(frameIndex) + "_conf.yml.gz";
        if (!writeCvMatStorage(confPath, *result.confidence, &saveErr))
        {
            LOG_WARN(QStringLiteral("[MVS] 保存%1置信图失败: %2")
                         .arg(stageLabel, QString::fromStdString(saveErr)));
        }
    }

    if (previewSaved && rawSaved && savePreviewPng)
    {
        const std::string pngPath = m_outputDir + "/depth_" + std::to_string(frameIndex) + ".png";
        emit depthMapSaved(
            QString::fromStdString(pngPath),
            result.depthMap->cols,
            result.depthMap->rows,
            QString::fromStdString(m_views[frameIndex].imagePath));
    }

    return previewSaved && rawSaved;
}

// =============================================================================
void DepthMapGenerator::runInBackground()
{
    bool allOk = true;
    const int NV = static_cast<int>(m_views.size());

    if (!m_config.runDepthEstimation)
    {
        emit errorOccurred(QStringLiteral("当前生成器配置未启用深度估计阶段"));
        emit finished(false);
        return;
    }

    // ── 预加载所有图像（一次性磁盘 I/O，后续全从内存读取）────────────────────
    emit progressChanged(QString("预加载 %1 张图像...").arg(NV), 0.f);
    preloadImages();
    emit progressChanged(QStringLiteral("预计算 MVS 可见性..."), 0.02f);
    prepareFrameCaches();
    m_depthFrames.resize(NV);

    // ── 阶段一：优先级队列并行估计深度图（GPU 优先高价值帧，CPU 处理其余帧）────
    int skippedFrames = 0;
    for (size_t i = 0; i < m_skipFrameMask.size(); ++i)
    {
        if (m_skipFrameMask[i] != 0)
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

    const bool cudaAvailable = m_config.patchMatch.useCuda && PatchMatchDepthEstimator::isCudaAvailable();
    const int cpuThreadCount = std::max(1, m_config.cpuWorkerCount);

    std::vector<FramePriority> framePriorities;
    framePriorities.reserve(static_cast<size_t>(NV));
    for (int i = 0; i < NV; ++i)
    {
        if (i >= 0 && i < static_cast<int>(m_skipFrameMask.size()) && m_skipFrameMask[static_cast<size_t>(i)] != 0)
        {
            continue;
        }

        float resolutionScore = 0.f;
        if (i >= 0 && i < static_cast<int>(m_grayCache.size()) && !m_grayCache[i].empty())
        {
            resolutionScore = static_cast<float>(m_grayCache[i].cols * m_grayCache[i].rows);
        }

        float contentRatio = 0.5f;
        if (i >= 0 && i < static_cast<int>(m_contentMasks.size()) && !m_contentMasks[i].empty())
        {
            const int totalPx = m_contentMasks[i].rows * m_contentMasks[i].cols;
            if (totalPx > 0)
            {
                contentRatio = static_cast<float>(cv::countNonZero(m_contentMasks[i])) / static_cast<float>(totalPx);
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
        ? std::clamp(std::max(1, m_config.gpuFrameWorkerCount), 1, maxFrameWorkers)
        : 0;
    const int cpuFrameWorkers = cudaAvailable
        ? std::clamp(std::max(0, m_config.cpuFrameWorkerCount), 0, maxFrameWorkers)
        : std::clamp(std::max(1, m_config.cpuFrameWorkerCount), 1, maxFrameWorkers);

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

    DepthFrameArtifactSaveQueue saveQueue([this](
                                              int frameIndex,
                                              const DepthFrameResult &result,
                                              const QString &stageLabel)
    {
        return saveDepthFrameArtifacts(frameIndex, result, stageLabel);
    });

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

    auto workerFunc = [this, NV, &completedTasks, &activeGpuTasks, &activeCpuTasks, &anyFailure,
                       &taskMutex, &gpuQueue, &cpuQueue, &emitDepthProgress, &saveQueue](bool useGpu) {
        DepthGenConfig workerConfig = m_config;
        workerConfig.patchMatch.useCuda = useGpu;
        const QString workerTag = useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU");

        while (!m_cancelled)
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

            DepthFrameResult res = computeDepthForView(i, &workerConfig);
            m_depthFrames[i] = res;
            const auto frameEnd = std::chrono::steady_clock::now();
            const double elapsedMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

            if (!res.success)
            {
                LOG_WARN(QStringLiteral("[MVS] 帧 %1 深度估计失败: device=%2 elapsed=%3 ms error=%4")
                             .arg(i)
                             .arg(useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU"))
                             .arg(elapsedMs, 0, 'f', 1)
                             .arg(QString::fromStdString(res.errorMsg)));
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
            }

            emit depthMapReady(res);
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

    saveQueue.waitUntilIdle();
    if (saveQueue.failed())
    {
        anyFailure = true;
    }

    // 释放图像缓存（深度估计完毕后不再需要）
    m_grayCache.clear();
    m_grayCache.shrink_to_fit();
    m_contentMasks.clear();
    m_contentMasks.shrink_to_fit();
    clearFrameCaches();

    if (m_cancelled) {
        saveQueue.stop();
        emit finished(false);
        return;
    }

    // ── 阶段 1.5：双视图深度图左右一致性检查 ────────────────────────────────
    // 在融合前剔除 PatchMatch 产生的幽灵深度（两视图深度互不一致的像素）
    if (NV >= 2) {
        emit progressChanged("深度一致性检查...", static_cast<float>(NV) / (NV + 3));
        crossCheckDepthConsistency();
    }

    const bool savePreviewPng = !m_outputDir.empty();
    const bool saveRawDepth = m_config.saveIntermediateDepthMaps && !m_config.intermediateDir.empty();
    if (savePreviewPng || saveRawDepth)
    {
        for (int i = 0; i < NV; ++i)
        {
            const DepthFrameResult &res = m_depthFrames[i];
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

    if (!m_config.runFusion)
    {
        emit progressChanged("完成", 1.f);
        emit finished(allOk);
        return;
    }

    // ── 阶段二：COLMAP BFS 深度图融合 → 直接输出 3D 点 ──────────────────────
    emit progressChanged("深度图融合...", static_cast<float>(NV) / (NV + 2));

    std::vector<FusionFrameInput> frames;
    for (const auto &fr : m_depthFrames) {
        if (fr.success && fr.depthMap && !fr.depthMap->empty())
            frames.push_back(buildFusionFrame(fr));
    }

    if (frames.empty()) {
        emit errorOccurred("没有有效的深度帧，融合失败");
        emit finished(false);
        return;
    }

    fprintf(stderr, "[MVS] COLMAP BFS 融合: %d 帧参与\n", (int)frames.size());

    // 使用新的 StereoFusionConfig
    StereoFusionConfig fusionCfg;
    fusionCfg.minNumPixels   = m_config.fusion.minConsistentViews;
    fusionCfg.maxReprojError = m_config.fusion.pixelThresh;
    fusionCfg.maxDepthError  = m_config.fusion.relDepthThresh;
    fusionCfg.checkNumImages = std::min(50, NV);
    fusionCfg.workerCount    = std::max(1, m_config.cpuWorkerCount);

    // 少视图场景（≤2张）：crossCheck 已保证深度一致性，
    // 融合只需 1 个观测即可通过（避免 BFS 因级联过滤找不到第二观测而全部拒绝）
    if (NV <= 2) {
        fusionCfg.minNumPixels   = 1;   // 单观测即可，crossCheck 已保证质量
        fusionCfg.maxDepthError  = std::max(fusionCfg.maxDepthError, 0.08f); // 放宽至 8%
        fusionCfg.maxReprojError = std::max(fusionCfg.maxReprojError, 3.0f); // 3 像素重投影误差
        fprintf(stderr, "[MVS] 少视图模式 (%d 帧)，minNumPixels=1, 放宽融合阈值\n", NV);
    }

    // 如果有稀疏云 AABB，设置包围盒过滤离群点
    // 但必须检查稀疏云与相机是否在同一坐标系（防止 GPS 稀疏云 vs SFM 相机失配）
    if (m_sparse.points.size() > 10) {
        // 计算相机中心的最大分量，作为坐标系尺度参考
        float camExtent = 1.0f;
        for (const auto &v : m_views) {
            PositiveDepthCameraModel vc = v.positiveDepthModel();
            for (int k = 0; k < 3; ++k)
                camExtent = std::max(camExtent, std::fabs(vc.C[k]));
        }
        // 稀疏云 AABB 最大分量
        float cloudExtent = 0.0f;
        for (int k = 0; k < 3; ++k)
            cloudExtent = std::max(cloudExtent,
                std::max(std::fabs(m_sparse.maxPt[k]), std::fabs(m_sparse.minPt[k])));
        // 坐标系兼容判断：尺度差不超过 50 倍才启用包围盒
        const float scaleRatio = cloudExtent / std::max(camExtent, 1.0f);
        if (scaleRatio < 50.0f)
        {
            fusionCfg.useBoundingBox = true;
            float pad = (NV <= 2) ? 0.5f : 0.2f;
            for (int k = 0; k < 3; ++k)
            {
                float range = m_sparse.maxPt[k] - m_sparse.minPt[k];
                fusionCfg.bboxMin[k] = m_sparse.minPt[k] - range * pad;
                fusionCfg.bboxMax[k] = m_sparse.maxPt[k] + range * pad;
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
    std::vector<DensePoint> cloud;
    std::string fuseErr;

    bool fuseOk = fusion.fuse(frames, cloud,
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

    // 保存每帧一致性过滤深度图（加锁，防止 GUI 线程并发读取）
    {
        std::lock_guard<std::mutex> lock(m_filteredDepthsMutex);
        m_filteredDepths = fusion.filteredDepths();
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
                m_config.patchMatch.useCuda ? plapoint::ProcessingDevice::Auto : plapoint::ProcessingDevice::CPU;

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

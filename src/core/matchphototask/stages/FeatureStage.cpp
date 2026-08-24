#include "FeatureStage.h"

#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosRuntime.h"
#include "FeaturePreparationQueue.h"
#include "io/PathIO.h"
#include "ImageMatchingRegistry.h"
#include "concurrency/SafeWorkerGroup.h"
#include "loma_r/LoMaRAlgorithm.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftComputeBackend.h"

#include <QElapsedTimer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace xjw::matchphotos
{
    namespace
    {

        MatchPhotosStageReport
        makeFeatureReport(MatchPhotosStageStatus status, const QString& message, int itemCount = 0)
        {
            MatchPhotosStageReport report;
            report.stageId = QStringLiteral("feature");
            report.displayName = QStringLiteral("特征提取");
            report.status = status;
            report.message = message;
            report.itemCount = itemCount;
            return report;
        }

        cv::Mat resizeImage(const cv::Mat& image, int maximumDimension, double* scale)
        {
            if (scale)
            {
                *scale = 1.0;
            }
            if (image.empty() || maximumDimension <= 0)
            {
                return image;
            }

            const int maximumSide = std::max(image.cols, image.rows);
            if (maximumSide <= maximumDimension)
            {
                return image;
            }

            const double resizeScale = static_cast<double>(maximumDimension) / static_cast<double>(maximumSide);
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(), resizeScale, resizeScale, cv::INTER_AREA);
            if (scale)
            {
                *scale = resizeScale;
            }
            return resized;
        }

    } // namespace

    MatchPhotosStageReport FeatureStage::run(const MatchPhotosContext& context,
                                             const MatchPhotosOptions& options,
                                             const MatchPhotosAlgorithmPlan& algorithmPlan,
                                             std::vector<MatchPhotosFeatureRecord>* featureRecords) const
    {
        if (options.planOnly)
        {
            return makeFeatureReport(MatchPhotosStageStatus::Skipped, QStringLiteral("plan-only 模式，跳过特征提取"));
        }
        if (!context.featureCache)
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("内部错误：没有创建任务级特征缓存"));
        }
        const QStringList images = context.pairInput.images;
        if (images.isEmpty())
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed, QStringLiteral("没有可用于提取特征的影像"));
        }

        ResolvedLoMaRTensorRtPackage loma_package;
        if (algorithmPlan.algorithmId == QLatin1String(image_matching::kLoMaRAlgorithmId))
        {
            reportMatchPhotosProgress(context,
                                      QStringLiteral("model_prepare"),
                                      QStringLiteral("正在检查 LoMa-R ONNX 和本机 TensorRT engine 缓存"),
                                      0,
                                      12);
            const ModelPreparationProgressCallback progress_callback =
                [&context](const QString& message, int current, int maximum)
            { reportMatchPhotosProgress(context, QStringLiteral("model_prepare"), message, current, maximum); };
            loma_package = resolveLoMaRTensorRtPackage(options, algorithmPlan.maxKeypoints, true, progress_callback);
            if (!loma_package.isValid())
            {
                return makeFeatureReport(
                    MatchPhotosStageStatus::Failed,
                    QStringLiteral("LoMa-R TensorRT 模型包不可用：%1").arg(loma_package.errorMessage));
            }
            reportMatchPhotosProgress(
                context,
                QStringLiteral("model_prepare"),
                QStringLiteral("LoMa-R 本机 TensorRT engine 已就绪：%1").arg(loma_package.environmentSummary),
                12,
                12);
        }

        const bool applyMask = shouldApplyMasksToKeypoints(options);
        const bool autoSift = algorithmPlan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId);
        const int prefetchDepth = std::clamp(options.featurePrefetchDepth, 1, 4);
        const unsigned int hardwareThreads = std::thread::hardware_concurrency();
        const int prepareWorkerBudget =
            hardwareThreads == 0 ? 2 : std::clamp(static_cast<int>((hardwareThreads + 3U) / 4U), 1, 4);
        const int prepareWorkerCount = std::min(prefetchDepth, prepareWorkerBudget);
        const int extractionWorkerCount =
            autoSift && algorithmPlan.executionBackend == image_matching::SiftComputeBackend::Cuda
                ? std::min(prefetchDepth, 2)
                : 1;
        const int totalImages = images.size();
        std::vector<FeaturePreparationRequest> requests;
        requests.reserve(static_cast<std::size_t>(totalImages));
        for (int index = 0; index < totalImages; ++index)
        {
            FeaturePreparationRequest request;
            request.index = index;
            request.imagePath = images.at(index);
            requests.push_back(std::move(request));
        }

        reportMatchPhotosProgress(context,
                                  QStringLiteral("feature"),
                                  QStringLiteral("%1：准备处理 %2 张影像%3，CPU 预取 %4 worker，提取流水线 %5 worker")
                                      .arg(algorithmPlan.displayName)
                                      .arg(totalImages)
                                      .arg(applyMask ? QStringLiteral("，蒙版约束关键点") : QString())
                                      .arg(prepareWorkerCount)
                                      .arg(extractionWorkerCount),
                                  0,
                                  totalImages);

        FeaturePreparationQueue queue(
            std::move(requests),
            prefetchDepth,
            context.cancelFlag,
            [&](const FeaturePreparationRequest& request)
            {
                PreparedFeatureImage prepared;
                prepared.index = request.index;
                prepared.imagePath = request.imagePath;
                QElapsedTimer totalTimer;
                QElapsedTimer phaseTimer;
                totalTimer.start();

                phaseTimer.start();
                const int readMode = algorithmPlan.requiresColorInput ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
                const cv::Mat original = common::io::readImage(request.imagePath, readMode);
                prepared.imageReadMs = phaseTimer.elapsed();
                if (original.empty())
                {
                    prepared.errorMessage = QStringLiteral("无法读取影像：%1").arg(request.imagePath);
                    return prepared;
                }

                prepared.originalWidth = original.cols;
                prepared.originalHeight = original.rows;
                prepared.effectiveKeypointLimit =
                    resolveFeatureKeypointLimit(options, algorithmPlan, original.cols, original.rows);

                phaseTimer.restart();
                const bool adaptiveSift =
                    algorithmPlan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId);
                prepared.inputImage =
                    adaptiveSift ? original : resizeImage(original, algorithmPlan.maxImageDim, &prepared.resizeScale);
                prepared.imageResizeMs = phaseTimer.elapsed();

                if (applyMask)
                {
                    phaseTimer.restart();
                    prepared.maskPath = maskPathForImage(context, request.imagePath);
                    const cv::Mat exclusionMask = loadMaskForImage(context, request.imagePath, original.size());
                    prepared.mask = makeExtractorValidMask(exclusionMask, prepared.inputImage.size(), options);
                    prepared.maskReadMs = phaseTimer.elapsed();
                }
                prepared.preparationMs = totalTimer.elapsed();
                return prepared;
            },
            prepareWorkerCount);

        std::atomic_int extractedCount{0};
        std::atomic<std::int64_t> totalQueueWaitMs{0};
        std::atomic<std::int64_t> totalPreparationMs{0};
        std::atomic<std::int64_t> totalExtractionMs{0};
        std::vector<MatchPhotosFeatureRecord> orderedRecords(static_cast<std::size_t>(totalImages));
        std::vector<std::uint8_t> populatedRecords(static_cast<std::size_t>(totalImages), 0U);
        std::mutex progressMutex;

        try
        {
            xjw::common::concurrency::runWorkerGroup(
                static_cast<std::size_t>(extractionWorkerCount),
                [&](std::stop_token stopToken)
                {
                    struct WorkspaceRelease
                    {
                        ~WorkspaceRelease()
                        {
                            image_matching::releaseSiftGpuThreadWorkspaces();
                        }
                    } workspaceRelease;

                    // 算法对象及其 TensorRT context 只在所属 worker 内使用。CUDA SIFT
                    // 的原始 GPU 段由后端按设备加锁，CPU 筛选和 RootSIFT 归一化仍可
                    // 与另一个 worker 的 GPU 提取重叠。
                    std::unordered_map<int, std::unique_ptr<image_matching::IImageMatchingAlgorithm>> algorithms;
                    while (!stopToken.stop_requested() && !shouldCancelMatchPhotos(context))
                    {
                        PreparedFeatureImage prepared;
                        QElapsedTimer queueWaitTimer;
                        queueWaitTimer.start();
                        if (!queue.take(&prepared))
                        {
                            break;
                        }
                        const std::int64_t queueWaitMs = queueWaitTimer.elapsed();
                        totalQueueWaitMs.fetch_add(queueWaitMs, std::memory_order_relaxed);
                        if (!prepared.errorMessage.isEmpty())
                        {
                            throw std::runtime_error(prepared.errorMessage.toStdString());
                        }

                        image_matching::ImageMatchingRuntimeConfig runtime;
                        runtime.cudaDevice = options.cudaDevice;
                        runtime.maxKeypoints = prepared.effectiveKeypointLimit;
                        runtime.removeBorders = algorithmPlan.featureRemoveBorders;
                        runtime.maxImageDimension = algorithmPlan.maxImageDim;
                        runtime.siftDetectionThreshold = algorithmPlan.siftDetectionThreshold;
                        runtime.siftContrastThreshold = algorithmPlan.siftContrastThreshold;
                        runtime.siftBackend =
                            autoSift ? algorithmPlan.executionBackend : image_matching::SiftComputeBackend::Cuda;
                        runtime.adaptiveSift = autoSift;
                        runtime.lowTextureRecovery = algorithmPlan.lowTextureRecovery;
                        runtime.rootSift = autoSift;
                        runtime.siftMaximumRatio = options.siftMaximumRatio;
                        runtime.siftMinimumAdaptiveRatio = options.siftMinimumAdaptiveRatio;
                        runtime.adaptiveSiftRatio = options.adaptiveSiftRatio;
                        if (algorithmPlan.algorithmId == QLatin1String(image_matching::kLoMaRAlgorithmId))
                        {
                            runtime.tensorRtFeatureEnginePath = loma_package.featureEnginePath;
                            runtime.tensorRtMatcherEnginePath = loma_package.matcherEnginePath;
                            runtime.modelInputWidth = loma_package.inputWidth;
                            runtime.modelInputHeight = loma_package.inputHeight;
                            runtime.maxMatcherKeypoints = loma_package.keypointCount;
                            runtime.featureKeypointCount = loma_package.featureKeypointCount;
                            runtime.descriptorDimension = loma_package.descriptorDimension;
                        }

                        image_matching::ImageFeatureInput input;
                        input.imagePath = prepared.imagePath;
                        if (algorithmPlan.requiresColorInput)
                        {
                            input.colorImage = prepared.inputImage;
                        }
                        else
                        {
                            input.grayImage = prepared.inputImage;
                        }
                        input.validMask = prepared.mask;
                        input.originalWidth = prepared.originalWidth;
                        input.originalHeight = prepared.originalHeight;
                        input.coordinateScale = prepared.resizeScale > 0.0 ? 1.0 / prepared.resizeScale : 1.0;

                        QElapsedTimer extractTimer;
                        extractTimer.start();
                        auto algorithmIt = algorithms.find(prepared.effectiveKeypointLimit);
                        if (algorithmIt == algorithms.end())
                        {
                            QString createError;
                            auto algorithm = image_matching::ImageMatchingRegistry::create(
                                algorithmPlan.algorithmId, runtime, &createError);
                            if (!algorithm)
                            {
                                throw std::runtime_error(createError.toStdString());
                            }
                            algorithmIt =
                                algorithms.emplace(prepared.effectiveKeypointLimit, std::move(algorithm)).first;
                        }
                        auto features =
                            std::make_shared<image_matching::FeatureSet>(algorithmIt->second->extract(input));
                        if (!features->isConsistent() || features->empty())
                        {
                            throw std::runtime_error(QStringLiteral("影像没有生成有效 %1 特征：%2")
                                                         .arg(algorithmPlan.displayName, prepared.imagePath)
                                                         .toStdString());
                        }

                        const std::int64_t extractionMs = extractTimer.elapsed();
                        context.featureCache->insert(prepared.imagePath, features);
                        totalPreparationMs.fetch_add(prepared.preparationMs, std::memory_order_relaxed);
                        totalExtractionMs.fetch_add(extractionMs, std::memory_order_relaxed);

                        QJsonObject settings = makeFeatureRecordSettings(algorithmPlan, options);
                        settings[QStringLiteral("storage")] = QStringLiteral("memory_only");
                        settings[QStringLiteral("effective_keypoint_limit")] = prepared.effectiveKeypointLimit;
                        settings[QStringLiteral("image_width")] = prepared.originalWidth;
                        settings[QStringLiteral("image_height")] = prepared.originalHeight;
                        settings[QStringLiteral("feature_backend")] =
                            autoSift ? QString::fromStdString(features->computeBackend) : algorithmPlan.algorithmId;
                        settings[QStringLiteral("backend_fallback")] = algorithmPlan.backendFallback;
                        settings[QStringLiteral("feature_prepare_ms")] = static_cast<double>(prepared.preparationMs);
                        settings[QStringLiteral("feature_prepare_workers")] = prepareWorkerCount;
                        settings[QStringLiteral("feature_pipeline_workers")] = extractionWorkerCount;
                        settings[QStringLiteral("feature_prefetch_depth")] = prefetchDepth;
                        settings[QStringLiteral("feature_queue_wait_ms")] = static_cast<double>(queueWaitMs);
                        settings[QStringLiteral("feature_extract_ms")] = static_cast<double>(extractionMs);
                        settings[QStringLiteral("mask_path")] = prepared.maskPath;
                        orderedRecords[static_cast<std::size_t>(prepared.index)] =
                            MatchPhotosFeatureRecord{prepared.imagePath, features->size(), settings};
                        populatedRecords[static_cast<std::size_t>(prepared.index)] = 1U;

                        advanceMatchPhotosProgress(context);
                        const int done = extractedCount.fetch_add(1, std::memory_order_relaxed) + 1;
                        std::lock_guard progressLock(progressMutex);
                        reportMatchPhotosProgress(
                            context,
                            QStringLiteral("feature"),
                            QStringLiteral("%1 特征提取：%2/%3，当前 %4 点，内存缓存约 %5 MiB")
                                .arg(algorithmPlan.displayName)
                                .arg(done)
                                .arg(totalImages)
                                .arg(features->size())
                                .arg(static_cast<double>(context.featureCache->approximateBytes()) / (1024.0 * 1024.0),
                                     0,
                                     'f',
                                     1),
                            done,
                            totalImages);
                    }
                });
        }
        catch (const std::exception& error)
        {
            return makeFeatureReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("%1 特征提取失败：%2").arg(algorithmPlan.displayName, QString::fromUtf8(error.what())),
                extractedCount.load());
        }

        if (shouldCancelMatchPhotos(context))
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("用户取消 %1 特征提取").arg(algorithmPlan.displayName),
                                     extractedCount.load());
        }
        if (extractedCount.load() != totalImages)
        {
            return makeFeatureReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("影像预取队列提前结束：已完成 %1/%2").arg(extractedCount.load()).arg(totalImages),
                extractedCount.load());
        }
        if (featureRecords)
        {
            for (int index = 0; index < totalImages; ++index)
            {
                if (populatedRecords[static_cast<std::size_t>(index)] != 0U)
                {
                    featureRecords->push_back(std::move(orderedRecords[static_cast<std::size_t>(index)]));
                }
            }
        }

        return makeFeatureReport(MatchPhotosStageStatus::Completed,
                                 QStringLiteral("%1 特征提取完成：%2 张，全部保存在任务内存中，%3；"
                                                "CPU 预取 %4 worker，峰值缓冲 %5/%6 张；"
                                                "提取流水线 %7 worker；累计准备 %8 ms、"
                                                "队列等待 %9 ms、提取后端 %10 ms")
                                     .arg(algorithmPlan.displayName)
                                     .arg(extractedCount.load())
                                     .arg(algorithmPlan.backendReason)
                                     .arg(prepareWorkerCount)
                                     .arg(queue.peakBufferedCount())
                                     .arg(prefetchDepth)
                                     .arg(extractionWorkerCount)
                                     .arg(totalPreparationMs.load())
                                     .arg(totalQueueWaitMs.load())
                                     .arg(totalExtractionMs.load()),
                                 extractedCount.load());
    }

} // namespace xjw::matchphotos

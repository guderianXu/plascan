#include "FeatureStage.h"

#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosRuntime.h"
#include "FeaturePreparationQueue.h"
#include "ImageFeaturePointFile.h"
#include "ImageMatchFile.h"
#include "io/PathIO.h"
#include "ImageMatchingRegistry.h"
#include "concurrency/SafeWorkerGroup.h"
#include "loma_r/LoMaRAlgorithm.h"
#include "plamatch_hct/PlaMatchHctAlgorithm.h"
#include "plamatch_hct/PlaMatchHctFeatureCacheFile.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftComputeBackend.h"

#include <QElapsedTimer>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
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

        double alignmentSourceScale(int downscale)
        {
            if (downscale == 0)
            {
                return 0.5;
            }
            if (downscale == 1 || downscale == 2 || downscale == 4 || downscale == 8)
            {
                return static_cast<double>(downscale);
            }
            throw std::invalid_argument("invalid alignment downscale");
        }

        cv::Mat integerDecimate(const cv::Mat& source, int factor)
        {
            if (factor == 1 || source.empty())
            {
                return source;
            }
            const int width = (source.cols + factor - 1) / factor;
            const int height = (source.rows + factor - 1) / factor;
            cv::Mat result(height, width, source.type());
            const std::size_t pixel_size = source.elemSize();
            for (int y = 0; y < height; ++y)
            {
                const unsigned char* source_row = source.ptr<unsigned char>(y * factor);
                unsigned char* destination_row = result.ptr<unsigned char>(y);
                for (int x = 0; x < width; ++x)
                {
                    std::memcpy(destination_row + static_cast<std::size_t>(x) * pixel_size,
                                source_row + static_cast<std::size_t>(x * factor) * pixel_size,
                                pixel_size);
                }
            }
            return result;
        }

        cv::Mat upsampleHighest(const cv::Mat& source)
        {
            if (source.empty())
            {
                return source;
            }
            const int channels = source.channels();
            cv::Mat source_float;
            source.convertTo(source_float, CV_MAKETYPE(CV_32F, channels));
            cv::Mat result_float(source.rows * 2 - 1, source.cols * 2 - 1, source_float.type());
            for (int source_y = 0; source_y < source.rows; ++source_y)
            {
                const float* source_row = source_float.ptr<float>(source_y);
                float* output_row = result_float.ptr<float>(source_y * 2);
                for (int source_x = 0; source_x + 1 < source.cols; ++source_x)
                {
                    for (int channel = 0; channel < channels; ++channel)
                    {
                        const float left = source_row[source_x * channels + channel];
                        output_row[(source_x * 2) * channels + channel] = left;
                        output_row[(source_x * 2 + 1) * channels + channel] =
                            (left + source_row[(source_x + 1) * channels + channel]) * 0.5F;
                    }
                }
                for (int channel = 0; channel < channels; ++channel)
                {
                    output_row[(result_float.cols - 1) * channels + channel] =
                        source_row[(source.cols - 1) * channels + channel];
                }
            }
            for (int source_y = 0; source_y + 1 < source.rows; ++source_y)
            {
                const float* top = result_float.ptr<float>(source_y * 2);
                const float* bottom = result_float.ptr<float>(source_y * 2 + 2);
                float* middle = result_float.ptr<float>(source_y * 2 + 1);
                const int row_values = result_float.cols * channels;
                for (int index = 0; index < row_values; ++index)
                {
                    middle[index] = (top[index] + bottom[index]) * 0.5F;
                }
            }
            cv::Mat result;
            result_float.convertTo(result, source.type());
            return result;
        }

        cv::Mat applyAlignmentAccuracy(const cv::Mat& source, int downscale)
        {
            return downscale == 0 ? upsampleHighest(source) : integerDecimate(source, downscale);
        }

        image_matching::ImageFeaturePointCatalog makeFeaturePointCatalog(const QString& imagePath,
                                                                         const image_matching::FeatureSet& features,
                                                                         const MatchPhotosAlgorithmPlan& algorithmPlan)
        {
            image_matching::ImageFeaturePointCatalog catalog;
            catalog.owner =
                image_matching::ImageMatchFile::identityForImage(imagePath, features.imageWidth, features.imageHeight);
            catalog.algorithmId = algorithmPlan.algorithmId;
            catalog.algorithmVersion = algorithmPlan.algorithmVersion;
            catalog.featureSchemaVersion = algorithmPlan.featureSchemaVersion;
            catalog.observations.reserve(features.keypoints.size());
            for (std::size_t index = 0; index < features.keypoints.size(); ++index)
            {
                const cv::KeyPoint& keypoint = features.keypoints[index];
                image_matching::KeypointObservation observation;
                observation.featureId = static_cast<std::uint32_t>(index);
                observation.x = keypoint.pt.x;
                observation.y = keypoint.pt.y;
                observation.scale = keypoint.size;
                observation.orientation = keypoint.angle;
                observation.response = keypoint.response;
                catalog.observations.push_back(observation);
            }
            return catalog;
        }

        QString plaMatchFeatureProducerSignature(const MatchPhotosOptions& options,
                                                 const MatchPhotosAlgorithmPlan& algorithmPlan,
                                                 const QString& maskPath)
        {
            const QFileInfo mask_info(maskPath);
            return QStringLiteral("plamatch-hct-feature-producer-v1;algorithm=%1;algorithm_version=%2;schema=%3;"
                                  "downscale=%4;max_keypoints=%5;keypoints_per_mpx=%6;max_image_dim=%7;"
                                  "remove_borders=%8;mask_mode=%9;mask_path=%10;mask_size=%11;mask_mtime=%12;"
                                  "backend=%13;device=%14")
                .arg(algorithmPlan.algorithmId)
                .arg(algorithmPlan.algorithmVersion)
                .arg(algorithmPlan.featureSchemaVersion)
                .arg(algorithmPlan.alignmentDownscale)
                .arg(options.maxKeypoints)
                .arg(options.keypointLimitPerMegapixel)
                .arg(algorithmPlan.maxImageDim)
                .arg(algorithmPlan.featureRemoveBorders)
                .arg(options.maskApplyMode)
                .arg(mask_info.exists() ? mask_info.absoluteFilePath() : QString())
                .arg(mask_info.exists() ? mask_info.size() : 0)
                .arg(mask_info.exists() ? mask_info.lastModified().toMSecsSinceEpoch() : 0)
                .arg(static_cast<int>(algorithmPlan.executionBackend))
                .arg(options.cudaDevice);
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
        const bool persistFeaturePoints = !context.projectPath.trimmed().isEmpty() ||
                                          !context.workingDirectory.trimmed().isEmpty() ||
                                          !context.matchDirectory.trimmed().isEmpty();
        const QString featurePointDirectory = persistFeaturePoints ? matchPhotosMatchDirectory(context) : QString();
        std::vector<FeaturePreparationRequest> requests;
        requests.reserve(static_cast<std::size_t>(totalImages));
        for (int index = 0; index < totalImages; ++index)
        {
            FeaturePreparationRequest request;
            request.index = index;
            request.imagePath = images.at(index);
            if (persistFeaturePoints &&
                algorithmPlan.algorithmId == QLatin1String(image_matching::kPlaMatchHctAlgorithmId))
            {
                request.featurePath = image_matching::PlaMatchHctFeatureCacheFile::filePathForImage(
                    featurePointDirectory, request.imagePath);
            }
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
                if (applyMask)
                {
                    prepared.maskPath = maskPathForImage(context, request.imagePath);
                }
                prepared.featurePath = request.featurePath;
                if (!request.featurePath.isEmpty() && options.reuseExistingMatches)
                {
                    const QString signature =
                        plaMatchFeatureProducerSignature(options, algorithmPlan, prepared.maskPath);
                    prepared.cachedFeatures = image_matching::PlaMatchHctFeatureCacheFile::read(
                        request.featurePath, request.imagePath, signature, &prepared.cacheMissReason);
                    if (prepared.cachedFeatures)
                    {
                        prepared.reused = true;
                        prepared.originalWidth = prepared.cachedFeatures->imageWidth;
                        prepared.originalHeight = prepared.cachedFeatures->imageHeight;
                        prepared.effectiveKeypointLimit = prepared.cachedFeatures->size();
                        prepared.preparationMs = totalTimer.elapsed();
                        return prepared;
                    }
                }

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
                const cv::Mat capped_image = resizeImage(original, algorithmPlan.maxImageDim, &prepared.resizeScale);
                const bool native_accuracy =
                    algorithmPlan.algorithmId == QLatin1String(image_matching::kPlaMatchHctAlgorithmId);
                const double source_scale =
                    native_accuracy ? 1.0 : alignmentSourceScale(algorithmPlan.alignmentDownscale);
                prepared.inputImage = native_accuracy
                                          ? capped_image
                                          : applyAlignmentAccuracy(capped_image, algorithmPlan.alignmentDownscale);
                prepared.coordinateScale = source_scale / prepared.resizeScale;
                // Accuracy sampling is an exact lattice transform, so it contributes scale only.
                // The offset below solely inverts OpenCV's pixel-centre mapping for an explicit cap resize.
                prepared.coordinateOffsetX = 0.5 / prepared.resizeScale - 0.5;
                prepared.coordinateOffsetY = prepared.coordinateOffsetX;
                prepared.imageResizeMs = phaseTimer.elapsed();

                if (applyMask)
                {
                    phaseTimer.restart();
                    const cv::Mat exclusionMask = loadMaskForImage(context, request.imagePath, original.size());
                    prepared.mask = makeExtractorValidMask(exclusionMask, prepared.inputImage.size(), options);
                    prepared.maskReadMs = phaseTimer.elapsed();
                }
                prepared.preparationMs = totalTimer.elapsed();
                return prepared;
            },
            prepareWorkerCount);

        std::atomic_int extractedCount{0};
        std::atomic_int reusedCount{0};
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

                        QElapsedTimer extractTimer;
                        extractTimer.start();
                        std::shared_ptr<image_matching::FeatureSet> features = prepared.cachedFeatures;
                        if (!features)
                        {
                            image_matching::ImageMatchingRuntimeConfig runtime;
                            runtime.cudaDevice = options.cudaDevice;
                            runtime.maxKeypoints = prepared.effectiveKeypointLimit;
                            runtime.removeBorders = algorithmPlan.featureRemoveBorders;
                            // 显式最长边限制和五档采样已在预取阶段完成，提取器不得再次缩放。
                            runtime.maxImageDimension = 0;
                            runtime.alignmentDownscale = algorithmPlan.alignmentDownscale;
                            runtime.siftDetectionThreshold = algorithmPlan.siftDetectionThreshold;
                            runtime.siftContrastThreshold = algorithmPlan.siftContrastThreshold;
                            runtime.siftBackend = algorithmPlan.executionBackend;
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
                            input.coordinateScale = prepared.coordinateScale;
                            input.coordinateOffsetX = prepared.coordinateOffsetX;
                            input.coordinateOffsetY = prepared.coordinateOffsetY;

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
                            features =
                                std::make_shared<image_matching::FeatureSet>(algorithmIt->second->extract(input));
                        }
                        if (!features->isConsistent() || features->empty())
                        {
                            throw std::runtime_error(QStringLiteral("影像没有生成有效 %1 特征：%2")
                                                         .arg(algorithmPlan.displayName, prepared.imagePath)
                                                         .toStdString());
                        }

                        const std::int64_t extractionMs = prepared.reused ? 0 : extractTimer.elapsed();
                        QString featurePointPath;
                        if (persistFeaturePoints)
                        {
                            featurePointPath = image_matching::ImageFeaturePointFile::filePathForImage(
                                featurePointDirectory, prepared.imagePath);
                            if (!prepared.reused)
                            {
                                const image_matching::ImageFeaturePointCatalog catalog =
                                    makeFeaturePointCatalog(prepared.imagePath, *features, algorithmPlan);
                                QString writeError;
                                if (!image_matching::ImageFeaturePointFile::write(
                                        featurePointPath, catalog, &writeError))
                                {
                                    throw std::runtime_error(writeError.toStdString());
                                }
                                if (!prepared.featurePath.isEmpty())
                                {
                                    const QString signature =
                                        plaMatchFeatureProducerSignature(options, algorithmPlan, prepared.maskPath);
                                    if (!image_matching::PlaMatchHctFeatureCacheFile::write(prepared.featurePath,
                                                                                            prepared.imagePath,
                                                                                            signature,
                                                                                            *features,
                                                                                            &writeError))
                                    {
                                        throw std::runtime_error(writeError.toStdString());
                                    }
                                }
                            }
                        }
                        if (prepared.reused)
                        {
                            reusedCount.fetch_add(1, std::memory_order_relaxed);
                        }
                        context.featureCache->insert(prepared.imagePath, features);
                        totalPreparationMs.fetch_add(prepared.preparationMs, std::memory_order_relaxed);
                        totalExtractionMs.fetch_add(extractionMs, std::memory_order_relaxed);

                        QJsonObject settings = makeFeatureRecordSettings(algorithmPlan, options);
                        settings[QStringLiteral("storage")] =
                            persistFeaturePoints ? QStringLiteral("pifeature") : QStringLiteral("memory_only");
                        settings[QStringLiteral("feature_point_path")] = featurePointPath;
                        settings[QStringLiteral("effective_keypoint_limit")] = prepared.effectiveKeypointLimit;
                        settings[QStringLiteral("image_width")] = prepared.originalWidth;
                        settings[QStringLiteral("image_height")] = prepared.originalHeight;
                        const QString featureBackend = QString::fromStdString(features->computeBackend);
                        settings[QStringLiteral("feature_backend")] =
                            featureBackend.isEmpty() ? algorithmPlan.algorithmId : featureBackend;
                        settings[QStringLiteral("backend_fallback")] = algorithmPlan.backendFallback;
                        settings[QStringLiteral("feature_prepare_ms")] = static_cast<double>(prepared.preparationMs);
                        settings[QStringLiteral("feature_prepare_workers")] = prepareWorkerCount;
                        settings[QStringLiteral("feature_pipeline_workers")] = extractionWorkerCount;
                        settings[QStringLiteral("feature_prefetch_depth")] = prefetchDepth;
                        settings[QStringLiteral("feature_queue_wait_ms")] = static_cast<double>(queueWaitMs);
                        settings[QStringLiteral("feature_extract_ms")] = static_cast<double>(extractionMs);
                        settings[QStringLiteral("feature_cache_reused")] = prepared.reused;
                        settings[QStringLiteral("feature_cache_path")] = prepared.featurePath;
                        settings[QStringLiteral("feature_cache_miss_reason")] = prepared.cacheMissReason;
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

        return makeFeatureReport(
            MatchPhotosStageStatus::Completed,
            QStringLiteral("%1 特征准备完成：%2 张，磁盘缓存命中 %3 张，描述子保存在任务内存中，"
                           "特征点几何%4，%5；"
                           "CPU 预取 %6 worker，峰值缓冲 %7/%8 张；"
                           "提取流水线 %9 worker；累计准备 %10 ms、"
                           "队列等待 %11 ms、提取后端 %12 ms")
                .arg(algorithmPlan.displayName)
                .arg(extractedCount.load())
                .arg(reusedCount.load())
                .arg(persistFeaturePoints ? QStringLiteral("已持久化") : QStringLiteral("未持久化"))
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

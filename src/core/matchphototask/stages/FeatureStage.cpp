#include "TraditionalFeatureExtractor.h"
#include "FeatureFileIO.h"
#include "FeaturePreparationQueue.h"
#include "FeatureStage.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosRuntime.h"
#include "io/PathIO.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <exception>

namespace xjw
{
namespace matchphotos
{
namespace
{

MatchPhotosStageReport makeFeatureReport(MatchPhotosStageStatus status,
                                         const QString &message,
                                         int itemCount = 0)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("feature");
    report.displayName = QStringLiteral("特征提取");
    report.status = status;
    report.message = message;
    report.itemCount = itemCount;
    return report;
}

bool shouldUseCudaSift(const MatchPhotosOptions &options,
                       const MatchPhotosAlgorithmPlan &algorithmPlan)
{
    return options.device == ComputeDevice::Cuda ||
        (options.device == ComputeDevice::Auto && algorithmPlan.preferCuda);
}

float siftDetectionThresholdForTiePoints(const MatchPhotosOptions &options)
{
    // CUDA SIFT 内部阈值量纲约为 UI 阈值 * 1000。空三连接点需要比通用特征提取
    // 更密的候选点，但阈值过低会引入大量弱响应点，并挤占 LightGlue 的有效预算。
    return options.profile == MatchPhotosProfile::Fast ? 0.003f : 0.0005f;
}

cv::Mat resizeForFeatureExtraction(const cv::Mat &grayImage,
                                   int maxImageDim,
                                   double *scale)
{
    if (scale)
    {
        *scale = 1.0;
    }
    if (grayImage.empty() || maxImageDim <= 0)
    {
        return grayImage;
    }

    const int maxSide = std::max(grayImage.cols, grayImage.rows);
    if (maxSide <= maxImageDim)
    {
        return grayImage;
    }

    const double resizeScale = static_cast<double>(maxImageDim) / static_cast<double>(maxSide);
    cv::Mat resized;
    cv::resize(grayImage,
               resized,
               cv::Size(),
               resizeScale,
               resizeScale,
               cv::INTER_AREA);
    if (scale)
    {
        *scale = resizeScale;
    }
    return resized;
}

void restoreOriginalCoordinates(FeatureOutput *output,
                                int originalWidth,
                                int originalHeight,
                                double resizeScale)
{
    if (!output)
    {
        return;
    }

    if (resizeScale > 0.0 && resizeScale != 1.0)
    {
        const float inverseScale = static_cast<float>(1.0 / resizeScale);
        for (cv::KeyPoint &keypoint : output->keypoints)
        {
            keypoint.pt.x *= inverseScale;
            keypoint.pt.y *= inverseScale;
            keypoint.size *= inverseScale;
        }
    }

    output->imageWidth = originalWidth;
    output->imageHeight = originalHeight;
}

} // namespace

MatchPhotosStageReport FeatureStage::run(const MatchPhotosContext &context,
                                         const MatchPhotosOptions &options,
                                         const MatchPhotosAlgorithmPlan &algorithmPlan,
                                         std::vector<MatchPhotosFeatureRecord> *featureRecords) const
{
    if (options.planOnly)
    {
        return makeFeatureReport(MatchPhotosStageStatus::Skipped,
                                 QStringLiteral("plan-only 模式，跳过特征提取"));
    }

    if (algorithmPlan.featureAlgorithm.compare(QStringLiteral("sift"), Qt::CaseInsensitive) != 0)
    {
        return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                 QStringLiteral("连接点匹配当前只支持 SIFT 特征"));
    }

    const QStringList images = context.pairInput.images;
    if (images.isEmpty())
    {
        return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                 QStringLiteral("没有可用于提取特征的影像"));
    }

    QDir featureDir(matchPhotosFeatureDirectory(context));
    if (!featureDir.exists() && !featureDir.mkpath(QStringLiteral(".")))
    {
        return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                 QStringLiteral("无法创建特征目录: %1").arg(featureDir.path()));
    }

    xjw::feature_extractors::TraditionalFeatureConfig config;
    config.maxImageSize = algorithmPlan.maxImageDim;
    config.removeBorders = 16;
    config.allowDeviceFallback = options.device != ComputeDevice::Cuda;

    const bool useCuda = shouldUseCudaSift(options, algorithmPlan);
    const bool applyKeypointMask = shouldApplyMasksToKeypoints(options);
    const int prefetchDepth = std::clamp(options.featurePrefetchDepth, 1, 4);
    int extractedCount = 0;
    int reusedCount = 0;
    const int totalImages = images.size();

    reportMatchPhotosProgress(context,
                              QStringLiteral("feature"),
                              QStringLiteral("SIFT 特征提取：准备处理 %1 张影像%2")
                                  .arg(totalImages)
                                  .arg(applyKeypointMask ? QStringLiteral("，按蒙版过滤关键点")
                                                         : QString()),
                              0,
                              totalImages);

    std::vector<FeaturePreparationRequest> requests;
    requests.reserve(static_cast<std::size_t>(totalImages));
    for (int index = 0; index < totalImages; ++index)
    {
        FeaturePreparationRequest request;
        request.index = index;
        request.imagePath = images.at(index);
        request.featurePath =
            matchPhotosFeaturePath(context, request.imagePath, algorithmPlan);
        requests.push_back(std::move(request));
    }

    FeaturePreparationQueue preparationQueue(
        std::move(requests),
        prefetchDepth,
        context.cancelFlag,
        [&](const FeaturePreparationRequest &request)
        {
            QElapsedTimer totalTimer;
            totalTimer.start();

            PreparedFeatureImage prepared;
            prepared.index = request.index;
            prepared.imagePath = request.imagePath;
            prepared.featurePath = request.featurePath;

            const int existingCount = FeatureFileIO::peekCount(request.featurePath);
            if (!applyKeypointMask &&
                options.reuseExistingFeatures &&
                existingCount > 0 &&
                FeatureFileIO::peekAlgorithm(request.featurePath) == "sift")
            {
                prepared.reused = true;
                prepared.existingKeypointCount = existingCount;
                prepared.preparationMs = totalTimer.elapsed();
                return prepared;
            }

            QElapsedTimer phaseTimer;
            phaseTimer.start();
            cv::Mat grayImage =
                xjw::common::io::readImage(request.imagePath, cv::IMREAD_GRAYSCALE);
            prepared.imageReadMs = phaseTimer.elapsed();
            if (grayImage.empty())
            {
                prepared.errorMessage =
                    QStringLiteral("无法读取影像: %1").arg(request.imagePath);
                prepared.preparationMs = totalTimer.elapsed();
                return prepared;
            }

            prepared.originalWidth = grayImage.cols;
            prepared.originalHeight = grayImage.rows;
            prepared.effectiveKeypointLimit =
                resolveFeatureKeypointLimit(options, algorithmPlan, grayImage.cols, grayImage.rows);

            phaseTimer.restart();
            prepared.inputImage = resizeForFeatureExtraction(grayImage,
                                                              algorithmPlan.maxImageDim,
                                                              &prepared.resizeScale);
            prepared.imageResizeMs = phaseTimer.elapsed();

            if (applyKeypointMask)
            {
                phaseTimer.restart();
                prepared.maskPath = maskPathForImage(context, request.imagePath);
                prepared.mask = loadMaskForImage(context,
                                                 request.imagePath,
                                                 grayImage.size());
                prepared.maskReadMs = phaseTimer.elapsed();
            }
            prepared.preparationMs = totalTimer.elapsed();
            return prepared;
        });

    for (int index = 0; index < totalImages; ++index)
    {
        if (shouldCancelMatchPhotos(context))
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("用户取消连接点特征提取"),
                                     extractedCount + reusedCount);
        }

        PreparedFeatureImage prepared;
        if (!preparationQueue.take(&prepared))
        {
            if (shouldCancelMatchPhotos(context))
            {
                return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                         QStringLiteral("用户取消连接点特征提取"),
                                         extractedCount + reusedCount);
            }
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("影像预取队列提前结束"),
                                     extractedCount + reusedCount);
        }
        if (!prepared.errorMessage.isEmpty())
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     prepared.errorMessage,
                                     extractedCount + reusedCount);
        }

        if (prepared.reused)
        {
            if (featureRecords)
            {
                QJsonObject settings = makeFeatureRecordSettings(algorithmPlan, options);
                settings[QStringLiteral("feature_prefetch_depth")] = prefetchDepth;
                settings[QStringLiteral("feature_reused")] = true;
                featureRecords->push_back(
                    MatchPhotosFeatureRecord{prepared.imagePath,
                                             prepared.featurePath,
                                             prepared.existingKeypointCount,
                                             settings});
            }
            ++reusedCount;
            advanceMatchPhotosProgress(context);
            reportMatchPhotosProgress(context,
                                      QStringLiteral("feature"),
                                      QStringLiteral("SIFT 特征提取：%1/%2，复用 %3 张，新提取 %4 张")
                                          .arg(extractedCount + reusedCount)
                                          .arg(totalImages)
                                          .arg(reusedCount)
                                          .arg(extractedCount),
                                      extractedCount + reusedCount,
                                      totalImages);
            continue;
        }

        try
        {
            xjw::feature_extractors::TraditionalFeatureConfig imageConfig = config;
            imageConfig.maxKeypoints = prepared.effectiveKeypointLimit;
            imageConfig.detectionThreshold = siftDetectionThresholdForTiePoints(options);

            QElapsedTimer phaseTimer;
            phaseTimer.start();
            FeatureOutput output = xjw::feature_extractors::TraditionalFeatureExtractor::detect(
                prepared.inputImage, imageConfig, "sift", useCuda, options.cudaDevice);
            const qint64 extractionMs = phaseTimer.elapsed();
            restoreOriginalCoordinates(&output,
                                       prepared.originalWidth,
                                       prepared.originalHeight,
                                       prepared.resizeScale);
            const int unmaskedKeypointCount = output.count();
            int maskRemovedKeypointCount = 0;
            phaseTimer.restart();
            if (applyKeypointMask && !prepared.mask.empty())
            {
                output = filterFeatureOutputByMask(output, prepared.mask);
                maskRemovedKeypointCount =
                    std::max(0, unmaskedKeypointCount - output.count());
            }
            const qint64 maskFilterMs = phaseTimer.elapsed();

            phaseTimer.restart();
            if (!FeatureFileIO::write(prepared.featurePath,
                                      QFileInfo(prepared.imagePath).fileName(),
                                      output,
                                      "sift"))
            {
                return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                         QStringLiteral("无法写入 SIFT 特征文件: %1")
                                             .arg(prepared.featurePath),
                                         extractedCount + reusedCount);
            }
            const qint64 writeMs = phaseTimer.elapsed();

            if (featureRecords)
            {
                QJsonObject settings = makeFeatureRecordSettings(algorithmPlan, options);
                settings[QStringLiteral("effective_keypoint_limit")] =
                    prepared.effectiveKeypointLimit;
                settings[QStringLiteral("image_width")] = prepared.originalWidth;
                settings[QStringLiteral("image_height")] = prepared.originalHeight;
                settings[QStringLiteral("feature_prefetch_depth")] = prefetchDepth;
                settings[QStringLiteral("feature_prepare_ms")] =
                    static_cast<double>(prepared.preparationMs);
                settings[QStringLiteral("feature_image_read_ms")] =
                    static_cast<double>(prepared.imageReadMs);
                settings[QStringLiteral("feature_image_resize_ms")] =
                    static_cast<double>(prepared.imageResizeMs);
                settings[QStringLiteral("feature_mask_read_ms")] =
                    static_cast<double>(prepared.maskReadMs);
                settings[QStringLiteral("feature_extract_ms")] =
                    static_cast<double>(extractionMs);
                settings[QStringLiteral("feature_mask_filter_ms")] =
                    static_cast<double>(maskFilterMs);
                settings[QStringLiteral("feature_write_ms")] =
                    static_cast<double>(writeMs);
                settings[QStringLiteral("feature_cuda_enabled")] = useCuda;
                if (applyKeypointMask)
                {
                    settings[QStringLiteral("mask_path")] = prepared.maskPath;
                    settings[QStringLiteral("mask_unfiltered_keypoints")] = unmaskedKeypointCount;
                    settings[QStringLiteral("mask_filtered_keypoints")] = maskRemovedKeypointCount;
                }
                featureRecords->push_back(
                    MatchPhotosFeatureRecord{prepared.imagePath,
                                             prepared.featurePath,
                                             output.count(),
                                             settings});
            }
            ++extractedCount;
            advanceMatchPhotosProgress(context);
            reportMatchPhotosProgress(context,
                                      QStringLiteral("feature"),
                                      QStringLiteral("SIFT 特征提取：%1/%2，新提取 %3 张，复用 %4 张，当前关键点 %5%6")
                                          .arg(extractedCount + reusedCount)
                                          .arg(totalImages)
                                          .arg(extractedCount)
                                          .arg(reusedCount)
                                          .arg(output.count())
                                          .arg(applyKeypointMask
                                                   ? QStringLiteral("，蒙版剔除 %1 点").arg(maskRemovedKeypointCount)
                                                   : QString()),
                                      extractedCount + reusedCount,
                                      totalImages);
        }
        catch (const std::exception &e)
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("SIFT 特征提取失败: %1").arg(QString::fromUtf8(e.what())),
                                     extractedCount + reusedCount);
        }
    }

    return makeFeatureReport(MatchPhotosStageStatus::Completed,
                             QStringLiteral("SIFT 特征完成：新提取 %1 张，复用 %2 张，CPU 预取深度 %3，目录 %4")
                                 .arg(extractedCount)
                                 .arg(reusedCount)
                                 .arg(prefetchDepth)
                                 .arg(featureDir.path()),
                             extractedCount + reusedCount);
}

} // namespace matchphotos
} // namespace xjw

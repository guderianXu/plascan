#include "FeatureStage.h"

#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosRuntime.h"
#include "FeaturePreparationQueue.h"
#include "io/PathIO.h"
#include "sift/SiftFeatureExtractor.h"
#include "sift_lightglue/SiftLightGlueAlgorithm.h"

#include <QElapsedTimer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <exception>
#include <memory>
#include <utility>

namespace xjw::matchphotos
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

float siftDetectionThreshold(const MatchPhotosOptions &options)
{
    // cudaSift 的阈值会在 SiftFeatureExtractor 内转换到其原生量纲。
    // 快速模式减少低响应点，其余模式保持低纹理摄影测量场景所需的较密检测。
    return options.profile == MatchPhotosProfile::Fast ? 0.003f : 0.0005f;
}

cv::Mat resizeImage(const cv::Mat &image, int maximumDimension, double *scale)
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

    const double resizeScale = static_cast<double>(maximumDimension) /
        static_cast<double>(maximumSide);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), resizeScale, resizeScale, cv::INTER_AREA);
    if (scale)
    {
        *scale = resizeScale;
    }
    return resized;
}

cv::Mat makeExtractorValidMask(const cv::Mat &exclusionMask,
                               const cv::Size &extractorSize)
{
    if (exclusionMask.empty())
    {
        return cv::Mat();
    }

    cv::Mat resized;
    if (exclusionMask.size() == extractorSize)
    {
        resized = exclusionMask;
    }
    else
    {
        cv::resize(exclusionMask, resized, extractorSize, 0.0, 0.0, cv::INTER_NEAREST);
    }

    // PlaScan 项目蒙版使用 0=有效、非 0=排除；统一算法接口使用非 0=有效。
    // 在模块边界只转换一次，避免不同提取器各自解释蒙版造成语义反转。
    cv::Mat validMask;
    cv::compare(resized, cv::Scalar(0), validMask, cv::CMP_EQ);
    return validMask;
}

} // namespace

MatchPhotosStageReport FeatureStage::run(
    const MatchPhotosContext &context,
    const MatchPhotosOptions &options,
    const MatchPhotosAlgorithmPlan &algorithmPlan,
    std::vector<MatchPhotosFeatureRecord> *featureRecords) const
{
    if (options.planOnly)
    {
        return makeFeatureReport(MatchPhotosStageStatus::Skipped,
                                 QStringLiteral("plan-only 模式，跳过特征提取"));
    }
    if (!context.featureCache)
    {
        return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                 QStringLiteral("内部错误：没有创建任务级特征缓存"));
    }
    if (algorithmPlan.algorithmId.compare(
            QString::fromLatin1(image_matching::kSiftLightGlueAlgorithmId),
            Qt::CaseInsensitive) != 0)
    {
        return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                 QStringLiteral("当前仅注册 CUDA SIFT + TensorRT LightGlue"));
    }

    const QStringList images = context.pairInput.images;
    if (images.isEmpty())
    {
        return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                 QStringLiteral("没有可用于提取特征的影像"));
    }

    const bool applyMask = shouldApplyMasksToKeypoints(options);
    const int prefetchDepth = std::clamp(options.featurePrefetchDepth, 1, 4);
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

    reportMatchPhotosProgress(
        context,
        QStringLiteral("feature"),
        QStringLiteral("CUDA SIFT：准备处理 %1 张影像%2")
            .arg(totalImages)
            .arg(applyMask ? QStringLiteral("，蒙版约束关键点") : QString()),
        0,
        totalImages);

    FeaturePreparationQueue queue(
        std::move(requests),
        prefetchDepth,
        context.cancelFlag,
        [&](const FeaturePreparationRequest &request)
        {
            PreparedFeatureImage prepared;
            prepared.index = request.index;
            prepared.imagePath = request.imagePath;
            QElapsedTimer totalTimer;
            QElapsedTimer phaseTimer;
            totalTimer.start();

            phaseTimer.start();
            const cv::Mat original =
                common::io::readImage(request.imagePath, cv::IMREAD_GRAYSCALE);
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
            prepared.inputImage = resizeImage(original,
                                              algorithmPlan.maxImageDim,
                                              &prepared.resizeScale);
            prepared.imageResizeMs = phaseTimer.elapsed();

            if (applyMask)
            {
                phaseTimer.restart();
                prepared.maskPath = maskPathForImage(context, request.imagePath);
                const cv::Mat exclusionMask =
                    loadMaskForImage(context, request.imagePath, original.size());
                prepared.mask = makeExtractorValidMask(exclusionMask, prepared.inputImage.size());
                prepared.maskReadMs = phaseTimer.elapsed();
            }
            prepared.preparationMs = totalTimer.elapsed();
            return prepared;
        });

    int extractedCount = 0;
    bool usedCudaForAll = true;
    for (int index = 0; index < totalImages; ++index)
    {
        if (shouldCancelMatchPhotos(context))
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("用户取消 SIFT 特征提取"),
                                     extractedCount);
        }

        PreparedFeatureImage prepared;
        if (!queue.take(&prepared))
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("影像预取队列提前结束"),
                                     extractedCount);
        }
        if (!prepared.errorMessage.isEmpty())
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     prepared.errorMessage,
                                     extractedCount);
        }

        image_matching::ImageMatchingRuntimeConfig runtime;
        runtime.cudaDevice = options.cudaDevice;
        runtime.maxKeypoints = prepared.effectiveKeypointLimit;
        runtime.removeBorders = 16;
        runtime.siftDetectionThreshold = siftDetectionThreshold(options);
        // 产品链路明确要求 CUDA SIFT。CUDA 不可用时必须报错，不能静默切换到
        // OpenCV CPU SIFT 后仍把结果标记为同一算法版本。
        runtime.allowCpuSiftFallback = false;

        image_matching::ImageFeatureInput input;
        input.imagePath = prepared.imagePath;
        input.grayImage = prepared.inputImage;
        input.validMask = prepared.mask;
        input.originalWidth = prepared.originalWidth;
        input.originalHeight = prepared.originalHeight;
        input.coordinateScale = prepared.resizeScale > 0.0
            ? 1.0 / prepared.resizeScale
            : 1.0;

        QElapsedTimer extractTimer;
        extractTimer.start();
        bool usedCuda = false;
        std::shared_ptr<image_matching::FeatureSet> features;
        try
        {
            features = std::make_shared<image_matching::FeatureSet>(
                image_matching::SiftFeatureExtractor::extract(input, runtime, &usedCuda));
        }
        catch (const std::exception &error)
        {
            return makeFeatureReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("SIFT 特征提取失败：%1；影像：%2")
                    .arg(QString::fromUtf8(error.what()), prepared.imagePath),
                extractedCount);
        }
        if (!features->isConsistent() || features->empty())
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("影像没有生成有效 SIFT 特征：%1")
                                         .arg(prepared.imagePath),
                                     extractedCount);
        }
        if (!usedCuda)
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("CUDA SIFT 未在 GPU 上执行：%1")
                                         .arg(prepared.imagePath),
                                     extractedCount);
        }

        context.featureCache->insert(prepared.imagePath, features);
        usedCudaForAll = usedCudaForAll && usedCuda;
        ++extractedCount;

        if (featureRecords)
        {
            QJsonObject settings = makeFeatureRecordSettings(algorithmPlan, options);
            settings[QStringLiteral("storage")] = QStringLiteral("memory_only");
            settings[QStringLiteral("effective_keypoint_limit")] =
                prepared.effectiveKeypointLimit;
            settings[QStringLiteral("image_width")] = prepared.originalWidth;
            settings[QStringLiteral("image_height")] = prepared.originalHeight;
            settings[QStringLiteral("feature_cuda_enabled")] = usedCuda;
            settings[QStringLiteral("feature_prepare_ms")] =
                static_cast<double>(prepared.preparationMs);
            settings[QStringLiteral("feature_extract_ms")] =
                static_cast<double>(extractTimer.elapsed());
            settings[QStringLiteral("mask_path")] = prepared.maskPath;
            featureRecords->push_back(
                MatchPhotosFeatureRecord{prepared.imagePath, features->size(), settings});
        }

        advanceMatchPhotosProgress(context);
        reportMatchPhotosProgress(
            context,
            QStringLiteral("feature"),
            QStringLiteral("SIFT 特征提取：%1/%2，当前 %3 点，内存缓存约 %4 MiB")
                .arg(extractedCount)
                .arg(totalImages)
                .arg(features->size())
                .arg(static_cast<double>(context.featureCache->approximateBytes()) /
                         (1024.0 * 1024.0),
                     0,
                     'f',
                     1),
            extractedCount,
            totalImages);
    }

    return makeFeatureReport(
        MatchPhotosStageStatus::Completed,
        QStringLiteral("SIFT 特征提取完成：%1 张，全部保存在任务内存中，%2")
            .arg(extractedCount)
            .arg(usedCudaForAll ? QStringLiteral("CUDA")
                                : QStringLiteral("含 CPU 回退")),
        extractedCount);
}

} // namespace xjw::matchphotos

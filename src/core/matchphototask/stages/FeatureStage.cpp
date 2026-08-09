#include "FeatureStage.h"

#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosRuntime.h"
#include "FeaturePreparationQueue.h"
#include "io/PathIO.h"
#include "ImageMatchingRegistry.h"
#include "loma_r/LoMaRAlgorithm.h"

#include <QElapsedTimer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <exception>
#include <memory>
#include <unordered_map>
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
    const QStringList images = context.pairInput.images;
    if (images.isEmpty())
    {
        return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                 QStringLiteral("没有可用于提取特征的影像"));
    }

    ResolvedLoMaRTensorRtPackage loma_package;
    if (algorithmPlan.algorithmId == QLatin1String(image_matching::kLoMaRAlgorithmId))
    {
        reportMatchPhotosProgress(
            context,
            QStringLiteral("model_prepare"),
            QStringLiteral("正在检查 LoMa-R ONNX，并为当前 TensorRT/GPU 准备本机 engine"),
            0,
            1);
        loma_package = resolveLoMaRTensorRtPackage(options, algorithmPlan.maxKeypoints);
        if (!loma_package.isValid())
        {
            return makeFeatureReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("LoMa-R TensorRT 模型包不可用：%1")
                    .arg(loma_package.errorMessage));
        }
        reportMatchPhotosProgress(
            context,
            QStringLiteral("model_prepare"),
            QStringLiteral("LoMa-R 本机 TensorRT engine 已就绪：%1")
                .arg(loma_package.environmentSummary),
            1,
            1);
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
        QStringLiteral("%1：准备处理 %2 张影像%3")
            .arg(algorithmPlan.displayName)
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
            const int readMode = algorithmPlan.requiresColorInput
                ? cv::IMREAD_COLOR
                : cv::IMREAD_GRAYSCALE;
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
    // 相同关键点预算的影像共享算法实例和 TensorRT execution context，避免逐图反序列化引擎。
    std::unordered_map<int, std::unique_ptr<image_matching::IImageMatchingAlgorithm>> algorithms;
    for (int index = 0; index < totalImages; ++index)
    {
        if (shouldCancelMatchPhotos(context))
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("用户取消 %1 特征提取")
                                         .arg(algorithmPlan.displayName),
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
        runtime.removeBorders = algorithmPlan.featureRemoveBorders;
        runtime.siftDetectionThreshold = algorithmPlan.siftDetectionThreshold;
        // 注册算法均要求 CUDA。不能静默切换到另一实现后仍沿用相同算法版本。
        runtime.allowCpuSiftFallback = false;
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
        input.coordinateScale = prepared.resizeScale > 0.0
            ? 1.0 / prepared.resizeScale
            : 1.0;

        QElapsedTimer extractTimer;
        extractTimer.start();
        std::shared_ptr<image_matching::FeatureSet> features;
        try
        {
            auto algorithm_it = algorithms.find(prepared.effectiveKeypointLimit);
            if (algorithm_it == algorithms.end())
            {
                QString create_error;
                auto algorithm = image_matching::ImageMatchingRegistry::create(
                    algorithmPlan.algorithmId, runtime, &create_error);
                if (!algorithm)
                {
                    throw std::runtime_error(create_error.toStdString());
                }
                algorithm_it = algorithms.emplace(prepared.effectiveKeypointLimit,
                                                  std::move(algorithm)).first;
            }
            features = std::make_shared<image_matching::FeatureSet>(
                algorithm_it->second->extract(input));
        }
        catch (const std::exception &error)
        {
            return makeFeatureReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("%1 特征提取失败：%2；影像：%3")
                    .arg(algorithmPlan.displayName)
                    .arg(QString::fromUtf8(error.what()), prepared.imagePath),
                extractedCount);
        }
        if (!features->isConsistent() || features->empty())
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("影像没有生成有效 %1 特征：%2")
                                         .arg(algorithmPlan.displayName,
                                              prepared.imagePath),
                                     extractedCount);
        }

        context.featureCache->insert(prepared.imagePath, features);
        ++extractedCount;

        if (featureRecords)
        {
            QJsonObject settings = makeFeatureRecordSettings(algorithmPlan, options);
            settings[QStringLiteral("storage")] = QStringLiteral("memory_only");
            settings[QStringLiteral("effective_keypoint_limit")] =
                prepared.effectiveKeypointLimit;
            settings[QStringLiteral("image_width")] = prepared.originalWidth;
            settings[QStringLiteral("image_height")] = prepared.originalHeight;
            settings[QStringLiteral("feature_cuda_enabled")] = algorithmPlan.requiresCuda;
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
            QStringLiteral("%1 特征提取：%2/%3，当前 %4 点，内存缓存约 %5 MiB")
                .arg(algorithmPlan.displayName)
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
        QStringLiteral("%1 特征提取完成：%2 张，全部保存在任务内存中，%3")
            .arg(algorithmPlan.displayName)
            .arg(extractedCount)
            .arg(algorithmPlan.requiresCuda ? QStringLiteral("CUDA")
                                            : QStringLiteral("CPU")),
        extractedCount);
}

} // namespace xjw::matchphotos

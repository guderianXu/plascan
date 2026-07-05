#include "TraditionalFeatureExtractor.h"
#include "FeatureFileIO.h"
#include "FeatureStage.h"
#include "MatchPhotosRuntime.h"

#include <QDir>
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
                                const cv::Mat &originalImage,
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

    output->imageWidth = originalImage.cols;
    output->imageHeight = originalImage.rows;
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
    config.maxKeypoints = algorithmPlan.maxKeypoints;
    config.maxImageSize = algorithmPlan.maxImageDim;
    config.removeBorders = 4;
    config.allowDeviceFallback = true;

    const bool useCuda = shouldUseCudaSift(options, algorithmPlan);
    int extractedCount = 0;
    int reusedCount = 0;

    for (const QString &imagePath : images)
    {
        if (shouldCancelMatchPhotos(context))
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("用户取消连接点特征提取"),
                                     extractedCount + reusedCount);
        }

        const QString featurePath = matchPhotosFeaturePath(context, imagePath, algorithmPlan);
        const int existingCount = FeatureFileIO::peekCount(featurePath);
        if (options.reuseExistingFeatures &&
            existingCount > 0 &&
            FeatureFileIO::peekAlgorithm(featurePath) == "sift")
        {
            if (featureRecords)
            {
                featureRecords->push_back(
                    MatchPhotosFeatureRecord{imagePath,
                                             featurePath,
                                             existingCount,
                                             makeFeatureRecordSettings(algorithmPlan, options)});
            }
            ++reusedCount;
            advanceMatchPhotosProgress(context);
            continue;
        }

        cv::Mat grayImage = cv::imread(imagePath.toStdString(), cv::IMREAD_GRAYSCALE);
        if (grayImage.empty())
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("无法读取影像: %1").arg(imagePath),
                                     extractedCount + reusedCount);
        }

        try
        {
            double resizeScale = 1.0;
            const cv::Mat inputImage = resizeForFeatureExtraction(grayImage,
                                                                  algorithmPlan.maxImageDim,
                                                                  &resizeScale);
            FeatureOutput output = xjw::feature_extractors::TraditionalFeatureExtractor::detect(
                inputImage, config, "sift", useCuda, options.cudaDevice);
            restoreOriginalCoordinates(&output, grayImage, resizeScale);

            if (!FeatureFileIO::write(featurePath,
                                      QFileInfo(imagePath).fileName(),
                                      output,
                                      "sift"))
            {
                return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                         QStringLiteral("无法写入 SIFT 特征文件: %1").arg(featurePath),
                                         extractedCount + reusedCount);
            }

            if (featureRecords)
            {
                featureRecords->push_back(
                    MatchPhotosFeatureRecord{imagePath,
                                             featurePath,
                                             output.count(),
                                             makeFeatureRecordSettings(algorithmPlan, options)});
            }
            ++extractedCount;
            advanceMatchPhotosProgress(context);
        }
        catch (const std::exception &e)
        {
            return makeFeatureReport(MatchPhotosStageStatus::Failed,
                                     QStringLiteral("SIFT 特征提取失败: %1").arg(QString::fromUtf8(e.what())),
                                     extractedCount + reusedCount);
        }
    }

    return makeFeatureReport(MatchPhotosStageStatus::Completed,
                             QStringLiteral("SIFT 特征完成：新提取 %1 张，复用 %2 张，目录 %3")
                                 .arg(extractedCount)
                                 .arg(reusedCount)
                                 .arg(featureDir.path()),
                             extractedCount + reusedCount);
}

} // namespace matchphotos
} // namespace xjw

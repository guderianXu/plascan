#include "MvsPipelineInternals.h"

namespace xjw::mvs::pipeline_detail
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    bool writeFastDepthMatStorage(const std::string& path, const cv::Mat& matrix, std::string* errorMsg)
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

    bool saveDepthPreviewPng(const std::string& path, const cv::Mat& depthMap, std::string* errorMsg)
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

    QString manifestPathForOutput(const DepthGenConfig& config, const std::string& outputDir)
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
} // namespace xjw::mvs::pipeline_detail

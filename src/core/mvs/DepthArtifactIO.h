#pragma once

#include <string>

#include <QString>

namespace cv
{
    class Mat;
}

namespace xjw::mvs
{
    struct DepthGenConfig;

    bool writeFastDepthMatStorage(const std::string& path, const cv::Mat& matrix, std::string* errorMsg);
    bool saveDepthPreviewPng(const std::string& path, const cv::Mat& depthMap, std::string* errorMsg);
    QString manifestPathForOutput(const DepthGenConfig& config, const std::string& outputDir);

} // namespace xjw::mvs

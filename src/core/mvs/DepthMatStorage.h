#pragma once

#include "result/OperationResult.h"

#include <QString>

namespace cv
{
    class Mat;
}

namespace xjw::core::project
{

    // Binary matrix IO only. No fusion, frame processing or workflow dependency.
    xjw::common::OperationResult loadDepthMatStorage(const QString& path, cv::Mat* matrix);
    xjw::common::OperationResult writeDepthMatStorage(const QString& path, const cv::Mat& matrix);

} // namespace xjw::core::project

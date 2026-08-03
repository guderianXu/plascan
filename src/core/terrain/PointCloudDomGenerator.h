#pragma once

#include "DemDomTypes.h"
#include "OrthoGenerationOptions.h"

#include <QJsonObject>

#include <atomic>
#include <functional>

namespace xjw
{

struct PointCloudDomResult
{
    cv::Mat imageBgr;
    cv::Mat validMask;
    DemGridData reference;
    OrthoGenerationOptions resolvedOptions;
    qint64 inputPointCount = 0;
    qint64 projectedPointCount = 0;
    qint64 validPixelCount = 0;
    double coverageRatio = 0.0;
};

class PointCloudDomGenerator
{
public:
    using ProgressCallback = std::function<void(const QString &, int)>;

    static bool estimate(const PlaPointCloud &pointCloud,
                         const OrthoGenerationOptions &options,
                         QJsonObject *result,
                         QString *errorMsg = nullptr);

    static bool generate(const PlaPointCloud &pointCloud,
                         const OrthoGenerationOptions &options,
                         PointCloudDomResult *result,
                         QString *errorMsg = nullptr,
                         const std::atomic_bool *cancelFlag = nullptr,
                         const ProgressCallback &progressCallback = {});
};

} // namespace xjw

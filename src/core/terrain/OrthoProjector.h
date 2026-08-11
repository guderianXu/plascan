#pragma once

#include "FramePinholeCamera.h"
#include "DemDomTypes.h"
#include "OrthoGenerationOptions.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <opencv2/core.hpp>

#include <atomic>
#include <functional>
#include <vector>

namespace xjw
{

struct OrthoImageInput
{
    QString imageId;
    QString imagePath;
    QString exclusionMaskPath;
    FramePinholeCamera camera;
};

struct OrthoOutputGrid
{
    DemGridData reference;
    OrthoGenerationOptions resolvedOptions;
    double minEdgeX = 0.0;
    double minEdgeY = 0.0;
    double maxEdgeX = 0.0;
    double maxEdgeY = 0.0;
    qint64 estimatedMemoryBytes = 0;
};

struct OrthoProjectionResult
{
    cv::Mat imageBgr;
    cv::Mat surfaceMask;
    cv::Mat coverageMask;
    cv::Mat holeFilledMask;
    OrthoOutputGrid outputGrid;
    int selectedCameraCount = 0;
    int loadedCameraCount = 0;
    int contributingCameraCount = 0;
    qint64 filledPixelCount = 0;
    qint64 holeFilledPixelCount = 0;
    double coverageRatio = 0.0;
};

class OrthoProjector
{
public:
    using ProgressCallback = std::function<void(const QString &, int)>;

    static bool planOutputGrid(const DemGridData &demGrid,
                               const OrthoGenerationOptions &options,
                               OrthoOutputGrid *outputGrid,
                               QString *errorMsg = nullptr);

    static bool buildImageInputs(const QStringList &selectedImages,
                                 const QJsonObject &projectMeta,
                                 std::vector<OrthoImageInput> *inputs,
                                 QString *errorMsg = nullptr);

    static bool project(const DemGridData &demGrid,
                        const std::vector<OrthoImageInput> &inputs,
                        const OrthoGenerationOptions &options,
                        double demElevationOffset,
                        OrthoProjectionResult *result,
                        QString *errorMsg = nullptr,
                        const std::atomic_bool *cancelFlag = nullptr,
                        const ProgressCallback &progressCallback = {});
};

} // namespace xjw

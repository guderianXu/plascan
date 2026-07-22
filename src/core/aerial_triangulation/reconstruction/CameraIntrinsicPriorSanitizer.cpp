#include "reconstruction/CameraIntrinsicPriorSanitizer.h"

#include "project/ProjectCameraIO.h"
#include "project/ProjectCommonUtils.h"

#include <Camera.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw::aerial_triangulation
{
namespace
{

constexpr double kDominantGroupMinRatio = 0.70;
constexpr double kDominantGroupLowerScale = 0.75;
constexpr double kDominantGroupUpperScale = 1.33;
constexpr double kOutlierLowerScale = 0.50;
constexpr double kOutlierUpperScale = 2.00;

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    const double upper = values[middle];
    if (values.size() % 2 != 0)
    {
        return upper;
    }

    const auto lowerEnd = values.begin() + static_cast<std::ptrdiff_t>(middle);
    const double lower = *std::max_element(values.begin(), lowerEnd);
    return (lower + upper) * 0.5;
}

struct CameraFocalRecord
{
    QString path;
    double focalPixels = 0.0;
};

} // namespace

CameraIntrinsicPriorSanitizationResult sanitizeProjectCameraIntrinsicPriors(
    const QStringList &imagePaths,
    QMap<QString, QJsonObject> *cameraByPath)
{
    CameraIntrinsicPriorSanitizationResult result;
    if (!cameraByPath || imagePaths.size() < 4)
    {
        return result;
    }

    std::vector<CameraFocalRecord> cameras;
    cameras.reserve(static_cast<std::size_t>(imagePaths.size()));
    for (const QString &imagePath : imagePaths)
    {
        const QString normalizedPath = xjw::common::project::normalizePath(imagePath);
        auto cameraIt = cameraByPath->constFind(normalizedPath);
        if (cameraIt == cameraByPath->cend())
        {
            // 兼容无头 CLI 或旧工程中未规范化的相对路径键。
            cameraIt = cameraByPath->constFind(imagePath);
        }
        if (cameraIt == cameraByPath->cend())
        {
            continue;
        }

        Camera camera;
        if (!xjw::common::project::cameraFromJson(cameraIt.value(), &camera) || !camera.isValid())
        {
            continue;
        }

        const double focalPixels = std::sqrt(camera.focalX() * camera.focalY());
        if (std::isfinite(focalPixels) && focalPixels > 1.0)
        {
            cameras.push_back({cameraIt.key(), focalPixels});
        }
    }
    result.inspectedCameraCount = static_cast<int>(cameras.size());
    if (cameras.size() < 4)
    {
        return result;
    }

    std::vector<double> allFocals;
    allFocals.reserve(cameras.size());
    for (const CameraFocalRecord &camera : cameras)
    {
        allFocals.push_back(camera.focalPixels);
    }
    const double initialMedian = median(allFocals);
    if (!(initialMedian > 1.0) || !std::isfinite(initialMedian))
    {
        return result;
    }

    std::vector<double> dominantFocals;
    for (const CameraFocalRecord &camera : cameras)
    {
        const double ratio = camera.focalPixels / initialMedian;
        if (ratio >= kDominantGroupLowerScale && ratio <= kDominantGroupUpperScale)
        {
            dominantFocals.push_back(camera.focalPixels);
        }
    }
    const int requiredDominantCount = std::max(3, static_cast<int>(std::ceil(
        static_cast<double>(cameras.size()) * kDominantGroupMinRatio)));
    if (static_cast<int>(dominantFocals.size()) < requiredDominantCount)
    {
        return result;
    }

    const double dominantMedian = median(dominantFocals);
    if (!(dominantMedian > 1.0) || !std::isfinite(dominantMedian))
    {
        return result;
    }
    result.dominantGroupCount = static_cast<int>(dominantFocals.size());
    result.dominantMedianFocalPixels = dominantMedian;

    for (const CameraFocalRecord &record : cameras)
    {
        const double ratio = record.focalPixels / dominantMedian;
        if (ratio >= kOutlierLowerScale && ratio <= kOutlierUpperScale)
        {
            continue;
        }

        auto cameraIt = cameraByPath->find(record.path);
        if (cameraIt == cameraByPath->end())
        {
            continue;
        }
        QJsonObject cameraObject = cameraIt.value();
        const double scale = dominantMedian / record.focalPixels;
        cameraObject.insert(QStringLiteral("fu"), cameraObject.value(QStringLiteral("fu")).toDouble() * scale);
        cameraObject.insert(QStringLiteral("fv"), cameraObject.value(QStringLiteral("fv")).toDouble() * scale);
        cameraIt.value() = cameraObject;
        result.normalizedImagePaths.append(record.path);
        ++result.normalizedCameraCount;
    }
    return result;
}

} // namespace xjw::aerial_triangulation

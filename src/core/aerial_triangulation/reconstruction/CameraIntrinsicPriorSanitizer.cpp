#include "reconstruction/CameraIntrinsicPriorSanitizer.h"

#include "ProjectCameraIO.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"

#include <algorithm>
#include <cmath>
#include <optional>
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
    camera_core::ImageId imageId;
    double focalPixels = 0.0;
};

template <typename CameraMap, typename FocalReader, typename FocalScaler>
CameraIntrinsicPriorSanitizationResult sanitizeByFocalGroup(
    const std::vector<camera_core::ImageId>& imageIds,
    CameraMap* cameraByImageId,
    FocalReader&& readFocal,
    FocalScaler&& scaleFocal)
{
    CameraIntrinsicPriorSanitizationResult result;
    if (!cameraByImageId || imageIds.size() < 4)
    {
        return result;
    }

    std::vector<CameraFocalRecord> cameras;
    cameras.reserve(imageIds.size());
    for (const camera_core::ImageId& imageId : imageIds)
    {
        const auto cameraIt = cameraByImageId->find(imageId);
        if (cameraIt == cameraByImageId->end())
        {
            continue;
        }

        const std::optional<double> focalPixels = readFocal(cameraIt->second);
        if (focalPixels.has_value() && std::isfinite(*focalPixels) && *focalPixels > 1.0)
        {
            cameras.push_back({imageId, *focalPixels});
        }
    }
    result.inspectedCameraCount = static_cast<int>(cameras.size());
    if (cameras.size() < 4)
    {
        return result;
    }

    std::vector<double> allFocals;
    allFocals.reserve(cameras.size());
    for (const CameraFocalRecord& camera : cameras)
    {
        allFocals.push_back(camera.focalPixels);
    }
    const double initialMedian = median(allFocals);
    if (!(initialMedian > 1.0) || !std::isfinite(initialMedian))
    {
        return result;
    }

    std::vector<double> dominantFocals;
    for (const CameraFocalRecord& camera : cameras)
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

    for (const CameraFocalRecord& record : cameras)
    {
        const double ratio = record.focalPixels / dominantMedian;
        if (ratio >= kOutlierLowerScale && ratio <= kOutlierUpperScale)
        {
            continue;
        }

        auto cameraIt = cameraByImageId->find(record.imageId);
        if (cameraIt == cameraByImageId->end())
        {
            continue;
        }
        scaleFocal(cameraIt->second, dominantMedian / record.focalPixels);
        result.normalizedImageIds.push_back(record.imageId);
        ++result.normalizedCameraCount;
    }
    return result;
}

} // namespace

bool isTrustedProjectCameraIntrinsic(const QJsonObject &cameraObject)
{
    if (cameraObject.isEmpty())
    {
        return false;
    }

    const QString intrinsicSource =
        cameraObject.value(QStringLiteral("intrinsic_source")).toString().trimmed().toLower();
    if (intrinsicSource == QLatin1String("sfm_estimated"))
    {
        return false;
    }
    if (intrinsicSource == QLatin1String("imported") ||
        intrinsicSource == QLatin1String("calibrated") ||
        intrinsicSource == QLatin1String("user") ||
        intrinsicSource == QLatin1String("exif"))
    {
        return true;
    }

    if (!cameraObject.value(QStringLiteral("source_file")).toString().trimmed().isEmpty())
    {
        return true;
    }

    const QString source =
        cameraObject.value(QStringLiteral("source")).toString().trimmed().toLower();
    if (source == QLatin1String("init_from_intrinsics") ||
        source == QLatin1String("init_pose_intrinsics_manual"))
    {
        return true;
    }
    if (source == QLatin1String("init_from_exif_or_default") ||
        source == QLatin1String("init_pose_intrinsics_from_exif_or_default"))
    {
        const QString focalSource =
            cameraObject.value(QStringLiteral("focal_source")).toString().trimmed().toLower();
        return !focalSource.isEmpty() && focalSource != QLatin1String("default_mm");
    }
    return false;
}

CameraIntrinsicPriorSanitizationResult sanitizeProjectCameraIntrinsicPriors(
    const std::vector<camera_core::ImageId> &imageIds,
    CameraIntrinsicsByImageId *cameraByImageId)
{
    return sanitizeByFocalGroup(
        imageIds,
        cameraByImageId,
        [](const QJsonObject& cameraObject) -> std::optional<double>
        {
            xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
            if (!xjw::common::project::decodeFramePinholeNumericState(cameraObject, &camera) || !camera.isValid())
            {
                return std::nullopt;
            }
            const double focalPixels = std::sqrt(camera.focalX() * camera.focalY());
            return std::isfinite(focalPixels) && focalPixels > 1.0
                       ? std::optional<double>(focalPixels)
                       : std::nullopt;
        },
        [](QJsonObject& cameraObject, double scale)
        {
            cameraObject.insert(QStringLiteral("fu"), cameraObject.value(QStringLiteral("fu")).toDouble() * scale);
            cameraObject.insert(QStringLiteral("fv"), cameraObject.value(QStringLiteral("fv")).toDouble() * scale);
        });
}

CameraIntrinsicPriorSanitizationResult sanitizeProjectCameraIntrinsicPriors(
    const std::vector<camera_core::ImageId>& imageIds,
    FramePinholeStatesByImageId* cameraByImageId)
{
    return sanitizeByFocalGroup(
        imageIds,
        cameraByImageId,
        [](const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera) -> std::optional<double>
        {
            if (!camera.isValid())
            {
                return std::nullopt;
            }
            const double focalPixels = std::sqrt(camera.focalX() * camera.focalY());
            return std::isfinite(focalPixels) && focalPixels > 1.0
                       ? std::optional<double>(focalPixels)
                       : std::nullopt;
        },
        [](xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera, double scale)
        {
            const xjw::camera_models::frame_pinhole::FramePinholeNumericState::Intrinsics intrinsics =
                camera.intrinsics();
            camera.setIntrinsics(intrinsics.focalX * scale,
                                 intrinsics.focalY * scale,
                                 intrinsics.principalX,
                                 intrinsics.principalY);
        });
}

} // namespace xjw::aerial_triangulation

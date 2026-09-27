#include "reconstruction/CameraIntrinsicPriorSanitizer.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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
            placamera::ImageId imageId;
            double focalPixels = 0.0;
        };

        template <typename CameraMap, typename FocalReader, typename FocalScaler>
        CameraIntrinsicPriorSanitizationResult sanitizeByFocalGroup(const std::vector<placamera::ImageId>& imageIds,
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
            for (const placamera::ImageId& imageId : imageIds)
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
            const int requiredDominantCount =
                std::max(3, static_cast<int>(std::ceil(static_cast<double>(cameras.size()) * kDominantGroupMinRatio)));
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

    bool isTrustedProjectCameraIntrinsic(const QJsonObject& instanceState)
    {
        if (instanceState.isEmpty())
        {
            return false;
        }

        const QJsonObject metadata = instanceState.value(QStringLiteral("metadata")).toObject();
        const auto property = [&instanceState, &metadata](const QString& key)
        { return instanceState.contains(key) ? instanceState.value(key) : metadata.value(key); };

        const QString intrinsicSource = property(QStringLiteral("intrinsic_source")).toString().trimmed().toLower();
        if (intrinsicSource == QLatin1String("sfm_estimated"))
        {
            return false;
        }
        if (intrinsicSource == QLatin1String("imported") || intrinsicSource == QLatin1String("calibrated") ||
            intrinsicSource == QLatin1String("user") || intrinsicSource == QLatin1String("exif"))
        {
            return true;
        }

        if (!property(QStringLiteral("source_file")).toString().trimmed().isEmpty())
        {
            return true;
        }

        const QString source = property(QStringLiteral("source")).toString().trimmed().toLower();
        if (source == QLatin1String("init_from_intrinsics") || source == QLatin1String("init_pose_intrinsics_manual"))
        {
            return true;
        }
        if (source == QLatin1String("init_from_exif_or_default") ||
            source == QLatin1String("init_pose_intrinsics_from_exif_or_default"))
        {
            const QString focalSource = property(QStringLiteral("focal_source")).toString().trimmed().toLower();
            return !focalSource.isEmpty() && focalSource != QLatin1String("default_mm");
        }
        return false;
    }

    CameraIntrinsicPriorSanitizationResult
    sanitizeProjectCameraIntrinsicPriors(const std::vector<placamera::ImageId>& imageIds,
                                         FramePinholeModelsByImageId* cameraByImageId)
    {
        return sanitizeByFocalGroup(
            imageIds,
            cameraByImageId,
            [](const std::shared_ptr<const placamera::FramePinholeModel>& camera) -> std::optional<double>
            {
                if (!camera)
                {
                    return std::nullopt;
                }
                const auto& intrinsics = camera->pinholeDefinition().intrinsics();
                const double focalPixels = std::sqrt(intrinsics.focalX * intrinsics.focalY);
                return std::isfinite(focalPixels) && focalPixels > 1.0 ? std::optional<double>(focalPixels)
                                                                       : std::nullopt;
            },
            [](std::shared_ptr<const placamera::FramePinholeModel>& camera, double scale)
            {
                auto intrinsics = camera->pinholeDefinition().intrinsics();
                intrinsics.focalX *= scale;
                intrinsics.focalY *= scale;
                const auto& original_definition = camera->pinholeDefinition();
                auto definition = placamera::FramePinholeDefinition::create(
                    placamera::CameraDefinitionId("sanitized-focal-" + std::string(camera->imageId().value())),
                    intrinsics,
                    original_definition.distortion(),
                    original_definition.pixelConvention(),
                    original_definition.groundFrame(),
                    original_definition.depthAxisFlipped());
                camera = std::make_shared<const placamera::FramePinholeModel>(
                    placamera::FramePinholeModel::create(camera->instanceId(),
                                                         camera->imageId(),
                                                         std::move(definition),
                                                         camera->imageSize(),
                                                         camera->pose(),
                                                         camera->captureTime()));
            });
    }

} // namespace xjw::aerial_triangulation

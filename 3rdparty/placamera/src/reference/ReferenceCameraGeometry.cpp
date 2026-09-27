#include "placamera/reference/ReferenceCameraGeometry.h"

#include <cmath>
#include <exception>
#include <unordered_set>
#include <utility>

namespace placamera::reference
{
    namespace
    {

        bool finiteCenter(const std::array<double, 3>& center)
        {
            for (const double value : center)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
            }
            return true;
        }

    } // namespace

    ReferenceCameraGeometry::ReferenceCameraGeometry(placamera::CameraModelPtr<placamera::FramePinholeModel> model)
        : _model(std::move(model))
    {
    }

    placamera::Result<ReferenceCameraGeometry>
    ReferenceCameraGeometry::create(placamera::CameraModelPtr<placamera::FramePinholeModel> model)
    {
        if (!model)
        {
            return placamera::Result<ReferenceCameraGeometry>::failure(
                placamera::CameraErrorCode::InvalidArgument,
                "reference camera geometry requires a PlaCamera frame-pinhole model");
        }
        if (!model->capabilities().contains(placamera::CapabilityKind::StaticPose) || !model->imageSize().isValid() ||
            model->pose().frame != model->groundFrame())
        {
            return placamera::Result<ReferenceCameraGeometry>::failure(
                placamera::CameraErrorCode::InvalidModelState,
                "reference camera geometry requires valid static pose, frame, and image size");
        }
        return placamera::Result<ReferenceCameraGeometry>::success(ReferenceCameraGeometry(std::move(model)));
    }

    const placamera::FramePinholeModel& ReferenceCameraGeometry::model() const noexcept
    {
        return *_model;
    }

    const placamera::CameraModelPtr<placamera::FramePinholeModel>& ReferenceCameraGeometry::modelPtr() const noexcept
    {
        return _model;
    }

    const placamera::CameraInstanceId& ReferenceCameraGeometry::instanceId() const noexcept
    {
        return _model->instanceId();
    }

    const placamera::ImageId& ReferenceCameraGeometry::imageId() const noexcept
    {
        return _model->imageId();
    }

    const placoordinate::CoordinateFrameId& ReferenceCameraGeometry::worldFrame() const noexcept
    {
        return _model->groundFrame();
    }

    ReferenceCameraPosition::ReferenceCameraPosition(placamera::ImageId imageId,
                                                     placoordinate::CoordinateFrameId worldFrame,
                                                     std::array<double, 3> center)
        : _imageId(std::move(imageId)), _worldFrame(std::move(worldFrame)), _center(center)
    {
    }

    placamera::Result<ReferenceCameraPosition> ReferenceCameraPosition::create(
        placamera::ImageId imageId, placoordinate::CoordinateFrameId worldFrame, std::array<double, 3> center)
    {
        if (!finiteCenter(center))
        {
            return placamera::Result<ReferenceCameraPosition>::failure(
                placamera::CameraErrorCode::InvalidArgument,
                "reference camera position must contain finite coordinates");
        }
        try
        {
            return placamera::Result<ReferenceCameraPosition>::success(
                ReferenceCameraPosition(std::move(imageId), std::move(worldFrame), center));
        }
        catch (const std::exception& exception)
        {
            return placamera::Result<ReferenceCameraPosition>::failure(placamera::CameraErrorCode::InvalidArgument,
                                                                       exception.what());
        }
        catch (...)
        {
            return placamera::Result<ReferenceCameraPosition>::failure(placamera::CameraErrorCode::InvalidArgument,
                                                                       "reference camera position identity is invalid");
        }
    }

    const placamera::ImageId& ReferenceCameraPosition::imageId() const noexcept
    {
        return _imageId;
    }

    const placoordinate::CoordinateFrameId& ReferenceCameraPosition::worldFrame() const noexcept
    {
        return _worldFrame;
    }

    const std::array<double, 3>& ReferenceCameraPosition::center() const noexcept
    {
        return _center;
    }

    placamera::Result<void> validateReferenceCameraGeometryMap(const ReferenceCameraGeometryMap& references)
    {
        for (const auto& [key, reference] : references)
        {
            if (key != reference.imageId())
            {
                return placamera::Result<void>::failure(placamera::CameraErrorCode::InvalidModelState,
                                                        "reference camera geometry map key does not match its ImageId");
            }
        }
        return placamera::Result<void>::success();
    }

    placamera::Result<void> validateReferenceCameraPositionMap(const ReferenceCameraPositionMap& references)
    {
        for (const auto& [key, reference] : references)
        {
            if (!(key == reference.imageId()))
            {
                return placamera::Result<void>::failure(placamera::CameraErrorCode::InvalidModelState,
                                                        "reference camera position map key does not match its ImageId");
            }
            if (!finiteCenter(reference.center()))
            {
                return placamera::Result<void>::failure(
                    placamera::CameraErrorCode::InvalidArgument,
                    "reference camera position map contains non-finite coordinates");
            }
        }
        return placamera::Result<void>::success();
    }

    placamera::Result<void> validateReferenceCameraInputs(const std::vector<placamera::ImageId>& imageIds,
                                                          std::size_t imageCount,
                                                          const ReferenceCameraGeometryMap& geometries,
                                                          const ReferenceCameraPositionMap& positions)
    {
        const auto geometry_validation = validateReferenceCameraGeometryMap(geometries);
        if (!geometry_validation)
        {
            return geometry_validation;
        }
        const auto position_validation = validateReferenceCameraPositionMap(positions);
        if (!position_validation)
        {
            return position_validation;
        }
        if (!geometries.empty() || !positions.empty())
        {
            if (imageIds.size() != imageCount)
            {
                return placamera::Result<void>::failure(
                    placamera::CameraErrorCode::InvalidArgument,
                    "reference geometry requires one ImageId for every input image");
            }
        }

        std::unordered_set<placamera::ImageId> inputIds;
        inputIds.reserve(imageIds.size());
        for (const auto& imageId : imageIds)
        {
            if (!inputIds.emplace(imageId).second)
            {
                return placamera::Result<void>::failure(placamera::CameraErrorCode::InvalidArgument,
                                                        "input ImageId list contains a duplicate identity: " +
                                                            imageId.value());
            }
        }

        const auto validateMembership = [&inputIds](const auto& references,
                                                    const char* label) -> placamera::Result<void>
        {
            for (const auto& [imageId, _] : references)
            {
                if (inputIds.find(imageId) == inputIds.end())
                {
                    return placamera::Result<void>::failure(
                        placamera::CameraErrorCode::InvalidArgument,
                        std::string(label) + " contains an ImageId outside the input image set: " + imageId.value());
                }
            }
            return placamera::Result<void>::success();
        };
        const auto geometry_membership = validateMembership(geometries, "reference camera geometry");
        if (!geometry_membership)
        {
            return geometry_membership;
        }
        return validateMembership(positions, "reference camera position");
    }

    placamera::Result<std::optional<placoordinate::CoordinateFrameId>>
    commonReferenceWorldFrame(const ReferenceCameraGeometryMap& geometries, const ReferenceCameraPositionMap& positions)
    {
        const auto geometry_validation = validateReferenceCameraGeometryMap(geometries);
        if (!geometry_validation)
        {
            return placamera::Result<std::optional<placoordinate::CoordinateFrameId>>::failure(
                geometry_validation.error());
        }
        const auto position_validation = validateReferenceCameraPositionMap(positions);
        if (!position_validation)
        {
            return placamera::Result<std::optional<placoordinate::CoordinateFrameId>>::failure(
                position_validation.error());
        }

        std::optional<placoordinate::CoordinateFrameId> frame;
        const auto visit = [&frame](const placoordinate::CoordinateFrameId& candidate)
        {
            if (!frame)
            {
                frame = candidate;
                return true;
            }
            if (*frame == candidate)
            {
                return true;
            }
            return false;
        };

        for (const auto& [_, reference] : geometries)
        {
            if (!visit(reference.worldFrame()))
            {
                return placamera::Result<std::optional<placoordinate::CoordinateFrameId>>::failure(
                    placamera::CameraErrorCode::FrameMismatch, "reference camera inputs use mixed coordinate frames");
            }
        }
        for (const auto& [_, reference] : positions)
        {
            if (!visit(reference.worldFrame()))
            {
                return placamera::Result<std::optional<placoordinate::CoordinateFrameId>>::failure(
                    placamera::CameraErrorCode::FrameMismatch, "reference camera inputs use mixed coordinate frames");
            }
        }
        return placamera::Result<std::optional<placoordinate::CoordinateFrameId>>::success(std::move(frame));
    }

} // namespace placamera::reference

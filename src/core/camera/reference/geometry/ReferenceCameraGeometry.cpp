#include "ReferenceCameraGeometry.h"

#include <cmath>
#include <exception>
#include <unordered_set>
#include <utility>

namespace xjw::camera_reference
{
    namespace
    {

        void clearError(std::string* error)
        {
            if (error)
            {
                error->clear();
            }
        }

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

    ReferenceCameraGeometry::ReferenceCameraGeometry(
        camera_models::frame_pinhole::FramePinholeNumericState numericState)
        : _numericState(std::move(numericState))
    {
    }

    std::optional<ReferenceCameraGeometry>
    ReferenceCameraGeometry::create(camera_models::frame_pinhole::FramePinholeNumericState numericState,
                                    std::string* error)
    {
        clearError(error);
        std::string validationError;
        if (!numericState.hasBoundIdentity())
        {
            if (error)
            {
                *error = "reference camera geometry requires a bound camera instance, image, and world frame";
            }
            return std::nullopt;
        }
        if (!numericState.validateNumericalState(&validationError))
        {
            if (error)
            {
                *error = "reference camera geometry has invalid numerical state: " + validationError;
            }
            return std::nullopt;
        }
        const auto imageSize = numericState.imageSize();
        if (!imageSize || !imageSize->isValid())
        {
            if (error)
            {
                *error = "reference camera geometry requires a valid image size";
            }
            return std::nullopt;
        }
        return ReferenceCameraGeometry(std::move(numericState));
    }

    const camera_models::frame_pinhole::FramePinholeNumericState& ReferenceCameraGeometry::numericState() const noexcept
    {
        return _numericState;
    }

    const camera_core::CameraInstanceId& ReferenceCameraGeometry::instanceId() const noexcept
    {
        return _numericState.instanceId();
    }

    const camera_core::ImageId& ReferenceCameraGeometry::imageId() const noexcept
    {
        return _numericState.imageId();
    }

    const xjw::coordinate_system::CoordinateFrameId& ReferenceCameraGeometry::worldFrame() const noexcept
    {
        return _numericState.worldFrame();
    }

    ReferenceCameraPosition::ReferenceCameraPosition(camera_core::ImageId imageId,
                                                     xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                     std::array<double, 3> center)
        : _imageId(std::move(imageId)), _worldFrame(std::move(worldFrame)), _center(center)
    {
    }

    std::optional<ReferenceCameraPosition> ReferenceCameraPosition::create(camera_core::ImageId imageId,
                                                                           xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                                           std::array<double, 3> center,
                                                                           std::string* error)
    {
        clearError(error);
        if (!finiteCenter(center))
        {
            if (error)
            {
                *error = "reference camera position must contain finite coordinates";
            }
            return std::nullopt;
        }
        try
        {
            return ReferenceCameraPosition(std::move(imageId), std::move(worldFrame), center);
        }
        catch (const std::exception& exception)
        {
            if (error)
            {
                *error = exception.what();
            }
            return std::nullopt;
        }
        catch (...)
        {
            if (error)
            {
                *error = "reference camera position identity is invalid";
            }
            return std::nullopt;
        }
    }

    const camera_core::ImageId& ReferenceCameraPosition::imageId() const noexcept
    {
        return _imageId;
    }

    const xjw::coordinate_system::CoordinateFrameId& ReferenceCameraPosition::worldFrame() const noexcept
    {
        return _worldFrame;
    }

    const std::array<double, 3>& ReferenceCameraPosition::center() const noexcept
    {
        return _center;
    }

    bool validateReferenceCameraGeometryMap(const ReferenceCameraGeometryMap& references, std::string* error)
    {
        clearError(error);
        for (const auto& [key, reference] : references)
        {
            if (!(key == reference.imageId()))
            {
                if (error)
                {
                    *error = "reference camera geometry map key does not match its ImageId";
                }
                return false;
            }
        }
        return true;
    }

    bool validateReferenceCameraPositionMap(const ReferenceCameraPositionMap& references, std::string* error)
    {
        clearError(error);
        for (const auto& [key, reference] : references)
        {
            if (!(key == reference.imageId()))
            {
                if (error)
                {
                    *error = "reference camera position map key does not match its ImageId";
                }
                return false;
            }
            if (!finiteCenter(reference.center()))
            {
                if (error)
                {
                    *error = "reference camera position map contains non-finite coordinates";
                }
                return false;
            }
        }
        return true;
    }

    bool validateReferenceCameraInputs(const std::vector<camera_core::ImageId>& imageIds,
                                       std::size_t imageCount,
                                       const ReferenceCameraGeometryMap& geometries,
                                       const ReferenceCameraPositionMap& positions,
                                       std::string* error)
    {
        clearError(error);
        if (!validateReferenceCameraGeometryMap(geometries, error) ||
            !validateReferenceCameraPositionMap(positions, error))
        {
            return false;
        }
        if (!geometries.empty() || !positions.empty())
        {
            if (imageIds.size() != imageCount)
            {
                if (error)
                {
                    *error = "reference geometry requires one ImageId for every input image";
                }
                return false;
            }
        }

        std::unordered_set<camera_core::ImageId> inputIds;
        inputIds.reserve(imageIds.size());
        for (const auto& imageId : imageIds)
        {
            if (!inputIds.emplace(imageId).second)
            {
                if (error)
                {
                    *error = "input ImageId list contains a duplicate identity: " + imageId.value();
                }
                return false;
            }
        }

        const auto validateMembership = [&inputIds, error](const auto& references, const char* label)
        {
            for (const auto& [imageId, _] : references)
            {
                if (inputIds.find(imageId) == inputIds.end())
                {
                    if (error)
                    {
                        *error =
                            std::string(label) + " contains an ImageId outside the input image set: " + imageId.value();
                    }
                    return false;
                }
            }
            return true;
        };
        return validateMembership(geometries, "reference camera geometry") &&
               validateMembership(positions, "reference camera position");
    }

    std::optional<xjw::coordinate_system::CoordinateFrameId> commonReferenceWorldFrame(
        const ReferenceCameraGeometryMap& geometries, const ReferenceCameraPositionMap& positions, std::string* error)
    {
        clearError(error);
        if (!validateReferenceCameraGeometryMap(geometries, error) ||
            !validateReferenceCameraPositionMap(positions, error))
        {
            return std::nullopt;
        }

        std::optional<xjw::coordinate_system::CoordinateFrameId> frame;
        const auto visit = [&frame, error](const xjw::coordinate_system::CoordinateFrameId& candidate)
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
            if (error)
            {
                *error = "reference camera inputs use mixed coordinate frames";
            }
            return false;
        };

        for (const auto& [_, reference] : geometries)
        {
            if (!visit(reference.worldFrame()))
            {
                return std::nullopt;
            }
        }
        for (const auto& [_, reference] : positions)
        {
            if (!visit(reference.worldFrame()))
            {
                return std::nullopt;
            }
        }
        return frame;
    }

} // namespace xjw::camera_reference

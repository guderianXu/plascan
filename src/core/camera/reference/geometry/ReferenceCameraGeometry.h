#pragma once

#include "camera/core/types/CameraIds.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw::camera_reference
{

    /**
     * A projection-ready reference attached to one canonical image identity.
     *
     * The path of an input image is deliberately absent from this type.  Paths
     * are an I/O locator and may change between project sessions; ImageId and
     * CameraInstanceId are the identities that make a reference safe to use in
     * guided matching and aerial reconstruction. Creation also requires a valid
     * raster size because downstream projectors operate in image coordinates.
     */
    class ReferenceCameraGeometry final
    {
    public:
        static std::optional<ReferenceCameraGeometry>
        create(camera_models::frame_pinhole::FramePinholeNumericState numericState, std::string* error = nullptr);

        const camera_models::frame_pinhole::FramePinholeNumericState& numericState() const noexcept;
        const camera_core::CameraInstanceId& instanceId() const noexcept;
        const camera_core::ImageId& imageId() const noexcept;
        const xjw::coordinate_system::CoordinateFrameId& worldFrame() const noexcept;

    private:
        explicit ReferenceCameraGeometry(camera_models::frame_pinhole::FramePinholeNumericState numericState);

        camera_models::frame_pinhole::FramePinholeNumericState _numericState;
    };

    /**
     * A position-only reference.  It is intentionally separate from camera
     * geometry so callers cannot accidentally use a position prior as a projector.
     */
    class ReferenceCameraPosition final
    {
    public:
        static std::optional<ReferenceCameraPosition> create(camera_core::ImageId imageId,
                                                             xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                             std::array<double, 3> center,
                                                             std::string* error = nullptr);

        const camera_core::ImageId& imageId() const noexcept;
        const xjw::coordinate_system::CoordinateFrameId& worldFrame() const noexcept;
        const std::array<double, 3>& center() const noexcept;

    private:
        ReferenceCameraPosition(camera_core::ImageId imageId,
                                xjw::coordinate_system::CoordinateFrameId worldFrame,
                                std::array<double, 3> center);

        camera_core::ImageId _imageId;
        xjw::coordinate_system::CoordinateFrameId _worldFrame;
        std::array<double, 3> _center;
    };

    using ReferenceCameraGeometryMap = std::unordered_map<camera_core::ImageId, ReferenceCameraGeometry>;
    using ReferenceCameraPositionMap = std::unordered_map<camera_core::ImageId, ReferenceCameraPosition>;

    /** Validate that a reference map is keyed by the same ImageId it stores. */
    bool validateReferenceCameraGeometryMap(const ReferenceCameraGeometryMap& references, std::string* error = nullptr);
    bool validateReferenceCameraPositionMap(const ReferenceCameraPositionMap& references, std::string* error = nullptr);

    /**
     * Validate the ordered ImageId boundary used by matching/aerial workflows.
     *
     * The ordered list is an input locator alignment, while the maps carry the
     * actual reference values.  Every map entry must belong to that list and the
     * list itself must not contain duplicate identities.  An empty reference set
     * may be used without an ImageId list; once a reference is supplied, the list
     * must be complete and have the same length as the image input.
     */
    bool validateReferenceCameraInputs(const std::vector<camera_core::ImageId>& imageIds,
                                       std::size_t imageCount,
                                       const ReferenceCameraGeometryMap& geometries,
                                       const ReferenceCameraPositionMap& positions,
                                       std::string* error = nullptr);

    /**
     * Validate the common world frame of all supplied references.  Empty maps are
     * valid and return no frame; mixed frames are rejected before any geometry is
     * evaluated.
     */
    std::optional<xjw::coordinate_system::CoordinateFrameId>
    commonReferenceWorldFrame(const ReferenceCameraGeometryMap& geometries,
                              const ReferenceCameraPositionMap& positions,
                              std::string* error = nullptr);

} // namespace xjw::camera_reference

#pragma once

#include <placamera/frame_camera.h>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace placamera::reference
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
        static placamera::Result<ReferenceCameraGeometry>
        create(placamera::CameraModelPtr<placamera::FramePinholeModel> model);

        const placamera::FramePinholeModel& model() const noexcept;
        const placamera::CameraModelPtr<placamera::FramePinholeModel>& modelPtr() const noexcept;
        const placamera::CameraInstanceId& instanceId() const noexcept;
        const placamera::ImageId& imageId() const noexcept;
        const placoordinate::CoordinateFrameId& worldFrame() const noexcept;

    private:
        explicit ReferenceCameraGeometry(placamera::CameraModelPtr<placamera::FramePinholeModel> model);

        placamera::CameraModelPtr<placamera::FramePinholeModel> _model;
    };

    /**
     * A position-only reference.  It is intentionally separate from camera
     * geometry so callers cannot accidentally use a position prior as a projector.
     */
    class ReferenceCameraPosition final
    {
    public:
        static placamera::Result<ReferenceCameraPosition>
        create(placamera::ImageId imageId, placoordinate::CoordinateFrameId worldFrame, std::array<double, 3> center);

        const placamera::ImageId& imageId() const noexcept;
        const placoordinate::CoordinateFrameId& worldFrame() const noexcept;
        const std::array<double, 3>& center() const noexcept;

    private:
        ReferenceCameraPosition(placamera::ImageId imageId,
                                placoordinate::CoordinateFrameId worldFrame,
                                std::array<double, 3> center);

        placamera::ImageId _imageId;
        placoordinate::CoordinateFrameId _worldFrame;
        std::array<double, 3> _center;
    };

    using ReferenceCameraGeometryMap = std::unordered_map<placamera::ImageId, ReferenceCameraGeometry>;
    using ReferenceCameraPositionMap = std::unordered_map<placamera::ImageId, ReferenceCameraPosition>;

    /** Validate that a reference map is keyed by the same ImageId it stores. */
    placamera::Result<void> validateReferenceCameraGeometryMap(const ReferenceCameraGeometryMap& references);
    placamera::Result<void> validateReferenceCameraPositionMap(const ReferenceCameraPositionMap& references);

    /**
     * Validate the ordered ImageId boundary used by matching/aerial workflows.
     *
     * The ordered list is an input locator alignment, while the maps carry the
     * actual reference values.  Every map entry must belong to that list and the
     * list itself must not contain duplicate identities.  An empty reference set
     * may be used without an ImageId list; once a reference is supplied, the list
     * must be complete and have the same length as the image input.
     */
    placamera::Result<void> validateReferenceCameraInputs(const std::vector<placamera::ImageId>& imageIds,
                                                          std::size_t imageCount,
                                                          const ReferenceCameraGeometryMap& geometries,
                                                          const ReferenceCameraPositionMap& positions);

    /**
     * Validate the common world frame of all supplied references.  Empty maps are
     * valid and return no frame; mixed frames are rejected before any geometry is
     * evaluated.
     */
    placamera::Result<std::optional<placoordinate::CoordinateFrameId>>
    commonReferenceWorldFrame(const ReferenceCameraGeometryMap& geometries,
                              const ReferenceCameraPositionMap& positions);

} // namespace placamera::reference

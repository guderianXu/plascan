#pragma once

#include "FramePinholeInstance.h"

#include "camera/core/types/CameraIds.h"
#include "camera/core/types/CameraPose.h"

#include <array>
#include <memory>
#include <optional>
#include <string>

namespace xjw::camera_models::frame_pinhole
{

    /**
     * Solver-owned numeric state for one frame-pinhole image instance.
     *
     * This is the state consumed by SfM/BA/MVS arithmetic. It is deliberately
     * independent from project JSON. The state keeps the identity of the typed
     * instance so a solver result can be committed back without reconstructing
     * geometry from flattened metadata.
     */
    class FramePinholeNumericState final
    {
    public:
        using Intrinsics = frame_pinhole::Intrinsics;
        using Distortion = frame_pinhole::Distortion;

        struct Pose
        {
            std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
            std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
            bool depthAxisFlipped = false;
        };

        struct Ray
        {
            std::array<double, 3> origin{{0.0, 0.0, 0.0}};
            std::array<double, 3> direction{{0.0, 0.0, 1.0}};
        };

        struct PairIntersection
        {
            std::array<double, 3> point{{0.0, 0.0, 0.0}};
            double rmsReprojectionPixels = 0.0;
            bool valid = false;
        };

        FramePinholeNumericState()
            : _instanceId("uninitialized-instance"), _imageId("uninitialized-image"), _worldFrame("uninitialized-frame")
        {
        }

        /** Build numeric state directly from a validated typed instance. */
        static bool fromInstance(const std::shared_ptr<const camera_core::CameraInstance>& source,
                                 FramePinholeNumericState* state,
                                 std::string* error = nullptr);
        static bool
        fromInstance(const FramePinholeInstance& source, FramePinholeNumericState* state, std::string* error = nullptr);

        /**
         * Return a new typed instance carrying the current solver state.
         *
         * Promotion requires both a valid numerical state and an explicit
         * identity binding; an unbound state cannot re-enter the project graph.
         */
        std::unique_ptr<FramePinholeInstance>
        toInstance(std::optional<camera_core::CameraDefinitionId> definitionId = std::nullopt) const;

        bool validateNumericalState(std::string* error = nullptr) const noexcept;
        bool isValid() const noexcept;
        /** True only when an explicit typed instance identity and world frame were bound. */
        bool hasBoundIdentity() const noexcept;
        /**
         * Bind an unbound solver state to explicit identity and frame metadata.
         * The caller must already know the frame of the numeric pose; this method
         * never derives one from a path or file name.
         */
        bool bindIdentity(camera_core::CameraInstanceId instanceId,
                          camera_core::ImageId imageId,
                          xjw::coordinate_system::CoordinateFrameId worldFrame,
                          std::string* error = nullptr);
        const camera_core::CameraInstanceId& instanceId() const noexcept;
        const camera_core::ImageId& imageId() const noexcept;
        const xjw::coordinate_system::CoordinateFrameId& worldFrame() const noexcept;
        const std::optional<xjw::coordinate_system::TimeReference>& captureTime() const noexcept;

        std::optional<camera_core::ImageSize> imageSize() const noexcept;
        void setImageSize(camera_core::ImageSize imageSize);

        Intrinsics intrinsics() const noexcept;
        Distortion distortion() const noexcept;
        Pose pose() const noexcept;
        std::array<double, 9> cameraToWorldRotation() const noexcept;
        std::array<double, 9> worldToCameraRotation() const noexcept;
        std::array<double, 3> cameraCenter() const noexcept;
        std::array<double, 3> worldToCameraTranslation() const noexcept;

        double focalX() const noexcept;
        double focalY() const noexcept;
        double principalX() const noexcept;
        double principalY() const noexcept;
        double pixelPitch() const noexcept;
        double focalXMillimeters() const noexcept;
        double focalYMillimeters() const noexcept;
        double principalXMillimeters() const noexcept;
        double principalYMillimeters() const noexcept;
        int uAxisSign() const noexcept;
        int vAxisSign() const noexcept;
        bool depthAxisFlipped() const noexcept;

        bool projectWorldPoint(const double world[3], double pixel[2]) const;
        bool projectWorldPointWithDepth(const double world[3], double pixel[2], double& positiveDepth) const;
        bool projectWorldPointSigned(const double world[3], double pixel[2]) const;
        double positiveDepth(const double world[3]) const;
        bool isPointInFront(const double world[3], double minimumDepth = 1.0e-9) const;
        void worldToCamera(const double world[3], double cameraPoint[3]) const;
        bool undistortPixel(const double pixel[2], double norm[2], int maxIter = 20, double tol = 1.0e-8) const;
        bool unprojectPixel(const double pixel[2], double positiveDepth, double world[3]) const;
        bool rayForPixel(const std::array<double, 2>& pixel, Ray* ray) const;
        static PairIntersection triangulatePair(const FramePinholeNumericState& left,
                                                const std::array<double, 2>& leftPixel,
                                                const FramePinholeNumericState& right,
                                                const std::array<double, 2>& rightPixel);

        void setPose(const std::array<double, 9>& rotation, const std::array<double, 3>& center);
        void setCameraCenter(const std::array<double, 3>& center);
        void setIntrinsics(double focalX, double focalY, double principalX, double principalY);
        void setIntrinsicsMillimeters(double focalXMillimeters,
                                      double focalYMillimeters,
                                      double principalXMillimeters,
                                      double principalYMillimeters,
                                      double pixelPitchMillimeters);
        void setPixelPitch(double pixelPitch);
        void setAxisDirections(int uDirection, int vDirection);
        void setDepthAxisFlipped(bool flipped);
        void setDistortion(const Distortion& distortion);
        void setDistortion(double k1, double k2, double k3, double p1, double p2);
        void applyDeltaPose(const double delta[6]);

        FramePinholeNumericState normalizedForPositiveDepth() const;
        FramePinholeNumericState scaledIntrinsics(double scaleX, double scaleY) const;

    private:
        static bool finiteArray(const double* values, std::size_t count) noexcept;
        static std::array<double, 9> transpose(const std::array<double, 9>& matrix) noexcept;
        static std::array<double, 9> multiply(const std::array<double, 9>& left,
                                              const std::array<double, 9>& right) noexcept;
        static std::array<double, 9> rotationFromDelta(const double delta[6]) noexcept;
        static std::array<double, 3> worldToCamera(const Pose& pose, const double world[3]) noexcept;
        static std::array<double, 3> cameraToWorld(const Pose& pose, const double camera[3]) noexcept;
        void applyDistortion(double x, double y, double* distortedX, double* distortedY) const noexcept;

        camera_core::CameraInstanceId _instanceId;
        camera_core::ImageId _imageId;
        xjw::coordinate_system::CoordinateFrameId _worldFrame;
        std::optional<xjw::coordinate_system::TimeReference> _captureTime;
        std::optional<camera_core::ImageSize> _imageSize;
        Intrinsics _intrinsics;
        Distortion _distortion;
        Pose _pose;
        std::shared_ptr<const FramePinholeDefinition> _definition;
        bool _isValid = false;
        bool _identityBound = false;
    };

    /**
     * Convert a typed instance at a numerical boundary.  Callers must retain the
     * returned state and must not create a second state from project metadata.
     */
    bool makeFramePinholeNumericState(const std::shared_ptr<const camera_core::CameraInstance>& source,
                                      FramePinholeNumericState* state,
                                      std::string* error = nullptr);

} // namespace xjw::camera_models::frame_pinhole

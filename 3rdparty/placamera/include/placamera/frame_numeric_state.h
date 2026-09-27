#pragma once

#include "placamera/frame_camera.h"

#include <array>
#include <optional>

namespace placamera
{

    struct FramePairIntersection
    {
        GroundCoordinate point;
        double triangulationAngleDegrees = 0.0;
        double rayMissDistance = 0.0;
        double firstReprojectionPixels = 0.0;
        double secondReprojectionPixels = 0.0;
        /** Root mean square of the two Euclidean image residuals. */
        double rmsReprojectionPixels = 0.0;
    };

    class FramePinholeNumericState
    {
    public:
        static FramePinholeNumericState fromModel(const FramePinholeModel& model);

        const CameraInstanceId& instanceId() const noexcept;
        const CameraDefinitionId& definitionId() const noexcept;
        const ImageId& imageId() const noexcept;
        const FrameId& groundFrame() const noexcept;
        const ImageSize& imageSize() const noexcept;
        const std::optional<TimeReference>& captureTime() const noexcept;
        const FrameIntrinsics& intrinsics() const noexcept;
        const BrownConradyDistortion& distortion() const noexcept;
        FrameCalibration calibration() const noexcept;
        const std::optional<PrincipalPointDecomposition>& principalPointDecomposition() const noexcept;
        PixelConvention pixelConvention() const noexcept;
        bool depthAxisFlipped() const noexcept;
        FrameProjectionModel projectionModel() const noexcept;
        const SensorMountState& sensorMount() const noexcept;
        const CameraAcquisitionState& acquisition() const noexcept;
        const Pose& pose() const noexcept;
        bool definitionDirty() const noexcept;

        void setPose(Pose pose);
        void setIntrinsics(FrameIntrinsics intrinsics);
        void setDistortion(BrownConradyDistortion distortion);
        void setCalibration(FrameCalibration calibration);
        void applyPoseDelta(const std::array<double, 6>& delta);

        /** Scale intrinsics and the bound image grid together; calibration writeback requires a new definition ID. */
        FramePinholeNumericState scaledIntrinsics(double scaleX, double scaleY) const;

        /** Express the same rays with a positive camera-Z depth axis. */
        FramePinholeNumericState normalizedForPositiveDepth() const;

        /** Signed optical-axis depth, including points behind the camera, for near-plane clipping. */
        EvaluationResult<double> signedDepth(const GroundCoordinate& ground) const;

        /** Project either side of the focal plane; positiveDepth is absent for points behind the camera. */
        EvaluationResult<Projection> groundToImageSigned(const GroundCoordinate& ground,
                                                         const EvaluationOptions& options = {}) const;

        EvaluationResult<Projection> groundToImage(const GroundCoordinate& ground,
                                                   const EvaluationOptions& options = {}) const;
        EvaluationResult<ImagingLocus> imageToImagingLocus(const ImageCoordinate& image,
                                                           const EvaluationOptions& options = {}) const;

        EvaluationResult<GroundCoordinate> imageToGroundAtDepth(const ImageCoordinate& image,
                                                                double positiveDepth,
                                                                const EvaluationOptions& options = {}) const;

        Result<CameraModelPtr<FramePinholeModel>>
        toModel(CameraInstanceId resultInstanceId,
                std::optional<CameraDefinitionId> resultDefinitionId = std::nullopt) const;

        static EvaluationResult<FramePairIntersection> triangulatePair(const FramePinholeNumericState& first,
                                                                       const ImageCoordinate& firstImage,
                                                                       const FramePinholeNumericState& second,
                                                                       const ImageCoordinate& secondImage,
                                                                       const EvaluationOptions& options = {});

    private:
        FramePinholeNumericState(CameraInstanceId instanceId,
                                 CameraDefinitionId definitionId,
                                 ImageId imageId,
                                 FrameId groundFrame,
                                 ImageSize imageSize,
                                 std::optional<TimeReference> captureTime,
                                 FrameIntrinsics intrinsics,
                                 BrownConradyDistortion distortion,
                                 std::optional<PrincipalPointDecomposition> principalPointDecomposition,
                                 PixelConvention pixelConvention,
                                 bool depthAxisFlipped,
                                 FrameProjectionModel projectionModel,
                                 SensorMountState sensorMount,
                                 CameraAcquisitionState acquisition,
                                 Pose pose);

        CameraInstanceId _instanceId;
        CameraDefinitionId _definitionId;
        ImageId _imageId;
        FrameId _groundFrame;
        ImageSize _imageSize;
        std::optional<TimeReference> _captureTime;
        FrameIntrinsics _intrinsics;
        BrownConradyDistortion _distortion;
        std::optional<PrincipalPointDecomposition> _principalPointDecomposition;
        PixelConvention _pixelConvention = PixelConvention::PixelCenter;
        bool _depthAxisFlipped = false;
        FrameProjectionModel _projectionModel = FrameProjectionModel::Perspective;
        SensorMountState _sensorMount;
        CameraAcquisitionState _acquisition;
        Pose _pose;
        bool _definitionDirty = false;
    };

    using CentralCameraNumericState = FramePinholeNumericState;
    using CentralCameraPairIntersection = FramePairIntersection;

} // namespace placamera

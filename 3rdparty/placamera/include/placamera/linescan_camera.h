#pragma once

#include "placamera/calibration.h"
#include "placamera/model.h"

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace placamera
{

    enum class LineScanPixelConvention
    {
        PixelCenter,
        ZeroBased,
    };

    enum class LineScanDistortionModel
    {
        RadialNormalized,
        LroNacFocalPlane,
    };

    struct LineScanDetectorGeometry
    {
        double detectorSampleSumming = 1.0;
        double detectorLineSumming = 1.0;
        double detectorSampleOrigin = 0.0;
        double detectorLineOrigin = 0.0;
        double startingDetectorSample = 0.0;
        double startingDetectorLine = 0.0;
        std::array<double, 3> focalToPixelSamples{{0.0, 0.0, 1.0}};
        std::array<double, 3> focalToPixelLines{{0.0, 1.0, 0.0}};
    };

    struct LineScanOptics
    {
        /** Physical focal length used to construct and recover sensor rays. */
        double focalLengthMillimeters = 0.0;
        /** Legacy detector pitch; also converts complete-calibration pixel offsets before detectorGeometry. */
        double samplePitchMillimeters = 0.0;
        /** Legacy k1-only principal sample. Ignored by completeCalibration. */
        double principalSample = 0.0;
        /** Legacy k1-only distortion. Ignored by completeCalibration. */
        double distortionK1 = 0.0;
        LineScanDistortionModel distortionModel = LineScanDistortionModel::RadialNormalized;
        std::optional<LineScanDetectorGeometry> detectorGeometry;
        /**
         * Full normalized-plane Brown calibration. When detectorGeometry is
         * also present, calibrated pixel offsets from cx/cy are converted to
         * millimetres with samplePitchMillimeters before the detector affine.
         * Its cx/cy use zero-based Metashape coordinates; PixelCenter models
         * add the explicit CSM half-pixel shift at the public boundary.
         */
        std::optional<MetashapeCalibration> completeCalibration;
    };

    struct FocalPlaneCoordinate
    {
        double xMillimeters = 0.0;
        double yMillimeters = 0.0;
    };

    struct LineScanDetectorProjection
    {
        double sample = 0.0;
        double lineResidualPixels = 0.0;
    };

    class LineScanDefinition final : public CameraDefinition
    {
    public:
        static constexpr int ParameterSchemaVersion = 3;

        static std::shared_ptr<const LineScanDefinition>
        create(CameraDefinitionId definitionId,
               FrameId groundFrame,
               LineScanOptics optics,
               LineScanPixelConvention pixelConvention = LineScanPixelConvention::PixelCenter);

        const CameraDefinitionId& definitionId() const noexcept override;
        std::string_view modelType() const noexcept override;
        int parameterSchemaVersion() const noexcept override;
        const FrameId& groundFrame() const noexcept override;
        const LineScanOptics& optics() const noexcept;
        LineScanPixelConvention pixelConvention() const noexcept;
        CapabilitySet capabilities() const noexcept override;

    private:
        LineScanDefinition(CameraDefinitionId definitionId,
                           FrameId groundFrame,
                           LineScanOptics optics,
                           LineScanPixelConvention pixelConvention);

        static void validate(const LineScanOptics& optics);

        CameraDefinitionId _definitionId;
        FrameId _groundFrame;
        LineScanOptics _optics;
        LineScanPixelConvention _pixelConvention;
    };

    EvaluationResult<FocalPlaneCoordinate> lineScanPixelToUndistortedFocal(const LineScanDefinition& definition,
                                                                           double sample,
                                                                           const EvaluationOptions& options = {});

    EvaluationResult<LineScanDetectorProjection> lineScanUndistortedFocalToPixel(const LineScanDefinition& definition,
                                                                                 const FocalPlaneCoordinate& focal,
                                                                                 const EvaluationOptions& options = {});

    struct TrajectoryKnotConstraints
    {
        bool positionFixed = true;
        bool rotationFixed = true;
        std::optional<Vector3> positionSigmaMeters;
        std::optional<Vector3> rotationSigmaRadians;
    };

    struct TrajectorySample
    {
        TimeReference time;
        Vector3 center{{0.0, 0.0, 0.0}};
        RotationMatrix cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        TrajectoryKnotConstraints constraints;

        TrajectorySample() = default;
        TrajectorySample(TimeReference sampleTime,
                         Vector3 sampleCenter = {{0.0, 0.0, 0.0}},
                         RotationMatrix sampleCameraToWorldRotation = {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}},
                         TrajectoryKnotConstraints sampleConstraints = {});
    };

    struct TranslationalStateSample
    {
        TimeReference time;
        Vector3 positionMeters{{0.0, 0.0, 0.0}};
        Vector3 velocityMetersPerSecond{{0.0, 0.0, 0.0}};
    };

    struct QuaternionTrajectorySample
    {
        TimeReference time;
        std::array<double, 4> scalarFirst{{1.0, 0.0, 0.0, 0.0}};
    };

    struct FrameRotationTrajectory
    {
        RotationMatrix constantRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::vector<QuaternionTrajectorySample> samples;
    };

    struct FrameComposedTrajectory
    {
        std::vector<TranslationalStateSample> inertialStates;
        FrameRotationTrajectory inertialToWorld;
        FrameRotationTrajectory inertialToSensor;
    };

    class LineScanTrajectory
    {
    public:
        static LineScanTrajectory create(std::vector<TrajectorySample> samples);
        static LineScanTrajectory createFrameComposed(FrameComposedTrajectory trajectory);

        const std::vector<TrajectorySample>& samples() const noexcept;
        const FrameComposedTrajectory* frameComposed() const noexcept;
        TimeScale timeScale() const noexcept;
        EvaluationResult<Pose> poseAt(TimeReference time, const FrameId& frame) const;

    private:
        explicit LineScanTrajectory(std::vector<TrajectorySample> samples);
        explicit LineScanTrajectory(FrameComposedTrajectory trajectory);

        std::vector<TrajectorySample> _samples;
        std::optional<FrameComposedTrajectory> _frameComposed;
        TimeScale _timeScale = TimeScale::Tdb;
    };

    struct LineRateSegment
    {
        double startLine = 0.5;
        double startTimeSeconds = 0.0;
        double secondsPerLine = 0.0;
    };

    struct LineTiming
    {
        double lineZero = 0.5;
        double startTimeSeconds = 0.0;
        double secondsPerLine = 0.0;
        TimeScale timeScale = TimeScale::Tdb;
        std::vector<LineRateSegment> segments;
    };

    struct LineScanTrajectoryBias
    {
        Vector3 translationMeters{{0.0, 0.0, 0.0}};
        Vector3 rotationVectorRadians{{0.0, 0.0, 0.0}};
        double timeOffsetSeconds = 0.0;
    };

    struct LineScanTimeOffsetPrior
    {
        double meanSeconds = 0.0;
        double sigmaSeconds = 0.0;
    };

    LineScanTrajectoryBias applyLineScanTrajectoryBiasUpdate(const LineScanTrajectoryBias& current,
                                                             const std::array<double, 7>& delta);

    struct LineScanProjectionDetails
    {
        Projection projection;
        double lineResidualPixels = 0.0;
        FocalPlaneCoordinate undistortedFocal;
    };

    class LineScanModel final : public RasterModel
    {
    public:
        static constexpr int InstanceSchemaVersion = 2;

        static LineScanModel create(CameraInstanceId instanceId,
                                    ImageId imageId,
                                    std::shared_ptr<const LineScanDefinition> definition,
                                    ImageSize imageSize,
                                    LineScanTrajectory trajectory,
                                    LineTiming timing,
                                    LineScanTrajectoryBias bias = {},
                                    std::optional<TimeReference> captureTime = std::nullopt,
                                    std::optional<LineScanTimeOffsetPrior> timeOffsetPrior = std::nullopt);

        const CameraInstanceId& instanceId() const noexcept override;
        const CameraDefinition& definition() const noexcept override;
        const CameraDefinitionId& definitionId() const noexcept override;
        const ImageId& imageId() const noexcept override;
        std::string_view modelType() const noexcept override;
        int parameterSchemaVersion() const noexcept override;
        const FrameId& groundFrame() const noexcept override;
        const ImageSize& imageSize() const noexcept override;
        const std::optional<TimeReference>& captureTime() const noexcept override;
        CapabilitySet capabilities() const noexcept override;

        const LineScanDefinition& lineScanDefinition() const noexcept;
        const LineScanTrajectory& trajectory() const noexcept;
        const LineTiming& lineTiming() const noexcept;
        const LineScanTrajectoryBias& trajectoryBias() const noexcept;
        const std::optional<LineScanTimeOffsetPrior>& timeOffsetPrior() const noexcept;

        EvaluationResult<TimeReference> timeForLine(double line) const;
        EvaluationResult<double> lineForTime(TimeReference time) const;

        EvaluationResult<Projection> groundToImage(const GroundCoordinate& ground,
                                                   const EvaluationOptions& options = {}) const override;

        EvaluationResult<LineScanProjectionDetails>
        projectAtLine(const GroundCoordinate& ground, double line, const EvaluationOptions& options = {}) const;
        EvaluationResult<LineScanProjectionDetails> projectAtLine(const GroundCoordinate& ground,
                                                                  double line,
                                                                  const LineScanTrajectoryBias& bias,
                                                                  const EvaluationOptions& options = {}) const;

        EvaluationResult<ImagingLocus> imageToImagingLocus(const ImageCoordinate& image,
                                                           const EvaluationOptions& options = {}) const override;

        OptimizationLayout optimizationLayout() const override;
        Result<RasterModelPtr> withOptimizationUpdate(const OptimizationUpdate& update) const override;

        LineScanModel withTrajectoryBias(CameraInstanceId instanceId, LineScanTrajectoryBias bias) const;

    private:
        LineScanModel(CameraInstanceId instanceId,
                      ImageId imageId,
                      std::shared_ptr<const LineScanDefinition> definition,
                      ImageSize imageSize,
                      LineScanTrajectory trajectory,
                      LineTiming timing,
                      LineScanTrajectoryBias bias,
                      std::optional<TimeReference> captureTime,
                      std::optional<LineScanTimeOffsetPrior> timeOffsetPrior);

        CameraInstanceId _instanceId;
        ImageId _imageId;
        std::shared_ptr<const LineScanDefinition> _definition;
        ImageSize _imageSize;
        LineScanTrajectory _trajectory;
        LineTiming _timing;
        LineScanTrajectoryBias _bias;
        std::optional<TimeReference> _captureTime;
        std::optional<LineScanTimeOffsetPrior> _timeOffsetPrior;
    };

} // namespace placamera

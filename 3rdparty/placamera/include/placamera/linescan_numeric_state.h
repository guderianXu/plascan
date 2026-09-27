#pragma once

#include "placamera/linescan_camera.h"

#include <span>

namespace placamera
{

    /** Explicit opt-in mask for a complete Metashape detector calibration. */
    struct LineScanCalibrationOptimizationMask
    {
        bool f = false;
        bool cx = false;
        bool cy = false;
        bool b1 = false;
        bool b2 = false;
        bool k1 = false;
        bool k2 = false;
        bool k3 = false;
        bool k4 = false;
        bool p1 = false;
        bool p2 = false;
        bool p3 = false;
        bool p4 = false;

        bool any() const noexcept;
    };

    /** Explicit opt-in mask for detector and optical definition parameters. */
    struct LineScanDetectorOptimizationMask
    {
        bool focalLength = false;
        bool distortionK1 = false;
        bool samplePitch = false;
        bool principalSample = false;
        bool detectorSampleSumming = false;
        bool detectorLineSumming = false;
        bool detectorSampleOrigin = false;
        bool detectorLineOrigin = false;
        bool startingDetectorSample = false;
        bool startingDetectorLine = false;
        bool focalToPixelSamples = false;
        bool focalToPixelLines = false;

        bool anyDetectorGeometryParameter() const noexcept;
        bool any() const noexcept;
    };

    struct LineScanOptimizationSelection
    {
        bool knotPositions = true;
        bool knotRotations = true;
        bool globalTranslation = false;
        bool globalRotation = false;
        bool timeOffset = true;
        LineScanCalibrationOptimizationMask calibration;
        LineScanDetectorOptimizationMask detector;

        /** Non-negative least-squares weights; residuals are multiplied by sqrt(weight). */
        double positionSecondDifferenceWeight = 0.0;
        double rotationSecondDifferenceWeight = 0.0;
    };

    /**
     * Mutable, solver-owned state for a direct-sample line-scan model.
     *
     * Construction allocates the trajectory and parameter layout once. Successful projection calls do not create
     * immutable camera models and do not resize internal storage.
     */
    class LineScanNumericState
    {
    public:
        static Result<LineScanNumericState> fromModel(const LineScanModel& model,
                                                      LineScanOptimizationSelection selection = {});

        const CameraInstanceId& instanceId() const noexcept;
        const CameraDefinitionId& definitionId() const noexcept;
        const ImageId& imageId() const noexcept;
        const FrameId& groundFrame() const noexcept;
        const ImageSize& imageSize() const noexcept;
        const std::optional<TimeReference>& captureTime() const noexcept;
        const std::vector<TrajectorySample>& trajectorySamples() const noexcept;
        const LineTiming& lineTiming() const noexcept;
        const LineScanTrajectoryBias& trajectoryBias() const noexcept;
        const std::optional<LineScanTimeOffsetPrior>& timeOffsetPrior() const noexcept;
        const LineScanOptics& optics() const noexcept;
        LineScanPixelConvention pixelConvention() const noexcept;
        const LineScanOptimizationSelection& selection() const noexcept;
        const OptimizationLayout& optimizationLayout() const noexcept;
        bool definitionDirty() const noexcept;

        Result<void> applyOptimizationDelta(std::span<const double> delta);

        std::size_t regularizationResidualCount() const noexcept;
        Result<void> writeRegularizationResiduals(std::span<double> residuals) const;

        EvaluationResult<TimeReference> timeForLine(double line) const;
        EvaluationResult<LineScanProjectionDetails>
        projectAtLine(const GroundCoordinate& ground, double line, const EvaluationOptions& options = {}) const;
        EvaluationResult<Projection> groundToImage(const GroundCoordinate& ground,
                                                   const EvaluationOptions& options = {}) const;
        EvaluationResult<ImagingLocus> imageToImagingLocus(const ImageCoordinate& image,
                                                           const EvaluationOptions& options = {}) const;

        Result<CameraModelPtr<LineScanModel>>
        toModel(CameraInstanceId resultInstanceId,
                std::optional<CameraDefinitionId> resultDefinitionId = std::nullopt) const;

    private:
        LineScanNumericState(CameraInstanceId instanceId,
                             CameraDefinitionId definitionId,
                             ImageId imageId,
                             FrameId groundFrame,
                             ImageSize imageSize,
                             std::optional<TimeReference> captureTime,
                             std::vector<TrajectorySample> samples,
                             LineTiming timing,
                             LineScanTrajectoryBias bias,
                             std::optional<LineScanTimeOffsetPrior> timeOffsetPrior,
                             LineScanOptics optics,
                             LineScanPixelConvention pixelConvention,
                             LineScanOptimizationSelection selection,
                             OptimizationLayout layout);

        CameraInstanceId _instanceId;
        CameraDefinitionId _definitionId;
        ImageId _imageId;
        FrameId _groundFrame;
        ImageSize _imageSize;
        std::optional<TimeReference> _captureTime;
        std::vector<TrajectorySample> _nominalSamples;
        std::vector<TrajectorySample> _samples;
        std::vector<Vector3> _positionDeltas;
        std::vector<Vector3> _rotationDeltas;
        LineTiming _timing;
        LineScanTrajectoryBias _bias;
        std::optional<LineScanTimeOffsetPrior> _timeOffsetPrior;
        LineScanOptics _optics;
        LineScanPixelConvention _pixelConvention = LineScanPixelConvention::PixelCenter;
        LineScanOptimizationSelection _selection;
        OptimizationLayout _layout;
        bool _definitionDirty = false;
    };

} // namespace placamera

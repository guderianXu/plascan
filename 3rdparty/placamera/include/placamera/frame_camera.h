#pragma once

#include "placamera/calibration.h"
#include "placamera/camera_topology.h"
#include "placamera/model.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace placamera
{

    enum class PixelConvention
    {
        PixelCenter,
        PixelCorner,
    };

    enum class FrameProjectionModel
    {
        Perspective,
        Fisheye,
        EquidistantFisheye,
        EquisolidFisheye,
        Spherical,
        Cylindrical,
    };

    std::string_view frameProjectionModelName(FrameProjectionModel model) noexcept;

    /**
     * Naming used by the two common Brown tangential coefficient conventions.
     *
     * OpenCv keeps the historical PlaCamera five-coefficient API. Metashape
     * uses p1 for the x-directed diagonal term and p2 for the y-directed term;
     * p3/p4 scale that complete tangential field by 1 + p3*r^2 + p4*r^4.
     */
    enum class BrownTangentialConvention
    {
        OpenCv,
        Metashape,
    };

    struct FrameIntrinsics
    {
        double focalX = 0.0;
        double focalY = 0.0;
        double principalX = 0.0;
        double principalY = 0.0;
        double pixelPitch = 1.0;
        int uAxisSign = 1;
        int vAxisSign = 1;
        /** Pixel affine term applied as sample = fx*x + skew*y + cx. */
        double skew = 0.0;
    };

    struct BrownConradyDistortion
    {
        double radialK1 = 0.0;
        double radialK2 = 0.0;
        double radialK3 = 0.0;
        double tangentialP1 = 0.0;
        double tangentialP2 = 0.0;
        double radialK4 = 0.0;
        double tangentialP3 = 0.0;
        double tangentialP4 = 0.0;
        BrownTangentialConvention tangentialConvention = BrownTangentialConvention::OpenCv;
    };

    enum class FrameOptimizationParameter : std::size_t
    {
        PoseRotation,
        PoseTranslation,
        F,
        Cx,
        Cy,
        B1,
        B2,
        K1,
        K2,
        K3,
        K4,
        P1,
        P2,
        P3,
        P4,
        RollingRotation,
        RollingTranslation,
        Count,
    };

    /** Explicit parameter mask for the extended frame optimization layout. */
    class FrameOptimizationSelection
    {
    public:
        static FrameOptimizationSelection none() noexcept;
        static FrameOptimizationSelection all() noexcept;

        bool enabled(FrameOptimizationParameter parameter) const noexcept;
        FrameOptimizationSelection& set(FrameOptimizationParameter parameter, bool enabled = true) noexcept;

    private:
        std::array<bool, static_cast<std::size_t>(FrameOptimizationParameter::Count)> _enabled{};
    };

    class FramePinholeDefinition final : public CameraDefinition
    {
    public:
        static constexpr int ParameterSchemaVersion = 4;

        static std::shared_ptr<const FramePinholeDefinition>
        create(CameraDefinitionId definitionId,
               FrameIntrinsics intrinsics,
               BrownConradyDistortion distortion,
               PixelConvention pixelConvention,
               FrameId groundFrame,
               bool depthAxisFlipped = false,
               FrameProjectionModel projectionModel = FrameProjectionModel::Perspective,
               SensorMountState sensorMount = {});

        static std::shared_ptr<const FramePinholeDefinition>
        create(CameraDefinitionId definitionId,
               FrameCalibration calibration,
               PixelConvention pixelConvention,
               FrameId groundFrame,
               bool depthAxisFlipped = false,
               double pixelPitch = 1.0,
               int uAxisSign = 1,
               int vAxisSign = 1,
               FrameProjectionModel projectionModel = FrameProjectionModel::Perspective,
               SensorMountState sensorMount = {});

        const CameraDefinitionId& definitionId() const noexcept override;
        std::string_view modelType() const noexcept override;
        int parameterSchemaVersion() const noexcept override;
        const FrameId& groundFrame() const noexcept override;
        const FrameIntrinsics& intrinsics() const noexcept;
        const BrownConradyDistortion& distortion() const noexcept;
        FrameCalibration calibration() const noexcept;
        const std::optional<PrincipalPointDecomposition>& principalPointDecomposition() const noexcept;
        PixelConvention pixelConvention() const noexcept;
        bool depthAxisFlipped() const noexcept;
        FrameProjectionModel projectionModel() const noexcept;
        const SensorMountState& sensorMount() const noexcept;
        CapabilitySet capabilities() const noexcept override;

        std::array<double, 3> positiveDepthAxisSigns() const noexcept;

        /** Invert pixel calibration without requiring an image instance or camera pose. */
        EvaluationResult<std::array<double, 2>> undistortPixel(const ImageCoordinate& image,
                                                               const EvaluationOptions& options = {}) const;

        std::shared_ptr<const FramePinholeDefinition>
        scaledIntrinsics(CameraDefinitionId definitionId, double scaleX, double scaleY) const;

        std::shared_ptr<const FramePinholeDefinition> normalizedForPositiveDepth(CameraDefinitionId definitionId) const;

    private:
        FramePinholeDefinition(CameraDefinitionId definitionId,
                               FrameIntrinsics intrinsics,
                               BrownConradyDistortion distortion,
                               PixelConvention pixelConvention,
                               FrameId groundFrame,
                               bool depthAxisFlipped,
                               FrameProjectionModel projectionModel,
                               SensorMountState sensorMount,
                               std::optional<PrincipalPointDecomposition> principalPointDecomposition);

        static void validate(const FrameIntrinsics& intrinsics, const BrownConradyDistortion& distortion);

        CameraDefinitionId _definitionId;
        FrameIntrinsics _intrinsics;
        BrownConradyDistortion _distortion;
        PixelConvention _pixelConvention;
        FrameId _groundFrame;
        bool _depthAxisFlipped = false;
        FrameProjectionModel _projectionModel = FrameProjectionModel::Perspective;
        SensorMountState _sensorMount;
        std::optional<PrincipalPointDecomposition> _principalPointDecomposition;
    };

    /** Calibration and pose without an application image identity. */
    struct FramePinholeGeometry
    {
        std::shared_ptr<const FramePinholeDefinition> definition;
        Pose pose;
    };

    /** Application identity and raster metadata used to bind reusable geometry. */
    struct FramePinholeBinding
    {
        CameraInstanceId instanceId;
        ImageId imageId;
        ImageSize imageSize;
        std::optional<TimeReference> captureTime;
        CameraAcquisitionState acquisition;
    };

    class FramePinholeModel final : public RasterModel
    {
    public:
        static constexpr int InstanceSchemaVersion = 2;

        static FramePinholeModel create(CameraInstanceId instanceId,
                                        ImageId imageId,
                                        std::shared_ptr<const FramePinholeDefinition> definition,
                                        ImageSize imageSize,
                                        Pose pose,
                                        std::optional<TimeReference> captureTime = std::nullopt,
                                        CameraAcquisitionState acquisition = {});

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

        const FramePinholeDefinition& pinholeDefinition() const noexcept;
        const Pose& pose() const noexcept;
        const CameraAcquisitionState& acquisition() const noexcept;

        /** Signed optical-axis depth, including points behind the camera, for near-plane clipping. */
        EvaluationResult<double> signedDepth(const GroundCoordinate& ground) const;

        /** Project either side of the focal plane; positiveDepth is absent for points behind the camera. */
        EvaluationResult<Projection> groundToImageSigned(const GroundCoordinate& ground,
                                                         const EvaluationOptions& options = {}) const;

        EvaluationResult<Projection> groundToImage(const GroundCoordinate& ground,
                                                   const EvaluationOptions& options = {}) const override;

        EvaluationResult<ImagingLocus> imageToImagingLocus(const ImageCoordinate& image,
                                                           const EvaluationOptions& options = {}) const override;

        EvaluationResult<GroundCoordinate> imageToGroundAtDepth(const ImageCoordinate& image,
                                                                double positiveDepth,
                                                                const EvaluationOptions& options = {}) const;

        OptimizationLayout optimizationLayout() const override;
        Result<RasterModelPtr> withOptimizationUpdate(const OptimizationUpdate& update) const override;

        /** Extended Metashape-compatible layout containing only selected parameters. */
        OptimizationLayout optimizationLayout(const FrameOptimizationSelection& selection) const;
        Result<RasterModelPtr> withOptimizationUpdate(const OptimizationUpdate& update,
                                                      const FrameOptimizationSelection& selection) const;

        FramePinholeModel withPose(CameraInstanceId instanceId, Pose pose) const;

        /** Rebind the raster grid while preserving the camera instance identity and calibration. */
        FramePinholeModel withImageSize(ImageSize imageSize) const;

        FramePinholeModel normalizedForPositiveDepth(CameraDefinitionId definitionId,
                                                     CameraInstanceId instanceId) const;

    private:
        FramePinholeModel(CameraInstanceId instanceId,
                          ImageId imageId,
                          std::shared_ptr<const FramePinholeDefinition> definition,
                          ImageSize imageSize,
                          Pose pose,
                          std::optional<TimeReference> captureTime,
                          CameraAcquisitionState acquisition);

        CameraInstanceId _instanceId;
        ImageId _imageId;
        std::shared_ptr<const FramePinholeDefinition> _definition;
        ImageSize _imageSize;
        Pose _pose;
        std::optional<TimeReference> _captureTime;
        CameraAcquisitionState _acquisition;
    };

    /** Preferred projection-neutral names; the original names remain source compatible. */
    using CentralCameraDefinition = FramePinholeDefinition;
    using CentralCameraModel = FramePinholeModel;
    using CentralCameraGeometry = FramePinholeGeometry;
    using CentralCameraBinding = FramePinholeBinding;

    /** Bind reusable central-camera geometry to one immutable image model. */
    Result<CameraModelPtr<CentralCameraModel>> bindCentralCamera(CentralCameraGeometry geometry,
                                                                 CentralCameraBinding binding);

    /** Legacy perspective-oriented spelling retained for source compatibility. */
    Result<CameraModelPtr<FramePinholeModel>> bindFramePinhole(FramePinholeGeometry geometry,
                                                               FramePinholeBinding binding);

} // namespace placamera

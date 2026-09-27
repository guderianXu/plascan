#include "placamera/frame_camera.h"

#include "internal/frame_camera_math.h"

#include <cmath>
#include <utility>

namespace placamera
{

    namespace
    {

        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

    } // namespace

    std::string_view frameProjectionModelName(FrameProjectionModel model) noexcept
    {
        switch (model)
        {
        case FrameProjectionModel::Perspective:
            return "frame_pinhole";
        case FrameProjectionModel::Fisheye:
            return "frame_fisheye";
        case FrameProjectionModel::EquidistantFisheye:
            return "frame_equidistant_fisheye";
        case FrameProjectionModel::EquisolidFisheye:
            return "frame_equisolid_fisheye";
        case FrameProjectionModel::Spherical:
            return "frame_spherical";
        case FrameProjectionModel::Cylindrical:
            return "frame_cylindrical";
        }
        return "invalid_frame_projection";
    }

    std::shared_ptr<const FramePinholeDefinition> FramePinholeDefinition::create(CameraDefinitionId definitionId,
                                                                                 FrameIntrinsics intrinsics,
                                                                                 BrownConradyDistortion distortion,
                                                                                 PixelConvention pixelConvention,
                                                                                 FrameId groundFrame,
                                                                                 bool depthAxisFlipped,
                                                                                 FrameProjectionModel projectionModel,
                                                                                 SensorMountState sensorMount)
    {
        validate(intrinsics, distortion);
        if (frameProjectionModelName(projectionModel) == "invalid_frame_projection")
        {
            throw CameraValidationError(CameraErrorCode::UnsupportedModel, "frame projection model is invalid");
        }
        if (projectionModel != FrameProjectionModel::Perspective && depthAxisFlipped)
        {
            throw CameraValidationError(CameraErrorCode::InvalidArgument,
                                        "non-perspective projections require the canonical positive depth axis");
        }
        const auto valid_mount = validateSensorMount(sensorMount, &definitionId);
        if (!valid_mount)
        {
            throw CameraValidationError(valid_mount.errorCode(), valid_mount.message());
        }
        if ((projectionModel == FrameProjectionModel::Spherical ||
             projectionModel == FrameProjectionModel::Cylindrical) &&
            (distortion.radialK1 != 0.0 || distortion.radialK2 != 0.0 || distortion.radialK3 != 0.0 ||
             distortion.radialK4 != 0.0 || distortion.tangentialP1 != 0.0 || distortion.tangentialP2 != 0.0 ||
             distortion.tangentialP3 != 0.0 || distortion.tangentialP4 != 0.0))
        {
            throw CameraValidationError(CameraErrorCode::InvalidDistortion,
                                        "spherical and cylindrical projections do not consume Brown distortion");
        }
        return std::shared_ptr<const FramePinholeDefinition>(new FramePinholeDefinition(std::move(definitionId),
                                                                                        intrinsics,
                                                                                        distortion,
                                                                                        pixelConvention,
                                                                                        std::move(groundFrame),
                                                                                        depthAxisFlipped,
                                                                                        projectionModel,
                                                                                        std::move(sensorMount),
                                                                                        std::nullopt));
    }

    std::shared_ptr<const FramePinholeDefinition> FramePinholeDefinition::create(CameraDefinitionId definitionId,
                                                                                 FrameCalibration calibration,
                                                                                 PixelConvention pixelConvention,
                                                                                 FrameId groundFrame,
                                                                                 bool depthAxisFlipped,
                                                                                 double pixelPitch,
                                                                                 int uAxisSign,
                                                                                 int vAxisSign,
                                                                                 FrameProjectionModel projectionModel,
                                                                                 SensorMountState sensorMount)
    {
        if (calibration.principalPointDecomposition)
        {
            const PrincipalPointDecomposition& principal = *calibration.principalPointDecomposition;
            if (!finite(principal.imageCenterX) || !finite(principal.imageCenterY) || !finite(principal.cxOffset) ||
                !finite(principal.cyOffset))
            {
                throw CameraValidationError(CameraErrorCode::InvalidIntrinsics,
                                            "principal-point decomposition must contain finite values");
            }
        }
        FrameIntrinsics intrinsics;
        intrinsics.focalX = calibration.f + calibration.b1;
        intrinsics.focalY = calibration.f;
        intrinsics.principalX = calibration.cx;
        intrinsics.principalY = calibration.cy;
        intrinsics.pixelPitch = pixelPitch;
        intrinsics.uAxisSign = uAxisSign;
        intrinsics.vAxisSign = vAxisSign;
        intrinsics.skew = calibration.b2;

        BrownConradyDistortion distortion;
        distortion.radialK1 = calibration.k1;
        distortion.radialK2 = calibration.k2;
        distortion.radialK3 = calibration.k3;
        distortion.tangentialP1 = calibration.p1;
        distortion.tangentialP2 = calibration.p2;
        distortion.radialK4 = calibration.k4;
        distortion.tangentialP3 = calibration.p3;
        distortion.tangentialP4 = calibration.p4;
        distortion.tangentialConvention = BrownTangentialConvention::Metashape;
        const auto validated = create(definitionId,
                                      intrinsics,
                                      distortion,
                                      pixelConvention,
                                      groundFrame,
                                      depthAxisFlipped,
                                      projectionModel,
                                      sensorMount);
        return std::shared_ptr<const FramePinholeDefinition>(
            new FramePinholeDefinition(std::move(definitionId),
                                       validated->intrinsics(),
                                       validated->distortion(),
                                       pixelConvention,
                                       std::move(groundFrame),
                                       depthAxisFlipped,
                                       projectionModel,
                                       std::move(sensorMount),
                                       calibration.principalPointDecomposition));
    }

    const CameraDefinitionId& FramePinholeDefinition::definitionId() const noexcept
    {
        return _definitionId;
    }

    std::string_view FramePinholeDefinition::modelType() const noexcept
    {
        return frameProjectionModelName(_projectionModel);
    }

    int FramePinholeDefinition::parameterSchemaVersion() const noexcept
    {
        return ParameterSchemaVersion;
    }

    const FrameId& FramePinholeDefinition::groundFrame() const noexcept
    {
        return _groundFrame;
    }

    const FrameIntrinsics& FramePinholeDefinition::intrinsics() const noexcept
    {
        return _intrinsics;
    }

    const BrownConradyDistortion& FramePinholeDefinition::distortion() const noexcept
    {
        return _distortion;
    }

    FrameCalibration FramePinholeDefinition::calibration() const noexcept
    {
        FrameCalibration result;
        result.f = _intrinsics.focalY;
        result.cx = _intrinsics.principalX;
        result.cy = _intrinsics.principalY;
        result.b1 = _intrinsics.focalX - _intrinsics.focalY;
        result.b2 = _intrinsics.skew;
        result.k1 = _distortion.radialK1;
        result.k2 = _distortion.radialK2;
        result.k3 = _distortion.radialK3;
        result.k4 = _distortion.radialK4;
        if (_distortion.tangentialConvention == BrownTangentialConvention::Metashape)
        {
            result.p1 = _distortion.tangentialP1;
            result.p2 = _distortion.tangentialP2;
            result.p3 = _distortion.tangentialP3;
            result.p4 = _distortion.tangentialP4;
        }
        else
        {
            result.p1 = _distortion.tangentialP2;
            result.p2 = _distortion.tangentialP1;
        }
        result.principalPointDecomposition = _principalPointDecomposition;
        return result;
    }

    const std::optional<PrincipalPointDecomposition>&
    FramePinholeDefinition::principalPointDecomposition() const noexcept
    {
        return _principalPointDecomposition;
    }

    PixelConvention FramePinholeDefinition::pixelConvention() const noexcept
    {
        return _pixelConvention;
    }

    bool FramePinholeDefinition::depthAxisFlipped() const noexcept
    {
        return _depthAxisFlipped;
    }

    FrameProjectionModel FramePinholeDefinition::projectionModel() const noexcept
    {
        return _projectionModel;
    }

    const SensorMountState& FramePinholeDefinition::sensorMount() const noexcept
    {
        return _sensorMount;
    }

    CapabilitySet FramePinholeDefinition::capabilities() const noexcept
    {
        return {CapabilityKind::Projection,
                CapabilityKind::InverseProjection,
                CapabilityKind::ImagingLocus,
                CapabilityKind::StaticPose,
                CapabilityKind::ImageCorrection,
                CapabilityKind::Optimization};
    }

    std::array<double, 3> FramePinholeDefinition::positiveDepthAxisSigns() const noexcept
    {
        const double z_sign = _depthAxisFlipped ? -1.0 : 1.0;
        std::array<double, 3> signs{z_sign * static_cast<double>(_intrinsics.uAxisSign),
                                    z_sign * static_cast<double>(_intrinsics.vAxisSign),
                                    z_sign};
        if (signs[0] * signs[1] * signs[2] < 0.0)
        {
            signs[0] = -signs[0];
        }
        return signs;
    }

    EvaluationResult<std::array<double, 2>>
    FramePinholeDefinition::undistortPixel(const ImageCoordinate& image, const EvaluationOptions& options) const
    {
        if (options.requireInsideImage)
        {
            return EvaluationResult<std::array<double, 2>>::failure(
                CameraErrorCode::InvalidArgument,
                "definition-only undistortion cannot check image bounds without an image instance");
        }
        return internal::undistort(*this, image, options);
    }

    std::shared_ptr<const FramePinholeDefinition>
    FramePinholeDefinition::scaledIntrinsics(CameraDefinitionId definitionId, double scaleX, double scaleY) const
    {
        if (!finite(scaleX) || !finite(scaleY) || scaleX <= 0.0 || scaleY <= 0.0)
        {
            throw CameraValidationError(CameraErrorCode::InvalidIntrinsics,
                                        "pinhole intrinsic scale must be finite and positive");
        }

        FrameIntrinsics scaled = _intrinsics;
        scaled.focalX *= scaleX;
        scaled.focalY *= scaleY;
        scaled.skew *= scaleX;
        if (_pixelConvention == PixelConvention::PixelCenter)
        {
            scaled.principalX = (scaled.principalX + 0.5) * scaleX - 0.5;
            scaled.principalY = (scaled.principalY + 0.5) * scaleY - 0.5;
        }
        else
        {
            scaled.principalX *= scaleX;
            scaled.principalY *= scaleY;
        }
        std::optional<PrincipalPointDecomposition> decomposition = _principalPointDecomposition;
        if (decomposition)
        {
            const bool exact_x = decomposition->imageCenterX + decomposition->cxOffset == _intrinsics.principalX;
            const bool exact_y = decomposition->imageCenterY + decomposition->cyOffset == _intrinsics.principalY;
            if (_pixelConvention == PixelConvention::PixelCenter)
            {
                decomposition->imageCenterX = (decomposition->imageCenterX + 0.5) * scaleX - 0.5;
                decomposition->imageCenterY = (decomposition->imageCenterY + 0.5) * scaleY - 0.5;
            }
            else
            {
                decomposition->imageCenterX *= scaleX;
                decomposition->imageCenterY *= scaleY;
            }
            decomposition->cxOffset *= scaleX;
            decomposition->cyOffset *= scaleY;
            if (exact_x)
            {
                scaled.principalX = decomposition->imageCenterX + decomposition->cxOffset;
            }
            if (exact_y)
            {
                scaled.principalY = decomposition->imageCenterY + decomposition->cyOffset;
            }
        }
        return std::shared_ptr<const FramePinholeDefinition>(new FramePinholeDefinition(std::move(definitionId),
                                                                                        scaled,
                                                                                        _distortion,
                                                                                        _pixelConvention,
                                                                                        _groundFrame,
                                                                                        _depthAxisFlipped,
                                                                                        _projectionModel,
                                                                                        _sensorMount,
                                                                                        decomposition));
    }

    std::shared_ptr<const FramePinholeDefinition>
    FramePinholeDefinition::normalizedForPositiveDepth(CameraDefinitionId definitionId) const
    {
        FrameIntrinsics normalized = _intrinsics;
        BrownConradyDistortion distortion = _distortion;
        const auto signs = positiveDepthAxisSigns();
        const double x_ratio_sign = signs[0] / signs[2];
        const double y_ratio_sign = signs[1] / signs[2];
        const int u_sign = normalized.uAxisSign < 0 ? -1 : 1;
        const int v_sign = normalized.vAxisSign < 0 ? -1 : 1;
        normalized.focalX = std::fabs(normalized.focalX);
        normalized.focalY = std::fabs(normalized.focalY);
        normalized.skew *= x_ratio_sign * y_ratio_sign;
        normalized.uAxisSign = u_sign * static_cast<int>(x_ratio_sign);
        normalized.vAxisSign = v_sign * static_cast<int>(y_ratio_sign);
        if (distortion.tangentialConvention == BrownTangentialConvention::Metashape)
        {
            distortion.tangentialP1 = x_ratio_sign * distortion.tangentialP1;
            distortion.tangentialP2 = y_ratio_sign * distortion.tangentialP2;
        }
        else
        {
            distortion.tangentialP1 = y_ratio_sign * distortion.tangentialP1;
            distortion.tangentialP2 = x_ratio_sign * distortion.tangentialP2;
        }
        return std::shared_ptr<const FramePinholeDefinition>(new FramePinholeDefinition(std::move(definitionId),
                                                                                        normalized,
                                                                                        distortion,
                                                                                        _pixelConvention,
                                                                                        _groundFrame,
                                                                                        false,
                                                                                        _projectionModel,
                                                                                        _sensorMount,
                                                                                        _principalPointDecomposition));
    }

    FramePinholeDefinition::FramePinholeDefinition(
        CameraDefinitionId definitionId,
        FrameIntrinsics intrinsics,
        BrownConradyDistortion distortion,
        PixelConvention pixelConvention,
        FrameId groundFrame,
        bool depthAxisFlipped,
        FrameProjectionModel projectionModel,
        SensorMountState sensorMount,
        std::optional<PrincipalPointDecomposition> principalPointDecomposition)
        : _definitionId(std::move(definitionId)), _intrinsics(intrinsics), _distortion(distortion),
          _pixelConvention(pixelConvention), _groundFrame(std::move(groundFrame)), _depthAxisFlipped(depthAxisFlipped),
          _projectionModel(projectionModel), _sensorMount(std::move(sensorMount)),
          _principalPointDecomposition(std::move(principalPointDecomposition))
    {
    }

    void FramePinholeDefinition::validate(const FrameIntrinsics& intrinsics, const BrownConradyDistortion& distortion)
    {
        if (!finite(intrinsics.focalX) || !finite(intrinsics.focalY) || !finite(intrinsics.principalX) ||
            !finite(intrinsics.principalY) || !finite(intrinsics.pixelPitch) || intrinsics.focalX <= 0.0 ||
            intrinsics.focalY <= 0.0 || intrinsics.pixelPitch <= 0.0 || !finite(intrinsics.skew) ||
            (intrinsics.uAxisSign != 1 && intrinsics.uAxisSign != -1) ||
            (intrinsics.vAxisSign != 1 && intrinsics.vAxisSign != -1))
        {
            throw CameraValidationError(CameraErrorCode::InvalidIntrinsics,
                                        "pinhole intrinsics must be finite, positive, and use signed axes");
        }

        if (!finite(distortion.radialK1) || !finite(distortion.radialK2) || !finite(distortion.radialK3) ||
            !finite(distortion.radialK4) || !finite(distortion.tangentialP1) || !finite(distortion.tangentialP2) ||
            !finite(distortion.tangentialP3) || !finite(distortion.tangentialP4))
        {
            throw CameraValidationError(CameraErrorCode::InvalidDistortion,
                                        "pinhole distortion coefficients must be finite");
        }
        if (distortion.tangentialConvention != BrownTangentialConvention::OpenCv &&
            distortion.tangentialConvention != BrownTangentialConvention::Metashape)
        {
            throw CameraValidationError(CameraErrorCode::InvalidDistortion,
                                        "pinhole tangential distortion convention is invalid");
        }
        if (distortion.tangentialConvention == BrownTangentialConvention::OpenCv &&
            (distortion.tangentialP3 != 0.0 || distortion.tangentialP4 != 0.0))
        {
            throw CameraValidationError(
                CameraErrorCode::InvalidDistortion,
                "p3/p4 require the Metashape tangential convention because they scale its full decentering field");
        }
    }

} // namespace placamera

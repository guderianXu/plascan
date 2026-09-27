#pragma once

#include "placamera/frame_camera.h"

#include <array>

namespace placamera::internal
{

    using NormalizedCoordinate = std::array<double, 2>;

    bool finiteVector(const Vector3& value) noexcept;
    bool finiteImage(const ImageCoordinate& value) noexcept;
    bool insideImage(const ImageCoordinate& value, const ImageSize& size) noexcept;
    bool validOptions(const EvaluationOptions& options) noexcept;

    void distort(
        const BrownConradyDistortion& distortion, double x, double y, double* distortedX, double* distortedY) noexcept;

    Vector3 worldToCamera(const Pose& pose, const Vector3& world) noexcept;
    Vector3 cameraToWorld(const Pose& pose, const Vector3& camera) noexcept;
    EvaluationResult<double> signedFrameDepth(const FrameId& groundFrame,
                                              bool depthAxisFlipped,
                                              const Pose& pose,
                                              const GroundCoordinate& ground);
    RotationMatrix normalizedCameraToWorldRotation(const RotationMatrix& cameraToWorld,
                                                   const std::array<double, 3>& axisSigns) noexcept;

    EvaluationResult<NormalizedCoordinate>
    undistort(const FramePinholeDefinition& definition, const ImageCoordinate& image, const EvaluationOptions& options);

    EvaluationResult<NormalizedCoordinate> undistort(const FrameIntrinsics& intrinsics,
                                                     const BrownConradyDistortion& distortion,
                                                     FrameProjectionModel projectionModel,
                                                     const std::optional<PrincipalPointDecomposition>&
                                                         principalPointDecomposition,
                                                     const ImageCoordinate& image,
                                                     const EvaluationOptions& options);

    EvaluationResult<Projection> projectFrame(const FrameId& groundFrame,
                                              const ImageSize& imageSize,
                                              const std::optional<TimeReference>& captureTime,
                                              const FrameIntrinsics& intrinsics,
                                              const BrownConradyDistortion& distortion,
                                              FrameProjectionModel projectionModel,
                                              bool depthAxisFlipped,
                                              const Pose& pose,
                                              const CameraAcquisitionState& acquisition,
                                              const GroundCoordinate& ground,
                                              const EvaluationOptions& options);

    EvaluationResult<Projection> projectFrameSigned(const FrameId& groundFrame,
                                                    const ImageSize& imageSize,
                                                    const std::optional<TimeReference>& captureTime,
                                                    const FrameIntrinsics& intrinsics,
                                                    const BrownConradyDistortion& distortion,
                                                    FrameProjectionModel projectionModel,
                                                    bool depthAxisFlipped,
                                                    const Pose& pose,
                                                    const CameraAcquisitionState& acquisition,
                                                    const GroundCoordinate& ground,
                                                    const EvaluationOptions& options);

    EvaluationResult<ImagingLocus> frameImagingLocus(const FrameId& groundFrame,
                                                     const ImageSize& imageSize,
                                                     const std::optional<TimeReference>& captureTime,
                                                     const FrameIntrinsics& intrinsics,
                                                     const BrownConradyDistortion& distortion,
                                                     const std::optional<PrincipalPointDecomposition>&
                                                         principalPointDecomposition,
                                                     FrameProjectionModel projectionModel,
                                                     bool depthAxisFlipped,
                                                     const Pose& pose,
                                                     const CameraAcquisitionState& acquisition,
                                                     const ImageCoordinate& image,
                                                     const EvaluationOptions& options);

    EvaluationResult<GroundCoordinate> frameGroundAtDepth(const FrameId& groundFrame,
                                                          const ImageSize& imageSize,
                                                          const FrameIntrinsics& intrinsics,
                                                          const BrownConradyDistortion& distortion,
                                                          const std::optional<PrincipalPointDecomposition>&
                                                              principalPointDecomposition,
                                                          FrameProjectionModel projectionModel,
                                                          bool depthAxisFlipped,
                                                          const Pose& pose,
                                                          const CameraAcquisitionState& acquisition,
                                                          const ImageCoordinate& image,
                                                          double positiveDepth,
                                                          const EvaluationOptions& options);

} // namespace placamera::internal

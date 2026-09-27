#pragma once

#include "placamera/result.h"
#include "placamera/types.h"

#include <cstddef>
#include <optional>

namespace placamera
{

    enum class CameraRole
    {
        Regular,
        Keyframe,
    };

    enum class RollingShutterMode
    {
        Disabled,
        Regularized,
        Full,
    };

    struct RollingShutterMotion
    {
        Vector3 translation{};
        Vector3 rotationVector{};

        bool isIdentity() const noexcept;
    };

    /** Sensor-to-master mount state, independent of the sensor projection family. */
    struct SensorMountState
    {
        std::optional<CameraDefinitionId> masterSensorId;
        Vector3 translation{};
        RotationMatrix rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        bool fixedTranslation = true;
        bool fixedRotation = true;
    };

    /** Per-image acquisition topology, independent of the sensor projection family. */
    struct CameraAcquisitionState
    {
        CameraRole role = CameraRole::Regular;
        std::optional<CaptureGroupId> captureGroupId;
        std::optional<CameraInstanceId> masterCameraId;
        std::size_t layerIndex = 0;
        RollingShutterMode rollingShutterMode = RollingShutterMode::Disabled;
        RollingShutterMotion rollingShutter;
        bool rollingShutterInitialized = false;
    };

    Result<void> validateSensorMount(const SensorMountState& mount, const CameraDefinitionId* definitionId = nullptr);
    Result<void> validateCameraAcquisition(const CameraAcquisitionState& acquisition,
                                           const CameraInstanceId* instanceId = nullptr);

} // namespace placamera

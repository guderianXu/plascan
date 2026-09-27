#include <placamera/camera_topology.h>

#include <gtest/gtest.h>

namespace
{

    using namespace placamera;

    TEST(CameraTopologyTest, ValidatesIndependentSensorAndAcquisitionIdentity)
    {
        SensorMountState mount;
        mount.masterSensorId = CameraDefinitionId("master-sensor");
        EXPECT_TRUE(validateSensorMount(mount, nullptr));
        const CameraDefinitionId own_sensor("master-sensor");
        EXPECT_FALSE(validateSensorMount(mount, &own_sensor));

        CameraAcquisitionState acquisition;
        acquisition.captureGroupId = CaptureGroupId("capture-1");
        acquisition.masterCameraId = CameraInstanceId("master-camera");
        acquisition.layerIndex = 2;
        acquisition.role = CameraRole::Keyframe;
        EXPECT_TRUE(validateCameraAcquisition(acquisition, nullptr));
        const CameraInstanceId own_camera("master-camera");
        EXPECT_FALSE(validateCameraAcquisition(acquisition, &own_camera));
    }

    TEST(CameraTopologyTest, EnforcesRollingShutterModeDomain)
    {
        CameraAcquisitionState regularized;
        regularized.rollingShutterMode = RollingShutterMode::Regularized;
        regularized.rollingShutter.translation = {0.1, -0.2, 0.0};
        EXPECT_TRUE(validateCameraAcquisition(regularized));
        regularized.rollingShutter.rotationVector[0] = 0.01;
        EXPECT_FALSE(validateCameraAcquisition(regularized));

        CameraAcquisitionState full = regularized;
        full.rollingShutterMode = RollingShutterMode::Full;
        full.rollingShutter.translation[2] = 0.3;
        EXPECT_TRUE(validateCameraAcquisition(full));
    }

} // namespace

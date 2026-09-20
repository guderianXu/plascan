#include "camera/models/CameraModelFactories.h"

#include "camera/core/capabilities/CapabilityRequirements.h"
#include "camera/models/frame_pinhole/FramePinholeDefinition.h"
#include "camera/models/frame_pinhole/FramePinholeInstance.h"
#include "camera/models/linescan/LineScanDefinition.h"
#include "camera/models/linescan/LineScanInstance.h"
#include "camera/models/rpc/RpcDefinition.h"

#include <gtest/gtest.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <array>
#include <memory>
#include <string>

namespace
{

    std::string rpcParametersJson()
    {
        QJsonArray numerator;
        QJsonArray denominator;
        for (int index = 0; index < 20; ++index)
        {
            numerator.append(index == 0 ? 1.0 : 0.0);
            denominator.append(index == 0 ? 1.0 : 0.0);
        }
        return QJsonDocument(QJsonObject{{QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                                         {QStringLiteral("line_offset"), 0.0},
                                         {QStringLiteral("sample_offset"), 0.0},
                                         {QStringLiteral("latitude_offset"), 0.0},
                                         {QStringLiteral("longitude_offset"), 0.0},
                                         {QStringLiteral("height_offset"), 0.0},
                                         {QStringLiteral("line_scale"), 1.0},
                                         {QStringLiteral("sample_scale"), 1.0},
                                         {QStringLiteral("latitude_scale"), 1.0},
                                         {QStringLiteral("longitude_scale"), 1.0},
                                         {QStringLiteral("height_scale"), 1.0},
                                         {QStringLiteral("line_numerator"), numerator},
                                         {QStringLiteral("line_denominator"), denominator},
                                         {QStringLiteral("sample_numerator"), numerator},
                                         {QStringLiteral("sample_denominator"), denominator}})
            .toJson(QJsonDocument::Compact)
            .toStdString();
    }

    std::string definitionParametersFor(const char* model)
    {
        if (std::string(model) == "rpc00b")
        {
            return rpcParametersJson();
        }
        if (std::string(model) == "planetary_linescan")
        {
            return R"({"optics":{"focal_length_mm":700,"distortion_model":"radial_normalized","distortion_k1":0,"sample_geometry":{"type":"uniform_pitch","sample_pitch_mm":0.01,"principal_sample":512}},"pixel_convention":"pixel_center"})";
        }
        return R"({"intrinsics":{"fx_px":100,"fy_px":100,"cx_px":50,"cy_px":50,"pixel_pitch_mm":0.01,"u_axis_sign":1,"v_axis_sign":1},"distortion":{"k1":0,"k2":0,"k3":0,"p1":0,"p2":0},"pixel_convention":"center","depth_axis_flipped":false})";
    }

    std::string instanceStateFor(const char* model)
    {
        if (std::string(model) == "planetary_linescan")
        {
            return R"({"image_size":{"samples":2,"lines":2},"line_timing":{"time_scale":"tdb","segments":[{"start_line":0.5,"start_time_seconds":0,"seconds_per_line":1}]},"trajectory":{"representation":"direct_pose_samples","time_scale":"tdb","samples":[{"time_seconds":0,"center_m":[0,0,0],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]},{"time_seconds":1,"center_m":[0,0,0],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]}]}})";
        }
        if (std::string(model) == "frame_pinhole")
        {
            return R"({"image_size":{"samples":2,"lines":2},"pose":{"frame":"project-world","center_m":[0,0,0],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]}})";
        }
        return R"({"image_size":{"samples":2,"lines":2}})";
    }

    TEST(CameraModelFactoriesTest, RegistersTheBuiltInModelKinds)
    {
        xjw::camera_core::CameraModelRegistry registry = xjw::camera_models::makeBuiltinCameraModelRegistry();

        const std::array<std::pair<const char*, xjw::camera_core::CapabilitySet>, 3> models{{
            {"frame_pinhole", {xjw::camera_core::CapabilityKind::StaticPose}},
            {"rpc00b", {xjw::camera_core::CapabilityKind::InverseProjection}},
            {"planetary_linescan", {xjw::camera_core::CapabilityKind::Trajectory}},
        }};

        for (const auto& [model, required] : models)
        {
            auto definition =
                registry.createDefinition(model,
                                          xjw::camera_core::CameraDefinitionId(std::string(model) + "-definition"),
                                          xjw::coordinate_system::CoordinateFrameId("project-world"),
                                          *xjw::camera_models::builtinCameraParameterSchemaVersion(model),
                                          definitionParametersFor(model));
            ASSERT_TRUE(definition);
            auto instance = registry.createInstance(
                model,
                xjw::camera_core::CameraInstanceId(std::string(model) + "-instance"),
                xjw::camera_core::ImageId(std::string(model) + "-image"),
                std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
                instanceStateFor(model));
            ASSERT_TRUE(instance);
            EXPECT_TRUE(xjw::camera_core::requireCapabilities(*instance, required).ok()) << model;
        }
    }

    TEST(CameraModelFactoriesTest, KeepsRpcAndLineScanWithoutStaticPose)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        for (const char* model : {"rpc00b", "planetary_linescan"})
        {
            auto definition =
                registry.createDefinition(model,
                                          xjw::camera_core::CameraDefinitionId(std::string(model) + "-definition"),
                                          xjw::coordinate_system::CoordinateFrameId("project-world"),
                                          *xjw::camera_models::builtinCameraParameterSchemaVersion(model),
                                          definitionParametersFor(model));
            auto instance = registry.createInstance(
                model,
                xjw::camera_core::CameraInstanceId(std::string(model) + "-instance"),
                xjw::camera_core::ImageId(std::string(model) + "-image"),
                std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
                instanceStateFor(model));
            ASSERT_TRUE(instance);
            EXPECT_FALSE(
                xjw::camera_core::requireCapabilities(*instance, {xjw::camera_core::CapabilityKind::StaticPose}).ok());
        }
    }

    TEST(CameraModelFactoriesTest, DecodesFramePinholeDefinitionAndInstanceState)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        auto definition = registry.createDefinition(
            "frame_pinhole",
            xjw::camera_core::CameraDefinitionId("frame-definition"),
            xjw::coordinate_system::CoordinateFrameId("project-world"),
            1,
            R"({"intrinsics":{"fx_px":1200,"fy_px":1190,"cx_px":512,"cy_px":384,"pixel_pitch_mm":0.01,"u_axis_sign":1,"v_axis_sign":1},"distortion":{"k1":0.02,"k2":0,"k3":0,"p1":0,"p2":0},"pixel_convention":"center","depth_axis_flipped":false})");
        auto pinholeDefinition =
            dynamic_cast<const xjw::camera_models::frame_pinhole::FramePinholeDefinition*>(definition.get());
        ASSERT_NE(pinholeDefinition, nullptr);
        EXPECT_DOUBLE_EQ(pinholeDefinition->intrinsics().focalX, 1200.0);
        EXPECT_DOUBLE_EQ(pinholeDefinition->distortion().radialK1, 0.02);

        auto instance = registry.createInstance(
            "frame_pinhole",
            xjw::camera_core::CameraInstanceId("frame-instance"),
            xjw::camera_core::ImageId("frame-image"),
            std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
            R"({"image_size":{"samples":1024,"lines":768},"pose":{"frame":"project-world","center_m":[1,2,3],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]},"capture_time":{"seconds":42.5,"time_scale":"utc"}})");
        ASSERT_NE(instance, nullptr);
        EXPECT_EQ(instance->imageSize().samples, 1024);
        EXPECT_EQ(instance->imageSize().lines, 768);
        const auto* pinholeInstance =
            dynamic_cast<const xjw::camera_models::frame_pinhole::FramePinholeInstance*>(instance.get());
        ASSERT_NE(pinholeInstance, nullptr);
        EXPECT_DOUBLE_EQ(pinholeInstance->pose().center[0], 1.0);
        ASSERT_TRUE(pinholeInstance->captureTime().has_value());
        EXPECT_EQ(pinholeInstance->captureTime()->scale, xjw::coordinate_system::TimeScale::Utc);
        EXPECT_DOUBLE_EQ(pinholeInstance->captureTime()->seconds, 42.5);
    }

    TEST(CameraModelFactoriesTest, RejectsMalformedSerializedStateAndIncompleteRpc)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        EXPECT_THROW(registry.createDefinition("frame_pinhole",
                                               xjw::camera_core::CameraDefinitionId("bad-definition"),
                                               xjw::coordinate_system::CoordinateFrameId("project-world"),
                                               1,
                                               "{not-json"),
                     xjw::camera_core::CameraRegistryError);
        EXPECT_THROW(registry.createDefinition("rpc00b",
                                               xjw::camera_core::CameraDefinitionId("bad-rpc"),
                                               xjw::coordinate_system::CoordinateFrameId("wgs84"),
                                               1,
                                               R"({"line_num_coeff":[1]})"),
                     xjw::camera_core::CameraRegistryError);
    }

    TEST(CameraModelFactoriesTest, RejectsLegacyFlatDefinitionFields)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        EXPECT_THROW(registry.createDefinition(
                         "frame_pinhole",
                         xjw::camera_core::CameraDefinitionId("flat-definition"),
                         xjw::coordinate_system::CoordinateFrameId("project-world"),
                         1,
                         R"({"fx_px":1200,"fy_px":1200,"cx_px":512,"cy_px":384,"pixel_pitch":0.01,"k1":0})"),
                     xjw::camera_core::CameraRegistryError);

        EXPECT_THROW(
            registry.createDefinition(
                "rpc00b",
                xjw::camera_core::CameraDefinitionId("aliased-rpc"),
                xjw::coordinate_system::CoordinateFrameId("wgs84-geodetic"),
                1,
                R"({"rpc_spec":"RPC00B","line_num_coeff":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"line_den_coeff":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"samp_num_coeff":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"samp_den_coeff":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]})"),
            xjw::camera_core::CameraRegistryError);
    }

    TEST(CameraModelFactoriesTest, RequiresCanonicalInstanceState)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        auto definition = registry.createDefinition("frame_pinhole",
                                                    xjw::camera_core::CameraDefinitionId("state-definition"),
                                                    xjw::coordinate_system::CoordinateFrameId("project-world"),
                                                    1,
                                                    definitionParametersFor("frame_pinhole"));
        EXPECT_THROW(
            registry.createInstance(
                "frame_pinhole",
                xjw::camera_core::CameraInstanceId("state-instance"),
                xjw::camera_core::ImageId("state-image"),
                std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
                R"({"pose":{"frame":"project-world","center_m":[0,0,0],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]}})"),
            xjw::camera_core::CameraRegistryError);
    }

    TEST(CameraModelFactoriesTest, RejectsImageSizeOutsideIntegerRange)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        auto definition = registry.createDefinition("frame_pinhole",
                                                    xjw::camera_core::CameraDefinitionId("large-image-definition"),
                                                    xjw::coordinate_system::CoordinateFrameId("project-world"),
                                                    1,
                                                    definitionParametersFor("frame_pinhole"));
        EXPECT_THROW(
            registry.createInstance(
                "frame_pinhole",
                xjw::camera_core::CameraInstanceId("large-image-instance"),
                xjw::camera_core::ImageId("large-image"),
                std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
                R"({"image_size":{"samples":2147483648,"lines":2},"pose":{"frame":"project-world","center_m":[0,0,0],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]}})"),
            xjw::camera_core::CameraRegistryError);
    }

    TEST(CameraModelFactoriesTest, RejectsUnknownParameterSchemaVersions)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        EXPECT_THROW(registry.createDefinition("frame_pinhole",
                                               xjw::camera_core::CameraDefinitionId("future-definition"),
                                               xjw::coordinate_system::CoordinateFrameId("project-world"),
                                               2,
                                               R"({"intrinsics":{"fx_px":100,"fy_px":100,"cx_px":50,"cy_px":50}})"),
                     xjw::camera_core::CameraRegistryError);
    }

    TEST(CameraModelFactoriesTest, DecodesLineScanOpticsTrajectoryAndTiming)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        auto definition = registry.createDefinition(
            "planetary_linescan",
            xjw::camera_core::CameraDefinitionId("line-definition"),
            xjw::coordinate_system::CoordinateFrameId("moon-fixed"),
            2,
            R"({"optics":{"focal_length_mm":700,"distortion_model":"radial_normalized","distortion_k1":0.001,"sample_geometry":{"type":"uniform_pitch","sample_pitch_mm":0.01,"principal_sample":512}},"pixel_convention":"zero_based"})");
        auto lineDefinition = dynamic_cast<const xjw::camera_models::linescan::LineScanDefinition*>(definition.get());
        ASSERT_NE(lineDefinition, nullptr);
        EXPECT_DOUBLE_EQ(lineDefinition->optics().focalLengthMillimeters, 700.0);
        EXPECT_EQ(lineDefinition->pixelConvention(), xjw::camera_models::linescan::PixelConvention::ZeroBased);

        auto instance = registry.createInstance(
            "planetary_linescan",
            xjw::camera_core::CameraInstanceId("line-instance"),
            xjw::camera_core::ImageId("line-image"),
            std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
            R"({"image_size":{"samples":2048,"lines":4096},"line_timing":{"time_scale":"tdb","segments":[{"start_line":0.5,"start_time_seconds":10,"seconds_per_line":0.01}]},"trajectory":{"representation":"direct_pose_samples","time_scale":"tdb","samples":[{"time_seconds":10,"center_m":[0,0,1],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]},{"time_seconds":20,"center_m":[0,0,2],"camera_to_world_rotation":[1,0,0,0,1,0,0,0,1]}]}})");
        ASSERT_NE(instance, nullptr);
        const auto* lineInstance = dynamic_cast<const xjw::camera_models::linescan::LineScanInstance*>(instance.get());
        ASSERT_NE(lineInstance, nullptr);
        EXPECT_EQ(lineInstance->imageSize().lines, 4096);
        EXPECT_DOUBLE_EQ(lineInstance->lineTiming().secondsPerLine, 0.01);
        EXPECT_EQ(lineInstance->trajectory().samples().size(), 2U);
    }

    TEST(CameraModelFactoriesTest, RejectsLineScanInstanceWithoutTrajectory)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        auto definition = registry.createDefinition(
            "planetary_linescan",
            xjw::camera_core::CameraDefinitionId("line-definition"),
            xjw::coordinate_system::CoordinateFrameId("moon-fixed"),
            2,
            R"({"optics":{"focal_length_mm":700,"distortion_model":"radial_normalized","distortion_k1":0,"sample_geometry":{"type":"uniform_pitch","sample_pitch_mm":0.01,"principal_sample":0}},"pixel_convention":"pixel_center"})");
        EXPECT_THROW(
            registry.createInstance("planetary_linescan",
                                    xjw::camera_core::CameraInstanceId("line-instance"),
                                    xjw::camera_core::ImageId("line-image"),
                                    std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
                                    R"({"image_size":{"samples":2048,"lines":4096}})"),
            xjw::camera_core::CameraRegistryError);
    }

    TEST(CameraModelFactoriesTest, DecodesFrameComposedLineScanWithoutFlatteningFrames)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        auto definition = registry.createDefinition(
            "planetary_linescan",
            xjw::camera_core::CameraDefinitionId("composed-definition"),
            xjw::coordinate_system::CoordinateFrameId("MOON_ME"),
            2,
            R"({"optics":{"focal_length_mm":700,"distortion_model":"lro_nac_focal_plane","distortion_k1":0.001,"sample_geometry":{"type":"detector_affine","detector_sample_summing":1,"detector_line_summing":1,"detector_sample_origin":512,"detector_line_origin":0,"starting_detector_sample":0,"starting_detector_line":0,"focal_to_pixel_samples":[0,0,1],"focal_to_pixel_lines":[0,1,0]}},"pixel_convention":"pixel_center"})");
        auto instance = registry.createInstance(
            "planetary_linescan",
            xjw::camera_core::CameraInstanceId("composed-instance"),
            xjw::camera_core::ImageId("composed-image"),
            std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
            R"({"image_size":{"samples":1024,"lines":200},"capture_time":{"time_scale":"tdb","seconds":15},"line_timing":{"time_scale":"tdb","segments":[{"start_line":0.5,"start_time_seconds":10,"seconds_per_line":0.01},{"start_line":100.5,"start_time_seconds":11,"seconds_per_line":0.02}]},"trajectory":{"representation":"frame_composed","time_scale":"tdb","inertial_states":[{"time_seconds":10,"position_m":[1,2,3],"velocity_m_per_s":[1,0,0]},{"time_seconds":20,"position_m":[11,2,3],"velocity_m_per_s":[1,0,0]}],"inertial_to_world":{"constant_rotation":[1,0,0,0,1,0,0,0,1],"samples":[{"time_seconds":10,"quaternion_scalar_first":[1,0,0,0]},{"time_seconds":20,"quaternion_scalar_first":[1,0,0,0]}]},"inertial_to_sensor":{"constant_rotation":[1,0,0,0,1,0,0,0,1],"samples":[{"time_seconds":10,"quaternion_scalar_first":[1,0,0,0]},{"time_seconds":20,"quaternion_scalar_first":[1,0,0,0]}]}}})");

        const auto* lineInstance = dynamic_cast<const xjw::camera_models::linescan::LineScanInstance*>(instance.get());
        ASSERT_NE(lineInstance, nullptr);
        ASSERT_NE(lineInstance->trajectory().frameComposed(), nullptr);
        EXPECT_EQ(lineInstance->trajectory().frameComposed()->inertialStates.size(), 2U);
        EXPECT_EQ(lineInstance->lineTiming().segments.size(), 2U);
        EXPECT_TRUE(lineInstance->lineScanDefinition().optics().detectorGeometry.has_value());
    }

    TEST(CameraModelFactoriesTest, RejectsLegacyLineScanSchemaAndArrayTrajectory)
    {
        auto registry = xjw::camera_models::makeBuiltinCameraModelRegistry();
        EXPECT_THROW(registry.createDefinition("planetary_linescan",
                                               xjw::camera_core::CameraDefinitionId("legacy-definition"),
                                               xjw::coordinate_system::CoordinateFrameId("moon-fixed"),
                                               1,
                                               definitionParametersFor("planetary_linescan")),
                     xjw::camera_core::CameraRegistryError);

        auto definition = registry.createDefinition("planetary_linescan",
                                                    xjw::camera_core::CameraDefinitionId("strict-definition"),
                                                    xjw::coordinate_system::CoordinateFrameId("moon-fixed"),
                                                    2,
                                                    definitionParametersFor("planetary_linescan"));
        EXPECT_THROW(
            registry.createInstance(
                "planetary_linescan",
                xjw::camera_core::CameraInstanceId("legacy-instance"),
                xjw::camera_core::ImageId("line-image"),
                std::shared_ptr<const xjw::camera_core::CameraDefinition>(std::move(definition)),
                R"({"image_size":{"samples":2,"lines":2},"line_timing":{"time_scale":"tdb","segments":[{"start_line":0.5,"start_time_seconds":0,"seconds_per_line":1}]},"trajectory":[]})"),
            xjw::camera_core::CameraRegistryError);
    }

} // namespace

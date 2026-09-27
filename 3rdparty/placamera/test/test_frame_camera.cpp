#include <placamera/frame_camera.h>
#include <placamera/frame_numeric_state.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace
{

    using namespace placamera;

    std::shared_ptr<const FramePinholeDefinition>
    makeDefinition(bool depthAxisFlipped = false,
                   int uAxisSign = 1,
                   int vAxisSign = 1,
                   BrownConradyDistortion distortion = {},
                   PixelConvention convention = PixelConvention::PixelCenter)
    {
        FrameIntrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 110.0;
        intrinsics.principalX = 50.0;
        intrinsics.principalY = 60.0;
        intrinsics.pixelPitch = 0.01;
        intrinsics.uAxisSign = uAxisSign;
        intrinsics.vAxisSign = vAxisSign;
        return FramePinholeDefinition::create(
            CameraDefinitionId("definition-1"), intrinsics, distortion, convention, FrameId("world"), depthAxisFlipped);
    }

    FramePinholeModel makeModel(std::shared_ptr<const FramePinholeDefinition> definition,
                                std::optional<TimeReference> captureTime = std::nullopt)
    {
        return FramePinholeModel::create(CameraInstanceId("instance-1"),
                                         ImageId("image-1"),
                                         std::move(definition),
                                         ImageSize{1000, 800},
                                         Pose::create(FrameId("world"),
                                                      {0.0, 0.0, 0.0},
                                                      RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}),
                                         captureTime);
    }

    TEST(FramePinholeModelTest, ProjectsAndUnprojectsWithExplicitFrames)
    {
        const FramePinholeModel model = makeModel(makeDefinition());
        const auto projected = model.groundToImage(GroundCoordinate{FrameId("world"), {1.0, 2.0, 10.0}});
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 60.0, 1.0e-12);
        EXPECT_NEAR(projected.value().image.line, 82.0, 1.0e-12);
        ASSERT_TRUE(projected.value().positiveDepth.has_value());
        EXPECT_DOUBLE_EQ(*projected.value().positiveDepth, 10.0);

        const auto ground = model.imageToGroundAtDepth(projected.value().image, *projected.value().positiveDepth);
        ASSERT_TRUE(ground) << ground.message();
        EXPECT_NEAR(ground.value().position[0], 1.0, 1.0e-9);
        EXPECT_NEAR(ground.value().position[1], 2.0, 1.0e-9);
        EXPECT_NEAR(ground.value().position[2], 10.0, 1.0e-9);
        EXPECT_EQ(ground.value().frame, FrameId("world"));
    }

    TEST(FramePinholeModelTest, RebindsImageGridWithoutLosingCameraIdentityOrCaptureTime)
    {
        const FramePinholeModel model = makeModel(makeDefinition(), TimeReference::create(TimeScale::Tdb, 42.0));
        const FramePinholeModel resized = model.withImageSize(ImageSize{1200, 900});

        EXPECT_EQ(resized.imageSize().samples, 1200);
        EXPECT_EQ(resized.imageSize().lines, 900);
        EXPECT_EQ(resized.instanceId(), model.instanceId());
        EXPECT_EQ(resized.definitionId(), model.definitionId());
        EXPECT_EQ(resized.imageId(), model.imageId());
        EXPECT_EQ(resized.groundFrame(), model.groundFrame());
        EXPECT_EQ(resized.captureTime(), model.captureTime());
        EXPECT_DOUBLE_EQ(resized.pinholeDefinition().intrinsics().focalX,
                         model.pinholeDefinition().intrinsics().focalX);
        EXPECT_THROW(model.withImageSize(ImageSize{0, 900}), CameraValidationError);
    }

    TEST(FramePinholeModelTest, BindsCentralGeometryWithCanonicalImmutableOwnership)
    {
        static_assert(std::is_same_v<CentralCameraGeometry, FramePinholeGeometry>);
        static_assert(std::is_same_v<CameraModelPtr<CentralCameraModel>, std::shared_ptr<const FramePinholeModel>>);

        CentralCameraGeometry geometry{makeDefinition(),
                                       Pose::create(FrameId("world"),
                                                    {0.0, 0.0, 0.0},
                                                    RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}})};
        CameraAcquisitionState acquisition;
        acquisition.role = CameraRole::Keyframe;
        const auto model = bindCentralCamera(std::move(geometry),
                                             {CameraInstanceId("bound-instance"),
                                              ImageId("bound-image"),
                                              ImageSize{1000, 800},
                                              std::nullopt,
                                              acquisition});

        ASSERT_TRUE(model) << model.message();
        EXPECT_EQ(model.value()->instanceId(), CameraInstanceId("bound-instance"));
        EXPECT_EQ(model.value()->acquisition().role, CameraRole::Keyframe);
    }

    TEST(FramePinholeModelTest, SignedDepthSupportsNearPlaneClipping)
    {
        const FramePinholeModel model = makeModel(makeDefinition());
        const auto front = model.signedDepth(GroundCoordinate{FrameId("world"), {0.0, 0.0, 2.0}});
        const auto behind = model.signedDepth(GroundCoordinate{FrameId("world"), {0.0, 0.0, -3.0}});
        const auto plane = model.signedDepth(GroundCoordinate{FrameId("world"), {0.0, 0.0, 0.0}});
        ASSERT_TRUE(front) << front.message();
        ASSERT_TRUE(behind) << behind.message();
        ASSERT_TRUE(plane) << plane.message();
        EXPECT_DOUBLE_EQ(front.value(), 2.0);
        EXPECT_DOUBLE_EQ(behind.value(), -3.0);
        EXPECT_DOUBLE_EQ(plane.value(), 0.0);
        const auto projected = model.groundToImage(GroundCoordinate{FrameId("world"), {0.0, 0.0, 2.0}});
        ASSERT_TRUE(projected) << projected.message();
        ASSERT_TRUE(projected.value().positiveDepth.has_value());
        EXPECT_DOUBLE_EQ(front.value(), *projected.value().positiveDepth);

        const FramePinholeModel flipped = makeModel(makeDefinition(true));
        const auto flipped_front = flipped.signedDepth(GroundCoordinate{FrameId("world"), {0.0, 0.0, -3.0}});
        ASSERT_TRUE(flipped_front) << flipped_front.message();
        EXPECT_DOUBLE_EQ(flipped_front.value(), 3.0);

        const auto wrong_frame = model.signedDepth(GroundCoordinate{FrameId("other"), {0.0, 0.0, 2.0}});
        EXPECT_FALSE(wrong_frame);
        EXPECT_EQ(wrong_frame.errorCode(), CameraErrorCode::FrameMismatch);
        const auto non_finite =
            model.signedDepth(GroundCoordinate{FrameId("world"), {0.0, 0.0, std::numeric_limits<double>::infinity()}});
        EXPECT_FALSE(non_finite);
        EXPECT_EQ(non_finite.errorCode(), CameraErrorCode::InvalidArgument);
    }

    TEST(FramePinholeModelTest, SignedProjectionKeepsBehindCameraDiagnosticsSeparateFromPositiveDepth)
    {
        const FramePinholeModel model = makeModel(makeDefinition());
        const FramePinholeNumericState state = FramePinholeNumericState::fromModel(model);
        const GroundCoordinate behind{FrameId("world"), {1.0, 2.0, -10.0}};

        EXPECT_FALSE(model.groundToImage(behind));
        const auto model_projection = model.groundToImageSigned(behind);
        const auto state_projection = state.groundToImageSigned(behind);
        ASSERT_TRUE(model_projection) << model_projection.message();
        ASSERT_TRUE(state_projection) << state_projection.message();
        EXPECT_FALSE(model_projection.value().positiveDepth);
        EXPECT_FALSE(state_projection.value().positiveDepth);
        EXPECT_DOUBLE_EQ(model_projection.value().image.sample, 40.0);
        EXPECT_DOUBLE_EQ(model_projection.value().image.line, 38.0);
        EXPECT_DOUBLE_EQ(state_projection.value().image.sample, model_projection.value().image.sample);
        EXPECT_DOUBLE_EQ(state_projection.value().image.line, model_projection.value().image.line);
        const auto signed_depth = state.signedDepth(behind);
        ASSERT_TRUE(signed_depth);
        EXPECT_DOUBLE_EQ(signed_depth.value(), -10.0);

        const GroundCoordinate focal_plane{FrameId("world"), {1.0, 2.0, 0.0}};
        EXPECT_FALSE(model.groundToImageSigned(focal_plane));
        EXPECT_EQ(model.groundToImageSigned(GroundCoordinate{FrameId("other"), behind.position}).errorCode(),
                  CameraErrorCode::FrameMismatch);
    }

    TEST(FramePinholeModelTest, ImplementsUniformRasterModelContract)
    {
        const FramePinholeModel frame_model = makeModel(makeDefinition(), TimeReference::create(TimeScale::Tdb, 42.0));
        const RasterModel& model = frame_model;

        EXPECT_EQ(model.modelType(), "frame_pinhole");
        EXPECT_EQ(model.parameterSchemaVersion(), FramePinholeDefinition::ParameterSchemaVersion);
        EXPECT_TRUE(model.capabilities().contains(CapabilityKind::Projection));
        EXPECT_TRUE(model.capabilities().contains(CapabilityKind::ImagingLocus));

        const auto locus = model.imageToImagingLocus(ImageCoordinate{50.0, 60.0});
        ASSERT_TRUE(locus) << locus.message();
        EXPECT_EQ(locus.value().origin.frame, FrameId("world"));
        EXPECT_EQ(locus.value().origin.position, (Vector3{0.0, 0.0, 0.0}));
        EXPECT_NEAR(locus.value().direction[0], 0.0, 1.0e-12);
        EXPECT_NEAR(locus.value().direction[1], 0.0, 1.0e-12);
        EXPECT_NEAR(locus.value().direction[2], 1.0, 1.0e-12);
        ASSERT_TRUE(locus.value().acquisitionTime.has_value());
        EXPECT_DOUBLE_EQ(locus.value().acquisitionTime->seconds, 42.0);
    }

    TEST(FramePinholeModelTest, InvertsBrownConradyDistortion)
    {
        BrownConradyDistortion distortion;
        distortion.radialK1 = 0.03;
        distortion.radialK2 = -0.002;
        distortion.tangentialP1 = 0.0015;
        distortion.tangentialP2 = -0.0007;
        const FramePinholeModel model = makeModel(makeDefinition(false, 1, 1, distortion));

        const GroundCoordinate expected{FrameId("world"), {0.4, -0.3, 5.0}};
        const auto projected = model.groundToImage(expected);
        ASSERT_TRUE(projected) << projected.message();
        const auto normalized = model.pinholeDefinition().undistortPixel(projected.value().image);
        ASSERT_TRUE(normalized) << normalized.message();
        EXPECT_NEAR(normalized.value()[0], expected.position[0] / expected.position[2], 1.0e-8);
        EXPECT_NEAR(normalized.value()[1], expected.position[1] / expected.position[2], 1.0e-8);
        EvaluationOptions bounded_options;
        bounded_options.requireInsideImage = true;
        const auto bounded = model.pinholeDefinition().undistortPixel(projected.value().image, bounded_options);
        EXPECT_FALSE(bounded);
        EXPECT_EQ(bounded.errorCode(), CameraErrorCode::InvalidArgument);
        const auto restored = model.imageToGroundAtDepth(projected.value().image, 5.0);
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_LT(restored.achievedPrecisionPixels(), 1.0e-7);
        EXPECT_NEAR(restored.value().position[0], expected.position[0], 1.0e-8);
        EXPECT_NEAR(restored.value().position[1], expected.position[1], 1.0e-8);
        EXPECT_NEAR(restored.value().position[2], expected.position[2], 1.0e-8);
    }

    TEST(FramePinholeModelTest, MatchesExtendedMetashapeFrameCalibration)
    {
        const FrameCalibration calibration{
            615.0, 512.25, 383.75, 2.0, -0.7, 0.012, -0.0015, 0.0002, -0.00003, 0.0004, -0.0003, 0.00007, -0.000006};
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("metashape-definition"), calibration, PixelConvention::PixelCenter, FrameId("world"));
        const FramePinholeModel model = makeModel(definition);
        const GroundCoordinate ground{FrameId("world"), {0.31, -0.19, 0.93}};
        const auto projected = model.groundToImage(ground);
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 718.547972308942, 1.0e-12);
        EXPECT_NEAR(projected.value().image.line, 257.8015871234144, 1.0e-12);

        const auto restored = model.imageToGroundAtDepth(projected.value().image, 0.93);
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_NEAR(restored.value().position[0], ground.position[0], 1.0e-10);
        EXPECT_NEAR(restored.value().position[1], ground.position[1], 1.0e-10);
        EXPECT_NEAR(restored.value().position[2], ground.position[2], 1.0e-12);
        EXPECT_EQ(definition->distortion().tangentialConvention, BrownTangentialConvention::Metashape);
        EXPECT_DOUBLE_EQ(definition->intrinsics().focalX, calibration.f + calibration.b1);
        EXPECT_DOUBLE_EQ(definition->intrinsics().skew, calibration.b2);
        EXPECT_DOUBLE_EQ(definition->calibration().p4, calibration.p4);
    }

    TEST(FramePinholeDefinitionTest, PreservesExactImageCenterAndPrincipalOffsets)
    {
        FrameCalibration calibration;
        calibration.f = 1345.2434677738593;
        calibration.principalPointDecomposition =
            PrincipalPointDecomposition{400.0, 300.0, 287.1311760596414, -84.61526522602229};
        calibration.cx =
            calibration.principalPointDecomposition->imageCenterX + calibration.principalPointDecomposition->cxOffset;
        calibration.cy =
            calibration.principalPointDecomposition->imageCenterY + calibration.principalPointDecomposition->cyOffset;
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("exact-principal"), calibration, PixelConvention::PixelCenter, FrameId("world"));
        const ImageCoordinate pixel{595.1800537109375, 498.72274780273438};
        const auto normalized = definition->undistortPixel(pixel);
        ASSERT_TRUE(normalized) << normalized.message();
        const double expected_x =
            ((pixel.sample - 400.0) - calibration.principalPointDecomposition->cxOffset) / calibration.f;
        const double expected_y =
            ((pixel.line - 300.0) - calibration.principalPointDecomposition->cyOffset) / calibration.f;
        EXPECT_DOUBLE_EQ(normalized.value()[0], expected_x);
        EXPECT_DOUBLE_EQ(normalized.value()[1], expected_y);
        EXPECT_NE(normalized.value()[0], (pixel.sample - calibration.cx) / calibration.f);

        FrameCalibration stale = calibration;
        stale.cx = std::nextafter(stale.cx, std::numeric_limits<double>::infinity());
        stale.cy = std::nextafter(stale.cy, std::numeric_limits<double>::infinity());
        const auto stale_definition = FramePinholeDefinition::create(
            CameraDefinitionId("stale-principal"), stale, PixelConvention::PixelCenter, FrameId("world"));
        const auto stale_normalized = stale_definition->undistortPixel(pixel);
        ASSERT_TRUE(stale_normalized) << stale_normalized.message();
        EXPECT_DOUBLE_EQ(stale_normalized.value()[0], (pixel.sample - stale.cx) / stale.f);
        EXPECT_DOUBLE_EQ(stale_normalized.value()[1], (pixel.line - stale.cy) / stale.f);
    }

    TEST(FramePinholeDefinitionTest, ScalesAndOptimizesPrincipalOffsetsWithoutLosingImageCenter)
    {
        FrameCalibration calibration;
        calibration.f = 800.0;
        calibration.principalPointDecomposition = PrincipalPointDecomposition{400.0, 300.0, 12.25, -7.75};
        calibration.cx = 412.25;
        calibration.cy = 292.25;
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("principal-source"), calibration, PixelConvention::PixelCenter, FrameId("world"));

        const auto scaled = definition->scaledIntrinsics(CameraDefinitionId("principal-scaled"), 0.5, 2.0);
        ASSERT_TRUE(scaled->principalPointDecomposition());
        const PrincipalPointDecomposition& scaled_principal = *scaled->principalPointDecomposition();
        EXPECT_DOUBLE_EQ(scaled_principal.imageCenterX, (400.0 + 0.5) * 0.5 - 0.5);
        EXPECT_DOUBLE_EQ(scaled_principal.imageCenterY, (300.0 + 0.5) * 2.0 - 0.5);
        EXPECT_DOUBLE_EQ(scaled_principal.cxOffset, 12.25 * 0.5);
        EXPECT_DOUBLE_EQ(scaled_principal.cyOffset, -7.75 * 2.0);
        EXPECT_DOUBLE_EQ(scaled->intrinsics().principalX, scaled_principal.imageCenterX + scaled_principal.cxOffset);
        EXPECT_DOUBLE_EQ(scaled->intrinsics().principalY, scaled_principal.imageCenterY + scaled_principal.cyOffset);

        FrameOptimizationSelection selection = FrameOptimizationSelection::none();
        selection.set(FrameOptimizationParameter::Cx).set(FrameOptimizationParameter::Cy);
        const OptimizationUpdate update{
            CameraInstanceId("principal-updated"), CameraDefinitionId("principal-updated-definition"), {0.25, -0.5}};
        const auto updated = makeModel(definition).withOptimizationUpdate(update, selection);
        ASSERT_TRUE(updated) << updated.message();
        const auto* frame = dynamic_cast<const FramePinholeModel*>(updated.value().get());
        ASSERT_NE(frame, nullptr);
        const FrameCalibration updated_calibration = frame->pinholeDefinition().calibration();
        ASSERT_TRUE(updated_calibration.principalPointDecomposition);
        EXPECT_DOUBLE_EQ(updated_calibration.principalPointDecomposition->imageCenterX, 400.0);
        EXPECT_DOUBLE_EQ(updated_calibration.principalPointDecomposition->imageCenterY, 300.0);
        EXPECT_DOUBLE_EQ(updated_calibration.principalPointDecomposition->cxOffset, 12.5);
        EXPECT_DOUBLE_EQ(updated_calibration.principalPointDecomposition->cyOffset, -8.25);
        EXPECT_DOUBLE_EQ(updated_calibration.cx, 412.5);
        EXPECT_DOUBLE_EQ(updated_calibration.cy, 291.75);
    }

    TEST(FramePinholeModelTest, KeepsLegacyOpenCvTangentialConventionStable)
    {
        BrownConradyDistortion legacy;
        legacy.tangentialP1 = 0.0015;
        legacy.tangentialP2 = -0.0007;
        const auto legacy_definition = makeDefinition(false, 1, 1, legacy);
        FrameCalibration equivalent = legacy_definition->calibration();
        const auto extended_definition = FramePinholeDefinition::create(CameraDefinitionId("extended-equivalent"),
                                                                        equivalent,
                                                                        PixelConvention::PixelCenter,
                                                                        FrameId("world"),
                                                                        false,
                                                                        0.01);
        const GroundCoordinate ground{FrameId("world"), {0.4, -0.3, 5.0}};
        const auto legacy_projection = makeModel(legacy_definition).groundToImage(ground);
        const auto extended_projection = makeModel(extended_definition).groundToImage(ground);
        ASSERT_TRUE(legacy_projection);
        ASSERT_TRUE(extended_projection);
        EXPECT_DOUBLE_EQ(legacy_projection.value().image.sample, extended_projection.value().image.sample);
        EXPECT_DOUBLE_EQ(legacy_projection.value().image.line, extended_projection.value().image.line);

        legacy.tangentialP3 = 0.1;
        EXPECT_THROW(makeDefinition(false, 1, 1, legacy), CameraValidationError);
    }

    TEST(FramePinholeModelTest, MatchesMetashapeProjectionFamiliesAndInverseRays)
    {
        FrameCalibration calibration;
        calibration.f = 400.0;
        calibration.cx = 407.5;
        calibration.cy = 295.75;
        struct Expectation
        {
            FrameProjectionModel model;
            double sample;
            double line;
        };
        const std::array<Expectation, 6> expectations{{
            {FrameProjectionModel::Perspective, 507.5, 255.75},
            {FrameProjectionModel::Fisheye, 505.1833041813559, 256.67667832745764},
            {FrameProjectionModel::EquidistantFisheye, 505.1833041813559, 256.67667832745764},
            {FrameProjectionModel::EquisolidFisheye, 504.9019766555284, 256.78920933778863},
            {FrameProjectionModel::Spherical, 505.49146525074565, 257.06536046298834},
            {FrameProjectionModel::Cylindrical, 505.49146525074565, 256.94429999418674},
        }};
        const Vector3 camera_point{0.25, -0.1, 1.0};
        const double norm = std::sqrt(camera_point[0] * camera_point[0] + camera_point[1] * camera_point[1] +
                                      camera_point[2] * camera_point[2]);

        for (const Expectation& expectation : expectations)
        {
            const auto definition = FramePinholeDefinition::create(CameraDefinitionId("projection-definition"),
                                                                   calibration,
                                                                   PixelConvention::PixelCenter,
                                                                   FrameId("world"),
                                                                   false,
                                                                   1.0,
                                                                   1,
                                                                   1,
                                                                   expectation.model);
            const FramePinholeModel model = makeModel(definition);
            const auto projected = model.groundToImage(GroundCoordinate{FrameId("world"), camera_point});
            ASSERT_TRUE(projected) << projected.message();
            EXPECT_NEAR(projected.value().image.sample, expectation.sample, 1.0e-12);
            EXPECT_NEAR(projected.value().image.line, expectation.line, 1.0e-12);
            EXPECT_EQ(model.modelType(), frameProjectionModelName(expectation.model));

            const auto locus = model.imageToImagingLocus(projected.value().image);
            ASSERT_TRUE(locus) << locus.message();
            EXPECT_NEAR(locus.value().direction[0], camera_point[0] / norm, 1.0e-10);
            EXPECT_NEAR(locus.value().direction[1], camera_point[1] / norm, 1.0e-10);
            EXPECT_NEAR(locus.value().direction[2], camera_point[2] / norm, 1.0e-10);
        }
    }

    TEST(FramePinholeModelTest, DistinguishesFrontOnlyAndFullSphereFisheyeDomains)
    {
        FrameCalibration calibration;
        calibration.f = 400.0;
        calibration.cx = 407.5;
        calibration.cy = 295.75;
        const auto make_projection = [&](FrameProjectionModel projection)
        {
            return makeModel(FramePinholeDefinition::create(CameraDefinitionId("fisheye-domain"),
                                                            calibration,
                                                            PixelConvention::PixelCenter,
                                                            FrameId("world"),
                                                            false,
                                                            1.0,
                                                            1,
                                                            1,
                                                            projection));
        };
        const GroundCoordinate rear{FrameId("world"), {-1.0, 0.0, -0.5}};
        EXPECT_FALSE(make_projection(FrameProjectionModel::Fisheye).groundToImage(rear));

        const auto equidistant = make_projection(FrameProjectionModel::EquidistantFisheye).groundToImage(rear);
        ASSERT_TRUE(equidistant) << equidistant.message();
        EXPECT_NEAR(equidistant.value().image.sample, -406.27757431828104, 1.0e-12);
        EXPECT_NEAR(equidistant.value().image.line, 295.75, 1.0e-12);
    }

    TEST(FramePinholeModelTest, RejectsPixelsOutsideEachInverseProjectionDomain)
    {
        FrameCalibration calibration;
        calibration.f = 100.0;
        const auto model_for = [&](FrameProjectionModel projection)
        {
            return makeModel(FramePinholeDefinition::create(CameraDefinitionId("inverse-domain"),
                                                            calibration,
                                                            PixelConvention::PixelCenter,
                                                            FrameId("world"),
                                                            false,
                                                            1.0,
                                                            1,
                                                            1,
                                                            projection));
        };

        EXPECT_EQ(model_for(FrameProjectionModel::Fisheye).imageToImagingLocus(ImageCoordinate{160.0, 0.0}).errorCode(),
                  CameraErrorCode::OutsideModelDomain);
        EXPECT_EQ(model_for(FrameProjectionModel::EquidistantFisheye)
                      .imageToImagingLocus(ImageCoordinate{315.0, 0.0})
                      .errorCode(),
                  CameraErrorCode::OutsideModelDomain);
        EXPECT_EQ(model_for(FrameProjectionModel::EquisolidFisheye)
                      .imageToImagingLocus(ImageCoordinate{200.0, 0.0})
                      .errorCode(),
                  CameraErrorCode::OutsideModelDomain);
        EXPECT_EQ(
            model_for(FrameProjectionModel::Spherical).imageToImagingLocus(ImageCoordinate{0.0, 158.0}).errorCode(),
            CameraErrorCode::OutsideModelDomain);
        EXPECT_EQ(
            model_for(FrameProjectionModel::Cylindrical).imageToImagingLocus(ImageCoordinate{315.0, 0.0}).errorCode(),
            CameraErrorCode::OutsideModelDomain);
    }

    TEST(FramePinholeModelTest, MatchesMetashapeRollingShutterProjectionAndInverseRay)
    {
        FrameCalibration calibration;
        calibration.f = 700.0;
        calibration.cx = 400.0;
        calibration.cy = 300.0;
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("rolling-definition"), calibration, PixelConvention::PixelCenter, FrameId("world"));
        CameraAcquisitionState acquisition;
        acquisition.rollingShutterMode = RollingShutterMode::Full;
        acquisition.rollingShutter.translation = {0.04, -0.025, 0.012};
        acquisition.rollingShutter.rotationVector = {0.0012, -0.0008, 0.0005};
        const FramePinholeModel model = FramePinholeModel::create(
            CameraInstanceId("rolling-instance"),
            ImageId("rolling-image"),
            definition,
            ImageSize{1000, 600},
            Pose::create(FrameId("world"), {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
            std::nullopt,
            acquisition);
        const GroundCoordinate ground{FrameId("world"), {0.25, -0.1, 1.0}};

        const auto projected = model.groundToImage(ground);
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 581.4944620108606, 1.0e-10);
        EXPECT_NEAR(projected.value().image.line, 226.09080567129956, 1.0e-10);
        const auto restored = model.imageToGroundAtDepth(projected.value().image, 1.0);
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_NEAR(restored.value().position[0], ground.position[0], 1.0e-8);
        EXPECT_NEAR(restored.value().position[1], ground.position[1], 1.0e-8);
        EXPECT_NEAR(restored.value().position[2], ground.position[2], 1.0e-12);

        const FramePinholeNumericState numeric = FramePinholeNumericState::fromModel(model);
        const auto numeric_projection = numeric.groundToImage(ground);
        ASSERT_TRUE(numeric_projection) << numeric_projection.message();
        EXPECT_DOUBLE_EQ(numeric_projection.value().image.sample, projected.value().image.sample);
        EXPECT_DOUBLE_EQ(numeric_projection.value().image.line, projected.value().image.line);
    }

    TEST(FramePinholeModelTest, PreservesProjectionWhenNormalizingPositiveDepth)
    {
        BrownConradyDistortion distortion;
        distortion.radialK1 = 0.03;
        distortion.radialK2 = -0.002;
        distortion.tangentialP1 = 0.0015;
        distortion.tangentialP2 = -0.0007;
        const FramePinholeModel model = makeModel(makeDefinition(true, -1, -1, distortion));
        const GroundCoordinate ground{FrameId("world"), {0.4, -0.3, -5.0}};

        const auto original = model.groundToImage(ground);
        ASSERT_TRUE(original) << original.message();
        const FramePinholeModel normalized = model.normalizedForPositiveDepth(
            CameraDefinitionId("definition-normalized"), CameraInstanceId("instance-normalized"));
        const auto reprojected = normalized.groundToImage(ground);
        ASSERT_TRUE(reprojected) << reprojected.message();
        EXPECT_NEAR(reprojected.value().image.sample, original.value().image.sample, 1.0e-9);
        EXPECT_NEAR(reprojected.value().image.line, original.value().image.line, 1.0e-9);
        EXPECT_FALSE(normalized.pinholeDefinition().depthAxisFlipped());
    }

    TEST(FramePinholeModelTest, ReportsFrameAndDomainFailuresWithoutThrowing)
    {
        const FramePinholeModel model = makeModel(makeDefinition());
        const auto mismatch = model.groundToImage(GroundCoordinate{FrameId("other"), {0.0, 0.0, 1.0}});
        EXPECT_FALSE(mismatch);
        EXPECT_EQ(mismatch.errorCode(), CameraErrorCode::FrameMismatch);

        const auto behind = model.groundToImage(GroundCoordinate{FrameId("world"), {0.0, 0.0, -1.0}});
        EXPECT_FALSE(behind);
        EXPECT_EQ(behind.errorCode(), CameraErrorCode::OutsideModelDomain);

        EvaluationOptions options;
        options.requireInsideImage = true;
        const auto outside = model.imageToImagingLocus(ImageCoordinate{-1.0, 10.0}, options);
        EXPECT_FALSE(outside);
        EXPECT_EQ(outside.errorCode(), CameraErrorCode::OutsideModelDomain);
    }

    TEST(FramePinholeDefinitionTest, ScalesPrincipalPointAccordingToPixelConvention)
    {
        const auto center = makeDefinition(false, 1, 1, {}, PixelConvention::PixelCenter);
        const auto scaled_center = center->scaledIntrinsics(CameraDefinitionId("scaled-center"), 2.0, 3.0);
        EXPECT_DOUBLE_EQ(scaled_center->intrinsics().principalX, 100.5);
        EXPECT_DOUBLE_EQ(scaled_center->intrinsics().principalY, 181.0);

        const auto corner = makeDefinition(false, 1, 1, {}, PixelConvention::PixelCorner);
        const auto scaled_corner = corner->scaledIntrinsics(CameraDefinitionId("scaled-corner"), 2.0, 3.0);
        EXPECT_DOUBLE_EQ(scaled_corner->intrinsics().principalX, 100.0);
        EXPECT_DOUBLE_EQ(scaled_corner->intrinsics().principalY, 180.0);
    }

    TEST(FramePinholeDefinitionTest, RejectsInvalidIntrinsics)
    {
        FrameIntrinsics intrinsics;
        intrinsics.focalX = -1.0;
        intrinsics.focalY = 100.0;
        EXPECT_THROW(FramePinholeDefinition::create(
                         CameraDefinitionId("invalid"), intrinsics, {}, PixelConvention::PixelCenter, FrameId("world")),
                     CameraValidationError);

        FrameCalibration calibration;
        calibration.f = 100.0;
        EXPECT_THROW(FramePinholeDefinition::create(CameraDefinitionId("invalid-fisheye-axis"),
                                                    calibration,
                                                    PixelConvention::PixelCenter,
                                                    FrameId("world"),
                                                    true,
                                                    1.0,
                                                    1,
                                                    1,
                                                    FrameProjectionModel::Fisheye),
                     CameraValidationError);
    }

} // namespace

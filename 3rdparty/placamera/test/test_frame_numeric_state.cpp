#include <placamera/frame_numeric_state.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace
{

    using namespace placamera;

    std::shared_ptr<const FramePinholeDefinition> makeDefinition()
    {
        FrameIntrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 100.0;
        intrinsics.principalX = 50.0;
        intrinsics.principalY = 50.0;
        return FramePinholeDefinition::create(
            CameraDefinitionId("definition"), intrinsics, {}, PixelConvention::PixelCenter, FrameId("world"));
    }

    FramePinholeModel makeModel(std::string instance, std::string image, Vector3 center)
    {
        return FramePinholeModel::create(
            CameraInstanceId(std::move(instance)),
            ImageId(std::move(image)),
            makeDefinition(),
            ImageSize{100, 100},
            Pose::create(FrameId("world"), center, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }

    TEST(FramePinholeNumericStateTest, AppliesPoseAndPromotesBackToTypedModel)
    {
        FramePinholeNumericState state =
            FramePinholeNumericState::fromModel(makeModel("instance", "image", {0.0, 0.0, 0.0}));
        EXPECT_FALSE(state.definitionDirty());
        state.applyPoseDelta({0.0, 0.0, 0.0, 1.0, 2.0, 3.0});
        EXPECT_EQ(state.pose().center, (Vector3{1.0, 2.0, 3.0}));

        const auto promoted = state.toModel(CameraInstanceId("updated"));
        ASSERT_TRUE(promoted) << promoted.message();
        EXPECT_EQ(promoted.value()->instanceId(), CameraInstanceId("updated"));
        EXPECT_EQ(promoted.value()->definitionId(), CameraDefinitionId("definition"));
        EXPECT_EQ(promoted.value()->pose().center, (Vector3{1.0, 2.0, 3.0}));
    }

    TEST(FramePinholeNumericStateTest, RequiresNewDefinitionIdentityForCalibrationChanges)
    {
        FramePinholeNumericState state =
            FramePinholeNumericState::fromModel(makeModel("instance", "image", {0.0, 0.0, 0.0}));
        FrameIntrinsics intrinsics = state.intrinsics();
        intrinsics.focalX = 120.0;
        state.setIntrinsics(intrinsics);
        EXPECT_TRUE(state.definitionDirty());
        EXPECT_FALSE(state.toModel(CameraInstanceId("missing-definition")));

        const auto promoted = state.toModel(CameraInstanceId("updated"), CameraDefinitionId("updated-definition"));
        ASSERT_TRUE(promoted) << promoted.message();
        EXPECT_EQ(promoted.value()->definitionId(), CameraDefinitionId("updated-definition"));
        EXPECT_DOUBLE_EQ(promoted.value()->pinholeDefinition().intrinsics().focalX, 120.0);
    }

    TEST(FramePinholeNumericStateTest, TriangulatesPairAndReportsReprojectionRms)
    {
        const FramePinholeNumericState left =
            FramePinholeNumericState::fromModel(makeModel("left-instance", "left-image", {0.0, 0.0, 0.0}));
        const FramePinholeNumericState right =
            FramePinholeNumericState::fromModel(makeModel("right-instance", "right-image", {1.0, 0.0, 0.0}));
        const GroundCoordinate expected{FrameId("world"), {0.0, 0.0, 10.0}};
        const auto left_image = left.groundToImage(expected);
        const auto right_image = right.groundToImage(expected);
        ASSERT_TRUE(left_image);
        ASSERT_TRUE(right_image);

        const auto intersection =
            FramePinholeNumericState::triangulatePair(left, left_image.value().image, right, right_image.value().image);
        ASSERT_TRUE(intersection) << intersection.message();
        EXPECT_NEAR(intersection.value().point.position[0], expected.position[0], 1.0e-10);
        EXPECT_NEAR(intersection.value().point.position[1], expected.position[1], 1.0e-10);
        EXPECT_NEAR(intersection.value().point.position[2], expected.position[2], 1.0e-9);
        EXPECT_NEAR(
            intersection.value().triangulationAngleDegrees, std::atan(0.1) * 180.0 / 3.14159265358979323846, 1.0e-10);
        EXPECT_NEAR(intersection.value().rayMissDistance, 0.0, 1.0e-10);
        EXPECT_NEAR(intersection.value().firstReprojectionPixels, 0.0, 1.0e-10);
        EXPECT_NEAR(intersection.value().secondReprojectionPixels, 0.0, 1.0e-10);
        EXPECT_NEAR(intersection.value().rmsReprojectionPixels, 0.0, 1.0e-10);
    }

    TEST(FramePinholeNumericStateTest, PairRmsUsesEuclideanResidualPerImage)
    {
        const FramePinholeNumericState left =
            FramePinholeNumericState::fromModel(makeModel("left-instance", "left-image", {0.0, 0.0, 0.0}));
        const FramePinholeNumericState right =
            FramePinholeNumericState::fromModel(makeModel("right-instance", "right-image", {1.0, 0.0, 0.0}));
        const GroundCoordinate point{FrameId("world"), {0.0, 0.0, 10.0}};
        const auto left_image = left.groundToImage(point);
        const auto right_image = right.groundToImage(point);
        ASSERT_TRUE(left_image);
        ASSERT_TRUE(right_image);

        ImageCoordinate perturbed = right_image.value().image;
        perturbed.line += 0.5;
        const auto intersection =
            FramePinholeNumericState::triangulatePair(left, left_image.value().image, right, perturbed);
        ASSERT_TRUE(intersection) << intersection.message();
        const double first_error = intersection.value().firstReprojectionPixels;
        const double second_error = intersection.value().secondReprojectionPixels;
        EXPECT_GT(intersection.value().rayMissDistance, 0.0);
        EXPECT_NEAR(intersection.value().rmsReprojectionPixels,
                    std::sqrt(0.5 * (first_error * first_error + second_error * second_error)),
                    1.0e-12);
    }

    TEST(FramePinholeNumericStateTest, PairRejectsRaysBehindCameras)
    {
        const FramePinholeNumericState left =
            FramePinholeNumericState::fromModel(makeModel("left-instance", "left-image", {0.0, 0.0, 0.0}));
        const FramePinholeNumericState right =
            FramePinholeNumericState::fromModel(makeModel("right-instance", "right-image", {1.0, 0.0, 0.0}));
        // Two rays diverging outward from the baseline cannot meet in front of both cameras.
        const auto intersection = FramePinholeNumericState::triangulatePair(
            left, ImageCoordinate{40.0, 50.0}, right, ImageCoordinate{60.0, 50.0});
        EXPECT_FALSE(intersection);
    }

    TEST(FramePinholeNumericStateTest, PairRejectsMixedGroundFrames)
    {
        const FramePinholeNumericState left =
            FramePinholeNumericState::fromModel(makeModel("left-instance", "left-image", {0.0, 0.0, 0.0}));
        FrameIntrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 100.0;
        const FrameId other_frame("other-world");
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("other-definition"), intrinsics, {}, PixelConvention::PixelCenter, other_frame);
        const FramePinholeModel other_model = FramePinholeModel::create(
            CameraInstanceId("other-instance"),
            ImageId("other-image"),
            definition,
            ImageSize{100, 100},
            Pose::create(other_frame, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        const FramePinholeNumericState right = FramePinholeNumericState::fromModel(other_model);

        const auto intersection = FramePinholeNumericState::triangulatePair(
            left, ImageCoordinate{50.0, 50.0}, right, ImageCoordinate{50.0, 50.0});
        EXPECT_FALSE(intersection);
        EXPECT_EQ(intersection.errorCode(), CameraErrorCode::FrameMismatch);
    }

    TEST(FramePinholeNumericStateTest, MatchesTypedModelForDistortedFlippedCamera)
    {
        FrameIntrinsics intrinsics{120.0, 125.0, 50.0, 45.0, 1.0, -1, 1};
        BrownConradyDistortion distortion{0.01, -0.002, 0.0001, 0.001, -0.0005};
        const FrameId frame("world");
        const auto definition = FramePinholeDefinition::create(CameraDefinitionId("distorted-definition"),
                                                               intrinsics,
                                                               distortion,
                                                               PixelConvention::PixelCenter,
                                                               frame,
                                                               true);
        const FramePinholeModel model = FramePinholeModel::create(
            CameraInstanceId("distorted-instance"),
            ImageId("distorted-image"),
            definition,
            ImageSize{100, 100},
            Pose::create(frame, {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        const FramePinholeNumericState state = FramePinholeNumericState::fromModel(model);
        const GroundCoordinate ground{frame, {2.0, 2.5, -7.0}};

        const auto model_projection = model.groundToImage(ground);
        const auto state_projection = state.groundToImage(ground);
        ASSERT_TRUE(model_projection);
        ASSERT_TRUE(state_projection);
        EXPECT_NEAR(state_projection.value().image.sample, model_projection.value().image.sample, 1.0e-12);
        EXPECT_NEAR(state_projection.value().image.line, model_projection.value().image.line, 1.0e-12);
        EXPECT_EQ(state_projection.value().positiveDepth, model_projection.value().positiveDepth);

        const auto model_ground = model.imageToGroundAtDepth(model_projection.value().image, 10.0);
        const auto state_ground = state.imageToGroundAtDepth(model_projection.value().image, 10.0);
        ASSERT_TRUE(model_ground);
        ASSERT_TRUE(state_ground);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(state_ground.value().position[axis], model_ground.value().position[axis], 1.0e-12);
            EXPECT_NEAR(state_ground.value().position[axis], ground.position[axis], 1.0e-8);
        }
        EXPECT_EQ(state.imageToGroundAtDepth(model_projection.value().image, -1.0).errorCode(),
                  CameraErrorCode::InvalidArgument);

        const auto model_ray = model.imageToImagingLocus(model_projection.value().image);
        const auto state_ray = state.imageToImagingLocus(model_projection.value().image);
        ASSERT_TRUE(model_ray);
        ASSERT_TRUE(state_ray);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(state_ray.value().direction[axis], model_ray.value().direction[axis], 1.0e-12);
        }
        EXPECT_EQ(state.groundToImage({FrameId("other-frame"), ground.position}).errorCode(),
                  CameraErrorCode::FrameMismatch);
    }

    TEST(FramePinholeNumericStateTest, SignedDepthMatchesTypedModelOnBothSidesOfCamera)
    {
        const FrameId frame("world");
        for (const bool flipped : {false, true})
        {
            const auto definition = FramePinholeDefinition::create(CameraDefinitionId("depth-definition"),
                                                                   FrameIntrinsics{100.0, 100.0, 50.0, 50.0},
                                                                   {},
                                                                   PixelConvention::PixelCenter,
                                                                   frame,
                                                                   flipped);
            const auto model = FramePinholeModel::create(
                CameraInstanceId("depth-instance"),
                ImageId("depth-image"),
                definition,
                {100, 100},
                Pose::create(frame, {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
            const auto state = FramePinholeNumericState::fromModel(model);
            for (const double z : {-7.0, 13.0})
            {
                const GroundCoordinate ground{frame, {1.0, 2.0, z}};
                const auto typed_depth = model.signedDepth(ground);
                const auto numeric_depth = state.signedDepth(ground);
                ASSERT_TRUE(typed_depth);
                ASSERT_TRUE(numeric_depth);
                EXPECT_DOUBLE_EQ(numeric_depth.value(), typed_depth.value());
            }
            EXPECT_EQ(state.signedDepth({FrameId("other-frame"), {1.0, 2.0, 13.0}}).errorCode(),
                      CameraErrorCode::FrameMismatch);
            EXPECT_EQ(state.signedDepth({frame, {1.0, 2.0, std::numeric_limits<double>::quiet_NaN()}}).errorCode(),
                      CameraErrorCode::InvalidArgument);
        }
    }

    TEST(FramePinholeNumericStateTest, EvaluatesModifiedStateBeforeDefinitionWriteback)
    {
        FramePinholeNumericState state =
            FramePinholeNumericState::fromModel(makeModel("working-instance", "image", {0.0, 0.0, 0.0}));
        FrameIntrinsics intrinsics = state.intrinsics();
        intrinsics.focalX = 130.0;
        state.setIntrinsics(intrinsics);
        state.applyPoseDelta({0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
        EXPECT_TRUE(state.definitionDirty());

        const GroundCoordinate ground{FrameId("world"), {2.0, 0.0, 10.0}};
        const auto projected = state.groundToImage(ground);
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 63.0, 1.0e-12);

        const auto committed =
            state.toModel(CameraInstanceId("result-instance"), CameraDefinitionId("result-definition"));
        ASSERT_TRUE(committed) << committed.message();
        const auto committed_projection = committed.value()->groundToImage(ground);
        ASSERT_TRUE(committed_projection) << committed_projection.message();
        EXPECT_EQ(projected.value().image.sample, committed_projection.value().image.sample);
        EXPECT_EQ(projected.value().image.line, committed_projection.value().image.line);
    }

    TEST(FramePinholeNumericStateTest, ScalesBoundImageGridAndCalibrationTogether)
    {
        const FramePinholeNumericState state =
            FramePinholeNumericState::fromModel(makeModel("scale-instance", "scale-image", {1.0, 2.0, 3.0}));
        const auto scaled = state.scaledIntrinsics(2.0, 0.5);
        EXPECT_EQ(scaled.instanceId(), state.instanceId());
        EXPECT_EQ(scaled.imageId(), state.imageId());
        EXPECT_EQ(scaled.groundFrame(), state.groundFrame());
        EXPECT_EQ(scaled.pose().center, state.pose().center);
        EXPECT_EQ(scaled.imageSize().samples, 200);
        EXPECT_EQ(scaled.imageSize().lines, 50);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().focalX, 200.0);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().focalY, 50.0);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().principalX, 100.5);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().principalY, 24.75);
        EXPECT_TRUE(scaled.definitionDirty());
        EXPECT_FALSE(scaled.toModel(CameraInstanceId("scaled-without-definition")));

        const auto committed =
            scaled.toModel(CameraInstanceId("scaled-instance"), CameraDefinitionId("scaled-definition"));
        ASSERT_TRUE(committed) << committed.message();
        EXPECT_EQ(committed.value()->imageSize().samples, 200);
        EXPECT_EQ(committed.value()->pinholeDefinition().intrinsics().principalX, 100.5);
        EXPECT_EQ(state.imageSize().samples, 100);
        EXPECT_FALSE(state.definitionDirty());

        EXPECT_THROW(state.scaledIntrinsics(0.0, 1.0), CameraValidationError);
        EXPECT_THROW(state.scaledIntrinsics(std::numeric_limits<double>::infinity(), 1.0), CameraValidationError);
        EXPECT_THROW(state.scaledIntrinsics(1.0e-4, 1.0), CameraValidationError);
        EXPECT_THROW(state.scaledIntrinsics(1.0e30, 1.0), CameraValidationError);
    }

    TEST(FramePinholeNumericStateTest, ScalingRespectsPixelCornerConvention)
    {
        FrameIntrinsics intrinsics{80.0, 90.0, 25.0, 40.0};
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("corner-definition"), intrinsics, {}, PixelConvention::PixelCorner, FrameId("world"));
        const auto model = FramePinholeModel::create(
            CameraInstanceId("corner-instance"),
            ImageId("corner-image"),
            definition,
            {100, 80},
            Pose::create(FrameId("world"), {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        const auto scaled = FramePinholeNumericState::fromModel(model).scaledIntrinsics(2.0, 0.5);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().principalX, 50.0);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().principalY, 20.0);
        EXPECT_EQ(scaled.imageSize().samples, 200);
        EXPECT_EQ(scaled.imageSize().lines, 40);
    }

    TEST(FramePinholeNumericStateTest, PreservesPrincipalDecompositionThroughScalingAndWriteback)
    {
        FrameCalibration calibration;
        calibration.f = 100.0;
        calibration.cx = 62.25;
        calibration.cy = 44.5;
        calibration.principalPointDecomposition = PrincipalPointDecomposition{50.0, 50.0, 12.25, -5.5};
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("decomposed-definition"), calibration, PixelConvention::PixelCenter, FrameId("world"));
        const auto model = FramePinholeModel::create(
            CameraInstanceId("decomposed-instance"),
            ImageId("decomposed-image"),
            definition,
            ImageSize{100, 100},
            Pose::create(FrameId("world"), {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));

        const FramePinholeNumericState state = FramePinholeNumericState::fromModel(model);
        ASSERT_TRUE(state.principalPointDecomposition());
        EXPECT_EQ(state.principalPointDecomposition(), calibration.principalPointDecomposition);

        const FramePinholeNumericState scaled = state.scaledIntrinsics(2.0, 0.5);
        ASSERT_TRUE(scaled.principalPointDecomposition());
        EXPECT_DOUBLE_EQ(scaled.principalPointDecomposition()->imageCenterX, 100.5);
        EXPECT_DOUBLE_EQ(scaled.principalPointDecomposition()->imageCenterY, 24.75);
        EXPECT_DOUBLE_EQ(scaled.principalPointDecomposition()->cxOffset, 24.5);
        EXPECT_DOUBLE_EQ(scaled.principalPointDecomposition()->cyOffset, -2.75);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().principalX, 125.0);
        EXPECT_DOUBLE_EQ(scaled.intrinsics().principalY, 22.0);

        const auto committed =
            scaled.toModel(CameraInstanceId("decomposed-scaled"), CameraDefinitionId("decomposed-scaled-definition"));
        ASSERT_TRUE(committed) << committed.message();
        const auto& restored = committed.value()->pinholeDefinition().principalPointDecomposition();
        ASSERT_TRUE(restored);
        EXPECT_EQ(restored, scaled.principalPointDecomposition());
        EXPECT_DOUBLE_EQ(committed.value()->pinholeDefinition().intrinsics().principalX, 125.0);
        EXPECT_DOUBLE_EQ(committed.value()->pinholeDefinition().intrinsics().principalY, 22.0);
    }

    TEST(FramePinholeNumericStateTest, NormalizedStateMatchesTypedModelForEveryAxisParity)
    {
        const FrameId frame("world");
        const BrownConradyDistortion distortion{0.01, -0.002, 0.0001, 0.001, -0.0005};
        for (bool flipped : {false, true})
        {
            for (int u_sign : {-1, 1})
            {
                for (int v_sign : {-1, 1})
                {
                    const FrameIntrinsics intrinsics{120.0, 125.0, 50.0, 45.0, 1.0, u_sign, v_sign};
                    const auto definition = FramePinholeDefinition::create(CameraDefinitionId("axis-definition"),
                                                                           intrinsics,
                                                                           distortion,
                                                                           PixelConvention::PixelCenter,
                                                                           frame,
                                                                           flipped);
                    const auto model = FramePinholeModel::create(
                        CameraInstanceId("axis-instance"),
                        ImageId("axis-image"),
                        definition,
                        {100, 100},
                        Pose::create(frame, {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
                    const auto state = FramePinholeNumericState::fromModel(model);
                    const auto normalized = state.normalizedForPositiveDepth();
                    const auto expected = model.normalizedForPositiveDepth(CameraDefinitionId("normalized-definition"),
                                                                           CameraInstanceId("normalized-instance"));
                    EXPECT_FALSE(normalized.depthAxisFlipped());
                    if (flipped)
                    {
                        EXPECT_TRUE(normalized.definitionDirty());
                        EXPECT_FALSE(normalized.toModel(CameraInstanceId("missing-definition")));
                    }
                    if (!flipped && u_sign == 1 && v_sign == 1)
                    {
                        EXPECT_FALSE(normalized.definitionDirty());
                    }
                    EXPECT_EQ(normalized.pose().cameraToWorldRotation, expected.pose().cameraToWorldRotation);
                    EXPECT_EQ(normalized.intrinsics().uAxisSign, expected.pinholeDefinition().intrinsics().uAxisSign);
                    EXPECT_EQ(normalized.intrinsics().vAxisSign, expected.pinholeDefinition().intrinsics().vAxisSign);
                    EXPECT_EQ(normalized.distortion().tangentialP1,
                              expected.pinholeDefinition().distortion().tangentialP1);
                    EXPECT_EQ(normalized.distortion().tangentialP2,
                              expected.pinholeDefinition().distortion().tangentialP2);

                    const GroundCoordinate ground{frame, {1.2, 1.7, flipped ? -7.0 : 13.0}};
                    const auto initial_projection = state.groundToImage(ground);
                    const auto normalized_projection = normalized.groundToImage(ground);
                    ASSERT_TRUE(initial_projection) << initial_projection.message();
                    ASSERT_TRUE(normalized_projection) << normalized_projection.message();
                    EXPECT_NEAR(
                        normalized_projection.value().image.sample, initial_projection.value().image.sample, 1.0e-9);
                    EXPECT_NEAR(
                        normalized_projection.value().image.line, initial_projection.value().image.line, 1.0e-9);

                    const auto committed = normalized.toModel(CameraInstanceId("committed-instance"),
                                                              CameraDefinitionId("committed-definition"));
                    ASSERT_TRUE(committed) << committed.message();
                    const auto committed_projection = committed.value()->groundToImage(ground);
                    ASSERT_TRUE(committed_projection) << committed_projection.message();
                    EXPECT_NEAR(
                        committed_projection.value().image.sample, normalized_projection.value().image.sample, 1.0e-12);
                    EXPECT_NEAR(
                        committed_projection.value().image.line, normalized_projection.value().image.line, 1.0e-12);
                }
            }
        }
    }

} // namespace

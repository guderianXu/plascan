#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <QtMath>

#include "CameraSceneViewMath.h"

using namespace xjw::gui::camera_scene;

TEST(CameraSceneViewMathTest, SortsFarCameraBeforeNearCameraRegardlessOfInputOrder)
{
    QMatrix4x4 world_to_view;
    world_to_view.setToIdentity();
    const QVector<QVector3D> centers{QVector3D(0.0f, 0.0f, -2.0f),
                                     QVector3D(0.0f, 0.0f, -9.0f)};

    EXPECT_EQ(farToNearCameraIndices(centers, world_to_view), QVector<int>({1, 0}));
}

TEST(CameraSceneViewMathTest, PreservesPortraitViewportAspectRatio)
{
    EXPECT_FLOAT_EQ(
        xjw::gui::camera_scene::sceneViewportAspectRatio(600, 1200),
        0.5f);
    EXPECT_FLOAT_EQ(
        xjw::gui::camera_scene::sceneViewportAspectRatio(1200, 600),
        2.0f);
    EXPECT_FLOAT_EQ(
        xjw::gui::camera_scene::sceneViewportAspectRatio(0, 0),
        1.0f);
}

TEST(CameraSceneViewMathTest, BoundsSceneZoomAgainstExtremeWheelInput)
{
    using namespace xjw::gui::camera_scene;

    EXPECT_DOUBLE_EQ(boundedSceneZoomScale(1.0, 1.1), 1.1);
    EXPECT_DOUBLE_EQ(
        boundedSceneZoomScale(1.0, std::numeric_limits<double>::max()),
        MaximumSceneZoomScale);
    EXPECT_DOUBLE_EQ(
        boundedSceneZoomScale(1.0, std::numeric_limits<double>::min()),
        MinimumSceneZoomScale);
    EXPECT_DOUBLE_EQ(
        boundedSceneZoomScale(8.0, -1.0),
        8.0);
    EXPECT_DOUBLE_EQ(
        boundedSceneZoomScale(8.0, 1.0, 2'000.0, -1.0),
        2'000.0);
}

TEST(CameraSceneViewMathTest, OrbitsAcrossCanvasWithoutUsingGizmoRadius)
{
    const QQuaternion identity;
    const QQuaternion yaw = orbitSceneViewRotation(
        identity, QVector2D(100.0f, 0.0f), 0.1f);
    const QQuaternion pitch = orbitSceneViewRotation(
        identity, QVector2D(0.0f, 100.0f), 0.1f);

    const QVector3D yawed = yaw.rotatedVector(QVector3D(0.0f, 0.0f, 1.0f));
    const QVector3D pitched = pitch.rotatedVector(QVector3D(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(yawed.x(), std::sin(qDegreesToRadians(10.0f)), 1.0e-5f);
    EXPECT_NEAR(yawed.y(), 0.0f, 1.0e-5f);
    EXPECT_NEAR(pitched.x(), 0.0f, 1.0e-5f);
    EXPECT_NEAR(pitched.y(), -std::sin(qDegreesToRadians(10.0f)), 1.0e-5f);
    EXPECT_NEAR(yaw.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(pitch.length(), 1.0f, 1.0e-5f);
}

TEST(CameraSceneViewMathTest, CancelsPerspectiveDepthAtFixedSceneZoom)
{
    constexpr int viewport_height = 1000;
    constexpr float half_fov_radians = 22.5f * 3.14159265358979323846f / 180.0f;
    auto projectedHalfExtentPixels = [viewport_height, half_fov_radians](
                                         float world_half_extent,
                                         float depth)
    {
        return world_half_extent * viewport_height
            / (2.0f * depth * std::tan(half_fov_radians));
    };

    QMatrix4x4 world_to_view;
    world_to_view.setToIdentity();
    const float near_extent = cameraPlaneHalfExtentForScreenSize(
        QVector3D(0.0f, 0.0f, -4.0f),
        world_to_view,
        viewport_height,
        1.0f);
    const float far_extent = cameraPlaneHalfExtentForScreenSize(
        QVector3D(0.0f, 0.0f, -16.0f),
        world_to_view,
        viewport_height,
        1.0f);

    EXPECT_NEAR(projectedHalfExtentPixels(near_extent, 4.0f), 34.0f, 1e-4f);
    EXPECT_NEAR(projectedHalfExtentPixels(far_extent, 16.0f), 34.0f, 1e-4f);
    EXPECT_NEAR(far_extent / near_extent, 4.0f, 1e-4f);
}

TEST(CameraSceneViewMathTest, MakesCameraCardsLargerAsSceneZoomsOut)
{
    constexpr int viewport_height = 1000;
    constexpr float camera_depth = 8.0f;
    constexpr float half_fov_radians = 22.5f * 3.14159265358979323846f / 180.0f;

    auto projectedHalfExtentPixels = [viewport_height, half_fov_radians](
                                         float world_half_extent,
                                         float depth)
    {
        return world_half_extent * viewport_height
            / (2.0f * depth * std::tan(half_fov_radians));
    };

    QMatrix4x4 world_to_view;
    world_to_view.setToIdentity();
    const QVector3D camera_center(0.0f, 0.0f, -camera_depth);
    const float normal_extent = cameraPlaneHalfExtentForScreenSize(
        camera_center,
        world_to_view,
        viewport_height,
        1.0f);
    const float zoomed_out_extent = cameraPlaneHalfExtentForScreenSize(
        camera_center,
        world_to_view,
        viewport_height,
        0.5f);
    const float zoomed_in_extent = cameraPlaneHalfExtentForScreenSize(
        camera_center,
        world_to_view,
        viewport_height,
        2.0f);

    EXPECT_NEAR(projectedHalfExtentPixels(normal_extent, camera_depth),
                34.0f,
                1e-4f);
    EXPECT_NEAR(projectedHalfExtentPixels(zoomed_out_extent, camera_depth),
                34.0f * std::pow(2.0f, 0.25f),
                1e-4f);
    EXPECT_NEAR(projectedHalfExtentPixels(zoomed_in_extent, camera_depth),
                34.0f / std::pow(2.0f, 0.25f),
                1e-4f);
    EXPECT_GT(zoomed_out_extent, normal_extent);
    EXPECT_LT(zoomed_in_extent, normal_extent);
}

TEST(CameraSceneViewMathTest, DoesNotClampZoomDrivenCameraScreenSize)
{
    constexpr int viewport_height = 1000;
    constexpr float half_fov_radians = 22.5f * 3.14159265358979323846f / 180.0f;

    auto projectedHalfExtentPixels = [viewport_height, half_fov_radians](
                                         float world_half_extent,
                                         float depth)
    {
        return world_half_extent * viewport_height
            / (2.0f * depth * std::tan(half_fov_radians));
    };

    QMatrix4x4 world_to_view;
    world_to_view.setToIdentity();
    const QVector3D camera_center(0.0f, 0.0f, -8.0f);
    const float deeply_zoomed_out_extent = cameraPlaneHalfExtentForScreenSize(
        camera_center,
        world_to_view,
        viewport_height,
        0.01);
    const float deeply_zoomed_in_extent = cameraPlaneHalfExtentForScreenSize(
        camera_center,
        world_to_view,
        viewport_height,
        100.0);

    EXPECT_NEAR(projectedHalfExtentPixels(deeply_zoomed_out_extent, 8.0f),
                34.0f * std::sqrt(10.0f),
                1e-3f);
    EXPECT_NEAR(projectedHalfExtentPixels(deeply_zoomed_in_extent, 8.0f),
                34.0f / std::sqrt(10.0f),
                1e-5f);
    EXPECT_NEAR(cameraPlaneScreenHalfExtentPixels(0.01),
                34.0 * std::sqrt(10.0),
                1e-10);
    EXPECT_NEAR(cameraPlaneScreenHalfExtentPixels(100.0),
                34.0 / std::sqrt(10.0),
                1e-12);
}

TEST(CameraSceneViewMathTest, RejectsInvalidCameraPlaneScreenSizeInputs)
{
    QMatrix4x4 world_to_view;
    world_to_view.setToIdentity();
    EXPECT_EQ(cameraPlaneHalfExtentForScreenSize(
                  QVector3D(0.0f, 0.0f, 1.0f),
                  world_to_view,
                  1000,
                  1.0f),
              0.0f);
    EXPECT_EQ(cameraPlaneHalfExtentForScreenSize(
                  QVector3D(0.0f, 0.0f, -4.0f),
                  world_to_view,
                  0,
                  1.0f),
              0.0f);
    EXPECT_EQ(cameraPlaneHalfExtentForScreenSize(
                  QVector3D(0.0f, 0.0f, -4.0f),
                  world_to_view,
                  1000,
                  0.0f),
              0.0f);
}

TEST(CameraSceneViewMathTest, BuildsPerspectiveIndependentScreenLeader)
{
    const QLineF leader = cameraPlaneLeaderLine(
        QPointF(10.0, 20.0), QPointF(1010.0, 20.0), 12.0, 50.0);

    EXPECT_FALSE(leader.isNull());
    EXPECT_NEAR(leader.p1().x(), 22.0, 1e-6);
    EXPECT_NEAR(leader.p1().y(), 20.0, 1e-6);
    EXPECT_NEAR(leader.p2().x(), 72.0, 1e-6);
    EXPECT_NEAR(leader.length(), 50.0, 1e-6);

    const QLineF closeProbeLeader = cameraPlaneLeaderLine(
        QPointF(10.0, 20.0), QPointF(11.0, 20.0), 12.0, 50.0);
    EXPECT_NEAR(closeProbeLeader.length(), leader.length(), 1e-6);
}

TEST(CameraSceneViewMathTest, RejectsLeaderWithoutScreenDirection)
{
    EXPECT_TRUE(cameraPlaneLeaderLine(
                    QPointF(4.0, 5.0),
                    QPointF(4.0, 5.0),
                    10.0,
                    40.0)
                    .isNull());

    EXPECT_TRUE(cameraPlaneLeaderLine(
                    QPointF(4.0, 5.0),
                    QPointF(8.0, 5.0),
                    -1.0,
                    40.0)
                    .isNull());
}

TEST(CameraSceneViewMathTest, DetectsPointHiddenBehindProjectedCameraQuad)
{
    const QVector<QVector3D> quad{
        QVector3D(-0.5f, -0.5f, 0.2f),
        QVector3D(0.5f, -0.5f, 0.2f),
        QVector3D(0.5f, 0.5f, 0.2f),
        QVector3D(-0.5f, 0.5f, 0.2f),
    };

    EXPECT_TRUE(pointIsBehindProjectedQuad(QVector3D(0.0f, 0.0f, 0.6f), quad));
    EXPECT_FALSE(pointIsBehindProjectedQuad(QVector3D(0.0f, 0.0f, -0.1f), quad));
    EXPECT_FALSE(pointIsBehindProjectedQuad(QVector3D(0.8f, 0.0f, 0.6f), quad));
}

TEST(CameraSceneViewMathTest, InterpolatesDepthAcrossTiltedCameraQuad)
{
    const QVector<QVector3D> quad{
        QVector3D(-1.0f, -1.0f, 0.1f),
        QVector3D(1.0f, -1.0f, 0.5f),
        QVector3D(1.0f, 1.0f, 0.5f),
        QVector3D(-1.0f, 1.0f, 0.1f),
    };

    EXPECT_TRUE(pointIsBehindProjectedQuad(QVector3D(0.0f, 0.0f, 0.4f), quad));
    EXPECT_FALSE(pointIsBehindProjectedQuad(QVector3D(0.0f, 0.0f, 0.2f), quad));
}

TEST(CameraSceneViewMathTest, ChoosesAvailableCameraWhoseForwardDirectionMatchesView)
{
    const QVector<CameraViewCandidate> candidates{
        {0, QVector3D(0.0f, 0.0f, -1.0f), QVector3D(0.0f, 0.0f, 3.0f), true},
        {1, QVector3D(1.0f, 0.0f, 0.0f), QVector3D(3.0f, 0.0f, 0.0f), true},
        {2, QVector3D(0.0f, 0.0f, -1.0f), QVector3D(0.0f, 0.0f, 2.0f), false},
    };

    EXPECT_EQ(selectCameraForView(candidates,
                                  QVector3D(0.0f, 0.0f, -1.0f),
                                  QVector3D()),
              0);
}

TEST(CameraSceneViewMathTest, UsesStableIndexWhenCameraScoresMatch)
{
    const QVector<CameraViewCandidate> candidates{
        {4, QVector3D(0.0f, 0.0f, -1.0f), QVector3D(0.0f, 0.0f, 3.0f), true},
        {2, QVector3D(0.0f, 0.0f, -1.0f), QVector3D(0.0f, 0.0f, 3.0f), true},
    };

    EXPECT_EQ(selectCameraForView(candidates,
                                  QVector3D(0.0f, 0.0f, -1.0f),
                                  QVector3D()),
              2);
}

TEST(CameraSceneViewMathTest, UsesCameraCenterSideBeforeNoisyOpticalAxis)
{
    const QVector<CameraViewCandidate> candidates{
        // 0 号相机位于当前观察方向一侧，但其光轴被错误翻转。
        {0, QVector3D(0.0f, 0.0f, 1.0f), QVector3D(0.0f, 0.0f, 3.0f), true},
        // 1 号相机在模型背面，光轴恰好会误导旧的纯光轴选择策略。
        {1, QVector3D(0.0f, 0.0f, -1.0f), QVector3D(0.0f, 0.0f, -3.0f), true},
    };

    EXPECT_EQ(selectCameraForView(candidates,
                                  QVector3D(0.0f, 0.0f, -1.0f),
                                  QVector3D()),
              0);
}

TEST(CameraSceneViewMathTest, RejectsCameraWhenCurrentViewHasNoMatchingDirection)
{
    const QVector<CameraViewCandidate> candidates{
        {0, QVector3D(-1.0f, 0.0f, 0.0f), QVector3D(3.0f, 0.0f, 0.0f), true},
        {1, QVector3D(1.0f, 0.0f, 0.0f), QVector3D(-3.0f, 0.0f, 0.0f), true},
    };

    EXPECT_EQ(selectCameraForView(candidates,
                                  QVector3D(0.0f, 0.0f, -1.0f),
                                  QVector3D()),
              -1);
}

TEST(CameraSceneViewMathTest, AcceptsCameraWithinViewMatchingAngle)
{
    constexpr float pi = 3.14159265358979323846f;
    const float angle_radians = 25.0f * pi / 180.0f;
    const QVector3D camera_to_scene(std::sin(angle_radians),
                                    0.0f,
                                    -std::cos(angle_radians));
    const QVector<CameraViewCandidate> candidates{
        {3, camera_to_scene, -camera_to_scene * 4.0f, true},
    };

    EXPECT_EQ(selectCameraForView(candidates,
                                  QVector3D(0.0f, 0.0f, -1.0f),
                                  QVector3D()),
              3);
}

TEST(CameraSceneViewMathTest, CalibratedProjectionMapsOpticalAxisToPrincipalPoint)
{
    const QMatrix4x4 projection = calibratedProjection(
        1000.0f, 1000.0f, 600.0f, 400.0f, 1600, 1000, 0.1f, 100.0f, 1, 1);
    const QVector3D ndc = (projection * QVector4D(0.0f, 0.0f, -2.0f, 1.0f)).toVector3DAffine();

    EXPECT_NEAR((ndc.x() * 0.5f + 0.5f) * 1600.0f, 600.0f, 1e-3f);
    EXPECT_NEAR((1.0f - (ndc.y() * 0.5f + 0.5f)) * 1000.0f, 400.0f, 1e-3f);
}

TEST(CameraSceneViewMathTest, SfMCameraViewAndProjectionRecoverObservedPixels)
{
    QMatrix3x3 camera_to_world;
    camera_to_world.fill(0.0f);
    camera_to_world(0, 0) = 1.0f;
    camera_to_world(1, 1) = 1.0f;
    camera_to_world(2, 2) = 1.0f;
    const QVector3D camera_center(10.0f, 20.0f, 30.0f);
    const QMatrix4x4 view = calibratedCameraView(
        camera_center, camera_to_world, false);
    const QMatrix4x4 projection = calibratedProjection(
        1000.0f, 1000.0f, 600.0f, 400.0f, 1600, 1000, 0.1f, 100.0f, 1, 1);

    const QVector4D camera_point = view * QVector4D(
        camera_center + QVector3D(2.0f, 3.0f, 8.0f), 1.0f);
    EXPECT_NEAR(camera_point.x(), 2.0f, 1.0e-5f);
    EXPECT_NEAR(camera_point.y(), 3.0f, 1.0e-5f);
    EXPECT_NEAR(camera_point.z(), -8.0f, 1.0e-5f);

    const QVector3D ndc = (projection * camera_point).toVector3DAffine();
    EXPECT_NEAR((ndc.x() * 0.5f + 0.5f) * 1600.0f, 850.0f, 1.0e-3f);
    EXPECT_NEAR((1.0f - (ndc.y() * 0.5f + 0.5f)) * 1000.0f, 775.0f, 1.0e-3f);
}

TEST(CameraSceneViewMathTest, FlippedDepthAxisPreservesSignedPinholeRatios)
{
    QMatrix3x3 camera_to_world;
    camera_to_world.fill(0.0f);
    camera_to_world(0, 0) = 1.0f;
    camera_to_world(1, 1) = 1.0f;
    camera_to_world(2, 2) = 1.0f;
    const QMatrix4x4 view = calibratedCameraView(
        QVector3D(), camera_to_world, true);
    const QMatrix4x4 projection = calibratedProjection(
        1000.0f, 1000.0f, 600.0f, 400.0f, 1600, 1000, 0.1f, 100.0f, 1, 1);

    const QVector3D ndc = (projection * view
        * QVector4D(2.0f, 3.0f, -8.0f, 1.0f)).toVector3DAffine();
    EXPECT_NEAR((ndc.x() * 0.5f + 0.5f) * 1600.0f, 350.0f, 1.0e-3f);
    EXPECT_NEAR((1.0f - (ndc.y() * 0.5f + 0.5f)) * 1000.0f, 25.0f, 1.0e-3f);
}

TEST(CameraSceneViewMathTest, FitsImageViewportWithoutChangingItsAspectRatio)
{
    EXPECT_EQ(fittedImageViewport(QSize(1600, 900), QSize(1000, 1000)),
              QRectF(350.0, 0.0, 900.0, 900.0));
    EXPECT_EQ(fittedImageViewport(QSize(900, 1600), QSize(1600, 900)),
              QRectF(0.0, 546.875, 900.0, 506.25));
    EXPECT_TRUE(fittedImageViewport(QSize(), QSize(1600, 900)).isEmpty());
}

TEST(CameraSceneViewMathTest, InvalidIntrinsicsUseFiniteFortyFiveDegreeFallback)
{
    const QMatrix4x4 projection = calibratedProjection(
        0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0.1f, 100.0f, 1, 1);
    const float *values = projection.constData();
    for (int i = 0; i < 16; ++i)
    {
        EXPECT_TRUE(std::isfinite(values[i]));
    }
}

TEST(CameraSceneViewMathTest, BuildsImagePlaneInCameraWorldCoordinates)
{
    const QVector<QVector3D> corners = cameraImagePlaneCorners(
        QVector3D(10.0f, 20.0f, 30.0f),
        QVector3D(1.0f, 0.0f, 0.0f),
        QVector3D(0.0f, 1.0f, 0.0f),
        4.0f,
        2.0f);

    ASSERT_EQ(corners.size(), 4);
    EXPECT_EQ(corners.at(0), QVector3D(14.0f, 22.0f, 30.0f));
    EXPECT_EQ(corners.at(1), QVector3D(6.0f, 22.0f, 30.0f));
    EXPECT_EQ(corners.at(2), QVector3D(6.0f, 18.0f, 30.0f));
    EXPECT_EQ(corners.at(3), QVector3D(14.0f, 18.0f, 30.0f));
}

TEST(CameraSceneViewMathTest, BuildsCalibratedImagePlaneAtTheModelDepth)
{
    const QVector<QVector3D> corners = calibratedImagePlaneCorners(
        QVector3D(0.0f, 0.0f, 10.0f),
        QVector3D(0.0f, 0.0f, -1.0f),
        QVector3D(1.0f, 0.0f, 0.0f),
        QVector3D(0.0f, 1.0f, 0.0f),
        QVector3D(),
        1000.0f,
        1000.0f,
        500.0f,
        250.0f,
        1000,
        500);

    ASSERT_EQ(corners.size(), 4);
    EXPECT_EQ(corners.at(0), QVector3D(5.0f, 2.5f, 0.0f));
    EXPECT_EQ(corners.at(1), QVector3D(-5.0f, 2.5f, 0.0f));
    EXPECT_EQ(corners.at(2), QVector3D(-5.0f, -2.5f, 0.0f));
    EXPECT_EQ(corners.at(3), QVector3D(5.0f, -2.5f, 0.0f));
}

TEST(CameraSceneViewMathTest, CalibratedImagePlaneRejectsModelBehindCamera)
{
    const QVector<QVector3D> corners = calibratedImagePlaneCorners(
        QVector3D(),
        QVector3D(0.0f, 0.0f, -1.0f),
        QVector3D(1.0f, 0.0f, 0.0f),
        QVector3D(0.0f, 1.0f, 0.0f),
        QVector3D(0.0f, 0.0f, 2.0f),
        1000.0f,
        1000.0f,
        500.0f,
        250.0f,
        1000,
        500);

    EXPECT_TRUE(corners.isEmpty());
}

TEST(CameraSceneViewMathTest, ConvertsPixelAxesToVisualImagePlaneAxes)
{
    QMatrix3x3 rotation;
    rotation.fill(0.0f);
    rotation(0, 0) = 1.0f;
    rotation(1, 1) = 1.0f;
    rotation(2, 2) = 1.0f;

    const CameraImagePlaneAxes axes = cameraImagePlaneAxes(rotation, 1, 1);
    EXPECT_EQ(axes.right, QVector3D(1.0f, 0.0f, 0.0f));
    EXPECT_EQ(axes.up, QVector3D(0.0f, -1.0f, 0.0f));

    const CameraImagePlaneAxes reversedAxes = cameraImagePlaneAxes(rotation, -1, -1);
    EXPECT_EQ(reversedAxes.right, QVector3D(-1.0f, 0.0f, 0.0f));
    EXPECT_EQ(reversedAxes.up, QVector3D(0.0f, 1.0f, 0.0f));
}

TEST(CameraSceneViewMathTest, BuildsCameraLocalAxesInWorldCoordinates)
{
    QMatrix3x3 rotation;
    rotation.fill(0.0f);
    rotation(0, 0) = 1.0f;
    rotation(1, 1) = 1.0f;
    rotation(2, 2) = 1.0f;

    const CameraLocalAxes axes = cameraLocalAxes(rotation, false);
    EXPECT_EQ(axes.x, QVector3D(1.0f, 0.0f, 0.0f));
    EXPECT_EQ(axes.y, QVector3D(0.0f, 1.0f, 0.0f));
    EXPECT_EQ(axes.z, QVector3D(0.0f, 0.0f, 1.0f));

    const CameraLocalAxes flippedAxes = cameraLocalAxes(rotation, true);
    EXPECT_EQ(flippedAxes.z, QVector3D(0.0f, 0.0f, -1.0f));
}

TEST(CameraSceneViewMathTest, BuildsTwelveEdgesForPointCloudBoundingBox)
{
    const QVector3D minimum(-1.0f, -2.0f, -3.0f);
    const QVector3D maximum(4.0f, 5.0f, 6.0f);

    const QVector<QVector3D> vertices =
        axisAlignedBoundingBoxLineVertices(minimum, maximum);

    ASSERT_EQ(vertices.size(), 24);
    EXPECT_EQ(vertices.at(0), QVector3D(-1.0f, -2.0f, -3.0f));
    EXPECT_EQ(vertices.at(1), QVector3D(4.0f, -2.0f, -3.0f));
    EXPECT_EQ(vertices.at(22), QVector3D(-1.0f, 5.0f, -3.0f));
    EXPECT_EQ(vertices.at(23), QVector3D(-1.0f, 5.0f, 6.0f));
}

TEST(CameraSceneViewMathTest, RejectsNonFinitePointCloudBoundingBox)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(axisAlignedBoundingBoxLineVertices(
                    QVector3D(nan, 0.0f, 0.0f),
                    QVector3D(1.0f, 1.0f, 1.0f))
                    .isEmpty());
}

TEST(CameraSceneViewMathTest, BuildsOrientedBoxAlongPointCloudPrincipalAxes)
{
    const QVector3D long_axis = QVector3D(1.0f, 1.0f, 0.0f).normalized();
    const QVector3D short_axis = QVector3D(-1.0f, 1.0f, 0.0f).normalized();
    QVector<QVector3D> points;
    for (int along = -4; along <= 4; ++along)
    {
        for (int across = -1; across <= 1; ++across)
        {
            points.push_back(long_axis * static_cast<float>(along)
                             + short_axis * static_cast<float>(across)
                             + QVector3D(0.0f, 0.0f, 0.05f * across));
        }
    }

    const PointCloudPrincipalAxes axes = pointCloudPrincipalAxes(points);
    ASSERT_TRUE(axes.valid);
    EXPECT_GT(std::abs(QVector3D::dotProduct(axes.first, long_axis)), 0.99f);
    EXPECT_GT(std::abs(QVector3D::dotProduct(axes.third, QVector3D(0, 0, 1))), 0.99f);

    const QVector<QVector3D> vertices = orientedBoundingBoxLineVertices(
        axes, QVector3D(-4.0f, -1.0f, -0.1f), QVector3D(4.0f, 1.0f, 0.1f));
    ASSERT_EQ(vertices.size(), 24);
    const QVector3D first_edge = (vertices.at(1) - vertices.at(0)).normalized();
    EXPECT_GT(std::abs(QVector3D::dotProduct(first_edge, long_axis)), 0.99f);
}

TEST(CameraSceneViewMathTest, FitsFootprintEdgeInsteadOfBiasedVarianceDirection)
{
    QVector<QVector3D> points{
        {0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f},
        {8.0f, 4.0f, 0.0f}, {0.0f, 4.0f, 0.0f},
    };
    for (int index = 0; index < 30; ++index)
    {
        points.push_back(QVector3D(0.2f + 0.01f * index,
                                   3.5f + 0.01f * index,
                                   0.0f));
    }

    const PointCloudPrincipalAxes axes = pointCloudPrincipalAxes(points);
    ASSERT_TRUE(axes.valid);
    EXPECT_GT(std::abs(QVector3D::dotProduct(axes.first, QVector3D(1, 0, 0))),
              0.999f);
    EXPECT_GT(std::abs(QVector3D::dotProduct(axes.third, QVector3D(0, 0, 1))),
              0.999f);
}

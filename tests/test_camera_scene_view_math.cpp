#include <cmath>
#include <limits>

#include <gtest/gtest.h>

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

TEST(CameraSceneViewMathTest, UsesLargerWorldPlaneForFartherCamera)
{
    QMatrix4x4 world_to_view;
    world_to_view.setToIdentity();

    const float near_half_extent = cameraPlaneHalfExtentForViewDepth(
        QVector3D(0.0f, 0.0f, -4.0f), world_to_view, 1000, 45.0f, 24.0f);
    const float far_half_extent = cameraPlaneHalfExtentForViewDepth(
        QVector3D(0.0f, 0.0f, -12.0f), world_to_view, 1000, 45.0f, 24.0f);

    EXPECT_GT(far_half_extent, near_half_extent);
    EXPECT_NEAR(far_half_extent, near_half_extent * 3.0f, 1e-6f);
}

TEST(CameraSceneViewMathTest, RejectsInvalidCameraPlaneDepthInputs)
{
    QMatrix4x4 world_to_view;
    world_to_view.setToIdentity();
    EXPECT_EQ(cameraPlaneHalfExtentForViewDepth(
                  QVector3D(0.0f, 0.0f, 1.0f), world_to_view, 1000),
              0.0f);
    EXPECT_EQ(cameraPlaneHalfExtentForViewDepth(
                  QVector3D(0.0f, 0.0f, -4.0f), world_to_view, 0),
              0.0f);
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

TEST(CameraSceneViewMathTest, LockKeepsActiveCameraUntilUnlocked)
{
    CameraImageSelectionState state;
    state.setActiveIndex(3);
    state.setLocked(true);
    EXPECT_EQ(state.resolveAutomaticIndex(7), 3);

    state.setLocked(false);
    EXPECT_EQ(state.resolveAutomaticIndex(7), 7);
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

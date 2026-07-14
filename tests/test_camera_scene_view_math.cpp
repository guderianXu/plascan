#include <cmath>

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

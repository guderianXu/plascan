#include "BundleAdjustNativeCudaMath.h"
#include "BundleAdjustNativeCudaTypes.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace nc = xjw::detail::native_cuda;

namespace
{

nc::HostCamera makeCamera()
{
    nc::HostCamera camera;
    camera.cameraToWorldRotation = {{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0}};
    camera.cameraCenter = {{0.0, 0.0, 0.0}};
    camera.focalX = 1000.0;
    camera.focalY = 1000.0;
    camera.principalX = 320.0;
    camera.principalY = 240.0;
    return camera;
}

double numericPointDerivative(const nc::HostCamera &camera,
                              std::array<double, 3> point,
                              int axis,
                              int pixelAxis)
{
    const double eps = 1e-6;
    point[axis] += eps;
    const auto plus = nc::projectHost(camera, point);
    point[axis] -= 2.0 * eps;
    const auto minus = nc::projectHost(camera, point);
    return (plus.pixel[pixelAxis] - minus.pixel[pixelAxis]) / (2.0 * eps);
}

} // namespace

TEST(NativeCudaMathTest, ProjectionMatchesSimplePinhole)
{
    const nc::HostCamera camera = makeCamera();
    const std::array<double, 3> point{{1.0, 2.0, 10.0}};
    const auto projected = nc::projectHost(camera, point);
    ASSERT_TRUE(projected.ok);
    EXPECT_NEAR(projected.pixel[0], 420.0, 1e-9);
    EXPECT_NEAR(projected.pixel[1], 440.0, 1e-9);
}

TEST(NativeCudaMathTest, PointJacobianMatchesFiniteDifference)
{
    const nc::HostCamera camera = makeCamera();
    const std::array<double, 3> point{{1.0, -0.5, 8.0}};
    const auto projected = nc::projectHost(camera, point);
    ASSERT_TRUE(projected.ok);

    nc::ObservationLinearization lin;
    ASSERT_TRUE(nc::linearizeObservationHost(camera,
                                             point,
                                             projected.pixel[0],
                                             projected.pixel[1],
                                             1.0,
                                             3.0,
                                             &lin));

    for (int pixelAxis = 0; pixelAxis < 2; ++pixelAxis)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const double numeric = numericPointDerivative(camera, point, axis, pixelAxis);
            EXPECT_NEAR(lin.jp[pixelAxis * 3 + axis], numeric, 1e-3);
        }
    }
}

#include <gtest/gtest.h>

#include "filtering/SparsePointCloudWorkspace.h"
#include "reconstruction/SfmReconstruction.h"

#include <cstdint>
#include <vector>

using namespace xjw;

namespace
{

SparsePointCloudPoint makeWorkspacePoint(double x,
                                         double y,
                                         double z,
                                         int track_len,
                                         std::uint8_t red,
                                         std::uint8_t green,
                                         std::uint8_t blue)
{
    SparsePointCloudPoint point;
    point.x = x;
    point.y = y;
    point.z = z;
    point.rmsReprojPx = 0.25 + x;
    point.minTriAngleDeg = 3.0 + y;
    point.trackLen = track_len;
    point.hasColor = true;
    point.red = red;
    point.green = green;
    point.blue = blue;
    return point;
}

ScenePoint3D makeScenePoint(Point3DId id,
                            double x,
                            double y,
                            double z,
                            std::uint8_t red,
                            std::uint8_t green,
                            std::uint8_t blue)
{
    ScenePoint3D point;
    point.id = id;
    point.xyz = {x, y, z};
    point.error = 0.5;
    point.color = {red, green, blue};
    point.track.elements.push_back({0, 0});
    point.track.elements.push_back({1, 1});
    return point;
}

} // namespace

TEST(SparsePointCloudWorkspaceTest, PreservesAttributesWhenFilteringPointVector)
{
    const std::vector<SparsePointCloudPoint> points = {
        makeWorkspacePoint(1.0, 2.0, 3.0, 4, 10, 20, 30),
        makeWorkspacePoint(4.0, 5.0, 6.0, 5, 40, 50, 60),
        makeWorkspacePoint(7.0, 8.0, 9.0, 6, 70, 80, 90)
    };

    const SparsePointCloudWorkspace workspace = SparsePointCloudWorkspace::fromPoints(points);

    ASSERT_EQ(workspace.size(), 3u);
    EXPECT_DOUBLE_EQ(workspace.cloud().points()(1, 2), 6.0);

    const std::vector<SparsePointCloudPoint> filtered =
        workspace.filteredPoints({true, false, true});

    ASSERT_EQ(filtered.size(), 2u);
    EXPECT_DOUBLE_EQ(filtered[0].x, 1.0);
    EXPECT_EQ(filtered[0].trackLen, 4);
    EXPECT_EQ(filtered[0].red, 10);
    EXPECT_EQ(filtered[1].trackLen, 6);
    EXPECT_EQ(filtered[1].green, 80);
    EXPECT_EQ(filtered[1].blue, 90);
}

TEST(SparsePointCloudWorkspaceTest, ReconstructionPointIdsAreDeterministicAndFilterable)
{
    SfmReconstruction reconstruction;
    reconstruction.addPoint3D(makeScenePoint(30, 3.0, 0.0, 0.0, 30, 31, 32));
    reconstruction.addPoint3D(makeScenePoint(10, 1.0, 0.0, 0.0, 10, 11, 12));
    reconstruction.addPoint3D(makeScenePoint(20, 2.0, 0.0, 0.0, 20, 21, 22));

    const SparsePointCloudWorkspace workspace =
        SparsePointCloudWorkspace::fromReconstruction(reconstruction);

    const std::vector<Point3DId> expected_ids = {10, 20, 30};
    EXPECT_EQ(workspace.pointIds(), expected_ids);

    const std::vector<Point3DId> removed = workspace.removedPointIds({true, false, true});

    ASSERT_EQ(removed.size(), 1u);
    EXPECT_EQ(removed.front(), 20u);
}

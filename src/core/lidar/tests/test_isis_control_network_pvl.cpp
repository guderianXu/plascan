#include "IsisControlNetworkPvl.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{

const char *kTwoPointNetwork = R"pvl(
Object = ControlNetwork
  NetworkId = lro_lidar
  TargetName = MOON

  Object = ControlPoint
    PointType = Free
    PointId = point_1
    Group = ControlMeasure
      SerialNumber = LRO/IMAGE_1
      Sample = 1342.25
      Line = 74.75
      Reference = True
    End_Group
    Group = ControlMeasure
      SerialNumber = LRO/IMAGE_2
      Sample = 1187.5
      Line = 17.5
    End_Group
  End_Object

  Object = ControlPoint
    PointType = Free
    PointId = point_2
    Group = ControlMeasure
      SerialNumber = LRO/IMAGE_1
      Sample = 200.5
      Line = 300.5
      Ignore = True
    End_Group
    Group = ControlMeasure
      SerialNumber = LRO/IMAGE_2
      Sample = 210.5
      Line = 301.5
    End_Group
  End_Object
End_Object
End
)pvl";

struct ScopedTestDirectory
{
    std::filesystem::path path;

    ~ScopedTestDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

} // namespace

TEST(IsisControlNetworkPvlTest, ParsesJigsawControlPointSubset)
{
    xjw::lidar::IsisControlNetwork network;
    std::string error;
    ASSERT_TRUE(xjw::lidar::parseIsisControlNetworkPvl(
        kTwoPointNetwork, &network, &error)) << error;

    EXPECT_EQ(network.networkId, "lro_lidar");
    EXPECT_EQ(network.targetName, "MOON");
    ASSERT_EQ(network.points.size(), 2u);
    EXPECT_EQ(network.usableMeasureCount(), 3);

    const auto &point = network.points.front();
    EXPECT_EQ(point.id, "point_1");
    EXPECT_EQ(point.type, xjw::lidar::IsisControlPointType::Free);
    ASSERT_EQ(point.measures.size(), 2u);
    EXPECT_EQ(point.measures[0].serialNumber, "LRO/IMAGE_1");
    EXPECT_DOUBLE_EQ(point.measures[0].samplePixels, 1342.25);
    EXPECT_DOUBLE_EQ(point.measures[0].linePixels, 74.75);
    EXPECT_FALSE(point.measures[0].ignored);
}

TEST(IsisControlNetworkPvlTest, RejectsIncompleteMeasure)
{
    const std::string pvl = R"pvl(
Object = ControlNetwork
  Object = ControlPoint
    PointType = Free
    PointId = broken
    Group = ControlMeasure
      SerialNumber = LRO/IMAGE_1
      Sample = 1.5
    End_Group
  End_Object
End_Object
)pvl";

    xjw::lidar::IsisControlNetwork network;
    std::string error;
    EXPECT_FALSE(xjw::lidar::parseIsisControlNetworkPvl(pvl, &network, &error));
    EXPECT_NE(error.find("incomplete ControlMeasure"), std::string::npos);
}

TEST(IsisControlNetworkPvlTest, RejectsDuplicateCameraInPoint)
{
    const std::string pvl = R"pvl(
Object = ControlNetwork
  Object = ControlPoint
    PointType = Free
    PointId = duplicate
    Group = ControlMeasure
      SerialNumber = SAME
      Sample = 1.5
      Line = 2.5
    End_Group
    Group = ControlMeasure
      SerialNumber = SAME
      Sample = 3.5
      Line = 4.5
    End_Group
  End_Object
End_Object
)pvl";

    xjw::lidar::IsisControlNetwork network;
    std::string error;
    EXPECT_FALSE(xjw::lidar::parseIsisControlNetworkPvl(pvl, &network, &error));
    EXPECT_NE(error.find("duplicate camera serial"), std::string::npos);
}

TEST(IsisControlNetworkPvlTest, LoadsUtf8FilePath)
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    ScopedTestDirectory directory{
        std::filesystem::temp_directory_path() /
        ("plascan_isis_control_network_" + std::to_string(nonce))};
    ASSERT_TRUE(std::filesystem::create_directories(directory.path));

    const std::filesystem::path path = directory.path /
        std::filesystem::path(u8"月球控制网.pvl");
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open());
    stream << kTwoPointNetwork;
    stream.close();

    const std::u8string utf8Path = path.generic_u8string();
    const std::string pathString(utf8Path.begin(), utf8Path.end());
    xjw::lidar::IsisControlNetwork network;
    std::string error;
    ASSERT_TRUE(xjw::lidar::loadIsisControlNetworkPvlFile(
        pathString, &network, &error)) << error;
    EXPECT_EQ(network.points.size(), 2u);
}

#include "IsisControlNetworkPvl.h"

#include <gtest/gtest.h>

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

#include <placamera/isd.h>

#include <gtest/gtest.h>

TEST(PlaCameraIsd, RejectsNonObjectDocument)
{
    const auto imported = placamera::parsePlanetaryLineScanIsd("[]",
                                                               placamera::CameraDefinitionId("definition"),
                                                               placamera::CameraInstanceId("instance"),
                                                               placamera::ImageId("image"));
    EXPECT_FALSE(imported);
    EXPECT_EQ(imported.errorCode(), placamera::CameraErrorCode::ParseFailure);
    EXPECT_NE(imported.message().find("JSON object"), std::string::npos);
}

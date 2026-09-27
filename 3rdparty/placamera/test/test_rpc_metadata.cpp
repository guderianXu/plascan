#include <placamera/rpc_metadata.h>

#include <gtest/gtest.h>

TEST(PlaCameraRpcMetadata, RejectsIncompleteDomainWithKeyName)
{
    const auto parameters = placamera::rpcParametersFromMetadata({{"line_off", "1"}});
    EXPECT_FALSE(parameters);
    EXPECT_EQ(parameters.errorCode(), placamera::CameraErrorCode::ParseFailure);
    EXPECT_NE(parameters.message().find("SAMP_OFF"), std::string::npos);
}

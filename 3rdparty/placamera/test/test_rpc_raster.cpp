#include <sstream>
#include <string>

#include <cpl_vsi.h>
#include <gdal_priv.h>
#include <gtest/gtest.h>

#include <placamera/rpc_raster.h>

namespace
{

    using namespace placamera;

    placamera::RpcParameters linearRpcParameters(double heightCoefficient)
    {
        placamera::RpcParameters parameters;
        parameters.lineOffset = 500.0;
        parameters.sampleOffset = 500.0;
        parameters.latitudeOffset = 20.0;
        parameters.longitudeOffset = 110.0;
        parameters.heightOffset = 1000.0;
        parameters.lineScale = 1000.0;
        parameters.sampleScale = 1000.0;
        parameters.latitudeScale = 0.1;
        parameters.longitudeScale = 0.1;
        parameters.heightScale = 1000.0;
        parameters.lineNumerator[2] = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleNumerator[1] = 1.0;
        parameters.sampleNumerator[3] = heightCoefficient;
        parameters.sampleDenominator[0] = 1.0;
        return parameters;
    }

    std::string coefficients(const placamera::RpcCoefficients& values)
    {
        std::ostringstream stream;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index > 0)
            {
                stream << ' ';
            }
            stream << values[index];
        }
        return stream.str();
    }

    RpcMetadata metadataFor(const placamera::RpcParameters& parameters)
    {
        return {{"LINE_OFF", std::to_string(parameters.lineOffset)},
                {"SAMP_OFF", std::to_string(parameters.sampleOffset)},
                {"LAT_OFF", std::to_string(parameters.latitudeOffset)},
                {"LONG_OFF", std::to_string(parameters.longitudeOffset)},
                {"HEIGHT_OFF", std::to_string(parameters.heightOffset)},
                {"LINE_SCALE", std::to_string(parameters.lineScale)},
                {"SAMP_SCALE", std::to_string(parameters.sampleScale)},
                {"LAT_SCALE", std::to_string(parameters.latitudeScale)},
                {"LONG_SCALE", std::to_string(parameters.longitudeScale)},
                {"HEIGHT_SCALE", std::to_string(parameters.heightScale)},
                {"LINE_NUM_COEFF", coefficients(parameters.lineNumerator)},
                {"LINE_DEN_COEFF", coefficients(parameters.lineDenominator)},
                {"SAMP_NUM_COEFF", coefficients(parameters.sampleNumerator)},
                {"SAMP_DEN_COEFF", coefficients(parameters.sampleDenominator)}};
    }

} // namespace

TEST(RpcRasterIO, ParsesStandardGdalMetadata)
{
    const placamera::RpcParameters source = linearRpcParameters(-0.25);
    RpcMetadata metadata = metadataFor(source);
    metadata["ERR_BIAS"] = "2.5";
    metadata["err_rand"] = "1.25";

    const auto restored_result = rpcParametersFromMetadata(metadata);
    ASSERT_TRUE(restored_result) << restored_result.message();
    const auto& restored = restored_result.value();
    ASSERT_TRUE(restored.errorBiasMeters.has_value());
    ASSERT_TRUE(restored.errorRandomMeters.has_value());
    EXPECT_DOUBLE_EQ(*restored.errorBiasMeters, 2.5);
    EXPECT_DOUBLE_EQ(*restored.errorRandomMeters, 1.25);
    EXPECT_EQ(restored.sampleNumerator, source.sampleNumerator);
}

TEST(RpcRasterIO, RejectsIncompleteMetadata)
{
    RpcMetadata metadata = metadataFor(linearRpcParameters(0.25));
    metadata.erase("LINE_NUM_COEFF");
    const auto parameters = rpcParametersFromMetadata(metadata);
    EXPECT_FALSE(parameters);
    EXPECT_NE(parameters.message().find("LINE_NUM_COEFF"), std::string::npos);
}

TEST(RpcRasterIO, ImportsIdentifiedTypedInstanceFromGdalRaster)
{
    GDALAllRegister();
    constexpr const char* rasterPath = "/vsimem/plascan_rpc_camera_test.tif";
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    ASSERT_NE(driver, nullptr);
    GDALDataset* dataset = driver->Create(rasterPath, 16, 12, 1, GDT_Byte, nullptr);
    ASSERT_NE(dataset, nullptr);
    const RpcMetadata metadata = metadataFor(linearRpcParameters(0.25));
    for (const auto& [key, value] : metadata)
    {
        ASSERT_EQ(dataset->SetMetadataItem(key.c_str(), value.c_str(), "RPC"), CE_None);
    }
    GDALClose(dataset);

    const auto camera = importRpcRasterModel(rasterPath,
                                             placamera::CameraDefinitionId("rpc-definition"),
                                             placamera::CameraInstanceId("rpc-instance"),
                                             placamera::ImageId("rpc-image"),
                                             placamera::FrameId("EPSG:4978"));
    ASSERT_TRUE(camera) << camera.message();
    EXPECT_EQ(camera.value()->imageSize().samples, 16);
    EXPECT_EQ(camera.value()->imageSize().lines, 12);
    EXPECT_DOUBLE_EQ(camera.value()->rpcDefinition().parameters().longitudeOffset, 110.0);
    EXPECT_EQ(camera.value()->instanceId().value(), "rpc-instance");
    EXPECT_EQ(VSIUnlink(rasterPath), 0);
}

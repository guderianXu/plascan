#include "RpcRasterIO.h"

#include <cpl_vsi.h>
#include <gdal_priv.h>
#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace
{

    using namespace xjw::camera_models::rpc;

    RpcDefinition::Parameters linearRpcParameters(double heightCoefficient)
    {
        RpcDefinition::Parameters parameters;
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

    std::string coefficients(const RpcDefinition::Coefficients& values)
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

    RpcMetadata metadataFor(const RpcDefinition::Parameters& parameters)
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
    const RpcDefinition::Parameters source = linearRpcParameters(-0.25);
    RpcMetadata metadata = metadataFor(source);
    metadata["ERR_BIAS"] = "2.5";
    metadata["err_rand"] = "1.25";

    RpcDefinition::Parameters restored;
    std::string error;
    ASSERT_TRUE(rpcParametersFromMetadata(metadata, &restored, &error)) << error;
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
    RpcDefinition::Parameters parameters;
    std::string error;
    EXPECT_FALSE(rpcParametersFromMetadata(metadata, &parameters, &error));
    EXPECT_NE(error.find("LINE_NUM_COEFF"), std::string::npos);
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

    std::string error;
    const auto camera = importRpcRasterInstance(rasterPath,
                                                xjw::camera_core::CameraDefinitionId("rpc-definition"),
                                                xjw::camera_core::CameraInstanceId("rpc-instance"),
                                                xjw::camera_core::ImageId("rpc-image"),
                                                xjw::coordinate_system::CoordinateFrameId("EPSG:4978"),
                                                &error);
    ASSERT_NE(camera, nullptr) << error;
    EXPECT_EQ(camera->imageSize().samples, 16);
    EXPECT_EQ(camera->imageSize().lines, 12);
    EXPECT_DOUBLE_EQ(camera->rpcDefinition().parameters().longitudeOffset, 110.0);
    EXPECT_EQ(camera->instanceId().value(), "rpc-instance");
    EXPECT_EQ(VSIUnlink(rasterPath), 0);
}

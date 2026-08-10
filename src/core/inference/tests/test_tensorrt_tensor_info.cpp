#include "inference/tensorrt/TensorRtTensorInfo.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace
{

    TEST(TensorRtTensorInfoTest, ComputesFixedTensorElementAndByteCounts)
    {
        xjw::inference::TensorRtTensorInfo tensor;
        tensor.name = QStringLiteral("input");
        tensor.mode = xjw::inference::TensorRtTensorMode::Input;
        tensor.dataType = xjw::inference::TensorRtTensorDataType::Float32;
        tensor.dimensions = {1, 3, 320, 320};

        EXPECT_TRUE(tensor.hasFixedShape());
        EXPECT_EQ(tensor.elementCount(), 307200U);
        EXPECT_EQ(tensor.byteCount(), 1228800U);
    }

    TEST(TensorRtTensorInfoTest, RejectsUnresolvedDynamicTensorShape)
    {
        xjw::inference::TensorRtTensorInfo tensor;
        tensor.dataType = xjw::inference::TensorRtTensorDataType::Float32;
        tensor.dimensions = {1, -1, 256};

        EXPECT_FALSE(tensor.hasFixedShape());
        EXPECT_EQ(tensor.elementCount(), 0U);
        EXPECT_EQ(tensor.byteCount(), 0U);
    }

    TEST(TensorRtTensorInfoTest, TreatsRankZeroTensorAsScalar)
    {
        xjw::inference::TensorRtTensorInfo tensor;
        tensor.dataType = xjw::inference::TensorRtTensorDataType::Int64;

        EXPECT_TRUE(tensor.hasFixedShape());
        EXPECT_EQ(tensor.elementCount(), 1U);
        EXPECT_EQ(tensor.byteCount(), sizeof(std::int64_t));
    }

    TEST(TensorRtTensorInfoTest, ConvertsStableMetadataTokens)
    {
        using xjw::inference::TensorRtTensorDataType;
        using xjw::inference::TensorRtTensorMode;

        EXPECT_EQ(xjw::inference::tensorRtTensorModeFromName(QStringLiteral("OUTPUT")), TensorRtTensorMode::Output);
        EXPECT_EQ(xjw::inference::tensorRtTensorDataTypeFromName(QStringLiteral("float16")),
                  TensorRtTensorDataType::Float16);
        EXPECT_EQ(xjw::inference::tensorRtTensorDataTypeName(TensorRtTensorDataType::Bool), QStringLiteral("bool"));
    }

} // namespace

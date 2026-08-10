#include "inference/tensorrt/TensorRtEngineCache.h"
#include "inference/tensorrt/TensorRtTensorInfo.h"

#include <QFile>
#include <QTemporaryDir>

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

    TEST(TensorRtEngineCacheTest, RejectsSameSizeEngineCorruption)
    {
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        const QString engine_path = temporary_directory.filePath(QStringLiteral("model.engine"));
        const QString metadata_path = engine_path + QStringLiteral(".json");

        QFile engine_file(engine_path);
        ASSERT_TRUE(engine_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(engine_file.write("engine-v1"), 9);
        engine_file.close();

        xjw::inference::TensorRtEngineBuildResult stored_result;
        stored_result.enginePath = engine_path;
        stored_result.cacheFingerprint = QStringLiteral("test-fingerprint");
        stored_result.environmentSummary = QStringLiteral("test environment");
        xjw::inference::TensorRtTensorInfo tensor;
        tensor.name = QStringLiteral("output");
        tensor.mode = xjw::inference::TensorRtTensorMode::Output;
        tensor.dataType = xjw::inference::TensorRtTensorDataType::Float32;
        tensor.dimensions = {1};
        stored_result.ioTensors.push_back(tensor);

        QString error_message;
        ASSERT_TRUE(xjw::inference::detail::saveEngineMetadata(metadata_path,
                                                               QJsonObject{{QStringLiteral("schema"), 2}},
                                                               stored_result,
                                                               QStringLiteral("model.onnx"),
                                                               QStringLiteral("test gpu"),
                                                               1.0,
                                                               9,
                                                               &error_message))
            << error_message.toStdString();

        xjw::inference::TensorRtEngineBuildResult loaded_result;
        EXPECT_TRUE(xjw::inference::detail::loadMatchingEngineMetadata(metadata_path,
                                                                       engine_path,
                                                                       stored_result.cacheFingerprint,
                                                                       &loaded_result));

        ASSERT_TRUE(engine_file.open(QIODevice::WriteOnly | QIODevice::Truncate));
        ASSERT_EQ(engine_file.write("engine-v2"), 9);
        engine_file.close();

        EXPECT_FALSE(xjw::inference::detail::loadMatchingEngineMetadata(metadata_path,
                                                                        engine_path,
                                                                        stored_result.cacheFingerprint,
                                                                        &loaded_result));
    }

} // namespace

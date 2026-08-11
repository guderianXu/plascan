#include "inference/tensorrt/TensorRtEngineCache.h"
#include "inference/tensorrt/TensorRtTensorInfo.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
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

    TEST(TensorRtEngineCacheTest, ExplainsLegacyMetadataRebuild)
    {
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        const QString legacy_directory =
            temporary_directory.filePath(QStringLiteral("legacy-fingerprint"));
        ASSERT_TRUE(QDir().mkpath(legacy_directory));
        const QString engine_name = QStringLiteral("feature.engine");
        const QString legacy_engine = QDir(legacy_directory).filePath(engine_name);
        QFile engine_file(legacy_engine);
        ASSERT_TRUE(engine_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(engine_file.write("legacy-engine"), 13);
        engine_file.close();

        const QJsonObject legacy_identity = {
            {QStringLiteral("schema"), 1},
            {QStringLiteral("onnx_sha256"), QStringLiteral("same-model")},
            {QStringLiteral("tensorrt"), QStringLiteral("10.16.0.72")}};
        QFile metadata_file(legacy_engine + QStringLiteral(".json"));
        ASSERT_TRUE(metadata_file.open(QIODevice::WriteOnly));
        ASSERT_GT(metadata_file.write(QJsonDocument(legacy_identity).toJson()), 0);
        metadata_file.close();

        const QJsonObject current_identity = {
            {QStringLiteral("schema"), 2},
            {QStringLiteral("onnx_sha256"), QStringLiteral("same-model")},
            {QStringLiteral("tensorrt"), QStringLiteral("10.16.0.72")}};
        const QString current_directory =
            temporary_directory.filePath(QStringLiteral("current-fingerprint"));
        const QString current_engine = QDir(current_directory).filePath(engine_name);
        const QString reason = xjw::inference::detail::describeEngineCacheMiss(
            temporary_directory.path(),
            engine_name,
            current_engine + QStringLiteral(".json"),
            current_engine,
            current_identity);

        EXPECT_TRUE(reason.contains(QStringLiteral("旧缓存格式 v1")));
        EXPECT_TRUE(reason.contains(QStringLiteral("当前要求 v2")));
        EXPECT_TRUE(reason.contains(QStringLiteral("完整性和 I/O 元数据")));
    }

    TEST(TensorRtEngineCacheTest, NamesChangedCacheIdentityFields)
    {
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        const QString previous_directory =
            temporary_directory.filePath(QStringLiteral("previous-fingerprint"));
        ASSERT_TRUE(QDir().mkpath(previous_directory));
        const QString engine_name = QStringLiteral("matcher.engine");
        const QString previous_engine = QDir(previous_directory).filePath(engine_name);
        QFile engine_file(previous_engine);
        ASSERT_TRUE(engine_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(engine_file.write("engine"), 6);
        engine_file.close();

        QJsonObject previous_identity = {
            {QStringLiteral("schema"), 2},
            {QStringLiteral("onnx_sha256"), QStringLiteral("same-model")},
            {QStringLiteral("cuda_driver"), 13000},
            {QStringLiteral("input_shapes"), QJsonArray{1, 1024, 2}}};
        QFile metadata_file(previous_engine + QStringLiteral(".json"));
        ASSERT_TRUE(metadata_file.open(QIODevice::WriteOnly));
        ASSERT_GT(metadata_file.write(QJsonDocument(previous_identity).toJson()), 0);
        metadata_file.close();

        QJsonObject current_identity = previous_identity;
        current_identity.insert(QStringLiteral("cuda_driver"), 13010);
        current_identity.insert(QStringLiteral("input_shapes"), QJsonArray{1, 2048, 2});
        const QString current_engine = QDir(
            temporary_directory.filePath(QStringLiteral("current-fingerprint")))
                                           .filePath(engine_name);
        const QString reason = xjw::inference::detail::describeEngineCacheMiss(
            temporary_directory.path(),
            engine_name,
            current_engine + QStringLiteral(".json"),
            current_engine,
            current_identity);

        EXPECT_TRUE(reason.contains(QStringLiteral("CUDA 驱动版本")));
        EXPECT_TRUE(reason.contains(QStringLiteral("输入形状")));
    }

} // namespace

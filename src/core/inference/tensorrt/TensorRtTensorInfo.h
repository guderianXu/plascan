#pragma once

#include <QString>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xjw::inference
{

    enum class TensorRtTensorMode
    {
        Input,
        Output
    };

    enum class TensorRtTensorDataType
    {
        Unknown,
        Float32,
        Float16,
        Int8,
        Int32,
        Int64,
        Bool,
        UInt8,
        BFloat16,
        Float8,
        Int4,
        Float4E2M1
    };

    struct TensorRtTensorInfo
    {
        QString name;
        TensorRtTensorMode mode = TensorRtTensorMode::Input;
        TensorRtTensorDataType dataType = TensorRtTensorDataType::Unknown;
        std::vector<std::int64_t> dimensions;

        bool hasFixedShape() const;
        std::uint64_t elementCount() const;
        std::uint64_t byteCount() const;
    };

    QString tensorRtTensorModeName(TensorRtTensorMode mode);
    QString tensorRtTensorDataTypeName(TensorRtTensorDataType type);
    TensorRtTensorMode tensorRtTensorModeFromName(const QString& name);
    TensorRtTensorDataType tensorRtTensorDataTypeFromName(const QString& name);
    std::size_t tensorRtTensorDataTypeSize(TensorRtTensorDataType type);

} // namespace xjw::inference

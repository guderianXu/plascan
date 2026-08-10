#include "TensorRtTensorInfo.h"

#include <limits>

namespace xjw::inference
{

    bool TensorRtTensorInfo::hasFixedShape() const
    {
        for (const std::int64_t dimension : dimensions)
        {
            if (dimension <= 0)
            {
                return false;
            }
        }
        return true;
    }

    std::uint64_t TensorRtTensorInfo::elementCount() const
    {
        if (!hasFixedShape())
        {
            return 0;
        }

        // TensorRT 使用 rank-0 Dims 表示标量，因此空 dimensions 的元素数是 1。
        std::uint64_t count = 1;
        for (const std::int64_t dimension : dimensions)
        {
            const auto value = static_cast<std::uint64_t>(dimension);
            if (count > std::numeric_limits<std::uint64_t>::max() / value)
            {
                return 0;
            }
            count *= value;
        }
        return count;
    }

    std::uint64_t TensorRtTensorInfo::byteCount() const
    {
        const std::uint64_t count = elementCount();
        const std::size_t element_size = tensorRtTensorDataTypeSize(dataType);
        if (count == 0 || element_size == 0 || count > std::numeric_limits<std::uint64_t>::max() / element_size)
        {
            return 0;
        }
        return count * static_cast<std::uint64_t>(element_size);
    }

    QString tensorRtTensorModeName(TensorRtTensorMode mode)
    {
        return mode == TensorRtTensorMode::Input ? QStringLiteral("input") : QStringLiteral("output");
    }

    QString tensorRtTensorDataTypeName(TensorRtTensorDataType type)
    {
        switch (type)
        {
        case TensorRtTensorDataType::Float32:
            return QStringLiteral("float32");
        case TensorRtTensorDataType::Float16:
            return QStringLiteral("float16");
        case TensorRtTensorDataType::Int8:
            return QStringLiteral("int8");
        case TensorRtTensorDataType::Int32:
            return QStringLiteral("int32");
        case TensorRtTensorDataType::Int64:
            return QStringLiteral("int64");
        case TensorRtTensorDataType::Bool:
            return QStringLiteral("bool");
        case TensorRtTensorDataType::UInt8:
            return QStringLiteral("uint8");
        case TensorRtTensorDataType::BFloat16:
            return QStringLiteral("bfloat16");
        case TensorRtTensorDataType::Float8:
            return QStringLiteral("float8");
        case TensorRtTensorDataType::Int4:
            return QStringLiteral("int4");
        case TensorRtTensorDataType::Float4E2M1:
            return QStringLiteral("float4e2m1");
        case TensorRtTensorDataType::Unknown:
            break;
        }
        return QStringLiteral("unknown");
    }

    TensorRtTensorMode tensorRtTensorModeFromName(const QString& name)
    {
        return name.compare(QStringLiteral("output"), Qt::CaseInsensitive) == 0 ? TensorRtTensorMode::Output
                                                                                : TensorRtTensorMode::Input;
    }

    TensorRtTensorDataType tensorRtTensorDataTypeFromName(const QString& name)
    {
        const QString normalized = name.trimmed().toLower();
        if (normalized == QStringLiteral("float32"))
        {
            return TensorRtTensorDataType::Float32;
        }
        if (normalized == QStringLiteral("float16"))
        {
            return TensorRtTensorDataType::Float16;
        }
        if (normalized == QStringLiteral("int8"))
        {
            return TensorRtTensorDataType::Int8;
        }
        if (normalized == QStringLiteral("int32"))
        {
            return TensorRtTensorDataType::Int32;
        }
        if (normalized == QStringLiteral("int64"))
        {
            return TensorRtTensorDataType::Int64;
        }
        if (normalized == QStringLiteral("bool"))
        {
            return TensorRtTensorDataType::Bool;
        }
        if (normalized == QStringLiteral("uint8"))
        {
            return TensorRtTensorDataType::UInt8;
        }
        if (normalized == QStringLiteral("bfloat16"))
        {
            return TensorRtTensorDataType::BFloat16;
        }
        if (normalized == QStringLiteral("float8"))
        {
            return TensorRtTensorDataType::Float8;
        }
        if (normalized == QStringLiteral("int4"))
        {
            return TensorRtTensorDataType::Int4;
        }
        if (normalized == QStringLiteral("float4e2m1"))
        {
            return TensorRtTensorDataType::Float4E2M1;
        }
        return TensorRtTensorDataType::Unknown;
    }

    std::size_t tensorRtTensorDataTypeSize(TensorRtTensorDataType type)
    {
        switch (type)
        {
        case TensorRtTensorDataType::Float32:
        case TensorRtTensorDataType::Int32:
            return 4;
        case TensorRtTensorDataType::Float16:
        case TensorRtTensorDataType::BFloat16:
            return 2;
        case TensorRtTensorDataType::Int8:
        case TensorRtTensorDataType::Bool:
        case TensorRtTensorDataType::UInt8:
        case TensorRtTensorDataType::Float8:
            return 1;
        case TensorRtTensorDataType::Int64:
            return 8;
        case TensorRtTensorDataType::Int4:
        case TensorRtTensorDataType::Float4E2M1:
        case TensorRtTensorDataType::Unknown:
            return 0;
        }
        return 0;
    }

} // namespace xjw::inference

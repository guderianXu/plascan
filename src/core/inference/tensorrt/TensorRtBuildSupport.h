#pragma once

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <QString>
#include <QStringList>

#include <memory>

namespace xjw::inference::detail
{

    class TensorRtBuildLogger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* message) noexcept override
        {
            if (severity <= Severity::kERROR && message)
            {
                try
                {
                    if (!_errors.isEmpty())
                    {
                        _errors += QLatin1Char('\n');
                    }
                    _errors += QString::fromUtf8(message);
                }
                catch (...)
                {
                }
            }
        }

        QString errors() const
        {
            return _errors;
        }

    private:
        QString _errors;
    };

    template <typename T> struct TensorRtDeleter
    {
        void operator()(T* value) const noexcept
        {
            delete value;
        }
    };

    template <typename T> using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

    inline QString compiledTensorRtVersion()
    {
        return QStringLiteral("%1.%2.%3.%4")
            .arg(NV_TENSORRT_MAJOR)
            .arg(NV_TENSORRT_MINOR)
            .arg(NV_TENSORRT_PATCH)
            .arg(NV_TENSORRT_BUILD);
    }

    inline QString parserErrors(nvonnxparser::IParser& parser)
    {
        QStringList errors;
        for (int index = 0; index < parser.getNbErrors(); ++index)
        {
            const nvonnxparser::IParserError* error = parser.getError(index);
            if (error)
            {
                errors.append(QString::fromUtf8(error->desc()));
            }
        }
        return errors.join(QLatin1Char('\n'));
    }

} // namespace xjw::inference::detail

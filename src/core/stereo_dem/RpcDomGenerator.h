#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <functional>
#include <atomic>

namespace xjw
{

    struct RpcDomOptions
    {
        bool blendAllImages = true;
        bool writePreview = true;
    };

    class RpcDomGenerator
    {
    public:
        using ProgressCallback = std::function<void(const QString&, int)>;

        static bool generate(const QStringList& imagePaths,
                             const QString& demPath,
                             const QString& outputPath,
                             const RpcDomOptions& options,
                             QJsonObject* result,
                             QString* errorMessage = nullptr,
                             const ProgressCallback& progress = {},
                             const std::atomic_bool* cancelFlag = nullptr);
    };

} // namespace xjw

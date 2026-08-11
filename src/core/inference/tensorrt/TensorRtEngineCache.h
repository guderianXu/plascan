#pragma once

#include "TensorRtEngineBuilder.h"

#include <QJsonObject>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xjw::inference::detail
{

    struct TensorRtCacheIdentityOptions
    {
        QString onnxHash;
        QString tensorRtVersion;
        int cudaRuntimeVersion = 0;
        int cudaDriverVersion = 0;
        QString computeCapability;
        std::uint64_t workspaceBytes = 0;
        std::vector<TensorRtInputShape> inputShapes;
        QStringList requiredOutputNames;
    };

    QString sha256File(const QString& path, QString* errorMessage);
    QJsonObject makeCacheIdentity(const TensorRtEngineBuildRequest& request,
                                  const TensorRtCacheIdentityOptions& options);
    QString fingerprintCacheIdentity(const QJsonObject& identity);
    QString describeEngineCacheMiss(const QString& cacheRoot,
                                    const QString& engineName,
                                    const QString& currentMetadataPath,
                                    const QString& currentEnginePath,
                                    const QJsonObject& currentIdentity);

    bool loadMatchingEngineMetadata(const QString& metadataPath,
                                    const QString& enginePath,
                                    const QString& fingerprint,
                                    TensorRtEngineBuildResult* result);

    bool saveEngineMetadata(const QString& metadataPath,
                            const QJsonObject& identity,
                            const TensorRtEngineBuildResult& result,
                            const QString& onnxFileName,
                            const QString& gpuName,
                            double buildSeconds,
                            std::size_t engineBytes,
                            QString* errorMessage);

} // namespace xjw::inference::detail

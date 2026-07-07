#pragma once

#include "model/TorchScriptModelResolver.h"

#include <QString>
#include <QStringList>
#include <QVector>

#include <optional>

namespace xjw::common::model
{

struct Sam21TorchScriptFiles
{
    QString encoder;
    QString decoder;
};

struct Sam21ModelSpec
{
    QString token;
    QString displayName;
    QString checkpointFileName;
    QString checkpointUrl;
};

struct Sam21ModelStatus
{
    Sam21ModelSpec spec;
    QString checkpointPath;
    QString cpuEncoderPath;
    QString cpuDecoderPath;
    QString cudaEncoderPath;
    QString cudaDecoderPath;
    QStringList missingFiles;
    QString label;
    QString detail;
    bool hasCheckpoint = false;
    bool hasCpuTorchScript = false;
    bool hasCudaTorchScript = false;
    bool isFullyInstalled = false;
};

QVector<Sam21ModelSpec> sam21ModelSpecs();
std::optional<Sam21ModelSpec> sam21ModelSpecForToken(const QString &token);
Sam21TorchScriptFiles sam21TorchScriptFiles(const QString &variantToken, bool useCuda);
Sam21ModelStatus sam21ModelStatus(const TorchScriptModelResolver &resolver, const QString &variantToken);
QVector<Sam21ModelStatus> sam21ModelStatuses(const TorchScriptModelResolver &resolver);

} // namespace xjw::common::model

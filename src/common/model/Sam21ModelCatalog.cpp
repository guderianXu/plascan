#include "model/Sam21ModelCatalog.h"

#include <QFileInfo>

namespace xjw::common::model
{
namespace
{

QVector<Sam21ModelSpec> buildSpecs()
{
    return {
        {QStringLiteral("tiny"),
         QStringLiteral("Hiera Tiny"),
         QStringLiteral("sam2.1_hiera_tiny.pt"),
         QStringLiteral("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")},
        {QStringLiteral("small"),
         QStringLiteral("Hiera Small"),
         QStringLiteral("sam2.1_hiera_small.pt"),
         QStringLiteral("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt")},
        {QStringLiteral("base_plus"),
         QStringLiteral("Hiera Base+"),
         QStringLiteral("sam2.1_hiera_base_plus.pt"),
         QStringLiteral("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt")},
        {QStringLiteral("large"),
         QStringLiteral("Hiera Large"),
         QStringLiteral("sam2.1_hiera_large.pt"),
         QStringLiteral("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")},
    };
}

QString displayMissingFiles(const QStringList &missingFiles)
{
    if (missingFiles.isEmpty())
    {
        return QStringLiteral("全部模型文件已安装");
    }
    return QStringLiteral("缺少：%1").arg(missingFiles.join(QStringLiteral(", ")));
}

} // namespace

QVector<Sam21ModelSpec> sam21ModelSpecs()
{
    static const QVector<Sam21ModelSpec> specs = buildSpecs();
    return specs;
}

std::optional<Sam21ModelSpec> sam21ModelSpecForToken(const QString &token)
{
    const QString cleanToken = token.trimmed();
    for (const Sam21ModelSpec &spec : sam21ModelSpecs())
    {
        if (spec.token == cleanToken)
        {
            return spec;
        }
    }
    return std::nullopt;
}

Sam21TorchScriptFiles sam21TorchScriptFiles(const QString &variantToken, bool useCuda)
{
    const QString device = useCuda ? QStringLiteral("cuda") : QStringLiteral("cpu");
    const QString token = variantToken.trimmed().isEmpty() ? QStringLiteral("tiny") : variantToken.trimmed();
    return {
        QStringLiteral("sam21_hiera_%1_encoder_%2.pt").arg(token, device),
        QStringLiteral("sam21_hiera_%1_decoder_%2.pt").arg(token, device),
    };
}

Sam21ModelStatus sam21ModelStatus(const TorchScriptModelResolver &resolver, const QString &variantToken)
{
    Sam21ModelStatus status;
    const auto spec = sam21ModelSpecForToken(variantToken).value_or(sam21ModelSpecs().front());
    status.spec = spec;

    const auto cpuFiles = sam21TorchScriptFiles(spec.token, false);
    const auto cudaFiles = sam21TorchScriptFiles(spec.token, true);

    status.checkpointPath = resolver.findModel(spec.checkpointFileName);
    status.cpuEncoderPath = resolver.findModel(cpuFiles.encoder);
    status.cpuDecoderPath = resolver.findModel(cpuFiles.decoder);
    status.cudaEncoderPath = resolver.findModel(cudaFiles.encoder);
    status.cudaDecoderPath = resolver.findModel(cudaFiles.decoder);

    status.hasCheckpoint = !status.checkpointPath.isEmpty();
    status.hasCpuTorchScript = !status.cpuEncoderPath.isEmpty() && !status.cpuDecoderPath.isEmpty();
    status.hasCudaTorchScript = !status.cudaEncoderPath.isEmpty() && !status.cudaDecoderPath.isEmpty();
    status.isFullyInstalled = status.hasCpuTorchScript && status.hasCudaTorchScript;

    if (!status.hasCheckpoint)
    {
        status.missingFiles << spec.checkpointFileName;
    }
    if (status.cpuEncoderPath.isEmpty())
    {
        status.missingFiles << cpuFiles.encoder;
    }
    if (status.cpuDecoderPath.isEmpty())
    {
        status.missingFiles << cpuFiles.decoder;
    }
    if (status.cudaEncoderPath.isEmpty())
    {
        status.missingFiles << cudaFiles.encoder;
    }
    if (status.cudaDecoderPath.isEmpty())
    {
        status.missingFiles << cudaFiles.decoder;
    }

    if (status.isFullyInstalled)
    {
        status.label = QStringLiteral("已安装");
    }
    else if (!status.hasCheckpoint && !status.hasCpuTorchScript && !status.hasCudaTorchScript)
    {
        status.label = QStringLiteral("未安装");
    }
    else if (status.hasCpuTorchScript && !status.hasCudaTorchScript)
    {
        status.label = QStringLiteral("缺 CUDA");
    }
    else if (!status.hasCpuTorchScript && status.hasCudaTorchScript)
    {
        status.label = QStringLiteral("缺 CPU");
    }
    else
    {
        status.label = QStringLiteral("未完成");
    }

    status.detail = displayMissingFiles(status.missingFiles);
    return status;
}

QVector<Sam21ModelStatus> sam21ModelStatuses(const TorchScriptModelResolver &resolver)
{
    QVector<Sam21ModelStatus> statuses;
    for (const Sam21ModelSpec &spec : sam21ModelSpecs())
    {
        statuses.append(sam21ModelStatus(resolver, spec.token));
    }
    return statuses;
}

} // namespace xjw::common::model

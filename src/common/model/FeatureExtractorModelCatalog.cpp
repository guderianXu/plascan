#include "FeatureExtractorModelCatalog.h"

#include <QFileInfo>

namespace xjw::common::model
{

QStringList featureExtractorModelCandidates(const QString &algorithm, bool useCuda)
{
    const QString normalizedAlgorithm = algorithm.trimmed().toLower();
    QStringList candidates;

    if (normalizedAlgorithm == QStringLiteral("superpoint"))
    {
        if (useCuda)
        {
            candidates << QStringLiteral("superpoint_extractor_cuda.torchscript")
                       << QStringLiteral("superpoint_extractor_cuda.pt");
        }
        candidates << QStringLiteral("superpoint_extractor_cpu.torchscript")
                   << QStringLiteral("superpoint_extractor_cpu.pt")
                   << QStringLiteral("superpoint_extractor.torchscript")
                   << QStringLiteral("superpoint_extractor.pt");
        return candidates;
    }

    if (normalizedAlgorithm == QStringLiteral("disk"))
    {
        if (useCuda)
        {
            candidates << QStringLiteral("disk_extractor_cuda_8192.torchscript")
                       << QStringLiteral("disk_extractor_cuda_8192.pt")
                       << QStringLiteral("disk_extractor_cuda_1200.torchscript")
                       << QStringLiteral("disk_extractor_cuda_1200.pt");
        }
        candidates << QStringLiteral("disk_extractor_cpu_8192.torchscript")
                   << QStringLiteral("disk_extractor_cpu_8192.pt")
                   << QStringLiteral("disk_extractor_cpu_1200.torchscript")
                   << QStringLiteral("disk_extractor_cpu_1200.pt")
                   << QStringLiteral("disk_extractor.torchscript")
                   << QStringLiteral("disk_extractor.pt");
        return candidates;
    }

    if (normalizedAlgorithm == QStringLiteral("aliked"))
    {
        if (useCuda)
        {
            candidates << QStringLiteral("aliked_extractor_cuda_480.torchscript")
                       << QStringLiteral("aliked_extractor_cuda_480.pt");
        }
        candidates << QStringLiteral("aliked_extractor_cpu_480.torchscript")
                   << QStringLiteral("aliked_extractor_cpu_480.pt")
                   << QStringLiteral("aliked_extractor.torchscript")
                   << QStringLiteral("aliked_extractor.pt");
        return candidates;
    }

    return candidates;
}

bool isManagedFeatureExtractorModelPath(const QString &path)
{
    const QString fileName = QFileInfo(path).fileName().toCaseFolded();
    return fileName.startsWith(QStringLiteral("superpoint_extractor"))
        || fileName.startsWith(QStringLiteral("disk_extractor"))
        || fileName.startsWith(QStringLiteral("aliked_extractor"));
}

} // namespace xjw::common::model

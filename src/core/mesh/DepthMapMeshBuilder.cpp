#include "DepthMapMeshBuilder.h"

#include <QDir>
#include <QFileInfo>
#include <QStringList>

namespace xjw::mesh
{

QVector<DepthFrameArtifact> DepthMapMeshBuilder::discoverDepthFrames(const QString &sourcePath)
{
    const QFileInfo info(sourcePath);
    const QDir dir(info.isDir() ? info.absoluteFilePath() : info.absolutePath());
    QVector<DepthFrameArtifact> frames;

    const QStringList depthFiles =
        dir.entryList(QStringList() << QStringLiteral("depth_*.raw"), QDir::Files, QDir::Name);
    frames.reserve(depthFiles.size());
    for (const QString &fileName : depthFiles)
    {
        const QString suffix = fileName.mid(QStringLiteral("depth_").size());
        DepthFrameArtifact frame;
        frame.depthPath = dir.filePath(fileName);

        const QString confidenceName = QStringLiteral("confidence_%1").arg(suffix);
        if (QFileInfo::exists(dir.filePath(confidenceName)))
        {
            frame.confidencePath = dir.filePath(confidenceName);
        }

        const QString previewName =
            QStringLiteral("depth_%1.png").arg(QFileInfo(suffix).completeBaseName());
        if (QFileInfo::exists(dir.filePath(previewName)))
        {
            frame.previewPath = dir.filePath(previewName);
        }

        frames.push_back(frame);
    }
    return frames;
}

QString DepthMapMeshBuilder::resolveReusableDenseCloud(const QString &sourcePath, QString *errorMessage)
{
    const QFileInfo info(sourcePath);
    const QDir dir(info.isDir() ? info.absoluteFilePath() : info.absolutePath());
    const QString densePath = dir.filePath(QStringLiteral("dense_cloud.ply"));
    if (QFileInfo::exists(densePath))
    {
        return QDir::cleanPath(densePath);
    }

    if (errorMessage)
    {
        *errorMessage = QStringLiteral("未找到可复用的深度图融合点云: %1").arg(densePath);
    }
    return QString();
}

} // namespace xjw::mesh

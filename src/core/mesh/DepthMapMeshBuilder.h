#pragma once

#include <QString>
#include <QVector>

namespace xjw::mesh
{

struct DepthFrameArtifact
{
    QString depthPath;
    QString confidencePath;
    QString previewPath;
};

class DepthMapMeshBuilder
{
public:
    static QVector<DepthFrameArtifact> discoverDepthFrames(const QString &sourcePath);
    static QString resolveReusableDenseCloud(const QString &sourcePath, QString *errorMessage = nullptr);
};

} // namespace xjw::mesh

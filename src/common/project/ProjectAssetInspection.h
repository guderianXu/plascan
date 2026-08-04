#pragma once

#include <QString>
#include <QStringList>

namespace xjw::common::project::detail
{

struct ProjectAssetInspection
{
    qint64 vertexCount = 0;
    qint64 faceCount = 0;
    bool hasVertexColors = false;
    QStringList materialLibraries;
};

bool inspectProjectAsset(const QString &path,
                         const QString &format,
                         ProjectAssetInspection *inspection,
                         QString *errorMessage);

QStringList collectObjDependencies(const QString &objPath,
                                   const ProjectAssetInspection &inspection,
                                   QStringList *warnings,
                                   bool *hasMaterial,
                                   bool *hasTexture);

} // namespace xjw::common::project::detail

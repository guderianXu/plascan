#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

class ProjectData;

namespace xjw::gui::project {

QJsonObject projectFilesMeta(ProjectData *projectData);

void persistProjectMeta(ProjectData *projectData,
                        const QJsonObject &meta,
                        bool markDirty = true);

QString resolveProjectOutputDir(const QString &projectPath,
                                const QString &requestedDir,
                                const QString &fallbackRelativeDir);

bool resolveLatestDenseCloudPath(ProjectData *projectData,
                                 QString *denseCloudPath,
                                 QString *errorMessage = nullptr);

void upsertProjectRecordByPath(ProjectData *projectData,
                               const QString &arrayKey,
                               const QString &pathKey,
                               const QJsonObject &record,
                               bool markDirty = true);

void replaceProjectRecordWithLatest(ProjectData *projectData,
                                    const QString &arrayKey,
                                    const QJsonObject &record,
                                    bool markDirty = true);

void appendAtResult(ProjectData *projectData,
                    const QString &sparseCloudPath,
                    int sparsePointCount,
                    const QStringList &selectedImages,
                    const QString &outputDir,
                    const QJsonObject &extraRecord = {},
                    int replaceIndex = -1);

void appendObsNetResult(ProjectData *projectData,
                        int nodeCount,
                        int edgeCount,
                        const QString &algorithmName,
                        const QJsonObject &extraInfo = {});

} // namespace xjw::gui::project
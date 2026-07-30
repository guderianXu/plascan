#pragma once

#include "ProjectChunkIndex.h"

#include <QJsonObject>
#include <QMap>
#include <QString>

namespace xjw::common::project
{

struct ProjectResourceRef
{
    QString id;
    QString kind;
    QString name;
    QString entryPath;
    QString mediaType;
    QString sha256;
    qint64 size = 0;
    qint64 modifiedMs = 0;

    bool isValid(QString *errorMessage = nullptr) const;
    QJsonObject toJson() const;
    static ProjectResourceRef fromJson(const QJsonObject &object,
                                       QString *errorMessage = nullptr);
};

class ProjectResourceIndex
{
public:
    static constexpr int CurrentSchemaVersion = 1;

    bool isEmpty() const;
    int size() const;
    bool contains(const QString &resourceId) const;
    ProjectResourceRef resource(const QString &resourceId) const;
    QList<ProjectResourceRef> resources() const;

    bool upsert(const ProjectResourceRef &resource, QString *errorMessage = nullptr);
    bool remove(const QString &resourceId);

    QJsonObject toJson() const;
    static ProjectResourceIndex fromJson(const QJsonObject &object,
                                         QString *errorMessage = nullptr);

private:
    QMap<QString, ProjectResourceRef> _resources;
};

class PortableProjectFormat
{
public:
    static constexpr const char *ProjectType = "plascan_project";
    static constexpr const char *ChunkType = "plascan_chunk";
    static constexpr const char *CurrentFormatVersion = "4.0";
    static constexpr const char *MinimumReaderVersion = "4.0";
    static constexpr const char *ChunkFormatVersion = "1.0";
    static constexpr const char *DocumentEntry = "doc.json";
    static constexpr const char *ChunkIndexSection = "chunk_index";
    static constexpr const char *ProjectUiStateSection = "ui_state";
    static constexpr const char *ChunkRecordSection = "chunk";
    static constexpr const char *ProjectFilesSection = "project_files";
    static constexpr const char *ProjectResultsSection = "project_results";
    static constexpr const char *ProjectConfigSection = "project_config";
    static constexpr const char *ResourceIndexSection = "resource_index";

    static QString createProjectId();
    static QJsonObject createProjectDocument(
        const QString &projectId,
        const ProjectChunkIndex &chunkIndex,
        const QJsonObject &uiState,
        const QString &createdWith = QStringLiteral("PlaScan"));
    static bool isCurrentProjectDocument(const QJsonObject &document);
    static bool readProjectIndex(const QJsonObject &document,
                                 ProjectChunkIndex *chunkIndex,
                                 QString *errorMessage = nullptr);

    static QJsonObject createChunkDocument(
        const ProjectChunkRecord &chunk,
        const QJsonObject &projectFiles,
        const QJsonObject &projectResults,
        const QJsonObject &projectConfig,
        const QJsonObject &resourceIndex);
    static QJsonObject normalizeProjectResults(
        const QJsonObject &projectResults);
    static bool isCurrentChunkDocument(
        const QJsonObject &document,
        const ProjectChunkRecord *expectedChunk = nullptr,
        QString *errorMessage = nullptr);

    static QString normalizeEntryPath(const QString &entryPath);
    static bool isSafeEntryPath(const QString &entryPath);
    static QString resolveEntryPath(const QString &rootPath,
                                    const QString &entryPath,
                                    QString *errorMessage = nullptr);

    static QString resourceUriForEntry(const QString &entryPath);
    static QString entryPathFromResourceUri(const QString &resourceUri);
};

} // namespace xjw::common::project

#pragma once

#include "project/ProjectChunkIndex.h"

#include <QJsonObject>
#include <QPair>
#include <QString>
#include <QVector>

class ProjectChunkStore
{
public:
    explicit ProjectChunkStore(const QString &projectPath);

    bool ensureLayout(QString *errorMessage = nullptr) const;
    bool loadProjectDocument(QJsonObject *document,
                             QString *errorMessage = nullptr) const;
    bool writeProjectDocument(const QJsonObject &document,
                              QString *errorMessage = nullptr) const;
    bool loadIndex(xjw::common::project::ProjectChunkIndex *index,
                   QString *errorMessage = nullptr) const;
    bool saveIndex(const xjw::common::project::ProjectChunkIndex &index,
                   QString *errorMessage = nullptr) const;
    bool saveProjectUiState(const QJsonObject &uiState,
                            QString *errorMessage = nullptr) const;
    QList<xjw::common::project::ProjectChunkRecord> chunks(
        QString *errorMessage = nullptr) const;
    bool createChunk(
        const QString &name,
        const QJsonObject &projectFiles,
        const QJsonObject &projectResults,
        const QJsonObject &projectConfig,
        const QJsonObject &resourceIndex,
        xjw::common::project::ProjectChunkRecord *createdChunk = nullptr,
        QString *errorMessage = nullptr) const;
    bool renameChunk(const QString &chunkId,
                     const QString &name,
                     QString *errorMessage = nullptr) const;
    bool removeChunk(const QString &chunkId,
                     QString *errorMessage = nullptr) const;
    bool setDefaultChunk(const QString &chunkId,
                         QString *errorMessage = nullptr) const;

    xjw::common::project::ProjectChunkRecord defaultChunk(
        QString *errorMessage = nullptr) const;
    QString defaultChunkDirectory(QString *errorMessage = nullptr) const;
    QString defaultChunkArchivePath(QString *errorMessage = nullptr) const;

    bool readChunkDocument(int chunkDirectory,
                           QJsonObject *document,
                           QString *errorMessage = nullptr) const;
    bool writeChunkDocument(int chunkDirectory,
                            const QJsonObject &document,
                            QString *errorMessage = nullptr) const;
    bool readDefaultChunkDocument(QJsonObject *document,
                                  QString *errorMessage = nullptr) const;
    bool readDefaultChunkSection(const QString &sectionName,
                                 QJsonObject *section,
                                 QString *errorMessage = nullptr) const;
    bool writeDefaultChunkSections(
        const QVector<QPair<QString, QJsonObject>> &sections,
        QString *errorMessage = nullptr) const;
    bool writeChunkSections(
        int chunkDirectory,
        const QVector<QPair<QString, QJsonObject>> &sections,
        QString *errorMessage = nullptr) const;

private:
    bool validateCurrentLayout(QString *errorMessage) const;

    QString _projectPath;
};

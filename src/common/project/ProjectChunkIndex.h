#pragma once

#include <QJsonObject>
#include <QList>
#include <QString>

namespace xjw::common::project
{

struct ProjectChunkRecord
{
    QString id;
    QString name;
    int directory = 0;
    int order = 0;
    qint64 revision = 0;

    bool isValid(QString *errorMessage = nullptr) const;
    QJsonObject toJson() const;
    static ProjectChunkRecord fromJson(const QJsonObject &object,
                                       QString *errorMessage = nullptr);
};

class ProjectChunkIndex
{
public:
    static constexpr int CurrentSchemaVersion = 1;

    static ProjectChunkIndex createInitial(
        const QString &chunkName = QStringLiteral("区块 1"));

    bool isValid(QString *errorMessage = nullptr) const;
    bool isEmpty() const;
    int size() const;
    QList<ProjectChunkRecord> chunks() const;
    QString defaultChunkId() const;
    ProjectChunkRecord defaultChunk() const;
    ProjectChunkRecord chunk(const QString &chunkId) const;
    int nextChunkDirectory() const;

    QString appendChunk(const QString &name,
                        QString *errorMessage = nullptr);
    bool removeChunk(const QString &chunkId,
                     QString *errorMessage = nullptr);
    bool renameChunk(const QString &chunkId,
                     const QString &name,
                     QString *errorMessage = nullptr);
    bool touchChunk(const QString &chunkId,
                    QString *errorMessage = nullptr);
    bool setDefaultChunk(const QString &chunkId,
                         QString *errorMessage = nullptr);

    QJsonObject toJson() const;
    static ProjectChunkIndex fromJson(const QJsonObject &object,
                                      QString *errorMessage = nullptr);

private:
    QList<ProjectChunkRecord> _chunks;
    QString _defaultChunkId;
    int _nextChunkDirectory = 1;
};

} // namespace xjw::common::project

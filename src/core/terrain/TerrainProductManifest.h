#pragma once

#include <QJsonObject>
#include <QString>
#include <QVector>

namespace xjw
{

struct TerrainProductRecord
{
    QString productId;
    QString productType;
    QString createdAt;
    QString sourcePath;
    QString demPath;
    QString domPath;
    QString errorPath;
    QString countPath;
    QString confidencePath;
    QString coveragePath;
    QString previewPath;
    QString projection;
    QString aggregation;
    double gridResolution = 0.0;
    int gridWidth = 0;
    int gridHeight = 0;
    QJsonObject extra;

    QString primaryPath() const;
    QJsonObject toJson() const;
    static TerrainProductRecord fromJson(const QJsonObject &object);
};

class TerrainProductManifest
{
public:
    bool load(const QString &path, QString *errorMsg = nullptr);
    bool saveAtomic(const QString &path, QString *errorMsg = nullptr) const;

    void clear();

    const QVector<TerrainProductRecord> &records() const;
    QVector<TerrainProductRecord> recordsSortedByPrimaryPath() const;

    void upsertRecord(const TerrainProductRecord &record);

    QJsonObject toJson() const;

private:
    int findRecordIndex(const QString &productId) const;

    QVector<TerrainProductRecord> m_records;
};

} // namespace xjw

#include "TerrainProductManifest.h"

#include <QCollator>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

#include <algorithm>

namespace xjw
{

namespace
{

QString jsonString(const QJsonObject &object, const QString &key)
{
    return object.value(key).toString();
}

void insertIfNotEmpty(QJsonObject *object, const QString &key, const QString &value)
{
    if (!object || value.isEmpty())
    {
        return;
    }
    object->insert(key, value);
}

} // namespace

QString TerrainProductRecord::primaryPath() const
{
    if (!demPath.isEmpty())
    {
        return demPath;
    }
    if (!domPath.isEmpty())
    {
        return domPath;
    }
    if (!errorPath.isEmpty())
    {
        return errorPath;
    }
    if (!coveragePath.isEmpty())
    {
        return coveragePath;
    }
    return sourcePath;
}

QJsonObject TerrainProductRecord::toJson() const
{
    QJsonObject object = extra;
    insertIfNotEmpty(&object, QStringLiteral("product_id"), productId);
    insertIfNotEmpty(&object, QStringLiteral("product_type"), productType);
    insertIfNotEmpty(&object, QStringLiteral("created_at"), createdAt);
    insertIfNotEmpty(&object, QStringLiteral("source_path"), sourcePath);
    insertIfNotEmpty(&object, QStringLiteral("dem_path"), demPath);
    insertIfNotEmpty(&object, QStringLiteral("dom_path"), domPath);
    insertIfNotEmpty(&object, QStringLiteral("error_path"), errorPath);
    insertIfNotEmpty(&object, QStringLiteral("count_path"), countPath);
    insertIfNotEmpty(&object, QStringLiteral("confidence_path"), confidencePath);
    insertIfNotEmpty(&object, QStringLiteral("coverage_path"), coveragePath);
    insertIfNotEmpty(&object, QStringLiteral("preview_path"), previewPath);
    insertIfNotEmpty(&object, QStringLiteral("projection"), projection);
    insertIfNotEmpty(&object, QStringLiteral("aggregation"), aggregation);
    object.insert(QStringLiteral("grid_resolution"), gridResolution);
    object.insert(QStringLiteral("grid_width"), gridWidth);
    object.insert(QStringLiteral("grid_height"), gridHeight);

    // Keep legacy GUI field names during the transition to the formal terrain manifest.
    insertIfNotEmpty(&object, QStringLiteral("dem_tif"), demPath);
    insertIfNotEmpty(&object, QStringLiteral("dom_png"), domPath);
    insertIfNotEmpty(&object, QStringLiteral("depth_png"), previewPath);
    return object;
}

TerrainProductRecord TerrainProductRecord::fromJson(const QJsonObject &object)
{
    TerrainProductRecord record;
    record.extra = object;
    record.productId = jsonString(object, QStringLiteral("product_id"));
    record.productType = jsonString(object, QStringLiteral("product_type"));
    record.createdAt = jsonString(object, QStringLiteral("created_at"));
    record.sourcePath = jsonString(object, QStringLiteral("source_path"));
    record.demPath = jsonString(object, QStringLiteral("dem_path"));
    record.domPath = jsonString(object, QStringLiteral("dom_path"));
    record.errorPath = jsonString(object, QStringLiteral("error_path"));
    record.countPath = jsonString(object, QStringLiteral("count_path"));
    record.confidencePath = jsonString(object, QStringLiteral("confidence_path"));
    record.coveragePath = jsonString(object, QStringLiteral("coverage_path"));
    record.previewPath = jsonString(object, QStringLiteral("preview_path"));
    record.projection = jsonString(object, QStringLiteral("projection"));
    record.aggregation = jsonString(object, QStringLiteral("aggregation"));
    record.gridResolution = object.value(QStringLiteral("grid_resolution")).toDouble(0.0);
    record.gridWidth = object.value(QStringLiteral("grid_width")).toInt(0);
    record.gridHeight = object.value(QStringLiteral("grid_height")).toInt(0);

    if (record.demPath.isEmpty())
    {
        record.demPath = jsonString(object, QStringLiteral("dem_tif"));
    }
    if (record.domPath.isEmpty())
    {
        record.domPath = jsonString(object, QStringLiteral("dom_png"));
    }
    if (record.previewPath.isEmpty())
    {
        record.previewPath = jsonString(object, QStringLiteral("depth_png"));
    }
    return record;
}

bool TerrainProductManifest::load(const QString &path, QString *errorMsg)
{
    clear();

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMsg)
        {
            *errorMsg = file.errorString();
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        if (errorMsg)
        {
            *errorMsg = parseError.errorString();
        }
        return false;
    }

    const QJsonArray records = document.object().value(QStringLiteral("products")).toArray();
    m_records.reserve(records.size());
    for (const QJsonValue &value : records)
    {
        if (value.isObject())
        {
            m_records.push_back(TerrainProductRecord::fromJson(value.toObject()));
        }
    }
    return true;
}

bool TerrainProductManifest::saveAtomic(const QString &path, QString *errorMsg) const
{
    const QFileInfo info(path);
    if (!info.absoluteDir().exists() && !QDir().mkpath(info.absolutePath()))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法创建目录: %1").arg(info.absolutePath());
        }
        return false;
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMsg)
        {
            *errorMsg = file.errorString();
        }
        return false;
    }

    file.write(QJsonDocument(toJson()).toJson(QJsonDocument::Indented));
    if (!file.commit())
    {
        if (errorMsg)
        {
            *errorMsg = file.errorString();
        }
        return false;
    }
    return true;
}

void TerrainProductManifest::clear()
{
    m_records.clear();
}

const QVector<TerrainProductRecord> &TerrainProductManifest::records() const
{
    return m_records;
}

QVector<TerrainProductRecord> TerrainProductManifest::recordsSortedByPrimaryPath() const
{
    QVector<TerrainProductRecord> result = m_records;
    QCollator collator;
    collator.setNumericMode(true);
    collator.setCaseSensitivity(Qt::CaseInsensitive);
    std::stable_sort(result.begin(),
                     result.end(),
                     [&collator](const TerrainProductRecord &lhs, const TerrainProductRecord &rhs)
                     {
                         return collator.compare(QFileInfo(lhs.primaryPath()).fileName(),
                                                 QFileInfo(rhs.primaryPath()).fileName()) < 0;
                     });
    return result;
}

void TerrainProductManifest::upsertRecord(const TerrainProductRecord &record)
{
    const int index = findRecordIndex(record.productId);
    if (index >= 0)
    {
        m_records[index] = record;
    }
    else
    {
        m_records.push_back(record);
    }
}

QJsonObject TerrainProductManifest::toJson() const
{
    QJsonObject root;
    root.insert(QStringLiteral("schema"), QStringLiteral("plascan.terrain.products.v1"));

    QJsonArray records;
    for (const TerrainProductRecord &record : m_records)
    {
        records.push_back(record.toJson());
    }
    root.insert(QStringLiteral("products"), records);
    return root;
}

int TerrainProductManifest::findRecordIndex(const QString &productId) const
{
    if (productId.isEmpty())
    {
        return -1;
    }
    for (int i = 0; i < m_records.size(); ++i)
    {
        if (m_records[i].productId == productId)
        {
            return i;
        }
    }
    return -1;
}

} // namespace xjw

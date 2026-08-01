// =============================================================================
// 文件名: ProjectFilesManager.cpp
// 描述:   ProjectFilesManager 内存数据模型实现。
//
//         数据拆分为两个对象，分别对应 Chunk doc.json 中的两个字段：
//
//   【_coreFiles → project_files】始终加载
//     { "images": [...] }
//
//   【_resultFiles → project_results】惰性加载（需要时再读）
//     {
//       "image_match_results": [
//         { "image": "...", "output": "...", "neighbors": ["..."] }
//         // 一幅影像一条记录；output 指向该影像唯一 `.pimatch` 分片。
//       ],
//       "intersection_results":   [...],
//       "bundle_adjust_results":  [...],
//       "depth_map_results":      [...],
//       "dense_cloud_results":    [...],
//       "model_results":          [...],
//       "dem_results":            [...],
//       "ortho_results":          [...],
//       "report_results":         [...],
//       "reference_datasets":     [...]
//     }
//
// =============================================================================
#include "ProjectDocumentModel.h"

#include <QDateTime>
#include <QFileInfo>
#include <QDir>
#include <QSet>
#include <QHash>

// ── 工具：判断 key 是否属于 results 域 ───────────────────────────────────────
bool ProjectFilesManager::isResultKey(const QString &key)
{
    return key == QLatin1String("image_match_results")
        || key == QLatin1String("intersection_results")
        || key == QLatin1String("bundle_adjust_results")
        || key == QLatin1String("aerial_triangulation_results")
        || key == QLatin1String("observation_network_results")
        || key == QLatin1String("depth_map_results")
        || key == QLatin1String("dense_cloud_results")
        || key == QLatin1String("model_results")
        || key == QLatin1String("dem_results")
        || key == QLatin1String("ortho_results")
        || key == QLatin1String("report_results")
        || key == QLatin1String("reference_datasets");
}

// ── 默认结构 ─────────────────────────────────────────────────────────────────
QJsonObject ProjectFilesManager::defaultFiles()
{
    QJsonObject files;
    files[QLatin1String("images")] = QJsonArray();
    return files;
}

QJsonObject ProjectFilesManager::defaultResults()
{
    return QJsonObject();   // 所有数组按需初始化为空，无需预建
}

// ── 拆分/合并接口 ─────────────────────────────────────────────────────────────

void ProjectFilesManager::setResultsData(const QJsonObject &data)
{
    _resultFiles  = data;
    _resultsDirty = false;
}

QJsonObject ProjectFilesManager::data() const
{
    // 合并 core + results，供需要全量数据的历史调用方使用
    QJsonObject merged = _coreFiles;
    for (auto it = _resultFiles.constBegin(); it != _resultFiles.constEnd(); ++it) {
        merged.insert(it.key(), it.value());
    }
    return merged;
}

void ProjectFilesManager::setData(const QJsonObject &data)
{
    // 将旧格式的整体 JSON 拆分到 core 和 results 两个内存对象
    QJsonObject core;
    QJsonObject results;
    for (auto it = data.constBegin(); it != data.constEnd(); ++it) {
        if (isResultKey(it.key())) {
            results.insert(it.key(), it.value());
        } else {
            core.insert(it.key(), it.value());
        }
    }
    _coreFiles    = core;
    _resultFiles  = results;
    _resultsDirty = false;
}

// ── 查询接口 ─────────────────────────────────────────────────────────────────

QStringList ProjectFilesManager::getAllImages() const
{
    QStringList result;
    const QJsonArray images = _coreFiles.value(QLatin1String("images")).toArray();
    for (const QJsonValue &val : images) {
        const QString path = val.toObject().value(QLatin1String("path")).toString();
        if (!path.isEmpty())
            result << path;
    }
    return result;
}

QStringList ProjectFilesManager::getImagesByCategory(const QString &category) const
{
    QStringList result;
    const QJsonArray images = _coreFiles.value(QLatin1String("images")).toArray();
    for (const QJsonValue &val : images) {
        const QJsonObject obj = val.toObject();
        if (category.isEmpty() || obj.value(QLatin1String("category")).toString() == category)
            result << obj.value(QLatin1String("path")).toString();
    }
    return result;
}

namespace
{
QString normalizedImageKey(const QString &path)
{
    QString key = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
#if defined(Q_OS_WIN)
    key = key.toLower();
#endif
    return key;
}

bool sameImage(const QString &left, const QString &right)
{
    return !left.trimmed().isEmpty() && normalizedImageKey(left) == normalizedImageKey(right);
}
}

QMap<QString, QString> ProjectFilesManager::getImageMatchOutputMap() const
{
    QMap<QString, QString> result;
    const QJsonArray records = _resultFiles.value(QLatin1String("image_match_results")).toArray();
    for (const QJsonValue &value : records)
    {
        const QJsonObject object = value.toObject();
        const QString image = object.value(QLatin1String("image")).toString();
        const QString output = object.value(QLatin1String("output")).toString();
        if (!image.isEmpty() && !output.isEmpty())
        {
            result.insert(image, output);
        }
    }
    return result;
}

QString ProjectFilesManager::findMatchFile(const QString &imgA, const QString &imgB) const
{
    const QJsonArray results = _resultFiles.value(QLatin1String("image_match_results")).toArray();
    for (const QJsonValue &value : results)
    {
        const QJsonObject object = value.toObject();
        const QString owner = object.value(QLatin1String("image")).toString();
        const QString output = object.value(QLatin1String("output")).toString();
        if (output.isEmpty())
        {
            continue;
        }
        const QJsonArray neighbors = object.value(QLatin1String("neighbors")).toArray();
        for (const QJsonValue &neighbor : neighbors)
        {
            const QString peer = neighbor.toString();
            if ((sameImage(owner, imgA) && sameImage(peer, imgB)) ||
                (sameImage(owner, imgB) && sameImage(peer, imgA)))
            {
                return output;
            }
        }
    }
    return {};
}

// ── 修改接口 ─────────────────────────────────────────────────────────────────

void ProjectFilesManager::setImages(const QJsonArray &images)
{
    _coreFiles[QLatin1String("images")] = images;
}

void ProjectFilesManager::appendImageMatchResult(const ProjectImageMatchResultRecord &record)
{
    appendImageMatchResults(QVector<ProjectImageMatchResultRecord>{record});
}

void ProjectFilesManager::appendImageMatchResults(
    const QVector<ProjectImageMatchResultRecord> &records)
{
    QJsonArray results = _resultFiles.value(QLatin1String("image_match_results")).toArray();

    QHash<QString, int> existingImages;
    existingImages.reserve(results.size());
    for (int i = 0; i < results.size(); ++i)
    {
        const QString existing = normalizedImageKey(
            results[i].toObject().value(QLatin1String("image")).toString());
        if (!existing.isEmpty())
        {
            existingImages.insert(existing, i);
        }
    }

    for (const ProjectImageMatchResultRecord &record : records)
    {
        const QString imageKey = normalizedImageKey(record.image);
        if (imageKey.isEmpty() || record.output.trimmed().isEmpty())
        {
            continue;
        }

        QJsonObject rec;
        rec[QLatin1String("image")] = record.image;
        rec[QLatin1String("output")] = record.output;
        rec[QLatin1String("neighbors")] = QJsonArray::fromStringList(record.neighbors);
        rec[QLatin1String("neighbor_count")] = record.neighbors.size();
        rec[QLatin1String("storage_format")] = QStringLiteral("pimatch");
        rec[QLatin1String("storage_format_version")] = 1;
        rec[QLatin1String("settings")] = record.settings;
        rec[QLatin1String("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);

        const auto found = existingImages.constFind(imageKey);
        if (found != existingImages.constEnd())
        {
            results[found.value()] = rec;
        }
        else
        {
            existingImages.insert(imageKey, results.size());
            results.append(rec);
        }
    }

    _resultFiles[QLatin1String("image_match_results")] = results;
    _resultsDirty = true;
}

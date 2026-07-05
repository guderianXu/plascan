// =============================================================================
// 文件名: ProjectFilesManager.cpp
// 描述:   ProjectFilesManager 内存数据模型实现。
//
//         数据拆分为两個对象，分别对应归档中的两个文件：
//
//   【_coreFiles → project_files.json】始终加载
//     { "images": [...] }
//
//   【_resultFiles → project_results.json】惰性加载（需要时再读）
//     {
//       "ipfind_results":  [...],
//       "ipmatch_results": [
//         { "output": "...", "image0": "...", "image1": "...", "created_at": "..." }
//         // 精简格式：去掉 sp0/sp1/pair_name 等冗余字段，sidecar 由 output+".json" 推导
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
//   旧格式兼容：读取时若 record 含 "settings" 字段则按旧格式解析，写入时统一精简格式。
// =============================================================================
#include "ProjectFilesManager.h"

#include <QDateTime>
#include <QDataStream>
#include <QFile>
#include <QFileInfo>
#include <QtEndian>
#include <QDir>
#include <QJsonDocument>
#include <QSet>
#include <QHash>

// ── 工具：判断 key 是否属于 results 域 ───────────────────────────────────────
bool ProjectFilesManager::isResultKey(const QString &key)
{
    return key == QLatin1String("ipfind_results")
        || key == QLatin1String("ipmatch_results")
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

QMap<QString, QString> ProjectFilesManager::getIpfindOutputMap() const
{
    QMap<QString, QString> result;
    const QJsonArray results = _resultFiles.value(QLatin1String("ipfind_results")).toArray();
    for (const QJsonValue &val : results) {
        const QJsonObject obj = val.toObject();
        const QString input  = obj.value(QLatin1String("input")).toString();
        const QString output = obj.value(QLatin1String("output")).toString();
        if (!input.isEmpty() && !output.isEmpty())
            result[input] = output;
    }
    return result;
}

QString ProjectFilesManager::findMatchFile(const QString &imgA, const QString &imgB) const
{
    const QJsonArray results = _resultFiles.value(QLatin1String("ipmatch_results")).toArray();
    for (const QJsonValue &val : results) {
        const QJsonObject obj = val.toObject();
        // 新格式：顶层 image0 / image1
        QString a = obj.value(QLatin1String("image0")).toString();
        QString b = obj.value(QLatin1String("image1")).toString();
        // 旧格式兼容：settings.image_files
        if (a.isEmpty() || b.isEmpty()) {
            const QJsonArray imageFiles = obj.value(QLatin1String("settings"))
                                            .toObject()
                                            .value(QLatin1String("image_files"))
                                            .toArray();
            if (imageFiles.size() >= 2) {
                a = imageFiles.at(0).toString();
                b = imageFiles.at(1).toString();
            }
        }
        if ((a == imgA && b == imgB) || (a == imgB && b == imgA))
            return obj.value(QLatin1String("output")).toString();
    }
    return {};
}

// ── 修改接口 ─────────────────────────────────────────────────────────────────

void ProjectFilesManager::setImages(const QJsonArray &images)
{
    _coreFiles[QLatin1String("images")] = images;
}

void ProjectFilesManager::appendIpfindResult(const QString &input,
                                              const QString &output,
                                              const QJsonObject &settings)
{
    appendIpfindResults(QVector<ProjectIpfindResultRecord>{
        ProjectIpfindResultRecord{input, output, settings}
    });
}

void ProjectFilesManager::appendIpfindResults(const QVector<ProjectIpfindResultRecord> &records)
{
    QJsonArray results = _resultFiles.value(QLatin1String("ipfind_results")).toArray();

    QHash<QString, int> existingInputs;
    existingInputs.reserve(results.size());
    for (int i = 0; i < results.size(); ++i)
    {
        const QString existing = QDir::cleanPath(results[i].toObject().value(QLatin1String("input")).toString());
        if (!existing.isEmpty())
        {
            existingInputs.insert(existing, i);
        }
    }

    for (const ProjectIpfindResultRecord &record : records)
    {
        const QString cleanInput = QDir::cleanPath(record.input);
        if (cleanInput.isEmpty() || record.output.trimmed().isEmpty())
        {
            continue;
        }

        QJsonObject rec;
        rec[QLatin1String("input")]      = record.input;
        rec[QLatin1String("output")]     = record.output;
        rec[QLatin1String("settings")]   = record.settings;   // ipfind settings 仍保留（字段少）
        rec[QLatin1String("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);

        const auto found = existingInputs.constFind(cleanInput);
        if (found != existingInputs.constEnd())
        {
            results[found.value()] = rec;
        }
        else
        {
            existingInputs.insert(cleanInput, results.size());
            results.append(rec);
        }
    }

    _resultFiles[QLatin1String("ipfind_results")] = results;
    _resultsDirty = true;
}

void ProjectFilesManager::appendIpmatchResult(const QStringList &outputs,
                                               const QJsonObject &settings)
{
    appendIpmatchResults(QVector<ProjectIpmatchResultRecord>{
        ProjectIpmatchResultRecord{outputs, settings}
    });
}

void ProjectFilesManager::appendIpmatchResults(const QVector<ProjectIpmatchResultRecord> &records)
{
    QJsonArray results = _resultFiles.value(QLatin1String("ipmatch_results")).toArray();

    // 去重索引：已有的 output 路径集合
    QSet<QString> existingOutputs;
    existingOutputs.reserve(results.size());
    for (const QJsonValue &val : results) {
        const QString existing = val.toObject().value(QLatin1String("output")).toString();
        if (!existing.isEmpty())
            existingOutputs.insert(QDir::cleanPath(existing));
    }

    for (const ProjectIpmatchResultRecord &record : records)
    {
        const QJsonObject &settings = record.settings;

        // 从 settings 中提取 image0/image1 路径（精简存储，不再存 sp0/sp1/pair_name 等冗余字段）
        QString image0, image1;
        const QJsonArray imageFiles = settings.value(QLatin1String("image_files")).toArray();
        if (imageFiles.size() >= 2) {
            image0 = imageFiles.at(0).toString();
            image1 = imageFiles.at(1).toString();
        }

        // 从路径提取影像基名（供 ObservationNetworkBuilder 等使用，避免后续重复 IO）
        const QString image0Name = QFileInfo(image0).completeBaseName();
        const QString image1Name = QFileInfo(image1).completeBaseName();
        const QString createdAt  = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);

        for (const QString &output : record.outputs) {
            const QString cleanOutput = QDir::cleanPath(output);
            if (existingOutputs.contains(cleanOutput))
                continue;

            // 读取 num_matches：先尝试 SuperGlue sidecar JSON，再尝试 SGMT 二进制
            int     numMatches = 0;
            QString nm0 = image0Name;
            QString nm1 = image1Name;
        {
            QFile sf(output + QStringLiteral(".json"));
            if (sf.open(QIODevice::ReadOnly)) {
                const QJsonDocument sc = QJsonDocument::fromJson(sf.readAll());
                if (sc.isObject()) {
                    const QJsonObject &scObj = sc.object();
                    numMatches = scObj.value(QLatin1String("num_matches")).toInt(0);
                    if (nm0.isEmpty()) nm0 = scObj.value(QLatin1String("image0_name")).toString();
                    if (nm1.isEmpty()) nm1 = scObj.value(QLatin1String("image1_name")).toString();
                }
            }
        }
        // 若 sidecar 不存在，尝试从匹配二进制文件直接读取匹配数
        if (numMatches == 0) {
            QFile mf(output);
            if (mf.open(QIODevice::ReadOnly)) {
                // 先尝试 SGMT 格式
                QDataStream in(&mf);
                in.setVersion(QDataStream::Qt_5_15);
                char magic[4];
                bool parsed = false;
                if (in.readRawData(magic, 4) == 4 && strncmp(magic, "SGMT", 4) == 0) {
                    quint32 ver = 0; in >> ver;
                    if (ver == 1) {
                        quint32 n0 = 0; in >> n0; in.skipRawData(n0);
                        quint32 n1 = 0; in >> n1; in.skipRawData(n1);
                        qint32 cnt = 0; in >> cnt;
                        numMatches = (int)cnt;
                        parsed = true;
                    }
                }
                // 不是 SGMT 则尝试 ASP/VisionWorkbench 格式
                if (!parsed) {
                    mf.seek(0);
                    QByteArray hdr = mf.read(16);
                    if (hdr.size() == 16) {
                        quint64 c1 = 0, c2 = 0;
                        memcpy(&c1, hdr.constData(),     8);
                        memcpy(&c2, hdr.constData() + 8, 8);
                        c1 = qFromLittleEndian<quint64>(c1);
                        c2 = qFromLittleEndian<quint64>(c2);
                        if (c1 == c2 && c1 > 0 && c1 < 1000000)
                            numMatches = (int)c1;
                    }
                }
            }
        }

        // ── 记录：匹配文件路径 + 影像路径对 + 名称 + 匹配数 ──
        QJsonObject rec;
        rec[QLatin1String("output")]      = output;
        rec[QLatin1String("image0")]      = image0;
        rec[QLatin1String("image1")]      = image1;
        rec[QLatin1String("image0_name")] = nm0;
        rec[QLatin1String("image1_name")] = nm1;
        rec[QLatin1String("num_matches")] = numMatches;
        rec[QLatin1String("created_at")]  = createdAt;

        results.append(rec);
        existingOutputs.insert(cleanOutput);
        }
    }

    _resultFiles[QLatin1String("ipmatch_results")] = results;
    _resultsDirty = true;
}

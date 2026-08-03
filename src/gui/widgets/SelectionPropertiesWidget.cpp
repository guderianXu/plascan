#include "SelectionPropertiesWidget.h"

#include <QAbstractItemView>
#include <QBrush>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFont>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonValue>
#include <QLabel>
#include <QLocale>
#include <QSize>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>

namespace
{

QString cleanPath(const QString &path)
{
    return QDir::cleanPath(path);
}

bool samePathText(const QString &left, const QString &right)
{
    if (left.isEmpty() || right.isEmpty())
    {
        return false;
    }
    return QString::compare(cleanPath(left), cleanPath(right), Qt::CaseInsensitive) == 0;
}

bool imageEntryMatches(const QJsonObject &entry,
                       const QString &targetPath,
                       const QString &targetAbsolutePath,
                       const QString &targetName)
{
    const QString path = entry.value(QStringLiteral("path")).toString();
    const QFileInfo pathInfo(path);
    const QString absolutePath = pathInfo.absoluteFilePath();
    const QString name = entry.value(QStringLiteral("name")).toString();
    const QString fileName = pathInfo.fileName();

    return samePathText(path, targetPath)
        || samePathText(path, targetAbsolutePath)
        || samePathText(absolutePath, targetPath)
        || samePathText(absolutePath, targetAbsolutePath)
        || (!targetName.isEmpty() && (name == targetName || fileName == targetName));
}

QJsonObject findImageEntryInArray(const QJsonArray &images,
                                  const QString &targetPath,
                                  const QString &targetAbsolutePath,
                                  const QString &targetName)
{
    for (const QJsonValue &value : images)
    {
        const QJsonObject entry = value.toObject();
        if (imageEntryMatches(entry, targetPath, targetAbsolutePath, targetName))
        {
            return entry;
        }
    }
    return {};
}

QString recordPath(const QJsonObject &record)
{
    for (const QString &key : {
             QStringLiteral("final_model_path"),
             QStringLiteral("model_obj"),
             QStringLiteral("model_ply"),
             QStringLiteral("mesh_ply"),
             QStringLiteral("dense_cloud_xyz"),
             QStringLiteral("dem_tif"),
             QStringLiteral("ortho_tif"),
             QStringLiteral("path")})
    {
        const QString path = record.value(key).toString().trimmed();
        if (!path.isEmpty())
        {
            return path;
        }
    }
    return {};
}

QString sourceDataText(const QString &source)
{
    if (source == QStringLiteral("depth_maps")) return QStringLiteral("深度图");
    if (source == QStringLiteral("point_cloud")) return QStringLiteral("点云");
    if (source == QStringLiteral("tie_points")) return QStringLiteral("连接点");
    if (source == QStringLiteral("model")) return QStringLiteral("模型");
    return source.isEmpty() ? QStringLiteral("不可用") : source;
}

QString surfaceTypeText(const QString &surfaceType)
{
    if (surfaceType == QStringLiteral("arbitrary_3d")) return QStringLiteral("任意 (3D)");
    if (surfaceType == QStringLiteral("height_field")) return QStringLiteral("高度场 (2.5D)");
    return surfaceType.isEmpty() ? QStringLiteral("不可用") : surfaceType;
}

QString interpolationText(const QString &interpolation)
{
    if (interpolation == QStringLiteral("disabled")) return QStringLiteral("已禁用");
    if (interpolation == QStringLiteral("enabled")) return QStringLiteral("已启用");
    if (interpolation == QStringLiteral("extrapolated")) return QStringLiteral("外推");
    return interpolation.isEmpty() ? QStringLiteral("不可用") : interpolation;
}

QString qualityText(QString profile)
{
    profile = profile.trimmed().toLower();
    if (profile == QStringLiteral("highest")) return QStringLiteral("超高");
    if (profile == QStringLiteral("high")) return QStringLiteral("高");
    if (profile == QStringLiteral("medium")) return QStringLiteral("中等");
    if (profile == QStringLiteral("low")) return QStringLiteral("低");
    if (profile == QStringLiteral("lowest")) return QStringLiteral("超低");
    if (profile == QStringLiteral("detail")) return QStringLiteral("细节");
    if (profile == QStringLiteral("balanced")) return QStringLiteral("均衡");
    if (profile == QStringLiteral("lite")) return QStringLiteral("轻量");
    return profile.isEmpty() ? QStringLiteral("不可用") : profile;
}

QString filterText(QString mode)
{
    mode = mode.trimmed().toLower();
    if (mode == QStringLiteral("disabled")) return QStringLiteral("已禁用");
    if (mode == QStringLiteral("mild")) return QStringLiteral("轻度");
    if (mode == QStringLiteral("moderate")) return QStringLiteral("中度");
    if (mode == QStringLiteral("aggressive")) return QStringLiteral("强度");
    if (mode == QStringLiteral("auto")) return QStringLiteral("自适应");
    return mode.isEmpty() ? QStringLiteral("不可用") : mode;
}

QString elapsedText(double milliseconds)
{
    if (milliseconds < 0.0)
    {
        return QStringLiteral("不可用");
    }
    if (milliseconds < 1000.0)
    {
        return QStringLiteral("%1 毫秒").arg(milliseconds, 0, 'f', 0);
    }
    return QStringLiteral("%1 秒").arg(milliseconds / 1000.0, 0, 'f', 2);
}

QString creationTimeText(const QString &text)
{
    const QDateTime parsed = QDateTime::fromString(text, Qt::ISODate);
    if (!parsed.isValid())
    {
        return text.isEmpty() ? QStringLiteral("不可用") : text;
    }
    return parsed.toLocalTime().toString(QStringLiteral("yyyy.MM.dd HH:mm:ss"));
}

bool pathBelongsToDirectory(const QString &path, const QString &directory)
{
    if (path.isEmpty() || directory.isEmpty())
    {
        return false;
    }
    const QString normalizedPath = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    QString normalizedDirectory = QDir::cleanPath(QFileInfo(directory).absoluteFilePath());
    if (!normalizedDirectory.endsWith(QDir::separator()))
    {
        normalizedDirectory += QDir::separator();
    }
    return normalizedPath.startsWith(normalizedDirectory, Qt::CaseInsensitive);
}

} // namespace

SelectionPropertiesWidget::SelectionPropertiesWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(4);

    _title = new QLabel(tr("未选择"), this);
    _title->setTextInteractionFlags(Qt::TextSelectableByMouse);

    _table = new QTableWidget(this);
    _table->setColumnCount(2);
    _table->setHorizontalHeaderLabels({tr("属性"), tr("值")});
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->verticalHeader()->setVisible(false);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setSelectionMode(QAbstractItemView::NoSelection);
    _table->setAlternatingRowColors(true);

    layout->addWidget(_title);
    layout->addWidget(_table, 1);
}

void SelectionPropertiesWidget::clearSelection()
{
    setRows(tr("未选择"), {});
}

void SelectionPropertiesWidget::showPhotoProperties(const QJsonObject &meta, const QString &imagePath)
{
    QVector<PropertyRow> rows;
    const QFileInfo info(imagePath);
    const QJsonObject entry = findImageEntry(meta, imagePath);

    rows.push_back({tr("名称"), info.fileName()});
    rows.push_back({tr("路径"), imagePath});
    rows.push_back({tr("定向状态"), imageAlignedText(entry)});

    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    const QSize size = reader.size();
    if (size.isValid())
    {
        rows.push_back({tr("尺寸"), QStringLiteral("%1 x %2").arg(size.width()).arg(size.height())});
    }
    const QByteArray format = reader.format();
    rows.push_back({tr("格式"), format.isEmpty() ? tr("未知") : QString::fromLatin1(format).toUpper()});

    appendFileRows(&rows, imagePath);

    const QString center = cameraCenterText(entry);
    if (!center.isEmpty())
    {
        rows.push_back({tr("相机中心"), center});
    }
    const QString intrinsics = intrinsicsText(entry);
    if (!intrinsics.isEmpty())
    {
        rows.push_back({tr("内方位"), intrinsics});
    }

    setRows(tr("照片属性"), rows);
}

void SelectionPropertiesWidget::showResourceProperties(const QJsonObject &meta,
                                                       const QString &section,
                                                       const QString &resourcePath)
{
    const QJsonObject record = findResourceRecord(meta, section, resourcePath);
    if (section.contains(tr("3D模型")) ||
        section.contains(tr("模型")) ||
        record.value(QStringLiteral("result_type")).toString() == QStringLiteral("mesh"))
    {
        setRows(tr("模型属性"), modelPropertyRows(meta, record, resourcePath));
        return;
    }

    QVector<PropertyRow> rows;
    const QFileInfo info(resourcePath);
    rows.push_back({tr("类型"), section});
    rows.push_back({tr("名称"), info.fileName().isEmpty() ? section : info.fileName()});
    rows.push_back({tr("路径"), resourcePath});
    rows.push_back({tr("扩展名"), info.suffix().isEmpty() ? tr("无") : info.suffix().toLower()});
    appendFileRows(&rows, resourcePath);

    if (section.contains(tr("点云")) || section.contains(tr("连接点")))
    {
        rows.push_back({tr("详细属性"), tr("未扫描详细属性，避免在主界面阻塞大点云加载")});
    }

    setRows(tr("资源属性"), rows);
}

void SelectionPropertiesWidget::setRows(const QString &title, const QVector<PropertyRow> &rows)
{
    if (_title)
    {
        _title->setText(title);
    }
    if (!_table)
    {
        return;
    }

    _table->setRowCount(rows.size());
    for (int row = 0; row < rows.size(); ++row)
    {
        auto *nameItem = new QTableWidgetItem(rows[row].name);
        nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
        _table->setItem(row, 0, nameItem);
        if (rows[row].sectionHeader)
        {
            QFont font = nameItem->font();
            font.setBold(true);
            nameItem->setFont(font);
            nameItem->setBackground(QBrush(palette().alternateBase()));
            _table->setSpan(row, 0, 1, 2);
            continue;
        }

        auto *valueItem = new QTableWidgetItem(rows[row].value);
        valueItem->setFlags(valueItem->flags() & ~Qt::ItemIsEditable);
        _table->setItem(row, 1, valueItem);
    }
    _table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    _table->resizeRowsToContents();
}

void SelectionPropertiesWidget::appendFileRows(QVector<PropertyRow> *rows, const QString &path) const
{
    if (!rows)
    {
        return;
    }
    const QFileInfo info(path);
    rows->push_back({tr("存在"), info.exists() ? tr("是") : tr("否")});
    if (info.exists())
    {
        rows->push_back({tr("文件大小"), fileSizeText(info.size())});
        rows->push_back({tr("修改时间"), info.lastModified().toString(Qt::ISODate)});
    }
}

QJsonObject SelectionPropertiesWidget::findImageEntry(const QJsonObject &meta, const QString &imagePath) const
{
    const QFileInfo targetInfo(imagePath);
    const QString targetPath = imagePath;
    const QString targetAbsolutePath = targetInfo.absoluteFilePath();
    const QString targetName = targetInfo.fileName();

    const QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    QJsonObject entry = findImageEntryInArray(images, targetPath, targetAbsolutePath, targetName);
    if (!entry.isEmpty())
    {
        return entry;
    }

    const QJsonObject projectFiles = meta.value(QStringLiteral("project_files")).toObject();
    const QJsonArray projectImages = projectFiles.value(QStringLiteral("images")).toArray();
    return findImageEntryInArray(projectImages, targetPath, targetAbsolutePath, targetName);
}

QJsonObject SelectionPropertiesWidget::findResourceRecord(const QJsonObject &meta,
                                                           const QString &section,
                                                           const QString &resourcePath) const
{
    QStringList keys;
    if (section.contains(tr("模型")))
    {
        keys.push_back(QStringLiteral("model_results"));
    }
    if (section.contains(tr("点云")) || section.contains(tr("连接点")))
    {
        keys.push_back(QStringLiteral("dense_cloud_results"));
        keys.push_back(QStringLiteral("aerial_triangulation_results"));
    }
    if (section.contains(QStringLiteral("DEM"))) keys.push_back(QStringLiteral("dem_results"));
    if (section.contains(tr("正射"))) keys.push_back(QStringLiteral("ortho_results"));
    if (keys.isEmpty())
    {
        keys = {QStringLiteral("model_results"),
                QStringLiteral("dense_cloud_results"),
                QStringLiteral("dem_results"),
                QStringLiteral("ortho_results")};
    }

    for (const QString &key : keys)
    {
        const QJsonArray records = meta.value(key).toArray();
        for (qsizetype index = records.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = records.at(index).toObject();
            if (samePathText(recordPath(record), resourcePath))
            {
                return record;
            }
        }
    }
    return {};
}

QVector<SelectionPropertiesWidget::PropertyRow> SelectionPropertiesWidget::modelPropertyRows(
    const QJsonObject &meta,
    const QJsonObject &record,
    const QString &resourcePath) const
{
    const QLocale locale;
    const QJsonObject reconstruction = record.value(
        QStringLiteral("reconstruction_parameters")).toObject();
    const QJsonObject depthSnapshot = record.value(
        QStringLiteral("depth_generation_parameters")).toObject();

    QVector<PropertyRow> rows;
    rows.push_back({tr("模型"), {}, true});
    const int faceCount = record.value(QStringLiteral("face_count")).toInt(-1);
    const int vertexCount = record.value(QStringLiteral("vertex_count")).toInt(-1);
    rows.push_back({tr("面"), faceCount >= 0 ? locale.toString(faceCount) : tr("不可用")});
    rows.push_back({tr("顶点"), vertexCount >= 0 ? locale.toString(vertexCount) : tr("不可用")});

    const bool hasVertexColorField = record.contains(QStringLiteral("has_vertex_colors"));
    const bool hasVertexColors = record.value(QStringLiteral("has_vertex_colors")).toBool(
        record.value(QStringLiteral("calculateVertexColors")).toBool(false) ||
        record.contains(QStringLiteral("reliably_colored_vertex_count")));
    rows.push_back({tr("顶点颜色"),
                    hasVertexColors
                        ? record.value(QStringLiteral("vertex_color_format"))
                              .toString(QStringLiteral("3波段, uint8"))
                        : (hasVertexColorField ? tr("无") : tr("不可用"))});
    rows.push_back({tr("格式"),
                    record.value(QStringLiteral("final_model_format"))
                        .toString(QFileInfo(resourcePath).suffix().toUpper())});

    const QString depthSource = record.value(QStringLiteral("depth_map_source_path"))
        .toString(record.value(QStringLiteral("source_path")).toString());
    QString depthQuality = depthSnapshot.value(QStringLiteral("quality_profile")).toString();
    QString depthFilter = depthSnapshot.value(QStringLiteral("filter_mode")).toString();
    int maximumNeighbors = depthSnapshot.value(QStringLiteral("maximum_neighbor_count")).toInt(-1);
    double depthElapsedMs = depthSnapshot.value(QStringLiteral("processing_elapsed_ms")).toDouble(-1.0);
    qint64 depthArtifactBytes = static_cast<qint64>(
        depthSnapshot.value(QStringLiteral("artifact_bytes")).toDouble(-1.0));
    int depthFrameCount = depthSnapshot.value(QStringLiteral("frame_count")).toInt(0);

    if (record.value(QStringLiteral("source_data")).toString() == QStringLiteral("depth_maps") &&
        depthSnapshot.isEmpty() &&
        !depthSource.isEmpty())
    {
        depthElapsedMs = 0.0;
        depthArtifactBytes = 0;
        for (const QJsonValue &value : meta.value(QStringLiteral("depth_map_results")).toArray())
        {
            const QJsonObject depthRecord = value.toObject();
            const QString depthPath = depthRecord.value(QStringLiteral("depth_png")).toString();
            const QString outputDir = depthRecord.value(QStringLiteral("mvs_output_dir")).toString();
            if (!depthSource.isEmpty() && !samePathText(outputDir, depthSource) &&
                !pathBelongsToDirectory(depthPath, depthSource))
            {
                continue;
            }
            ++depthFrameCount;
            if (depthQuality.isEmpty())
            {
                depthQuality = depthRecord.value(QStringLiteral("quality_profile")).toString();
            }
            if (depthFilter.isEmpty())
            {
                depthFilter = depthRecord.value(QStringLiteral("filter_mode"))
                    .toString(depthRecord.value(QStringLiteral("depth_filter_mode")).toString());
            }
            maximumNeighbors = std::max(maximumNeighbors,
                depthRecord.value(QStringLiteral("requested_source_view_count"))
                    .toInt(depthRecord.value(QStringLiteral("source_view_count")).toInt(-1)));
            depthElapsedMs += depthRecord.value(QStringLiteral("elapsed_ms")).toDouble(0.0);
            for (const QString &pathKey : {QStringLiteral("depth_png"),
                                           QStringLiteral("raw_depth_path"),
                                           QStringLiteral("raw_confidence_path"),
                                           QStringLiteral("valid_mask_path")})
            {
                const QFileInfo artifactInfo(depthRecord.value(pathKey).toString());
                if (artifactInfo.exists()) depthArtifactBytes += artifactInfo.size();
            }
        }
        if (depthFrameCount == 0)
        {
            depthElapsedMs = -1.0;
            depthArtifactBytes = -1;
        }
    }

    if (record.value(QStringLiteral("source_data")).toString() == QStringLiteral("depth_maps"))
    {
        rows.push_back({tr("深度图生成参数"), {}, true});
        rows.push_back({tr("质量"), qualityText(depthQuality)});
        rows.push_back({tr("筛选模式"), filterText(depthFilter)});
        rows.push_back({tr("最大邻域数量"),
                        maximumNeighbors >= 0 ? locale.toString(maximumNeighbors) : tr("不可用")});
        rows.push_back({tr("深度图数量"),
                        depthFrameCount > 0 ? locale.toString(depthFrameCount) : tr("不可用")});
        rows.push_back({tr("处理时间"), elapsedText(depthElapsedMs)});
        rows.push_back({tr("深度产物大小"),
                        depthArtifactBytes >= 0 ? fileSizeText(depthArtifactBytes) : tr("不可用")});
    }

    rows.push_back({tr("重建参数"), {}, true});
    rows.push_back({tr("表面类型"), surfaceTypeText(
                        reconstruction.value(QStringLiteral("surface_type"))
                            .toString(record.value(QStringLiteral("surface_type")).toString()))});
    rows.push_back({tr("源数据"), sourceDataText(record.value(QStringLiteral("source_data")).toString())});
    rows.push_back({tr("插值"), interpolationText(
                        reconstruction.value(QStringLiteral("interpolation"))
                            .toString(record.value(QStringLiteral("configured_interpolation")).toString()))});
    const QJsonValue strictMasks = reconstruction.contains(QStringLiteral("strict_volumetric_masks"))
        ? reconstruction.value(QStringLiteral("strict_volumetric_masks"))
        : record.value(QStringLiteral("strictVolumetricMasks"));
    rows.push_back({tr("严格的体积掩模"),
                    strictMasks.isBool() ? (strictMasks.toBool() ? tr("是") : tr("否")) : tr("不可用")});
    rows.push_back({tr("重建算法"),
                    record.value(QStringLiteral("actual_mesh_algorithm"))
                        .toString(record.value(QStringLiteral("mesh_algorithm")).toString(tr("不可用")))});
    const int resolution = record.value(QStringLiteral("configured_tsdf_resolution")).toInt(-1);
    if (resolution > 0)
    {
        rows.push_back({tr("TSDF 分辨率"), locale.toString(resolution)});
    }
    const int targetFaces = record.value(QStringLiteral("configured_target_faces")).toInt(
        reconstruction.value(QStringLiteral("target_faces")).toInt(-1));
    if (targetFaces >= 0)
    {
        rows.push_back({tr("目标面数"), locale.toString(targetFaces)});
    }
    rows.push_back({tr("处理时间"), elapsedText(
                        reconstruction.value(QStringLiteral("processing_elapsed_ms"))
                            .toDouble(record.value(QStringLiteral("processing_elapsed_ms")).toDouble(-1.0)))});
    const qint64 requiredBytes = static_cast<qint64>(record.value(
        QStringLiteral("tsdf_required_bytes")).toDouble(-1.0));
    rows.push_back({tr("内存用量（估算）"),
                    requiredBytes >= 0 ? fileSizeText(requiredBytes) : tr("不可用")});

    rows.push_back({tr("文件信息"), {}, true});
    rows.push_back({tr("创建日期"), creationTimeText(record.value(QStringLiteral("created_at")).toString())});
    QString softwareVersion = record.value(QStringLiteral("software_version")).toString();
    if (softwareVersion.isEmpty()) softwareVersion = QCoreApplication::applicationVersion();
    rows.push_back({tr("软件版本"), softwareVersion.isEmpty() ? tr("不可用") : softwareVersion});
    rows.push_back({tr("文件大小"),
                    QFileInfo(resourcePath).exists() ? fileSizeText(QFileInfo(resourcePath).size()) : tr("不可用")});
    rows.push_back({tr("路径"), resourcePath});
    return rows;
}

QString SelectionPropertiesWidget::imageAlignedText(const QJsonObject &entry) const
{
    if (entry.isEmpty())
    {
        return tr("未知");
    }
    const QJsonObject camera = entry.value(QStringLiteral("camera")).toObject();
    const bool hasCenter = entry.contains(QStringLiteral("center"))
        || entry.contains(QStringLiteral("camera_center"))
        || camera.contains(QStringLiteral("C"));
    const bool aligned = entry.value(QStringLiteral("aligned")).toBool(hasCenter);
    return aligned ? tr("已定向") : tr("未定向");
}

QString SelectionPropertiesWidget::cameraCenterText(const QJsonObject &entry) const
{
    QJsonArray center = entry.value(QStringLiteral("center")).toArray();
    if (center.isEmpty())
    {
        center = entry.value(QStringLiteral("camera_center")).toArray();
    }
    if (center.isEmpty())
    {
        const QJsonObject camera = entry.value(QStringLiteral("camera")).toObject();
        center = camera.value(QStringLiteral("C")).toArray();
    }
    if (center.size() < 3)
    {
        return {};
    }
    return QStringLiteral("%1, %2, %3")
        .arg(center.at(0).toDouble(), 0, 'f', 3)
        .arg(center.at(1).toDouble(), 0, 'f', 3)
        .arg(center.at(2).toDouble(), 0, 'f', 3);
}

QString SelectionPropertiesWidget::intrinsicsText(const QJsonObject &entry) const
{
    const QJsonObject intrinsics = entry.value(QStringLiteral("intrinsics")).toObject();
    if (!intrinsics.isEmpty())
    {
        return QStringLiteral("fx=%1, fy=%2, cx=%3, cy=%4")
            .arg(intrinsics.value(QStringLiteral("fx")).toDouble(), 0, 'f', 2)
            .arg(intrinsics.value(QStringLiteral("fy")).toDouble(), 0, 'f', 2)
            .arg(intrinsics.value(QStringLiteral("cx")).toDouble(), 0, 'f', 2)
            .arg(intrinsics.value(QStringLiteral("cy")).toDouble(), 0, 'f', 2);
    }

    const QJsonObject camera = entry.value(QStringLiteral("camera")).toObject();
    if (camera.isEmpty())
    {
        return {};
    }
    return QStringLiteral("fu=%1, fv=%2, cu=%3, cv=%4")
        .arg(camera.value(QStringLiteral("fu")).toDouble(), 0, 'f', 2)
        .arg(camera.value(QStringLiteral("fv")).toDouble(), 0, 'f', 2)
        .arg(camera.value(QStringLiteral("cu")).toDouble(), 0, 'f', 2)
        .arg(camera.value(QStringLiteral("cv")).toDouble(), 0, 'f', 2);
}

QString SelectionPropertiesWidget::fileSizeText(qint64 bytes)
{
    if (bytes < 1024)
    {
        return QStringLiteral("%1 B").arg(bytes);
    }
    if (bytes < 1024 * 1024)
    {
        return QStringLiteral("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    }
    return QStringLiteral("%1 MB").arg(bytes / 1024.0 / 1024.0, 0, 'f', 1);
}

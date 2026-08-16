#include "DataTreeWidget.h"

#include "ui_DataTreeWidget.h"
#include "ProjectWorkflowUtils.h"
#include "WorkspaceSectionIcons.h"
#include "project/ProjectIO.h"

#include <QTreeView>
#include <QStandardItemModel>
#include <QHeaderView>
#include <QMenu>
#include <QAction>
#include <QJsonObject>
#include <QJsonArray>
#include <QFileInfo>
#include <QDir>
#include <QItemSelectionModel>
#include <QLocale>
#include <QMessageBox>
#include <QImageReader>
#include <QSet>
#include <QVector>
#include <QFont>

#include <initializer_list>

// DataTreeWidget 的实现：
// - 只消费 ProjectData/ProjectManager 提供的内存元数据快照
// - 将每个资源作为一行显示，列为：名称 / 路径 / 存储方式（internal/reference）
// - 右键菜单提供打开 / 在文件管理器中显示 / 打包 / 移除等操作（只发出信号，由上层处理）
#include "DataTreeResourceUtils.h"

using namespace xjw::gui::widgets::data_tree;

void DataTreeWidget::populateFromMeta(const QJsonObject &meta)
{
    QJsonObject normalized = normalizeMeta(meta);
    if (normalized.isEmpty()) return;

    QJsonArray images = normalized.value("images").toArray();
    QJsonArray modelResults = normalized.value("model_results").toArray();
    QJsonArray denseResults = normalized.value("dense_cloud_results").toArray();
    QJsonArray depthResults = normalized.value("depth_map_results").toArray();
    QJsonArray atResults = normalized.value("aerial_triangulation_results").toArray();
    QJsonArray demResults = normalized.value("dem_results").toArray();
    QJsonArray orthoResults = normalized.value("ortho_results").toArray();
    QJsonArray obsNetResults = normalized.value("observation_network_results").toArray();
    QJsonArray reportResults = normalized.value("report_results").toArray();
    QJsonArray referenceDatasets = normalized.value("reference_datasets").toArray();

    QJsonObject currentTiePointRecord;
    QString currentTiePointPath;
    int totalSparsePoints = -1;
    QSet<QString> alignedImageKeys;
    for (int index = atResults.size() - 1; index >= 0; --index)
    {
        const QJsonObject candidate = atResults.at(index).toObject();
        const QString sparsePath = candidate.value(QStringLiteral("files"))
                                       .toObject()
                                       .value(QStringLiteral("sparse_cloud_xyz"))
                                       .toString()
                                       .trimmed();
        if (!sparsePath.isEmpty())
        {
            currentTiePointRecord = candidate;
            currentTiePointPath = sparsePath;
            break;
        }
    }

    if (!currentTiePointRecord.isEmpty())
    {
        totalSparsePoints = currentTiePointRecord.value(QStringLiteral("sparse_point_count")).toInt(-1);
        if (totalSparsePoints < 0)
        {
            totalSparsePoints = currentTiePointRecord.value(QStringLiteral("point_count")).toInt(-1);
        }
        if (totalSparsePoints < 0)
        {
            totalSparsePoints = currentTiePointRecord.value(QStringLiteral("quality"))
                                    .toObject()
                                    .value(QStringLiteral("point_count"))
                                    .toInt(-1);
        }
        for (const QJsonValue &imageValue :
             currentTiePointRecord.value(QStringLiteral("selected_images")).toArray())
        {
            const QString key = imagePathKey(imagePathFromValue(imageValue));
            if (!key.isEmpty())
            {
                alignedImageKeys.insert(key);
            }
    }
}
    int alignedImageCount = 0;
    int maskCount = 0;
    for (const QJsonValue &v : images)
    {
        if (imageIsAligned(v, alignedImageKeys))
        {
            ++alignedImageCount;
        }
        if (v.isObject()
            && !v.toObject().value(QStringLiteral("mask_path")).toString().trimmed().isEmpty())
        {
            ++maskCount;
        }
    }

    QStringList depthPaths;
    QSet<QString> depthPathKeys;
    QSet<QString> depthFrameKeys;
    QJsonObject latestDepthRecord;
    for (int index = depthResults.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = depthResults.at(index).toObject();
        const QString primaryPath = depthRecordPrimaryPath(record);
        if (primaryPath.isEmpty())
        {
            continue;
        }

        const QString pathKey = QDir::cleanPath(QDir::fromNativeSeparators(primaryPath)).toCaseFolded();
        if (!depthPathKeys.contains(pathKey))
        {
            depthPathKeys.insert(pathKey);
            depthPaths.append(primaryPath);
        }

        QString frameKey = record.value(QStringLiteral("ref_image")).toString().trimmed();
        if (frameKey.isEmpty())
        {
            frameKey = primaryPath;
        }
        frameKey = QDir::cleanPath(QDir::fromNativeSeparators(frameKey)).toCaseFolded();
        if (depthFrameKeys.contains(frameKey))
        {
            continue;
        }

        depthFrameKeys.insert(frameKey);
        if (latestDepthRecord.isEmpty())
        {
            latestDepthRecord = record;
        }
    }

    const int denseCount = countObjectsWithPath(denseResults, {"dense_cloud_xyz", "source_sparse_cloud"});
    const int modelCount = displayableMeshResultCount(modelResults) + _transientModels.size();
    const int demCount = countObjectsWithPath(demResults, {"dem_tif", "dem_path"});
    const int orthoCount = countObjectsWithPath(orthoResults, {"output_path"});
    const int reportCount = countObjectsWithPath(reportResults, {"path", "json_path", "report_path"});
    const int referenceCount = countObjectsWithPath(referenceDatasets, {"path", "file_path", "dem_path",
                                                                        "lidar_path", "cloud_path"});

    // ── 保存展开状态 ──────────────────────────────────────────────────────
    QSet<int> expandedSections;
    QSet<QString> expandedChunks;
    const auto captureSectionExpansion =
        [this, &expandedSections](QStandardItem *item)
    {
        if (!item)
        {
            return;
        }
        const QModelIndex index = _model->indexFromItem(item);
        const QVariant sectionValue =
            item->data(WorkspaceSectionRole);
        if (_view->isExpanded(index) && sectionValue.isValid())
        {
            expandedSections.insert(sectionValue.toInt());
        }
    };
    const auto captureChunkExpansion =
        [this, &expandedChunks, &captureSectionExpansion](
            QStandardItem *chunk)
    {
        if (!chunk
            || chunk->data(ChunkIdRole).toString().isEmpty())
        {
            return;
        }
        if (_view->isExpanded(_model->indexFromItem(chunk)))
        {
            expandedChunks.insert(
                chunk->data(ChunkIdRole).toString());
        }
        for (int row = 0; row < chunk->rowCount(); ++row)
        {
            captureSectionExpansion(chunk->child(row, 0));
        }
    };
    for (int i = 0; i < _model->rowCount(); ++i)
    {
        QStandardItem *topLevel = _model->item(i, 0);
        if (topLevel
            && topLevel->data(WorkspaceRootRole).toBool())
        {
            for (int row = 0; row < topLevel->rowCount(); ++row)
            {
                captureChunkExpansion(topLevel->child(row, 0));
            }
            continue;
        }
        if (!topLevel->data(ChunkIdRole).toString().isEmpty())
        {
            captureChunkExpansion(topLevel);
        }
        else
        {
            captureSectionExpansion(topLevel);
        }
    }

    _model->removeRows(0, _model->rowCount());
    _workspaceRoot = nullptr;
    _activeChunkRoot = nullptr;

    int totalImageCount = 0;
    for (const QJsonValue &value : _chunks)
    {
        const QJsonObject chunk = value.toObject();
        const QString chunkId =
            chunk.value(QStringLiteral("id")).toString();
        const int imageCount = chunkId == _activeChunkId
            ? images.size()
            : chunk.value(QStringLiteral("image_count")).toInt(0);
        totalImageCount += qMax(0, imageCount);
    }

    if (!_chunks.isEmpty())
    {
        auto *nameItem = new QStandardItem(
            workspaceSummaryLabel(
                _chunks.size(), totalImageCount));
        auto *pathItem = new QStandardItem(QString());
        auto *storageItem =
            new QStandardItem(QStringLiteral("workspace"));
        nameItem->setFlags(
            nameItem->flags() & ~Qt::ItemIsEditable);
        pathItem->setFlags(
            pathItem->flags() & ~Qt::ItemIsEditable);
        storageItem->setFlags(
            storageItem->flags() & ~Qt::ItemIsEditable);
        nameItem->setIcon(
            xjw::gui::widgets::workspaceRootIcon());
        nameItem->setData(true, WorkspaceRootRole);
        nameItem->setToolTip(QStringLiteral(
            "工程中的 Chunk 与影像汇总"));
        _model->appendRow(
            {nameItem, pathItem, storageItem});
        _workspaceRoot = nameItem;
    }

    for (const QJsonValue &value : _chunks)
    {
        const QJsonObject chunk = value.toObject();
        const QString chunkId =
            chunk.value(QStringLiteral("id")).toString();
        if (chunkId.isEmpty())
        {
            continue;
        }
        QString name = chunk.value(
            QStringLiteral("name")).toString().trimmed();
        const QJsonValue directoryValue =
            chunk.value(QStringLiteral("directory"));
        const int directory = directoryValue.isDouble()
            ? directoryValue.toInt()
            : directoryValue.toString().toInt();
        if (name.isEmpty())
        {
            name = QStringLiteral("Chunk %1").arg(directory);
        }
        const bool active = chunkId == _activeChunkId;
        const int imageCount = active
            ? images.size()
            : chunk.value(QStringLiteral("image_count")).toInt(-1);
        const int tiePointCount = active
            ? totalSparsePoints
            : chunk.value(
                  QStringLiteral("tie_point_count")).toInt(-1);
        auto *nameItem = new QStandardItem(
            chunkSummaryLabel(name, imageCount, tiePointCount));
        auto *pathItem =
            new QStandardItem(QString::number(directory));
        auto *storageItem =
            new QStandardItem(QStringLiteral("chunk"));
        nameItem->setFlags(
            nameItem->flags() & ~Qt::ItemIsEditable);
        pathItem->setFlags(
            pathItem->flags() & ~Qt::ItemIsEditable);
        storageItem->setFlags(
            storageItem->flags() & ~Qt::ItemIsEditable);
        nameItem->setIcon(
            xjw::gui::widgets::workspaceChunkIcon());
        nameItem->setData(chunkId, ChunkIdRole);
        nameItem->setData(directory, ChunkDirectoryRole);
        QFont font = nameItem->font();
        font.setBold(active);
        nameItem->setFont(font);
        nameItem->setToolTip(
            active
                ? QStringLiteral("当前 Chunk")
                : QStringLiteral("双击切换到此 Chunk"));
        if (_workspaceRoot)
        {
            _workspaceRoot->appendRow(
                {nameItem, pathItem, storageItem});
        }
        else
        {
            _model->appendRow(
                {nameItem, pathItem, storageItem});
        }
        if (active)
        {
            _activeChunkRoot = nameItem;
        }
    }

    using xjw::gui::widgets::WorkspaceSection;
    auto *photos = images.isEmpty()
        ? nullptr
        : createSection(QStringLiteral("图像 (%1/%2 对齐)").arg(alignedImageCount).arg(images.size()),
                        WorkspaceSection::Photos);
    if (maskCount > 0)
    {
        createSection(QStringLiteral("掩膜"), maskCount, WorkspaceSection::Masks);
    }
    auto *obsNet = obsNetResults.isEmpty()
        ? nullptr
        : createSection(QStringLiteral("观测网络"), obsNetResults.size(), WorkspaceSection::ObservationNetwork);
    if (!currentTiePointPath.isEmpty())
    {
        QString tiePointLabel = QStringLiteral("连接点");
        if (totalSparsePoints >= 0)
        {
            tiePointLabel = QStringLiteral("连接点 (%1个点)")
                                .arg(QLocale().toString(totalSparsePoints));
        }
        appendTopLevelResource(tiePointLabel,
                               WorkspaceSection::TiePoints,
                               QStringLiteral("连接点"),
                               currentTiePointPath,
                               QStringLiteral("generated"));
    }
    if (!depthPaths.isEmpty())
    {
        QString qualityProfile = latestDepthRecord.value(QStringLiteral("quality_profile")).toString();
        if (qualityProfile.isEmpty())
        {
            qualityProfile = latestDepthRecord.value(QStringLiteral("qualityProfile")).toString();
        }
        QStringList summaryParts{QLocale().toString(depthFrameKeys.size())};
        const QString qualityLabel = depthQualityLabel(qualityProfile);
        const QString filterLabel = depthFilterLabel(
            latestDepthRecord.value(QStringLiteral("filter_mode")).toString());
        if (!qualityLabel.isEmpty()) summaryParts.append(qualityLabel);
        if (!filterLabel.isEmpty()) summaryParts.append(filterLabel);
        appendTopLevelAggregate(QStringLiteral("深度图（%1）").arg(summaryParts.join(QStringLiteral("，"))),
                                WorkspaceSection::DepthMaps,
                                QStringLiteral("深度图"),
                                depthPaths);
    }
    auto *denseCloud = denseCount <= 0
        ? nullptr
        : createSection(QStringLiteral("稠密点云"), denseCount, WorkspaceSection::DenseCloud);
    auto *model3d = modelCount <= 0
        ? nullptr
        : createSection(QStringLiteral("3D模型"), modelCount, WorkspaceSection::Model3D);
    auto *dem = demCount <= 0
        ? nullptr
        : createSection(QStringLiteral("DEM"), demCount, WorkspaceSection::Dem);
    auto *ortho = orthoCount <= 0
        ? nullptr
        : createSection(QStringLiteral("正射影像"), orthoCount, WorkspaceSection::Orthomosaic);
    auto *references = referenceCount <= 0
        ? nullptr
        : createSection(QStringLiteral("参考数据"), referenceCount, WorkspaceSection::ReferenceData);
    auto *reports = reportCount <= 0
        ? nullptr
        : createSection(QStringLiteral("报告"), reportCount, WorkspaceSection::Reports);

    // ── 照片（每张后面标注是否已定向）────────────────────────────────────
    for (const QJsonValue &v : images) {
        QString path;
        QString storage;
        if (v.isObject()) {
            QJsonObject o = v.toObject();
            path    = imagePathFromValue(v);
            storage = o.value("storage").toString();
        } else if (v.isString()) {
            path    = imagePathFromValue(v);
            storage = QStringLiteral("internal");
        }
        QFileInfo fi(path);
        QString name = fi.fileName();
        if (name.isEmpty()) name = path;
        const bool aligned = imageIsAligned(v, alignedImageKeys);
        QStandardItem *imageItem =
            appendItemRow(photos, name, path, storage);
        if (imageItem)
        {
            imageItem->setIcon(
                xjw::gui::widgets::workspaceImageIcon());
            imageItem->setToolTip(
                aligned
                    ? QStringLiteral("已对齐")
                    : QStringLiteral("未对齐"));
        }
    }

    // ── 观测网络结果 ──────────────────────────────────────────────────────
    for (int i = 0; i < obsNetResults.size(); ++i) {
        const QJsonValue &v = obsNetResults.at(i);
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        const QString algo  = obj.value(QStringLiteral("algorithm")).toString();
        const int nodes     = obj.value(QStringLiteral("node_count")).toInt(0);
        const int edges     = obj.value(QStringLiteral("edge_count")).toInt(0);
        const QString ts    = obj.value(QStringLiteral("timestamp")).toString();
        QString name = QStringLiteral("%1  [N:%2 E:%3]").arg(algo).arg(nodes).arg(edges);
        if (!ts.isEmpty()) name += QStringLiteral("  %1").arg(ts.left(10));
        appendItemRow(obsNet, name, QString::number(i), QStringLiteral("generated"));
    }

    // ── 3D 模型 ───────────────────────────────────────────────────────────
    for (const QJsonValue &v : modelResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        if (!isDisplayableMeshResult(obj)) continue;
        const QString modelObjPath = obj.value(
            QStringLiteral("model_obj")).toString().trimmed();
        const bool texturedRecord = obj.value(
            QStringLiteral("textured")).toBool(false);
        for (const QString &modelPath : displayableMeshAssetPaths(obj))
        {
            QString name = QFileInfo(modelPath).fileName();
            if (name.isEmpty()) name = modelPath;
            const int vtx = obj.value(QStringLiteral("vertex_count")).toInt(-1);
            const int face = obj.value(QStringLiteral("face_count")).toInt(-1);
            if (vtx >= 0 && face >= 0)
                name = QStringLiteral("%1  [V:%2 F:%3]").arg(name).arg(vtx).arg(face);
            const QString format = QFileInfo(modelPath).suffix().toUpper();
            if (!format.isEmpty()) {
                name += QStringLiteral("  [%1]").arg(format);
            }
            if (texturedRecord && modelPath.compare(
                    modelObjPath, Qt::CaseInsensitive) == 0) {
                name += QStringLiteral("  [纹理]");
            }
            appendItemRow(model3d, name, modelPath, QStringLiteral("generated"));
        }
    }

    for (const QString &modelPath : _transientModels) {
        if (modelPath.trimmed().isEmpty()) continue;
        QString name = QFileInfo(modelPath).fileName();
        if (name.isEmpty()) name = modelPath;
        name += QStringLiteral("  [临时]");
        appendItemRow(model3d, name, modelPath, QStringLiteral("temporary"));
    }

    // ── 稠密点云 ──────────────────────────────────────────────────────────
    for (const QJsonValue &v : denseResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        QString path = obj.value(QStringLiteral("dense_cloud_xyz")).toString();
        if (path.isEmpty()) path = obj.value(QStringLiteral("source_sparse_cloud")).toString();
        if (path.isEmpty()) continue;
        QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        const int pts = obj.value(QStringLiteral("point_count")).toInt(-1);
        if (pts >= 0)
            name = QStringLiteral("%1  [点: %2]").arg(name).arg(pts);
        appendItemRow(denseCloud, name, path, QStringLiteral("generated"));
    }

    // ── DEM ───────────────────────────────────────────────────────────────
    for (const QJsonValue &v : demResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        QString path = obj.value(QStringLiteral("dem_tif")).toString();
        if (path.isEmpty()) path = obj.value(QStringLiteral("dem_path")).toString();
        if (path.isEmpty()) continue;
        QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        const QString typ = obj.value(QStringLiteral("dem_type")).toString();
        if (!typ.isEmpty()) name = QStringLiteral("%1  [%2]").arg(name, typ);
        appendItemRow(dem, name, path, QStringLiteral("generated"));
        QString previewPath = obj.value(QStringLiteral("depth_preview_png")).toString();
        if (previewPath.isEmpty()) previewPath = obj.value(QStringLiteral("preview_path")).toString();
        if (previewPath.isEmpty()) previewPath = obj.value(QStringLiteral("depth_png")).toString();
        if (!previewPath.isEmpty())
        {
            QString previewName = QFileInfo(previewPath).fileName().isEmpty()
                ? previewPath
                : QFileInfo(previewPath).fileName();
            previewName = QStringLiteral("预览 %1").arg(previewName);
            appendItemRow(dem, previewName, previewPath, QStringLiteral("generated"));
        }
        const auto appendQualityRaster = [&](const QString &key, const QString &label)
        {
            const QString qualityPath = obj.value(key).toString();
            if (qualityPath.isEmpty())
            {
                return;
            }
            QString qualityName = QFileInfo(qualityPath).fileName().isEmpty()
                ? qualityPath
                : QFileInfo(qualityPath).fileName();
            qualityName = QStringLiteral("%1 %2").arg(label, qualityName);
            appendItemRow(dem, qualityName, qualityPath, QStringLiteral("generated"));
        };
        appendQualityRaster(QStringLiteral("error_path"), QStringLiteral("误差"));
        appendQualityRaster(QStringLiteral("count_path"), QStringLiteral("点数"));
        appendQualityRaster(QStringLiteral("confidence_path"), QStringLiteral("置信度"));
        appendQualityRaster(QStringLiteral("coverage_path"), QStringLiteral("覆盖率"));
    }

    // ── 正射影像 ──────────────────────────────────────────────────────────
    for (const QJsonValue &v : orthoResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        const QString path = obj.value(QStringLiteral("output_path")).toString();
        if (path.isEmpty()) continue;
        QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        appendItemRow(ortho, name, path, QStringLiteral("generated"));
    }

    // ── 报告 ─────────────────────────────────────────────────────────────
    for (const QJsonValue &v : reportResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        QString path = obj.value(QStringLiteral("path")).toString();
        if (path.isEmpty()) path = obj.value(QStringLiteral("json_path")).toString();
        if (path.isEmpty()) path = obj.value(QStringLiteral("report_path")).toString();
        if (path.isEmpty()) continue;
        QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        const QString type = obj.value(QStringLiteral("type")).toString();
        if (!type.isEmpty())
        {
            name += QStringLiteral("  [%1]").arg(type);
        }
        appendItemRow(reports, name, path, QStringLiteral("generated"));
    }

    // ── 参考数据（外部 DEM/LiDAR/点云，不默认复制进项目）────────────────
    for (const QJsonValue &v : referenceDatasets) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        const QString path = referenceDatasetPath(obj);
        if (path.isEmpty()) continue;

        QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        const QString typeLabel = referenceDatasetTypeLabel(obj.value(QStringLiteral("type")).toString());
        if (!typeLabel.isEmpty())
        {
            name += QStringLiteral("  [%1]").arg(typeLabel);
        }
        const QString roleLabel = referenceDatasetRoleLabel(obj.value(QStringLiteral("role")).toString());
        if (!roleLabel.isEmpty())
        {
            name += QStringLiteral(" [%1]").arg(roleLabel);
        }

        const QString storage = obj.value(QStringLiteral("storage")).toString(QStringLiteral("reference"));
        appendItemRow(references, name, path, storage);
    }

    const int sectionCount = _activeChunkRoot
        ? _activeChunkRoot->rowCount()
        : _model->rowCount();
    for (int i = 0; i < sectionCount; ++i)
    {
        sortSectionChildrenByFileName(
            _activeChunkRoot
                ? _activeChunkRoot->child(i, 0)
                : _model->item(i, 0));
    }

    // ── 恢复展开状态 ──────────────────────────────────────────────────────
    // 仅恢复用户先前显式展开过的分组，不在数据更新时自动展开任何默认分组。
    // 之前这里对第一个分组（通常是“照片”）做了强制展开，导致新增深度图/点云等
    // 元数据写回后，工作区会突然自动展开“照片”树，打断用户当前浏览位置。
    for (int i = 0; i < sectionCount; ++i) {
        QStandardItem *item = _activeChunkRoot
            ? _activeChunkRoot->child(i, 0)
            : _model->item(i, 0);
        if (item) {
            const QVariant sectionValue = item->data(WorkspaceSectionRole);
            if (sectionValue.isValid() && expandedSections.contains(sectionValue.toInt())) {
                QModelIndex idx = _model->indexFromItem(item);
                _view->expand(idx);
            }
        }
    }
    if (_workspaceRoot)
    {
        _view->expand(
            _model->indexFromItem(_workspaceRoot));
    }
    if (_activeChunkRoot)
    {
        const QModelIndex activeIndex =
            _model->indexFromItem(_activeChunkRoot);
        if (expandedChunks.isEmpty()
            || expandedChunks.contains(_activeChunkId))
        {
            _view->expand(activeIndex);
        }
    }
}

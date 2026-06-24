#include "DataTreeWidget.h"

#include "ui_DataTreeWidget.h"
#include "ProjectWorkflowUtils.h"

#include <QTreeView>
#include <QStandardItemModel>
#include <QHeaderView>
#include <QMenu>
#include <QAction>
#include <QJsonObject>
#include <QJsonArray>
#include <QFileInfo>
#include <QDir>
#include <QMessageBox>
#include <QImageReader>
#include <QStyle>
#include <QVector>

// DataTreeWidget 的实现：
// - 只消费 ProjectData/ProjectManager 提供的内存元数据快照
// - 将每个资源作为一行显示，列为：名称 / 路径 / 存储方式（internal/reference）
// - 右键菜单提供打开 / 在文件管理器中显示 / 打包 / 移除等操作（只发出信号，由上层处理）

namespace
{

QString depthResultKind(const QJsonObject &record)
{
    const QString explicitKind = record.value(QStringLiteral("result_type")).toString();
    if (!explicitKind.isEmpty())
    {
        return explicitKind;
    }

    if (!record.value(QStringLiteral("raw_depth_path")).toString().isEmpty() ||
        !record.value(QStringLiteral("ref_image")).toString().isEmpty())
    {
        return QStringLiteral("mvs_depth");
    }

    return QStringLiteral("legacy_preview");
}

int mvsDepthResultCount(const QJsonArray &depthResults)
{
    int count = 0;
    for (const QJsonValue &value : depthResults)
    {
        if (!value.isObject())
        {
            continue;
        }
        if (depthResultKind(value.toObject()) == QStringLiteral("mvs_depth"))
        {
            ++count;
        }
    }
    return count;
}

bool isTreeResultKey(const QString &key)
{
    return key == QStringLiteral("ipfind_results")
        || key == QStringLiteral("ipmatch_results")
        || key == QStringLiteral("observation_network_results")
        || key == QStringLiteral("aerial_triangulation_results")
        || key == QStringLiteral("depth_map_results")
        || key == QStringLiteral("dense_cloud_results")
        || key == QStringLiteral("model_results")
        || key == QStringLiteral("dem_results")
        || key == QStringLiteral("ortho_results")
        || key == QStringLiteral("report_results")
        || key == QStringLiteral("reference_datasets");
}

bool hasTreeResultKeys(const QJsonObject &meta)
{
    for (auto it = meta.constBegin(); it != meta.constEnd(); ++it)
    {
        if (isTreeResultKey(it.key()))
        {
            return true;
        }
    }
    return false;
}

int compareNaturalText(QString lhs, QString rhs)
{
    int li = 0;
    int ri = 0;
    const int ln = lhs.size();
    const int rn = rhs.size();

    while (li < ln && ri < rn)
    {
        const QChar lc = lhs.at(li);
        const QChar rc = rhs.at(ri);
        if (lc.isDigit() && rc.isDigit())
        {
            const int lhsStart = li;
            const int rhsStart = ri;
            while (li < ln && lhs.at(li).isDigit()) ++li;
            while (ri < rn && rhs.at(ri).isDigit()) ++ri;

            int lhsSig = lhsStart;
            int rhsSig = rhsStart;
            while (lhsSig + 1 < li && lhs.at(lhsSig) == QLatin1Char('0')) ++lhsSig;
            while (rhsSig + 1 < ri && rhs.at(rhsSig) == QLatin1Char('0')) ++rhsSig;

            const int lhsDigits = li - lhsSig;
            const int rhsDigits = ri - rhsSig;
            if (lhsDigits != rhsDigits)
            {
                return lhsDigits < rhsDigits ? -1 : 1;
            }
            for (int offset = 0; offset < lhsDigits; ++offset)
            {
                const int ld = lhs.at(lhsSig + offset).digitValue();
                const int rd = rhs.at(rhsSig + offset).digitValue();
                if (ld != rd)
                {
                    return ld < rd ? -1 : 1;
                }
            }

            const int lhsRun = li - lhsStart;
            const int rhsRun = ri - rhsStart;
            if (lhsRun != rhsRun)
            {
                return lhsRun < rhsRun ? -1 : 1;
            }
            continue;
        }

        const QChar lf = lc.toCaseFolded();
        const QChar rf = rc.toCaseFolded();
        if (lf != rf)
        {
            return lf.unicode() < rf.unicode() ? -1 : 1;
        }
        ++li;
        ++ri;
    }

    if (li == ln && ri == rn)
    {
        return 0;
    }
    return li == ln ? -1 : 1;
}

QString referenceDatasetPath(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("file_path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("dem_path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("lidar_path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("cloud_path")).toString();
    return path;
}

QString referenceDatasetTypeLabel(QString type)
{
    type = type.trimmed().toLower();
    if (type == QStringLiteral("dem") || type == QStringLiteral("reference_dem"))
    {
        return QStringLiteral("DEM");
    }
    if (type == QStringLiteral("lidar") || type == QStringLiteral("las") || type == QStringLiteral("laz") ||
        type == QStringLiteral("copc") || type == QStringLiteral("reference_lidar"))
    {
        return QStringLiteral("LiDAR");
    }
    if (type == QStringLiteral("point_cloud") || type == QStringLiteral("cloud"))
    {
        return QStringLiteral("点云");
    }
    if (type.isEmpty())
    {
        return QStringLiteral("参考");
    }
    return type;
}

QString referenceDatasetRoleLabel(QString role)
{
    role = role.trimmed().toLower();
    if (role == QStringLiteral("validation") || role == QStringLiteral("quality_check"))
    {
        return QStringLiteral("精度检查");
    }
    if (role == QStringLiteral("ba_prior") || role == QStringLiteral("bundle_adjustment") ||
        role == QStringLiteral("reference_prior"))
    {
        return QStringLiteral("BA约束");
    }
    if (role == QStringLiteral("alignment") || role == QStringLiteral("registration"))
    {
        return QStringLiteral("配准");
    }
    if (role.isEmpty())
    {
        return QString();
    }
    return role;
}

} // namespace

DataTreeWidget::DataTreeWidget(QWidget *parent)
    : QWidget(parent)
{
    Ui::DataTreeWidget ui;
    ui.setupUi(this);

    _view = ui.m_view;
    _model = new QStandardItemModel(this);
    // 左侧列表显示策略：
    // - 只展示“名称”一列，保持列表简洁（用户不希望直接看到路径/存储）。
    // - 但为了右键菜单“属性”弹窗仍能显示完整信息，这里仍保留 path/storage 作为隐藏列。
    _model->setHorizontalHeaderLabels({QObject::tr("名称"), QObject::tr("路径"), QObject::tr("存储")});

    _view->setModel(_model);
    // 隐藏“路径/存储”两列
    _view->setColumnHidden(1, true);
    _view->setColumnHidden(2, true);
    // 禁用双击进入编辑（用户要求：双击不应改名）
    _view->setEditTriggers(QAbstractItemView::NoEditTriggers);
    // 允许使用 Ctrl / Shift 多选
    _view->setSelectionMode(QAbstractItemView::ExtendedSelection);
    connect(_view, &QTreeView::customContextMenuRequested, this, &DataTreeWidget::onContextMenuRequested);

    // 双击或回车激活资源时，通知上层切换中央显示。
    // 单击只负责选择，避免浏览资源树时同步加载大图造成界面卡顿。
    connect(_view, &QTreeView::activated, this, [this](const QModelIndex &idx) {
        if (!idx.isValid()) return;
        // Use the index to access the hidden 'path' column in the same row/parent
        QModelIndex pathIdx = idx.sibling(idx.row(), 1);
        if (!pathIdx.isValid()) return;
        QString path = _model->data(pathIdx).toString();
        QModelIndex nameIdx = idx.sibling(idx.row(), 0);
        QModelIndex parentNameIdx = nameIdx.parent().isValid() ? nameIdx.parent() : QModelIndex();
        QString section;
        if (parentNameIdx.isValid()) {
            section = _model->data(parentNameIdx).toString().section(' ', 0, 0);
        }
        if (!path.trimmed().isEmpty())
        {
            path = resolveResourcePath(path);
            emit resourceActivated(section, path);
            if (section == QStringLiteral("照片")
                || section == QStringLiteral("深度图")
                || section == QStringLiteral("DEM")
                || section == QStringLiteral("正射影像")) {
                emit imageActivated(path);
            }
        }
    });

}

DataTreeWidget::~DataTreeWidget()
{
}

void DataTreeWidget::setProjectPath(const QString &plascanPath)
{
    _currentPlascanPath = plascanPath;
    _lastMeta = QJsonObject();
    _model->removeRows(0, _model->rowCount());
}

void DataTreeWidget::loadFromArchive(const QString &plascanPath)
{
    setProjectPath(plascanPath);
}

void DataTreeWidget::loadFromJson(const QJsonObject &meta)
{
    // 注意：这里不能在 images 为空时就清空列表并返回。
    // 因为 projectOpened/metadataChanged 等信号可能会短暂发送一个空 meta，
    // 若直接清空会覆盖掉当前项目树，造成 UI 闪烁或误清空。
    if (meta.isEmpty()) return;

    // 支持两种结构：根对象的 "images" 或嵌套在 "project_files" 中的 "images"。
    // 如果 meta 明确包含 images（即使为空数组），我们应该以该值为准并更新视图。
    QJsonObject normalized = normalizeMeta(meta);
    bool hasImagesKey = normalized.contains(QStringLiteral("images"));
    QJsonArray images = normalized.value("images").toArray();
    if (!hasImagesKey && meta.value(QStringLiteral("project_files")).isObject())
    {
        QJsonObject pf = meta.value(QStringLiteral("project_files")).toObject();
        if (pf.contains(QStringLiteral("images"))) {
            hasImagesKey = true;
            images = pf.value(QStringLiteral("images")).toArray();
        }
    }

    if (!hasImagesKey)
    {
        const QJsonObject projectFiles = meta.value(QStringLiteral("project_files")).toObject();
        const QJsonObject resultSource = projectFiles.isEmpty() ? meta : projectFiles;
        if (!hasTreeResultKeys(resultSource))
        {
            return;
        }

        normalized = normalizeMeta(_lastMeta);
        if (normalized.isEmpty())
        {
            normalized.insert(QStringLiteral("images"), QJsonArray());
        }
        for (auto it = resultSource.constBegin(); it != resultSource.constEnd(); ++it)
        {
            if (isTreeResultKey(it.key()))
            {
                normalized.insert(it.key(), it.value());
            }
        }
    }

    // 无论 images 是否为空，只要 meta 明确提供了 images，我们就清空并重建模型（允许清空列表）。
    _lastMeta = normalized;
    populateFromMeta(normalized);
}

void DataTreeWidget::addTransientModel(const QString &modelPath)
{
    const QString cleanPath = QDir::cleanPath(modelPath.trimmed());
    if (cleanPath.isEmpty())
    {
        return;
    }

    if (!_transientModels.contains(cleanPath))
    {
        _transientModels.append(cleanPath);
    }

    if (_lastMeta.isEmpty())
    {
        QJsonObject emptyProject;
        emptyProject.insert(QStringLiteral("images"), QJsonArray());
        _lastMeta = emptyProject;
    }
    populateFromMeta(_lastMeta);
}

void DataTreeWidget::clearTransientResources()
{
    if (_transientModels.isEmpty())
    {
        return;
    }

    _transientModels.clear();
    if (!_lastMeta.isEmpty())
    {
        populateFromMeta(_lastMeta);
    }
    else
    {
        _model->removeRows(0, _model->rowCount());
    }
}

QJsonObject DataTreeWidget::normalizeMeta(const QJsonObject &meta) const
{
    QJsonObject normalized;
    if (meta.value(QStringLiteral("project_files")).isObject())
    {
        normalized = meta.value(QStringLiteral("project_files")).toObject();
    }
    else
    {
        normalized = meta;
    }

    for (auto it = meta.constBegin(); it != meta.constEnd(); ++it)
    {
        if (isTreeResultKey(it.key()))
        {
            normalized.insert(it.key(), it.value());
        }
    }
    return normalized;
}

QStandardItem *DataTreeWidget::createSection(const QString &title, int count)
{
    const QString label = QStringLiteral("%1 (%2)").arg(title).arg(count);
    auto *nameItem = new QStandardItem(label);
    nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
    nameItem->setIcon(style()->standardIcon(QStyle::SP_DirIcon));

    auto *pathItem = new QStandardItem(QString());
    auto *storageItem = new QStandardItem(QString());
    pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
    storageItem->setFlags(storageItem->flags() & ~Qt::ItemIsEditable);

    _model->appendRow({nameItem, pathItem, storageItem});
    return nameItem;
}

void DataTreeWidget::appendItemRow(QStandardItem *parent, const QString &name, const QString &path, const QString &storage)
{
    if (!parent) return;
    auto *nameItem = new QStandardItem(name);
    auto *pathItem = new QStandardItem(path);
    auto *storageItem = new QStandardItem(storage);
    nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
    pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
    storageItem->setFlags(storageItem->flags() & ~Qt::ItemIsEditable);
    parent->appendRow({nameItem, pathItem, storageItem});
}

void DataTreeWidget::sortSectionChildrenByFileName(QStandardItem *section)
{
    if (!section || section->rowCount() < 2)
    {
        return;
    }

    struct Row
    {
        QList<QStandardItem *> items;
        QString fileNameKey;
        QString displayName;
        int originalRow = 0;
    };

    QVector<Row> rows;
    rows.reserve(section->rowCount());
    const int rowCount = section->rowCount();
    for (int row = 0; row < rowCount; ++row)
    {
        QList<QStandardItem *> items = section->takeRow(0);
        if (items.isEmpty())
        {
            continue;
        }

        const QString displayName = items.at(0) ? items.at(0)->text() : QString();
        const QString path = items.size() > 1 && items.at(1) ? items.at(1)->text() : QString();
        QString fileNameKey = QFileInfo(path).fileName();
        if (fileNameKey.isEmpty())
        {
            fileNameKey = displayName;
        }
        rows.push_back({items, fileNameKey, displayName, row});
    }

    std::stable_sort(rows.begin(), rows.end(), [](const Row &lhs, const Row &rhs)
    {
        int cmp = compareNaturalText(lhs.fileNameKey, rhs.fileNameKey);
        if (cmp != 0)
        {
            return cmp < 0;
        }

        cmp = compareNaturalText(lhs.displayName, rhs.displayName);
        if (cmp != 0)
        {
            return cmp < 0;
        }

        return lhs.originalRow < rhs.originalRow;
    });

    for (const Row &row : rows)
    {
        section->appendRow(row.items);
    }
}

QString DataTreeWidget::resolveResourcePath(const QString &resourcePath) const
{
    const QString trimmedPath = resourcePath.trimmed();
    if (trimmedPath.isEmpty())
    {
        return QString();
    }

    if (QFileInfo(trimmedPath).isAbsolute() || _currentPlascanPath.trimmed().isEmpty())
    {
        return QDir::cleanPath(trimmedPath);
    }

    const QString projectRoot = QFileInfo(_currentPlascanPath).absolutePath();
    return QDir::cleanPath(QDir(projectRoot).filePath(trimmedPath));
}

void DataTreeWidget::populateFromMeta(const QJsonObject &meta)
{
    QJsonObject normalized = normalizeMeta(meta);
    if (normalized.isEmpty()) return;

    QJsonArray images = normalized.value("images").toArray();
    QJsonArray ipfindResults = normalized.value("ipfind_results").toArray();
    QJsonArray ipmatchResults = normalized.value("ipmatch_results").toArray();
    QJsonArray modelResults = normalized.value("model_results").toArray();
    QJsonArray depthResults = normalized.value("depth_map_results").toArray();
    QJsonArray denseResults = normalized.value("dense_cloud_results").toArray();
    QJsonArray atResults = normalized.value("aerial_triangulation_results").toArray();
    QJsonArray demResults = normalized.value("dem_results").toArray();
    QJsonArray orthoResults = normalized.value("ortho_results").toArray();
    QJsonArray obsNetResults = normalized.value("observation_network_results").toArray();
    QJsonArray reportResults = normalized.value("report_results").toArray();
    QJsonArray referenceDatasets = normalized.value("reference_datasets").toArray();

    // 最近一次 AT 的总稀疏点数（用于"连接点"括号里显示）
    int totalSparsePoints = -1;
    // 已做过 AT 的影像集合（用于判断每张照片是否已定向）
    QSet<QString> atProcessedImages;
    for (const QJsonValue &v : atResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        // 更新总点数（取最后一次 AT 的结果）
        const int pts = obj.value(QStringLiteral("sparse_point_count")).toInt(-1);
        if (pts >= 0) totalSparsePoints = pts;
        // 收集该 AT 使用过的影像
        for (const QJsonValue &imgV : obj.value(QStringLiteral("selected_images")).toArray())
            atProcessedImages.insert(imgV.toString());
    }

    // ── 哪些影像已有相机参数（来自 AT 或手动导入）──────────────────────
    QSet<QString> cameraImages;
    for (const QJsonValue &v : images) {
        if (!v.isObject()) continue;
        const QJsonObject o = v.toObject();
        if (!o.value(QStringLiteral("camera")).toObject().isEmpty())
            cameraImages.insert(o.value(QStringLiteral("path")).toString());
    }

    const int denseCount = denseResults.size();
    const int modelCount = modelResults.size() + _transientModels.size();

    // ── 保存展开状态 ──────────────────────────────────────────────────────
    QSet<QString> expandedSections;
    for (int i = 0; i < _model->rowCount(); ++i) {
        QStandardItem *item = _model->item(i, 0);
        if (item) {
            QModelIndex idx = _model->indexFromItem(item);
            if (_view->isExpanded(idx)) {
                expandedSections.insert(item->text().section(' ', 0, 0));
            }
        }
    }
    
    _model->removeRows(0, _model->rowCount());

    // "连接点" 括号里显示稀疏点总数（若有AT结果），否则显示文件数
    const int matchesCount = (totalSparsePoints >= 0) ? totalSparsePoints : atResults.size();

    auto *photos    = createSection(QStringLiteral("照片"),    images.size());
    auto *obsNet    = createSection(QStringLiteral("观测网络"), obsNetResults.size());
    auto *matches   = createSection(QStringLiteral("连接点"),  matchesCount);
    auto *depthMaps = createSection(QStringLiteral("深度图"),  mvsDepthResultCount(depthResults));
    auto *denseCloud= createSection(QStringLiteral("稠密点云"),denseCount);
    auto *model3d   = createSection(QStringLiteral("3D模型"),  modelCount);
    auto *dem       = createSection(QStringLiteral("DEM"),      demResults.size());
    auto *ortho     = createSection(QStringLiteral("正射影像"),orthoResults.size());
    auto *references= createSection(QStringLiteral("参考数据"), referenceDatasets.size());
    auto *reports   = createSection(QStringLiteral("报告"),     reportResults.size());

    Q_UNUSED(ipfindResults);
    Q_UNUSED(ipmatchResults);

    // ── 照片（每张后面标注是否已定向）────────────────────────────────────
    for (const QJsonValue &v : images) {
        QString path;
        QString storage;
        if (v.isObject()) {
            QJsonObject o = v.toObject();
            path    = o.value("path").toString();
            storage = o.value("storage").toString();
        } else if (v.isString()) {
            path    = v.toString();
            storage = QStringLiteral("internal");
        }
        QFileInfo fi(path);
        QString name = fi.fileName();
        if (name.isEmpty()) name = path;
        // 相机状态标记：✓ = 有相机参数（AT完成）
        if (cameraImages.contains(path) || atProcessedImages.contains(path))
            name += QStringLiteral("  [✓]");
        appendItemRow(photos, name, path, storage);
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

    // ── 连接点（稀疏点云文件） ────────────────────────────────────────────
    for (int i = 0; i < atResults.size(); ++i) {
        const QJsonObject obj = atResults.at(i).toObject();
        const QJsonObject files = obj.value(QStringLiteral("files")).toObject();
        const QString sparsePath = files.value(QStringLiteral("sparse_cloud_xyz")).toString();
        if (sparsePath.isEmpty()) {
            continue;
        }

        const QString operation = obj.value(QStringLiteral("operation")).toString(QStringLiteral("triangulation"));
        QString operationLabel = obj.value(QStringLiteral("operation_display_name")).toString();
        if (operationLabel.isEmpty()) {
            operationLabel = xjw::gui::project::sparseOperationDisplayName(operation);
        }

        QString name = QStringLiteral("#%1 %2").arg(i).arg(operationLabel);
        const int pts = obj.value(QStringLiteral("sparse_point_count")).toInt(-1);
        if (pts >= 0) {
            name += QStringLiteral("  [点: %1]").arg(pts);
        }
        const int sourceIndex = obj.value(QStringLiteral("source_result_index")).toInt(-1);
        if (sourceIndex >= 0) {
            name += QStringLiteral("  [源 #%1]").arg(sourceIndex);
        }
        const QString dirName = QFileInfo(obj.value(QStringLiteral("output_dir")).toString()).fileName();
        if (!dirName.isEmpty()) {
            name += QStringLiteral("  (%1)").arg(dirName);
        }
        if (i == atResults.size() - 1) {
            name += QStringLiteral("  [当前]");
        }
        appendItemRow(matches, name, sparsePath, QStringLiteral("generated"));
    }

    // ── 3D 模型 ───────────────────────────────────────────────────────────
    for (const QJsonValue &v : modelResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        QString modelPath = obj.value(QStringLiteral("final_model_path")).toString();
        bool texturedModel = obj.value(QStringLiteral("textured")).toBool(false);
        if (modelPath.isEmpty()) {
            modelPath = obj.value(QStringLiteral("model_obj")).toString();
        }
        if (modelPath.isEmpty()) {
            modelPath = obj.value(QStringLiteral("model_ply")).toString();
        }
        if (modelPath.isEmpty()) continue;
        QString name = QFileInfo(modelPath).fileName();
        if (name.isEmpty()) name = modelPath;
        const int vtx  = obj.value(QStringLiteral("vertex_count")).toInt(-1);
        const int face = obj.value(QStringLiteral("face_count")).toInt(-1);
        if (vtx >= 0 && face >= 0)
            name = QStringLiteral("%1  [V:%2 F:%3]").arg(name).arg(vtx).arg(face);
        const QString finalFormat = obj.value(QStringLiteral("final_model_format")).toString();
        if (!finalFormat.isEmpty()) {
            name += QStringLiteral("  [%1]").arg(finalFormat);
        }
        if (texturedModel) {
            name += QStringLiteral("  [纹理]");
        }
        appendItemRow(model3d, name, modelPath, QStringLiteral("generated"));
    }

    for (const QString &modelPath : _transientModels) {
        if (modelPath.trimmed().isEmpty()) continue;
        QString name = QFileInfo(modelPath).fileName();
        if (name.isEmpty()) name = modelPath;
        name += QStringLiteral("  [临时]");
        appendItemRow(model3d, name, modelPath, QStringLiteral("temporary"));
    }

    // ── 深度图 ────────────────────────────────────────────────────────────
    for (const QJsonValue &v : depthResults) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();
        if (depthResultKind(obj) == QStringLiteral("mvs_depth"))
        {
            const QString path = obj.value(QStringLiteral("depth_png")).toString();
            if (path.isEmpty()) continue;
            QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
            const int gw = obj.value(QStringLiteral("grid_width")).toInt(-1);
            const int gh = obj.value(QStringLiteral("grid_height")).toInt(-1);
            if (gw > 0 && gh > 0)
                name = QStringLiteral("%1  [%2x%3]").arg(name).arg(gw).arg(gh);
            const QString device = obj.value(QStringLiteral("device")).toString();
            if (!device.isEmpty() && device != QStringLiteral("unknown"))
            {
                name += QStringLiteral("  [%1]").arg(device);
            }
            const QString status = obj.value(QStringLiteral("status")).toString();
            if (!status.isEmpty() && status != QStringLiteral("completed"))
            {
                name += QStringLiteral("  [%1]").arg(status);
            }
            appendItemRow(depthMaps, name, path, QStringLiteral("generated"));
        }
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

    for (int i = 0; i < _model->rowCount(); ++i)
    {
        sortSectionChildrenByFileName(_model->item(i, 0));
    }
    
    // ── 恢复展开状态 ──────────────────────────────────────────────────────
    // 仅恢复用户先前显式展开过的分组，不在数据更新时自动展开任何默认分组。
    // 之前这里对第一个分组（通常是“照片”）做了强制展开，导致新增深度图/点云等
    // 元数据写回后，工作区会突然自动展开“照片”树，打断用户当前浏览位置。
    for (int i = 0; i < _model->rowCount(); ++i) {
        QStandardItem *item = _model->item(i, 0);
        if (item) {
            QString sectionName = item->text().section(' ', 0, 0);
            if (expandedSections.contains(sectionName)) {
                QModelIndex idx = _model->indexFromItem(item);
                _view->expand(idx);
            }
        }
    }
}

void DataTreeWidget::onContextMenuRequested(const QPoint &pos)
{
    QModelIndex idx = _view->indexAt(pos);
    if (!idx.isValid()) return;

    // 收集当前选中的所有条目（支持树状选择），如果没有多选，则只包含右键所在项
    QModelIndexList sel = _view->selectionModel()->selectedIndexes();
    QSet<QModelIndex> uniqueRows;
    for (const QModelIndex &si : sel) {
        if (si.isValid() && si.column() == 0) uniqueRows.insert(si);
    }
    if (uniqueRows.isEmpty()) uniqueRows.insert(idx);

    // 从模型中收集路径列表
    QStringList paths;
    QList<QModelIndex> rows;
    QString sectionName;
    bool sameSection = true;
    bool hasSectionRoot = false;
    for (const QModelIndex &i : uniqueRows) {
        if (!i.parent().isValid()) {
            hasSectionRoot = true;
            continue;
        }
        QModelIndex pathIdx = i.sibling(i.row(), 1);
        if (pathIdx.isValid()) {
            paths << resolveResourcePath(_model->data(pathIdx).toString());
            rows.append(i);

            const QString rowSection = _model->data(i.parent()).toString().section(' ', 0, 0);
            if (sectionName.isEmpty()) {
                sectionName = rowSection;
            } else if (sectionName != rowSection) {
                sameSection = false;
            }
        }
    }

    if (rows.isEmpty() && hasSectionRoot) {
        return;
    }

    QMenu menu(this);
    QAction *openAct = menu.addAction(tr("打开"));
    QAction *revealAct = menu.addAction(tr("在文件管理器中显示"));
    QAction *propAct = menu.addAction(tr("属性..."));
    QAction *packAct = menu.addAction(tr("打包到归档 (.plascan)"));
    QAction *removeAct = menu.addAction(tr("移除引用"));
    QAction *deleteAct = nullptr;
    QAction *sideOpenAct = nullptr;
    if (sameSection && !sectionName.isEmpty() && sectionName != QStringLiteral("照片"))
    {
        deleteAct = menu.addAction(tr("删除数据"));
    }
    if (sameSection
        && paths.size() == 1
        && (sectionName == QStringLiteral("照片")
            || sectionName == QStringLiteral("深度图")
            || sectionName == QStringLiteral("DEM")
            || sectionName == QStringLiteral("正射影像")))
    {
        sideOpenAct = menu.addAction(tr("在侧边打开"));
    }

    QAction *act = menu.exec(_view->viewport()->mapToGlobal(pos));
    if (!act) return;
    if (act == openAct)
    {
        // 仅处理第一个（打开多个文件一次性打开可能令人困惑）
        if (!paths.isEmpty()) emit openRequested(paths.first());
    }
    else if (act == revealAct)
    {
        if (!paths.isEmpty()) emit revealRequested(paths.first());
    }
    else if (act == propAct)
    {
        // 仅展示第一个影像的属性（多选时避免弹出多个窗口）
        if (paths.isEmpty()) return;

        const QString path = paths.first();
        QFileInfo fi(path);

        // 使用 QImageReader 获取尽可能多的信息。
        // 说明：
        // - QImageReader 对很多常规格式支持良好。
        // - 对 16/32 位 TIF 等非常规数据，QImageReader 的解析能力取决于 Qt 插件与编译选项。
        // - 即使无法完整读取像素，也应能拿到尺寸等基本元数据。
        QImageReader reader(path);
        const QSize size = reader.size();
        const QImage::Format fmt = reader.imageFormat();

        QStringList lines;
        lines << tr("名称：%1").arg(fi.fileName());
        lines << tr("路径：%1").arg(QDir::toNativeSeparators(fi.absoluteFilePath()));
        if (size.isValid())
        {
            lines << tr("尺寸：%1 x %2").arg(size.width()).arg(size.height());
        }
        else
        {
            lines << tr("尺寸：未知（QImageReader 无法解析）");
        }
        // 将 QImage::Format 转为可读名称（例如 Format_Grayscale8），便于用户判断像素深度/通道
        auto qtFormatName = [&](QImage::Format f) -> QString {
            switch (f) {
                case QImage::Format_Invalid: return QStringLiteral("Invalid");
                case QImage::Format_Mono: return QStringLiteral("Mono (1-bit)");
                case QImage::Format_MonoLSB: return QStringLiteral("MonoLSB (1-bit)");
                case QImage::Format_Indexed8: return QStringLiteral("Indexed8 (8-bit paletted)");
                case QImage::Format_Grayscale8: return QStringLiteral("Grayscale8 (8-bit)");
                case QImage::Format_RGB32: return QStringLiteral("RGB32 (32-bit packed)");
                case QImage::Format_ARGB32: return QStringLiteral("ARGB32 (32-bit with alpha)");
                case QImage::Format_ARGB32_Premultiplied: return QStringLiteral("ARGB32 Premultiplied");
                case QImage::Format_RGB16: return QStringLiteral("RGB16 (16-bit)");
                case QImage::Format_ARGB8565_Premultiplied: return QStringLiteral("ARGB8565 Premultiplied");
                case QImage::Format_RGB666: return QStringLiteral("RGB666");
                case QImage::Format_RGB555: return QStringLiteral("RGB555");
                case QImage::Format_RGB888: return QStringLiteral("RGB888 (24-bit)");
                case QImage::Format_RGB444: return QStringLiteral("RGB444");
                case QImage::Format_Grayscale16: return QStringLiteral("Grayscale16 (16-bit)");
                case QImage::Format_RGBX8888: return QStringLiteral("RGBX8888");
                case QImage::Format_RGBA8888: return QStringLiteral("RGBA8888");
                case QImage::Format_RGBA8888_Premultiplied: return QStringLiteral("RGBA8888 Premultiplied");
                default:
                    return QStringLiteral("Unknown (enum value: %1)").arg(static_cast<int>(f));
            }
        };
        lines << tr("Qt 解析格式：%1").arg(qtFormatName(fmt));
        if (!reader.format().isEmpty())
        {
            lines << tr("文件格式：%1").arg(QString::fromLatin1(reader.format()));
        }
        if (!reader.supportsOption(QImageIOHandler::ImageFormat))
        {
            // 仅做提示：不是错误
        }

        // 从隐藏列拿到 storage 字段（如果存在）。使用我们计算的 rows 列表（对应于 paths）
        QString storage;
        if (!rows.isEmpty()) {
            QModelIndex sidx = rows.first().sibling(rows.first().row(), 2);
            if (sidx.isValid()) storage = _model->data(sidx).toString();
        }
        if (!storage.isEmpty())
        {
            lines << tr("存储方式：%1").arg(storage);
        }

        QMessageBox::information(this, tr("影像属性"), lines.join("\n"));
    }
    else if (act == packAct)
    {
        // 同样仅打包第一个项（批量打包可在后续实现）
        if (!paths.isEmpty()) emit packRequested(paths.first());
    }
    else if (act == removeAct)
    {
        // 批量删除：发出所有被选中文件的路径
        emit removeRequested(paths);
    }
    else if (deleteAct && act == deleteAct)
    {
        emit deleteDataRequested(sectionName, paths);
    }
    else if (sideOpenAct && act == sideOpenAct)
    {
        emit sideOpenRequested(sectionName, paths.first());
    }
}

#include "DataTreeWidget.h"

#include <QTreeView>
#include <QStandardItemModel>
#include <QHeaderView>
#include <QMenu>
#include <QAction>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFileInfo>
#include <QFile>
#include <QDir>
#include <QVBoxLayout>
#include <QDebug>
#include <QMessageBox>
#include <QImageReader>
#include <QStyle>

#include "PlascanArchive.h"
#include "Logger.h"

// DataTreeWidget 的实现：
// - 通过 PlascanArchive 读取 project_files.json（或回退到磁盘上的文件），解析 images 数组
// - 将每个资源作为一行显示，列为：名称 / 路径 / 存储方式（internal/reference）
// - 右键菜单提供打开 / 在文件管理器中显示 / 打包 / 移除等操作（只发出信号，由上层处理）

DataTreeWidget::DataTreeWidget(QWidget *parent)
    : QWidget(parent)
{
    m_view = new QTreeView(this);
    m_model = new QStandardItemModel(this);
    // 左侧列表显示策略：
    // - 只展示“名称”一列，保持列表简洁（用户不希望直接看到路径/存储）。
    // - 但为了右键菜单“属性”弹窗仍能显示完整信息，这里仍保留 path/storage 作为隐藏列。
    m_model->setHorizontalHeaderLabels({QObject::tr("名称"), QObject::tr("路径"), QObject::tr("存储")});

    m_view->setModel(m_model);
    // 隐藏“路径/存储”两列
    m_view->setColumnHidden(1, true);
    m_view->setColumnHidden(2, true);
    // 禁用双击进入编辑（用户要求：双击不应改名）
    m_view->setEditTriggers(QAbstractItemView::NoEditTriggers);
    // 允许使用 Ctrl / Shift 多选
    m_view->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_view->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_view, &QTreeView::customContextMenuRequested, this, &DataTreeWidget::onContextMenuRequested);

    // 单击/双击任意项时，通知上层切换中央影像显示
    connect(m_view, &QTreeView::clicked, this, [this](const QModelIndex &idx) {
        if (!idx.isValid()) return;
        // Use the index to access the hidden 'path' column in the same row/parent
        QModelIndex pathIdx = idx.sibling(idx.row(), 1);
        if (!pathIdx.isValid()) return;
        QString path = m_model->data(pathIdx).toString();
        QModelIndex nameIdx = idx.sibling(idx.row(), 0);
        QModelIndex parentNameIdx = nameIdx.parent().isValid() ? nameIdx.parent() : QModelIndex();
        QString section;
        if (parentNameIdx.isValid()) {
            section = m_model->data(parentNameIdx).toString().section(' ', 0, 0);
        }
        if (!path.trimmed().isEmpty())
        {
            if (!QFileInfo(path).isAbsolute() && !m_currentPlascanPath.trimmed().isEmpty()) {
                const QString root = QFileInfo(m_currentPlascanPath).absolutePath();
                path = QDir(root).filePath(path);
            }
            emit resourceActivated(section, path);
            if (section == QStringLiteral("照片")
                || section == QStringLiteral("深度图")
                || section == QStringLiteral("DEM")
                || section == QStringLiteral("正射影像")) {
                emit imageActivated(path);
            }
        }
    });

    auto layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_view);
    setLayout(layout);
}

DataTreeWidget::~DataTreeWidget()
{
}

void DataTreeWidget::loadFromArchive(const QString &plascanPath)
{
    m_currentPlascanPath = plascanPath;
    m_model->removeRows(0, m_model->rowCount());

        // debug logs removed

    QByteArray content;

    // 优先检查运行时临时目录中的元数据（避免打开归档每次写入）
    QString projectRoot = QFileInfo(plascanPath).absolutePath();
    QString tempMeta = QDir(projectRoot).filePath(".plascan_tmp/project_files.json");
    QFile tf(tempMeta);
    if (tf.open(QIODevice::ReadOnly))
    {
        content = tf.readAll();
        tf.close();
    }

    // 临时文件不存在时再尝试从归档中读取
    if (content.isEmpty())
    {
        QString err;
        PlascanArchive arch(plascanPath);
        if (arch.isValid())
        {
            content = arch.readEntry("project_files.json", &err);
        }
        else
        {
        }
    }

    // 最后回退到磁盘上的 project_files.json
    if (content.isEmpty())
    {
        QString metaPath = QDir(projectRoot).filePath("project_files.json");
        QFile f(metaPath);
        if (f.open(QIODevice::ReadOnly))
        {
            content = f.readAll();
            f.close();
        }
        else
        {
        }
    }

    if (content.isEmpty())
    {
        m_lastMeta = QJsonObject();
        return; // 没有元数据可显示
    }

    QJsonDocument doc = QJsonDocument::fromJson(content);
    if (!doc.isObject()) return;
    m_lastMeta = doc.object();
    populateFromMeta(m_lastMeta);

}

void DataTreeWidget::loadFromJson(const QJsonObject &meta)
{
    // 注意：这里不能在 images 为空时就清空列表并返回。
    // 因为 projectOpened/metadataChanged 等信号可能会短暂发送一个空 meta，
    // 若直接清空会覆盖掉刚从 loadFromArchive() 填充的列表，造成“解析出 rowCount=2 但 UI 仍空”。
    if (meta.isEmpty()) return;

    // 支持两种结构：根对象的 "images" 或嵌套在 "project_files" 中的 "images"。
    // 如果 meta 明确包含 images（即使为空数组），我们应该以该值为准并更新视图。
    bool hasImagesKey = meta.contains(QStringLiteral("images"));
    QJsonArray images = meta.value("images").toArray();
    if (!hasImagesKey && meta.value(QStringLiteral("project_files")).isObject())
    {
        QJsonObject pf = meta.value(QStringLiteral("project_files")).toObject();
        if (pf.contains(QStringLiteral("images"))) {
            hasImagesKey = true;
            images = pf.value(QStringLiteral("images")).toArray();
        }
    }

    // 如果 meta 中没有 images 字段（也没有 project_files.images），则视为非资源更新，忽略。
    if (!hasImagesKey) return;

    // 无论 images 是否为空，只要 meta 明确提供了 images，我们就清空并重建模型（允许清空列表）。
    m_lastMeta = meta;
    populateFromMeta(meta);
}

void DataTreeWidget::addTransientModel(const QString &modelPath)
{
    const QString cleanPath = QDir::cleanPath(modelPath.trimmed());
    if (cleanPath.isEmpty())
    {
        return;
    }

    if (!m_transientModels.contains(cleanPath))
    {
        m_transientModels.append(cleanPath);
    }

    if (m_lastMeta.isEmpty())
    {
        QJsonObject emptyProject;
        emptyProject.insert(QStringLiteral("images"), QJsonArray());
        m_lastMeta = emptyProject;
    }
    populateFromMeta(m_lastMeta);
}

void DataTreeWidget::clearTransientResources()
{
    if (m_transientModels.isEmpty())
    {
        return;
    }

    m_transientModels.clear();
    if (!m_lastMeta.isEmpty())
    {
        populateFromMeta(m_lastMeta);
    }
    else
    {
        m_model->removeRows(0, m_model->rowCount());
    }
}

QJsonObject DataTreeWidget::normalizeMeta(const QJsonObject &meta) const
{
    if (meta.contains(QStringLiteral("images"))) {
        return meta;
    }
    if (meta.value(QStringLiteral("project_files")).isObject()) {
        return meta.value(QStringLiteral("project_files")).toObject();
    }
    return QJsonObject();
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

    m_model->appendRow({nameItem, pathItem, storageItem});
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
    const int modelCount = modelResults.size() + m_transientModels.size();

    // ── 保存展开状态 ──────────────────────────────────────────────────────
    QSet<QString> expandedSections;
    for (int i = 0; i < m_model->rowCount(); ++i) {
        QStandardItem *item = m_model->item(i, 0);
        if (item) {
            QModelIndex idx = m_model->indexFromItem(item);
            if (m_view->isExpanded(idx)) {
                expandedSections.insert(item->text().section(' ', 0, 0));
            }
        }
    }
    
    m_model->removeRows(0, m_model->rowCount());

    // "连接点" 括号里显示稀疏点总数（若有AT结果），否则显示文件数
    const int matchesCount = (totalSparsePoints >= 0) ? totalSparsePoints : atResults.size();

    auto *photos    = createSection(QStringLiteral("照片"),    images.size());
    auto *obsNet    = createSection(QStringLiteral("观测网络"), obsNetResults.size());
    auto *matches   = createSection(QStringLiteral("连接点"),  matchesCount);
    auto *depthMaps = createSection(QStringLiteral("深度图"),  depthResults.size());
    auto *denseCloud= createSection(QStringLiteral("稠密点云"),denseCount);
    auto *model3d   = createSection(QStringLiteral("3D模型"),  modelCount);
    auto *dem       = createSection(QStringLiteral("DEM"),      demResults.size());
    auto *ortho     = createSection(QStringLiteral("正射影像"),orthoResults.size());

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
            if (operation == QStringLiteral("triangulation")) {
                operationLabel = QStringLiteral("初始稀疏点云");
            } else if (operation == QStringLiteral("outlier_removal")) {
                operationLabel = QStringLiteral("离群点剔除");
            } else if (operation == QStringLiteral("sparse_refine")) {
                operationLabel = QStringLiteral("稀疏点云精修");
            } else {
                operationLabel = QStringLiteral("稀疏点云");
            }
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

    for (const QString &modelPath : m_transientModels) {
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
        const QString path = obj.value(QStringLiteral("depth_png")).toString();
        if (path.isEmpty()) continue;
        QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        const int gw = obj.value(QStringLiteral("grid_width")).toInt(-1);
        const int gh = obj.value(QStringLiteral("grid_height")).toInt(-1);
        if (gw > 0 && gh > 0)
            name = QStringLiteral("%1  [%2x%3]").arg(name).arg(gw).arg(gh);
        appendItemRow(depthMaps, name, path, QStringLiteral("generated"));
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
        const QString path = obj.value(QStringLiteral("dem_tif")).toString();
        if (path.isEmpty()) continue;
        QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        const QString typ = obj.value(QStringLiteral("dem_type")).toString();
        if (!typ.isEmpty()) name = QStringLiteral("%1  [%2]").arg(name, typ);
        appendItemRow(dem, name, path, QStringLiteral("generated"));
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
    
    // ── 恢复展开状态 ──────────────────────────────────────────────────────
    // 仅恢复用户先前显式展开过的分组，不在数据更新时自动展开任何默认分组。
    // 之前这里对第一个分组（通常是“照片”）做了强制展开，导致新增深度图/点云等
    // 元数据写回后，工作区会突然自动展开“照片”树，打断用户当前浏览位置。
    for (int i = 0; i < m_model->rowCount(); ++i) {
        QStandardItem *item = m_model->item(i, 0);
        if (item) {
            QString sectionName = item->text().section(' ', 0, 0);
            if (expandedSections.contains(sectionName)) {
                QModelIndex idx = m_model->indexFromItem(item);
                m_view->expand(idx);
            }
        }
    }
}

void DataTreeWidget::onContextMenuRequested(const QPoint &pos)
{
    QModelIndex idx = m_view->indexAt(pos);
    if (!idx.isValid()) return;

    // 收集当前选中的所有条目（支持树状选择），如果没有多选，则只包含右键所在项
    QModelIndexList sel = m_view->selectionModel()->selectedIndexes();
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
            paths << m_model->data(pathIdx).toString();
            rows.append(i);

            const QString rowSection = m_model->data(i.parent()).toString().section(' ', 0, 0);
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

    QAction *act = menu.exec(m_view->viewport()->mapToGlobal(pos));
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
            if (sidx.isValid()) storage = m_model->data(sidx).toString();
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

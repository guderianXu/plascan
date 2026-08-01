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

void DataTreeWidget::onContextMenuRequested(const QPoint &pos)
{
    QModelIndex idx = _view->indexAt(pos);
    if (!idx.isValid()) return;

    const QModelIndex nameIndex = idx.sibling(idx.row(), 0);
    if (_model->data(nameIndex, WorkspaceRootRole).toBool())
    {
        QMenu menu(this);
        QAction *createAction =
            menu.addAction(tr("新建 Chunk..."));
        if (menu.exec(_view->viewport()->mapToGlobal(pos))
            == createAction)
        {
            emit createChunkRequested();
        }
        return;
    }
    const QString chunkId =
        _model->data(nameIndex, ChunkIdRole).toString();
    if (!chunkId.isEmpty())
    {
        QMenu menu(this);
        QAction *activateAction = nullptr;
        if (chunkId != _activeChunkId)
        {
            activateAction = menu.addAction(tr("设为当前 Chunk"));
        }
        QAction *createAction = menu.addAction(tr("新建 Chunk..."));
        QAction *renameAction = menu.addAction(tr("重命名 Chunk..."));
        QAction *removeAction = menu.addAction(tr("删除 Chunk..."));
        QAction *selected =
            menu.exec(_view->viewport()->mapToGlobal(pos));
        if (selected == activateAction)
        {
            emit switchChunkRequested(chunkId);
        }
        else if (selected == createAction)
        {
            emit createChunkRequested();
        }
        else if (selected == renameAction)
        {
            emit renameChunkRequested(chunkId);
        }
        else if (selected == removeAction)
        {
            emit removeChunkRequested(chunkId);
        }
        return;
    }

    QString contextSection;
    QString contextPath;
    if (resourceFromIndex(nameIndex, &contextSection, &contextPath)
        && contextSection == QStringLiteral("连接点"))
    {
        QMenu menu(this);
        QAction *zoomToAction = menu.addAction(tr("放大至"));
        QAction *removeTiePointsAction = menu.addAction(tr("移除连接点"));
        QAction *showInformationAction = menu.addAction(tr("显示信息"));
        QAction *revealAction = menu.addAction(tr("在文件管理器中显示"));
        const bool resourceExists = QFileInfo::exists(contextPath);
        zoomToAction->setEnabled(resourceExists);
        revealAction->setEnabled(resourceExists);

        QAction *selectedAction =
            menu.exec(_view->viewport()->mapToGlobal(pos));
        if (selectedAction == zoomToAction)
        {
            emit resourceActivated(contextSection, contextPath);
        }
        else if (selectedAction == removeTiePointsAction)
        {
            emit deleteDataRequested(
                contextSection,
                QStringList{contextPath});
        }
        else if (selectedAction == showInformationAction)
        {
            const QFileInfo fileInfo(contextPath);
            QStringList lines{
                tr("类型：连接点"),
                tr("名称：%1").arg(_model->data(nameIndex).toString()),
                tr("文件：%1").arg(fileInfo.fileName()),
                tr("路径：%1").arg(
                    QDir::toNativeSeparators(fileInfo.absoluteFilePath()))
            };
            if (fileInfo.exists())
            {
                lines.append(
                    tr("大小：%1").arg(
                        QLocale().formattedDataSize(fileInfo.size())));
                lines.append(
                    tr("修改时间：%1").arg(
                        QLocale().toString(
                            fileInfo.lastModified(),
                            QLocale::ShortFormat)));
            }
            else
            {
                lines.append(tr("状态：文件不存在"));
            }
            QMessageBox::information(
                this,
                tr("连接点信息"),
                lines.join(QLatin1Char('\n')));
        }
        else if (selectedAction == revealAction)
        {
            emit revealRequested(contextPath);
        }
        return;
    }

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
    bool hasAggregateRow = false;
    bool hasRegularRow = false;
    for (const QModelIndex &i : uniqueRows) {
        QString rowSection;
        QString rowPath;
        const QModelIndex nameIndex = i.sibling(i.row(), 0);
        const QStringList aggregatePaths = _model->data(nameIndex, AggregateResourcePathsRole).toStringList();
        if (!aggregatePaths.isEmpty())
        {
            rowSection = _model->data(nameIndex, SectionRole).toString();
            for (const QString &aggregatePath : aggregatePaths)
            {
                const QString resolvedPath = resolveResourcePath(aggregatePath);
                if (!resolvedPath.isEmpty() && !paths.contains(resolvedPath))
                {
                    paths.append(resolvedPath);
                }
            }
            rows.append(i);
            hasAggregateRow = true;
            if (sectionName.isEmpty())
            {
                sectionName = rowSection;
            }
            else if (sectionName != rowSection)
            {
                sameSection = false;
            }
            continue;
        }
        if (resourceFromIndex(i, &rowSection, &rowPath)) {
            paths << rowPath;
            rows.append(i);
            hasRegularRow = true;
            if (sectionName.isEmpty()) {
                sectionName = rowSection;
            } else if (sectionName != rowSection) {
                sameSection = false;
            }
        }
    }

    if (rows.isEmpty()) {
        return;
    }

    QMenu menu(this);
    const bool depthAggregateOnly = hasAggregateRow && !hasRegularRow
        && sameSection && sectionName == QStringLiteral("深度图");
    QAction *openAct = nullptr;
    QAction *revealAct = nullptr;
    QAction *propAct = nullptr;
    QAction *packAct = nullptr;
    QAction *removeAct = nullptr;
    QAction *deleteAct = nullptr;
    QAction *sideOpenAct = nullptr;
    QAction *viewMatchesAct = nullptr;
    if (depthAggregateOnly)
    {
        deleteAct = menu.addAction(tr("删除深度图"));
    }
    else
    {
        openAct = menu.addAction(tr("打开"));
        revealAct = menu.addAction(tr("在文件管理器中显示"));
        propAct = menu.addAction(tr("属性..."));
        packAct = menu.addAction(tr("打包到归档 (.plascan)"));
        removeAct = menu.addAction(tr("移除引用"));
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
        if (sameSection
            && hasRegularRow
            && !hasAggregateRow
            && paths.size() == 1
            && sectionName == QStringLiteral("照片"))
        {
            viewMatchesAct = menu.addAction(tr("查看匹配..."));
        }
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

        QString path = paths.first();
        path = resolveResourcePath(path);
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
    else if (viewMatchesAct && act == viewMatchesAct)
    {
        emit viewMatchesRequested(paths.first());
    }
}

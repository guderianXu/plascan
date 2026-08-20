#include "DataTreeWidget.h"

#include "ui_DataTreeWidget.h"
#include "ProjectWorkflowOperations.h"
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
QStandardItem *DataTreeWidget::createSection(
    const QString &label,
    xjw::gui::widgets::WorkspaceSection section)
{
    auto *nameItem = new QStandardItem(label);
    nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
    nameItem->setIcon(xjw::gui::widgets::workspaceSectionIcon(section));
    nameItem->setData(workspaceSectionName(section), SectionRole);
    nameItem->setData(static_cast<int>(section), WorkspaceSectionRole);

    auto *pathItem = new QStandardItem(QString());
    auto *storageItem = new QStandardItem(QString());
    pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
    storageItem->setFlags(storageItem->flags() & ~Qt::ItemIsEditable);

    if (_activeChunkRoot)
    {
        _activeChunkRoot->appendRow(
            {nameItem, pathItem, storageItem});
    }
    else
    {
        _model->appendRow({nameItem, pathItem, storageItem});
    }
    return nameItem;
}

QStandardItem *DataTreeWidget::createSection(
    const QString &title,
    int count,
    xjw::gui::widgets::WorkspaceSection section)
{
    return createSection(QStringLiteral("%1 (%2)").arg(title).arg(count), section);
}

QStandardItem *DataTreeWidget::appendItemRow(QStandardItem *parent,
                                             const QString &name,
                                             const QString &path,
                                             const QString &storage)
{
    if (!parent) return nullptr;
    auto *nameItem = new QStandardItem(name);
    auto *pathItem = new QStandardItem(path);
    auto *storageItem = new QStandardItem(storage);
    nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
    pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
    storageItem->setFlags(storageItem->flags() & ~Qt::ItemIsEditable);
    nameItem->setData(parent->data(SectionRole), SectionRole);
    nameItem->setData(parent->data(WorkspaceSectionRole), WorkspaceSectionRole);
    nameItem->setData(path, ResourcePathRole);
    parent->appendRow({nameItem, pathItem, storageItem});
    return nameItem;
}

QStandardItem *DataTreeWidget::appendTopLevelResource(
    const QString &name,
    xjw::gui::widgets::WorkspaceSection section,
    const QString &sectionName,
    const QString &path,
    const QString &storage)
{
    auto *nameItem = new QStandardItem(name);
    auto *pathItem = new QStandardItem(path);
    auto *storageItem = new QStandardItem(storage);
    nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
    pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
    storageItem->setFlags(storageItem->flags() & ~Qt::ItemIsEditable);
    nameItem->setIcon(xjw::gui::widgets::workspaceSectionIcon(section));
    nameItem->setData(sectionName, SectionRole);
    nameItem->setData(static_cast<int>(section), WorkspaceSectionRole);
    nameItem->setData(path, ResourcePathRole);
    if (_activeChunkRoot)
    {
        _activeChunkRoot->appendRow(
            {nameItem, pathItem, storageItem});
    }
    else
    {
        _model->appendRow({nameItem, pathItem, storageItem});
    }
    return nameItem;
}

QStandardItem *DataTreeWidget::appendTopLevelAggregate(
    const QString &name,
    xjw::gui::widgets::WorkspaceSection section,
    const QString &sectionName,
    const QStringList &paths)
{
    if (paths.isEmpty())
    {
        return nullptr;
    }

    auto *nameItem = new QStandardItem(name);
    auto *pathItem = new QStandardItem(QString());
    auto *storageItem = new QStandardItem(QStringLiteral("generated"));
    nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
    pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
    storageItem->setFlags(storageItem->flags() & ~Qt::ItemIsEditable);
    nameItem->setIcon(xjw::gui::widgets::workspaceSectionIcon(section));
    nameItem->setData(sectionName, SectionRole);
    nameItem->setData(static_cast<int>(section), WorkspaceSectionRole);
    nameItem->setData(paths, AggregateResourcePathsRole);
    if (_activeChunkRoot)
    {
        _activeChunkRoot->appendRow(
            {nameItem, pathItem, storageItem});
    }
    else
    {
        _model->appendRow({nameItem, pathItem, storageItem});
    }
    return nameItem;
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

bool DataTreeWidget::resourceFromIndex(const QModelIndex &index, QString *section, QString *resourcePath) const
{
    if (!index.isValid() || !_model)
    {
        return false;
    }

    const QModelIndex nameIndex = index.sibling(index.row(), 0);
    const QModelIndex parentIndex = nameIndex.parent();
    const QString topLevelPath = _model->data(nameIndex, ResourcePathRole).toString();
    const QString topLevelSection = _model->data(nameIndex, SectionRole).toString();
    if (!topLevelPath.trimmed().isEmpty() && !topLevelSection.trimmed().isEmpty())
    {
        if (section)
        {
            *section = topLevelSection;
        }
        if (resourcePath)
        {
            *resourcePath = resolveResourcePath(topLevelPath);
        }
        return true;
    }

    if (!parentIndex.isValid())
    {
        return false;
    }

    const QModelIndex pathIndex = index.sibling(index.row(), 1);
    if (!pathIndex.isValid())
    {
        return false;
    }

    const QString rawPath = _model->data(pathIndex).toString();
    if (rawPath.trimmed().isEmpty())
    {
        return false;
    }

    const QString parentSection = _model->data(parentIndex, SectionRole).toString();
    if (parentSection.trimmed().isEmpty())
    {
        return false;
    }
    if (section)
    {
        *section = parentSection;
    }
    if (resourcePath)
    {
        *resourcePath = resolveResourcePath(rawPath);
    }
    return true;
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

    return xjw::common::project::ProjectIO::resolveProjectResourcePath(
        _currentPlascanPath, trimmedPath);
}

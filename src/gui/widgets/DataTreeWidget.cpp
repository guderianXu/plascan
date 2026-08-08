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
    _view->setIconSize(QSize(16, 16));
    _view->header()->hide();
    _view->setIndentation(18);
    _view->setUniformRowHeights(true);
    _view->setRootIsDecorated(true);
    _view->setItemsExpandable(true);
    _view->setAllColumnsShowFocus(true);
    _view->setStyleSheet(QStringLiteral(
        "QTreeView {"
        "  border: none;"
        "  background: palette(base);"
        "  outline: 0;"
        "}"
        "QTreeView::item {"
        "  min-height: 23px;"
        "  padding: 1px 2px;"
        "}"
        "QTreeView::item:selected {"
        "  background: palette(highlight);"
        "  color: palette(highlighted-text);"
        "}"));
    // 隐藏“路径/存储”两列
    _view->setColumnHidden(1, true);
    _view->setColumnHidden(2, true);
    // 禁用双击进入编辑（用户要求：双击不应改名）
    _view->setEditTriggers(QAbstractItemView::NoEditTriggers);
    // 允许使用 Ctrl / Shift 多选
    _view->setSelectionMode(QAbstractItemView::ExtendedSelection);
    connect(_view, &QTreeView::customContextMenuRequested, this, &DataTreeWidget::onContextMenuRequested);

    if (_view->selectionModel())
    {
        connect(_view->selectionModel(), &QItemSelectionModel::currentChanged,
                this, [this](const QModelIndex &current, const QModelIndex &)
        {
            QString section;
            QString path;
            if (resourceFromIndex(current, &section, &path))
            {
                emit resourceSelected(section, path);
            }
        });
    }

    // 双击或回车激活资源时，通知上层切换中央显示。
    // 单击只负责选择，避免浏览资源树时同步加载大图造成界面卡顿。
    connect(_view, &QTreeView::activated, this, [this](const QModelIndex &idx) {
        const QModelIndex nameIndex = idx.sibling(idx.row(), 0);
        const QString chunkId =
            _model->data(nameIndex, ChunkIdRole).toString();
        if (!chunkId.isEmpty())
        {
            if (chunkId != _activeChunkId)
            {
                emit switchChunkRequested(chunkId);
            }
            return;
        }
        QString section;
        QString path;
        if (resourceFromIndex(idx, &section, &path))
        {
            emit resourceActivated(section, path);
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
    _workspaceRoot = nullptr;
    _activeChunkRoot = nullptr;
    _model->removeRows(0, _model->rowCount());
}

void DataTreeWidget::clearProject()
{
    _currentPlascanPath.clear();
    _lastMeta = QJsonObject();
    _transientModels.clear();
    _chunks = QJsonArray();
    _activeChunkId.clear();
    _workspaceRoot = nullptr;
    _activeChunkRoot = nullptr;
    _model->removeRows(0, _model->rowCount());
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

void DataTreeWidget::setChunkContext(
    const QJsonArray &chunks,
    const QString &activeChunkId)
{
    _chunks = chunks;
    _activeChunkId = activeChunkId;
    if (!_lastMeta.isEmpty())
    {
        populateFromMeta(_lastMeta);
    }
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

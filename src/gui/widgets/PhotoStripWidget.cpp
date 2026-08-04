#include "PhotoStripWidget.h"

#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "project/ProjectIO.h"

#include "../views/LayerImageLoader.h"

#include <QAbstractItemView>
#include <QAction>
#include <QColor>
#include <QDir>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QItemSelectionModel>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QListView>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMenu>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QPixmap>
#include <QRectF>
#include <QSize>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent>

namespace
{
constexpr int PathRole = Qt::UserRole + 1;
constexpr int ThumbWidth = 132;
constexpr int ThumbHeight = 88;
constexpr int GridWidth = 220;
constexpr int GridHeight = 140;
constexpr int AsyncListThreshold = 100;
constexpr int ImageListBatchSize = 40;

bool hasAlignmentEvidence(const QJsonObject &entry)
{
    if (entry.value(QStringLiteral("camera")).isObject())
    {
        return true;
    }
    if (entry.contains(QStringLiteral("center")))
    {
        return true;
    }
    return entry.contains(QStringLiteral("camera_center"));
}

bool isAlignedEntry(const QJsonObject &entry)
{
    if (entry.contains(QStringLiteral("aligned")))
    {
        return entry.value(QStringLiteral("aligned")).toBool(false);
    }
    return hasAlignmentEvidence(entry);
}

QString displayNameForEntry(const QJsonObject &entry, const QString &imagePath)
{
    const QString name = entry.value(QStringLiteral("name")).toString().trimmed();
    if (!name.isEmpty())
    {
        return name;
    }

    const QString fileName = QFileInfo(imagePath).fileName();
    return fileName.isEmpty() ? imagePath : fileName;
}

QIcon placeholderPhotoIcon()
{
    static const QIcon icon = []()
    {
        QImage image(QSize(ThumbWidth, ThumbHeight), QImage::Format_ARGB32_Premultiplied);
        image.fill(QColor(246, 249, 252));

        QPainter painter(&image);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(QColor(188, 199, 211), 1));
        painter.setBrush(QColor(236, 242, 248));
        painter.drawRoundedRect(QRectF(0.5, 0.5, ThumbWidth - 1.0, ThumbHeight - 1.0), 4.0, 4.0);

        painter.setPen(QPen(QColor(117, 135, 153), 2));
        painter.drawLine(QPointF(28.0, 60.0), QPointF(56.0, 38.0));
        painter.drawLine(QPointF(56.0, 38.0), QPointF(76.0, 54.0));
        painter.drawLine(QPointF(76.0, 54.0), QPointF(102.0, 30.0));
        painter.setBrush(QColor(117, 135, 153));
        painter.setPen(Qt::NoPen);
        painter.drawEllipse(QPointF(38.0, 28.0), 5.0, 5.0);
        return QIcon(QPixmap::fromImage(image));
    }();
    return icon;
}
} // namespace

PhotoStripWidget::PhotoStripWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    _list = new QListWidget(this);
    _list->setObjectName(QStringLiteral("photoStripList"));
    _list->setViewMode(QListView::IconMode);
    _list->setFlow(QListView::LeftToRight);
    _list->setWrapping(true);
    _list->setMovement(QListView::Static);
    _list->setResizeMode(QListView::Adjust);
    _list->setSelectionMode(QAbstractItemView::ExtendedSelection);
    _list->setSelectionBehavior(QAbstractItemView::SelectItems);
    _list->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _list->setContextMenuPolicy(Qt::CustomContextMenu);
    _list->setIconSize(QSize(ThumbWidth, ThumbHeight));
    _list->setGridSize(QSize(GridWidth, GridHeight));
    _list->setSpacing(6);
    _list->setUniformItemSizes(true);
    _list->setWordWrap(true);
    _list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _list->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    layout->addWidget(_list);

    connect(_list, &QListWidget::itemClicked, this, [this](QListWidgetItem *item)
    {
        if (!item || !item->isSelected())
        {
            return;
        }
        emit photoSelected(item->data(PathRole).toString());
    });
    connect(_list, &QListWidget::itemActivated, this, [this](QListWidgetItem *item)
    {
        if (!item)
        {
            return;
        }
        emit photoActivated(item->data(PathRole).toString());
    });
    connect(_list,
            &QListWidget::customContextMenuRequested,
            this,
            &PhotoStripWidget::showPhotoContextMenu);
}

PhotoStripWidget::~PhotoStripWidget()
{
    advanceThumbnailGeneration(false);
    const QSet<QFutureWatcher<ThumbnailResult> *> watchers = _thumbnailWatchers;
    for (QFutureWatcher<ThumbnailResult> *watcher : watchers)
    {
        disconnect(watcher, nullptr, this, nullptr);
        watcher->cancel();
    }
    for (QFutureWatcher<ThumbnailResult> *watcher : watchers)
    {
        watcher->waitForFinished();
    }
    _thumbnailWatchers.clear();
}

void PhotoStripWidget::setProjectPath(const QString &plascanPath)
{
    QString projectRootPath;
    const QString cleanProjectPath = plascanPath.trimmed();
    if (!cleanProjectPath.isEmpty())
    {
        projectRootPath =
            xjw::common::project::ProjectIO::projectRootFromPlascan(
                cleanProjectPath);
    }

    if (_projectRootPath == projectRootPath && _projectFilePath == cleanProjectPath)
    {
        return;
    }

    _projectFilePath = cleanProjectPath;
    _projectRootPath = projectRootPath;
    advanceThumbnailGeneration(true);
    clearPhotos();
}

void PhotoStripWidget::loadFromJson(const QJsonObject &meta)
{
    if (meta.isEmpty())
    {
        return;
    }

    advanceThumbnailGeneration(false);
    clearPhotos();

    const QJsonArray images = xjw::common::project::projectImageEntries(meta);
    const int total = images.size();
    if (total == 0)
    {
        emit imageLoadingFinished(true, tr("项目中没有影像"));
        return;
    }

    _imageListLoading = true;
    emit imageLoadingProgressChanged(tr("正在加载影像列表..."), 0, total);
    if (total <= AsyncListThreshold)
    {
        for (const QJsonValue &value : images)
        {
            appendImageEntry(value);
        }
        _imageListLoading = false;
        emit imageLoadingProgressChanged(tr("正在加载影像列表..."), total, total);
        emit imageLoadingFinished(true, tr("影像列表加载完成"));
        return;
    }

    _pendingImageEntries = images;
    _pendingImageIndex = 0;
    const quint64 generation = _thumbnailGeneration;
    QTimer::singleShot(0, this, [this, generation]()
    {
        processPendingImageBatch(generation);
    });
}

void PhotoStripWidget::appendImageEntry(const QJsonValue &value)
{
    QJsonObject entry;
    if (value.isObject())
    {
        entry = value.toObject();
    }
    else if (value.isString())
    {
        const QString imagePath = value.toString().trimmed();
        if (imagePath.isEmpty())
        {
            return;
        }
        entry.insert(QStringLiteral("path"), imagePath);
    }
    else
    {
        return;
    }

    QListWidgetItem *item = createItem(entry);
    if (!item)
    {
        return;
    }

    _list->addItem(item);
    const QString imagePath = item->data(PathRole).toString();
    const QString key = normalizedPath(imagePath);
    _itemsByPath[key].append(item);
    startThumbnailLoad(imagePath);
}

void PhotoStripWidget::processPendingImageBatch(quint64 generation)
{
    if (generation != _thumbnailGeneration || !_imageListLoading)
    {
        return;
    }

    const int total = _pendingImageEntries.size();
    const int end = std::min(_pendingImageIndex + ImageListBatchSize, total);
    while (_pendingImageIndex < end)
    {
        appendImageEntry(_pendingImageEntries.at(_pendingImageIndex));
        ++_pendingImageIndex;
    }
    emit imageLoadingProgressChanged(
        tr("正在加载影像列表..."), _pendingImageIndex, total);

    if (_pendingImageIndex >= total)
    {
        _pendingImageEntries = QJsonArray();
        _pendingImageIndex = 0;
        _imageListLoading = false;
        emit imageLoadingFinished(true, tr("影像列表加载完成"));
        return;
    }

    QTimer::singleShot(0, this, [this, generation]()
    {
        processPendingImageBatch(generation);
    });
}

void PhotoStripWidget::setCurrentPhoto(const QString &imagePath)
{
    if (!_list)
    {
        return;
    }

    const QString key = normalizedPath(imagePath);
    const QList<QListWidgetItem *> items = _itemsByPath.value(key);
    if (items.isEmpty() || !items.first())
    {
        _list->clearSelection();
        _list->setCurrentItem(nullptr);
        return;
    }

    QListWidgetItem *item = items.first();
    const QItemSelectionModel::SelectionFlags command = item->isSelected()
        ? QItemSelectionModel::NoUpdate
        : QItemSelectionModel::ClearAndSelect;
    _list->setCurrentItem(item, command);
    _list->scrollToItem(item, QAbstractItemView::PositionAtCenter);
}

QStringList PhotoStripWidget::selectedPhotoPaths() const
{
    QStringList paths;
    QSet<QString> seen;
    if (!_list)
    {
        return paths;
    }

    for (int row = 0; row < _list->count(); ++row)
    {
        QListWidgetItem *item = _list->item(row);
        if (!item || !item->isSelected())
        {
            continue;
        }

        const QString path = item->data(PathRole).toString().trimmed();
        const QString key = normalizedPath(path);
        if (!path.isEmpty() && !key.isEmpty() && !seen.contains(key))
        {
            seen.insert(key);
            paths.push_back(path);
        }
    }
    return paths;
}

void PhotoStripWidget::showPhotoContextMenu(const QPoint &position)
{
    if (!_list)
    {
        return;
    }

    QListWidgetItem *item = _list->itemAt(position);
    if (!item)
    {
        return;
    }

    const QItemSelectionModel::SelectionFlags command = item->isSelected()
        ? QItemSelectionModel::NoUpdate
        : QItemSelectionModel::ClearAndSelect;
    _list->setCurrentItem(item, command);
    emit photoSelected(item->data(PathRole).toString());

    const QStringList imagePaths = selectedPhotoPaths();
    if (imagePaths.isEmpty())
    {
        return;
    }

    auto *menu = new QMenu(this);
    menu->setAttribute(Qt::WA_DeleteOnClose);
    QAction *generateAction = menu->addAction(tr("生成蒙版..."));
    connect(generateAction, &QAction::triggered, this, [this, menu, imagePaths]()
    {
        menu->close();
        emit generateMaskRequested(imagePaths);
    });
    menu->popup(_list->viewport()->mapToGlobal(position));
}

void PhotoStripWidget::clearPhotos()
{
    _itemsByPath.clear();
    if (_list)
    {
        _list->clear();
    }
}

QListWidgetItem *PhotoStripWidget::createItem(const QJsonObject &entry)
{
    const QString imagePath = resolveImagePath(entry.value(QStringLiteral("path")).toString());
    if (imagePath.isEmpty())
    {
        return nullptr;
    }

    const QString key = normalizedPath(imagePath);
    auto *item = new QListWidgetItem(displayNameForEntry(entry, imagePath));
    item->setData(PathRole, imagePath);
    item->setTextAlignment(Qt::AlignHCenter);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);

    const QIcon cachedIcon = _thumbnailCache.value(key);
    item->setIcon(cachedIcon.isNull() ? placeholderPhotoIcon() : cachedIcon);

    const QString alignedText = isAlignedEntry(entry) ? tr("已对齐") : tr("未对齐");
    item->setToolTip(tr("%1\n状态: %2").arg(imagePath, alignedText));
    return item;
}

void PhotoStripWidget::startThumbnailLoad(const QString &imagePath)
{
    const QString resolvedPath = resolveImagePath(imagePath);
    const QString key = normalizedPath(resolvedPath);
    const quint64 generation = _thumbnailGeneration;
    if (key.isEmpty() || _thumbnailCache.contains(key)
        || _thumbnailLoadsInFlight.value(key, 0) == generation
        || _queuedThumbnailKeys.contains(key))
    {
        return;
    }

    _pendingThumbnailPaths.enqueue(resolvedPath);
    _queuedThumbnailKeys.insert(key);
    pumpThumbnailLoads();
}

void PhotoStripWidget::pumpThumbnailLoads()
{
    const int maximumLoads = std::clamp(QThread::idealThreadCount(), 2, 8);
    while (_activeThumbnailLoads < maximumLoads && !_pendingThumbnailPaths.isEmpty())
    {
        const QString resolvedPath = _pendingThumbnailPaths.dequeue();
        const QString key = normalizedPath(resolvedPath);
        _queuedThumbnailKeys.remove(key);
        if (key.isEmpty() || _thumbnailCache.contains(key)
            || _thumbnailLoadsInFlight.value(key, 0) == _thumbnailGeneration)
        {
            continue;
        }

        const quint64 generation = _thumbnailGeneration;
        _thumbnailLoadsInFlight.insert(key, generation);
        auto *watcher = new QFutureWatcher<ThumbnailResult>(this);
        _thumbnailWatchers.insert(watcher);
        ++_activeThumbnailLoads;
        const QString projectPath = _projectFilePath;
        connect(watcher, &QFutureWatcher<ThumbnailResult>::finished, this,
                [this, watcher, key, generation, projectPath]()
        {
            if (generation == _thumbnailGeneration && projectPath == _projectFilePath)
            {
                applyThumbnail(watcher->result(), generation, projectPath);
            }
            if (_thumbnailLoadsInFlight.value(key, 0) == generation)
            {
                _thumbnailLoadsInFlight.remove(key);
            }
            _thumbnailWatchers.remove(watcher);
            _activeThumbnailLoads = std::max(0, _activeThumbnailLoads - 1);
            watcher->deleteLater();
            pumpThumbnailLoads();
        });
        watcher->setFuture(
            QtConcurrent::run(&PhotoStripWidget::loadThumbnail, resolvedPath, projectPath));
    }
}

void PhotoStripWidget::applyThumbnail(const ThumbnailResult &result,
                                      quint64 generation,
                                      const QString &projectPath)
{
    if (generation != _thumbnailGeneration || projectPath != _projectFilePath)
    {
        return;
    }
    const QString key = normalizedPath(result.path);
    if (key.isEmpty() || !result.loaded || result.image.isNull())
    {
        return;
    }

    const QList<QListWidgetItem *> items = _itemsByPath.value(key);
    if (items.isEmpty())
    {
        return;
    }

    const QIcon icon(QPixmap::fromImage(result.image));
    _thumbnailCache.insert(key, icon);
    for (QListWidgetItem *item : items)
    {
        if (item)
        {
            item->setIcon(icon);
        }
    }
}

void PhotoStripWidget::advanceThumbnailGeneration(bool clearCache)
{
    const bool wasLoading = _imageListLoading;
    ++_thumbnailGeneration;
    _pendingImageEntries = QJsonArray();
    _pendingImageIndex = 0;
    _imageListLoading = false;
    _pendingThumbnailPaths.clear();
    _queuedThumbnailKeys.clear();
    if (clearCache)
    {
        _thumbnailCache.clear();
    }
    if (wasLoading)
    {
        emit imageLoadingFinished(false, tr("影像列表加载已停止"));
    }
}

QString PhotoStripWidget::resolveImagePath(const QString &imagePath) const
{
    const QString path = QDir::cleanPath(imagePath.trimmed());
    if (path.isEmpty())
    {
        return QString();
    }

    if (QFileInfo(path).isAbsolute())
    {
        return path;
    }

    if (!_projectRootPath.isEmpty())
    {
        return QDir::cleanPath(QDir(_projectRootPath).filePath(path));
    }

    const QString absolutePath = QFileInfo(path).absoluteFilePath();
    return absolutePath.isEmpty() ? path : QDir::cleanPath(absolutePath);
}

PhotoStripWidget::ThumbnailResult PhotoStripWidget::loadThumbnail(const QString &imagePath, const QString &projectPath)
{
    ThumbnailResult result;
    result.path = imagePath;
    result.image = xjw::gui::views::loadImageForDisplay(imagePath, projectPath);

    if (result.image.isNull())
    {
        return result;
    }

    result.image = result.image.scaled(QSize(ThumbWidth, ThumbHeight),
                                       Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation);
    result.loaded = true;
    return result;
}

QString PhotoStripWidget::normalizedPath(const QString &imagePath) const
{
    const QString path = resolveImagePath(imagePath);
    if (path.isEmpty())
    {
        return QString();
    }

    const QFileInfo info(path);
    const QString absolutePath = info.absoluteFilePath();
    return absolutePath.isEmpty() ? path : QDir::cleanPath(absolutePath);
}

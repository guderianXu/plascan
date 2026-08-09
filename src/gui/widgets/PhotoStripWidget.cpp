#include "PhotoStripWidget.h"

#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "project/ProjectIO.h"
#include "Logger.h"

#include "../views/LayerImageLoader.h"

#include <algorithm>
#include <exception>

#include <QAbstractItemView>
#include <QAction>
#include <QColor>
#include <QDir>
#include <QEvent>
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
#include <QScrollBar>
#include <QThread>
#include <QThreadPool>
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
constexpr int MaximumPendingThumbnailLoads = 64;
constexpr int MaximumThumbnailCacheEntries = 256;
constexpr int HiddenListFallbackCount = 8;

class ThumbnailLoadPool final : public QThreadPool
{
public:
    ThumbnailLoadPool()
    {
        setMaxThreadCount(std::clamp(QThread::idealThreadCount(), 2, 8));
        setExpiryTimeout(30'000);
    }
};

QThreadPool *thumbnailLoadPool()
{
    static ThumbnailLoadPool pool;
    return &pool;
}

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
    _thumbnailCancellation = std::make_shared<std::atomic_bool>(false);
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
    connect(_list->verticalScrollBar(), &QScrollBar::valueChanged,
            this, [this]()
    {
        scheduleVisibleThumbnailLoads();
    });
    connect(_list->verticalScrollBar(), &QScrollBar::rangeChanged,
            this, [this]()
    {
        scheduleVisibleThumbnailLoads();
    });
    _list->viewport()->installEventFilter(this);
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
    _thumbnailWatchers.clear();
}

bool PhotoStripWidget::eventFilter(QObject *watched, QEvent *event)
{
    if (_list && watched == _list->viewport()
        && (event->type() == QEvent::Resize || event->type() == QEvent::Show))
    {
        scheduleVisibleThumbnailLoads();
    }
    return QWidget::eventFilter(watched, event);
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
        scheduleVisibleThumbnailLoads();
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
    scheduleVisibleThumbnailLoads();

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
    scheduleVisibleThumbnailLoads();
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
    ++_thumbnailVisibilityGeneration;
    _pendingThumbnailRequests.clear();
    _queuedThumbnailKeys.clear();
    _desiredThumbnailKeys.clear();
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

    const QString key = thumbnailCacheKey(imagePath);
    auto *item = new QListWidgetItem(displayNameForEntry(entry, imagePath));
    item->setData(PathRole, imagePath);
    item->setTextAlignment(Qt::AlignHCenter);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);

    const QIcon cachedIcon = cachedThumbnail(key);
    item->setIcon(cachedIcon.isNull() ? placeholderPhotoIcon() : cachedIcon);

    const QString alignedText = isAlignedEntry(entry) ? tr("已对齐") : tr("未对齐");
    item->setToolTip(tr("%1\n状态: %2").arg(imagePath, alignedText));
    return item;
}

void PhotoStripWidget::scheduleVisibleThumbnailLoads()
{
    if (_thumbnailRefreshScheduled)
    {
        return;
    }

    _thumbnailRefreshScheduled = true;
    QTimer::singleShot(0, this, [this]()
    {
        _thumbnailRefreshScheduled = false;
        refreshVisibleThumbnailLoads();
    });
}

void PhotoStripWidget::refreshVisibleThumbnailLoads()
{
    ++_thumbnailVisibilityGeneration;
    _pendingThumbnailRequests.clear();
    _queuedThumbnailKeys.clear();
    _desiredThumbnailKeys.clear();

    if (!_list || _list->count() == 0)
    {
        return;
    }

    QStringList paths;
    QSet<QString> keys;
    const QRect visible_rect = _list->viewport()->rect();
    appendThumbnailCandidates(visible_rect, &paths, &keys);

    if (paths.size() < MaximumPendingThumbnailLoads)
    {
        const int prefetch_height = std::max(visible_rect.height(), GridHeight);
        const QRect prefetch_rect = visible_rect.adjusted(
            0, -prefetch_height, 0, prefetch_height);
        appendThumbnailCandidates(prefetch_rect, &paths, &keys);
    }

    _desiredThumbnailKeys = keys;
    const quint64 visibility_generation = _thumbnailVisibilityGeneration;
    for (const QString &path : paths)
    {
        startThumbnailLoad(path, visibility_generation);
    }
    pumpThumbnailLoads();
}

void PhotoStripWidget::appendThumbnailCandidates(const QRect &viewport_rect,
                                                 QStringList *paths,
                                                 QSet<QString> *keys) const
{
    if (!_list || !paths || !keys || paths->size() >= MaximumPendingThumbnailLoads)
    {
        return;
    }

    const int item_count = _list->count();
    auto item_rect = [this](int row)
    {
        QListWidgetItem *item = _list->item(row);
        return item ? _list->visualItemRect(item) : QRect();
    };

    int first = 0;
    int last = std::min(item_count, HiddenListFallbackCount);
    const QRect first_rect = item_rect(0);
    const QRect final_rect = item_rect(item_count - 1);
    if (first_rect.isValid() && final_rect.isValid())
    {
        int low = 0;
        int high = item_count;
        while (low < high)
        {
            const int middle = low + (high - low) / 2;
            if (item_rect(middle).bottom() < viewport_rect.top())
            {
                low = middle + 1;
            }
            else
            {
                high = middle;
            }
        }
        first = low;

        low = first;
        high = item_count;
        while (low < high)
        {
            const int middle = low + (high - low) / 2;
            if (item_rect(middle).top() <= viewport_rect.bottom())
            {
                low = middle + 1;
            }
            else
            {
                high = middle;
            }
        }
        last = low;
    }

    for (int row = first;
         row < last && paths->size() < MaximumPendingThumbnailLoads;
         ++row)
    {
        QListWidgetItem *item = _list->item(row);
        if (!item)
        {
            continue;
        }
        const QString path = item->data(PathRole).toString();
        const QString key = normalizedPath(path);
        if (key.isEmpty() || keys->contains(key))
        {
            continue;
        }
        keys->insert(key);
        paths->append(path);
    }
}

void PhotoStripWidget::startThumbnailLoad(const QString &imagePath,
                                          quint64 visibility_generation)
{
    const QString resolvedPath = resolveImagePath(imagePath);
    const QString key = normalizedPath(resolvedPath);
    const QString cache_key = thumbnailCacheKey(resolvedPath);
    const quint64 generation = _thumbnailGeneration;
    if (key.isEmpty() || cache_key.isEmpty())
    {
        return;
    }
    if (_thumbnailCache.contains(cache_key))
    {
        const QIcon icon = cachedThumbnail(cache_key);
        for (QListWidgetItem *item : _itemsByPath.value(key))
        {
            if (item)
            {
                item->setIcon(icon);
            }
        }
        return;
    }
    if (_thumbnailLoadsInFlight.value(key, 0) == generation
        || _queuedThumbnailKeys.contains(key))
    {
        return;
    }

    _pendingThumbnailRequests.enqueue({resolvedPath,
                                       cache_key,
                                       _thumbnailGeneration,
                                       visibility_generation});
    _queuedThumbnailKeys.insert(key);
}

void PhotoStripWidget::pumpThumbnailLoads()
{
    const int maximumLoads = thumbnailLoadPool()->maxThreadCount();
    while (_activeThumbnailLoads < maximumLoads
           && !_pendingThumbnailRequests.isEmpty())
    {
        const ThumbnailRequest request = _pendingThumbnailRequests.dequeue();
        const QString resolvedPath = request.path;
        const QString key = normalizedPath(resolvedPath);
        const QString cache_key = request.cacheKey;
        _queuedThumbnailKeys.remove(key);
        if (request.projectGeneration != _thumbnailGeneration
            || request.visibilityGeneration != _thumbnailVisibilityGeneration)
        {
            continue;
        }
        if (key.isEmpty() || cache_key.isEmpty() || _thumbnailCache.contains(cache_key)
            || _thumbnailLoadsInFlight.value(key, 0) == _thumbnailGeneration)
        {
            continue;
        }

        const quint64 generation = _thumbnailGeneration;
        _thumbnailLoadsInFlight.insert(key, generation);
        auto *watcher = new QFutureWatcher<ThumbnailResult>();
        _thumbnailWatchers.insert(watcher);
        ++_activeThumbnailLoads;
        const QString projectPath = _projectFilePath;
        const auto cancellation = _thumbnailCancellation;
        const quint64 visibility_generation = request.visibilityGeneration;
        connect(watcher, &QFutureWatcher<ThumbnailResult>::finished,
                watcher, &QObject::deleteLater);
        connect(watcher, &QFutureWatcher<ThumbnailResult>::finished, this,
                [this, watcher, key, generation, visibility_generation, projectPath]()
        {
            const bool is_latest_request =
                visibility_generation == _thumbnailVisibilityGeneration;
            const bool is_still_wanted = _desiredThumbnailKeys.contains(key);
            if (generation == _thumbnailGeneration
                && projectPath == _projectFilePath
                && (is_latest_request || is_still_wanted))
            {
                ThumbnailResult result;
                try
                {
                    result = watcher->result();
                }
                catch (const std::exception &exception)
                {
                    result.error = QString::fromUtf8(exception.what());
                }
                catch (...)
                {
                    result.error = tr("未知后台异常");
                }
                if (!result.error.isEmpty())
                {
                    LOG_WARN(QStringLiteral("缩略图后台加载失败：%1（%2）")
                                 .arg(key, result.error));
                }
                applyThumbnail(result, generation, projectPath);
            }
            if (_thumbnailLoadsInFlight.value(key, 0) == generation)
            {
                _thumbnailLoadsInFlight.remove(key);
            }
            _thumbnailWatchers.remove(watcher);
            _activeThumbnailLoads = std::max(0, _activeThumbnailLoads - 1);
            pumpThumbnailLoads();
        });
        watcher->setFuture(QtConcurrent::run(
            thumbnailLoadPool(),
            &PhotoStripWidget::loadThumbnail,
            resolvedPath,
            projectPath,
            cache_key,
            cancellation));
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
    if (result.cacheKey.isEmpty() || thumbnailCacheKey(result.path) != result.cacheKey)
    {
        scheduleVisibleThumbnailLoads();
        return;
    }

    const QList<QListWidgetItem *> items = _itemsByPath.value(key);
    if (items.isEmpty())
    {
        return;
    }

    const QIcon icon(QPixmap::fromImage(result.image));
    insertThumbnailCache(result.cacheKey, icon);
    for (QListWidgetItem *item : items)
    {
        if (item)
        {
            item->setIcon(icon);
        }
    }
}

QIcon PhotoStripWidget::cachedThumbnail(const QString &key)
{
    const auto cached = _thumbnailCache.constFind(key);
    if (cached == _thumbnailCache.cend())
    {
        return QIcon();
    }
    _thumbnailCacheLru.removeAll(key);
    _thumbnailCacheLru.enqueue(key);
    return cached.value();
}

void PhotoStripWidget::insertThumbnailCache(const QString &key, const QIcon &icon)
{
    if (key.isEmpty() || icon.isNull())
    {
        return;
    }

    _thumbnailCache.insert(key, icon);
    _thumbnailCacheLru.removeAll(key);
    _thumbnailCacheLru.enqueue(key);
    while (_thumbnailCacheLru.size() > MaximumThumbnailCacheEntries)
    {
        const QString expired_key = _thumbnailCacheLru.dequeue();
        _thumbnailCache.remove(expired_key);
        resetItemIcons(expired_key);
    }
}

void PhotoStripWidget::resetItemIcons(const QString &key)
{
    const QString path_key = key.section(QLatin1Char('\n'), 0, 0);
    if (thumbnailCacheKey(path_key) != key)
    {
        return;
    }
    const QList<QListWidgetItem *> items = _itemsByPath.value(path_key);
    for (QListWidgetItem *item : items)
    {
        if (item)
        {
            item->setIcon(placeholderPhotoIcon());
        }
    }
}

void PhotoStripWidget::advanceThumbnailGeneration(bool clearCache)
{
    const bool wasLoading = _imageListLoading;
    ++_thumbnailGeneration;
    if (_thumbnailCancellation)
    {
        _thumbnailCancellation->store(true, std::memory_order_relaxed);
    }
    _thumbnailCancellation = std::make_shared<std::atomic_bool>(false);
    _pendingImageEntries = QJsonArray();
    _pendingImageIndex = 0;
    _imageListLoading = false;
    ++_thumbnailVisibilityGeneration;
    _pendingThumbnailRequests.clear();
    _queuedThumbnailKeys.clear();
    _desiredThumbnailKeys.clear();
    if (clearCache)
    {
        _thumbnailCache.clear();
        _thumbnailCacheLru.clear();
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

PhotoStripWidget::ThumbnailResult PhotoStripWidget::loadThumbnail(
    const QString &imagePath,
    const QString &projectPath,
    const QString &cacheKey,
    const std::shared_ptr<std::atomic_bool> &cancellation)
{
    ThumbnailResult result;
    result.path = imagePath;
    result.cacheKey = cacheKey;
    if (cancellation && cancellation->load(std::memory_order_relaxed))
    {
        return result;
    }
    try
    {
        result.image = xjw::gui::views::loadImageForDisplay(
            imagePath, projectPath, QSize(ThumbWidth, ThumbHeight), nullptr);
    }
    catch (const std::exception &exception)
    {
        result.error = QString::fromUtf8(exception.what());
        return result;
    }
    catch (...)
    {
        result.error = QStringLiteral("未知后台异常");
        return result;
    }

    if ((cancellation && cancellation->load(std::memory_order_relaxed))
        || result.image.isNull())
    {
        return result;
    }

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

QString PhotoStripWidget::thumbnailCacheKey(const QString &imagePath) const
{
    const QString path = normalizedPath(imagePath);
    if (path.isEmpty())
    {
        return {};
    }
    const QFileInfo info(path);
    return QStringLiteral("%1\n%2\n%3")
        .arg(path)
        .arg(info.exists() ? info.lastModified().toMSecsSinceEpoch() : -1)
        .arg(info.exists() ? info.size() : -1);
}

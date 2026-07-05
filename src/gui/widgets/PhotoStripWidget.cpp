#include "PhotoStripWidget.h"

#include "ProjectSupportUtils.h"

#include <QAbstractItemView>
#include <QDir>
#include <QFileIconProvider>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QImageReader>
#include <QItemSelectionModel>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QListView>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPixmap>
#include <QSize>
#include <QVBoxLayout>
#include <QtConcurrent>

namespace
{
constexpr int PathRole = Qt::UserRole + 1;
constexpr int ThumbWidth = 132;
constexpr int ThumbHeight = 88;
constexpr int GridWidth = 220;
constexpr int GridHeight = 140;

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

QIcon fileIconForPath(const QString &imagePath)
{
    QFileIconProvider provider;
    const QIcon pathIcon = provider.icon(QFileInfo(imagePath));
    return pathIcon.isNull() ? provider.icon(QFileIconProvider::File) : pathIcon;
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
    _list->setSelectionMode(QAbstractItemView::SingleSelection);
    _list->setSelectionBehavior(QAbstractItemView::SelectItems);
    _list->setEditTriggers(QAbstractItemView::NoEditTriggers);
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
        if (!item)
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
}

void PhotoStripWidget::setProjectPath(const QString &plascanPath)
{
    QString projectRootPath;
    const QString cleanProjectPath = plascanPath.trimmed();
    if (!cleanProjectPath.isEmpty())
    {
        projectRootPath = QDir::cleanPath(QFileInfo(cleanProjectPath).absolutePath());
    }

    if (_projectRootPath == projectRootPath)
    {
        return;
    }

    _projectRootPath = projectRootPath;
    _thumbnailCache.clear();
    _thumbnailLoadsInFlight.clear();
    clearPhotos();
}

void PhotoStripWidget::loadFromJson(const QJsonObject &meta)
{
    if (meta.isEmpty())
    {
        return;
    }

    clearPhotos();

    const QJsonArray images = xjw::gui::project::projectImageEntries(meta);
    for (const QJsonValue &value : images)
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
                continue;
            }
            entry.insert(QStringLiteral("path"), imagePath);
        }
        else
        {
            continue;
        }

        QListWidgetItem *item = createItem(entry);
        if (!item)
        {
            continue;
        }

        _list->addItem(item);

        const QString imagePath = item->data(PathRole).toString();
        const QString key = normalizedPath(imagePath);
        _itemsByPath[key].append(item);
        startThumbnailLoad(imagePath);
    }
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
    _list->setCurrentItem(item, QItemSelectionModel::ClearAndSelect);
    _list->scrollToItem(item, QAbstractItemView::PositionAtCenter);
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
    item->setIcon(cachedIcon.isNull() ? fileIconForPath(imagePath) : cachedIcon);

    const QString alignedText = isAlignedEntry(entry) ? tr("已对齐") : tr("未对齐");
    item->setToolTip(tr("%1\n状态: %2").arg(imagePath, alignedText));
    return item;
}

void PhotoStripWidget::startThumbnailLoad(const QString &imagePath)
{
    const QString resolvedPath = resolveImagePath(imagePath);
    const QString key = normalizedPath(resolvedPath);
    if (key.isEmpty() || _thumbnailCache.contains(key) || _thumbnailLoadsInFlight.contains(key))
    {
        return;
    }

    _thumbnailLoadsInFlight.insert(key);
    auto *watcher = new QFutureWatcher<ThumbnailResult>(this);
    connect(watcher, &QFutureWatcher<ThumbnailResult>::finished, this, [this, watcher, key]()
    {
        applyThumbnail(watcher->result());
        _thumbnailLoadsInFlight.remove(key);
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run(&PhotoStripWidget::loadThumbnail, resolvedPath));
}

void PhotoStripWidget::applyThumbnail(const ThumbnailResult &result)
{
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

PhotoStripWidget::ThumbnailResult PhotoStripWidget::loadThumbnail(const QString &imagePath)
{
    ThumbnailResult result;
    result.path = imagePath;

    QImageReader reader(imagePath);
    reader.setAutoTransform(true);

    const QSize originalSize = reader.size();
    if (originalSize.isValid())
    {
        reader.setScaledSize(originalSize.scaled(QSize(ThumbWidth, ThumbHeight), Qt::KeepAspectRatio));
        result.image = reader.read();
    }
    else
    {
        result.image.load(imagePath);
    }

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

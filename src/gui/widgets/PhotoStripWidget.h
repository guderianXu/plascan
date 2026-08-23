#pragma once

#include <atomic>
#include <memory>

#include <QHash>
#include <QFutureWatcher>
#include <QIcon>
#include <QImage>
#include <QJsonObject>
#include <QJsonArray>
#include <QList>
#include <QQueue>
#include <QRect>
#include <QSet>
#include <QStringList>
#include <QWidget>

class QListWidget;
class QListWidgetItem;
class QEvent;

class PhotoStripWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PhotoStripWidget(QWidget *parent = nullptr);
    ~PhotoStripWidget() override;
    void setProjectPath(const QString &plascanPath);

public slots:
    void loadFromJson(const QJsonObject &meta);
    void setCurrentPhoto(const QString &imagePath);
    void clearPhotos();

signals:
    void photoSelected(const QString &imagePath);
    void photoActivated(const QString &imagePath);
    void generateMaskRequested(const QStringList &imagePaths);
    void clearMasksRequested(const QStringList &imagePaths);
    void imageLoadingProgressChanged(const QString &stage, int done, int total);
    void imageLoadingFinished(bool success, const QString &message);

private slots:
    void showPhotoContextMenu(const QPoint &position);

private:
    bool eventFilter(QObject *watched, QEvent *event) override;

    struct ThumbnailResult
    {
        QString path;
        QString cacheKey;
        QImage image;
        QString error;
        bool loaded = false;
    };

    struct ThumbnailRequest
    {
        QString path;
        QString cacheKey;
        quint64 projectGeneration = 0;
        quint64 visibilityGeneration = 0;
    };

    QListWidgetItem *createItem(const QJsonObject &entry);
    void appendImageEntry(const QJsonValue &value);
    void processPendingImageBatch(quint64 generation);
    void scheduleVisibleThumbnailLoads();
    void refreshVisibleThumbnailLoads();
    void appendThumbnailCandidates(const QRect &viewport_rect,
                                   QStringList *paths,
                                   QSet<QString> *keys) const;
    void startThumbnailLoad(const QString &imagePath,
                            quint64 visibility_generation);
    void pumpThumbnailLoads();
    void applyThumbnail(const ThumbnailResult &result,
                        quint64 generation,
                        const QString &projectPath);
    QIcon cachedThumbnail(const QString &key);
    void insertThumbnailCache(const QString &key, const QIcon &icon);
    void resetItemIcons(const QString &key);
    void advanceThumbnailGeneration(bool clearCache);
    QStringList selectedPhotoPaths() const;
    bool selectedPhotosHaveMasks() const;
    QString resolveImagePath(const QString &imagePath) const;
    QString normalizedPath(const QString &imagePath) const;
    QString thumbnailCacheKey(const QString &imagePath) const;
    static ThumbnailResult loadThumbnail(
        const QString &imagePath,
        const QString &projectPath,
        const QString &cacheKey,
        const std::shared_ptr<std::atomic_bool> &cancellation);

    QListWidget *_list = nullptr;
    QString _projectFilePath;
    QString _projectRootPath;
    QHash<QString, QList<QListWidgetItem *>> _itemsByPath;
    QHash<QString, QIcon> _thumbnailCache;
    QQueue<QString> _thumbnailCacheLru;
    QHash<QString, quint64> _thumbnailLoadsInFlight;
    QSet<QFutureWatcher<ThumbnailResult> *> _thumbnailWatchers;
    QQueue<ThumbnailRequest> _pendingThumbnailRequests;
    QSet<QString> _queuedThumbnailKeys;
    QSet<QString> _desiredThumbnailKeys;
    QJsonArray _pendingImageEntries;
    int _pendingImageIndex = 0;
    int _activeThumbnailLoads = 0;
    bool _imageListLoading = false;
    bool _thumbnailRefreshScheduled = false;
    quint64 _thumbnailGeneration{};
    quint64 _thumbnailVisibilityGeneration{};
    std::shared_ptr<std::atomic_bool> _thumbnailCancellation;
};

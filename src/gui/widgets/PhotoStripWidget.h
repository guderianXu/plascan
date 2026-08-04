#pragma once

#include <QHash>
#include <QFutureWatcher>
#include <QIcon>
#include <QImage>
#include <QJsonObject>
#include <QJsonArray>
#include <QList>
#include <QQueue>
#include <QSet>
#include <QStringList>
#include <QWidget>

class QListWidget;
class QListWidgetItem;

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
    void imageLoadingProgressChanged(const QString &stage, int done, int total);
    void imageLoadingFinished(bool success, const QString &message);

private slots:
    void showPhotoContextMenu(const QPoint &position);

private:
    struct ThumbnailResult
    {
        QString path;
        QImage image;
        bool loaded = false;
    };

    QListWidgetItem *createItem(const QJsonObject &entry);
    void appendImageEntry(const QJsonValue &value);
    void processPendingImageBatch(quint64 generation);
    void startThumbnailLoad(const QString &imagePath);
    void pumpThumbnailLoads();
    void applyThumbnail(const ThumbnailResult &result, quint64 generation, const QString &projectPath);
    void advanceThumbnailGeneration(bool clearCache);
    QStringList selectedPhotoPaths() const;
    QString resolveImagePath(const QString &imagePath) const;
    QString normalizedPath(const QString &imagePath) const;
    static ThumbnailResult loadThumbnail(const QString &imagePath, const QString &projectPath);

    QListWidget *_list = nullptr;
    QString _projectFilePath;
    QString _projectRootPath;
    QHash<QString, QList<QListWidgetItem *>> _itemsByPath;
    QHash<QString, QIcon> _thumbnailCache;
    QHash<QString, quint64> _thumbnailLoadsInFlight;
    QSet<QFutureWatcher<ThumbnailResult> *> _thumbnailWatchers;
    QQueue<QString> _pendingThumbnailPaths;
    QSet<QString> _queuedThumbnailKeys;
    QJsonArray _pendingImageEntries;
    int _pendingImageIndex = 0;
    int _activeThumbnailLoads = 0;
    bool _imageListLoading = false;
    quint64 _thumbnailGeneration{};
};

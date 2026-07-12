#pragma once

#include <QHash>
#include <QIcon>
#include <QImage>
#include <QJsonObject>
#include <QList>
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
    void setProjectPath(const QString &plascanPath);

public slots:
    void loadFromJson(const QJsonObject &meta);
    void setCurrentPhoto(const QString &imagePath);
    void clearPhotos();

signals:
    void photoSelected(const QString &imagePath);
    void photoActivated(const QString &imagePath);
    void generateMaskRequested(const QStringList &imagePaths);

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
    void startThumbnailLoad(const QString &imagePath);
    void applyThumbnail(const ThumbnailResult &result);
    QStringList selectedPhotoPaths() const;
    QString resolveImagePath(const QString &imagePath) const;
    QString normalizedPath(const QString &imagePath) const;
    static ThumbnailResult loadThumbnail(const QString &imagePath, const QString &projectPath);

    QListWidget *_list = nullptr;
    QString _projectFilePath;
    QString _projectRootPath;
    QHash<QString, QList<QListWidgetItem *>> _itemsByPath;
    QHash<QString, QIcon> _thumbnailCache;
    QSet<QString> _thumbnailLoadsInFlight;
};

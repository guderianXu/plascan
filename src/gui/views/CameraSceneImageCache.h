#pragma once

#include <QHash>
#include <QImage>
#include <QQueue>
#include <QSet>
#include <QString>

namespace xjw::gui::camera_scene
{

class CameraSceneImageCache
{
public:
    explicit CameraSceneImageCache(
        qint64 fullImageByteLimit = 128LL * 1024LL * 1024LL);

    QImage image(const QString &key) const;
    bool contains(const QString &key) const;

    bool hasFailure(const QString &key) const;
    bool markFailure(const QString &key);
    bool clearFailure(const QString &key);

    void store(const QString &key,
               const QImage &image,
               bool retainAsFullImage,
               const QString &protectedKey = {});
    void remove(const QString &key);
    void clearWithPrefix(const QString &prefix);
    void clear();

private:
    QHash<QString, QImage> _images;
    QQueue<QString> _fullImageLru;
    QSet<QString> _failures;
    qint64 _fullImageBytes = 0;
    qint64 _fullImageByteLimit;
};

} // namespace xjw::gui::camera_scene

#include "CameraSceneImageCache.h"

namespace xjw::gui::camera_scene
{

CameraSceneImageCache::CameraSceneImageCache(qint64 fullImageByteLimit)
    : _fullImageByteLimit(qMax<qint64>(0, fullImageByteLimit))
{
}

QImage CameraSceneImageCache::image(const QString &key) const
{
    return _images.value(key);
}

bool CameraSceneImageCache::contains(const QString &key) const
{
    return _images.contains(key);
}

bool CameraSceneImageCache::hasFailure(const QString &key) const
{
    return _failures.contains(key);
}

bool CameraSceneImageCache::markFailure(const QString &key)
{
    if (key.isEmpty())
    {
        return false;
    }
    const bool firstFailure = !_failures.contains(key);
    _failures.insert(key);
    return firstFailure;
}

bool CameraSceneImageCache::clearFailure(const QString &key)
{
    return _failures.remove(key);
}

void CameraSceneImageCache::store(const QString &key,
                                  const QImage &image,
                                  bool retainAsFullImage,
                                  const QString &protectedKey)
{
    if (key.isEmpty() || image.isNull())
    {
        return;
    }

    if (retainAsFullImage)
    {
        const auto existing = _images.constFind(key);
        if (existing != _images.cend())
        {
            _fullImageBytes -= static_cast<qint64>(existing.value().sizeInBytes());
        }
        _fullImageLru.removeAll(key);
        _fullImageLru.enqueue(key);
        _fullImageBytes += static_cast<qint64>(image.sizeInBytes());
    }
    _images.insert(key, image);

    int remainingCandidates = _fullImageLru.size();
    while (retainAsFullImage
           && _fullImageBytes > _fullImageByteLimit
           && _fullImageLru.size() > 1
           && remainingCandidates-- > 0)
    {
        const QString candidate = _fullImageLru.dequeue();
        if (candidate == key || candidate == protectedKey)
        {
            _fullImageLru.enqueue(candidate);
            continue;
        }
        const auto cached = _images.find(candidate);
        if (cached != _images.end())
        {
            _fullImageBytes -= static_cast<qint64>(cached.value().sizeInBytes());
            _images.erase(cached);
        }
    }
}

void CameraSceneImageCache::remove(const QString &key)
{
    const auto existing = _images.find(key);
    if (existing != _images.end())
    {
        if (_fullImageLru.removeAll(key) > 0)
        {
            _fullImageBytes -= static_cast<qint64>(existing.value().sizeInBytes());
        }
        _images.erase(existing);
    }
}

void CameraSceneImageCache::clearWithPrefix(const QString &prefix)
{
    for (auto it = _images.begin(); it != _images.end();)
    {
        if (it.key().startsWith(prefix))
        {
            if (_fullImageLru.removeAll(it.key()) > 0)
            {
                _fullImageBytes -= static_cast<qint64>(it.value().sizeInBytes());
            }
            it = _images.erase(it);
        }
        else
        {
            ++it;
        }
    }
    for (auto it = _failures.begin(); it != _failures.end();)
    {
        if (it->startsWith(prefix))
        {
            it = _failures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void CameraSceneImageCache::clear()
{
    _images.clear();
    _fullImageLru.clear();
    _failures.clear();
    _fullImageBytes = 0;
}

} // namespace xjw::gui::camera_scene

#include "MatchPhotosFeatureCache.h"

#include <QDir>
#include <QFileInfo>

#include <mutex>

namespace xjw::matchphotos
{

QString MatchPhotosFeatureCache::cacheKey(const QString &imagePath)
{
    const QFileInfo info(imagePath);
    const QString absolute = info.isAbsolute()
        ? info.absoluteFilePath()
        : QFileInfo(QDir::current(), imagePath).absoluteFilePath();
    return QDir::cleanPath(QDir::fromNativeSeparators(absolute)).toLower();
}

void MatchPhotosFeatureCache::insert(
    const QString &imagePath,
    std::shared_ptr<const image_matching::FeatureSet> features)
{
    if (imagePath.trimmed().isEmpty() || !features)
    {
        return;
    }

    std::unique_lock lock(_mutex);
    _features.insert(cacheKey(imagePath), std::move(features));
}

std::shared_ptr<const image_matching::FeatureSet> MatchPhotosFeatureCache::find(
    const QString &imagePath) const
{
    std::shared_lock lock(_mutex);
    const auto it = _features.constFind(cacheKey(imagePath));
    return it == _features.constEnd() ? nullptr : it.value();
}

bool MatchPhotosFeatureCache::contains(const QString &imagePath) const
{
    return static_cast<bool>(find(imagePath));
}

int MatchPhotosFeatureCache::imageCount() const
{
    std::shared_lock lock(_mutex);
    return _features.size();
}

std::size_t MatchPhotosFeatureCache::approximateBytes() const
{
    std::shared_lock lock(_mutex);
    std::size_t bytes = 0;
    for (auto it = _features.constBegin(); it != _features.constEnd(); ++it)
    {
        if (it.value())
        {
            bytes += it.value()->approximateBytes();
        }
    }
    return bytes;
}

void MatchPhotosFeatureCache::clear()
{
    std::unique_lock lock(_mutex);
    _features.clear();
}

} // namespace xjw::matchphotos

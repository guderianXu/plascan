#include "DepthOverlayController.h"

#include "GuiTaskRunner.h"
#include "LayerImageLoader.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>

#include <algorithm>
#include <limits>

namespace xjw::gui::widgets
{
namespace
{

QString fileIdentity(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return QStringLiteral("<none>");
    }
    const QFileInfo file_info(QDir::fromNativeSeparators(path));
    return QStringLiteral("%1|%2|%3")
        .arg(QDir::cleanPath(file_info.absoluteFilePath()))
        .arg(file_info.size())
        .arg(file_info.lastModified().toMSecsSinceEpoch());
}

int levelNumber(views::DepthOverlayLevel level)
{
    return static_cast<int>(level);
}

} // namespace

DepthOverlayController::DepthOverlayController(QObject *parent)
    : QObject(parent)
{
}

bool DepthOverlayController::setProjectMetadata(
    const QJsonObject &metadata,
    const QString &observed_image_path)
{
    if (_metadata == metadata)
    {
        return false;
    }

    const bool observed_record_changed = observed_image_path.trimmed().isEmpty()
        || views::resolveDepthOverlayRecord(_metadata, observed_image_path)
            != views::resolveDepthOverlayRecord(metadata, observed_image_path);
    _metadata = metadata;
    if (observed_record_changed)
    {
        cancelPending();
        _cache.clear();
        _cacheBytes = 0;
    }
    return observed_record_changed;
}

void DepthOverlayController::setProjectPath(const QString &project_path)
{
    const QString normalized_path = QDir::cleanPath(project_path);
    if (_projectPath == normalized_path)
    {
        return;
    }
    _projectPath = normalized_path;
    cancelPending();
    _cache.clear();
    _cacheBytes = 0;
}

void DepthOverlayController::request(
    const QString &image_path,
    views::DepthOverlayLevel level,
    const views::DepthOverlayRenderOptions &options)
{
    const quint64 generation = ++_requestGeneration;
    const views::DepthOverlayAvailability availability = artifactAvailability(
        image_path, level);
    if (!availability.available)
    {
        emit overlayFailed(image_path, availability.reason);
        return;
    }
    const auto artifact = views::resolveDepthOverlayArtifact(_metadata, image_path, level);
    if (!artifact)
    {
        emit overlayFailed(image_path, QStringLiteral("当前照片没有所选级别的深度图"));
        return;
    }
    const views::DepthOverlayArtifact resolved_artifact =
        views::resolveDepthOverlayArtifactPaths(*artifact, _projectPath);

    const QString cache_key = cacheKeyForArtifact(
        resolved_artifact, level, options, image_path);
    QImage overlay;
    QImage intensity_base;
    if (readCache(cache_key, &overlay, &intensity_base))
    {
        emit overlayReady(image_path, overlay, intensity_base);
        return;
    }

    const QString project_path = _projectPath;
    xjw::gui::tasks::runGuarded(
        this,
        [artifact = resolved_artifact, options, image_path, project_path]()
        {
            QImage source_image;
            if (options.showIntensity)
            {
                source_image = views::loadImageForDisplay(image_path, project_path);
            }
            return views::loadDepthOverlay(artifact, options, source_image);
        },
        [generation, image_path, cache_key](DepthOverlayController *self,
                                            views::DepthOverlayRenderResult result)
        {
            if (!self->isCurrentGeneration(generation))
            {
                return;
            }
            if (!result.errorMessage.isEmpty() || result.overlay.isNull())
            {
                emit self->overlayFailed(
                    image_path,
                    result.errorMessage.isEmpty()
                        ? QStringLiteral("深度图叠加生成失败")
                        : result.errorMessage);
                return;
            }
            self->writeCache(cache_key, result);
            emit self->overlayReady(image_path, result.overlay, result.intensityBase);
        });
}

void DepthOverlayController::cancelPending()
{
    ++_requestGeneration;
}

bool DepthOverlayController::anyArtifactAvailable(const QString &image_path) const
{
    for (const views::DepthOverlayLevel level : {
             views::DepthOverlayLevel::Final,
             views::DepthOverlayLevel::Level1,
             views::DepthOverlayLevel::Level2,
             views::DepthOverlayLevel::Level3})
    {
        if (artifactAvailable(image_path, level))
        {
            return true;
        }
    }
    return false;
}

bool DepthOverlayController::artifactAvailable(
    const QString &image_path,
    views::DepthOverlayLevel level) const
{
    return artifactAvailability(image_path, level).available;
}

views::DepthOverlayAvailability DepthOverlayController::artifactAvailability(
    const QString &image_path,
    views::DepthOverlayLevel level) const
{
    return views::resolveDepthOverlayAvailability(
        _metadata, image_path, level, _projectPath);
}

bool DepthOverlayController::isCurrentGeneration(quint64 generation) const noexcept
{
    return generation == _requestGeneration;
}

QString DepthOverlayController::cacheKeyForArtifact(
    const views::DepthOverlayArtifact &artifact,
    views::DepthOverlayLevel level,
    const views::DepthOverlayRenderOptions &options,
    const QString &source_image_path)
{
    return QStringLiteral("%1\n%2\nlevel=%3\nopacity=%4\nintensity=%5\nsource=%6")
        .arg(fileIdentity(artifact.rawDepthPath),
             fileIdentity(artifact.validMaskPath))
        .arg(levelNumber(level))
        .arg(std::clamp(options.opacity, 0, 255))
        .arg(options.showIntensity ? 1 : 0)
        .arg(options.showIntensity ? fileIdentity(source_image_path) : QStringLiteral("<unused>"));
}

bool DepthOverlayController::readCache(
    const QString &key,
    QImage *overlay,
    QImage *intensity_base)
{
    auto iterator = _cache.find(key);
    if (iterator == _cache.end())
    {
        return false;
    }
    iterator->accessSerial = ++_accessSerial;
    if (overlay)
    {
        *overlay = iterator->overlay;
    }
    if (intensity_base)
    {
        *intensity_base = iterator->intensityBase;
    }
    return true;
}

void DepthOverlayController::writeCache(
    const QString &key,
    const views::DepthOverlayRenderResult &result)
{
    const qint64 bytes = result.overlay.sizeInBytes() + result.intensityBase.sizeInBytes();
    if (bytes <= 0 || bytes > MaximumCacheBytes)
    {
        return;
    }

    const auto existing = _cache.find(key);
    if (existing != _cache.end())
    {
        _cacheBytes -= existing->bytes;
        _cache.erase(existing);
    }
    evictFor(bytes);

    CacheEntry entry;
    entry.overlay = result.overlay;
    entry.intensityBase = result.intensityBase;
    entry.bytes = bytes;
    entry.accessSerial = ++_accessSerial;
    _cache.insert(key, entry);
    _cacheBytes += bytes;
}

void DepthOverlayController::evictFor(qint64 required_bytes)
{
    while (!_cache.isEmpty() && _cacheBytes + required_bytes > MaximumCacheBytes)
    {
        auto oldest = _cache.begin();
        quint64 oldest_serial = std::numeric_limits<quint64>::max();
        for (auto iterator = _cache.begin(); iterator != _cache.end(); ++iterator)
        {
            if (iterator->accessSerial < oldest_serial)
            {
                oldest = iterator;
                oldest_serial = iterator->accessSerial;
            }
        }
        _cacheBytes -= oldest->bytes;
        _cache.erase(oldest);
    }
}

} // namespace xjw::gui::widgets

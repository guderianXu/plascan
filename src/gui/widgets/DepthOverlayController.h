#pragma once

#include "DepthOverlayData.h"

#include <QHash>
#include <QImage>
#include <QJsonObject>
#include <QObject>
#include <QString>

namespace xjw::gui::widgets
{

class DepthOverlayController : public QObject
{
    Q_OBJECT
public:
    explicit DepthOverlayController(QObject *parent = nullptr);

    /// 更新项目元数据；仅当 observed_image_path 对应的深度记录变化时作废当前加载。
    /// 返回 true 表示调用方需要刷新该照片的叠加可用性。
    bool setProjectMetadata(const QJsonObject &metadata,
                            const QString &observed_image_path = QString());
    void setProjectPath(const QString &project_path);
    void request(const QString &image_path,
                 views::DepthOverlayLevel level,
                 const views::DepthOverlayRenderOptions &options);
    void cancelPending();

    bool anyArtifactAvailable(const QString &image_path) const;
    bool artifactAvailable(const QString &image_path, views::DepthOverlayLevel level) const;
    views::DepthOverlayAvailability artifactAvailability(
        const QString &image_path,
        views::DepthOverlayLevel level) const;
    bool isCurrentGeneration(quint64 generation) const noexcept;

    static QString cacheKeyForArtifact(const views::DepthOverlayArtifact &artifact,
                                       views::DepthOverlayLevel level,
                                       const views::DepthOverlayRenderOptions &options,
                                       const QString &source_image_path);

signals:
    void overlayReady(const QString &image_path,
                      const QImage &overlay,
                      const QImage &intensity_base);
    void overlayFailed(const QString &image_path, const QString &error_message);

private:
    struct CacheEntry
    {
        QImage overlay;
        QImage intensityBase;
        qint64 bytes = 0;
        quint64 accessSerial = 0;
    };

    bool readCache(const QString &key, QImage *overlay, QImage *intensity_base);
    void writeCache(const QString &key, const views::DepthOverlayRenderResult &result);
    void evictFor(qint64 required_bytes);

    QJsonObject _metadata;
    QString _projectPath;
    QHash<QString, CacheEntry> _cache;
    qint64 _cacheBytes = 0;
    quint64 _accessSerial = 0;
    quint64 _requestGeneration = 0;
    static constexpr qint64 MaximumCacheBytes = 256LL * 1024LL * 1024LL;
};

} // namespace xjw::gui::widgets

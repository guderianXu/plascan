#pragma once

#include <QImage>
#include <QJsonObject>
#include <QString>

#include <optional>

namespace cv
{
class Mat;
}

namespace xjw::gui::views
{

enum class DepthOverlayLevel
{
    Final,
    Level1,
    Level2,
    Level3
};

struct DepthOverlayArtifact
{
    QString referenceImage;
    QString rawDepthPath;
    QString validMaskPath;
    QString previewPath;
    int level = 0;
};

struct DepthOverlayRenderOptions
{
    int opacity = 150;
    bool showIntensity = false;
};

struct DepthOverlayRenderResult
{
    QImage overlay;
    QImage intensityBase;
    QString errorMessage;
};

enum class DepthOverlayAvailabilityCode
{
    Available,
    NotComputedForResolution,
    NotPersisted,
    ArtifactMissing,
    NoDepthRecord
};

struct DepthOverlayAvailability
{
    bool available = false;
    DepthOverlayAvailabilityCode code = DepthOverlayAvailabilityCode::NoDepthRecord;
    QString reason;
};

std::optional<QJsonObject> resolveDepthOverlayRecord(
    const QJsonObject &project_metadata,
    const QString &image_path);

std::optional<DepthOverlayArtifact> resolveDepthOverlayArtifact(
    const QJsonObject &project_metadata,
    const QString &image_path,
    DepthOverlayLevel level);

DepthOverlayArtifact resolveDepthOverlayArtifactPaths(
    const DepthOverlayArtifact &artifact,
    const QString &project_path);

DepthOverlayAvailability resolveDepthOverlayAvailability(
    const QJsonObject &project_metadata,
    const QString &image_path,
    DepthOverlayLevel level,
    const QString &project_path = {});

QImage colorizeDepthOverlay(const cv::Mat &depth,
                            const cv::Mat &valid_mask,
                            int opacity);

DepthOverlayRenderResult renderDepthOverlay(
    const cv::Mat &depth,
    const cv::Mat &valid_mask,
    const DepthOverlayRenderOptions &options,
    const QImage &source_image = {});

DepthOverlayRenderResult loadDepthOverlay(
    const DepthOverlayArtifact &artifact,
    const DepthOverlayRenderOptions &options,
    const QImage &source_image = {});

} // namespace xjw::gui::views

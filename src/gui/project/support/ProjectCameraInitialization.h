#pragma once

#include <placamera/instance_set.h>

#include <QJsonObject>
#include <QMap>
#include <QSet>
#include <QSize>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>
#include <optional>

namespace xjw::gui::project
{

    enum class CameraInitializationMode
    {
        ExifOrDefault,
        Intrinsics
    };

    struct CameraInitializationRequest
    {
        QStringList images;
        QJsonObject projectMetadata;
        QJsonObject settings;
        CameraInitializationMode mode = CameraInitializationMode::ExifOrDefault;
        QString source;
    };

    struct CameraInitializationResult
    {
        placamera::CameraInstanceSet cameras;
        QMap<QString, QJsonObject> annotationsByImageId;
        QSet<QString> existingImages;
        int skippedExisting = 0;
        int exifCount = 0;
        int fallbackCount = 0;
        int invalidSizeCount = 0;
        int autoPrincipalPointCount = 0;
        bool cancelled = false;
        QString error;
    };

    using CameraInitializationProgress = std::function<void(int completed, int total)>;

    std::optional<double> parsePossiblyFractionalNumber(const QString& text);

    std::optional<double>
    focalPixelsFromExif(const QString& imagePath, const QSize& size, double sensorWidthMm, QString* sourceTag);

    QStringList resolveInitTargets(const QStringList& allImages, const QJsonObject& settings, QString* errorMsg);

    bool withPreparedCameras(const QJsonObject& baseMeta,
                             const CameraInitializationResult& prepared,
                             bool overwriteExisting,
                             QJsonObject* output,
                             QString* error = nullptr);

    CameraInitializationResult prepareCameraInitializations(const CameraInitializationRequest& request,
                                                            const std::atomic<bool>* cancelFlag = nullptr,
                                                            const CameraInitializationProgress& progress = {});

} // namespace xjw::gui::project

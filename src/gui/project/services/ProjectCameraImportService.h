#pragma once

#include <placamera/dataset_formats.h>
#include <placamera/instance_set.h>

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>
#include <optional>
#include <vector>

namespace xjw::gui::project
{

    using CameraImportProgress = std::function<void(int completed, int total)>;

    using PreparedFrameCamera = placamera::CentralCameraGeometry;

    struct PreparedFrameCameraImport
    {
        QString imageAbsPath;
        QString sourceFile;
        QString sourceFormat;
        QString sourceKind;
        placamera::ImageSize imageSize;
        std::optional<PreparedFrameCamera> camera;
        placamera::CameraAcquisitionState acquisition;
        QString error;
    };

    enum class SingleCameraImportStatus
    {
        Ok,
        EmptyImagePath,
        ParseFailed,
        Cancelled
    };

    SingleCameraImportStatus buildSingleCameraImport(const QString& imagePath,
                                                     const QString& tsaiPath,
                                                     PreparedFrameCameraImport* out,
                                                     const std::atomic<bool>* cancelFlag = nullptr,
                                                     const CameraImportProgress& progress = {});

    enum class CameraProjectImportStatus
    {
        Ok,
        NoProjectImages,
        NoImportable,
        ParseFailed,
        Cancelled
    };

    struct CameraProjectImportResult
    {
        std::vector<PreparedFrameCameraImport> cameras;
        QString sourceFormat;
        int ambiguousCount = 0;
        int unmatchedCount = 0;
        int unsupportedCount = 0;
        QStringList warnings;
        QStringList importErrors;
        QString error;
    };

    CameraProjectImportStatus buildCameraProjectImport(const QString& inputPath,
                                                       const QStringList& projectImages,
                                                       CameraProjectImportResult* out,
                                                       const std::atomic<bool>* cancelFlag = nullptr,
                                                       const CameraImportProgress& progress = {});

    bool bindImportedFrameCameras(const QJsonObject& projectCore,
                                  const QString& projectPath,
                                  const std::vector<PreparedFrameCameraImport>& imports,
                                  placamera::CameraInstanceSet* cameras,
                                  QMap<QString, QJsonObject>* annotationsByImageId,
                                  QString* error);

} // namespace xjw::gui::project

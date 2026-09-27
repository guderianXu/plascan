#include "CameraCalibrationData.h"

#include "project/ProjectFramePinholeMetadataIO.h"
#include "project/ProjectMetadata.h"
#include "placamera_runtime/ProjectCameraStore.h"

#include <placamera/rpc_camera.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>
#include <QSize>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

namespace xjw::gui::camera_calibration
{
    namespace
    {

        QString normalizedPathKey(const QString& path)
        {
            return xjw::common::project::normalizePath(path);
        }

        using ProjectCameras = QHash<QString, std::shared_ptr<const placamera::RasterModel>>;

        ProjectCameras projectCamerasByPath(const QJsonObject& metadata, QString* error = nullptr)
        {
            ProjectCameras cameras;
            if (error)
            {
                error->clear();
            }
            const auto loaded =
                xjw::placamera_runtime::loadProjectCameras(xjw::common::project::projectFilesRootObject(metadata));
            if (!loaded.ok())
            {
                if (error)
                {
                    *error = loaded.errors.join(QStringLiteral("；"));
                }
                return cameras;
            }
            for (const QJsonValue& value : xjw::common::project::projectImageEntries(metadata))
            {
                const QJsonObject image = value.toObject();
                const QString path = image.value(QStringLiteral("path")).toString();
                const QString image_id = image.value(QStringLiteral("image_uuid")).toString().trimmed();
                if (path.isEmpty() || image_id.isEmpty())
                {
                    continue;
                }
                const auto model = loaded.instances.forImage(placamera::ImageId(image_id.toStdString()));
                if (model)
                {
                    cameras.insert(normalizedPathKey(path), model.value());
                }
            }
            return cameras;
        }

        QString calibrationModelKey(const QJsonObject& camera)
        {
            const QString model = camera.value(QStringLiteral("model")).toString().trimmed();
            return model.isEmpty() ? QStringLiteral("__generic_camera__") : model.toCaseFolded();
        }

        void addResolutionEvidence(const QString& modelKey,
                                   const QSize& size,
                                   QHash<QString, QSize>* consensus,
                                   QSet<QString>* conflicts)
        {
            if (!consensus || !conflicts || size.width() <= 0 || size.height() <= 0 || conflicts->contains(modelKey))
            {
                return;
            }
            const auto existing = consensus->constFind(modelKey);
            if (existing == consensus->constEnd())
            {
                consensus->insert(modelKey, size);
            }
            else if (*existing != size)
            {
                consensus->remove(modelKey);
                conflicts->insert(modelKey);
            }
        }

        void applyResolvedImageSize(const QSize& size, QJsonObject* calibration)
        {
            if (!calibration || calibration->isEmpty() || size.width() <= 0 || size.height() <= 0)
            {
                return;
            }
            calibration->insert(QStringLiteral("image_width"), size.width());
            calibration->insert(QStringLiteral("image_height"), size.height());
            if (calibration->contains(QStringLiteral("cu")))
            {
                calibration->insert(QStringLiteral("cx"),
                                    calibration->value(QStringLiteral("cu")).toDouble() - size.width() * 0.5);
            }
            if (calibration->contains(QStringLiteral("cv")))
            {
                calibration->insert(QStringLiteral("cy"),
                                    calibration->value(QStringLiteral("cv")).toDouble() - size.height() * 0.5);
            }
        }

        void inferMissingRecordImageSizes(QVector<CameraCalibrationRecord>* records)
        {
            if (!records)
            {
                return;
            }
            QHash<QString, QSize> consensus;
            QSet<QString> conflicts;
            for (const CameraCalibrationRecord& record : *records)
            {
                addResolutionEvidence(record.model.trimmed().isEmpty() ? QStringLiteral("__generic_camera__")
                                                                       : record.model.trimmed().toCaseFolded(),
                                      QSize(record.imageWidth, record.imageHeight),
                                      &consensus,
                                      &conflicts);
            }
            for (CameraCalibrationRecord& record : *records)
            {
                if (record.imageWidth > 0 && record.imageHeight > 0)
                {
                    continue;
                }
                const QString modelKey = record.model.trimmed().isEmpty() ? QStringLiteral("__generic_camera__")
                                                                          : record.model.trimmed().toCaseFolded();
                const QSize size = consensus.value(modelKey);
                if (size.width() <= 0 || size.height() <= 0)
                {
                    continue;
                }
                record.imageWidth = size.width();
                record.imageHeight = size.height();
                applyResolvedImageSize(size, &record.initial);
                applyResolvedImageSize(size, &record.adjusted);
            }
        }

        QSize resolveImageSize(const QString& path,
                               const QJsonObject& image,
                               const QJsonObject& camera,
                               const std::shared_ptr<const placamera::RasterModel>& projectCamera)
        {
            int width = camera.value(QStringLiteral("image_width"))
                            .toInt(camera.value(QStringLiteral("image_samples"))
                                       .toInt(image.value(QStringLiteral("width"))
                                                  .toInt(image.value(QStringLiteral("samples")).toInt())));
            int height = camera.value(QStringLiteral("image_height"))
                             .toInt(camera.value(QStringLiteral("image_lines"))
                                        .toInt(image.value(QStringLiteral("height"))
                                                   .toInt(image.value(QStringLiteral("lines")).toInt())));
            if (projectCamera && (width <= 0 || height <= 0))
            {
                width = projectCamera->imageSize().samples;
                height = projectCamera->imageSize().lines;
            }
            if (width > 0 && height > 0)
            {
                return QSize(width, height);
            }

            const QSize size = QImageReader(path).size();
            return size.isValid() ? size : QSize();
        }

        QJsonObject normalizedCalibration(const placamera::FramePinholeDefinition& definition, const QSize& imageSize)
        {
            const auto& intrinsics = definition.intrinsics();
            const auto& distortion = definition.distortion();
            const double focalX = intrinsics.focalX;
            const double focalY = intrinsics.focalY;
            const double principalX = intrinsics.principalX;
            const double principalY = intrinsics.principalY;

            QJsonObject calibration{
                {QStringLiteral("model"), QStringLiteral("frame_pinhole")},
                {QStringLiteral("intrinsics_unit"), QStringLiteral("px")},
                {QStringLiteral("principal_point_convention"), QStringLiteral("offset_from_image_center")},
                {QStringLiteral("fu"), focalX},
                {QStringLiteral("fv"), focalY},
                {QStringLiteral("f"), std::sqrt(std::abs(focalX * focalY))},
                {QStringLiteral("cu"), principalX},
                {QStringLiteral("cv"), principalY},
                {QStringLiteral("cx"), imageSize.isValid() ? principalX - imageSize.width() * 0.5 : principalX},
                {QStringLiteral("cy"), imageSize.isValid() ? principalY - imageSize.height() * 0.5 : principalY},
                {QStringLiteral("image_width"), imageSize.width()},
                {QStringLiteral("image_height"), imageSize.height()}};
            calibration.insert(QStringLiteral("k1"), distortion.radialK1);
            calibration.insert(QStringLiteral("k2"), distortion.radialK2);
            calibration.insert(QStringLiteral("k3"), distortion.radialK3);
            calibration.insert(QStringLiteral("p1"), distortion.tangentialP1);
            calibration.insert(QStringLiteral("p2"), distortion.tangentialP2);
            return calibration;
        }

        QJsonObject normalizedCalibration(const QJsonObject& camera, const QSize& imageSize)
        {
            const auto parsed = xjw::common::project::decodeFramePinholeMetadata(
                camera, placamera::CameraDefinitionId("calibration-report"));
            return parsed ? normalizedCalibration(*parsed->definition, imageSize) : QJsonObject{};
        }

        QJsonObject projectCalibration(const placamera::RasterModel& camera, const QSize& imageSize)
        {
            if (const auto* frame = dynamic_cast<const placamera::FramePinholeModel*>(&camera))
            {
                return normalizedCalibration(frame->pinholeDefinition(), imageSize);
            }
            const auto* rpc = dynamic_cast<const placamera::RpcModel*>(&camera);
            if (!rpc)
            {
                return {};
            }
            const auto& parameters = rpc->rpcDefinition().parameters();
            QJsonObject calibration{{QStringLiteral("model"), QStringLiteral("rpc00b")},
                                    {QStringLiteral("line_off"), parameters.lineOffset},
                                    {QStringLiteral("samp_off"), parameters.sampleOffset},
                                    {QStringLiteral("lat_off"), parameters.latitudeOffset},
                                    {QStringLiteral("long_off"), parameters.longitudeOffset},
                                    {QStringLiteral("height_off"), parameters.heightOffset},
                                    {QStringLiteral("line_scale"), parameters.lineScale},
                                    {QStringLiteral("samp_scale"), parameters.sampleScale},
                                    {QStringLiteral("lat_scale"), parameters.latitudeScale},
                                    {QStringLiteral("long_scale"), parameters.longitudeScale},
                                    {QStringLiteral("height_scale"), parameters.heightScale},
                                    {QStringLiteral("image_width"), imageSize.width()},
                                    {QStringLiteral("image_height"), imageSize.height()}};
            if (parameters.errorBiasMeters)
            {
                calibration.insert(QStringLiteral("err_bias_m"), *parameters.errorBiasMeters);
            }
            if (parameters.errorRandomMeters)
            {
                calibration.insert(QStringLiteral("err_rand_m"), *parameters.errorRandomMeters);
            }
            return calibration;
        }

        QJsonObject automaticInitialCalibration(const QSize& imageSize, double focalScale)
        {
            if (!imageSize.isValid())
            {
                return {};
            }
            const double focal = std::max(imageSize.width(), imageSize.height()) * std::max(0.1, focalScale);
            return QJsonObject{
                {QStringLiteral("model"), QStringLiteral("brown_conrady")},
                {QStringLiteral("intrinsics_unit"), QStringLiteral("px")},
                {QStringLiteral("principal_point_convention"), QStringLiteral("offset_from_image_center")},
                {QStringLiteral("f"), focal},
                {QStringLiteral("fu"), focal},
                {QStringLiteral("fv"), focal},
                {QStringLiteral("cu"), imageSize.width() * 0.5},
                {QStringLiteral("cv"), imageSize.height() * 0.5},
                {QStringLiteral("cx"), 0.0},
                {QStringLiteral("cy"), 0.0},
                {QStringLiteral("k1"), 0.0},
                {QStringLiteral("k2"), 0.0},
                {QStringLiteral("k3"), 0.0},
                {QStringLiteral("p1"), 0.0},
                {QStringLiteral("p2"), 0.0},
                {QStringLiteral("image_width"), imageSize.width()},
                {QStringLiteral("image_height"), imageSize.height()}};
        }

        void applyImageMetadata(const QJsonObject& image,
                                const std::shared_ptr<const placamera::RasterModel>& camera,
                                CameraCalibrationRecord* record)
        {
            if (!record)
            {
                return;
            }

            record->hasProjectCamera = camera != nullptr;
            record->path = image.value(QStringLiteral("path")).toString(record->path);
            record->name = QFileInfo(record->path).fileName();
            if (camera)
            {
                record->model = QString::fromStdString(std::string(camera->modelType()));
                record->imageWidth = camera->imageSize().samples;
                record->imageHeight = camera->imageSize().lines;
            }
            else
            {
                record->imageWidth = image.value(QStringLiteral("width"))
                                         .toInt(image.value(QStringLiteral("samples")).toInt(record->imageWidth));
                record->imageHeight = image.value(QStringLiteral("height"))
                                          .toInt(image.value(QStringLiteral("lines")).toInt(record->imageHeight));
            }

            if (camera && !record->hasInitial && !record->hasAdjusted)
            {
                record->initial = projectCalibration(*camera, QSize(record->imageWidth, record->imageHeight));
                record->hasInitial = !record->initial.isEmpty();
                record->initialSource = QStringLiteral("project_camera_prior");
                record->adjustmentStatus = QStringLiteral("not_run");
                if (record->model == QStringLiteral("rpc00b"))
                {
                    record->initialSource = QStringLiteral("embedded_rpc00b");
                    record->adjustmentStatus = QStringLiteral("rpc_fixed_model");
                }
            }
        }

    } // namespace

    static QJsonArray buildCameraCalibrationComparisonImpl(const QJsonObject& projectMetadata,
                                                           const QMap<QString, QJsonObject>& adjustedCameras,
                                                           const QJsonObject& sfmDiagnostics,
                                                           const ProjectCameras* nativeAdjustedCameras)
    {
        const ProjectCameras projectCameras = projectCamerasByPath(projectMetadata);
        QHash<QString, QJsonObject> imagesByPath;
        for (const QJsonValue& value : xjw::common::project::projectImageEntries(projectMetadata))
        {
            const QJsonObject image = value.toObject();
            const QString path = image.value(QStringLiteral("path")).toString();
            if (!path.isEmpty())
            {
                imagesByPath.insert(normalizedPathKey(path), image);
            }
        }

        const double focalScale = sfmDiagnostics.value(QStringLiteral("adaptive_focal_seed_scale")).toDouble(1.0);
        const QString adjustmentStatus =
            sfmDiagnostics.value(QStringLiteral("camera_self_calibration_status")).toString(QStringLiteral("fixed"));
        const bool requiresReview =
            sfmDiagnostics.value(QStringLiteral("camera_self_calibration_requires_review")).toBool(false);
        const bool adaptiveFittingApplied =
            sfmDiagnostics.value(QStringLiteral("adaptive_camera_model_fitting_applied")).toBool(false);
        const QJsonObject enabledParameters =
            sfmDiagnostics.value(QStringLiteral("ba_intrinsic_parameter_enabled")).toObject();
        const QJsonObject parameterReliability =
            sfmDiagnostics.value(QStringLiteral("ba_intrinsic_parameter_reliability")).toObject();
        const QJsonObject metadataPrior = sfmDiagnostics.value(QStringLiteral("image_metadata_focal_prior")).toObject();
        const QString priorModel = metadataPrior.value(QStringLiteral("model")).toString();
        const bool usedMetadataPrior = metadataPrior.value(QStringLiteral("used")).toBool(false);

        QHash<QString, QSize> resolvedSizesByPath;
        QHash<QString, QSize> resolutionConsensusByModel;
        QSet<QString> resolutionConflicts;
        for (auto it = adjustedCameras.constBegin(); it != adjustedCameras.constEnd(); ++it)
        {
            const QString path = normalizedPathKey(it.key());
            const QJsonObject image = imagesByPath.value(path);
            const auto projectCamera = projectCameras.value(path);
            const QSize size = resolveImageSize(path, image, it.value(), projectCamera);
            resolvedSizesByPath.insert(path, size);
            const QString modelKey =
                priorModel.isEmpty() ? calibrationModelKey(it.value()) : priorModel.trimmed().toCaseFolded();
            addResolutionEvidence(modelKey, size, &resolutionConsensusByModel, &resolutionConflicts);
        }

        QJsonArray comparisons;
        for (auto it = adjustedCameras.constBegin(); it != adjustedCameras.constEnd(); ++it)
        {
            const QString path = normalizedPathKey(it.key());
            const QJsonObject image = imagesByPath.value(path);
            const auto projectCamera = projectCameras.value(path);
            const auto projectFrame = std::dynamic_pointer_cast<const placamera::FramePinholeModel>(projectCamera);
            const bool usesProjectCamera = projectFrame != nullptr;
            const QString modelKey =
                priorModel.isEmpty() ? calibrationModelKey(it.value()) : priorModel.trimmed().toCaseFolded();
            QSize imageSize = resolvedSizesByPath.value(path);
            if (imageSize.width() <= 0 || imageSize.height() <= 0)
            {
                imageSize = resolutionConsensusByModel.value(modelKey);
            }
            const QJsonObject initial = usesProjectCamera
                                            ? normalizedCalibration(projectFrame->pinholeDefinition(), imageSize)
                                            : automaticInitialCalibration(imageSize, focalScale);
            QJsonObject adjusted;
            if (nativeAdjustedCameras)
            {
                const auto nativeAdjusted = nativeAdjustedCameras->constFind(path);
                if (nativeAdjusted != nativeAdjustedCameras->constEnd())
                {
                    adjusted = projectCalibration(*nativeAdjusted.value(), imageSize);
                }
            }
            else
            {
                adjusted = normalizedCalibration(it.value(), imageSize);
            }
            QStringList optimized;
            if (adaptiveFittingApplied)
            {
                if (!enabledParameters.isEmpty())
                {
                    static const std::array<const char*, 9> parameterNames{{
                        "f",
                        "aspect",
                        "cx",
                        "cy",
                        "k1",
                        "k2",
                        "k3",
                        "p1",
                        "p2",
                    }};
                    for (const char* parameterName : parameterNames)
                    {
                        const QString name = QString::fromLatin1(parameterName);
                        if (enabledParameters.value(name).toBool(false))
                        {
                            optimized.append(name);
                        }
                    }
                }
            }
            const QString effectiveStatus =
                adjustmentStatus == QStringLiteral("trusted_prior")
                    ? (adaptiveFittingApplied ? QStringLiteral("trusted_prior_limited_refinement")
                                              : QStringLiteral("trusted_prior_fixed"))
                    : adjustmentStatus;

            QJsonObject comparison{
                {QStringLiteral("path"), path},
                {QStringLiteral("name"), QFileInfo(path).fileName()},
                {QStringLiteral("camera_model"),
                 priorModel.isEmpty() ? adjusted.value(QStringLiteral("model")).toString() : priorModel},
                {QStringLiteral("had_before"), !initial.isEmpty()},
                {QStringLiteral("initial_camera"), initial},
                {QStringLiteral("adjusted_camera"), adjusted},
                {QStringLiteral("initial_source"),
                 !usesProjectCamera ? (usedMetadataPrior ? QStringLiteral("image_metadata_focal_prior")
                                                         : QStringLiteral("automatic_focal_seed"))
                                    : QStringLiteral("project_camera_prior")},
                {QStringLiteral("adjustment_status"), effectiveStatus},
                {QStringLiteral("intrinsics_refined"), !optimized.isEmpty()},
                {QStringLiteral("requires_review"), requiresReview},
                {QStringLiteral("parameter_reliability"), parameterReliability},
                {QStringLiteral("optimized_parameters"), QJsonArray::fromStringList(optimized)}};
            comparisons.append(comparison);
        }
        return comparisons;
    }

    QJsonArray buildCameraCalibrationComparison(const QJsonObject& projectMetadata,
                                                const QMap<QString, QJsonObject>& adjustedCameras,
                                                const QJsonObject& sfmDiagnostics)
    {
        return buildCameraCalibrationComparisonImpl(projectMetadata, adjustedCameras, sfmDiagnostics, nullptr);
    }

    QJsonArray buildCameraCalibrationComparison(const QJsonObject& projectMetadata,
                                                const placamera::CameraInstanceSet& adjustedCameras,
                                                const QJsonObject& sfmDiagnostics)
    {
        QHash<QString, QString> pathByImageId;
        for (const QJsonValue& value : xjw::common::project::projectImageEntries(projectMetadata))
        {
            const QJsonObject image = value.toObject();
            pathByImageId.insert(image.value(QStringLiteral("image_uuid")).toString(),
                                 normalizedPathKey(image.value(QStringLiteral("path")).toString()));
        }
        QMap<QString, QJsonObject> summaries;
        ProjectCameras nativeCameras;
        for (const auto& camera : adjustedCameras.values())
        {
            const QString path = pathByImageId.value(QString::fromStdString(camera->imageId().value()));
            if (path.isEmpty())
            {
                continue;
            }
            summaries.insert(path,
                             QJsonObject{{QStringLiteral("model"), QString::fromStdString(std::string(camera->modelType()))},
                                         {QStringLiteral("image_width"), camera->imageSize().samples},
                                         {QStringLiteral("image_height"), camera->imageSize().lines}});
            nativeCameras.insert(path, camera);
        }
        return buildCameraCalibrationComparisonImpl(projectMetadata, summaries, sfmDiagnostics, &nativeCameras);
    }

    QVector<CameraCalibrationRecord> buildCameraCalibrationRecords(const QJsonObject& projectMetadata,
                                                                   const QJsonObject& bundleAdjustReport,
                                                                   QString* cameraError)
    {
        const ProjectCameras projectCameras = projectCamerasByPath(projectMetadata, cameraError);
        QVector<CameraCalibrationRecord> records;
        QHash<QString, int> recordIndexByPath;

        const QJsonArray comparisons = bundleAdjustReport.value(QStringLiteral("camera_comparison")).toArray();
        for (const QJsonValue& value : comparisons)
        {
            const QJsonObject comparison = value.toObject();
            CameraCalibrationRecord record;
            record.path = comparison.value(QStringLiteral("path")).toString();
            record.name = comparison.value(QStringLiteral("name")).toString(QFileInfo(record.path).fileName());
            record.initial = comparison.value(QStringLiteral("initial_camera")).toObject();
            record.adjusted = comparison.value(QStringLiteral("adjusted_camera")).toObject();
            record.hasInitial = !record.initial.isEmpty();
            record.hasAdjusted = !record.adjusted.isEmpty();
            record.initialSource = comparison.value(QStringLiteral("initial_source")).toString();
            record.adjustmentStatus = comparison.value(QStringLiteral("adjustment_status")).toString();
            record.parameterReliability = comparison.value(QStringLiteral("parameter_reliability")).toObject();
            for (const QJsonValue& parameter : comparison.value(QStringLiteral("optimized_parameters")).toArray())
            {
                record.optimizedParameters.append(parameter.toString());
            }
            record.requiresReview = comparison.value(QStringLiteral("requires_review")).toBool(false);

            const QJsonObject preferred = record.hasAdjusted ? record.adjusted : record.initial;
            record.model = comparison.value(QStringLiteral("camera_model"))
                               .toString(preferred.value(QStringLiteral("model")).toString());
            record.imageWidth = preferred.value(QStringLiteral("image_width")).toInt();
            record.imageHeight = preferred.value(QStringLiteral("image_height")).toInt();

            const QString key = normalizedPathKey(record.path);
            recordIndexByPath.insert(key, records.size());
            records.append(record);
        }

        const QJsonArray images = xjw::common::project::projectImageEntries(projectMetadata);
        for (const QJsonValue& value : images)
        {
            const QJsonObject image = value.toObject();
            const QString path = image.value(QStringLiteral("path")).toString();
            if (path.trimmed().isEmpty())
            {
                continue;
            }

            const QString key = normalizedPathKey(path);
            const auto camera = projectCameras.value(key);
            if (!camera)
            {
                continue;
            }
            const auto existing = recordIndexByPath.constFind(key);
            if (existing != recordIndexByPath.constEnd())
            {
                applyImageMetadata(image, camera, &records[*existing]);
                continue;
            }

            CameraCalibrationRecord record;
            applyImageMetadata(image, camera, &record);
            recordIndexByPath.insert(key, records.size());
            records.append(record);
        }

        inferMissingRecordImageSizes(&records);

        std::sort(records.begin(),
                  records.end(),
                  [](const auto& left, const auto& right)
                  { return QString::localeAwareCompare(left.name, right.name) < 0; });
        return records;
    }

    QJsonObject readLatestCameraCalibrationReport(const QString& projectAssetsDir, QString* errorMessage)
    {
        if (errorMessage)
        {
            errorMessage->clear();
        }
        if (projectAssetsDir.trimmed().isEmpty())
        {
            return {};
        }

        const QString reportPath = QDir(projectAssetsDir).filePath(QStringLiteral("reports/at_report.json"));
        QFile reportFile(reportPath);
        if (!reportFile.exists())
        {
            return {};
        }
        if (!reportFile.open(QIODevice::ReadOnly))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取空三报告：%1").arg(reportPath);
            }
            return {};
        }

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(reportFile.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("空三报告格式无效：%1（%2）").arg(reportPath, parseError.errorString());
            }
            return {};
        }
        return document.object();
    }

} // namespace xjw::gui::camera_calibration

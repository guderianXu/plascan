#include "MvsWorkspaceReplay.h"

#include "MvsWorkspaceManifest.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace xjw::mvs
{
namespace
{

bool readDoubleArray(const QJsonValue &value, double *output, int count)
{
    const QJsonArray array = value.toArray();
    if (!output || array.size() != count)
    {
        return false;
    }
    for (int index = 0; index < count; ++index)
    {
        output[index] = array.at(index).toDouble();
    }
    return true;
}

QString maskPathForImage(const QString &maskDirectory, const QString &imagePath)
{
    if (maskDirectory.trimmed().isEmpty())
    {
        return QString();
    }
    return QDir(maskDirectory).filePath(
        QFileInfo(imagePath).completeBaseName() + QStringLiteral("_mask.png"));
}

bool readJsonObject(const QString &path, QJsonObject *object, QString *errorMessage)
{
    QFile file(path);
    if (!object || !file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 JSON：%1").arg(QDir::toNativeSeparators(path));
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("JSON 格式无效：%1（%2）")
                                .arg(QDir::toNativeSeparators(path), parseError.errorString());
        }
        return false;
    }
    *object = document.object();
    return true;
}

QString resolveManifestPath(const QString &manifestPath, const QString &storedPath)
{
    const QString trimmedPath = storedPath.trimmed();
    if (trimmedPath.isEmpty())
    {
        return QString();
    }

    QFileInfo pathInfo(trimmedPath);
    if (pathInfo.isRelative())
    {
        const QDir manifestDirectory(QFileInfo(manifestPath).absolutePath());
        pathInfo.setFile(manifestDirectory.filePath(trimmedPath));
    }

    const QString canonicalPath = pathInfo.canonicalFilePath();
    return QDir::cleanPath(
        canonicalPath.isEmpty() ? pathInfo.absoluteFilePath() : canonicalPath);
}

QString replayImageIdentity(const QString &path)
{
    QString identity = QDir::fromNativeSeparators(QDir::cleanPath(path));
#ifdef Q_OS_WIN
    identity = identity.toCaseFolded();
#endif
    return identity;
}

} // namespace

bool cameraFromMvsWorkspaceJson(const QJsonObject &object, FramePinholeCamera *camera)
{
    if (!camera || object.isEmpty())
    {
        return false;
    }

    std::array<double, 9> worldToCamera{};
    std::array<double, 3> center{};
    if (!readDoubleArray(object.value(QStringLiteral("rotation_world_to_camera")),
                         worldToCamera.data(),
                         9) ||
        !readDoubleArray(object.value(QStringLiteral("camera_center")), center.data(), 3))
    {
        return false;
    }

    const double focalX = object.value(QStringLiteral("fx")).toDouble();
    const double focalY = object.value(QStringLiteral("fy")).toDouble();
    const double principalX = object.value(QStringLiteral("cx")).toDouble();
    const double principalY = object.value(QStringLiteral("cy")).toDouble();
    if (!std::isfinite(focalX) || !std::isfinite(focalY) ||
        !std::isfinite(principalX) || !std::isfinite(principalY) ||
        std::abs(focalX) <= 1.0e-12 || std::abs(focalY) <= 1.0e-12)
    {
        return false;
    }

    const std::array<double, 9> cameraToWorld{{
        worldToCamera[0], worldToCamera[3], worldToCamera[6],
        worldToCamera[1], worldToCamera[4], worldToCamera[7],
        worldToCamera[2], worldToCamera[5], worldToCamera[8]
    }};

    FramePinholeCamera parsed;
    parsed.setIntrinsics(focalX, focalY, principalX, principalY);
    parsed.setPose(cameraToWorld, center);
    parsed.setDistortion(FramePinholeCamera::Distortion{});
    if (!parsed.isValid())
    {
        return false;
    }
    *camera = parsed;
    return true;
}

bool loadMvsReplayViews(const QString &manifestPath,
                        const QString &maskDirectory,
                        std::vector<CameraView> *views,
                        QString *errorMessage)
{
    if (!views)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS replay 视图输出参数为空");
        }
        return false;
    }
    views->clear();

    MvsWorkspaceManifest manifest;
    QString manifestError;
    if (!manifest.load(manifestPath, &manifestError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法加载 MVS manifest：%1（%2）")
                                .arg(QDir::toNativeSeparators(manifestPath), manifestError);
        }
        return false;
    }

    std::map<int, MvsDepthFrameRecord> recordsByIndex;
    for (const MvsDepthFrameRecord &record : manifest.frames())
    {
        if (record.refIndex < 0 || record.refImage.trimmed().isEmpty())
        {
            continue;
        }
        if (!recordsByIndex.emplace(record.refIndex, record).second)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS manifest 含重复 ref_index：%1")
                                    .arg(record.refIndex);
            }
            return false;
        }
    }

    if (recordsByIndex.size() < 2)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS manifest 中可重放视图少于 2 张");
        }
        return false;
    }

    views->reserve(recordsByIndex.size());
    QSet<QString> replayImageIdentities;
    QSet<QString> replayRasterIdentities;
    int expectedIndex = 0;
    for (const auto &[index, record] : recordsByIndex)
    {
        if (index != expectedIndex)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "MVS manifest 的 ref_index 不连续：期望 %1，实际 %2")
                                    .arg(expectedIndex)
                                    .arg(index);
            }
            views->clear();
            return false;
        }

        const QString source_image_path = resolveManifestPath(
            manifestPath, record.refImage);
        const bool has_prepared_image =
            !record.preparedImage.trimmed().isEmpty();
        const bool has_prepared_mask =
            !record.preparedValidMaskPath.trimmed().isEmpty();
        const bool has_prepared_camera = !record.preparedCameraModel.isEmpty();
        if (has_prepared_image != has_prepared_mask ||
            has_prepared_image != has_prepared_camera)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "MVS manifest 第 %1 帧的 prepared_image、prepared "
                    "有效蒙版与 prepared_camera_model 不完整")
                                    .arg(index);
            }
            views->clear();
            return false;
        }
        const QString raster_path = resolveManifestPath(
            manifestPath,
            has_prepared_image ? record.preparedImage : record.refImage);
        if (!QFileInfo::exists(raster_path))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS replay 工作栅格不存在：%1")
                                    .arg(QDir::toNativeSeparators(raster_path));
            }
            views->clear();
            return false;
        }
        const QString imageIdentity = replayImageIdentity(source_image_path);
        if (replayImageIdentities.contains(imageIdentity))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "MVS manifest 含重复 ref_image：第 %1 帧与已有帧指向同一影像 %2")
                                    .arg(index)
                                    .arg(QDir::toNativeSeparators(source_image_path));
            }
            views->clear();
            return false;
        }
        replayImageIdentities.insert(imageIdentity);
        const QString raster_identity = replayImageIdentity(raster_path);
        if (replayRasterIdentities.contains(raster_identity))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "MVS manifest 第 %1 帧与已有帧指向同一 prepared raster：%2")
                                    .arg(index)
                                    .arg(QDir::toNativeSeparators(raster_path));
            }
            views->clear();
            return false;
        }
        replayRasterIdentities.insert(raster_identity);

        CameraView view;
        view.imagePath = xjw::common::io::toUtf8Path(source_image_path);
        if (has_prepared_image)
        {
            view.preparedImagePath = xjw::common::io::toUtf8Path(raster_path);
        }
        const QJsonObject replay_camera = has_prepared_camera
            ? record.preparedCameraModel
            : record.cameraModel;
        if (!cameraFromMvsWorkspaceJson(replay_camera, &view.camera))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS manifest 第 %1 帧相机模型无效").arg(index);
            }
            views->clear();
            return false;
        }

        const cv::Mat image = xjw::common::io::readImage(
            has_prepared_image ? view.preparedImagePath : view.imagePath,
            cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取 MVS replay 影像：%1")
                                    .arg(QDir::toNativeSeparators(raster_path));
            }
            views->clear();
            return false;
        }
        view.imageWidth = image.cols;
        view.imageHeight = image.rows;
        view.camera.setImageSize(CameraImageSize{image.cols, image.rows});

        const QString prepared_mask_path = resolveManifestPath(
            manifestPath, record.preparedValidMaskPath);
        if (!prepared_mask_path.isEmpty())
        {
            const cv::Mat prepared_mask = xjw::common::io::readImage(
                prepared_mask_path, cv::IMREAD_GRAYSCALE);
            if (prepared_mask.empty())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral(
                        "MVS replay prepared 有效蒙版不存在：%1")
                                        .arg(QDir::toNativeSeparators(
                                            prepared_mask_path));
                }
                views->clear();
                return false;
            }
            if (prepared_mask.size() != image.size())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral(
                        "MVS replay prepared 有效蒙版与工作栅格尺寸不一致：%1")
                                        .arg(QDir::toNativeSeparators(
                                            prepared_mask_path));
                }
                views->clear();
                return false;
            }
            view.preparedValidMaskPath = xjw::common::io::toUtf8Path(
                prepared_mask_path);
            view.preparedValidMaskSource =
                record.maskSource.toUtf8().toStdString();
        }
        else
        {
            const QString mask_path = maskPathForImage(
                maskDirectory, source_image_path);
            if (!mask_path.isEmpty() && !QFileInfo::exists(mask_path))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("MVS replay 蒙版不存在：%1")
                                        .arg(QDir::toNativeSeparators(mask_path));
                }
                views->clear();
                return false;
            }
            if (!mask_path.isEmpty())
            {
                view.validRegionMaskPath = xjw::common::io::toUtf8Path(
                    mask_path);
            }
        }
        views->push_back(std::move(view));
        ++expectedIndex;
    }

    return true;
}

bool loadMvsPairAuditReport(
    const QString &reportPath,
    std::vector<MvsSourcePairQuality> *qualities,
    MvsPairAuditSummary *summary,
    QString *errorMessage)
{
    if (!qualities)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS pair audit 输出参数为空");
        }
        return false;
    }
    qualities->clear();
    if (summary)
    {
        *summary = MvsPairAuditSummary{};
    }

    QJsonObject root;
    if (!readJsonObject(reportPath, &root, errorMessage))
    {
        return false;
    }

    const QJsonArray pairs = root.value(QStringLiteral("pairs")).toArray();
    qualities->reserve(static_cast<std::size_t>(pairs.size()));
    MvsPairAuditSummary parsedSummary;
    for (const QJsonValue &value : pairs)
    {
        const QJsonObject pair = value.toObject();
        const QString storedImageA =
            pair.value(QStringLiteral("image_a")).toString().trimmed();
        const QString storedImageB =
            pair.value(QStringLiteral("image_b")).toString().trimmed();
        if (storedImageA.isEmpty() || storedImageB.isEmpty())
        {
            continue;
        }
        const QString imageA = resolveManifestPath(reportPath, storedImageA);
        const QString imageB = resolveManifestPath(reportPath, storedImageB);

        MvsSourcePairQuality quality;
        quality.imageA = xjw::common::io::toUtf8Path(imageA);
        quality.imageB = xjw::common::io::toUtf8Path(imageB);
        quality.totalMatches = std::max(
            0, pair.value(QStringLiteral("total_matches")).toInt());
        quality.geometricInliers = std::max(
            0, pair.value(QStringLiteral("geometric_inliers")).toInt());
        quality.geometricCoverage = static_cast<float>(std::clamp(
            pair.value(QStringLiteral("coverage_score")).toDouble(0.0), 0.0, 1.0));
        const QString status =
            pair.value(QStringLiteral("status")).toString().trimmed().toLower();
        quality.hasVerificationStatistics =
            status == QStringLiteral("verified") || status == QStringLiteral("failed");
        quality.verified = status == QStringLiteral("verified");
        quality.verificationReason = xjw::common::io::toUtf8Path(
            pair.value(QStringLiteral("reason")).toString());

        ++parsedSummary.auditedPairCount;
        if (quality.verified)
        {
            ++parsedSummary.verifiedPairCount;
        }
        else if (quality.hasVerificationStatistics)
        {
            ++parsedSummary.failedPairCount;
        }
        else
        {
            ++parsedSummary.missingStatisticsPairCount;
        }
        qualities->push_back(std::move(quality));
    }

    if (qualities->empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS pair audit 报告不含有效像对：%1")
                                .arg(QDir::toNativeSeparators(reportPath));
        }
        return false;
    }
    if (summary)
    {
        *summary = parsedSummary;
    }
    return true;
}

} // namespace xjw::mvs

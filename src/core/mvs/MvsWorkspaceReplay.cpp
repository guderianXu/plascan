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

bool cameraFromMvsWorkspaceJson(const QJsonObject &object, Camera *camera)
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

    Camera parsed;
    parsed.setIntrinsics(focalX, focalY, principalX, principalY);
    parsed.setPose(cameraToWorld, center);
    parsed.setDistortion(Camera::Distortion{});
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

        const QString imagePath = resolveManifestPath(manifestPath, record.refImage);
        if (!QFileInfo::exists(imagePath))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS replay 影像不存在：%1")
                                    .arg(QDir::toNativeSeparators(imagePath));
            }
            views->clear();
            return false;
        }
        const QString imageIdentity = replayImageIdentity(imagePath);
        if (replayImageIdentities.contains(imageIdentity))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "MVS manifest 含重复 ref_image：第 %1 帧与已有帧指向同一影像 %2")
                                    .arg(index)
                                    .arg(QDir::toNativeSeparators(imagePath));
            }
            views->clear();
            return false;
        }
        replayImageIdentities.insert(imageIdentity);

        CameraView view;
        view.imagePath = xjw::common::io::toUtf8Path(imagePath);
        if (!cameraFromMvsWorkspaceJson(record.cameraModel, &view.camera))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS manifest 第 %1 帧相机模型无效").arg(index);
            }
            views->clear();
            return false;
        }

        const cv::Mat image = xjw::common::io::readImage(
            view.imagePath, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取 MVS replay 影像：%1")
                                    .arg(QDir::toNativeSeparators(imagePath));
            }
            views->clear();
            return false;
        }
        view.imageWidth = image.cols;
        view.imageHeight = image.rows;

        const QString maskPath = maskPathForImage(maskDirectory, imagePath);
        if (!maskPath.isEmpty())
        {
            if (!QFileInfo::exists(maskPath))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("MVS replay 蒙版不存在：%1")
                                        .arg(QDir::toNativeSeparators(maskPath));
                }
                views->clear();
                return false;
            }
            view.validRegionMaskPath = xjw::common::io::toUtf8Path(maskPath);
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

#include "CameraCalibrationData.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>

#include <algorithm>

namespace xjw::gui::camera_calibration
{
namespace
{

QString normalizedPathKey(const QString &path)
{
    QString key = QDir::cleanPath(QDir::fromNativeSeparators(path.trimmed()));
#ifdef Q_OS_WIN
    key = key.toCaseFolded();
#endif
    return key;
}

void copyLegacyParameter(const QJsonObject &comparison,
                         const QString &parameter,
                         const QString &suffix,
                         QJsonObject *camera)
{
    const QString key = parameter + suffix;
    if (camera && comparison.contains(key))
    {
        camera->insert(parameter, comparison.value(key));
    }
}

QJsonObject legacyCamera(const QJsonObject &comparison, const QString &suffix)
{
    QJsonObject camera;
    for (const QString &parameter : {
             QStringLiteral("fu"), QStringLiteral("fv"), QStringLiteral("cu"),
             QStringLiteral("cv"), QStringLiteral("k1"), QStringLiteral("k2"),
             QStringLiteral("k3"), QStringLiteral("p1"), QStringLiteral("p2")})
    {
        copyLegacyParameter(comparison, parameter, suffix, &camera);
    }
    return camera;
}

void applyImageMetadata(const QJsonObject &image, CameraCalibrationRecord *record)
{
    if (!record)
    {
        return;
    }

    const QJsonObject camera = image.value(QStringLiteral("camera")).toObject();
    record->path = image.value(QStringLiteral("path")).toString(record->path);
    record->name = QFileInfo(record->path).fileName();
    record->model = camera.value(QStringLiteral("model")).toString(record->model);
    record->imageWidth = camera.value(QStringLiteral("image_width"))
                             .toInt(image.value(QStringLiteral("width")).toInt(record->imageWidth));
    record->imageHeight = camera.value(QStringLiteral("image_height"))
                              .toInt(image.value(QStringLiteral("height")).toInt(record->imageHeight));

    if (!camera.isEmpty())
    {
        if (record->hasAdjusted)
        {
            record->adjusted = camera;
        }
        else if (!record->hasInitial)
        {
            record->initial = camera;
            record->hasInitial = true;
        }
    }
}

} // namespace

QVector<CameraCalibrationRecord> buildCameraCalibrationRecords(
    const QJsonObject &projectMetadata,
    const QJsonObject &bundleAdjustReport)
{
    QVector<CameraCalibrationRecord> records;
    QHash<QString, int> recordIndexByPath;

    const QJsonArray comparisons =
        bundleAdjustReport.value(QStringLiteral("camera_comparison")).toArray();
    for (const QJsonValue &value : comparisons)
    {
        const QJsonObject comparison = value.toObject();
        CameraCalibrationRecord record;
        record.path = comparison.value(QStringLiteral("path")).toString();
        record.name = comparison.value(QStringLiteral("name")).toString(
            QFileInfo(record.path).fileName());
        record.initial = comparison.value(QStringLiteral("initial_camera")).toObject();
        record.adjusted = comparison.value(QStringLiteral("adjusted_camera")).toObject();
        if (record.initial.isEmpty())
        {
            record.initial = legacyCamera(comparison, QStringLiteral("_before"));
        }
        if (record.adjusted.isEmpty())
        {
            record.adjusted = legacyCamera(comparison, QStringLiteral("_after"));
        }
        record.hasInitial = comparison.value(QStringLiteral("had_before"))
                                .toBool(!record.initial.isEmpty());
        record.hasAdjusted = !record.adjusted.isEmpty();

        const QJsonObject preferred = record.hasAdjusted ? record.adjusted : record.initial;
        record.model = preferred.value(QStringLiteral("model")).toString();
        record.imageWidth = preferred.value(QStringLiteral("image_width")).toInt();
        record.imageHeight = preferred.value(QStringLiteral("image_height")).toInt();

        const QString key = normalizedPathKey(record.path);
        recordIndexByPath.insert(key, records.size());
        records.append(record);
    }

    const QJsonArray images = projectMetadata.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        const QString path = image.value(QStringLiteral("path")).toString();
        const QJsonObject camera = image.value(QStringLiteral("camera")).toObject();
        if (path.trimmed().isEmpty() || camera.isEmpty())
        {
            continue;
        }

        const QString key = normalizedPathKey(path);
        const auto existing = recordIndexByPath.constFind(key);
        if (existing != recordIndexByPath.constEnd())
        {
            applyImageMetadata(image, &records[*existing]);
            continue;
        }

        CameraCalibrationRecord record;
        applyImageMetadata(image, &record);
        recordIndexByPath.insert(key, records.size());
        records.append(record);
    }

    std::sort(records.begin(), records.end(), [](const auto &left, const auto &right)
    {
        return QString::localeAwareCompare(left.name, right.name) < 0;
    });
    return records;
}

QJsonObject readLatestCameraCalibrationReport(const QString &projectAssetsDir,
                                              QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (projectAssetsDir.trimmed().isEmpty())
    {
        return {};
    }

    const QString reportPath = QDir(projectAssetsDir).filePath(
        QStringLiteral("reports/at_report.json"));
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
            *errorMessage = QStringLiteral("空三报告格式无效：%1（%2）")
                                .arg(reportPath, parseError.errorString());
        }
        return {};
    }
    return document.object();
}

} // namespace xjw::gui::camera_calibration

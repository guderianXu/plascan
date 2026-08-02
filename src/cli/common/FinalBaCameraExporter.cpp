#include "FinalBaCameraExporter.h"

#include "io/PathIO.h"
#include "ProjectCameraIO.h"

#include <QDir>
#include <QFileInfo>
#include <QSaveFile>
#include <QSet>
#include <QStringConverter>
#include <QTemporaryDir>
#include <QTextStream>

#include <utility>
#include <vector>

namespace xjw::cli
{
namespace
{

QString normalizedCameraLookupKey(const QString &path)
{
    QString normalized = QDir::fromNativeSeparators(
        QDir::cleanPath(QFileInfo(path).absoluteFilePath()));
#ifdef Q_OS_WIN
    normalized = normalized.toCaseFolded();
#endif
    return normalized;
}

QString safeCameraStem(const QString &imagePath, QSet<QString> *usedNames)
{
    QString stem = QFileInfo(imagePath).completeBaseName().trimmed();
    if (stem.isEmpty())
    {
        stem = QStringLiteral("camera");
    }
    for (int index = 0; index < stem.size(); ++index)
    {
        const QChar ch = stem.at(index);
        if (QStringLiteral("<>:\"/\\|?*").contains(ch) || ch.unicode() < 32)
        {
            stem[index] = QLatin1Char('_');
        }
    }
    while (stem.endsWith(QLatin1Char('.')) || stem.endsWith(QLatin1Char(' ')))
    {
        stem.chop(1);
    }
    if (stem.isEmpty())
    {
        stem = QStringLiteral("camera");
    }

    const QString upper = stem.toUpper();
    const bool reserved = upper == QStringLiteral("CON") || upper == QStringLiteral("PRN")
        || upper == QStringLiteral("AUX") || upper == QStringLiteral("NUL")
        || (upper.size() == 4
            && (upper.startsWith(QStringLiteral("COM")) || upper.startsWith(QStringLiteral("LPT")))
            && upper.at(3) >= QLatin1Char('1') && upper.at(3) <= QLatin1Char('9'));
    if (reserved)
    {
        stem.prepend(QLatin1Char('_'));
    }

    const QString base = stem;
    int suffix = 2;
    while (usedNames && usedNames->contains(stem.toCaseFolded()))
    {
        stem = QStringLiteral("%1_%2").arg(base).arg(suffix++);
    }
    if (usedNames)
    {
        usedNames->insert(stem.toCaseFolded());
    }
    return stem;
}

QString quotedListToken(QString token)
{
    token = QDir::fromNativeSeparators(token);
    token.replace(QLatin1Char('"'), QStringLiteral("\\\""));
    return QStringLiteral("\"") + token + QStringLiteral("\"");
}

bool fail(const QString &message, QString *errorMessage)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
    return false;
}

struct CameraToWrite
{
    QString imagePath;
    QString fileName;
    xjw::Camera camera;
};

} // namespace

bool exportFinalBaCameras(const QStringList &images,
                          const QMap<QString, QJsonObject> &finalCameraMetadata,
                          const QString &outputDir,
                          FinalBaCameraExportResult *result,
                          QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (result)
    {
        *result = {};
    }
    if (images.isEmpty())
    {
        return fail(QStringLiteral("无法导出最终 BA 相机：输入影像集合为空"), errorMessage);
    }
    if (outputDir.trimmed().isEmpty())
    {
        return fail(QStringLiteral("最终 BA 相机导出目录为空"), errorMessage);
    }
    const QString targetDir = QDir::cleanPath(QFileInfo(outputDir).absoluteFilePath());
    if (QFileInfo::exists(targetDir))
    {
        return fail(
            QStringLiteral("最终 BA 相机导出目录已存在，拒绝覆盖: %1").arg(targetDir),
            errorMessage);
    }

    QMap<QString, QJsonObject> metadataByPath;
    for (auto it = finalCameraMetadata.constBegin(); it != finalCameraMetadata.constEnd(); ++it)
    {
        metadataByPath.insert(normalizedCameraLookupKey(it.key()), it.value());
    }

    std::vector<CameraToWrite> cameras;
    cameras.reserve(static_cast<std::size_t>(images.size()));
    QSet<QString> seenImages;
    QSet<QString> usedNames;
    for (const QString &image : images)
    {
        const QString normalizedImage = normalizedCameraLookupKey(image);
        if (normalizedImage.isEmpty() || seenImages.contains(normalizedImage))
        {
            return fail(
                QStringLiteral("无法导出最终 BA 相机：输入影像为空或重复: %1").arg(image),
                errorMessage);
        }
        seenImages.insert(normalizedImage);
        if (!QFileInfo::exists(image))
        {
            return fail(
                QStringLiteral("无法导出最终 BA 相机：输入影像不存在: %1").arg(image),
                errorMessage);
        }

        const auto metadata = metadataByPath.constFind(normalizedImage);
        if (metadata == metadataByPath.constEnd())
        {
            return fail(
                QStringLiteral("无法导出最终 BA 相机：正式模型没有影像对应的相机: %1").arg(image),
                errorMessage);
        }
        xjw::Camera camera;
        if (!xjw::common::project::cameraFromJson(metadata.value(), &camera) || !camera.isValid())
        {
            return fail(
                QStringLiteral("无法导出最终 BA 相机：相机元数据无效: %1").arg(image),
                errorMessage);
        }

        CameraToWrite entry;
        entry.imagePath = QDir::cleanPath(QFileInfo(image).absoluteFilePath());
        entry.fileName = safeCameraStem(image, &usedNames) + QStringLiteral(".tsai");
        entry.camera = camera;
        cameras.push_back(std::move(entry));
    }

    const QFileInfo targetInfo(targetDir);
    const QString parentPath = targetInfo.absolutePath();
    if (!QDir().mkpath(parentPath))
    {
        return fail(
            QStringLiteral("无法创建最终 BA 相机导出目录的父目录: %1").arg(parentPath),
            errorMessage);
    }
    QTemporaryDir staging(
        QDir(parentPath).filePath(QStringLiteral(".plascan_camera_export_XXXXXX")));
    if (!staging.isValid())
    {
        return fail(
            QStringLiteral("无法创建最终 BA 相机暂存目录: %1").arg(parentPath),
            errorMessage);
    }
    const QString stagingCameraDir = QDir(staging.path()).filePath(QStringLiteral("cameras"));
    if (!QDir().mkpath(stagingCameraDir))
    {
        return fail(QStringLiteral("无法创建相机子目录"), errorMessage);
    }

    QStringList listLines;
    QStringList finalCameraPaths;
    listLines.reserve(static_cast<qsizetype>(cameras.size()));
    finalCameraPaths.reserve(static_cast<qsizetype>(cameras.size()));
    const QDir finalDir(targetDir);
    for (const CameraToWrite &entry : cameras)
    {
        const QString stagedPath = QDir(stagingCameraDir).filePath(entry.fileName);
        if (!entry.camera.saveToFile(xjw::common::io::toUtf8Path(stagedPath)))
        {
            return fail(QStringLiteral("无法写入最终 BA 相机: %1").arg(stagedPath), errorMessage);
        }
        const QString relativeImage = finalDir.relativeFilePath(entry.imagePath);
        const QString relativeCamera = QStringLiteral("cameras/%1").arg(entry.fileName);
        listLines.append(
            quotedListToken(relativeImage) + QLatin1Char(' ') + quotedListToken(relativeCamera));
        finalCameraPaths.append(finalDir.filePath(relativeCamera));
    }

    const QString stagedList = QDir(staging.path()).filePath(QStringLiteral("image_camera.lis"));
    QSaveFile listFile(stagedList);
    if (!listFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        return fail(QStringLiteral("无法写入影像相机清单: %1").arg(stagedList), errorMessage);
    }
    QTextStream stream(&listFile);
    stream.setEncoding(QStringConverter::Utf8);
    for (const QString &line : listLines)
    {
        stream << line << '\n';
    }
    stream.flush();
    if (stream.status() != QTextStream::Ok || !listFile.commit())
    {
        return fail(QStringLiteral("无法提交影像相机清单: %1").arg(stagedList), errorMessage);
    }

    QDir parentDir(parentPath);
    const QString stagingName = QFileInfo(staging.path()).fileName();
    const QString targetName = targetInfo.fileName();
    staging.setAutoRemove(false);
    if (!parentDir.rename(stagingName, targetName))
    {
        staging.setAutoRemove(true);
        return fail(QStringLiteral("无法提交最终 BA 相机目录: %1").arg(targetDir), errorMessage);
    }

    if (result)
    {
        result->outputDir = targetDir;
        result->imageCameraList = finalDir.filePath(QStringLiteral("image_camera.lis"));
        result->cameraPaths = finalCameraPaths;
    }
    return true;
}

} // namespace xjw::cli

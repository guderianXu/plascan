#include "FinalBaCameraExporter.h"

#include "io/PathIO.h"
#include <placamera/tsai.h>

#include <placamera/frame_camera.h>

#include <QDir>
#include <QFileInfo>
#include <QSaveFile>
#include <QSet>
#include <QStringConverter>
#include <QTemporaryDir>
#include <QTextStream>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace xjw::cli
{
    namespace
    {

        QString safeCameraStem(const QString& imagePath, QSet<QString>* usedNames)
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
            const bool reserved =
                upper == QStringLiteral("CON") || upper == QStringLiteral("PRN") || upper == QStringLiteral("AUX") ||
                upper == QStringLiteral("NUL") ||
                (upper.size() == 4 &&
                 (upper.startsWith(QStringLiteral("COM")) || upper.startsWith(QStringLiteral("LPT"))) &&
                 upper.at(3) >= QLatin1Char('1') && upper.at(3) <= QLatin1Char('9'));
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

        bool fail(const QString& message, QString* errorMessage)
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
            std::shared_ptr<const placamera::FramePinholeModel> camera;
        };

    } // namespace

    bool exportFinalBaCameras(const QStringList& images,
                              const std::vector<placamera::ImageId>& imageIds,
                              const placamera::CameraInstanceSet& finalCameras,
                              const QString& outputDir,
                              FinalBaCameraExportResult* result,
                              QString* errorMessage)
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
        if (imageIds.size() != static_cast<std::size_t>(images.size()))
        {
            return fail(QStringLiteral("无法导出最终 BA 相机：ImageId 与影像数量不一致"), errorMessage);
        }
        if (outputDir.trimmed().isEmpty())
        {
            return fail(QStringLiteral("最终 BA 相机导出目录为空"), errorMessage);
        }
        const QString targetDir = QDir::cleanPath(QFileInfo(outputDir).absoluteFilePath());
        if (QFileInfo::exists(targetDir))
        {
            const QFileInfo existingTarget(targetDir);
            if (!existingTarget.isDir() ||
                !QDir(targetDir).entryList(QDir::AllEntries | QDir::NoDotAndDotDot).isEmpty())
            {
                return fail(QStringLiteral("最终 BA 相机导出目录已存在且非空，拒绝覆盖: %1").arg(targetDir),
                            errorMessage);
            }
        }

        std::vector<CameraToWrite> cameras;
        cameras.reserve(static_cast<std::size_t>(images.size()));
        QSet<QString> seenImages;
        QSet<QString> usedNames;
        QSet<QString> seenImageIds;
        for (int index = 0; index < images.size(); ++index)
        {
            const QString& image = images.at(index);
            const QString normalizedImage = QDir::cleanPath(QFileInfo(image).absoluteFilePath());
#ifdef Q_OS_WIN
            const QString imageKey = normalizedImage.toCaseFolded();
#else
            const QString imageKey = normalizedImage;
#endif
            if (normalizedImage.isEmpty() || seenImages.contains(imageKey))
            {
                return fail(QStringLiteral("无法导出最终 BA 相机：输入影像为空或重复: %1").arg(image), errorMessage);
            }
            seenImages.insert(imageKey);
            if (!QFileInfo::exists(image))
            {
                return fail(QStringLiteral("无法导出最终 BA 相机：输入影像不存在: %1").arg(image), errorMessage);
            }

            const placamera::ImageId& imageId = imageIds.at(static_cast<std::size_t>(index));
            const QString imageIdText = QString::fromStdString(imageId.value());
            if (seenImageIds.contains(imageIdText))
            {
                return fail(QStringLiteral("无法导出最终 BA 相机：重复 ImageId: %1").arg(imageIdText), errorMessage);
            }
            seenImageIds.insert(imageIdText);
            const auto lookup = finalCameras.forImage(imageId);
            const auto camera =
                lookup.ok() ? std::dynamic_pointer_cast<const placamera::FramePinholeModel>(lookup.value()) : nullptr;
            if (!camera)
            {
                return fail(
                    QStringLiteral("无法导出最终 BA 相机：正式模型没有影像对应的帧相机: %1 (%2)")
                        .arg(image,
                             lookup.ok() ? QStringLiteral("非帧相机") : QString::fromStdString(lookup.message())),
                    errorMessage);
            }

            cameras.push_back(
                CameraToWrite{normalizedImage, safeCameraStem(image, &usedNames) + QStringLiteral(".tsai"), camera});
        }

        const QFileInfo targetInfo(targetDir);
        const QString parentPath = targetInfo.absolutePath();
        if (!QDir().mkpath(parentPath))
        {
            return fail(QStringLiteral("无法创建最终 BA 相机导出目录的父目录: %1").arg(parentPath), errorMessage);
        }
        QTemporaryDir staging(QDir(parentPath).filePath(QStringLiteral(".plascan_camera_export_XXXXXX")));
        if (!staging.isValid())
        {
            return fail(QStringLiteral("无法创建最终 BA 相机暂存目录: %1").arg(parentPath), errorMessage);
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
        for (const CameraToWrite& entry : cameras)
        {
            const QString stagedPath = QDir(stagingCameraDir).filePath(entry.fileName);
            const placamera::TsaiFramePinhole tsai_camera{std::shared_ptr<const placamera::FramePinholeDefinition>(
                                                              entry.camera, &entry.camera->pinholeDefinition()),
                                                          entry.camera->pose()};
            const auto saved = placamera::saveTsaiFramePinhole(tsai_camera, xjw::common::io::toUtf8Path(stagedPath));
            if (!saved)
            {
                return fail(QStringLiteral("无法写入最终 BA 相机: %1 (%2)")
                                .arg(stagedPath, QString::fromStdString(saved.message())),
                            errorMessage);
            }
            const QString relativeImage = finalDir.relativeFilePath(entry.imagePath);
            const QString relativeCamera = QStringLiteral("cameras/%1").arg(entry.fileName);
            listLines.append(quotedListToken(relativeImage) + QLatin1Char(' ') + quotedListToken(relativeCamera));
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
        for (const QString& line : listLines)
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
        if (QFileInfo::exists(targetDir) &&
            (!QDir(targetDir).entryList(QDir::AllEntries | QDir::NoDotAndDotDot).isEmpty() ||
             !parentDir.rmdir(targetName)))
        {
            staging.setAutoRemove(true);
            return fail(QStringLiteral("最终 BA 相机导出目录在提交前已非空或无法替换: %1").arg(targetDir),
                        errorMessage);
        }
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

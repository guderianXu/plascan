#include "ProjectCameraImportService.h"

#include "io/ImageIO.h"
#include "io/PathIO.h"

#include <placamera/project_import.h>

#include <QDir>
#include <QFileInfo>
#include <QMap>
#include <QUuid>

#include <algorithm>
#include <exception>
#include <utility>

namespace xjw::gui::project
{
    namespace
    {
        bool isCancelled(const std::atomic<bool>* cancelFlag)
        {
            return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
        }

        QString absolutePath(const QString& path)
        {
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
        }

        QString cameraProjectFormatName(placamera::CameraProjectFormat format)
        {
            switch (format)
            {
            case placamera::CameraProjectFormat::MiddleburyPar:
                return QStringLiteral("middlebury-par");
            case placamera::CameraProjectFormat::EpflCamera:
                return QStringLiteral("epfl-camera");
            case placamera::CameraProjectFormat::ColmapText:
                return QStringLiteral("colmap-text");
            case placamera::CameraProjectFormat::MetashapeXml:
                return QStringLiteral("metashape-xml");
            case placamera::CameraProjectFormat::MetashapeReferenceText:
                return QStringLiteral("metashape-reference");
            case placamera::CameraProjectFormat::Auto:
                return QStringLiteral("auto");
            }
            return QStringLiteral("unknown");
        }

        bool readImageSize(const QString& imagePath, placamera::ImageSize* size, QString* error)
        {
            QString image_error;
            const QSize image_size = xjw::common::io::readImageSize(imagePath, &image_error);
            if (!image_size.isValid())
            {
                if (error)
                {
                    *error = image_error;
                }
                return false;
            }
            *size = {image_size.width(), image_size.height()};
            return true;
        }

        QString leafName(std::string name)
        {
            QString path = QString::fromStdString(std::move(name));
            path.replace(QLatin1Char('\\'), QLatin1Char('/'));
            return QFileInfo(path).fileName();
        }

        QString withoutMetashapeNumericSuffix(QString stem)
        {
            const qsizetype separator = stem.lastIndexOf(QLatin1Char('_'));
            if (separator < 0 || separator + 1 >= stem.size())
            {
                return stem;
            }
            const QString suffix = stem.sliced(separator + 1);
            if (std::all_of(suffix.cbegin(), suffix.cend(), [](QChar ch) { return ch.isDigit(); }))
            {
                stem.truncate(separator);
            }
            return stem;
        }

        struct ProjectImageIndex
        {
            QMap<QString, QStringList> byFileName;
            QMap<QString, QStringList> byStem;
        };

        ProjectImageIndex indexProjectImages(const QStringList& projectImages)
        {
            ProjectImageIndex index;
            for (const QString& image : projectImages)
            {
                const QString path = absolutePath(image);
                const QFileInfo info(path);
                index.byFileName[info.fileName().toLower()].push_back(path);
                index.byStem[info.completeBaseName().toLower()].push_back(path);
            }
            return index;
        }

        QStringList matchingImages(const ProjectImageIndex& index,
                                   const placamera::ImportedProjectCamera& imported,
                                   placamera::CameraProjectFormat format)
        {
            const QString label = leafName(imported.imageName);
            QStringList matches = index.byFileName.value(label.toLower());
            if (matches.isEmpty())
            {
                const QString stem = QFileInfo(label).completeBaseName().toLower();
                matches = index.byStem.value(stem);
                if (matches.isEmpty() && format == placamera::CameraProjectFormat::MetashapeXml)
                {
                    matches = index.byStem.value(withoutMetashapeNumericSuffix(stem));
                }
            }
            matches.removeDuplicates();
            return matches;
        }

        bool prepareProjectCamera(const placamera::ImportedProjectCamera& imported,
                                  const QString& imagePath,
                                  const QString& sourcePath,
                                  const QString& sourceFormat,
                                  PreparedFrameCameraImport* output)
        {
            output->imageAbsPath = absolutePath(imagePath);
            output->sourceFile = absolutePath(sourcePath);
            output->sourceFormat = sourceFormat;
            output->sourceKind = QStringLiteral("camera_project_import");
            output->camera.reset();
            output->acquisition = imported.acquisition;
            output->error.clear();

            if (!readImageSize(output->imageAbsPath, &output->imageSize, &output->error))
            {
                return false;
            }
            if (imported.sourceColmapCamera.has_value() &&
                (imported.sourceColmapCamera->width != output->imageSize.samples ||
                 imported.sourceColmapCamera->height != output->imageSize.lines))
            {
                output->error = QStringLiteral("COLMAP 相机栅格尺寸与工程影像不一致: %1").arg(output->imageAbsPath);
                return false;
            }

            auto camera = placamera::makeCentralCameraGeometry(
                imported,
                placamera::CameraDefinitionId("project-import-" +
                                              QUuid::createUuid().toString(QUuid::WithoutBraces).toStdString()),
                placamera::FrameId("project-world"));
            if (!camera)
            {
                const QString model = imported.sourceColmapCamera.has_value()
                                          ? QString::fromStdString(imported.sourceColmapCamera->model)
                                          : sourceFormat;
                output->error = QStringLiteral("相机模型 %1 无法无损导入: %2 (%3)")
                                    .arg(model, output->imageAbsPath, QString::fromStdString(camera.message()));
                return false;
            }
            output->camera = camera.takeValue();
            return true;
        }
    } // namespace

    CameraProjectImportStatus buildCameraProjectImport(const QString& inputPath,
                                                       const QStringList& projectImages,
                                                       CameraProjectImportResult* out,
                                                       const std::atomic<bool>* cancelFlag,
                                                       const CameraImportProgress& progress)
    {
        if (!out)
        {
            return CameraProjectImportStatus::ParseFailed;
        }
        *out = {};
        if (isCancelled(cancelFlag))
        {
            return CameraProjectImportStatus::Cancelled;
        }
        if (projectImages.isEmpty())
        {
            return CameraProjectImportStatus::NoProjectImages;
        }

        try
        {
            const auto imported = placamera::importCameraProject(xjw::common::io::toUtf8Path(inputPath));
            if (!imported)
            {
                out->error = QString::fromStdString(imported.message());
                return CameraProjectImportStatus::ParseFailed;
            }
            const auto& project = imported.value();
            out->sourceFormat = cameraProjectFormatName(project.format);
            for (const std::string& warning : project.warnings)
            {
                out->warnings.push_back(QString::fromStdString(warning));
            }
            if (project.cameras.empty())
            {
                out->error = QStringLiteral("所选文件不包含可绑定的相机几何");
                return CameraProjectImportStatus::NoImportable;
            }

            const ProjectImageIndex image_index = indexProjectImages(projectImages);
            struct Match
            {
                const placamera::ImportedProjectCamera* camera = nullptr;
                QString imagePath;
            };
            std::vector<Match> matches;
            QMap<QString, int> target_counts;
            matches.reserve(project.cameras.size());
            for (const auto& camera : project.cameras)
            {
                const QStringList candidates = matchingImages(image_index, camera, project.format);
                if (candidates.isEmpty())
                {
                    ++out->unmatchedCount;
                    continue;
                }
                if (candidates.size() != 1)
                {
                    ++out->ambiguousCount;
                    continue;
                }
                matches.push_back({&camera, candidates.front()});
                ++target_counts[candidates.front()];
            }

            if (progress)
            {
                progress(0, static_cast<int>(matches.size()));
            }
            int completed = 0;
            for (const Match& match : matches)
            {
                if (isCancelled(cancelFlag))
                {
                    return CameraProjectImportStatus::Cancelled;
                }
                if (target_counts.value(match.imagePath) != 1)
                {
                    ++out->ambiguousCount;
                }
                else
                {
                    PreparedFrameCameraImport camera;
                    if (prepareProjectCamera(*match.camera, match.imagePath, inputPath, out->sourceFormat, &camera))
                    {
                        out->cameras.push_back(std::move(camera));
                    }
                    else
                    {
                        ++out->unsupportedCount;
                        out->importErrors.push_back(camera.error);
                    }
                }
                if (progress)
                {
                    progress(++completed, static_cast<int>(matches.size()));
                }
            }
        }
        catch (const std::exception& exception)
        {
            out->error = QString::fromUtf8(exception.what());
            return CameraProjectImportStatus::ParseFailed;
        }

        if (out->cameras.empty())
        {
            return CameraProjectImportStatus::NoImportable;
        }
        return CameraProjectImportStatus::Ok;
    }

} // namespace xjw::gui::project

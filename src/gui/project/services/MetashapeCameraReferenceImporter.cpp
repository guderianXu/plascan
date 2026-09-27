#include "MetashapeCameraReferenceImporter.h"

#include <placamera/project_import.h>

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace xjw::gui::reference_import
{
    namespace
    {

        std::filesystem::path filePath(const QString& value)
        {
            return std::filesystem::path(value.toStdU16String());
        }

        QString fromUtf8(const std::string& value)
        {
            return QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
        }

    } // namespace

    bool importMetashapeCameraReferenceTxt(const QString& cameraTxtPath,
                                           const QString& gnssOffsetTxtPath,
                                           MetashapeCameraReferenceImportResult* result,
                                           QString* errorMessage)
    {
        if (!result)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("相机参考导入结果输出不能为空");
            }
            return false;
        }
        if (cameraTxtPath.trimmed().isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("相机参考文件路径不能为空");
            }
            return false;
        }

        try
        {
            const auto imported = placamera::importCameraProject(
                filePath(cameraTxtPath),
                placamera::CameraProjectFormat::MetashapeReferenceText,
                gnssOffsetTxtPath.isEmpty() ? std::filesystem::path{} : filePath(gnssOffsetTxtPath));
            if (!imported)
            {
                if (errorMessage)
                {
                    *errorMessage = fromUtf8(imported.message());
                }
                return false;
            }
            const auto& project = imported.value();
            MetashapeCameraReferenceImportResult converted;
            for (const auto& source : project.references)
            {
                RawCameraReferenceRecord record;
                record.fileName = fromUtf8(source.imageName);
                record.wgs84LatitudeDegrees = source.latitudeDegrees;
                record.wgs84LongitudeDegrees = source.longitudeDegrees;
                record.wgs84EllipsoidalHeightMeters = source.ellipsoidalHeightMeters;
                record.rollDegrees = source.rollDegrees;
                record.pitchDegrees = source.pitchDegrees;
                record.yawDegrees = source.yawDegrees;
                record.timeText = fromUtf8(source.timeText);
                record.stdDevNorthMeters = source.stdDevNorthMeters;
                record.stdDevEastMeters = source.stdDevEastMeters;
                record.stdDevUpMeters = source.stdDevUpMeters;
                record.stdDevHorizontalMeters = source.stdDevHorizontalMeters;
                converted.records.push_back(std::move(record));
            }
            if (project.leverArm)
            {
                converted.leverArm =
                    LeverArm{project.leverArm->xMeters, project.leverArm->yMeters, project.leverArm->zMeters};
            }
            for (const auto& warning : project.warnings)
            {
                converted.warnings.push_back(fromUtf8(warning));
            }
            *result = std::move(converted);
            if (errorMessage)
            {
                errorMessage->clear();
            }
            return true;
        }
        catch (const std::exception& error)
        {
            if (errorMessage)
            {
                *errorMessage = fromUtf8(error.what());
            }
            return false;
        }
    }

} // namespace xjw::gui::reference_import

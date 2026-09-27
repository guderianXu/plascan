#pragma once

#include "placamera/colmap.h"
#include "placamera/dataset_formats.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace placamera
{

    enum class CameraProjectFormat
    {
        Auto,
        MiddleburyPar,
        EpflCamera,
        ColmapText,
        MetashapeXml,
        MetashapeReferenceText
    };

    /** Raw Metashape GNSS/YPR observation. Its orientation and lever-arm axes are unresolved. */
    struct UnresolvedCameraReference
    {
        std::string imageName;
        double latitudeDegrees = 0.0;
        double longitudeDegrees = 0.0;
        double ellipsoidalHeightMeters = 0.0;
        double rollDegrees = 0.0;
        double pitchDegrees = 0.0;
        double yawDegrees = 0.0;
        std::string timeText;
        std::optional<double> stdDevNorthMeters;
        std::optional<double> stdDevEastMeters;
        std::optional<double> stdDevUpMeters;
        std::optional<double> stdDevHorizontalMeters;
    };

    struct UnresolvedLeverArm
    {
        double xMeters = 0.0;
        double yMeters = 0.0;
        double zMeters = 0.0;
    };

    /** One unbound imported camera plus optional source-format identity. */
    struct ImportedProjectCamera : ImportedCamera
    {
        std::optional<ColmapCamera> sourceColmapCamera;
        std::optional<int> sourceImageId;
    };

    using ImportedCameraFrame = ImportedProjectCamera;

    struct CameraProjectImportResult
    {
        CameraProjectFormat format = CameraProjectFormat::Auto;
        std::vector<ImportedProjectCamera> cameras;
        std::vector<UnresolvedCameraReference> references;
        std::optional<UnresolvedLeverArm> leverArm;
        std::vector<std::string> warnings;
    };

    /** Import camera geometry or raw reference TXT without binding application image IDs.
     *  gnssOffsetPath is only valid for MetashapeReferenceText.
     */
    Result<CameraProjectImportResult> importCameraProject(const std::filesystem::path& path,
                                                          CameraProjectFormat format = CameraProjectFormat::Auto,
                                                          const std::filesystem::path& gnssOffsetPath = {});

    /** Convert an already-extracted Metashape chunk document, for example from chunk.zip. */
    Result<CameraProjectImportResult> importMetashapeDocument(std::string_view xml);

} // namespace placamera

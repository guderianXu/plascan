#include "plafs/PlaChunkLayout.h"

#include "file/FileIO.h"

#include <string_view>
#include <utility>

namespace xjw::common::plafs
{

    std::optional<PlaChunkLayout> PlaChunkLayout::open(const std::filesystem::path& root, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (root.empty())
        {
            if (error)
            {
                *error = "chunk root path is empty";
            }
            return std::nullopt;
        }
        return PlaChunkLayout(root.lexically_normal());
    }

    PlaChunkLayout::PlaChunkLayout(std::filesystem::path root) : _root(std::move(root))
    {
    }

    const std::filesystem::path& PlaChunkLayout::root() const noexcept
    {
        return _root;
    }

    std::filesystem::path PlaChunkLayout::assetsDirectory() const
    {
        return _root / "assets";
    }

    std::filesystem::path PlaChunkLayout::imageMatchesDirectory() const
    {
        return assetsDirectory() / "image_matches";
    }

    std::filesystem::path PlaChunkLayout::tiePointsDirectory() const
    {
        return assetsDirectory() / "tie_points";
    }

    std::filesystem::path PlaChunkLayout::controlPointsDirectory() const
    {
        return assetsDirectory() / "control_points";
    }

    std::filesystem::path PlaChunkLayout::cameraReferencesDirectory() const
    {
        return assetsDirectory() / "camera_references";
    }

    std::filesystem::path PlaChunkLayout::importedDirectory() const
    {
        return assetsDirectory() / "imported";
    }

    std::filesystem::path PlaChunkLayout::importedCategoryDirectory(std::string_view category) const
    {
        if (category.empty())
        {
            return {};
        }

        const std::filesystem::path component = xjw::common::file::pathFromUtf8(category);
        if (component.empty() || component.is_absolute() || component.has_root_name() || component.has_parent_path() ||
            component == std::filesystem::path(".") || component == std::filesystem::path(".."))
        {
            return {};
        }
        return importedDirectory() / component;
    }

    std::filesystem::path PlaChunkLayout::packedDirectory() const
    {
        return assetsDirectory() / "packed";
    }

    std::filesystem::path PlaChunkLayout::masksDirectory() const
    {
        return assetsDirectory() / "masks";
    }

    std::filesystem::path PlaChunkLayout::bundleAdjustDirectory() const
    {
        return _root / "bundle_adjust";
    }

    std::filesystem::path PlaChunkLayout::reportsDirectory() const
    {
        return _root / "reports";
    }

    std::filesystem::path PlaChunkLayout::reconstructionDirectory() const
    {
        return _root / "reconstruction";
    }

    std::filesystem::path PlaChunkLayout::resourcesDirectory() const
    {
        return _root / "resources";
    }

    std::filesystem::path PlaChunkLayout::temporaryDirectory() const
    {
        return _root / ".plascan_tmp";
    }

    std::filesystem::path PlaChunkLayout::temporaryFilesPath() const
    {
        return temporaryDirectory() / "project_files.json";
    }

    std::filesystem::path PlaChunkLayout::temporaryResultsPath() const
    {
        return temporaryDirectory() / "project_results.json";
    }

    std::filesystem::path PlaChunkLayout::temporaryConfigPath() const
    {
        return temporaryDirectory() / "project_config.json";
    }

    std::filesystem::path PlaChunkLayout::temporaryUiStatePath() const
    {
        return temporaryDirectory() / "project_ui_state.json";
    }

    std::filesystem::path PlaChunkLayout::markerSetPath() const
    {
        return controlPointsDirectory() / "marker_set.json";
    }

    std::filesystem::path PlaChunkLayout::markerDetectionReviewPath() const
    {
        return controlPointsDirectory() / "detection_review.json";
    }

    std::filesystem::path PlaChunkLayout::cameraReferenceSetPath() const
    {
        return cameraReferencesDirectory() / "camera_reference_set.json";
    }

} // namespace xjw::common::plafs

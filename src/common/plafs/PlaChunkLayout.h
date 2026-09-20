#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace xjw::common::plafs
{

    /// Physical paths owned by one active PlaScan Chunk directory.
    ///
    /// This value object only describes the on-disk layout. It does not know
    /// how a Chunk was selected, read project metadata, open archives, or
    /// remove files. Those responsibilities stay in the project/session
    /// adapters.
    class PlaChunkLayout
    {
    public:
        static std::optional<PlaChunkLayout> open(const std::filesystem::path& root, std::string* error = nullptr);

        const std::filesystem::path& root() const noexcept;

        std::filesystem::path assetsDirectory() const;
        std::filesystem::path imageMatchesDirectory() const;
        std::filesystem::path tiePointsDirectory() const;
        std::filesystem::path controlPointsDirectory() const;
        std::filesystem::path cameraReferencesDirectory() const;
        std::filesystem::path importedDirectory() const;
        std::filesystem::path importedCategoryDirectory(std::string_view category) const;
        std::filesystem::path packedDirectory() const;
        std::filesystem::path masksDirectory() const;

        std::filesystem::path bundleAdjustDirectory() const;
        std::filesystem::path reportsDirectory() const;
        std::filesystem::path reconstructionDirectory() const;
        std::filesystem::path resourcesDirectory() const;

        std::filesystem::path temporaryDirectory() const;
        std::filesystem::path temporaryFilesPath() const;
        std::filesystem::path temporaryResultsPath() const;
        std::filesystem::path temporaryConfigPath() const;
        std::filesystem::path temporaryUiStatePath() const;

        std::filesystem::path markerSetPath() const;
        std::filesystem::path markerDetectionReviewPath() const;
        std::filesystem::path cameraReferenceSetPath() const;

    private:
        explicit PlaChunkLayout(std::filesystem::path root);

        std::filesystem::path _root;
    };

} // namespace xjw::common::plafs

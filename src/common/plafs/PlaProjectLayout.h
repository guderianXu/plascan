#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace xjw::common::plafs
{

    /// Physical layout of a PlaScan `.plascan` project and its `.files` data.
    ///
    /// This class only calculates paths. It does not read the XML descriptor,
    /// create directories, open archives, or register a GUI session.
    class PlaProjectLayout
    {
    public:
        static std::optional<PlaProjectLayout> open(const std::filesystem::path& projectFile,
                                                    std::string* error = nullptr);

        const std::filesystem::path& projectFile() const noexcept;
        std::filesystem::path projectDirectory() const;
        std::filesystem::path filesDirectory() const;
        std::filesystem::path metadataArchive() const;
        std::filesystem::path sharedDirectory() const;
        std::filesystem::path sharedImagesDirectory() const;

        std::filesystem::path chunkDirectory(int directoryNumber) const;
        std::filesystem::path chunkArchive(int directoryNumber) const;
        std::filesystem::path assetsDirectory(int directoryNumber) const;
        std::filesystem::path imageMatchesDirectory(int directoryNumber) const;
        std::filesystem::path tiePointsDirectory(int directoryNumber) const;
        std::filesystem::path controlPointsDirectory(int directoryNumber) const;
        std::filesystem::path importedDirectory(int directoryNumber) const;
        std::filesystem::path reportsDirectory(int directoryNumber) const;
        std::filesystem::path reconstructionDirectory(int directoryNumber) const;
        std::filesystem::path bundleAdjustDirectory(int directoryNumber) const;
        std::filesystem::path resourcesDirectory(int directoryNumber) const;

    private:
        explicit PlaProjectLayout(std::filesystem::path projectFile);

        std::filesystem::path _projectFile;
    };

} // namespace xjw::common::plafs

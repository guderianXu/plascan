#include "plafs/PlaProjectLayout.h"

#include "file/FileIO.h"
#include "plafs/PlaChunkLayout.h"

#include <string_view>
#include <system_error>
#include <utility>

namespace xjw::common::plafs
{

    namespace
    {

        std::filesystem::path absoluteNormalized(std::filesystem::path path, std::string* error)
        {
            if (path.empty())
            {
                if (error)
                {
                    *error = "project file path is empty";
                }
                return {};
            }

            std::error_code filesystemError;
            path = std::filesystem::absolute(path, filesystemError).lexically_normal();
            if (filesystemError)
            {
                if (error)
                {
                    *error = "cannot normalize project file path '" + xjw::common::file::pathToUtf8(path) +
                             "': " + filesystemError.message();
                }
                return {};
            }
            return path;
        }

        std::filesystem::path appendChunkPath(const std::filesystem::path& chunk, std::string_view relative)
        {
            return chunk.empty() ? std::filesystem::path{} : chunk / relative;
        }

        using ChunkPathMember = std::filesystem::path (PlaChunkLayout::*)() const;

        std::filesystem::path chunkPath(const PlaProjectLayout& layout, int directoryNumber, ChunkPathMember member)
        {
            const auto chunk = PlaChunkLayout::open(layout.chunkDirectory(directoryNumber));
            return chunk ? (chunk.value().*member)() : std::filesystem::path{};
        }

    } // namespace

    std::optional<PlaProjectLayout> PlaProjectLayout::open(const std::filesystem::path& projectFile, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        std::filesystem::path normalized = absoluteNormalized(projectFile, error);
        if (normalized.empty())
        {
            return std::nullopt;
        }
        return PlaProjectLayout(std::move(normalized));
    }

    PlaProjectLayout::PlaProjectLayout(std::filesystem::path projectFile) : _projectFile(std::move(projectFile))
    {
    }

    const std::filesystem::path& PlaProjectLayout::projectFile() const noexcept
    {
        return _projectFile;
    }

    std::filesystem::path PlaProjectLayout::projectDirectory() const
    {
        return _projectFile.parent_path();
    }

    std::filesystem::path PlaProjectLayout::filesDirectory() const
    {
        std::filesystem::path directoryName = _projectFile.stem();
        directoryName += ".files";
        return projectDirectory() / directoryName;
    }

    std::filesystem::path PlaProjectLayout::metadataArchive() const
    {
        return filesDirectory() / "project.zip";
    }

    std::filesystem::path PlaProjectLayout::sharedDirectory() const
    {
        return filesDirectory() / "shared";
    }

    std::filesystem::path PlaProjectLayout::sharedImagesDirectory() const
    {
        return sharedDirectory() / "images";
    }

    std::filesystem::path PlaProjectLayout::chunkDirectory(int directoryNumber) const
    {
        if (directoryNumber <= 0)
        {
            return {};
        }
        return filesDirectory() / std::to_string(directoryNumber);
    }

    std::filesystem::path PlaProjectLayout::chunkArchive(int directoryNumber) const
    {
        return appendChunkPath(chunkDirectory(directoryNumber), "chunk.zip");
    }

    std::filesystem::path PlaProjectLayout::assetsDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::assetsDirectory);
    }

    std::filesystem::path PlaProjectLayout::imageMatchesDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::imageMatchesDirectory);
    }

    std::filesystem::path PlaProjectLayout::tiePointsDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::tiePointsDirectory);
    }

    std::filesystem::path PlaProjectLayout::controlPointsDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::controlPointsDirectory);
    }

    std::filesystem::path PlaProjectLayout::importedDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::importedDirectory);
    }

    std::filesystem::path PlaProjectLayout::reportsDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::reportsDirectory);
    }

    std::filesystem::path PlaProjectLayout::reconstructionDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::reconstructionDirectory);
    }

    std::filesystem::path PlaProjectLayout::bundleAdjustDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::bundleAdjustDirectory);
    }

    std::filesystem::path PlaProjectLayout::resourcesDirectory(int directoryNumber) const
    {
        return chunkPath(*this, directoryNumber, &PlaChunkLayout::resourcesDirectory);
    }

} // namespace xjw::common::plafs

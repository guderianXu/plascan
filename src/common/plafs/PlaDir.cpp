#include "plafs/PlaDir.h"

#include "file/FileIO.h"

#include <system_error>
#include <utility>

namespace xjw::common::plafs
{

    PlaDir::PlaDir(std::filesystem::path path) : _path(std::move(path))
    {
    }

    const std::filesystem::path& PlaDir::path() const noexcept
    {
        return _path;
    }

    bool PlaDir::empty() const noexcept
    {
        return _path.empty();
    }

    bool PlaDir::exists() const noexcept
    {
        std::error_code error;
        return !_path.empty() && std::filesystem::exists(_path, error) && !error;
    }

    bool PlaDir::isDirectory() const noexcept
    {
        std::error_code error;
        return !_path.empty() && std::filesystem::is_directory(_path, error) && !error;
    }

    bool PlaDir::ensure(std::string* error) const
    {
        return xjw::common::file::ensureDirectory(_path, error);
    }

    bool PlaDir::isEmpty(std::string* error) const
    {
        if (_path.empty())
        {
            if (error)
            {
                *error = "directory path is empty";
            }
            return false;
        }

        std::error_code filesystemError;
        const bool empty = std::filesystem::is_empty(_path, filesystemError);
        if (filesystemError)
        {
            if (error)
            {
                *error = "cannot inspect directory '" + xjw::common::file::pathToUtf8(_path) +
                         "': " + filesystemError.message();
            }
            return false;
        }
        if (error)
        {
            error->clear();
        }
        return empty;
    }

    bool PlaDir::contains(const std::filesystem::path& relativeName) const
    {
        if (_path.empty() || relativeName.empty() || relativeName.is_absolute())
        {
            return false;
        }

        for (const auto& component : relativeName)
        {
            if (component == std::filesystem::path(".."))
            {
                return false;
            }
        }

        std::error_code error;
        return std::filesystem::exists(_path / relativeName, error) && !error;
    }

} // namespace xjw::common::plafs

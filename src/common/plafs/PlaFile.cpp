#include "plafs/PlaFile.h"

#include "file/FileIO.h"

#include <system_error>
#include <utility>

namespace xjw::common::plafs
{

    PlaFile::PlaFile(std::filesystem::path path) : _path(std::move(path))
    {
    }

    const std::filesystem::path& PlaFile::path() const noexcept
    {
        return _path;
    }

    bool PlaFile::empty() const noexcept
    {
        return _path.empty();
    }

    bool PlaFile::exists() const noexcept
    {
        std::error_code error;
        return !_path.empty() && std::filesystem::exists(_path, error) && !error;
    }

    bool PlaFile::isRegularFile() const noexcept
    {
        std::error_code error;
        return !_path.empty() && std::filesystem::is_regular_file(_path, error) && !error;
    }

    bool PlaFile::read(std::string* bytes, std::string* error) const
    {
        return xjw::common::file::readFile(_path, bytes, error);
    }

    bool PlaFile::writeAtomic(std::string_view bytes, std::string* error) const
    {
        return xjw::common::file::writeFileAtomic(_path, bytes, error);
    }

} // namespace xjw::common::plafs

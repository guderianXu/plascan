#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace xjw::common::plafs
{

    /// Value object for one physical PlaScan file.
    ///
    /// PlaFile owns path and basic file operations only. It does not interpret
    /// project metadata or any domain-specific artifact format.
    class PlaFile
    {
    public:
        explicit PlaFile(std::filesystem::path path = {});

        const std::filesystem::path& path() const noexcept;
        bool empty() const noexcept;
        bool exists() const noexcept;
        bool isRegularFile() const noexcept;

        bool read(std::string* bytes, std::string* error = nullptr) const;
        bool writeAtomic(std::string_view bytes, std::string* error = nullptr) const;

    private:
        std::filesystem::path _path;
    };

} // namespace xjw::common::plafs

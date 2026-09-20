#pragma once

#include <filesystem>
#include <string>

namespace xjw::common::plafs
{

    /// Value object for one physical PlaScan directory.
    ///
    /// PlaDir deliberately has no recursive deletion API. Destructive cleanup
    /// belongs to a project/resource store that can validate ownership first.
    class PlaDir
    {
    public:
        explicit PlaDir(std::filesystem::path path = {});

        const std::filesystem::path& path() const noexcept;
        bool empty() const noexcept;
        bool exists() const noexcept;
        bool isDirectory() const noexcept;

        bool ensure(std::string* error = nullptr) const;
        bool isEmpty(std::string* error = nullptr) const;
        bool contains(const std::filesystem::path& relativeName) const;

    private:
        std::filesystem::path _path;
    };

} // namespace xjw::common::plafs

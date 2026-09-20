#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace xjw::common::file
{

    std::filesystem::path pathFromUtf8(std::string_view text);
    std::string pathToUtf8(const std::filesystem::path& path);
    std::filesystem::path absoluteNormalizedPath(const std::filesystem::path& path, std::string* error = nullptr);
    bool ensureDirectory(const std::filesystem::path& path, std::string* error = nullptr);
    bool readFile(const std::filesystem::path& path, std::string* bytes, std::string* error = nullptr);

    /// Owns a temporary file in the destination directory; destruction discards an uncommitted write.
    class AtomicFile
    {
    public:
        explicit AtomicFile(std::filesystem::path destination);
        ~AtomicFile();
        AtomicFile(const AtomicFile&) = delete;
        AtomicFile& operator=(const AtomicFile&) = delete;

        bool open(std::string* error = nullptr);
        std::ofstream& stream();
        bool commit(std::string* error = nullptr);

    private:
        std::filesystem::path _destination;
        std::filesystem::path _temporary;
        std::ofstream _stream;
        bool _committed = false;
    };

    bool writeFileAtomic(const std::filesystem::path& path, std::string_view bytes, std::string* error = nullptr);

} // namespace xjw::common::file

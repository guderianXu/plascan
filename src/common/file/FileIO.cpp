#include "file/FileIO.h"

#include <limits>
#include <system_error>

namespace xjw::common::file
{

    std::filesystem::path pathFromUtf8(std::string_view text)
    {
        std::u8string encoded;
        encoded.reserve(text.size());
        for (unsigned char character : text)
        {
            encoded.push_back(static_cast<char8_t>(character));
        }
        return std::filesystem::path(encoded);
    }

    std::string pathToUtf8(const std::filesystem::path& path)
    {
        const std::u8string encoded = path.u8string();
        return std::string(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    }

    std::filesystem::path absoluteNormalizedPath(const std::filesystem::path& path, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        std::error_code failure;
        if (path.empty())
        {
            if (error)
            {
                *error = "路径不能为空";
            }
            return {};
        }
        const auto absolute = std::filesystem::absolute(path, failure);
        if (failure)
        {
            if (error)
            {
                *error = "无法规范化路径 " + pathToUtf8(path) + ": " + failure.message();
            }
            return {};
        }
        return absolute.lexically_normal();
    }

    bool ensureDirectory(const std::filesystem::path& path, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        std::error_code failure;
        if (!path.empty() && (std::filesystem::create_directories(path, failure) ||
                              (!failure && std::filesystem::is_directory(path, failure))))
        {
            return true;
        }
        if (error)
        {
            *error = "无法创建目录 " + pathToUtf8(path) + ": " + (failure ? failure.message() : "路径为空或不是目录");
        }
        return false;
    }

    bool readFile(const std::filesystem::path& path, std::string* bytes, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        const auto fail = [&](std::string reason)
        {
            if (error)
            {
                *error = "读取文件失败 " + pathToUtf8(path) + ": " + reason;
            }
            return false;
        };
        if (!bytes)
        {
            return fail("输出缓冲区为空");
        }
        bytes->clear();
        std::error_code failure;
        if (!std::filesystem::is_regular_file(path, failure))
        {
            return fail(failure ? failure.message() : "输入不是普通文件");
        }
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
        {
            return fail("无法打开文件");
        }
        const auto length = input.tellg();
        if (length < 0 || static_cast<std::uintmax_t>(length) > bytes->max_size() ||
            static_cast<std::uintmax_t>(length) >
                static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        {
            return fail("文件长度无效或过大");
        }
        std::string buffer(static_cast<std::size_t>(length), '\0');
        input.seekg(0);
        if (!buffer.empty() && !input.read(buffer.data(), static_cast<std::streamsize>(buffer.size())))
        {
            return fail("无法完整读取文件");
        }
        if (!input)
        {
            return fail("无法定位文件起点");
        }
        *bytes = std::move(buffer);
        return true;
    }

} // namespace xjw::common::file

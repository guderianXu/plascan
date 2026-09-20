#include "file/JsonFile.h"
#include "file/FileIO.h"

namespace xjw::common::file
{

    bool readJson(const std::filesystem::path& path, nlohmann::json* document, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!document)
        {
            if (error)
            {
                *error = "JSON 输出为空: " + pathToUtf8(path);
            }
            return false;
        }
        std::string bytes;
        if (!readFile(path, &bytes, error))
        {
            return false;
        }
        try
        {
            auto parsed = nlohmann::json::parse(bytes);
            *document = std::move(parsed);
            return true;
        }
        catch (const nlohmann::json::exception& exception)
        {
            if (error)
            {
                *error = "JSON 解析失败 " + pathToUtf8(path) + ": " + exception.what();
            }
            return false;
        }
    }

    bool writeJsonAtomic(const std::filesystem::path& path, const nlohmann::json& document, std::string* error)
    {
        try
        {
            return writeFileAtomic(path, document.dump(), error);
        }
        catch (const nlohmann::json::exception& exception)
        {
            if (error)
            {
                *error = "JSON 序列化失败 " + pathToUtf8(path) + ": " + exception.what();
            }
            return false;
        }
    }

} // namespace xjw::common::file

#pragma once

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace xjw::common::file
{

    bool readJson(const std::filesystem::path& path, nlohmann::json* document, std::string* error = nullptr);
    bool
    writeJsonAtomic(const std::filesystem::path& path, const nlohmann::json& document, std::string* error = nullptr);

} // namespace xjw::common::file

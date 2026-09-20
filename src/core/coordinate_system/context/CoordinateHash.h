#pragma once

#include <string>
#include <string_view>

namespace xjw::coordinate_system
{

    std::string sha256Hash(std::string_view value);
    bool isSha256Hash(std::string_view value) noexcept;

} // namespace xjw::coordinate_system

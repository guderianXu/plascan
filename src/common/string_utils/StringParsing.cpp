#include <regex>
#include <stdexcept>

#include "string_utils/StringParsing.h"

namespace xjw::common::string_utils
{

bool extractDoublesFromText(const std::string &text, std::vector<double> &out)
{
    out.clear();
    static const std::regex kNumberPattern(
        R"([+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
    );

    const auto begin = std::sregex_iterator(text.begin(), text.end(), kNumberPattern);
    const auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it)
    {
        try
        {
            out.push_back(std::stod(it->str()));
        }
        catch (const std::invalid_argument &)
        {
        }
        catch (const std::out_of_range &)
        {
        }
    }
    return !out.empty();
}

} // namespace xjw::common::string_utils

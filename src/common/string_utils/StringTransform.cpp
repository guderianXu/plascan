#include "string_utils/StringTransform.h"

namespace xjw::common::string_utils
{
namespace
{

char asciiLower(char value)
{
    if (value >= 'A' && value <= 'Z')
    {
        return static_cast<char>(value + ('a' - 'A'));
    }
    return value;
}

bool isAsciiWhitespace(char value)
{
    return value == ' ' || value == '\t' || value == '\n' ||
           value == '\r' || value == '\f' || value == '\v';
}

} // namespace

std::string asciiLowerCopy(std::string_view text)
{
    std::string result;
    result.reserve(text.size());
    for (const char value : text)
    {
        result.push_back(asciiLower(value));
    }
    return result;
}

std::string trimAsciiWhitespace(std::string_view text)
{
    std::size_t begin = 0;
    while (begin < text.size() && isAsciiWhitespace(text[begin]))
    {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && isAsciiWhitespace(text[end - 1]))
    {
        --end;
    }

    return std::string(text.substr(begin, end - begin));
}

bool endsWithAsciiIgnoreCase(std::string_view text, std::string_view suffix)
{
    if (suffix.size() > text.size())
    {
        return false;
    }

    const std::size_t offset = text.size() - suffix.size();
    for (std::size_t i = 0; i < suffix.size(); ++i)
    {
        if (asciiLower(text[offset + i]) != asciiLower(suffix[i]))
        {
            return false;
        }
    }
    return true;
}

} // namespace xjw::common::string_utils

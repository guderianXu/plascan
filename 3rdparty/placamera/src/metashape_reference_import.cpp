#include "project_import_internal.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace placamera::internal
{
    namespace
    {

        std::string trim(std::string_view text)
        {
            const auto first = text.find_first_not_of(" \t\r\n");
            if (first == std::string_view::npos)
            {
                return {};
            }
            const auto last = text.find_last_not_of(" \t\r\n");
            return std::string(text.substr(first, last - first + 1));
        }

        std::string lower(std::string text)
        {
            std::transform(text.begin(),
                           text.end(),
                           text.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            return text;
        }

        std::string pathUtf8(const std::filesystem::path& path)
        {
            const auto bytes = path.u8string();
            return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
        }

        std::string normalizeHeader(std::string_view text)
        {
            auto source = trim(text);
            if (!source.empty() && source.front() == '#')
            {
                source = trim(std::string_view(source).substr(1));
            }
            std::string normalized;
            bool space = false;
            for (unsigned char ch : source)
            {
                if (std::isspace(ch) != 0)
                {
                    space = true;
                }
                else
                {
                    if (space && !normalized.empty())
                    {
                        normalized.push_back(' ');
                    }
                    normalized.push_back(static_cast<char>(std::tolower(ch)));
                    space = false;
                }
            }
            return normalized;
        }

        std::vector<std::string> fields(std::string_view line)
        {
            std::vector<std::string> result;
            std::size_t start = 0;
            while (start <= line.size())
            {
                const auto end = line.find('\t', start);
                result.push_back(trim(line.substr(start, end == std::string_view::npos ? end : end - start)));
                if (end == std::string_view::npos)
                {
                    break;
                }
                start = end + 1;
            }
            while (!result.empty() && result.back().empty())
            {
                result.pop_back();
            }
            return result;
        }

        bool validUtf8(std::string_view text)
        {
            for (std::size_t index = 0; index < text.size();)
            {
                const auto lead = static_cast<unsigned char>(text[index]);
                if (lead < 0x80)
                {
                    ++index;
                    continue;
                }
                int length = 0;
                unsigned int codepoint = 0;
                if (lead >= 0xC2 && lead <= 0xDF)
                {
                    length = 2;
                    codepoint = lead & 0x1F;
                }
                else if (lead >= 0xE0 && lead <= 0xEF)
                {
                    length = 3;
                    codepoint = lead & 0x0F;
                }
                else if (lead >= 0xF0 && lead <= 0xF4)
                {
                    length = 4;
                    codepoint = lead & 0x07;
                }
                else
                {
                    return false;
                }
                if (index + static_cast<std::size_t>(length) > text.size())
                {
                    return false;
                }
                for (int offset = 1; offset < length; ++offset)
                {
                    const auto byte = static_cast<unsigned char>(text[index + static_cast<std::size_t>(offset)]);
                    if ((byte & 0xC0) != 0x80)
                    {
                        return false;
                    }
                    codepoint = (codepoint << 6) | (byte & 0x3F);
                }
                if ((length == 3 && (codepoint < 0x800 || (codepoint >= 0xD800 && codepoint <= 0xDFFF))) ||
                    (length == 4 && (codepoint < 0x10000 || codepoint > 0x10FFFF)))
                {
                    return false;
                }
                index += static_cast<std::size_t>(length);
            }
            return true;
        }

        std::vector<std::string> readLines(const std::filesystem::path& path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("无法读取相机参考文件: " + pathUtf8(path));
            }
            std::string content{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            if (!input.eof() && input.fail())
            {
                throw std::runtime_error("读取相机参考文件失败: " + pathUtf8(path));
            }
            if (!validUtf8(content))
            {
                throw std::runtime_error("相机参考文件不是有效的 UTF-8 文本: " + pathUtf8(path));
            }
            if (content.starts_with("\xEF\xBB\xBF"))
            {
                content.erase(0, 3);
            }
            std::vector<std::string> lines;
            std::istringstream stream(content);
            std::string line;
            while (std::getline(stream, line))
            {
                lines.push_back(std::move(line));
            }
            return lines;
        }

        std::string lineError(const std::filesystem::path& path, std::size_t line, const std::string& error)
        {
            return pathUtf8(path) + "：第 " + std::to_string(line) + " 行：" + error;
        }

        int column(const std::map<std::string, int>& columns, std::initializer_list<const char*> aliases)
        {
            int result = -1;
            for (const char* alias : aliases)
            {
                const auto it = columns.find(alias);
                if (it == columns.end())
                {
                    continue;
                }
                if (result >= 0 && result != it->second)
                {
                    throw std::invalid_argument("表头中存在含义重复的列");
                }
                result = it->second;
            }
            return result;
        }

        std::string field(const std::vector<std::string>& row, int index)
        {
            return index >= 0 && static_cast<std::size_t>(index) < row.size() ? row[static_cast<std::size_t>(index)]
                                                                              : "";
        }

        bool finiteNumber(const std::string& text, double* output)
        {
            try
            {
                std::size_t consumed = 0;
                const double value = std::stod(text, &consumed);
                if (consumed != text.size() || !std::isfinite(value))
                {
                    return false;
                }
                *output = value;
                return true;
            }
            catch (const std::exception&)
            {
                return false;
            }
        }

        std::string basenameKey(const std::string& name)
        {
            const auto slash = name.find_last_of("/\\");
            return lower(trim(slash == std::string::npos ? name : name.substr(slash + 1)));
        }

        std::string join(const std::vector<std::string>& values, std::string_view delimiter)
        {
            std::string result;
            for (const auto& value : values)
            {
                if (!result.empty())
                {
                    result += delimiter;
                }
                result += value;
            }
            return result;
        }

        void importReferences(const std::filesystem::path& path, CameraProjectImportResult* result)
        {
            const auto lines = readLines(path);
            const auto header =
                std::find_if(lines.begin(), lines.end(), [](const std::string& line) { return !trim(line).empty(); });
            if (header == lines.end())
            {
                throw std::runtime_error("相机参考文件为空: " + pathUtf8(path));
            }
            const auto headerLine = static_cast<std::size_t>(std::distance(lines.begin(), header)) + 1;
            const auto names = fields(*header);
            if (names.size() < 2)
            {
                throw std::runtime_error(lineError(path, headerLine, "表头必须使用制表符分隔"));
            }
            std::map<std::string, int> columns;
            for (std::size_t index = 0; index < names.size(); ++index)
            {
                const auto normalized = normalizeHeader(names[index]);
                if (!normalized.empty() && !columns.emplace(normalized, static_cast<int>(index)).second)
                {
                    throw std::runtime_error(lineError(path, headerLine, "表头列重复：" + names[index]));
                }
            }
            struct NamedColumn
            {
                int index;
                const char* name;
            };
            std::array<NamedColumn, 8> required{};
            std::array<NamedColumn, 4> optional{};
            try
            {
                required = {{{column(columns, {"file"}), "file"},
                             {column(columns, {"wgs84_lat", "wgs84 lat"}), "WGS84_lat"},
                             {column(columns, {"wgs84_lon", "wgs84 lon"}), "WGS84_lon"},
                             {column(columns, {"wgs84_h", "wgs84 h", "wgs84_height", "wgs84 height"}), "WGS84_H"},
                             {column(columns, {"roll"}), "roll"},
                             {column(columns, {"pitch"}), "pitch"},
                             {column(columns, {"yaw"}), "yaw"},
                             {column(columns, {"time"}), "time"}}};
                optional = {{{column(columns, {"std dev n (m)", "std dev north (m)"}), "Std Dev n (m)"},
                             {column(columns, {"std dev e (m)", "std dev east (m)"}), "Std Dev e (m)"},
                             {column(columns, {"std dev u (m)", "std dev up (m)"}), "Std Dev u (m)"},
                             {column(columns, {"std dev hz (m)", "std dev h (m)", "std dev horizontal (m)"}),
                              "Std Dev Hz (m)"}}};
            }
            catch (const std::invalid_argument& error)
            {
                throw std::runtime_error(lineError(path, headerLine, error.what()));
            }
            std::vector<std::string> missing;
            for (const auto& item : required)
            {
                if (item.index < 0)
                {
                    missing.emplace_back(item.name);
                }
            }
            if (!missing.empty())
            {
                throw std::runtime_error(lineError(path, headerLine, "缺少必需列：" + join(missing, ", ")));
            }
            for (const auto& item : optional)
            {
                if (item.index < 0)
                {
                    result->warnings.push_back("未找到可选列 " + std::string(item.name) + "；对应精度将留空");
                }
            }

            std::set<std::string> seen;
            std::vector<std::string> errors;
            for (std::size_t lineIndex = headerLine; lineIndex < lines.size(); ++lineIndex)
            {
                const auto text = trim(lines[lineIndex]);
                if (text.empty() || text.front() == '#')
                {
                    continue;
                }
                const auto row = fields(lines[lineIndex]);
                const auto lineNumber = lineIndex + 1;
                if (row.size() > names.size())
                {
                    errors.push_back(lineError(path, lineNumber, "数据列多于表头，且尾部包含非空值"));
                    continue;
                }
                UnresolvedCameraReference record;
                record.imageName = field(row, required[0].index);
                const auto key = basenameKey(record.imageName);
                if (key.empty())
                {
                    errors.push_back(lineError(path, lineNumber, "字段 file 不能为空且必须是有效文件名"));
                    continue;
                }
                if (!seen.insert(key).second)
                {
                    errors.push_back(lineError(path, lineNumber, "文件名重复：" + record.imageName));
                    continue;
                }
                record.timeText = field(row, required[7].index);
                const std::array<double UnresolvedCameraReference::*, 6> numbers{
                    &UnresolvedCameraReference::latitudeDegrees,
                    &UnresolvedCameraReference::longitudeDegrees,
                    &UnresolvedCameraReference::ellipsoidalHeightMeters,
                    &UnresolvedCameraReference::rollDegrees,
                    &UnresolvedCameraReference::pitchDegrees,
                    &UnresolvedCameraReference::yawDegrees};
                bool valid = true;
                for (std::size_t index = 0; index < numbers.size(); ++index)
                {
                    const auto value = field(row, required[index + 1].index);
                    if (!finiteNumber(value, &(record.*numbers[index])))
                    {
                        errors.push_back(lineError(path,
                                                   lineNumber,
                                                   "字段 " + std::string(required[index + 1].name) +
                                                       " 必须是有限数值，当前值为“" + value + "”"));
                        valid = false;
                    }
                }
                const std::array<std::optional<double> UnresolvedCameraReference::*, 4> sigmas{
                    &UnresolvedCameraReference::stdDevNorthMeters,
                    &UnresolvedCameraReference::stdDevEastMeters,
                    &UnresolvedCameraReference::stdDevUpMeters,
                    &UnresolvedCameraReference::stdDevHorizontalMeters};
                for (std::size_t index = 0; index < sigmas.size(); ++index)
                {
                    const auto value = field(row, optional[index].index);
                    if (value.empty())
                    {
                        continue;
                    }
                    double number = 0.0;
                    if (!finiteNumber(value, &number))
                    {
                        errors.push_back(lineError(path,
                                                   lineNumber,
                                                   "字段 " + std::string(optional[index].name) +
                                                       " 必须是有限数值，当前值为“" + value + "”"));
                        valid = false;
                    }
                    else
                    {
                        record.*sigmas[index] = number;
                    }
                }
                if (record.latitudeDegrees < -90.0 || record.latitudeDegrees > 90.0)
                {
                    errors.push_back(lineError(path, lineNumber, "WGS84_lat 必须位于 [-90, 90]"));
                    valid = false;
                }
                if (record.longitudeDegrees < -180.0 || record.longitudeDegrees > 180.0)
                {
                    errors.push_back(lineError(path, lineNumber, "WGS84_lon 必须位于 [-180, 180]"));
                    valid = false;
                }
                for (std::size_t index = 0; index < sigmas.size(); ++index)
                {
                    const auto& sigma = record.*sigmas[index];
                    if (sigma && *sigma <= 0.0)
                    {
                        errors.push_back(
                            lineError(path, lineNumber, "字段 " + std::string(optional[index].name) + " 必须大于 0"));
                        valid = false;
                    }
                }
                const int sigmaCount = static_cast<int>(record.stdDevNorthMeters.has_value()) +
                                       static_cast<int>(record.stdDevEastMeters.has_value()) +
                                       static_cast<int>(record.stdDevUpMeters.has_value());
                if (sigmaCount != 0 && sigmaCount != 3)
                {
                    errors.push_back(lineError(path, lineNumber, "Std Dev n/e/u 必须同时提供或同时留空"));
                    valid = false;
                }
                if (valid)
                {
                    result->references.push_back(std::move(record));
                }
            }
            if (!errors.empty())
            {
                throw std::runtime_error(join(errors, "\n"));
            }
            if (result->references.empty())
            {
                throw std::runtime_error("相机参考文件不包含数据记录：" + pathUtf8(path));
            }
        }

        UnresolvedLeverArm importLeverArm(const std::filesystem::path& path)
        {
            const auto lines = readLines(path);
            std::map<char, double> values;
            std::vector<std::string> errors;
            for (std::size_t index = 0; index < lines.size(); ++index)
            {
                const auto line = trim(lines[index]);
                if (line.empty() || line.front() == '#')
                {
                    continue;
                }
                const auto separator = line.find('=');
                const auto axisText = trim(std::string_view(line).substr(0, separator));
                const char axis = axisText.size() == 1
                                      ? static_cast<char>(std::toupper(static_cast<unsigned char>(axisText[0])))
                                      : '\0';
                if (separator == std::string::npos || (axis != 'X' && axis != 'Y' && axis != 'Z'))
                {
                    errors.push_back(lineError(path, index + 1, "GNSS 偏移必须使用 X=、Y= 或 Z= 格式"));
                    continue;
                }
                if (values.contains(axis))
                {
                    errors.push_back(lineError(path, index + 1, "GNSS 偏移分量 " + std::string(1, axis) + " 重复"));
                    continue;
                }
                double value = 0.0;
                if (!finiteNumber(trim(std::string_view(line).substr(separator + 1)), &value))
                {
                    errors.push_back(
                        lineError(path, index + 1, "字段 GNSS " + std::string(1, axis) + " 必须是有限数值"));
                    continue;
                }
                values.emplace(axis, value);
            }
            if (!errors.empty())
            {
                throw std::runtime_error(join(errors, "\n"));
            }
            std::vector<std::string> missing;
            for (const char axis : {'X', 'Y', 'Z'})
            {
                if (!values.contains(axis))
                {
                    missing.emplace_back(1, axis);
                }
            }
            if (!missing.empty())
            {
                throw std::runtime_error("GNSS 偏移文件 " + pathUtf8(path) + " 缺少分量：" + join(missing, ", "));
            }
            return {values.at('X'), values.at('Y'), values.at('Z')};
        }

    } // namespace

    CameraProjectImportResult importMetashapeReferenceFiles(const std::filesystem::path& cameraPath,
                                                            const std::filesystem::path& gnssOffsetPath)
    {
        CameraProjectImportResult result;
        result.format = CameraProjectFormat::MetashapeReferenceText;
        importReferences(cameraPath, &result);
        if (!gnssOffsetPath.empty())
        {
            result.leverArm = importLeverArm(gnssOffsetPath);
        }
        return result;
    }

} // namespace placamera::internal

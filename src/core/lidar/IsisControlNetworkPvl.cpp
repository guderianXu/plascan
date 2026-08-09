#include "IsisControlNetworkPvl.h"

#include "io/PathIO.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <utility>

namespace xjw
{
namespace lidar
{
namespace
{

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

std::string trim(const std::string &value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char character)
    {
        return std::isspace(character) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char character)
    {
        return std::isspace(character) != 0;
    }).base();
    return first < last ? std::string(first, last) : std::string{};
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character)
    {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string unquote(const std::string &value)
{
    const std::string cleaned = trim(value);
    if (cleaned.size() >= 2 &&
        ((cleaned.front() == '"' && cleaned.back() == '"') ||
         (cleaned.front() == '\'' && cleaned.back() == '\'')))
    {
        return cleaned.substr(1, cleaned.size() - 2);
    }
    return cleaned;
}

bool parseFiniteDouble(const std::string &value, double *result)
{
    if (!result)
    {
        return false;
    }
    try
    {
        std::size_t parsed = 0;
        const std::string cleaned = trim(value);
        const double number = std::stod(cleaned, &parsed);
        if (parsed != cleaned.size() || !std::isfinite(number))
        {
            return false;
        }
        *result = number;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parseBool(const std::string &value, bool *result)
{
    if (!result)
    {
        return false;
    }
    const std::string normalized = lower(unquote(value));
    if (normalized == "true" || normalized == "yes" || normalized == "1")
    {
        *result = true;
        return true;
    }
    if (normalized == "false" || normalized == "no" || normalized == "0")
    {
        *result = false;
        return true;
    }
    return false;
}

IsisControlPointType parsePointType(const std::string &value)
{
    const std::string normalized = lower(unquote(value));
    if (normalized == "free")
    {
        return IsisControlPointType::Free;
    }
    if (normalized == "constrained")
    {
        return IsisControlPointType::Constrained;
    }
    if (normalized == "fixed")
    {
        return IsisControlPointType::Fixed;
    }
    return IsisControlPointType::Unknown;
}

struct MeasureBuilder
{
    IsisControlMeasure measure;
    bool hasSerial = false;
    bool hasSample = false;
    bool hasLine = false;
};

} // namespace

bool IsisControlNetwork::validate(std::string *errorMessage) const
{
    if (points.empty())
    {
        setError(errorMessage, "ISIS control network contains no ControlPoint objects");
        return false;
    }

    std::set<std::string> pointIds;
    for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
    {
        const IsisControlPoint &point = points[pointIndex];
        if (point.id.empty())
        {
            setError(errorMessage, "ControlPoint at index " + std::to_string(pointIndex) +
                                       " has no PointId");
            return false;
        }
        if (!pointIds.insert(point.id).second)
        {
            setError(errorMessage, "duplicate ISIS ControlPoint PointId: " + point.id);
            return false;
        }
        if (point.type == IsisControlPointType::Unknown)
        {
            setError(errorMessage, "ControlPoint " + point.id +
                                       " has a missing or unsupported PointType");
            return false;
        }
        if (point.measures.empty())
        {
            setError(errorMessage, "ControlPoint " + point.id + " has no ControlMeasure");
            return false;
        }

        std::set<std::string> serialNumbers;
        for (const IsisControlMeasure &measure : point.measures)
        {
            if (measure.serialNumber.empty())
            {
                setError(errorMessage, "ControlPoint " + point.id +
                                           " contains a measure without SerialNumber");
                return false;
            }
            if (!std::isfinite(measure.samplePixels) || !std::isfinite(measure.linePixels))
            {
                setError(errorMessage, "ControlPoint " + point.id +
                                           " contains a non-finite image coordinate");
                return false;
            }
            if (!serialNumbers.insert(measure.serialNumber).second)
            {
                setError(errorMessage, "ControlPoint " + point.id +
                                           " contains duplicate camera serial " +
                                           measure.serialNumber);
                return false;
            }
        }
    }
    return true;
}

int IsisControlNetwork::usableMeasureCount() const
{
    int count = 0;
    for (const IsisControlPoint &point : points)
    {
        if (point.ignored)
        {
            continue;
        }
        count += static_cast<int>(std::count_if(
            point.measures.begin(), point.measures.end(), [](const IsisControlMeasure &measure)
            {
                return !measure.ignored;
            }));
    }
    return count;
}

bool parseIsisControlNetworkPvl(const std::string &pvl,
                                IsisControlNetwork *network,
                                std::string *errorMessage)
{
    if (!network)
    {
        setError(errorMessage, "output ISIS control network pointer is null");
        return false;
    }

    IsisControlNetwork parsed;
    IsisControlPoint point;
    MeasureBuilder measure;
    bool inPoint = false;
    bool inMeasure = false;
    int lineNumber = 0;

    std::istringstream stream(pvl);
    std::string rawLine;
    while (std::getline(stream, rawLine))
    {
        ++lineNumber;
        const std::string line = trim(rawLine);
        if (line.empty() || line.starts_with('#'))
        {
            continue;
        }

        const std::size_t equals = line.find('=');
        const std::string key = lower(trim(line.substr(0, equals)));
        const std::string value = equals == std::string::npos
            ? std::string{}
            : unquote(line.substr(equals + 1));
        const std::string normalizedValue = lower(value);

        if (key == "object" && normalizedValue == "controlpoint")
        {
            if (inPoint || inMeasure)
            {
                setError(errorMessage, "nested ControlPoint at PVL line " +
                                           std::to_string(lineNumber));
                return false;
            }
            point = IsisControlPoint{};
            inPoint = true;
            continue;
        }
        if (key == "group" && normalizedValue == "controlmeasure")
        {
            if (!inPoint || inMeasure)
            {
                setError(errorMessage, "ControlMeasure outside ControlPoint at PVL line " +
                                           std::to_string(lineNumber));
                return false;
            }
            measure = MeasureBuilder{};
            inMeasure = true;
            continue;
        }
        if (key == "end_group" || key == "endgroup")
        {
            if (!inMeasure)
            {
                continue;
            }
            if (!measure.hasSerial || !measure.hasSample || !measure.hasLine)
            {
                setError(errorMessage, "incomplete ControlMeasure ending at PVL line " +
                                           std::to_string(lineNumber));
                return false;
            }
            point.measures.push_back(std::move(measure.measure));
            measure = MeasureBuilder{};
            inMeasure = false;
            continue;
        }
        if ((key == "end_object" || key == "endobject") && inPoint)
        {
            if (inMeasure)
            {
                setError(errorMessage, "ControlPoint ended before ControlMeasure at PVL line " +
                                           std::to_string(lineNumber));
                return false;
            }
            parsed.points.push_back(std::move(point));
            point = IsisControlPoint{};
            inPoint = false;
            continue;
        }

        if (inMeasure)
        {
            if (key == "serialnumber")
            {
                measure.measure.serialNumber = value;
                measure.hasSerial = !value.empty();
            }
            else if (key == "sample")
            {
                measure.hasSample = parseFiniteDouble(value, &measure.measure.samplePixels);
                if (!measure.hasSample)
                {
                    setError(errorMessage, "invalid ControlMeasure Sample at PVL line " +
                                               std::to_string(lineNumber));
                    return false;
                }
            }
            else if (key == "line")
            {
                measure.hasLine = parseFiniteDouble(value, &measure.measure.linePixels);
                if (!measure.hasLine)
                {
                    setError(errorMessage, "invalid ControlMeasure Line at PVL line " +
                                               std::to_string(lineNumber));
                    return false;
                }
            }
            else if (key == "ignore")
            {
                if (!parseBool(value, &measure.measure.ignored))
                {
                    setError(errorMessage, "invalid ControlMeasure Ignore flag at PVL line " +
                                               std::to_string(lineNumber));
                    return false;
                }
            }
            continue;
        }

        if (inPoint)
        {
            if (key == "pointid")
            {
                point.id = value;
            }
            else if (key == "pointtype")
            {
                point.type = parsePointType(value);
                if (point.type == IsisControlPointType::Unknown)
                {
                    setError(errorMessage, "unsupported ControlPoint PointType at PVL line " +
                                               std::to_string(lineNumber));
                    return false;
                }
            }
            else if (key == "ignore")
            {
                if (!parseBool(value, &point.ignored))
                {
                    setError(errorMessage, "invalid ControlPoint Ignore flag at PVL line " +
                                               std::to_string(lineNumber));
                    return false;
                }
            }
            continue;
        }

        if (key == "networkid")
        {
            parsed.networkId = value;
        }
        else if (key == "targetname")
        {
            parsed.targetName = value;
        }
    }

    if (inMeasure || inPoint)
    {
        setError(errorMessage, "unterminated ISIS ControlPoint or ControlMeasure in PVL input");
        return false;
    }
    if (!parsed.validate(errorMessage))
    {
        return false;
    }
    *network = std::move(parsed);
    return true;
}

bool loadIsisControlNetworkPvlFile(const std::string &path,
                                   IsisControlNetwork *network,
                                   std::string *errorMessage)
{
    std::ifstream stream = common::io::openInputFile(path);
    if (!stream)
    {
        setError(errorMessage, "failed to open ISIS control network PVL: " + path);
        return false;
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof())
    {
        setError(errorMessage, "failed while reading ISIS control network PVL: " + path);
        return false;
    }
    return parseIsisControlNetworkPvl(contents.str(), network, errorMessage);
}

} // namespace lidar
} // namespace xjw

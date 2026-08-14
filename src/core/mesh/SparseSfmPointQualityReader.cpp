#include "SparseSfmPointQualityReader.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string_view>
#include <vector>

namespace xjw::mesh
{
namespace
{

class JsonStream
{
public:
    explicit JsonStream(std::ifstream stream) : _stream(std::move(stream)) {}

    bool good() const
    {
        return _stream.good() || _stream.eof();
    }

    std::size_t bytesRead() const
    {
        return _bytesRead;
    }

    void skipWhitespace()
    {
        while (true)
        {
            const int value = _stream.peek();
            if (value != ' ' && value != '\t' && value != '\r' && value != '\n')
            {
                return;
            }
            take();
        }
    }

    int peek()
    {
        skipWhitespace();
        return _stream.peek();
    }

    bool consume(char expected)
    {
        skipWhitespace();
        if (_stream.peek() != expected)
        {
            return false;
        }
        take();
        return true;
    }

    bool readString(std::string *text)
    {
        text->clear();
        if (!consume('"'))
        {
            return false;
        }
        bool escaped = false;
        while (true)
        {
            const int raw = take();
            if (raw == EOF)
            {
                return false;
            }
            const char value = static_cast<char>(raw);
            if (escaped)
            {
                escaped = false;
                text->push_back(value);
            }
            else if (value == '\\')
            {
                escaped = true;
            }
            else if (value == '"')
            {
                return true;
            }
            else
            {
                text->push_back(value);
            }
        }
    }

    bool readNumber(double *number)
    {
        skipWhitespace();
        std::string token;
        token.reserve(32);
        while (true)
        {
            const int raw = _stream.peek();
            if ((raw >= '0' && raw <= '9') || raw == '-' || raw == '+'
                || raw == '.' || raw == 'e' || raw == 'E')
            {
                if (token.size() >= 128)
                {
                    return false;
                }
                token.push_back(static_cast<char>(take()));
                continue;
            }
            break;
        }
        if (token.empty())
        {
            return false;
        }
        char *end = nullptr;
        *number = std::strtod(token.c_str(), &end);
        return end == token.c_str() + token.size() && std::isfinite(*number);
    }

    bool skipValue()
    {
        skipWhitespace();
        const int first = _stream.peek();
        if (first == '"')
        {
            std::string ignored;
            return readString(&ignored);
        }
        if (first != '{' && first != '[')
        {
            bool consumed = false;
            while (true)
            {
                const int raw = _stream.peek();
                if (raw == EOF || raw == ',' || raw == '}' || raw == ']'
                    || raw == ' ' || raw == '\t' || raw == '\r' || raw == '\n')
                {
                    return consumed;
                }
                take();
                consumed = true;
            }
        }

        std::vector<char> closing;
        closing.reserve(16);
        closing.push_back(first == '{' ? '}' : ']');
        take();
        bool in_string = false;
        bool escaped = false;
        while (!closing.empty())
        {
            const int raw = take();
            if (raw == EOF)
            {
                return false;
            }
            const char value = static_cast<char>(raw);
            if (in_string)
            {
                if (escaped)
                {
                    escaped = false;
                }
                else if (value == '\\')
                {
                    escaped = true;
                }
                else if (value == '"')
                {
                    in_string = false;
                }
                continue;
            }
            if (value == '"')
            {
                in_string = true;
            }
            else if (value == '{' || value == '[')
            {
                if (closing.size() >= 128)
                {
                    return false;
                }
                closing.push_back(value == '{' ? '}' : ']');
            }
            else if (value == '}' || value == ']')
            {
                if (closing.back() != value)
                {
                    return false;
                }
                closing.pop_back();
            }
        }
        return true;
    }

private:
    int take()
    {
        const int value = _stream.get();
        if (value != EOF)
        {
            ++_bytesRead;
        }
        return value;
    }

    std::ifstream _stream;
    std::size_t _bytesRead = 0;
};

bool readPointArray(JsonStream *json, std::array<float, 3> *point)
{
    if (!json->consume('['))
    {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        double coordinate = 0.0;
        if (!json->readNumber(&coordinate)
            || coordinate < -std::numeric_limits<float>::max()
            || coordinate > std::numeric_limits<float>::max())
        {
            return false;
        }
        (*point)[static_cast<std::size_t>(axis)] = static_cast<float>(coordinate);
        if (axis < 2 && !json->consume(','))
        {
            return false;
        }
    }
    return json->consume(']');
}

bool readPointObject(JsonStream *json, SparseSfmPointQuality *point)
{
    if (!json->consume('{'))
    {
        return false;
    }
    bool has_track = false;
    bool has_rms = false;
    bool has_angle = false;
    while (json->peek() != '}')
    {
        std::string key;
        if (!json->readString(&key) || !json->consume(':'))
        {
            return false;
        }
        if (key == "point_xyz")
        {
            point->hasPoint = readPointArray(json, &point->point);
            if (!point->hasPoint)
            {
                return false;
            }
        }
        else if (key == "track_len")
        {
            double value = 0.0;
            if (!json->readNumber(&value) || value < 0.0
                || value > static_cast<double>(std::numeric_limits<int>::max()))
            {
                return false;
            }
            point->trackLength = static_cast<int>(value);
            has_track = true;
        }
        else if (key == "rms_reproj_px")
        {
            double value = 0.0;
            if (!json->readNumber(&value)
                || value > std::numeric_limits<float>::max())
            {
                return false;
            }
            point->rmsReprojectionPixels = static_cast<float>(value);
            has_rms = true;
        }
        else if (key == "triangulation_angle_deg" || key == "min_tri_angle_deg")
        {
            double value = 0.0;
            if (!json->readNumber(&value)
                || value > std::numeric_limits<float>::max())
            {
                return false;
            }
            point->triangulationAngleDegrees = static_cast<float>(value);
            has_angle = true;
        }
        else if (!json->skipValue())
        {
            return false;
        }
        if (json->peek() == ',')
        {
            json->consume(',');
        }
        else if (json->peek() != '}')
        {
            return false;
        }
    }
    if (!json->consume('}'))
    {
        return false;
    }
    point->hasRequiredQuality = has_track && has_rms && has_angle;
    return true;
}

bool readPointsArray(JsonStream *json, std::vector<SparseSfmPointQuality> *points)
{
    if (!json->consume('['))
    {
        return false;
    }
    while (json->peek() != ']')
    {
        SparseSfmPointQuality point;
        if (!readPointObject(json, &point))
        {
            return false;
        }
        points->push_back(point);
        if (json->peek() == ',')
        {
            json->consume(',');
        }
        else if (json->peek() != ']')
        {
            return false;
        }
    }
    return json->consume(']');
}

bool readRoot(JsonStream *json, std::vector<SparseSfmPointQuality> *points)
{
    if (!json->consume('{'))
    {
        return false;
    }
    bool found_points = false;
    while (json->peek() != '}')
    {
        std::string key;
        if (!json->readString(&key) || !json->consume(':'))
        {
            return false;
        }
        if (key == "points")
        {
            if (found_points || !readPointsArray(json, points))
            {
                return false;
            }
            found_points = true;
        }
        else if (!json->skipValue())
        {
            return false;
        }
        if (json->peek() == ',')
        {
            json->consume(',');
        }
        else if (json->peek() != '}')
        {
            return false;
        }
    }
    return json->consume('}') && found_points;
}

} // namespace

SparseSfmPointQualityReadResult SparseSfmPointQualityReader::read(
    const std::filesystem::path &path)
{
    SparseSfmPointQualityReadResult result;
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if (!stream)
    {
        result.error = "无法打开 SfM 点质量 sidecar";
        return result;
    }

    JsonStream json(std::move(stream));
    if (!readRoot(&json, &result.points))
    {
        result.error = "SfM 点质量 sidecar JSON 结构无效或已截断";
        result.points.clear();
    }
    result.bytesRead = json.bytesRead();
    return result;
}

} // namespace xjw::mesh

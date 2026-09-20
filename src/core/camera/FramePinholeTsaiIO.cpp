#include "FramePinholeTsaiIO.h"

#include "io/PathIO.h"
#include "string_utils/StringParsing.h"
#include "string_utils/StringTransform.h"

#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace xjw::camera_io
{
    namespace
    {

        using xjw::common::string_utils::asciiLowerCopy;
        using xjw::common::string_utils::extractDoublesFromText;
        using xjw::common::string_utils::trimAsciiWhitespace;
        using NumericState = xjw::camera_models::frame_pinhole::FramePinholeNumericState;

        bool startsWithKey(const std::string& textLower, const std::string& keyLower)
        {
            if (textLower.rfind(keyLower, 0) != 0)
            {
                return false;
            }
            if (textLower.size() == keyLower.size())
            {
                return true;
            }
            const char next = textLower[keyLower.size()];
            return std::isspace(static_cast<unsigned char>(next)) || next == '=' || next == ':';
        }

        bool firstNumberAfterSeparator(const std::string& line, double* value)
        {
            if (!value)
            {
                return false;
            }
            const std::size_t separator = line.find_first_of("=:");
            if (separator == std::string::npos)
            {
                return false;
            }
            std::vector<double> values;
            if (!extractDoublesFromText(line.substr(separator + 1), values) || values.empty())
            {
                return false;
            }
            *value = values.front();
            return true;
        }

        void setError(std::string* error, const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
        }

    } // namespace

    bool loadFramePinholeNumericStateFromTsaiFile(const std::string& path, NumericState* state, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!state)
        {
            setError(error, "pinhole numeric state output is null");
            return false;
        }

        std::ifstream input = xjw::common::io::openInputFile(path);
        if (!input)
        {
            setError(error, "cannot open Tsai camera file: " + path);
            return false;
        }

        double focal_x_mm = 0.0;
        double focal_y_mm = 0.0;
        double principal_x_mm = 0.0;
        double principal_y_mm = 0.0;
        double pixel_pitch_mm = 1.0;
        std::array<double, 3> center{{0.0, 0.0, 0.0}};
        std::array<double, 9> rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        double radial_k1 = 0.0;
        double radial_k2 = 0.0;
        double radial_k3 = 0.0;
        double tangential_p1 = 0.0;
        double tangential_p2 = 0.0;
        int u_axis_sign = 1;
        int v_axis_sign = 1;
        bool depth_axis_flipped = false;
        bool has_focal_x = false;
        bool has_focal_y = false;
        bool has_principal_x = false;
        bool has_principal_y = false;
        bool has_center = false;
        bool has_rotation = false;

        std::string line;
        while (std::getline(input, line))
        {
            const std::string trimmed = trimAsciiWhitespace(line);
            if (trimmed.empty())
            {
                continue;
            }
            const std::string lower = asciiLowerCopy(trimmed);
            std::vector<double> values;
            if (startsWithKey(lower, "fu"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    focal_x_mm = values.front();
                    has_focal_x = true;
                }
            }
            else if (startsWithKey(lower, "fv"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    focal_y_mm = values.front();
                    has_focal_y = true;
                }
            }
            else if (startsWithKey(lower, "cu"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    principal_x_mm = values.front();
                    has_principal_x = true;
                }
            }
            else if (startsWithKey(lower, "cv"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    principal_y_mm = values.front();
                    has_principal_y = true;
                }
            }
            else if (startsWithKey(lower, "c"))
            {
                if (extractDoublesFromText(trimmed, values) && values.size() >= 3)
                {
                    center = {values[0], values[1], values[2]};
                    has_center = true;
                }
            }
            else if (startsWithKey(lower, "r"))
            {
                if (extractDoublesFromText(trimmed, values) && values.size() >= 9)
                {
                    for (std::size_t index = 0; index < rotation.size(); ++index)
                    {
                        rotation[index] = values[index];
                    }
                    has_rotation = true;
                }
            }
            else if (startsWithKey(lower, "k1"))
            {
                firstNumberAfterSeparator(trimmed, &radial_k1);
            }
            else if (startsWithKey(lower, "k2"))
            {
                firstNumberAfterSeparator(trimmed, &radial_k2);
            }
            else if (startsWithKey(lower, "k3"))
            {
                firstNumberAfterSeparator(trimmed, &radial_k3);
            }
            else if (startsWithKey(lower, "p1"))
            {
                firstNumberAfterSeparator(trimmed, &tangential_p1);
            }
            else if (startsWithKey(lower, "p2"))
            {
                firstNumberAfterSeparator(trimmed, &tangential_p2);
            }
            else if (startsWithKey(lower, "pitch"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    pixel_pitch_mm = values.front();
                }
            }
            else if (startsWithKey(lower, "u_direction"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    const double dominant =
                        values.size() >= 3 ? (values[0] == 0.0 ? values[1] : values[0]) : values.front();
                    u_axis_sign = dominant < 0.0 ? -1 : 1;
                }
            }
            else if (startsWithKey(lower, "v_direction"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    const double dominant =
                        values.size() >= 3 ? (values[1] == 0.0 ? values[0] : values[1]) : values.front();
                    v_axis_sign = dominant < 0.0 ? -1 : 1;
                }
            }
            else if (startsWithKey(lower, "w_direction"))
            {
                if (extractDoublesFromText(trimmed, values) && !values.empty())
                {
                    const double depth = values.size() >= 3 ? values[2] : values.front();
                    depth_axis_flipped = depth < 0.0;
                }
            }
        }

        if (!(has_focal_x && has_focal_y && has_principal_x && has_principal_y && has_center && has_rotation))
        {
            setError(error, "Tsai camera is missing fu/fv/cu/cv/c/r fields: " + path);
            return false;
        }
        if (!std::isfinite(pixel_pitch_mm) || pixel_pitch_mm <= 0.0)
        {
            setError(error, "Tsai camera pixel pitch must be positive: " + path);
            return false;
        }
        if (!std::isfinite(focal_x_mm) || !std::isfinite(focal_y_mm) || focal_x_mm <= 0.0 || focal_y_mm <= 0.0)
        {
            setError(error, "Tsai camera focal lengths must be positive: " + path);
            return false;
        }

        NumericState parsed;
        parsed.setPixelPitch(pixel_pitch_mm);
        parsed.setIntrinsicsMillimeters(focal_x_mm, focal_y_mm, principal_x_mm, principal_y_mm, pixel_pitch_mm);
        parsed.setAxisDirections(u_axis_sign, v_axis_sign);
        parsed.setDepthAxisFlipped(depth_axis_flipped);
        parsed.setDistortion(radial_k1, radial_k2, radial_k3, tangential_p1, tangential_p2);
        parsed.setPose(rotation, center);
        if (!parsed.isValid())
        {
            setError(error, "Tsai camera did not produce a valid numeric state: " + path);
            return false;
        }
        *state = std::move(parsed);
        return true;
    }

} // namespace xjw::camera_io

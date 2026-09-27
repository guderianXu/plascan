#include "state_json_common.h"

#include "placamera/state_codec.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <unordered_set>
#include <utility>

namespace placamera::state_detail
{

    namespace
    {

        const char* timeScaleName(TimeScale scale) noexcept
        {
            switch (scale)
            {
            case TimeScale::Utc:
                return "utc";
            case TimeScale::Tai:
                return "tai";
            case TimeScale::Tdb:
                return "tdb";
            case TimeScale::Relative:
                return "relative";
            }
            return "relative";
        }

        std::optional<TimeScale> parseTimeScale(std::string_view value) noexcept
        {
            if (value == "utc")
            {
                return TimeScale::Utc;
            }
            if (value == "tai")
            {
                return TimeScale::Tai;
            }
            if (value == "tdb")
            {
                return TimeScale::Tdb;
            }
            if (value == "relative")
            {
                return TimeScale::Relative;
            }
            return std::nullopt;
        }

    } // namespace

    bool hasObjectKeys(const Json& value,
                       std::initializer_list<std::string_view> required,
                       std::initializer_list<std::string_view> optional)
    {
        if (!value.is_object())
        {
            return false;
        }
        for (const std::string_view key : required)
        {
            if (!value.contains(std::string(key)))
            {
                return false;
            }
        }
        for (auto iterator = value.begin(); iterator != value.end(); ++iterator)
        {
            const std::string_view key = iterator.key();
            const bool known = std::find(required.begin(), required.end(), key) != required.end() ||
                               std::find(optional.begin(), optional.end(), key) != optional.end();
            if (!known)
            {
                return false;
            }
        }
        return true;
    }

    bool readString(const Json& object, std::string_view key, std::string* output)
    {
        if (!output || !object.contains(std::string(key)) || !object.at(std::string(key)).is_string())
        {
            return false;
        }
        *output = object.at(std::string(key)).get<std::string>();
        return !output->empty();
    }

    bool readInt(const Json& object, std::string_view key, int* output)
    {
        if (!output || !object.contains(std::string(key)) || !object.at(std::string(key)).is_number_integer())
        {
            return false;
        }
        *output = object.at(std::string(key)).get<int>();
        return true;
    }

    bool readFinite(const Json& object, std::string_view key, double* output)
    {
        if (!output || !object.contains(std::string(key)) || !object.at(std::string(key)).is_number())
        {
            return false;
        }
        *output = object.at(std::string(key)).get<double>();
        return std::isfinite(*output);
    }

    bool readBool(const Json& object, std::string_view key, bool* output)
    {
        if (!output || !object.contains(std::string(key)) || !object.at(std::string(key)).is_boolean())
        {
            return false;
        }
        *output = object.at(std::string(key)).get<bool>();
        return true;
    }

    Json timeReference(const TimeReference& time)
    {
        return Json{{"scale", timeScaleName(time.scale)}, {"seconds", time.seconds}};
    }

    bool readTimeReference(const Json& value, TimeReference* output)
    {
        if (!output || !hasObjectKeys(value, {"scale", "seconds"}))
        {
            return false;
        }
        std::string scale_name;
        double seconds = 0.0;
        if (!readString(value, "scale", &scale_name) || !readFinite(value, "seconds", &seconds))
        {
            return false;
        }
        const auto scale = parseTimeScale(scale_name);
        if (!scale)
        {
            return false;
        }
        *output = TimeReference::create(*scale, seconds);
        return true;
    }

    Json optionalTimeReference(const std::optional<TimeReference>& time)
    {
        return time ? timeReference(*time) : Json(nullptr);
    }

    bool readOptionalTimeReference(const Json& object, std::string_view key, std::optional<TimeReference>* output)
    {
        if (!output || !object.contains(std::string(key)))
        {
            return false;
        }
        const Json& value = object.at(std::string(key));
        if (value.is_null())
        {
            *output = std::nullopt;
            return true;
        }
        TimeReference time = TimeReference::create(TimeScale::Relative, 0.0);
        if (!readTimeReference(value, &time))
        {
            return false;
        }
        *output = time;
        return true;
    }

    Json principalPointDecomposition(const std::optional<PrincipalPointDecomposition>& principal)
    {
        if (!principal)
        {
            return nullptr;
        }
        return Json{
            {"image_center", finiteArray(std::array<double, 2>{{principal->imageCenterX, principal->imageCenterY}})},
            {"cx_offset", principal->cxOffset},
            {"cy_offset", principal->cyOffset}};
    }

    bool readPrincipalPointDecomposition(const Json& value, std::optional<PrincipalPointDecomposition>* output)
    {
        if (!output)
        {
            return false;
        }
        if (value.is_null())
        {
            *output = std::nullopt;
            return true;
        }
        if (!hasObjectKeys(value, {"image_center", "cx_offset", "cy_offset"}))
        {
            return false;
        }
        std::array<double, 2> center{};
        PrincipalPointDecomposition principal;
        if (!readFiniteArray(value.at("image_center"), &center) ||
            !readFinite(value, "cx_offset", &principal.cxOffset) ||
            !readFinite(value, "cy_offset", &principal.cyOffset))
        {
            return false;
        }
        principal.imageCenterX = center[0];
        principal.imageCenterY = center[1];
        *output = principal;
        return true;
    }

    Json metashapeCalibration(const std::optional<MetashapeCalibration>& calibration)
    {
        if (!calibration)
        {
            return nullptr;
        }
        return Json{
            {"f", calibration->f},
            {"cx", calibration->cx},
            {"cy", calibration->cy},
            {"b1", calibration->b1},
            {"b2", calibration->b2},
            {"k1", calibration->k1},
            {"k2", calibration->k2},
            {"k3", calibration->k3},
            {"k4", calibration->k4},
            {"p1", calibration->p1},
            {"p2", calibration->p2},
            {"p3", calibration->p3},
            {"p4", calibration->p4},
            {"principal_point_decomposition", principalPointDecomposition(calibration->principalPointDecomposition)}};
    }

    bool readMetashapeCalibration(const Json& value, std::optional<MetashapeCalibration>* output)
    {
        if (!output)
        {
            return false;
        }
        if (value.is_null())
        {
            *output = std::nullopt;
            return true;
        }
        if (!hasObjectKeys(value,
                           {"f",
                            "cx",
                            "cy",
                            "b1",
                            "b2",
                            "k1",
                            "k2",
                            "k3",
                            "k4",
                            "p1",
                            "p2",
                            "p3",
                            "p4",
                            "principal_point_decomposition"}))
        {
            return false;
        }
        MetashapeCalibration calibration;
        if (!readFinite(value, "f", &calibration.f) || !readFinite(value, "cx", &calibration.cx) ||
            !readFinite(value, "cy", &calibration.cy) || !readFinite(value, "b1", &calibration.b1) ||
            !readFinite(value, "b2", &calibration.b2) || !readFinite(value, "k1", &calibration.k1) ||
            !readFinite(value, "k2", &calibration.k2) || !readFinite(value, "k3", &calibration.k3) ||
            !readFinite(value, "k4", &calibration.k4) || !readFinite(value, "p1", &calibration.p1) ||
            !readFinite(value, "p2", &calibration.p2) || !readFinite(value, "p3", &calibration.p3) ||
            !readFinite(value, "p4", &calibration.p4) ||
            !readPrincipalPointDecomposition(value.at("principal_point_decomposition"),
                                             &calibration.principalPointDecomposition))
        {
            return false;
        }
        *output = calibration;
        return true;
    }

    Result<Json> parseObject(std::string_view payload)
    {
        try
        {
            Json parsed = Json::parse(payload.begin(), payload.end(), nullptr, false);
            if (parsed.is_discarded() || !parsed.is_object())
            {
                return Result<Json>::failure(CameraErrorCode::InvalidModelState, "camera state must be a JSON object");
            }
            return Result<Json>::success(std::move(parsed));
        }
        catch (const std::exception& error)
        {
            return Result<Json>::failure(CameraErrorCode::InvalidModelState,
                                         std::string("camera JSON parsing failed: ") + error.what());
        }
    }

    Result<Json> parseStateObject(const ModelState& state, std::string_view expectedMediaType)
    {
        if (state.mediaType != expectedMediaType)
        {
            return Result<Json>::failure(CameraErrorCode::InvalidModelState,
                                         "camera model state uses an unsupported media type");
        }
        return parseObject(state.payloadAsText());
    }

    ModelState jsonState(std::string modelType, int schemaVersion, std::string_view mediaType, const Json& payload)
    {
        return ModelState::fromText(std::move(modelType), schemaVersion, std::string(mediaType), payload.dump());
    }

} // namespace placamera::state_detail

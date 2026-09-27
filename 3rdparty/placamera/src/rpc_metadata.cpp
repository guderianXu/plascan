#include "placamera/rpc_metadata.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <sstream>
#include <utility>

namespace placamera
{
    namespace
    {
        std::string uppercase(std::string value)
        {
            std::transform(value.begin(),
                           value.end(),
                           value.begin(),
                           [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
            return value;
        }

        RpcMetadata normalizedMetadata(const RpcMetadata& metadata)
        {
            RpcMetadata normalized;
            for (const auto& [key, value] : metadata)
            {
                normalized[uppercase(key)] = value;
            }
            return normalized;
        }

        bool parseDouble(const RpcMetadata& metadata,
                         const std::string& key,
                         double* value,
                         std::string* error,
                         bool required = true)
        {
            const auto found = metadata.find(key);
            if (found == metadata.end())
            {
                if (!required)
                {
                    return false;
                }
                if (error)
                {
                    *error = "RPC metadata is missing required key " + key;
                }
                return false;
            }
            try
            {
                std::size_t consumed = 0;
                const double parsed = std::stod(found->second, &consumed);
                while (consumed < found->second.size() &&
                       std::isspace(static_cast<unsigned char>(found->second[consumed])))
                {
                    ++consumed;
                }
                if (consumed != found->second.size() || !std::isfinite(parsed))
                {
                    throw std::invalid_argument("trailing RPC scalar content");
                }
                *value = parsed;
                return true;
            }
            catch (const std::exception&)
            {
                if (error)
                {
                    *error = "RPC metadata key " + key + " is not a finite number";
                }
                return false;
            }
        }

        bool parseCoefficients(const RpcMetadata& metadata,
                               const std::string& key,
                               RpcCoefficients* coefficients,
                               std::string* error)
        {
            const auto found = metadata.find(key);
            if (found == metadata.end())
            {
                if (error)
                {
                    *error = "RPC metadata is missing required key " + key;
                }
                return false;
            }

            std::string values = found->second;
            std::replace(values.begin(), values.end(), ',', ' ');
            std::istringstream stream(values);
            for (double& coefficient : *coefficients)
            {
                if (!(stream >> coefficient) || !std::isfinite(coefficient))
                {
                    if (error)
                    {
                        *error = "RPC metadata key " + key + " must contain exactly 20 finite coefficients";
                    }
                    return false;
                }
            }
            double extra = 0.0;
            if (stream >> extra)
            {
                if (error)
                {
                    *error = "RPC metadata key " + key + " contains more than 20 coefficients";
                }
                return false;
            }
            stream.clear();
            stream >> std::ws;
            if (!stream.eof())
            {
                if (error)
                {
                    *error = "RPC metadata key " + key + " contains invalid trailing content";
                }
                return false;
            }
            return true;
        }

    } // namespace

    Result<RpcParameters> rpcParametersFromMetadata(const RpcMetadata& metadata)
    {
        const RpcMetadata normalized = normalizedMetadata(metadata);
        RpcParameters parsed;
        std::string error;
        if (!parseDouble(normalized, "LINE_OFF", &parsed.lineOffset, &error) ||
            !parseDouble(normalized, "SAMP_OFF", &parsed.sampleOffset, &error) ||
            !parseDouble(normalized, "LAT_OFF", &parsed.latitudeOffset, &error) ||
            !parseDouble(normalized, "LONG_OFF", &parsed.longitudeOffset, &error) ||
            !parseDouble(normalized, "HEIGHT_OFF", &parsed.heightOffset, &error) ||
            !parseDouble(normalized, "LINE_SCALE", &parsed.lineScale, &error) ||
            !parseDouble(normalized, "SAMP_SCALE", &parsed.sampleScale, &error) ||
            !parseDouble(normalized, "LAT_SCALE", &parsed.latitudeScale, &error) ||
            !parseDouble(normalized, "LONG_SCALE", &parsed.longitudeScale, &error) ||
            !parseDouble(normalized, "HEIGHT_SCALE", &parsed.heightScale, &error) ||
            !parseCoefficients(normalized, "LINE_NUM_COEFF", &parsed.lineNumerator, &error) ||
            !parseCoefficients(normalized, "LINE_DEN_COEFF", &parsed.lineDenominator, &error) ||
            !parseCoefficients(normalized, "SAMP_NUM_COEFF", &parsed.sampleNumerator, &error) ||
            !parseCoefficients(normalized, "SAMP_DEN_COEFF", &parsed.sampleDenominator, &error))
        {
            return Result<RpcParameters>::failure(CameraErrorCode::ParseFailure, std::move(error), "RPC metadata");
        }

        double optionalError = 0.0;
        if (parseDouble(normalized, "ERR_BIAS", &optionalError, nullptr, false) && optionalError >= 0.0)
        {
            parsed.errorBiasMeters = optionalError;
        }
        if (parseDouble(normalized, "ERR_RAND", &optionalError, nullptr, false) && optionalError >= 0.0)
        {
            parsed.errorRandomMeters = optionalError;
        }
        try
        {
            (void)RpcDefinition::create(CameraDefinitionId("rpc-metadata-validation"), FrameId("EPSG:4978"), parsed);
        }
        catch (const std::exception& exception)
        {
            return Result<RpcParameters>::failure(CameraErrorCode::InvalidModelState, exception.what(), "RPC metadata");
        }
        return Result<RpcParameters>::success(std::move(parsed));
    }

} // namespace placamera

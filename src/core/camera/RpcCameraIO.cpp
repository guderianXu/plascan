#include "RpcCameraIO.h"

#include <cpl_conv.h>
#include <cpl_string.h>
#include <gdal_priv.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

namespace xjw
{
    namespace
    {

        struct GdalDatasetDeleter
        {
            void operator()(GDALDataset* dataset) const
            {
                if (dataset)
                {
                    GDALClose(dataset);
                }
            }
        };

        using GdalDatasetPtr = std::unique_ptr<GDALDataset, GdalDatasetDeleter>;

        void ensureGdalRegistered()
        {
            static std::once_flag flag;
            std::call_once(flag, []() { GDALAllRegister(); });
        }

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
                         std::string* errorMessage,
                         bool required = true)
        {
            const auto found = metadata.find(key);
            if (found == metadata.end())
            {
                if (!required)
                {
                    return false;
                }
                if (errorMessage)
                {
                    *errorMessage = "RPC metadata is missing required key " + key;
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
                if (errorMessage)
                {
                    *errorMessage = "RPC metadata key " + key + " is not a finite number";
                }
                return false;
            }
        }

        bool parseCoefficients(const RpcMetadata& metadata,
                               const std::string& key,
                               RpcCameraModel::Coefficients* coefficients,
                               std::string* errorMessage)
        {
            const auto found = metadata.find(key);
            if (found == metadata.end())
            {
                if (errorMessage)
                {
                    *errorMessage = "RPC metadata is missing required key " + key;
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
                    if (errorMessage)
                    {
                        *errorMessage = "RPC metadata key " + key + " must contain exactly 20 finite coefficients";
                    }
                    return false;
                }
            }
            double extra = 0.0;
            if (stream >> extra)
            {
                if (errorMessage)
                {
                    *errorMessage = "RPC metadata key " + key + " contains more than 20 coefficients";
                }
                return false;
            }
            stream.clear();
            stream >> std::ws;
            if (!stream.eof())
            {
                if (errorMessage)
                {
                    *errorMessage = "RPC metadata key " + key + " contains invalid trailing content";
                }
                return false;
            }
            return true;
        }

        RpcMetadata metadataFromStringList(char** metadata)
        {
            RpcMetadata result;
            for (char** item = metadata; item && *item; ++item)
            {
                char* key = nullptr;
                const char* value = CPLParseNameValue(*item, &key);
                if (key && value)
                {
                    result[key] = value;
                }
                CPLFree(key);
            }
            return result;
        }

    } // namespace

    bool rpcCameraFromMetadata(const RpcMetadata& metadata, RpcCameraModel* camera, std::string* errorMessage)
    {
        if (!camera)
        {
            if (errorMessage)
            {
                *errorMessage = "RPC camera output is null";
            }
            return false;
        }
        const RpcMetadata normalized = normalizedMetadata(metadata);
        RpcCameraModel::Parameters parameters;
        if (!parseDouble(normalized, "LINE_OFF", &parameters.lineOffset, errorMessage) ||
            !parseDouble(normalized, "SAMP_OFF", &parameters.sampleOffset, errorMessage) ||
            !parseDouble(normalized, "LAT_OFF", &parameters.latitudeOffset, errorMessage) ||
            !parseDouble(normalized, "LONG_OFF", &parameters.longitudeOffset, errorMessage) ||
            !parseDouble(normalized, "HEIGHT_OFF", &parameters.heightOffset, errorMessage) ||
            !parseDouble(normalized, "LINE_SCALE", &parameters.lineScale, errorMessage) ||
            !parseDouble(normalized, "SAMP_SCALE", &parameters.sampleScale, errorMessage) ||
            !parseDouble(normalized, "LAT_SCALE", &parameters.latitudeScale, errorMessage) ||
            !parseDouble(normalized, "LONG_SCALE", &parameters.longitudeScale, errorMessage) ||
            !parseDouble(normalized, "HEIGHT_SCALE", &parameters.heightScale, errorMessage) ||
            !parseCoefficients(normalized, "LINE_NUM_COEFF", &parameters.lineNumerator, errorMessage) ||
            !parseCoefficients(normalized, "LINE_DEN_COEFF", &parameters.lineDenominator, errorMessage) ||
            !parseCoefficients(normalized, "SAMP_NUM_COEFF", &parameters.sampleNumerator, errorMessage) ||
            !parseCoefficients(normalized, "SAMP_DEN_COEFF", &parameters.sampleDenominator, errorMessage))
        {
            return false;
        }

        double optional_error = 0.0;
        if (parseDouble(normalized, "ERR_BIAS", &optional_error, nullptr, false) && optional_error >= 0.0)
        {
            parameters.errorBiasMeters = optional_error;
        }
        if (parseDouble(normalized, "ERR_RAND", &optional_error, nullptr, false) && optional_error >= 0.0)
        {
            parameters.errorRandomMeters = optional_error;
        }
        return camera->setParameters(parameters, errorMessage);
    }

    bool loadRpcCameraFromRaster(const std::string& rasterPath, RpcCameraModel* camera, std::string* errorMessage)
    {
        if (!camera || rasterPath.empty())
        {
            if (errorMessage)
            {
                *errorMessage = camera ? "RPC raster path is empty" : "RPC camera output is null";
            }
            return false;
        }
        ensureGdalRegistered();
        GdalDatasetPtr dataset(static_cast<GDALDataset*>(
            GDALOpenEx(rasterPath.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
        if (!dataset)
        {
            if (errorMessage)
            {
                *errorMessage = "Cannot open RPC raster: " + rasterPath;
            }
            return false;
        }

        const RpcMetadata metadata = metadataFromStringList(dataset->GetMetadata("RPC"));
        if (metadata.empty())
        {
            if (errorMessage)
            {
                *errorMessage = "Raster has no GDAL RPC metadata: " + rasterPath;
            }
            return false;
        }
        RpcCameraModel parsed;
        if (!rpcCameraFromMetadata(metadata, &parsed, errorMessage))
        {
            return false;
        }
        parsed.setImageSize(CameraImageSize{dataset->GetRasterXSize(), dataset->GetRasterYSize()});
        *camera = std::move(parsed);
        return true;
    }

} // namespace xjw

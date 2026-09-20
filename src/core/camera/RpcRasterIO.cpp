#include "RpcRasterIO.h"

#include <cpl_conv.h>
#include <cpl_string.h>
#include <gdal_priv.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

namespace xjw::camera_models::rpc
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
                               RpcDefinition::Coefficients* coefficients,
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

    bool rpcParametersFromMetadata(const RpcMetadata& metadata,
                                   RpcDefinition::Parameters* parameters,
                                   std::string* error)
    {
        if (!parameters)
        {
            if (error)
            {
                *error = "RPC parameter output is null";
            }
            return false;
        }
        const RpcMetadata normalized = normalizedMetadata(metadata);
        RpcDefinition::Parameters parsed;
        if (!parseDouble(normalized, "LINE_OFF", &parsed.lineOffset, error) ||
            !parseDouble(normalized, "SAMP_OFF", &parsed.sampleOffset, error) ||
            !parseDouble(normalized, "LAT_OFF", &parsed.latitudeOffset, error) ||
            !parseDouble(normalized, "LONG_OFF", &parsed.longitudeOffset, error) ||
            !parseDouble(normalized, "HEIGHT_OFF", &parsed.heightOffset, error) ||
            !parseDouble(normalized, "LINE_SCALE", &parsed.lineScale, error) ||
            !parseDouble(normalized, "SAMP_SCALE", &parsed.sampleScale, error) ||
            !parseDouble(normalized, "LAT_SCALE", &parsed.latitudeScale, error) ||
            !parseDouble(normalized, "LONG_SCALE", &parsed.longitudeScale, error) ||
            !parseDouble(normalized, "HEIGHT_SCALE", &parsed.heightScale, error) ||
            !parseCoefficients(normalized, "LINE_NUM_COEFF", &parsed.lineNumerator, error) ||
            !parseCoefficients(normalized, "LINE_DEN_COEFF", &parsed.lineDenominator, error) ||
            !parseCoefficients(normalized, "SAMP_NUM_COEFF", &parsed.sampleNumerator, error) ||
            !parseCoefficients(normalized, "SAMP_DEN_COEFF", &parsed.sampleDenominator, error))
        {
            return false;
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
            (void)RpcDefinition::create(camera_core::CameraDefinitionId("rpc-metadata-validation"),
                                        xjw::coordinate_system::CoordinateFrameId("EPSG:4978"),
                                        parsed);
        }
        catch (const std::exception& exception)
        {
            if (error)
            {
                *error = exception.what();
            }
            return false;
        }
        *parameters = std::move(parsed);
        if (error)
        {
            error->clear();
        }
        return true;
    }

    bool readRpcRasterData(const std::string& rasterPath, RpcRasterData* data, std::string* error)
    {
        if (!data || rasterPath.empty())
        {
            if (error)
            {
                *error = data ? "RPC raster path is empty" : "RPC raster output is null";
            }
            return false;
        }
        ensureGdalRegistered();
        GdalDatasetPtr dataset(static_cast<GDALDataset*>(
            GDALOpenEx(rasterPath.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
        if (!dataset)
        {
            if (error)
            {
                *error = "Cannot open RPC raster: " + rasterPath;
            }
            return false;
        }

        const RpcMetadata metadata = metadataFromStringList(dataset->GetMetadata("RPC"));
        if (metadata.empty())
        {
            if (error)
            {
                *error = "Raster has no GDAL RPC metadata: " + rasterPath;
            }
            return false;
        }
        RpcRasterData parsed;
        if (!rpcParametersFromMetadata(metadata, &parsed.parameters, error))
        {
            return false;
        }
        parsed.imageSize = {dataset->GetRasterXSize(), dataset->GetRasterYSize()};
        if (!parsed.imageSize.isValid())
        {
            if (error)
            {
                *error = "RPC raster has invalid dimensions: " + rasterPath;
            }
            return false;
        }
        *data = std::move(parsed);
        if (error)
        {
            error->clear();
        }
        return true;
    }

    std::shared_ptr<const RpcInstance> importRpcRasterInstance(const std::string& rasterPath,
                                                               camera_core::CameraDefinitionId definitionId,
                                                               camera_core::CameraInstanceId instanceId,
                                                               camera_core::ImageId imageId,
                                                               xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                               std::string* error)
    {
        RpcRasterData data;
        if (!readRpcRasterData(rasterPath, &data, error))
        {
            return nullptr;
        }
        try
        {
            auto definition =
                RpcDefinition::create(std::move(definitionId), std::move(worldFrame), std::move(data.parameters));
            auto instance = RpcInstance::create(
                std::move(instanceId), std::move(imageId), std::move(definition), data.imageSize);
            if (error)
            {
                error->clear();
            }
            return std::make_shared<const RpcInstance>(std::move(instance));
        }
        catch (const std::exception& exception)
        {
            if (error)
            {
                *error = exception.what();
            }
            return nullptr;
        }
    }

} // namespace xjw::camera_models::rpc

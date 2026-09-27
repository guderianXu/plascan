#include <exception>
#include <memory>
#include <mutex>
#include <utility>

#include <cpl_conv.h>
#include <cpl_string.h>
#include <gdal_priv.h>

#include "placamera/rpc_raster.h"

namespace placamera
{
    namespace
    {

        struct RasterDatasetDeleter
        {
            void operator()(GDALDataset* dataset) const
            {
                if (dataset)
                {
                    GDALClose(dataset);
                }
            }
        };

        using RasterDatasetPtr = std::unique_ptr<GDALDataset, RasterDatasetDeleter>;

        void ensureRasterDriversRegistered()
        {
            static std::once_flag flag;
            std::call_once(flag, []() { GDALAllRegister(); });
        }

        std::string pathText(const std::filesystem::path& path)
        {
            const auto utf8 = path.u8string();
            return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
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

    Result<RpcRasterData> readRpcRasterData(const std::filesystem::path& rasterPath)
    {
        if (rasterPath.empty())
        {
            return Result<RpcRasterData>::failure(CameraErrorCode::InvalidArgument, "RPC raster path is empty");
        }
        ensureRasterDriversRegistered();
        const std::string raster_path = pathText(rasterPath);
        RasterDatasetPtr dataset(static_cast<GDALDataset*>(
            GDALOpenEx(raster_path.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
        if (!dataset)
        {
            return Result<RpcRasterData>::failure(CameraErrorCode::IoFailure, "cannot open RPC raster", raster_path);
        }

        const RpcMetadata metadata = metadataFromStringList(dataset->GetMetadata("RPC"));
        if (metadata.empty())
        {
            return Result<RpcRasterData>::failure(
                CameraErrorCode::ParseFailure, "raster has no RPC metadata", raster_path);
        }
        RpcRasterData parsed;
        auto parameters = rpcParametersFromMetadata(metadata);
        if (!parameters)
        {
            CameraError error = parameters.error();
            error.source = raster_path;
            return Result<RpcRasterData>::failure(std::move(error));
        }
        parsed.parameters = parameters.takeValue();
        parsed.imageSize = {dataset->GetRasterXSize(), dataset->GetRasterYSize()};
        if (!parsed.imageSize.isValid())
        {
            return Result<RpcRasterData>::failure(
                CameraErrorCode::InvalidImageSize, "RPC raster has invalid dimensions", raster_path);
        }
        return Result<RpcRasterData>::success(std::move(parsed));
    }

    Result<CameraModelPtr<RpcModel>> importRpcRasterModel(const std::filesystem::path& rasterPath,
                                                          CameraDefinitionId definitionId,
                                                          CameraInstanceId instanceId,
                                                          ImageId imageId,
                                                          FrameId worldFrame)
    {
        auto data = readRpcRasterData(rasterPath);
        if (!data)
        {
            return Result<CameraModelPtr<RpcModel>>::failure(data.error());
        }
        try
        {
            RpcRasterData raster = data.takeValue();
            auto definition =
                RpcDefinition::create(std::move(definitionId), std::move(worldFrame), std::move(raster.parameters));
            auto instance =
                RpcModel::create(std::move(instanceId), std::move(imageId), std::move(definition), raster.imageSize);
            return Result<CameraModelPtr<RpcModel>>::success(std::make_shared<const RpcModel>(std::move(instance)));
        }
        catch (const CameraValidationError& error)
        {
            return Result<CameraModelPtr<RpcModel>>::failure(error.code(), error.what(), pathText(rasterPath));
        }
        catch (const std::exception& exception)
        {
            return Result<CameraModelPtr<RpcModel>>::failure(
                CameraErrorCode::InvalidModelState, exception.what(), pathText(rasterPath));
        }
    }

} // namespace placamera

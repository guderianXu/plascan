#include "MvsImageMetadataProbe.h"

#include <gdal_priv.h>

#include <mutex>

namespace xjw::mvs
{
namespace
{

void registerGdalOnce()
{
    static std::once_flag registered;
    std::call_once(registered, []()
    {
        GDALAllRegister();
    });
}

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

} // namespace

bool probeMvsImageMetadata(const std::string &imagePath,
                           MvsImageMetadata *metadata,
                           std::string *errorMessage)
{
    if (!metadata)
    {
        setError(errorMessage, "MVS image metadata output is null");
        return false;
    }
    *metadata = {};
    if (imagePath.empty())
    {
        setError(errorMessage, "MVS image metadata path is empty");
        return false;
    }

    registerGdalOnce();
    GDALDataset *dataset = static_cast<GDALDataset *>(GDALOpenEx(
        imagePath.c_str(),
        GDAL_OF_RASTER | GDAL_OF_READONLY,
        nullptr,
        nullptr,
        nullptr));
    if (!dataset)
    {
        setError(errorMessage, "unable to open image header: " + imagePath);
        return false;
    }

    metadata->width = dataset->GetRasterXSize();
    metadata->height = dataset->GetRasterYSize();
    GDALClose(dataset);
    if (metadata->width <= 0 || metadata->height <= 0)
    {
        *metadata = {};
        setError(errorMessage, "image header has invalid dimensions: " + imagePath);
        return false;
    }
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

} // namespace xjw::mvs

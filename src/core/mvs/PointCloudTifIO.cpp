#include "PointCloudTifIO.h"

#include <gdal_priv.h>
#include <cpl_string.h>

#include <cmath>
#include <cstdio>
#include <string>

namespace xjw
{
namespace mvs
{

static void ensureGdalRegistered()
{
    static bool done = false;
    if (!done)
    {
        GDALAllRegister();
        done = true;
    }
}

bool PointCloudTifIO::writeTif(const std::string &path,
                               const TriangulationResult &tri,
                               std::string *errorMsg)
{
    if (tri.pointCloud.empty())
    {
        if (errorMsg) *errorMsg = "Empty point cloud";
        return false;
    }

    ensureGdalRegistered();

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
        if (errorMsg) *errorMsg = "GTiff driver not available";
        return false;
    }

    const int rows = tri.pointCloud.rows;
    const int cols = tri.pointCloud.cols;

    char **opts = nullptr;
    opts = CSLSetNameValue(opts, "COMPRESS", "LZW");

    GDALDataset *ds = driver->Create(path.c_str(), cols, rows, 4,
                                     GDT_Float32, opts);
    CSLDestroy(opts);
    if (!ds)
    {
        if (errorMsg) *errorMsg = "Failed to create TIF: " + path;
        return false;
    }

    // Set POINT_OFFSET metadata (ASP convention)
    char offsetStr[256];
    snprintf(offsetStr, sizeof(offsetStr), "%.12f %.12f %.12f",
             tri.pointOffset[0], tri.pointOffset[1], tri.pointOffset[2]);
    ds->SetMetadataItem("POINT_OFFSET", offsetStr);

    // Write 4 bands: X, Y, Z (offset-subtracted), error
    std::vector<float> bandBuf(static_cast<size_t>(rows) * cols, 0.0f);

    for (int b = 0; b < 4; ++b)
    {
        std::fill(bandBuf.begin(), bandBuf.end(), 0.0f);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                if (tri.validMask.at<uint8_t>(r, c) == 0) continue;
                if (b < 3)
                {
                    bandBuf[r * cols + c] = static_cast<float>(
                        tri.pointCloud.at<cv::Vec3d>(r, c)[b]);
                }
                else
                {
                    bandBuf[r * cols + c] = tri.errorMap.at<float>(r, c);
                }
            }
        }
        CPLErr err = ds->GetRasterBand(b + 1)->RasterIO(
            GF_Write, 0, 0, cols, rows,
            bandBuf.data(), cols, rows, GDT_Float32, 0, 0);
        if (err != CE_None)
        {
            if (errorMsg) *errorMsg = "RasterIO write failed for band " + std::to_string(b + 1);
            GDALClose(ds);
            return false;
        }
    }

    GDALClose(ds);
    fprintf(stderr, "[PointCloudTifIO] wrote %s (%dx%d, %d valid)\n",
            path.c_str(), cols, rows, tri.validPoints);
    return true;
}

bool PointCloudTifIO::readTif(const std::string &path,
                              TriangulationResult &tri,
                              std::string *errorMsg)
{
    ensureGdalRegistered();

    GDALDataset *ds = static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (!ds)
    {
        if (errorMsg) *errorMsg = "Cannot open TIF: " + path;
        return false;
    }

    int cols = ds->GetRasterXSize();
    int rows = ds->GetRasterYSize();
    int bands = ds->GetRasterCount();
    if (bands < 4)
    {
        if (errorMsg) *errorMsg = "Expected 4 bands, got " + std::to_string(bands);
        GDALClose(ds);
        return false;
    }

    // Read POINT_OFFSET
    const char *offsetMeta = ds->GetMetadataItem("POINT_OFFSET");
    tri.pointOffset = {0, 0, 0};
    if (offsetMeta)
    {
        sscanf(offsetMeta, "%lf %lf %lf",
               &tri.pointOffset[0], &tri.pointOffset[1], &tri.pointOffset[2]);
    }

    tri.pointCloud = cv::Mat(rows, cols, CV_64FC3, cv::Scalar(0, 0, 0));
    tri.errorMap = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    tri.validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    tri.totalPixels = rows * cols;
    tri.validPoints = 0;

    std::vector<float> buf(static_cast<size_t>(rows) * cols);
    std::vector<std::vector<float>> bandData(4);
    for (int b = 0; b < 4; ++b)
    {
        bandData[b].resize(static_cast<size_t>(rows) * cols);
        ds->GetRasterBand(b + 1)->RasterIO(
            GF_Read, 0, 0, cols, rows,
            bandData[b].data(), cols, rows, GDT_Float32, 0, 0);
    }
    GDALClose(ds);

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx = r * cols + c;
            float x = bandData[0][idx];
            float y = bandData[1][idx];
            float z = bandData[2][idx];
            float e = bandData[3][idx];
            if (x == 0.0f && y == 0.0f && z == 0.0f) continue;

            tri.pointCloud.at<cv::Vec3d>(r, c) = {x, y, z};
            tri.errorMap.at<float>(r, c) = e;
            tri.validMask.at<uint8_t>(r, c) = 255;
            ++tri.validPoints;
        }
    }

    fprintf(stderr, "[PointCloudTifIO] read %s: %dx%d, %d valid, offset=(%.4f,%.4f,%.4f)\n",
            path.c_str(), cols, rows, tri.validPoints,
            tri.pointOffset[0], tri.pointOffset[1], tri.pointOffset[2]);
    return true;
}

} // namespace mvs
} // namespace xjw

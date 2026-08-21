#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace xjw::camera
{

struct ColmapRasterModel
{
    std::string model;
    int width = 0;
    int height = 0;
    std::vector<double> params;
};

struct ColmapUndistortedRaster
{
    cv::Mat image;
    cv::Mat validMask;
    std::array<double, 9> rasterIndexIntrinsics{};
    double focalScale = 1.0;
};

bool isSupportedColmapPreUndistortModel(const ColmapRasterModel &camera);

ColmapUndistortedRaster undistortColmapRaster(
    const cv::Mat &source,
    const ColmapRasterModel &camera);

} // namespace xjw::camera

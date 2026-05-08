#include "StereoDenseCloudPipelineOutput.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <cstdio>

namespace xjw
{
namespace mvs
{

bool writeStereoPipelinePly(const std::string &path,
                            const TriangulationResult &triResult,
                            const cv::Mat &grayImage,
                            std::string &errorMessage)
{
    size_t numValid = static_cast<size_t>(triResult.validPoints);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(numValid, 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(numValid, 3);
    const auto &off = triResult.pointOffset;

    size_t idx = 0;
    for (int r = 0; r < triResult.pointCloud.rows; ++r)
    {
        for (int c = 0; c < triResult.pointCloud.cols; ++c)
        {
            if (triResult.validMask.at<uint8_t>(r, c) == 0) continue;
            auto &pt = triResult.pointCloud.at<cv::Vec3d>(r, c);
            float x = static_cast<float>(pt[0] + off[0]);
            float y = static_cast<float>(pt[1] + off[1]);
            float z = static_cast<float>(pt[2] + off[2]);
            uint8_t gray = 128;
            if (r < grayImage.rows && c < grayImage.cols)
                gray = grayImage.at<uint8_t>(r, c);
            pts(idx, 0) = x;
            pts(idx, 1) = y;
            pts(idx, 2) = z;
            colors(idx, 0) = gray;
            colors(idx, 1) = gray;
            colors(idx, 2) = gray;
            ++idx;
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(pts));
    cloud.setColors(std::move(colors));

    try
    {
        plapoint::io::writePly<float>(path, cloud, plapoint::io::PlyFormat::BinaryLE);
    }
    catch (const std::exception &e)
    {
        errorMessage = e.what();
        return false;
    }

    std::fprintf(stderr, "[StereoPipeline] PLY: %s (%zu points)\n", path.c_str(), cloud.size());
    return true;
}

} // namespace mvs
} // namespace xjw

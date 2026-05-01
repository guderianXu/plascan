#include "StereoDenseCloudPipelineOutput.h"

#include "pointcloud/data/PointCloud.h"
#include "pointcloud/io/PointCloudIO.h"

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
    pointcloud::PointCloud pc;
    pc.reserve(triResult.validPoints);
    const auto &off = triResult.pointOffset;

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
            pc.addPoint({x, y, z}, {gray, gray, gray, 255});
        }
    }

    pointcloud::PointCloudWriteOptions plyOpts;
    plyOpts.format = pointcloud::PointCloudFileFormat::PlyBinaryLittleEndian;
    plyOpts.writeNormals = false;
    pointcloud::PointCloudIOResult plyRes;
    if (!pointcloud::writePlyPointCloud(path, pc, plyOpts, &plyRes))
    {
        errorMessage = plyRes.errorMessage;
        return false;
    }

    std::fprintf(stderr, "[StereoPipeline] PLY: %s (%zu points)\n", path.c_str(), pc.size());
    return true;
}

} // namespace mvs
} // namespace xjw

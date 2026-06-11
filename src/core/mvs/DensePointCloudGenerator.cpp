// =============================================================================
// 文件: DensePointCloudGenerator.cpp
// 模块: MVS - 稠密点云生成（简化版）
// =============================================================================

#include "DensePointCloudGenerator.h"
#include "DensePointCloudCUDA.h"
#include "log/Logger.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace xjw
{
namespace mvs
{

// =============================================================================
DensePointCloudGenerator::DensePointCloudGenerator(
    const std::vector<CameraView> &views,
    const DenseCloudGenConfig     &config)
    : m_views(views), m_config(config)
{
}

// =============================================================================
bool DensePointCloudGenerator::generate(const UnprojectInput &input,
                                         int                   frameIdx,
                                         QVector<DensePoint>  &cloud,
                                         void                 * /*unused*/,
                                         std::string          *errorMsg)
{
    if (!input.fusedDepth || input.fusedDepth->empty())
    {
        if (errorMsg)
        {
            *errorMsg = "融合深度图为空";
        }
        return false;
    }

    const cv::Mat &depth = *input.fusedDepth;
    const cv::Mat mask   = (input.validMask && !input.validMask->empty())
                           ? *input.validMask
                           : cv::Mat();

    // 选取参考帧
    int refIdx = -1;
    if (frameIdx >= 0 && frameIdx < static_cast<int>(m_views.size()))
    {
        refIdx = frameIdx;
    }
    if (refIdx < 0)
    {
        for (int ci : input.colorFrameIndices)
        {
            if (ci >= 0 && ci < static_cast<int>(m_views.size()))
            {
                refIdx = ci;
                break;
            }
        }
    }
    if (refIdx < 0 && !m_views.empty())
    {
        refIdx = 0;
    }
    if (refIdx < 0)
    {
        if (errorMsg)
        {
            *errorMsg = "没有可用的参考相机视图";
        }
        return false;
    }

    PositiveDepthCameraModel refCam = m_views[refIdx].positiveDepthModel();

    // 构建选项
    DenseCloudOptions opt;
    opt.useGPU    = m_config.useCudaUnprojection;
    opt.minDepth  = m_config.minDepth;
    opt.maxDepth  = m_config.maxDepth;
    opt.subsample = m_config.subsample;

    if (m_config.clipToSparseAabb)
    {
        opt.clipAABB = true;
        const float pad = m_config.aabbPaddingRatio;
        for (int k = 0; k < 3; ++k)
        {
            float range = m_config.sparseMaxPt[k] - m_config.sparseMinPt[k];
            (void)range;
            (void)k; // used only for iteration
        }
        float rx = m_config.sparseMaxPt[0] - m_config.sparseMinPt[0];
        float ry = m_config.sparseMaxPt[1] - m_config.sparseMinPt[1];
        float rz = m_config.sparseMaxPt[2] - m_config.sparseMinPt[2];
        opt.minX = m_config.sparseMinPt[0] - rx * pad;
        opt.maxX = m_config.sparseMaxPt[0] + rx * pad;
        opt.minY = m_config.sparseMinPt[1] - ry * pad;
        opt.maxY = m_config.sparseMaxPt[1] + ry * pad;
        opt.minZ = m_config.sparseMinPt[2] - rz * pad;
        opt.maxZ = m_config.sparseMaxPt[2] + rz * pad;
    }

    // 加载颜色图
    cv::Mat colorImg = cv::imread(m_views[refIdx].imagePath, cv::IMREAD_COLOR);

    // 如果深度图和颜色图尺寸不同，缩放内参
    PositiveDepthCameraModel depthCam = refCam;
    if (!colorImg.empty() && (depth.cols != colorImg.cols || depth.rows != colorImg.rows))
    {
        const float sx = static_cast<float>(depth.cols) / static_cast<float>(colorImg.cols);
        const float sy = static_cast<float>(depth.rows) / static_cast<float>(colorImg.rows);
        depthCam.fx *= sx;
        depthCam.fy *= sy;
        depthCam.cx *= sx;
        depthCam.cy *= sy;
    }

    // 反投影
    std::vector<DensePoint> raw;
    if (opt.useGPU)
    {
        std::string gpuErr;
        raw = DensePointCloudCUDA::unprojectGPU(depth, mask, depthCam, colorImg,
                                                 opt.minDepth, opt.maxDepth, &gpuErr);
        if (raw.empty() && !gpuErr.empty())
        {
            LOG_WARN("[DenseGen] GPU 反投影失败: %s，回退 CPU", gpuErr.c_str());
            raw = DenseCloudBuilder::unproject(depth, mask, depthCam, colorImg, opt);
        }
    }
    else
    {
        raw = DenseCloudBuilder::unproject(depth, mask, depthCam, colorImg, opt);
    }
    LOG_INFO("[DenseGen] 反投影得到 %d 个 3D 点", (int)raw.size());

    cloud.clear();
    cloud.reserve(static_cast<int>(raw.size()));
    for (const auto &p : raw)
    {
        cloud.push_back(p);
    }
    return true;
}

// =============================================================================
void DensePointCloudGenerator::postProcess(QVector<DensePoint> &cloud)
{
    if (cloud.isEmpty())
    {
        return;
    }

    // 体素降采样
    if (m_config.voxelSize > 0.0f)
    {
        std::vector<DensePoint> input(cloud.begin(), cloud.end());
        std::vector<DensePoint> filtered = DenseCloudBuilder::voxelDownsample(
            input, m_config.voxelSize, m_config.processingDevice);
        QVector<DensePoint> downsampled;
        downsampled.reserve(static_cast<int>(filtered.size()));
        for (const DensePoint &point : filtered)
        {
            downsampled.push_back(point);
        }
        fprintf(stderr, "[PostProcess] 体素降采样: %d → %d 点 (vs=%.4f)\n",
                static_cast<int>(cloud.size()), static_cast<int>(downsampled.size()), m_config.voxelSize);
        cloud = std::move(downsampled);
    }
}

// =============================================================================
bool DensePointCloudGenerator::saveToXyz(const QVector<DensePoint> &cloud,
                                          const std::string          &path,
                                          std::string                *errorMsg)
{
    std::ofstream ofs(path);
    if (!ofs)
    {
        if (errorMsg)
        {
            *errorMsg = "无法创建文件: " + path;
        }
        return false;
    }
    for (const auto &p : cloud)
    {
        ofs << p.x << ' ' << p.y << ' ' << p.z
            << ' ' << (int)p.r << ' ' << (int)p.g << ' ' << (int)p.b << '\n';
    }

    return ofs.good();
}

// =============================================================================
bool DensePointCloudGenerator::saveToPly(const QVector<DensePoint> &cloud,
                                          const std::string          &path,
                                          std::string                *errorMsg)
{
    std::vector<DensePoint> std_cloud(cloud.begin(), cloud.end());
    return DenseCloudBuilder::savePLY(path, std_cloud, errorMsg);
}

} // namespace mvs
} // namespace xjw

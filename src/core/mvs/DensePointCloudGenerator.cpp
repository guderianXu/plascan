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
#include <unordered_map>
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
        float xMin = cloud[0].x, xMax = cloud[0].x;
        float yMin = cloud[0].y, yMax = cloud[0].y;
        float zMin = cloud[0].z, zMax = cloud[0].z;
        for (const auto &p : cloud)
        {
            xMin = std::min(xMin, p.x);
            xMax = std::max(xMax, p.x);
            yMin = std::min(yMin, p.y);
            yMax = std::max(yMax, p.y);
            zMin = std::min(zMin, p.z);
            zMax = std::max(zMax, p.z);
        }
        const float vs = m_config.voxelSize;
        const int nx = std::max(1, static_cast<int>((xMax - xMin) / vs) + 1);
        const int ny = std::max(1, static_cast<int>((yMax - yMin) / vs) + 1);

        struct VoxelAcc
        {
            double sx = 0;
            double sy = 0;
            double sz = 0;
            float sr = 0;
            float sg = 0;
            float sb = 0;
            int cnt = 0;
        };
        std::unordered_map<int64_t, VoxelAcc> voxelMap;
        voxelMap.reserve(cloud.size());

        for (const auto &p : cloud)
        {
            int ix = std::max(0, (int)((p.x - xMin) / vs));
            int iy = std::max(0, (int)((p.y - yMin) / vs));
            int iz = std::max(0, (int)((p.z - zMin) / vs));
            int64_t key = (int64_t)iz * nx * ny + (int64_t)iy * nx + ix;
            auto &acc = voxelMap[key];
            acc.sx += p.x;
            acc.sy += p.y;
            acc.sz += p.z;
            acc.sr += p.r;
            acc.sg += p.g;
            acc.sb += p.b;
            ++acc.cnt;
        }

        QVector<DensePoint> downsampled;
        downsampled.reserve(static_cast<int>(voxelMap.size()));
        for (const auto &kv : voxelMap)
        {
            const VoxelAcc &a = kv.second;
            float n = static_cast<float>(a.cnt);
            DensePoint p;
            p.x = float(a.sx / n);
            p.y = float(a.sy / n);
            p.z = float(a.sz / n);
            p.r = uint8_t(a.sr / n);
            p.g = uint8_t(a.sg / n);
            p.b = uint8_t(a.sb / n);
            downsampled.push_back(p);
        }
        fprintf(stderr, "[PostProcess] 体素降采样: %d → %d 点 (vs=%.4f)\n",
                cloud.size(), downsampled.size(), vs);
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

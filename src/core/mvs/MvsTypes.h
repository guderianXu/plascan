// =============================================================================
// 文件: MvsTypes.h
// 模块: MVS (Multi-View Stereo)
// 说明:
//   MVS 管线的所有公共数据结构定义。
//
//   核心设计原则（COLMAP 方法）：
//     - 所有相机参数在进入 MVS 之前统一转换为 COLMAP 正深度约定
//     - Z_cam > 0 恒成立，不再需要追踪 depthFlippedZ / uDir / vDir 等 hack
//
//   深度图约定：
//     depthMap(v, u) = Z_cam（正值，代表相机坐标系 Z 轴距离）
//     0.0 = 无效
// =============================================================================
#pragma once

#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <functional>

#include "camera/Camera.h"

#include <opencv2/core.hpp>

namespace xjw
{
namespace mvs
{

using PositiveDepthCameraModel = xjw::Camera::PositiveDepthModel;

// =============================================================================
// PatchMatch 配置
// =============================================================================
struct PatchMatchConfig
{
    int   numIterations      = 16;      ///< PatchMatch 迭代次数（16轮保证宽深度范围下充分收敛）
    int   patchHalf          = 7;       ///< NCC 块半径（块大小 = 2r+1 = 15×15），更大→更鲁棒
    int   numSourceViews     = 4;
    int   cpuThreadCount     = 1;       ///< CPU 路径的像素级并行线程数
    float confidenceThresh   = 0.30f;   ///< 多图默认阈值；少视图时由调用方自适应降低
    bool  useCuda            = true;
    int   downsampleFactor   = 2;       ///< 降采样因子（2=半分辨率，速度提升约4倍）
    bool  doMedianBlur       = true;
    int   medianKernelSize   = 5;
    bool  doBilateralFilter  = true;
    int   bilateralD         = 9;
    float bilateralSigmaColor= 50.f;
    float bilateralSigmaSpace= 5.f;

    // 几何一致性检查（PatchMatch 第二趟）
    bool  geomConsistency      = true;   ///< 是否启用几何一致性检查
    float geomConsistencyMaxErr= 1.0f;   ///< 允许的最大质心深度偏差（像素）

    // CUDA 线程块尺寸（调优不同 GPU 架构时修改）
    int   cudaBlockW           = 16;     ///< 2D kernel 线程块宽度（必须为 2 的幂）
    int   cudaBlockH           = 16;     ///< 2D kernel 线程块高度（必须为 2 的幂）
    int   cudaBlockSweep       = 32;     ///< 1D sweep kernel 线程块大小（必须为 2 的幂）

    bool  epipolarRectified    = false;  ///< 图像已极线校正，偏向水平传播
};

// =============================================================================
// 融合配置
// =============================================================================
struct FusionConfig
{
    int   minConsistentViews = 2;      ///< 至少有多少视图一致才保留像素
    float relDepthThresh     = 0.05f;  ///< 深度相对误差阈值（收紧→更严格一致性）
    float pixelThresh        = 2.0f;   ///< 投影误差阈值（像素），收紧→减少噎点
    float confidenceThresh   = 0.25f;  ///< 融合前二次置信度过滤阈值（与PatchMatch 0.30对应）
    bool  doSigmaFusion      = true;   ///< 是否做 sigma 加权深度融合
    float sigmaMultiplier    = 2.0f;   ///< sigma 乘数放宽→少剔除内点
    bool  doInpaint          = true;   ///< 对小洞做 inpaint（填补）
    float inpaintRadiusFactor= 3.0f;
    int   inpaintRadius      = 5;      ///< inpaint 搜索半径（像素），加大→填补更多小洞
};

// =============================================================================
// 融合输入帧
// =============================================================================
struct FusionFrameInput
{
    cv::Mat   depthMap;    ///< CV_32F, 正深度（COLMAP 约定），0=无效
    cv::Mat   normalMap;   ///< CV_32FC3, 法线（相机坐标系），可为空
    cv::Mat   confidence;  ///< CV_32F, [0,1]，可为空
    PositiveDepthCameraModel cameraModel;  ///< 正深度相机模型
    int       imgW = 0;
    int       imgH = 0;
    std::string imagePath; ///< 原始彩色图像路径（用于取色）
};

// =============================================================================
// 稠密点（含颜色）
// =============================================================================
struct DensePoint
{
    float   x = 0.f, y = 0.f, z = 0.f;
    uint8_t r = 0,   g = 0,   b = 0;
};

// =============================================================================
// 深度图生成配置
// =============================================================================
struct DepthGenConfig
{
    PatchMatchConfig patchMatch;
    FusionConfig     fusion;
    int   numSourceViews        = 4;
    int   cpuWorkerCount        = 1;     ///< CPU 路径的像素级并行线程数
    float zNearScale            = 0.75f;  ///< IQR 内 2%ile × 0.75（原0.5太宽致PatchMatch难收敛）
    float zFarScale             = 1.5f;   ///< IQR 内 98%ile × 1.5（原2.0产生6x搜索比，现3x更合理）
    bool  runDepthEstimation    = true;
    bool  runFusion             = true;
    bool  saveIntermediateDepthMaps = false;
    std::string intermediateDir = "";
};

// =============================================================================
// 输入相机视图（图像元数据 + 唯一相机表达）
// =============================================================================
struct CameraView
{
    std::string imagePath;
    int         imageWidth  = 0;
    int         imageHeight = 0;
    xjw::Camera camera;

    PositiveDepthCameraModel positiveDepthModel() const
    {
        return camera.toPositiveDepthModel();
    }
};

// =============================================================================
// 稀疏点云预处理结果（供深度范围估计）
// =============================================================================
struct SparseCloud
{
    std::vector<std::array<float,3>> points;
    std::array<float,3> minPt = {}, maxPt = {};
};

// =============================================================================
// 进度回调
// =============================================================================
using MvsProgressCallback = std::function<void(const std::string &, float)>;

} // namespace mvs
} // namespace xjw

// =============================================================================
// 文件: PatchMatchCUDA.h
// 模块: MVS - PatchMatch 深度图估计
// 说明:
//   基于 COLMAP 同源方法（warpPixel homography）的 GPU/CPU PatchMatch。
//
//   NCC 计算方法（COLMAP 风格）：
//     给定参考帧像素 (u_r, v_r) 和深度假设 d，将其映射到源帧：
//       xn = (u_r - cx_r)/fx_r,  yn = (v_r - cy_r)/fy_r
//       X = R_rel[0..2] * [xn,yn,1] + T_rel[0] / d
//       Y = R_rel[3..5] * [xn,yn,1] + T_rel[1] / d
//       Z = R_rel[6..8] * [xn,yn,1] + T_rel[2] / d
//       u_s = fx_s * X/Z + cx_s
//     其中 R_rel = R_cw_src * R_cw_ref^T, T_rel = T_src - R_rel * T_ref
//
//   源相机数据布局（每个源帧 16 个 float）：
//     [0..3]  = {fx_s, cx_s, fy_s, cy_s}
//     [4..12] = R_rel[9] (行优先)
//     [13..15]= T_rel[3]
// =============================================================================
#pragma once

#include "MvsTypes.h"
#include <vector>
#include <string>

namespace xjw 
{
namespace mvs 
{

class PatchMatchDepthEstimator
{
public:
    static bool isCudaAvailable();

    /// 释放全局 GPU 图像缓存占用的所有显存。
    /// 在程序退出前或不再需要 MVS 功能时调用，防止显存泄漏。
    static void cleanupGpuImageCache();

    // ── 主接口 ──────────────────────────────────────────────────────────────
    // refGray      参考帧灰度图 (CV_8U)
    // srcGrays     源帧灰度图列表（与 srcCams 等长）
    // refCam       参考相机（COLMAP 约定，Z_cam>0）
    // srcCams      源相机列表（COLMAP 约定）
    // zNear/zFar   深度搜索范围（正值）
    // config       配置
    // depthOut     [out] 深度图 (CV_32F, 0=无效)
    // confOut      [out, 可选] 置信度图 (CV_32F, 0~1)
    // errorMsg     [out, 可选] 错误信息
    // hintDepth    [可选] 逐像素提示深度图 (CV_32F, 0=无)
    // hintRadius   [可选] 逐像素搜索半径 (CV_32F, <=0=自动半径)
    // refValidMask [可选] 参考帧可信前景 (CV_8U, 非零=有效)
    // srcValidMasks[可选] 源帧可信前景，与 srcGrays 等长；空 Mat=全图有效
    static bool estimate(
        const cv::Mat                          &refGray,
        const std::vector<cv::Mat>             &srcGrays,
        const Camera                           &refCam,
        const std::vector<Camera>              &srcCams,
        float                                   zNear,
        float                                   zFar,
        const PatchMatchConfig                 &config,
        cv::Mat                                &depthOut,
        cv::Mat                                *confOut    = nullptr,
        std::string                            *errorMsg   = nullptr,
        const cv::Mat                          *hintDepth  = nullptr,
        const cv::Mat                          *hintRadius = nullptr,
        const cv::Mat                          *refValidMask = nullptr,
        const std::vector<cv::Mat>             *srcValidMasks = nullptr);

private:
    static bool estimateGPU(
        const cv::Mat                          &refGray,
        const std::vector<cv::Mat>             &srcGrays,
        const Camera                           &refCam,
        const std::vector<Camera>              &srcCams,
        float zNear, float zFar,
        const PatchMatchConfig       &config,
        cv::Mat &depthOut, cv::Mat *confOut,
        std::string *errorMsg,
        const cv::Mat *hintDepth,
        const cv::Mat *hintRadius,
        const cv::Mat *refValidMask,
        const std::vector<cv::Mat> *srcValidMasks);

    static bool estimateCPU(
        const cv::Mat                          &refGray,
        const std::vector<cv::Mat>             &srcGrays,
        const Camera                           &refCam,
        const std::vector<Camera>              &srcCams,
        float zNear, float zFar,
        const PatchMatchConfig       &config,
        cv::Mat &depthOut, cv::Mat *confOut,
        std::string *errorMsg,
        const cv::Mat *hintDepth,
        const cv::Mat *hintRadius,
        const cv::Mat *refValidMask,
        const std::vector<cv::Mat> *srcValidMasks);
};

} // namespace mvs
} // namespace xjw

#pragma once

// ============================================================
// 文件：PnpSolver.h
// 功能：PnP（Perspective-n-Point）绝对位姿估计。
//
// 给定一组 3D-2D 对应关系（已知三维点 + 图像像素坐标），
// 使用 OpenCV 的 solvePnPRansac 估计相机的绝对位姿（R, t）。
//
// 输出采用 PlaScan 约定：
//   - R 为 camera-to-world 旋转矩阵（与 Camera.h 一致）
//   - C 为相机中心在世界坐标系中的位置
//
// 参考：COLMAP 的 absolute_pose.h，简化适配。
// ============================================================

#include "Camera.h"

#include <array>
#include <vector>

namespace xjw
{

/**
 * @brief PnP 求解选项。
 */
struct PnpOptions
{
    /// RANSAC 最大重投影误差阈值（像素）
    double maxReprojError = 4.0;

    /// RANSAC 最大迭代次数
    int maxIterations = 1000;

    /// 最少所需内点数
    int minNumInliers = 10;

    /// 最小内点比例
    double minInlierRatio = 0.25;

    /// 是否允许低于 minInlierRatio 但绝对内点数足够的宽松通过。
    /// 无相机增量 SfM 默认应关闭，避免弱约束位姿过早进入模型。
    bool allowRelaxedInlierRatio = false;

    /// 宽松通过时仍要求的最低内点率。
    /// 仅当 allowRelaxedInlierRatio=true 时生效，默认值保持相对保守。
    double relaxedMinInlierRatio = 0.10;

    /// 宽松通过时仍要求的最低绝对内点数；<=0 时使用 max(minNumInliers, 15)。
    int relaxedMinNumInliers = 0;

    /// 是否向 solvePnPRansac 提供外参初值。
    bool useInitialPose = false;

    /// 初始 camera-to-world 旋转矩阵（行优先 3×3）。
    std::array<double, 9> initialCameraToWorldRotation{{1.0, 0.0, 0.0,
                                                        0.0, 1.0, 0.0,
                                                        0.0, 0.0, 1.0}};

    /// 初始相机中心。
    std::array<double, 3> initialCameraCenter{{0.0, 0.0, 0.0}};

    /// 是否先用初始位姿重投影门控 2D-3D 对应，再执行 PnP RANSAC。
    bool useInitialPosePrefilter = false;

    /// 初始位姿对应门控的最大重投影误差（像素）。
    double initialPosePrefilterMaxReprojError = 48.0;

    /// 至少保留多少候选才采用初始位姿门控；不足时回退到全量 RANSAC。
    int initialPosePrefilterMinCandidates = 20;

    /// RANSAC 置信度
    double confidence = 0.999;

    /// OpenCV RANSAC 稳定种子。调用方应根据影像 ID 和当前注册阶段派生该值。
    int ransacSeed = 0;
};

/**
 * @brief PnP 求解结果。
 */
struct PnpResult
{
    bool success = false; ///< 是否成功求解

    /// camera-to-world 旋转矩阵（行优先 3×3），与 `Camera::cameraToWorldRotation()` 一致
    std::array<double, 9> R{{1, 0, 0, 0, 1, 0, 0, 0, 1}};

    /// 相机中心在世界坐标系中的位置，与 `Camera::cameraCenter()` 一致
    std::array<double, 3> C{{0, 0, 0}};

    int numInliers = 0;     ///< RANSAC 内点数
    double inlierRatio = 0; ///< 内点比例
    int inputCandidateCount = 0; ///< 输入 PnP 的原始 2D-3D 对应数
    int prefilterCandidateCount = 0; ///< 初始位姿门控后保留的对应数
    bool usedInitialPosePrefilter = false; ///< 本次求解是否实际采用了初始位姿门控

    /// 内点掩码（与输入点对等长，1=内点，0=外点）
    std::vector<unsigned char> inlierMask;
};

/**
 * @brief PnP 绝对位姿估计器（静态接口）。
 *
 * 核心方法 solve() 封装了 OpenCV 的 cv::solvePnPRansac，
 * 并将输出转换为 PlaScan 的 camera-to-world 约定。
 */
class PnpSolver
{
  public:
    /**
     * @brief 从 3D-2D 对应关系估计相机绝对位姿。
     *
     * @param worldPoints   三维点坐标列表 [X, Y, Z]
     * @param imagePoints   对应的图像像素坐标列表 [u, v]
     * @param fu            x 方向焦距（像素）
     * @param fv            y 方向焦距（像素）
     * @param cu            主点 u 坐标（像素）
     * @param cv            主点 v 坐标（像素）
     * @param options       PnP 求解选项
     * @return PnpResult    求解结果
     *
     * @note worldPoints 和 imagePoints 必须等长，且长度 >= 4。
     */
    static PnpResult solve(const std::vector<std::array<double, 3>> &worldPoints,
                           const std::vector<std::array<double, 2>> &imagePoints, double fu, double fv, double cu,
                           double cv, int uDir = 1, int vDir = 1, bool depthFlipped = false,
                           const PnpOptions &options = PnpOptions());

    /**
     * @brief 使用已有 Camera 内参从 3D-2D 对应关系估计绝对位姿。
     *
     * @param worldPoints   三维点坐标列表
     * @param imagePoints   对应的图像像素坐标列表
     * @param cam           提供内参的相机（仅使用 fu/fv/cu/cv）
     * @param options       PnP 求解选项
     * @return PnpResult    求解结果
     */
    static PnpResult solveWithCamera(const std::vector<std::array<double, 3>> &worldPoints,
                                     const std::vector<std::array<double, 2>> &imagePoints, const Camera &cam,
                                     const PnpOptions &options = PnpOptions());
};

} // namespace xjw

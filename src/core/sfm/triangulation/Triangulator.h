#pragma once

// ============================================================
// 文件：Triangulator.h
// 功能：SfM 流水线内的多视图三角化器。
//
// 职责：
//   1. 对新注册图像的已匹配特征执行三角化，创建新三维点
//   2. 将新观测追加到已有三维点（延续轨迹）
//   3. 过滤质量不佳的三维点
//
// 复用 Intersection::intersectPair 进行双目三角化。
// 参考：COLMAP 的 IncrementalTriangulator，简化适配。
// ============================================================

#include "common/SfmTypes.h"
#include "graph/CorrespondenceGraph.h"
#include "reconstruction/SfmReconstruction.h"

#include "Camera.h"
#include "Intersection.h"

#include <vector>

namespace xjw
{

/**
 * @brief 三角化选项。
 */
struct TriangulatorOptions
{
    /// 创建新三维点时的最小三角化角（度）
    double minTriAngle = 2.0;

    /// 创建新三维点时的最大重投影误差（像素）
    double maxReprojError = 2.5;

    /// 延续已有轨迹时的最大重投影误差（像素）
    double continueMaxReprojError = 2.5;

    /// 完成/合并轨迹时的最大重投影误差（像素）
    double completeMaxReprojError = 2.5;
};

/**
 * @brief 三角化统计。
 */
struct TriangulationStats
{
    int numCreated = 0;   ///< 新创建的三维点数
    int numContinued = 0; ///< 延续现有轨迹的观测数
    int numFiltered = 0;  ///< 被过滤的三维点数
};

/**
 * @brief SfM 三角化器。
 *
 * 持有对 SfmReconstruction 和 CorrespondenceGraph 的引用，
 * 负责在增量注册循环中执行三角化操作。
 */
class Triangulator
{
  public:
    /**
     * @brief 构造三角化器。
     * @param reconstruction  重建容器引用
     * @param graph           对应关系图引用
     */
    Triangulator(SfmReconstruction &reconstruction, const CorrespondenceGraph &graph);

    /**
     * @brief 对新注册的图像执行三角化。
     *
     * 对该图像的所有特征，查找其在已注册图像中的对应点：
     *   - 如果对应点已关联三维点 → 延续轨迹（将当前特征追加到轨迹）
     *   - 如果对应点未关联三维点 → 尝试三角化创建新三维点
     *
     * @param imageId  刚注册的图像 ID
     * @param options  三角化选项
     * @return 三角化统计
     */
    TriangulationStats triangulateImage(ImageId imageId, const TriangulatorOptions &options = TriangulatorOptions());

    /**
     * @brief 过滤重投影误差过大的三维点。
     *
     * 遍历所有三维点，重新计算重投影误差，
     * 删除超过阈值或三角化角过小的点。
     *
     * @param maxReprojError  最大允许重投影误差（像素）
     * @param minTriAngle     最小允许三角化角（度）
     * @return 被删除的三维点数量
     */
    int filterPoints(double maxReprojError = 2.0, double minTriAngle = 2.0);

    /**
     * @brief 过滤轨迹长度过短的三维点。
     *
     * 仅被少量图像观测到的三维点可靠性较差，容易成为离群点。
     *
     * @param minTrackLen  最小轨迹长度，观测数 < 此值的点将被删除
     * @return 被删除的三维点数量
     */
    int filterShortTracks(int minTrackLen = 2);

    /**
     * @brief 尝试将未关联的观测补全到已有三维点。
     *
     * 对已注册图像中尚未关联三维点的特征，检查对应关系中
     * 是否有其他图像的对应特征已关联三维点。若有，且重投影
     * 误差在阈值内且深度为正，则追加到轨迹。
     *
     * @param options  三角化选项
     * @return 新追加的观测数
     */
    int completeTracks(const TriangulatorOptions &options = TriangulatorOptions());

    /**
     * @brief 利用当前相机位姿重新三角化所有三维点。
     *
     * BA 优化相机位姿后，原来的 3D 点坐标基于旧位姿，精度较差。
     * 此方法对每个三维点的所有观测执行多视图 DLT 三角化，
     * 更新坐标并重算重投影误差。如果重三角化结果更差则保留原值。
     *
     * 参考 COLMAP 的 Retriangulate 策略。
     *
     * @param maxReprojError  重三角化后允许的最大重投影误差
     * @return 成功重三角化的点数
     */
    int retriangulatePoints(double maxReprojError = 2.0);

    /**
     * @brief 用当前相机位姿重新计算所有三维点的重投影误差。
     *
     * 在 BA 或重三角化后调用，确保 ScenePoint3D::error 字段
     * 反映最新的相机位姿，为后续过滤提供准确依据。
     */
    void recomputeReprojErrors();

  private:
    SfmReconstruction &_reconstruction;
    const CorrespondenceGraph &_correspondenceGraph;

    /**
     * @brief 尝试对两个特征执行双目三角化。
     *
     * @param imgId1   图像 1 ID
     * @param featIdx1 图像 1 中特征索引
     * @param imgId2   图像 2 ID
     * @param featIdx2 图像 2 中特征索引
     * @param options  三角化选项
     * @param outXyz   输出：三角化得到的三维坐标
     * @return 三角化成功返回 true
     */
    bool triangulatePair(ImageId imgId1, FeatureIdx featIdx1, ImageId imgId2, FeatureIdx featIdx2,
                         const TriangulatorOptions &options, std::array<double, 3> &outXyz);

    /**
     * @brief 计算三维点在某相机中的重投影误差。
     * @return 重投影误差（像素），失败返回极大值
     */
    double computeReprojError(const std::array<double, 3> &xyz, ImageId imageId, FeatureIdx featureIdx) const;

    /**
     * @brief 检查三维点在指定相机中的深度是否为正。
     * @return 深度为正返回 true
     */
    bool hasPositiveDepth(const std::array<double, 3> &xyz, ImageId imageId) const;

    /**
     * @brief 多视图 DLT 三角化：利用 >= 2 个观测求解最优三维点。
     * @param observations  (imageId, featureIdx) 列表
     * @param outXyz        输出三维坐标
     * @return 成功返回 true
     */
    bool triangulateMultiView(const std::vector<TrackElement> &observations, std::array<double, 3> &outXyz) const;
};

} // namespace xjw

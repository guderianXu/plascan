#pragma once

// ============================================================
// 文件：IncrementalSfm.h
// 功能：增量式运动恢复结构（Incremental SfM）主控制器。
//
// 算法流程（参考 COLMAP IncrementalMapper，简化适配 PlaScan）：
//   1. 构建对应关系图
//   2. 选择初始像对，利用相对定向恢复 R/t，三角化种子点
//   3. 逐帧注册循环：
//      a) 选择下一幅最佳图像（按可见三维点数排序）
//      b) PnP 绝对定向
//      c) 三角化新可见特征
//      d) 局部/全局光束法平差
//      e) 过滤低质量三维点和坏帧
//   4. 输出最终重建结果
//
// 依赖模块：Camera, Intersection,
//           BundleAdjust, PnpSolver, Triangulator
// ============================================================

#include "common/SfmTypes.h"
#include "graph/CorrespondenceGraph.h"
#include "pose/PnpSolver.h"
#include "reconstruction/SfmReconstruction.h"
#include "triangulation/Triangulator.h"

#include "BundleAdjust.h"
#include "Camera.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace xjw
{

// ---- 选项 ----

/**
 * @brief 增量 SfM 全局选项。
 */
struct IncrementalSfmOptions
{
    // --- 初始化选项 ---
    /// 初始像对最少匹配数
    int initMinNumMatches = 100;
    /// 初始像对相对定向最少内点数
    int initMinNumInliers = 50;
    /// 初始像对 chirality 检查最少通过数（可独立于 initMinNumInliers，软阈值）
    int initMinChiralityInliers = 10;
    /// 初始像对最小三角化角（度）
    double initMinTriAngle = 8.0;
    /// 自动选择初始像对（true）还是使用指定的 initImageId1/2
    bool autoSelectInitPair = true;
    ImageId initImageId1 = kInvalidImageId;
    ImageId initImageId2 = kInvalidImageId;
    /// 使用输入相机中的已知外参，跳过初始像对相对定向和 PnP 注册
    bool useKnownCameraPoses = false;
    /// 自动选择初始像对时的最大候选对数量（参考 COLMAP 多候选重试策略）
    int maxInitPairCandidates = 10;

    // --- 注册选项 ---
    PnpOptions pnpOptions;

    // --- 三角化选项 ---
    TriangulatorOptions triangulatorOptions;

    // --- 光束法平差选项 ---
    /// 每注册多少幅图像执行一次局部 BA
    int localBAInterval = 3;
    /// 局部 BA 包含的邻域图像数量
    int localBANumImages = 6;
    /// 全局 BA 执行间隔（注册图像数）
    int globalBAInterval = 10;
    /// BA 选项
    BAOptions baOptions;

    // --- 迭代 BA + 过滤选项（参考 COLMAP IterativeGlobalRefinement）---
    /// 全局 BA 迭代精化轮数：每轮 = BA + 过滤，直到变化率 < 阈值或达到上限
    int iterativeBARounds = 3;
    /// 在全局 BA 前过滤负深度点（在所有已注册相机前方检查）
    bool filterNegativeDepth = true;

    // --- 过滤选项 ---
    /// 过滤三维点的最大重投影误差（像素）
    double filterMaxReprojError = 2.0;
    /// 过滤三维点的最小三角化角（度）
    double filterMinTriAngle = 2.0;
    /// 过滤三维点的最小轨迹长度（观测次数 < 此值的点将被剔除）
    int filterMinTrackLen = 2;
};

// ---- 回调 ----

/**
 * @brief SfM 进度回调函数类型。
 *
 * @param numRegistered  已注册图像数
 * @param numTotal       总图像数
 * @param message        进度消息
 * @return false 表示请求中止
 */
using SfmProgressCallback = std::function<bool(int numRegistered, int numTotal, const std::string &message)>;

// ---- 结果 ----

/**
 * @brief 增量 SfM 的最终结果。
 */
struct IncrementalSfmResult
{
    bool success = false;         ///< 是否成功完成重建
    int numRegisteredImages = 0;  ///< 已注册图像数
    int numPoints3D = 0;          ///< 三维点数
    double meanReprojError = 0.0; ///< 平均重投影误差（像素，最终 BA 后）
    std::string summary;          ///< 可读结果摘要

    // ── 光束法平差统计（最终一轮全局 BA 的结果）──
    double baRmsBefore = 0.0;  ///< 最终 BA 前的平均重投影 RMS（px）
    double baRmsAfter = 0.0;   ///< 最终 BA 后的平均重投影 RMS（px）
    int baTracksTotal = 0;     ///< 最终全局 BA 参与轨迹总数
    int baTracksOptimized = 0; ///< 最终全局 BA 成功优化轨迹数
    int baTracksFiltered = 0;  ///< 最终全局 BA 过滤的离群点数

    /// 重建容器的指针（调用方获得所有权）
    std::shared_ptr<SfmReconstruction> reconstruction;
};

// ---- 主控制器 ----

/**
 * @brief 增量式 SfM 主控制器。
 *
 * 典型使用流程：
 * @code
 *   IncrementalSfm sfm(options);
 *
 *   // 添加图像和特征
 *   sfm.addImage(id1, path1, camPath1, keypoints1);
 *   sfm.addImage(id2, path2, camPath2, keypoints2);
 *   ...
 *
 *   // 添加匹配
 *   sfm.addMatches(id1, id2, matches12);
 *   ...
 *
 *   // 执行重建
 *   auto result = sfm.run(progressCallback);
 * @endcode
 */
class IncrementalSfm
{
  public:
    /**
     * @brief 构造增量 SfM 控制器。
     * @param options  全局选项
     */
    explicit IncrementalSfm(const IncrementalSfmOptions &options = IncrementalSfmOptions());

    // ---- 数据输入 ----

    /**
     * @brief 添加一幅图像。
     * @param id         图像唯一 ID
     * @param imagePath  图像文件路径
     * @param cameraPath 相机参数文件路径（.tsai），可为空
     * @param keypoints  该图像的特征点列表
     */
    void addImage(ImageId id, const std::string &imagePath, const std::string &cameraPath,
                  const std::vector<FeatureKeypoint> &keypoints);

    /**
     * @brief 添加一幅图像，并提供预设的相机内参（无需相机文件）。
     *
     * 当没有 .tsai 相机文件时（如纯 SFM 场景），调用方可根据
     * 影像尺寸估算内参（焦距≈max(w,h)*1.2，主点≈图像中心），
     * 通过此接口传入。外参（R、C）留为默认值即可，SFM 会恢复。
     *
     * @param id         图像唯一 ID
     * @param imagePath  图像文件路径
     * @param camera     预设相机对象（至少包含 fu/fv/cu/cv 内参）
     * @param keypoints  该图像的特征点列表
     */
    void addImageWithCamera(ImageId id, const std::string &imagePath, const Camera &camera,
                            const std::vector<FeatureKeypoint> &keypoints);

    /**
     * @brief 添加两幅图像之间的匹配。
     * @param id1      图像 1 ID
     * @param id2      图像 2 ID
     * @param matches  匹配列表
     */
    void addMatches(ImageId id1, ImageId id2, const std::vector<FeatureMatch> &matches);

    // ---- 执行 ----

    /**
     * @brief 执行增量式 SfM 重建。
     *
     * 完整流程：
     *   1. 构建对应关系图
     *   2. 选择并初始化初始像对
     *   3. 逐帧注册循环（PnP → 三角化 → BA → 过滤）
     *   4. 最终全局 BA 和过滤
     *   5. 返回重建结果
     *
     * @param progressCb  可选的进度回调，返回 false 中止
     * @return IncrementalSfmResult 重建结果
     */
    IncrementalSfmResult run(SfmProgressCallback progressCb = nullptr);

    // ---- 查询 ----

    /// 获取当前重建容器（只读）
    const SfmReconstruction *reconstruction() const
    {
        return _reconstruction.get();
    }

  private:
    IncrementalSfmOptions _sfmOptions;

    /// 对应关系图
    CorrespondenceGraph _correspondenceGraph;

    /// 重建容器
    std::shared_ptr<SfmReconstruction> _reconstruction;

    /// 输入的相机文件路径（imageId → cameraPath）
    std::unordered_map<ImageId, std::string> _cameraPaths;

    /// 预设的相机对象（imageId → Camera），由 addImageWithCamera 填充
    std::unordered_map<ImageId, Camera> _preloadedCameras;

    /// 最近一次内部操作的错误描述（供 run() 写入 result.summary）
    std::string _lastErrorMessage;

    /// 中止标志
    bool _isAborted = false;

    // ── 最后一次全局 BA 的统计（填入 IncrementalSfmResult）──
    double _lastGlobalBARmsBefore = 0.0;
    double _lastGlobalBARmsAfter = 0.0;
    int _lastGlobalBATracksTotal = 0;
    int _lastGlobalBATracksOptimized = 0;
    int _lastGlobalBATracksFiltered = 0;

    /// 增量维护的可见三维点计数缓存（imageId → 可见已三角化点数）
    std::unordered_map<ImageId, size_t> _visibilityCache;
    /// 可见计数缓存是否有效
    bool _visibilityCacheDirty = true;

    /// 图像注册失败次数（imageId → 连续失败次数）
    std::unordered_map<ImageId, int> _registerFailCount;
    /// 永久失败图像集合（超过最大重试次数后移入）
    std::unordered_set<ImageId> _permanentlyFailedImages;

    // ---- 内部流程 ----

    /**
     * @brief 获取某张影像的相机对象。
     *
     * 优先使用预设相机（addImageWithCamera），
     * 其次从 _cameraPaths 中的 .tsai 文件加载。
     *
     * @param imageId  图像 ID
     * @param cam      输出相机对象
     * @return 成功返回 true
     */
    bool getCamera(ImageId imageId, Camera &cam) const;

    std::vector<BACameraPosePrior> buildCameraPosePriorsFromInputCameras(
        const std::vector<ImageId> &imageIds) const;

    void alignReconstructionToKnownPosePriors(const std::vector<ImageId> &imageIds,
                                              std::vector<Camera> *baCameras);

    void refineKnownCameraPosesWithPnp();

    /**
     * @brief 使用输入相机外参执行固定相机位姿三角化。
     *
     * 适用于 `.tsai` 等外部相机文件已经提供可靠位姿的场景。
     * 此路径不会重新估计相机位姿，只注册所有输入相机并基于匹配生成稀疏点。
     */
    IncrementalSfmResult runKnownCameraPoseReconstruction(SfmProgressCallback progressCb);

    /**
     * @brief 从 .tsai 文件加载相机参数。
     * @param cameraPath  .tsai 文件路径
     * @param cam         输出相机对象
     * @return 成功返回 true
     */
    bool loadCamera(const std::string &cameraPath, Camera &cam) const;

    /**
     * @brief 返回多个初始像对候选（按匹配数降序排序）。
     *
     * 参考 COLMAP 的多候选重试策略，在 run() 中逐个尝试直到成功。
     *
     * @param maxCandidates  最大候选数量
     * @return 候选像对列表（按匹配数降序排序）
     */
    std::vector<std::pair<ImageId, ImageId>> selectInitialPairCandidates(int maxCandidates) const;

    /**
     * @brief 初始化初始像对。
     *
     * 执行相对定向 → 注册两幅图像 → 三角化初始点云。
     * 参考 COLMAP：同时尝试 E（Essential）和 H（Homography）路径，
     * 选择 chirality 内点最多的分解结果。
     *
     * @param id1  图像 1 ID
     * @param id2  图像 2 ID
     * @return 成功返回 true
     */
    bool initializeFromPair(ImageId id1, ImageId id2);

    /**
     * @brief 选择下一幅最佳待注册图像。
     *
     * 策略：在所有未注册图像中，选择能看到最多已三角化三维点的图像。
     *
     * @return 最佳图像 ID，无可注册图像则返回 kInvalidImageId
     */
    ImageId selectNextImage() const;

    /**
     * @brief 注册一幅图像到重建中。
     *
     * 收集该图像可见的 3D-2D 对应，调用 PnP 求解绝对位姿。
     *
     * @param imageId  待注册图像 ID
     * @return 成功返回 true
     */
    bool registerImage(ImageId imageId);

    /**
     * @brief 执行光束法平差。
     *
     * 收集重建中的所有相机和轨迹，调用 BundleAdjust::optimizePoints。
     * 将优化结果回写到重建容器。
     *
     * @param localOnly  是否仅局部 BA（仅优化最近注册图像的邻域）
     * @param anchorIds  局部 BA 时锚定的图像 ID
     */
    void runBundleAdjust(bool localOnly = false, const std::vector<ImageId> &anchorIds = {});

    /**
     * @brief 迭代全局 BA 精化（参考 COLMAP IterativeGlobalRefinement）。
     *
     * 循环执行：全局 BA → 过滤 → 补三角化，直到变化率 < 阈值或达到上限。
     * 每轮中先过滤负深度点（可选），再执行 BA + 点过滤。
     */
    void iterativeGlobalBA();

    /**
     * @brief 过滤负深度三维点（参考 COLMAP FilterObservationsWithNegativeDepth）。
     *
     * 遍历所有三维点，检查其在每个观测相机中的深度。
     * 若在某台相机前方深度 < 0，则移除该观测；
     * 若移除后轨迹长度 < 2，则删除整个三维点。
     *
     * @return 删除的三维点数量
     */
    int filterNegativeDepthPoints();

    /**
     * @brief 报告进度。
     * @return false 如果回调要求中止
     */
    bool reportProgress(int numRegistered, int numTotal, const std::string &msg, SfmProgressCallback &cb);

    /**
     * @brief 重建可见性缓存（在缓存失效时调用）。
     */
    void rebuildVisibilityCache();

    /**
     * @brief 标记可见性缓存需要重建（在 BA/过滤后调用）。
     */
    void invalidateVisibilityCache();

    /**
     * @brief 基于新建立的 3D 关联增量更新可见性缓存。
     *
     * @param imageId         刚完成三角化的已注册图像
     * @param previousPointIds 三角化前该图像的 point3DIds 快照
     */
    void updateVisibilityCacheForImage(ImageId imageId, const std::vector<Point3DId> &previousPointIds);
};

} // namespace xjw

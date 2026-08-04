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
#include "registration/ControlNetworkSolver.h"
#include "triangulation/Triangulator.h"
#include "registration/PriorTrack.h"

#include "BundleAdjust.h"
#include "Camera.h"

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace xjw
{

class ImageRegistrationEngine;
class InitialPairInitializer;
class KnownPoseReconstructor;
class SfmBundleAdjustCoordinator;

// ---- 选项 ----

enum class SfmExecutionProfile
{
    FullRefinement,
    CoarseEvaluation
};

/**
 * @brief 增量 SfM 全局选项。
 */
struct IncrementalSfmOptions
{
    /// 正式精化或候选粗筛。粗筛会统一收紧高成本 BA 参数。
    SfmExecutionProfile executionProfile = SfmExecutionProfile::FullRefinement;
    /// 候选探测最多注册的影像数；0 表示注册全部影像。正式重建必须保持为 0。
    int maxRegisteredImages = 0;

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
    /// 已知外参作为 soft prior 时仍允许 BA 微调相机位姿。
    bool refineKnownCameraPoseWithSoftPrior = true;
    /// 航测弱几何第一轮 BA 不释放内参/畸变，避免弱基线下内参吸收几何误差。
    bool keepIntrinsicsFixedInKnownPoseBa = true;
    /// 根据相机中心范围自适应位置先验 sigma 的倍率。
    double knownPosePriorPositionSigmaScale = 1.0;
    /// 已知外参 soft prior 的旋转 sigma（deg）。
    double knownPosePriorRotationSigmaDegrees = 2.0;
    /// 自动选择初始像对时的最大候选对数量（参考 COLMAP 多候选重试策略）
    int maxInitPairCandidates = 10;
    /// 小型无相机数据集是否完整评估多个初始像对模型，避免第一个可初始化 seed 困在局部子图。
    bool evaluateMultipleInitialPairModels = true;
    /// 多初始模型评估只在图像数不超过该阈值时启用，避免大工程重复跑完整 SfM。
    int multiInitialPairMaxImages = 32;

    // --- 注册选项 ---
    PnpOptions pnpOptions;
    /// PnP 注册新影像时，3D 点至少需要的轨迹长度。无相机序列建议使用 3，
    /// 避免大量两视点把新相机吸附到错误环段；注册影像少于 3 张时内部仍允许两视点启动模型。
    int pnpMinTrackLength = 2;
    /// 是否利用序号相邻相机插值/外推 PnP 初值并放宽有强绝对支撑的序列缺口。
    /// 与严格距离门控分离，避免必须启用等距轨迹假设才能使用序列恢复。
    bool useSequencePoseRecovery = false;
    /// 照片序列模式下，PnP 后检查相邻序号相机中心距离，避免闭环序列被错误跨接。
    bool enforceSequencePoseConsistency = false;
    /// 序列首尾是否按闭环相邻处理。
    bool sequenceLoopClosure = true;
    /// 估计相邻距离中位数至少需要的已注册相邻边数量。
    int sequencePoseConsistencyMinSamples = 3;
    /// 新相机到已注册相邻序号相机的距离不能低于中位相邻距离的该倍率。
    double sequenceAdjacentDistanceMinFactor = 0.15;
    /// 新相机到已注册相邻序号相机的距离不能高于中位相邻距离的该倍率。
    double sequenceAdjacentDistanceMaxFactor = 3.0;
    /// 当前后紧邻序列相机均已注册时，允许以强绝对内点支撑放宽 PnP 内点率。
    bool allowBracketedSequencePnpRelaxation = true;
    /// 仅有一侧序列邻居时，使用相邻位姿外推、初值预过滤和绝对内点门槛恢复连续缺口。
    bool allowOneSidedSequencePoseRecovery = true;
    /// 单侧外推旋转误差通常大于夹逼插值，预过滤采用更宽的像素门槛。
    double oneSidedSequencePosePrefilterMaxReprojError = 128.0;
    /// 单侧外推只提供较弱的序列先验，使用独立的绝对内点门槛。
    /// 不能沿用夹逼模式的 28 点门槛，否则长序列在局部纹理较弱处会永久断链。
    int oneSidedSequencePnpMinInliers = 16;
    /// 单侧外推 PnP 的最低内点率；仍同时要求上述绝对内点数。
    double oneSidedSequencePnpMinInlierRatio = 0.05;
    /// 夹逼式序列 PnP 的最低内点率；仍需同时满足绝对内点数和序列距离检查。
    double bracketedSequencePnpMinInlierRatio = 0.025;
    /// 夹逼式序列 PnP 的最低绝对内点数。
    int bracketedSequencePnpMinInliers = 28;
    /// 最终全局 BA 收敛后，是否用更新后的三维点和相机位姿重试未注册影像。
    bool retryUnregisteredAfterFinalBA = true;
    /// 最终 BA 后最多执行的缺口注册轮数。
    int maxFinalRegistrationRetryPasses = 2;
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

    // --- 输入多视轨迹质量管理 ---
    /// 每张影像最多保留的 tie points，<=0 表示不限制。
    int maxTracksPerImage = 6000;
    /// 每张影像每个网格最多保留的 tie points，<=0 表示不限制。
    int maxTracksPerGridCell = 300;
    /// tie point 空间均匀化网格列数。
    int trackThinningGridColumns = 8;
    /// tie point 空间均匀化网格行数。
    int trackThinningGridRows = 8;

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

/**
 * @brief 根据执行模式生成实际生效的 SfM 参数。
 */
IncrementalSfmOptions effectiveSfmOptions(const IncrementalSfmOptions &options);

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
    ImageId selectedInitialImageId1 = kInvalidImageId; ///< 最终采用的初始像对第一幅影像
    ImageId selectedInitialImageId2 = kInvalidImageId; ///< 最终采用的初始像对第二幅影像

    // ── 光束法平差统计（最终一轮全局 BA 的结果）──
    double baRmsBefore = 0.0;  ///< 最终 BA 前的平均重投影 RMS（px）
    double baRmsAfter = 0.0;   ///< 最终 BA 后的平均重投影 RMS（px）
    int baTracksTotal = 0;     ///< 最终全局 BA 参与轨迹总数
    int baTracksOptimized = 0; ///< 最终全局 BA 成功优化轨迹数
    int baTracksFiltered = 0;  ///< 最终全局 BA 过滤的离群点数
    int baRefinedIntrinsicCount = 0;    ///< 最终全局 BA 中发生共享内参更新的相机数量
    double baSharedFocalScale = 1.0;    ///< 最终全局 BA 后焦距相对输入焦距的平均尺度
    double baSharedFocalAspectScale = 1.0; ///< 最终全局 BA 后 fy/fx 比例倍率
    double baSharedPrincipalOffsetX = 0.0; ///< 最终全局 BA 主点 X 平均偏移（像素）
    double baSharedPrincipalOffsetY = 0.0; ///< 最终全局 BA 主点 Y 平均偏移（像素）
    double baSharedRadialK1 = 0.0;         ///< 最终全局 BA 的共享一阶径向畸变系数
    double baSharedRadialK2 = 0.0;         ///< 最终全局 BA 的共享二阶径向畸变系数
    BABackend baRequestedBackend = BABackend::LegacyCpu; ///< 最终全局 BA 请求后端
    BABackend baUsedBackend = BABackend::LegacyCpu;      ///< 最终全局 BA 实际后端
    BASolveStatus baSolveStatus = BASolveStatus::NotRun; ///< 最终全局 BA 求解状态
    bool baSolutionUsable = false;                       ///< 求解结果是否满足写回前置条件
    bool baResultApplied = false;                        ///< 求解结果是否通过 SfM 质量门并写回
    bool baBackendFallback = false;                      ///< 是否发生后端回退
    int baObservationCount = 0;                         ///< 最终全局 BA 观测数量
    double baTotalSeconds = 0.0;                        ///< 最终全局 BA 总耗时
    std::string baBackendMessage;                       ///< 后端选择、回退或失败原因
    int priorTracksAccepted = 0;
    int priorTracksRejected = 0;
    int priorObservationsAccepted = 0;
    int priorObservationConflicts = 0;
    std::vector<std::string> priorTrackDiagnostics;
    bool controlNetworkApplied = false;
    int controlPointConstraintCount = 0;
    int checkPointResidualCount = 0;
    int controlScaleBarConstraintCount = 0;
    int checkScaleBarResidualCount = 0;
    double controlPointRms = 0.0;
    double checkPointRms = 0.0;
    double controlScaleBarRms = 0.0;
    double checkScaleBarRms = 0.0;
    std::string controlNetworkError;

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

    void addPriorTrack(const control_points::PriorTrack &track);
    void addPriorScaleBar(const control_points::PriorScaleBar &scaleBar);

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
    friend class ImageRegistrationEngine;
    friend class InitialPairInitializer;
    friend class KnownPoseReconstructor;
    friend class SfmBundleAdjustCoordinator;

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
    int _lastGlobalBARefinedIntrinsicCount = 0;
    double _lastGlobalBASharedFocalScale = 1.0;
    double _lastGlobalBASharedFocalAspectScale = 1.0;
    double _lastGlobalBASharedPrincipalOffsetX = 0.0;
    double _lastGlobalBASharedPrincipalOffsetY = 0.0;
    double _lastGlobalBASharedRadialK1 = 0.0;
    double _lastGlobalBASharedRadialK2 = 0.0;
    BABackend _lastGlobalBARequestedBackend = BABackend::LegacyCpu;
    BABackend _lastGlobalBAUsedBackend = BABackend::LegacyCpu;
    BASolveStatus _lastGlobalBASolveStatus = BASolveStatus::NotRun;
    bool _lastGlobalBASolutionUsable = false;
    bool _lastGlobalBAResultApplied = false;
    bool _lastGlobalBABackendFallback = false;
    int _lastGlobalBAObservationCount = 0;
    double _lastGlobalBATotalSeconds = 0.0;
    std::string _lastGlobalBABackendMessage;

    /// 增量维护的可见三维点计数缓存（imageId → 可见已三角化点数）
    std::unordered_map<ImageId, size_t> _visibilityCache;
    /// 可见计数缓存是否有效
    bool _visibilityCacheDirty = true;

    /// 图像注册失败次数（imageId → 连续失败次数）
    std::unordered_map<ImageId, int> _registerFailCount;
    /// 当前注册轮次内暂缓重试的图像，避免同一张高重叠坏候选连续耗尽重试次数
    std::unordered_set<ImageId> _deferredFailedImages;
    /// 永久失败图像集合（超过最大重试次数后移入）
    std::unordered_set<ImageId> _permanentlyFailedImages;

    std::vector<control_points::PriorTrack> _pendingPriorTracks;
    std::vector<control_points::PriorScaleBar> _pendingPriorScaleBars;
    std::vector<Track> _materializedPriorTracks;
    control_points::PriorTrackDiagnostics _priorTrackDiagnostics;
    bool _priorTracksMaterialized = false;
    bool _controlNetworkApplied = false;
    control_points::ControlNetworkResult _controlNetworkResult;
    int _lastControlPointConstraintCount = 0;
    int _lastControlScaleBarConstraintCount = 0;
    control_points::SimilarityTransform3D _controlNetworkTransform;

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

    void materializePriorTracks();
    void applyPriorTrackDiagnostics(IncrementalSfmResult *result) const;
    void applyControlNetworkDiagnostics(IncrementalSfmResult *result) const;
    bool tryApplyControlNetwork(const std::vector<ImageId> &baImageIds,
                                std::vector<Camera> *baCameras);
    const control_points::PriorTrack *priorTrack(const std::string &markerId) const;
    void tagPriorTrackSource(Track *track) const;

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
     * @brief 初始像对已经注册后，执行后续增量注册、BA 和结果组装。
     */
    IncrementalSfmResult runRegistrationFromCurrentInitialization(int totalImages,
                                                                  SfmProgressCallback progressCb);

    /**
     * @brief 清理一次初始像对试跑产生的运行态，并恢复到同一份输入影像/匹配。
     */
    void resetForInitialPairTrial(const SfmReconstruction &baseReconstruction);

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
     * @brief 最终全局 BA 后重试仍未注册、且已有序列邻居的影像。
     *
     * 仅接受真实 PnP 位姿；成功后重新三角化并再次执行全局 BA。
     */
    int retryUnregisteredImagesAfterFinalBA(int totalImages);

    /**
     * @brief 用最近的已注册序列相机辅助连续缺口补位。
     */
    bool findRegisteredSequenceNeighbor(ImageId imageId,
                                        int direction,
                                        ImageId *neighborOut,
                                        int *stepsOut) const;

    /**
     * @brief 序列模式下判断待注册影像是否至少有一个已注册的前/后序号邻居。
     */
    bool hasRegisteredSequenceNeighbor(ImageId imageId) const;

    /**
     * @brief 计算已注册相邻序号相机中心距离的中位数。
     */
    double registeredSequenceAdjacentDistanceMedian(ImageId excludedImageId) const;

    /**
     * @brief 检查 PnP 结果是否破坏照片序列的局部相机中心距离。
     */
    bool validateSequencePoseConsistency(ImageId imageId,
                                         const Camera &candidateCamera,
                                         std::string *reason) const;

    /**
     * @brief 使用已注册的序列相邻相机为 PnP 生成外参初值。
     */
    bool makeSequenceInitialPoseGuess(ImageId imageId, Camera *guessCamera) const;

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

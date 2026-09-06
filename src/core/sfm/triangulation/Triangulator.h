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

#include "FramePinholeCamera.h"
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

        /// 参考流程按特征尺度归一化重投影误差后再与阈值比较。
        bool normalizeReprojectionByFeatureScale = false;

        /// 几何估计仍只使用已注册观测，但新点绑定完整输入轨迹。
        /// 参考增量流程用此保持 track -> point 身份，使后续注册相机无需全图补轨。
        bool bindCompleteInputTrack = false;

        /// 多影像重建中延迟创建对应图里只有两个观测的纯两视点。
        /// 初始影像对、真正的双影像任务、人工标记以及具有潜在第三视图支持的轨迹自动豁免。
        bool deferPureTwoViewTracks = true;
    };

    /**
     * @brief 三角化统计。
     */
    struct TriangulationStats
    {
        int numCreated = 0;   ///< 本轮新增有效三维点数（包含原槽恢复）
        int numRestored = 0;  ///< 在稳定 point slot 上原位恢复的三维点数
        int numContinued = 0; ///< 延续现有轨迹的观测数
        int numFiltered = 0;  ///< 被过滤的三维点数

        int inputTracks = 0;               ///< 输入轨迹数
        int inputLongTracks = 0;           ///< 输入观测数 >= 3 的轨迹数
        int unusableTracks = 0;            ///< 因影像/特征/已占用观测不可用而跳过的轨迹数
        int noCandidateTracks = 0;         ///< 未找到任何可三角化候选的轨迹数
        int createdTwoViewTracks = 0;      ///< 最终只生成两视点的数量
        int createdLongTracks = 0;         ///< 最终生成 >=3 观测点的数量
        int deferredPureTwoViewTracks = 0; ///< 多影像场景中延迟创建的纯两视图组件数
        int reprojObservationRejected = 0; ///< 候选点补观测时因重投影误差被拒数量
        int depthObservationRejected = 0;  ///< 候选点补观测时因深度被拒数量
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
        Triangulator(SfmReconstruction& reconstruction, const CorrespondenceGraph& graph, int threadCount = 1);

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
        TriangulationStats triangulateImage(ImageId imageId,
                                            const TriangulatorOptions& options = TriangulatorOptions());

        /**
         * @brief 基于已经合并好的多视图轨迹批量创建三维点。
         *
         * 适用于已知相机位姿路径：先把 pairwise matches 合并为一致的多视观测，
         * 再按参考流程对每条轨迹做一次最近射线最小二乘三角化。候选并行计算、
         * 按输入顺序提交；任一已注册观测深度或重投影超限时拒绝整条轨迹。
         *
         * @param tracks   已经去除同图像冲突的轨迹
         * @param options  三角化选项
         * @return 三角化统计
         */
        TriangulationStats triangulateTracks(const std::vector<Track>& tracks,
                                             const TriangulatorOptions& options = TriangulatorOptions());

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
        int filterPoints(double maxReprojError = 2.0, double minTriAngle = 2.0, bool normalizeByFeatureScale = false);

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
        int completeTracks(const TriangulatorOptions& options = TriangulatorOptions());

        /**
         * @brief 利用当前相机位姿重新三角化所有三维点。
         *
         * BA 优化相机位姿后，原来的 3D 点坐标基于旧位姿，精度较差。
         * 此方法对每个三维点的所有观测执行参考版 closest-rays 多视三角化，
         * 更新坐标并重算重投影误差。如果重三角化结果更差则保留原值。
         *
         * 参考 COLMAP 的 Retriangulate 策略。
         *
         * @param maxReprojError  重三角化后允许的最大重投影误差
         * @return 成功重三角化的点数
         */
        int retriangulatePoints(double maxReprojError = 2.0, bool normalizeByFeatureScale = false);

        /**
         * @brief 用当前相机位姿重新计算所有三维点的重投影误差。
         *
         * 在 BA 或重三角化后调用，确保 ScenePoint3D::error 字段
         * 反映最新的相机位姿，为后续过滤提供准确依据。
         */
        void recomputeReprojErrors();

    private:
        SfmReconstruction& _reconstruction;
        const CorrespondenceGraph& _correspondenceGraph;
        int _threadCount = 1;

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
        bool triangulatePair(ImageId imgId1,
                             FeatureIdx featIdx1,
                             ImageId imgId2,
                             FeatureIdx featIdx2,
                             const TriangulatorOptions& options,
                             std::array<double, 3>& outXyz);

        /**
         * @brief 计算三维点在某相机中的重投影误差。
         * @return 重投影误差（像素），失败返回极大值
         */
        double computeReprojError(const std::array<double, 3>& xyz, ImageId imageId, FeatureIdx featureIdx) const;

        double
        normalizedReprojError(double error, ImageId imageId, FeatureIdx featureIdx, bool normalizeByFeatureScale) const;

        /// 两观测在完整对应图中是否构成没有潜在第三视图支持的孤立组件。
        bool isPureTwoViewComponent(const Track& track) const;

        /**
         * @brief 检查三维点在指定相机中的深度是否为正。
         * @return 深度为正返回 true
         */
        bool hasPositiveDepth(const std::array<double, 3>& xyz, ImageId imageId) const;

        /**
         * @brief 计算轨迹观测相机之间的最大三角化角。
         */
        double computeMaxTriangulationAngle(const std::array<double, 3>& xyz,
                                            const std::vector<TrackElement>& observations) const;

        /**
         * @brief 多视图最近射线三角化：求解 sum(I-dd^T)X=sum(I-dd^T)C。
         * @param observations  (imageId, featureIdx) 列表
         * @param outXyz        输出三维坐标
         * @return 成功返回 true
         */
        bool triangulateMultiView(const std::vector<TrackElement>& observations, std::array<double, 3>& outXyz) const;
    };

} // namespace xjw

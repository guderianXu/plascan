#pragma once
// =============================================================================
// 文件: MvsPipelineService.h
// 模块: MVS - 同步深度流程服务
// 说明:
//   管理完整 MVS 流程：
//     1. 对每个参考帧调用 PatchMatchCUDA::estimate
//     2. DepthMapFusion::fuse (COLMAP BFS) → 直接输出 3D FusedPoint
//     3. 通过回调暴露中间结果，不负责线程启动
// =============================================================================

#include "WorkflowExecution.h"
#include "MvsTypes.h"
#include "DepthFrameResult.h"
#include "depth_processing/DepthPostprocessor.h"
#include "PatchMatchCUDA.h"
#include "DepthMapFusion.h"
#include "DepthPyramidEstimator.h"
#include "DepthFrameQualityGate.h"
#include "DepthGeometryHypothesisReranker.h"
#include "DepthEvidenceConfidence.h"
#include "DepthCompletenessMetrics.h"
#include "DepthGapTargetedRecovery.h"
#include "DepthResidualReestimation.h"
#include "DepthMissingReason.h"
#include "DepthMemoryPolicy.h"
#include "MvsImageCache.h"
#include "MvsImagePreprocessor.h"
#include "MvsQualityReport.h"
#include "MvsSceneClassifier.h"
#include "DenseCloudBuilder.h"
#include "DensePointCloudCUDA.h"
#include "MvsSourcePlanner.h"
#include "MvsStageSnapshot.h"
#include "MvsVisibilityGraphBuilder.h"
#include "MvsViewSelection.h"
#include "MvsWorkspaceManifest.h"
#include "LearnedDepthCandidateGate.h"
#include "SparseCloudPreprocessor.h"

#include <QString>
#include <QSharedPointer>
#include <QJsonObject>
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <cstdint>

namespace xjw
{
    namespace mvs
    {

        struct DepthConsistencyFrameSourcePlan
        {
            std::vector<int> consistencySourceIndices;
            std::vector<int> geometrySourceViewIndices;
        };

        /// Freeze every frame's source plan before consistency updates any frame-level
        /// acceptance. This prevents processing order from removing sources from later
        /// frames in the same batch.
        std::vector<DepthConsistencyFrameSourcePlan>
        freezeDepthConsistencySourcePlans(const std::vector<DepthFrameResult>& frames,
                                          int viewCount,
                                          MvsSceneProfile sceneProfile,
                                          int requestedRepairSourceCount);

        namespace detail
        {
            /// Run one background task and invoke the failure handler exactly once when
            /// the task throws. The handler receives a user-facing diagnostic.
            void
            runDepthMapBackgroundTaskWithExceptionBoundary(const std::function<void()>& task,
                                                           const std::function<void(const QString&)>& failureHandler);

            /// An explicit source-angle experiment must never promote a frame whose
            /// PatchMatch source plan is short. Repeated quality refreshes remain idempotent.
            void applySourceAngleCapShortfallSafety(DepthFrameResult& result);

            /// Post-consistency evidence is mandatory only after all publication counts
            /// have been recorded. Frames rejected before consistency remain publishable
            /// as diagnostics without pretending that geometry evidence exists.
            bool hasCompletedConsistencyPublication(const DepthFrameResult& result);

            /// Whether this frame was expected to enter cross-view consistency. The
            /// initial quality role is immutable, so a later rejection cannot erase the
            /// obligation to publish complete consistency evidence.
            bool expectsConsistencyPublication(const DepthFrameResult& result, int viewCount);
        } // namespace detail

        struct MvsPipelineEvents
        {
            std::function<void(DepthFrameResult)> depthMapReady;
            std::function<void(QString, int, int, QString)> depthMapSaved;
            std::function<void(QJsonObject)> depthMapArtifactSaved;
            std::function<void(std::vector<DensePoint>)> pointCloudReady;
            std::function<void(QString, float)> progressChanged;
            std::function<void(QString)> errorOccurred;
            std::function<void(bool)> finished;
        };

        class MvsPipelineService final
        {
        public:
            explicit MvsPipelineService(task_runtime::WorkflowControl control = {});
            ~MvsPipelineService();
            MvsPipelineService(const MvsPipelineService&) = delete;
            MvsPipelineService& operator=(const MvsPipelineService&) = delete;
            void setEvents(MvsPipelineEvents events);
            task_runtime::WorkflowOutcome execute();
            void resetCancellation()
            {
                _cancelled.store(false);
            }

            /// 设置输入数据
            void setViews(const std::vector<CameraView>& views);
            void setSparseCloud(const SparseCloud& sparse);
            void setConfig(const DepthGenConfig& config);
            void setSkippedFrameIndices(const std::vector<int>& indices);

            const std::vector<CameraView>& views() const
            {
                return _views;
            }

            const SparseCloud& sparse() const
            {
                return _sparse;
            }

            const DepthGenConfig& config() const
            {
                return _config;
            }

            /// 设置输出目录（深度图 PNG 保存位置）
            void setOutputDir(const std::string& dir)
            {
                _outputDir = dir;
            }

            /// 请求取消
            void cancel()
            {
                _cancelled = true;
            }

            void requestCancel()
            {
                _cancelled = true;
            }

            bool isCancelled() const
            {
                return _cancelled.load();
            }

            /// 基于稀疏点投影生成深度过滤支撑区，返回 CV_8U 掩码 (255=保留)；覆盖率不可靠时返回空 Mat
            static cv::Mat buildSparseSupportMask(const std::vector<CameraView>& views,
                                                  const SparseCloud& sparse,
                                                  int refIdx,
                                                  int W,
                                                  int H,
                                                  const std::vector<int>& sourceIndices = {});

            /// 将同一帧可见稀疏点投影一次，供 hint 与支撑掩码在不同工作分辨率复用
            static std::vector<ProjectedSparseDepthSample>
            collectProjectedSparseDepthSamples(const SparseCloud& sparse,
                                               const FramePinholeCamera& camera,
                                               int imageWidth,
                                               int imageHeight,
                                               const std::vector<size_t>& visiblePointIndices);

            /// 基于已投影样本生成 PatchMatch hint 深度图
            static cv::Mat buildHintDepthFromProjectedSamples(int refIdx,
                                                              int W,
                                                              int H,
                                                              const std::vector<ProjectedSparseDepthSample>& samples);

            /// 基于已投影样本仅生成局部种子深度，不做距离传播；用于精细层覆盖粗层 hint
            static cv::Mat buildSparseSeedDepthFromProjectedSamples(
                int refIdx, int W, int H, const std::vector<ProjectedSparseDepthSample>& samples, int seedRadius = 3);

            /// 基于已投影样本生成稀疏支撑掩码，避免重复投影同一批稀疏点
            static cv::Mat buildSparseSupportMaskFromProjectedSamples(
                int refIdx, int W, int H, const std::vector<ProjectedSparseDepthSample>& samples);

            /// 基于原始灰度图生成内容区域掩码；近似全图有效时返回空 Mat 表示跳过过滤
            static cv::Mat buildContentMask(const cv::Mat& gray,
                                            float* coverage = nullptr,
                                            double* otsuThreshold = nullptr,
                                            int* adaptiveThreshold = nullptr);

            /// 将项目蒙版转换为 MVS 有效区：项目蒙版非零=排除，返回值 255=有效
            static cv::Mat projectMaskToValidMask(const cv::Mat& projectMask, cv::Size targetSize);

            /// 暗背景环拍物体可用内容亮度挖出项目蒙版内部开口；保护外轮廓并限制最大移除比例
            static cv::Mat refineOrbitalProjectValidMask(const cv::Mat& gray,
                                                         const cv::Mat& projectValidMask,
                                                         bool* refined = nullptr,
                                                         float* retainedRatio = nullptr);

            /// CUDA PatchMatch 显存不足后的下一次重试配置
            static PatchMatchConfig
            nextCudaRetryPatchMatchConfig(const PatchMatchConfig& config, int imageWidth, int imageHeight);

        private:
            /// 每估计完一帧就发出
            void depthMapReady(DepthFrameResult result);
            /// 每帧深度图保存为 PNG 后发出（path, width, height, refImagePath）
            void depthMapSaved(QString pngPath, int width, int height, QString refImagePath);
            /// 每帧深度图全部产物保存后发出结构化元数据，供项目树增量刷新
            void depthMapArtifactSaved(QJsonObject artifact);
            /// 点云生成完毕
            void pointCloudReady(std::vector<DensePoint> cloud);
            /// 进度更新
            void progressChanged(QString stage, float ratio);
            /// 出错
            void errorOccurred(QString msg);
            /// 整个流程完成
            void finished(bool success);

        private:
            struct FrameMvsCache
            {
                std::vector<size_t> visiblePointIndices;
                std::vector<size_t> sourceSharedPointIndices;
                std::vector<int> sourceViewIndices;
                std::vector<MvsSourcePlanEntry> sourceViewScores;
                int requestedSourceViewCount = 0;
                int sourceViewShortfall = 0;
                std::string sourceViewShortfallReason;
                QJsonObject sourceAngleDiagnostics;
                bool completeVisibilityCandidatePoolEnabled = false;
                bool completePoolChangedLegacyPlan = false;
            };

            /// 同步执行；调用方决定线程与生命周期
            void runInBackground();
            void runInBackgroundImpl();
            void clearRuntimeCachesAfterFailure();
            void emitFinishedOnce(bool success);

            /// 计算单帧深度图
            DepthFrameResult computeDepthForView(
                int refIdx,
                const DepthGenConfig* configOverride = nullptr,
                const std::function<bool(const DepthLevelSummary&, std::string*)>& firstLevelCompletionGate = {},
                const std::vector<cv::Mat>* frozenDepthMaps = nullptr);

            /// 预计算 MVS 可见性与源视图候选，避免每帧重复全量扫描稀疏点
            void prepareFrameCaches();
            void clearFrameCaches();
            std::vector<int> sourceViewIndicesForFrame(int refIdx, int maxSources) const;
            std::vector<size_t> visibleSparsePointIndicesForFrame(int refIdx,
                                                                  const std::vector<int>& sourceIndices,
                                                                  int minSourceViews) const;
            bool isSparsePointVisibleInFrame(int viewIdx, size_t pointIndex) const;

            /// 将 DepthFrameResult 组装为 FusionFrameInput
            FusionFrameInput buildFusionFrame(const DepthFrameResult& res);

            /// 估计参考帧的深度范围
            bool
            estimateDepthRange(int refIdx, float& zNear, float& zFar, const std::vector<int>& sourceIndices = {}) const;
            bool estimateDepthRangeFromVisiblePoints(int refIdx,
                                                     const std::vector<size_t>& visiblePointIndices,
                                                     float& zNear,
                                                     float& zFar) const;

            /// 从稀疏点云生成提示深度图
            cv::Mat buildHintDepth(int refIdx, int W, int H, const std::vector<int>& sourceIndices = {}) const;
            cv::Mat buildHintDepthFromVisiblePoints(int refIdx,
                                                    int W,
                                                    int H,
                                                    const std::vector<size_t>& visiblePointIndices) const;
            cv::Mat buildHintDepthForCamera(int refIdx,
                                            const FramePinholeCamera& camera,
                                            int W,
                                            int H,
                                            const std::vector<size_t>& visiblePointIndices) const;

            cv::Mat buildSparseSupportMaskFromVisiblePoints(int refIdx,
                                                            int W,
                                                            int H,
                                                            const std::vector<size_t>& visiblePointIndices) const;
            cv::Mat buildSparseSupportMaskForCamera(int refIdx,
                                                    const FramePinholeCamera& camera,
                                                    int W,
                                                    int H,
                                                    const std::vector<size_t>& visiblePointIndices) const;

            /// 双视图深度图左右一致性检查（剔除互不一致的深度像素）
            void crossCheckDepthConsistency();
            bool crossCheckDepthConsistencyStreaming();
            void recoverResidualDepthAfterConsistency();
            void applyLearnedDepthCandidatesAfterConsistency();
            void runDepthPoseRefinementCandidateStage(bool residentDepthFrames);

            /// 保存单帧深度图预览、原始深度和置信图，并通知 GUI 更新项目结果树
            bool saveDepthFrameArtifacts(int frameIndex, const DepthFrameResult& result, const QString& stageLabel);
            void captureStageSnapshot(int frameIndex,
                                      MvsStageSnapshotStage stage,
                                      const QString& boundary,
                                      const DepthFrameResult& result,
                                      const cv::Mat& depth,
                                      const cv::Mat& confidence,
                                      const cv::Mat& validMask = cv::Mat());
            bool ensurePreparedRasterArtifact(int frameIndex,
                                              MvsPreparedRasterArtifact* artifact,
                                              QString* errorMessage = nullptr);
            void initializeWorkspaceManifest();
            void markManifestFrameRunning(int frameIndex);
            void markManifestFrameFailed(int frameIndex, const QString& error);
            bool publishTerminalDepthCheckpoints();
            bool persistWorkspaceManifest(QString* errorMsg = nullptr);

            /// 在任何全量像素解码前从影像头部补齐尺寸，并配置统一图像 provider。
            bool probeImageMetadata(QString* errorMessage);
            bool initializeImageProvider(const MvsPipelineMemoryPolicyDecision& decision, QString* errorMessage);
            bool preloadImages(QString* errorMessage = nullptr);
            bool loadMvsImageFrame(int frameIndex,
                                   const std::atomic_bool* cancelFlag,
                                   MvsImageFrame* frame,
                                   std::string* errorMessage);
            MvsImageCache::ImageLease acquireImageFrame(int frameIndex, std::string* errorMessage = nullptr);

            std::vector<CameraView> _views;
            SparseCloud _sparse;
            DepthGenConfig _config;
            MvsSceneClassification _sceneClassification;
            MvsSceneProfile _effectiveSceneProfile = MvsSceneProfile::Custom;
            DepthFilterMode _effectiveDepthFilterMode = DepthFilterMode::Moderate;
            int _configuredSourceViewCount = 0;
            std::shared_ptr<std::atomic_bool> _cancellation;
            std::atomic_bool& _cancelled;
            std::atomic<bool> _finishedEmitted{false};
            MvsPipelineEvents _events;
            task_runtime::WorkflowControl _control;
            task_runtime::WorkflowOutcome _outcome;
            std::mutex _executionMutex;
            std::string _outputDir;
            std::string _consistencyDepthDirectory;
            bool _streamConsistencyStorageEnabled = false;
            std::unique_ptr<MvsStageSnapshotRecorder> _stageSnapshotRecorder;

            /// 缓存已估计的深度帧
            std::vector<DepthFrameResult> _depthFrames;
            std::vector<uint8_t> _skipFrameMask;

            /// 统一 MVS 图像 provider。eager 模式常驻全部帧，bounded 模式仅保留当前 worker 的引用帧和源帧。
            std::unique_ptr<MvsImageCache> _imageCache;
            MvsPipelineMemoryPolicyDecision _pipelineMemoryDecision;
            std::vector<MvsPreparedRasterArtifact> _preparedRasterArtifacts;
            std::mutex _preparedRasterArtifactsMutex;

            /// MVS 稀疏点可见性与源视图缓存；runInBackground 中预计算一次，帧 worker 仅读取
            std::vector<FrameMvsCache> _frameCaches;
            std::vector<uint64_t> _visibilityBits;
            std::vector<std::vector<MvsVisibilityNeighbor>> _visibilityAdjacency;
            size_t _visibilityWordCount = 0;
            bool _frameCachesReady = false;

            QString _workspaceManifestPath;
            QString _depthConfigHash;
            MvsWorkspaceManifest _workspaceManifest;
            std::mutex _workspaceManifestMutex;

        public:
            /// 融合完可获取每帧一致性过滤的深度图（返回副本，线程安全）
            std::vector<cv::Mat> filteredDepths() const
            {
                std::lock_guard<std::mutex> lock(_filteredDepthsMutex);
                return _filteredDepths;
            }

        private:
            std::vector<cv::Mat> _filteredDepths;
            mutable std::mutex _filteredDepthsMutex;
        };

    } // namespace mvs
} // namespace xjw

#pragma once

// Public depth-frame data contract. No QObject lifecycle, scheduling or generator dependency.

#include <string>
#include <vector>

#include <QJsonObject>
#include <QSharedPointer>

#include "DepthCompletenessMetrics.h"
#include "DepthEvidenceConfidence.h"
#include "DepthFrameQualityGate.h"
#include "DepthGeometryHypothesisReranker.h"
#include "DepthPyramidTypes.h"
#include "MvsQualityReport.h"
#include "MvsSourcePlanner.h"

namespace xjw::mvs
{

    /// 单帧深度估计结果
    struct DepthFrameResult
    {
        int refViewIdx = -1; ///< 参考帧在 views 数组中的下标
        bool depthFlippedZ = false;
        cv::Size preparedRasterSize;                    ///< MVS 准备后的全分辨率影像尺寸；不随深度缓存释放丢失
        bool effectiveNativeFinalDepthGrid = false;     ///< 该帧实际采用实验原生最终网格策略
        QJsonObject pixelDomainDiagnostics;             ///< full-raster 参数到实际深度网格参数的显式审计
        FramePinholeCamera cameraModel;                 ///< 与输出深度栅格严格对应的正深度、零畸变工作相机
        std::vector<int> sourceViewIndices;             ///< PatchMatch 实际使用的源视图下标，用于限制一致性检查范围
        std::vector<int> geometrySourceViewIndices;     ///< geometrySourceMask 的精确位序表，最多 16 个来源
        std::vector<MvsSourcePlanEntry> sourceViewPlan; ///< 实际源视图的可审计几何选择依据
        QJsonObject sourceAngleDiagnostics;             ///< 场景推导上限、实验 cap 与实际源视图角度上限
        bool sourceAngleCapEnabled = false;             ///< 强类型安全标志；不依赖 JSON 诊断执行 shortfall 降级
        bool completeVisibilityCandidatePoolEnabled = false; ///< 默认关闭的完整候选池实验
        bool completePoolChangedLegacyPlan = false;          ///< 本帧最终计划是否真正不同于 legacy early-stop
        bool initialQualityAcceptanceAvailable = false;      ///< 最终准入不得伪造初始阶段角色
        DepthFrameAcceptance initialQualityAcceptance = DepthFrameAcceptance::Rejected;
        bool geometricGuidancePassExpected = false;    ///< 当前帧是否满足冻结源深度第二轮的执行前提
        bool geometricGuidancePassApplied = false;     ///< 第二轮是否实际成功，未执行结果不得被当作完整缓存复用
        int requestedSourceViewCount = 0;              ///< 选择阶段请求的源视图数，不随缓存释放丢失
        int sourceViewShortfall = 0;                   ///< 请求数与实际可用源视图数之差
        std::string sourceViewShortfallReason;         ///< 源视图不足的主要原因
        QSharedPointer<cv::Mat> depthMap;              ///< 深度图 (CV_32F)
        QSharedPointer<cv::Mat> confidence;            ///< 置信图 (CV_32F)
        QSharedPointer<cv::Mat> photometricConfidence; ///< 原始光度置信度 (CV_32F)，双通道实验启用时保留
        QSharedPointer<cv::Mat> geometricConfidence;   ///< 独立跨视几何置信度 (CV_32F)，双通道实验启用时保留
        QSharedPointer<cv::Mat> normalMap;             ///< 最终层法线图 (CV_32FC3)，可为空
        QSharedPointer<cv::Mat> supportCount;          ///< PatchMatch 候选来源数诊断图；不得作为最终几何支持 (CV_16U)
        QSharedPointer<cv::Mat> photometricSourceMask; ///< PatchMatch 每像素光度来源 bitset (CV_32SC1)
        QSharedPointer<cv::Mat> geometrySupportCount;  ///< 参考帧+跨视几何确认数 (CV_16U)
        QSharedPointer<cv::Mat> geometrySourceMask;    ///< bit N 对应 geometrySourceViewIndices 的第 N 个来源 (CV_16U)
        QSharedPointer<cv::Mat> inverseDepthMean;      ///< 几何确认观测的逆深度均值 (CV_32F)
        QSharedPointer<cv::Mat> inverseDepthRelativeSpread;         ///< 逆深度相对标准差 (CV_32F)
        QSharedPointer<cv::Mat> adaptiveGeometrySupportWeight;      ///< 连续跨视几何支持权重 (CV_32F)
        QSharedPointer<cv::Mat> adaptiveGeometryEffectiveViewCount; ///< 连续证据有效视图数 (CV_32F)
        QSharedPointer<cv::Mat> adaptiveGeometryConflictRatio;      ///< 可观测证据中的冲突比例 [0, 1] (CV_32F)
        QSharedPointer<cv::Mat> depthLayerReliabilityClass;         ///< 观测用深度层可靠性分类 (CV_8U)，不参与准入
        QSharedPointer<DepthGeometryHypothesisRerankMaps> geometryRerankMaps; ///< 几何候选代价与独立性逐像素审计
        QSharedPointer<cv::Mat> crossViewRepairedMask;            ///< 跨视图补回像素；不参与帧准入评分 (CV_8U)
        QSharedPointer<cv::Mat> targetedGapRecoveredMask;         ///< 定向二源 PatchMatch 恢复像素 (CV_8U)
        bool targetedGapRecoveredMaskExpected = false;            ///< 流式释放后仍保留的权威存在性，用于检测工件丢失
        QSharedPointer<cv::Mat> residualReestimatedMask;          ///< 一致性后局部 PatchMatch 恢复像素 (CV_8U)
        QSharedPointer<cv::Mat> learnedCandidateAcceptedMask;     ///< 学习候选通过独立几何门控的像素 (CV_8U)
        QSharedPointer<cv::Mat> depthProvenance;                  ///< 最终深度来源码 (CV_8U, DepthProvenance)
        QSharedPointer<cv::Mat> missingReasonMap;                 ///< 最终缺失像素的逐像素原因码 (CV_8U)
        QSharedPointer<cv::Mat> validMask;                        ///< 最终输出空间的权威有效蒙版 (CV_8U)
        QSharedPointer<cv::Mat> supportRegionMask;                ///< 项目/内容允许参与重建的区域，不含深度孔洞
        DepthPostProcessStats depthPostprocess;                   ///< 融合前深度图后处理统计
        DepthCompletenessDiagnostics depthCompleteness;           ///< 蒙版内覆盖和逐阶段损失诊断
        QJsonObject crossViewRepairDiagnostics;                   ///< 跨视补回和锚定插值的逐原因统计
        QJsonObject targetedGapRecoveryDiagnostics;               ///< 缺口定向 PatchMatch 请求、接受和拒绝统计
        QJsonObject residualReestimationDiagnostics;              ///< 一致性后残余缺口局部实测恢复统计
        QJsonObject evidenceConfidenceDiagnostics;                ///< 光度/几何双通道置信度与因果准入统计
        DepthEvidenceConfidenceSummary evidenceConfidenceSummary; ///< 双通道置信度的强类型帧级摘要
        QJsonObject learnedCandidateDiagnostics;                  ///< 学习候选加载与最终几何门控统计
        QJsonObject poseRefinementDiagnostics;                    ///< 深度约束位姿细化候选与安全门诊断
        FramePinholeCamera derivedCameraModel;                    ///< 可选派生相机候选；绝不覆盖 cameraModel 或项目相机
        std::vector<DepthLevelSummary> pyramidLevels;             ///< 三级深度估计逐层摘要
        std::vector<DepthLevelResult> intermediatePyramidLevels;  ///< 可选的 L3/L2 调试结果
        std::vector<ProjectedSparseDepthSample> projectedSparseDepthSamples; ///< 最终输出栅格上的稀疏绝对深度锚点
        SparseDepthResidualSummary sparseDepthResidual;                      ///< 最终深度相对稀疏锚点的稳健残差审计
        DepthMapQualityMetrics qualityMetrics;                               ///< 帧级覆盖、连通性与搜索边界统计
        DepthFrameQualityDecision qualityDecision;                           ///< 是否允许进入多视融合
        std::string maskSource;                                              ///< project/content/technical/full_image
        float maskCoverage = 1.0f;                                           ///< 参考影像中允许参与 MVS 的像素比例
        int selectedLevel = 0;                                               ///< 实际采用的深度金字塔层级
        std::string fallbackReason;                                          ///< 层级减少或细层失败原因
        int pyramidRequestedLevelCount = 3;
        int pyramidActiveLevelCount = 0;
        int pyramidMinimumShortSide = 0;
        std::string pyramidDegradedReason;
        float effectivePatchMatchConfidenceThreshold = 0.0f;
        bool depthPostprocessApplied = false; ///< true 表示 depthMap/confidence 已应用上述后处理
        bool success = false;
        double elapsedMs = 0.0; ///< 单帧深度估计耗时，不含异步写盘
        std::string device;     ///< 实际调度设备：GPU/CPU
        std::string errorMsg;

        bool eligibleForFusion() const
        {
            return success && qualityDecision.acceptance == DepthFrameAcceptance::Accepted;
        }

        bool eligibleForConsistencyCheck() const
        {
            return success && qualityDecision.acceptance != DepthFrameAcceptance::Rejected;
        }

        bool eligibleAsConsistencySource() const
        {
            return eligibleForConsistencyCheck();
        }

        /// Release large per-pixel products while retaining the content/support
        /// domain required by streaming consistency and final artifact diagnostics.
        void releaseStreamingPixelStorage()
        {
            depthMap.clear();
            confidence.clear();
            photometricConfidence.clear();
            geometricConfidence.clear();
            normalMap.clear();
            supportCount.clear();
            photometricSourceMask.clear();
            geometrySupportCount.clear();
            geometrySourceMask.clear();
            inverseDepthMean.clear();
            inverseDepthRelativeSpread.clear();
            adaptiveGeometrySupportWeight.clear();
            adaptiveGeometryEffectiveViewCount.clear();
            adaptiveGeometryConflictRatio.clear();
            depthLayerReliabilityClass.clear();
            geometryRerankMaps.clear();
            crossViewRepairedMask.clear();
            targetedGapRecoveredMask.clear();
            residualReestimatedMask.clear();
            learnedCandidateAcceptedMask.clear();
            depthProvenance.clear();
            missingReasonMap.clear();
            validMask.clear();
            intermediatePyramidLevels.clear();
        }

        /// Release every per-pixel allocation, including the support domain.
        /// Use this for terminal cleanup paths where no later artifact is produced.
        void releasePixelStorage()
        {
            releaseStreamingPixelStorage();
            supportRegionMask.clear();
        }
    };

} // namespace xjw::mvs

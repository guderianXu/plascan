#pragma once

#include "PairSelectionPolicy.h"

#include <QString>

namespace xjw
{
    namespace matchphotos
    {

        /// “对齐照片”精度只控制首层特征图采样，不改变关键点配额、像对预选或 SfM/BA。
        enum class AlignmentAccuracy
        {
            Highest = 0,
            High = 1,
            Medium = 2,
            Low = 4,
            Lowest = 8
        };

        constexpr int alignmentAccuracyDownscale(AlignmentAccuracy accuracy)
        {
            return static_cast<int>(accuracy);
        }

        inline QString alignmentAccuracyName(AlignmentAccuracy accuracy)
        {
            switch (accuracy)
            {
            case AlignmentAccuracy::Highest:
                return QStringLiteral("highest");
            case AlignmentAccuracy::High:
                return QStringLiteral("high");
            case AlignmentAccuracy::Medium:
                return QStringLiteral("medium");
            case AlignmentAccuracy::Low:
                return QStringLiteral("low");
            case AlignmentAccuracy::Lowest:
                return QStringLiteral("lowest");
            }
            return QStringLiteral("high");
        }

        inline AlignmentAccuracy alignmentAccuracyFromName(const QString& name,
                                                           AlignmentAccuracy fallback = AlignmentAccuracy::High)
        {
            const QString normalized = name.trimmed().toLower();
            if (normalized == QLatin1String("highest") || normalized == QLatin1String("0"))
            {
                return AlignmentAccuracy::Highest;
            }
            if (normalized == QLatin1String("high") || normalized == QLatin1String("1"))
            {
                return AlignmentAccuracy::High;
            }
            if (normalized == QLatin1String("medium") || normalized == QLatin1String("standard") ||
                normalized == QLatin1String("2"))
            {
                return AlignmentAccuracy::Medium;
            }
            if (normalized == QLatin1String("low") || normalized == QLatin1String("4"))
            {
                return AlignmentAccuracy::Low;
            }
            if (normalized == QLatin1String("lowest") || normalized == QLatin1String("fast") ||
                normalized == QLatin1String("8"))
            {
                return AlignmentAccuracy::Lowest;
            }
            return fallback;
        }

        // 面向用户的预设应映射到合理默认值，而不是在主流程 UI 中暴露每个检测器
        // 和匹配器的细碎参数。
        enum class MatchPhotosProfile
        {
            Auto,
            Fast,
            HighAccuracy,
            DifficultTexture,
            CpuCompatible,
            CudaAccelerated
        };

        enum class ComputeDevice
        {
            Auto,
            Cpu,
            Cuda,
            OpenCl,
            Metal
        };

        enum class ReferencePreselectionMode
        {
            Source,
            Estimated,
            Sequential
        };

        inline QString referencePreselectionModeName(ReferencePreselectionMode mode)
        {
            switch (mode)
            {
            case ReferencePreselectionMode::Source:
                return QStringLiteral("source");
            case ReferencePreselectionMode::Estimated:
                return QStringLiteral("estimated");
            case ReferencePreselectionMode::Sequential:
                return QStringLiteral("sequential");
            }
            return QStringLiteral("source");
        }

        inline ReferencePreselectionMode
        referencePreselectionModeFromName(const QString& name,
                                          ReferencePreselectionMode fallback = ReferencePreselectionMode::Source)
        {
            const QString normalized = name.trimmed().toLower();
            if (normalized == QLatin1String("source") || normalized == QLatin1String("source_code"))
            {
                return ReferencePreselectionMode::Source;
            }
            if (normalized == QLatin1String("estimated") || normalized == QLatin1String("estimated_pose"))
            {
                return ReferencePreselectionMode::Estimated;
            }
            if (normalized == QLatin1String("sequence") || normalized == QLatin1String("sequential") ||
                normalized == QLatin1String("photo_sequence"))
            {
                return ReferencePreselectionMode::Sequential;
            }
            return fallback;
        }

        enum class GuidedMatchingMode
        {
            Disabled,
            Automatic,
            Forced
        };

        inline QString guidedMatchingModeName(GuidedMatchingMode mode)
        {
            switch (mode)
            {
            case GuidedMatchingMode::Disabled:
                return QStringLiteral("off");
            case GuidedMatchingMode::Automatic:
                return QStringLiteral("auto");
            case GuidedMatchingMode::Forced:
                return QStringLiteral("force");
            }
            return QStringLiteral("off");
        }

        inline bool guidedMatchingEnabled(GuidedMatchingMode mode)
        {
            return mode != GuidedMatchingMode::Disabled;
        }

        inline GuidedMatchingMode guidedMatchingModeFromName(const QString& name,
                                                             GuidedMatchingMode fallback = GuidedMatchingMode::Disabled)
        {
            const QString normalized = name.trimmed().toLower();
            if (normalized == QLatin1String("off") || normalized == QLatin1String("disabled"))
            {
                return GuidedMatchingMode::Disabled;
            }
            if (normalized == QLatin1String("auto") || normalized == QLatin1String("automatic"))
            {
                return GuidedMatchingMode::Automatic;
            }
            if (normalized == QLatin1String("force") || normalized == QLatin1String("forced"))
            {
                return GuidedMatchingMode::Forced;
            }
            return fallback;
        }

        struct MatchPhotosOptions
        {
            MatchPhotosProfile profile = MatchPhotosProfile::Auto;
            AlignmentAccuracy accuracy = AlignmentAccuracy::High;
            ComputeDevice device = ComputeDevice::Auto;

            // 推荐指定便携 LightGlue ONNX；历史本机 `.engine` 仅作为兼容输入。
            QString lightGlueTensorRtEnginePath;
            // LoMa-R schema 2 清单绑定特征/匹配 ONNX，运行时据本机环境生成 engine。
            QString lomaRTensorRtPackagePath;
            // 0 表示按 GPU 总显存与关键点上限自动选择 1024/2048/3840 bucket；
            // 非 0 值表示用户明确指定档位。显式 manifest 路径始终优先于档位选择。
            int lomaRKeypointBudget = 0;

            // 影像对规划保持显式配置，方便调用方复用 overlap 结果，
            // 或在测试、批处理中强制使用确定性模式。
            PairSelectionPolicy pairPolicy = makePairSelectionPolicy(PairSelectionPreset::Auto);

            // 组合算法通过统一注册表选择。当前注册 auto_sift、plamatch_hct、
            // sift_lightglue 和 loma_r；
            // 新实现只需实现 IImageMatchingAlgorithm 并注册，不再增加特征/匹配双重 token。
            QString algorithmId = QStringLiteral("plamatch_hct");
            // 蒙版应用阶段：none=不使用，keypoints=提取后过滤关键点，tiepoints=匹配后过滤连接点。
            // 项目蒙版约定为 0 表示有效区域，非 0 表示排除区域。
            QString maskApplyMode = QStringLiteral("none");

            // 0 不额外限制；正值是独立于 Accuracy 的显式内存安全上限。
            int maxImageDim = 0;
            int maxKeypoints = 0;
            int keypointLimitPerMegapixel = 0;
            int cudaDevice = 0;
            // 0 表示根据可用显存自动选择；CPU 路径始终按 1 处理。
            int cudaParallelPairs = 0;
            // CUDA SIFT 流水线最多预读的影像数，避免大 TIFF 占满主机内存。
            int featurePrefetchDepth = 2;
            float matchThreshold = 0.15f;
            // SIFT 最近邻/次近邻距离比。自适应模式把该值作为宽松上限，在候选充足时
            // 根据当前像对的双向互检分布自动收紧；小像对保留上限，避免少量匹配被裁空。
            float siftMaximumRatio = 0.98f;
            float siftMinimumAdaptiveRatio = 0.78f;
            bool adaptiveSiftRatio = true;
            double geometryReprojThreshold = 1.5;
            int geometryMinInliers = 20;
            double geometryMinInlierRatio = 0.18;
            double geometryMinGridCoverage = 0.12;
            int geometryGridColumns = 4;
            int geometryGridRows = 4;
            int geometryMaxIterations = 10000;
            int maxTiePointsPerImage = 4000;
            // 旧 PlaScan 网格硬配额字段只为项目/调用兼容保留。默认轨迹链采用
            // “对齐照片”自行确定的至多 16×16 水位网格，不读取以下三个值。
            int maxTiePointsPerGridCell = 500;
            int tiePointGridColumns = 4;
            int tiePointGridRows = 4;
            bool enableGeometryVerification = true;
            bool enableTrackBuild = true;
            // Automatic 只补救几何可靠但支持度、内点率或覆盖率偏弱的像对；Forced
            // 对所有具备可靠基础矩阵或可信参考位姿的像对执行引导搜索。
            GuidedMatchingMode guidedMatchingMode = GuidedMatchingMode::Disabled;
            bool guidedUseReferenceCameraPoses = false;
            bool guidedRequireMultiViewConsistency = true;
            bool useExplicitKeypointLimit = false;
            bool useGenericPreselection = true;
            bool useReferencePreselection = false;
            // PlaMatch-HCT 与“对齐照片”一致：Source/Estimated 只决定上游传入哪套
            // 相机中心，二者使用相同的最近位置预选；Sequential 依赖序列组元数据。
            ReferencePreselectionMode referencePreselectionMode = ReferencePreselectionMode::Source;
            int referencePreselectionNeighbors = 10;
            bool excludeStationaryTiePoints = true;
            // 复用与当前影像指纹、算法版本和配置指纹完全一致的 `.pimatch` 数据；
            // PlaMatch-HCT 同时复用完整描述子与 coarse/global 预选特征缓存。
            bool reuseExistingMatches = true;
            float stationaryTiePointMaxPixelMotion = 1.0f; ///< 旧固定像素阈值，仅为配置兼容保留。

            // 项目蒙版是排除概率（0=有效，255=确定排除）。只硬裁高置信度且位于
            // 排除区内部的点；边界和不确定区域以软权重保留。
            float maskHardExclusionThreshold = 0.90f;
            float maskMinimumTiepointWeight = 0.20f;
            int maskRelaxationRadius = 2;

            // 在特征、匹配、几何验证和轨迹阶段接入现有核心模块前，
            // 框架默认先以 plan-only 方式运行。
            bool planOnly = true;
        };

    } // namespace matchphotos
} // namespace xjw

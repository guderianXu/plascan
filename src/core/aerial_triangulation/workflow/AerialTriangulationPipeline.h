#pragma once

/**
 * @file AerialTriangulationPipeline.h
 * @brief 连接点已就绪后的焦距搜索、SfM 候选择优和正式结果写出管线。
 *
 * Pipeline 不提取特征、不匹配影像。无可信内参时，它并行运行多个焦距初始化；
 * 大规模工程先用受限注册规模探测候选，再只对最佳焦距重放一次完整 SfM/BA。
 * SfmSearchPolicy 负责选择摄影测量网络更可靠的模型，最后只写出正式重放结果。
 */

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"
#include "reconstruction/SfmAttemptRunner.h"

#include <functional>

namespace xjw::aerial_triangulation
{

/// 正式重建管线；函数对象注入点用于测试职责边界和候选策略。
class AerialTriangulationPipeline
{
public:
    /// 执行一次具体焦距/配置的 SfM，默认由 SfmAttemptRunner 提供。
    using AttemptRunner = std::function<SfmAttemptExecutionResult(
        const PreparedAerialTriangulationInput &input)>;

    /// 把胜出候选写成正式资产，默认由 AerialTriangulationResultWriter 提供。
    using ResultWriter = std::function<bool(
        const PreparedAerialTriangulationInput &input,
        SfmAttemptExecutionResult *execution,
        QString *errorMessage)>;

    /**
     * @brief 构造可执行管线。
     * @param attemptRunner 为空时安装生产 SfmAttemptRunner。
     * @param resultWriter 为空时安装生产结果写出器。
     */
    explicit AerialTriangulationPipeline(AttemptRunner attemptRunner = {},
                                         ResultWriter resultWriter = {});

    /**
     * @brief 执行焦距初始化搜索、可选自标定细化和正式写出。
     *
     * input.tiePointPath 必须已存在。函数同步等待全部候选 worker，GUI 应放在后台任务。
     */
    AerialTriangulationReconstructionResult run(
        const PreparedAerialTriangulationInput &input) const;

    /// 对无绝对控制的近垂直摄影块标记可能的整体穹顶，提示用户复核而不强行拉平。
    static bool shouldFlagAerialDomingRisk(
        bool geometryValid,
        double opticalAxisConcentration,
        double cameraCenterNormalSpanRmsRatio,
        bool hasAbsoluteGeometryConstraint);

private:
    bool _usesProductionAttemptRunner = false; ///< 生产路径可在候选启动前共享连接点图。
    AttemptRunner _attemptRunner; ///< 单次候选执行器。
    ResultWriter _resultWriter; ///< 正式提交器。
};

} // namespace xjw::aerial_triangulation

#pragma once

/**
 * @file AerialTriangulationWorkflow.h
 * @brief Metashape“对齐照片”语义的唯一核心编排入口。
 *
 * Workflow 负责解析 GUI/CLI 参数、决定复用/补齐/重建连接点，再调用正式
 * AerialTriangulationPipeline。它不包含 SfM 数值算法，也不直接写工程文件。
 */

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResolvedConfig.h"
#include "model/AerialTriangulationResult.h"

#include <functional>

namespace xjw::aerial_triangulation
{

/**
 * @brief 对齐照片的外观层与算法层边界。
 *
 * GUI 和 CLI 只构造 AerialTriangulationOptions。所有默认值、质量预设、路径、
 * 蒙版、预选模式和进度区间必须在此统一解析，避免两个入口产生不同算法行为。
 */
class AerialTriangulationWorkflow
{
public:
    /// 已有连接点输入下的正式管线注入点。
    using PipelineRunner = std::function<AerialTriangulationReconstructionResult(
        const PreparedAerialTriangulationInput &input)>;

    /// 连接点任务注入点；为空时使用 MatchPhotosTask。
    using TiePointRunner = std::function<matchphotos::MatchPhotosResult(
        const matchphotos::MatchPhotosOptions &options,
        const matchphotos::MatchPhotosContext &context)>;

    /**
     * @brief 解析实际生效配置，但不访问缓存、不运行算法。
     *
     * 关键语义：resetAlignment 只决定是否复用工程相机位姿；
     * reuseExistingMatches 才决定是否保留匹配/连接点缓存。
     */
    static AerialTriangulationResolvedConfig resolveConfig(
        const AerialTriangulationOptions &options);

    /// 使用生产连接点任务和生产 SfM 管线执行完整工作流。
    static AerialTriangulationResult run(const AerialTriangulationOptions &options);

    /**
     * @brief 带执行器注入的完整工作流。
     *
     * 仅在 resolved.prepareTiePoints=true 时调用 tiePointRunner；连接点准备失败后
     * 不会启动 pipelineRunner。返回结果始终包含 resolved config 供上层回显。
     */
    static AerialTriangulationResult run(const AerialTriangulationOptions &options,
                                         const PipelineRunner &pipelineRunner,
                                         const TiePointRunner &tiePointRunner = {});
};

} // namespace xjw::aerial_triangulation

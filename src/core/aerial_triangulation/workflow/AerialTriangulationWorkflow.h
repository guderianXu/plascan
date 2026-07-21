#pragma once

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResolvedConfig.h"
#include "model/AerialTriangulationResult.h"

#include <functional>

namespace xjw::aerial_triangulation
{

// Metashape“对齐照片”语义的唯一核心入口。GUI 和 CLI 只构造 Options，
// 连接点准备与 SfM/BA 的职责边界由此处统一编排。
class AerialTriangulationWorkflow
{
public:
    using PipelineRunner = std::function<AerialTriangulationReconstructionResult(
        const PreparedAerialTriangulationInput &input)>;
    using TiePointRunner = std::function<matchphotos::MatchPhotosResult(
        const matchphotos::MatchPhotosOptions &options,
        const matchphotos::MatchPhotosContext &context)>;

    static AerialTriangulationResolvedConfig resolveConfig(
        const AerialTriangulationOptions &options);

    static AerialTriangulationResult run(const AerialTriangulationOptions &options);

    static AerialTriangulationResult run(const AerialTriangulationOptions &options,
                                         const PipelineRunner &pipelineRunner,
                                         const TiePointRunner &tiePointRunner = {});
};

} // namespace xjw::aerial_triangulation

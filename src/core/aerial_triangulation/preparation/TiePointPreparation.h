#pragma once

/**
 * @file TiePointPreparation.h
 * @brief 空三工作流到 matchphototask 的唯一连接点准备适配器。
 *
 * 类本身不实现特征提取或匹配；默认调用 MatchPhotosTask，测试可注入 Runner。
 */

#include "matchphototask/task/MatchPhotosTask.h"

#include <functional>

namespace xjw::aerial_triangulation
{

/// 无状态连接点任务入口。
class TiePointPreparation
{
public:
    using Runner = std::function<matchphotos::MatchPhotosResult(
        const matchphotos::MatchPhotosOptions &options,
        const matchphotos::MatchPhotosContext &context)>;

    /**
     * @brief 执行特征、候选对、匹配、几何验证和多视轨迹落盘。
     *
     * runner 非空时完全替代默认任务，便于验证 Workflow 是否正确传递解析配置。
     */
    static matchphotos::MatchPhotosResult run(
        const matchphotos::MatchPhotosOptions &options,
        const matchphotos::MatchPhotosContext &context,
        const Runner &runner = {});
};

} // namespace xjw::aerial_triangulation

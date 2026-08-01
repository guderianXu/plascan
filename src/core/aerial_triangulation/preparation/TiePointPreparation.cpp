/**
 * @file TiePointPreparation.cpp
 * @brief 空三对 MatchPhotosTask 的最薄执行适配层。
 *
 * 该适配层不修改任何参数。测试可注入 runner，生产路径则直接调用唯一的
 * matchphototask 实现，避免空三维护第二套特征/匹配流程。
 */

#include "TiePointPreparation.h"

namespace xjw::aerial_triangulation
{

matchphotos::MatchPhotosResult TiePointPreparation::run(
    const matchphotos::MatchPhotosOptions &options,
    const matchphotos::MatchPhotosContext &context,
    const Runner &runner)
{
    // 注入执行器和生产执行器接收完全相同的 options/context。
    if (runner)
    {
        return runner(options, context);
    }
    return matchphotos::MatchPhotosTask(options).run(context);
}

} // namespace xjw::aerial_triangulation

#include "TiePointPreparation.h"

namespace xjw::aerial_triangulation
{

matchphotos::MatchPhotosResult TiePointPreparation::run(
    const matchphotos::MatchPhotosOptions &options,
    const matchphotos::MatchPhotosContext &context,
    const Runner &runner)
{
    if (runner)
    {
        return runner(options, context);
    }
    return matchphotos::MatchPhotosTask(options).run(context);
}

} // namespace xjw::aerial_triangulation

#pragma once

#include "matchphototask/task/MatchPhotosTask.h"

#include <functional>

namespace xjw::aerial_triangulation
{

class TiePointPreparation
{
public:
    using Runner = std::function<matchphotos::MatchPhotosResult(
        const matchphotos::MatchPhotosOptions &options,
        const matchphotos::MatchPhotosContext &context)>;

    static matchphotos::MatchPhotosResult run(
        const matchphotos::MatchPhotosOptions &options,
        const matchphotos::MatchPhotosContext &context,
        const Runner &runner = {});
};

} // namespace xjw::aerial_triangulation

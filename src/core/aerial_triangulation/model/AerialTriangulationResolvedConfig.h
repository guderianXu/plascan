#pragma once

#include "model/AerialTriangulationOptions.h"
#include "matchphototask/task/MatchPhotosTask.h"

#include <QJsonObject>

namespace xjw::aerial_triangulation
{

struct AerialTriangulationResolvedConfig
{
    PreparedAerialTriangulationInput pipelineInput;
    matchphotos::MatchPhotosOptions tiePointOptions;
    matchphotos::MatchPhotosContext tiePointContext;
    bool prepareTiePoints = false;
    bool forceRebuildTiePoints = false;
    QJsonObject resolvedSettings;
};

} // namespace xjw::aerial_triangulation

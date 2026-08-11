#pragma once

#include "ObjRenderPreparation.h"

ObjRenderPreparation prepareObjPointPreview(
    const ObjRenderCloud &cloud,
    const std::atomic_bool *cancellationFlag,
    const ObjPrepareProgressCallback &progress,
    const ObjRenderPreparationLimits &limits);

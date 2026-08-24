#pragma once

#include "BundleAdjustTypes.h"

#include <plamatrix/optimization/block_schur.h>

namespace xjw::detail
{

plamatrix::SchurComplementLinearBackend plaMatrixLinearBackend(BABackend backend);

const char* plaMatrixLinearBackendName(
    plamatrix::SchurComplementLinearBackend backend);

} // namespace xjw::detail

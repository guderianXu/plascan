#include "placamera/model.h"

namespace placamera
{

    OptimizationLayout RasterModel::optimizationLayout() const
    {
        return {};
    }

    Result<RasterModelPtr> RasterModel::withOptimizationUpdate(const OptimizationUpdate& update) const
    {
        (void)update;
        return Result<RasterModelPtr>::failure(CameraErrorCode::UnsupportedModel,
                                               "camera model does not expose an optimization parameterization");
    }

} // namespace placamera

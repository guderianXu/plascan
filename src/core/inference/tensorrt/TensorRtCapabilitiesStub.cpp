#include "TensorRtCapabilities.h"

namespace xjw::inference
{

    TensorRtCapabilities queryTensorRtCapabilities(int cudaDevice) noexcept
    {
        TensorRtCapabilities result;
        result.cudaDevice = cudaDevice;
        result.errorMessage = QStringLiteral("当前 PlaScan 构建未启用 TensorRT");
        return result;
    }

} // namespace xjw::inference

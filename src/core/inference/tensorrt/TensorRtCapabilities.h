#pragma once

#include <QString>

namespace xjw::inference
{

    struct TensorRtCapabilities
    {
        bool compiled = false;
        bool cudaAvailable = false;
        int deviceCount = 0;
        int cudaDevice = 0;
        bool fastFp16 = false;
        QString tensorRtVersion;
        QString gpuName;
        QString errorMessage;

        bool isAvailable() const
        {
            return compiled && cudaAvailable && errorMessage.isEmpty();
        }
    };

    TensorRtCapabilities queryTensorRtCapabilities(int cudaDevice = 0) noexcept;

} // namespace xjw::inference

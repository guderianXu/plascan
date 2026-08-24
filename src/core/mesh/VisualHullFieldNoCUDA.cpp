#include "VisualHullFieldBackend.h"

namespace xjw::mesh::detail
{

    bool cudaVisualHullFieldAvailable(int) noexcept
    {
        return false;
    }

    bool evaluateVisualHullFieldCuda(
        const VisualHullFieldDeviceInput&, int, std::vector<float>*, int* actualDeviceIndex, std::string* errorMessage)
    {
        if (actualDeviceIndex)
        {
            *actualDeviceIndex = -1;
        }
        if (errorMessage)
        {
            *errorMessage = "CUDA is unavailable in this build";
        }
        return false;
    }

} // namespace xjw::mesh::detail

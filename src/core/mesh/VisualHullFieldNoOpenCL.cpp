#include "VisualHullFieldBackend.h"

namespace xjw::mesh::detail
{

    bool openClVisualHullFieldAvailable(int) noexcept
    {
        return false;
    }

    bool evaluateVisualHullFieldOpenCl(
        const VisualHullFieldDeviceInput&, int, std::vector<float>*, int* actualDeviceIndex, std::string* errorMessage)
    {
        if (actualDeviceIndex)
        {
            *actualDeviceIndex = -1;
        }
        if (errorMessage)
        {
            *errorMessage = "OpenCL is unavailable in this build";
        }
        return false;
    }

} // namespace xjw::mesh::detail

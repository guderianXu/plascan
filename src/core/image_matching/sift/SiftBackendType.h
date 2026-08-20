#pragma once

namespace xjw::image_matching
{

    enum class SiftComputeBackend
    {
        Automatic,
        Cpu,
        Cuda,
        OpenCl,
        Metal
    };

} // namespace xjw::image_matching

#pragma once

#include "RegularGrid3D.h"
#include "VisualHullReconstructor.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh::detail
{

    inline constexpr int kVisualHullCameraParameterStride = 25;
    inline constexpr int kVisualHullViewMetadataStride = 6;
    inline constexpr int kVisualHullMaximumGpuSilhouetteViolations = 63;
    inline constexpr int kVisualHullDefaultGpuSlabDepth = 16;

    struct VisualHullFieldDeviceInput
    {
        RegularGrid3D grid;
        std::vector<float> cameraParameters;
        std::vector<std::int32_t> viewMetadata;
        std::vector<float> silhouetteSamples;
        std::vector<float> depthSamples;
        std::vector<float> gridCoordinates;
        int viewCount = 0;
        int minimumVisibleViews = 0;
        int allowedSilhouetteViolations = 0;
        bool continuousSilhouetteField = false;
        bool enableDepthFreeSpaceCarving = false;
        int minimumDepthFreeSpaceViolations = 0;
        float relativeDepthTolerance = 0.0f;
        bool closeVolumeBoundary = false;
        int gpuSlabDepth = kVisualHullDefaultGpuSlabDepth;
        std::function<bool()> isCancelled;
    };

    bool cudaVisualHullFieldAvailable(int deviceIndex) noexcept;
    bool evaluateVisualHullFieldCuda(const VisualHullFieldDeviceInput& input,
                                     int deviceIndex,
                                     std::vector<float>* field,
                                     int* actualDeviceIndex,
                                     std::string* errorMessage);

    bool openClVisualHullFieldAvailable(int deviceIndex) noexcept;
    bool evaluateVisualHullFieldOpenCl(const VisualHullFieldDeviceInput& input,
                                       int deviceIndex,
                                       std::vector<float>* field,
                                       int* actualDeviceIndex,
                                       std::string* errorMessage);

} // namespace xjw::mesh::detail

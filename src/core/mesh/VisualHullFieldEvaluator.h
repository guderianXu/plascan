#pragma once

#include "RegularGrid3D.h"
#include "VisualHullReconstructor.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace xjw::mesh::detail
{

    struct PreparedVisualHullView
    {
        const VisualHullView* view = nullptr;
        cv::Mat signedSilhouetteDistance;
    };

    std::vector<PreparedVisualHullView> prepareVisualHullFieldViews(const std::vector<VisualHullView>& views);

    float evaluateContinuousVisualHullField(float worldX,
                                            float worldY,
                                            float worldZ,
                                            const std::vector<PreparedVisualHullView>& views,
                                            const VisualHullConfig& config);

    bool evaluateBinaryVisualHullField(float worldX,
                                       float worldY,
                                       float worldZ,
                                       const std::vector<VisualHullView>& views,
                                       const VisualHullConfig& config);

    bool isVisualHullFieldBackendAvailable(VisualHullComputeBackend backend, int deviceIndex = -1) noexcept;

    bool evaluateVisualHullFieldGrid(const std::vector<VisualHullView>& views,
                                     const VisualHullConfig& config,
                                     const RegularGrid3D& grid,
                                     std::vector<float>* field,
                                     VisualHullComputeBackend* usedBackend = nullptr,
                                     std::string* errorMessage = nullptr,
                                     VisualHullExecutionInfo* executionInfo = nullptr);

} // namespace xjw::mesh::detail

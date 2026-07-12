#pragma once

#include "MeshTypes.h"
#include "PositiveDepthCameraModel.h"

#include <opencv2/core.hpp>

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct VisualHullView
{
    xjw::PositiveDepthCameraModel camera;
    cv::Mat silhouetteMask;
    cv::Mat depthMap;
    cv::Mat colorImage;
    std::string imagePath;
};

struct VisualHullConfig
{
    std::array<float, 3> boundsMin{-1.0f, -1.0f, -1.0f};
    std::array<float, 3> boundsMax{1.0f, 1.0f, 1.0f};
    int resolution = 96;
    int minimumVisibleViews = 4;
    int allowedSilhouetteViolations = 1;
    bool enableDepthFreeSpaceCarving = true;
    int minimumDepthFreeSpaceViolations = 2;
    float relativeDepthTolerance = 0.01f;
    int workerCount = 0;
    std::function<bool()> isCancelled;
    std::function<void(const std::string &, float)> progressFn;
};

class VisualHullReconstructor
{
public:
    static bool reconstruct(const std::vector<VisualHullView> &views,
                            const VisualHullConfig &config,
                            TriMesh *mesh,
                            std::string *errorMessage = nullptr);
};

} // namespace xjw::mesh

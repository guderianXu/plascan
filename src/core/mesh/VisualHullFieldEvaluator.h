#pragma once

#include "VisualHullReconstructor.h"

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::mesh::detail
{

struct PreparedVisualHullView
{
    const VisualHullView *view = nullptr;
    cv::Mat signedSilhouetteDistance;
};

std::vector<PreparedVisualHullView> prepareVisualHullFieldViews(
    const std::vector<VisualHullView> &views);

float evaluateContinuousVisualHullField(
    float worldX,
    float worldY,
    float worldZ,
    const std::vector<PreparedVisualHullView> &views,
    const VisualHullConfig &config);

} // namespace xjw::mesh::detail

#pragma once

#include "engine/TiePointGraph.h"
#include "pipeline/IncrementalSfm.h"
#include <placamera/frame_camera.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace xjw::aerial_triangulation::engine
{

    struct PinholeImage
    {
        ImageId id = kInvalidImageId;
        std::filesystem::path path;
        std::shared_ptr<const placamera::FramePinholeModel> camera;
        std::string sensorKey;
    };

    /// Numerical input contains resolved cameras and observations, never project JSON or UI objects.
    struct PinholeInput
    {
        IncrementalSfmOptions options;
        std::vector<PinholeImage> images;
        std::shared_ptr<const TiePointGraph> graph;
        std::vector<control_points::PriorTrack> priorTracks;
        std::vector<control_points::PriorScaleBar> scaleBars;
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        std::function<bool(int registered, int total, const std::string& message)> progressFn;
    };

    IncrementalSfmResult runPinhole(const PinholeInput& input);

} // namespace xjw::aerial_triangulation::engine

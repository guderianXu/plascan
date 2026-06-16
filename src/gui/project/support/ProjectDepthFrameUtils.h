#pragma once

#include "DepthFrameUtils.h"

namespace xjw::gui::project {

using StoredDepthFrameRecord = xjw::core::project::StoredDepthFrameRecord;
using StoredDepthFramesResult = xjw::core::project::StoredDepthFramesResult;
using FusionFrameBuildResult = xjw::core::project::FusionFrameBuildResult;
using xjw::core::project::rawDepthStoragePath;
using xjw::core::project::rawConfidenceStoragePath;
using xjw::core::project::depthFrameArtifactsExist;
using xjw::core::project::collectLatestStoredDepthFrames;
using xjw::core::project::buildStoredFusionFrame;

} // namespace xjw::gui::project

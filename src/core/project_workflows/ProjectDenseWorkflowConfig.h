#pragma once

// Legacy CLI include name. New code should include PointCloudWorkflowConfig.h.
#include "PointCloudWorkflowConfig.h"

namespace xjw::gui::project
{
using xjw::core::project::DenseGenerationSettings;
using xjw::core::project::DenseRefineSettings;
using xjw::core::project::DepthQualityProfile;
using xjw::core::project::buildDepthGenConfig;
using xjw::core::project::denseGenerationSettingsFromJson;
using xjw::core::project::depthQualityDownsample;
using xjw::core::project::depthQualityProfileFromId;
using xjw::core::project::depthQualityProfileId;
}

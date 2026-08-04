#pragma once

// Compatibility shim. Point-cloud workflow configuration is headless and lives
// in core/project_workflows; new code should include PointCloudWorkflowConfig.h.
#include "PointCloudWorkflowConfig.h"

namespace xjw::gui::project
{

using xjw::core::project::DepthQualityProfile;
using xjw::core::project::DenseGenerationSettings;
using xjw::core::project::DenseRefineSettings;
using xjw::core::project::buildDepthGenConfig;
using xjw::core::project::denseGenerationSettingsFromJson;
using xjw::core::project::denseRefineSettingsFromJson;
using xjw::core::project::depthQualityDownsample;
using xjw::core::project::depthQualityParameters;
using xjw::core::project::depthQualityProfileFromId;
using xjw::core::project::depthQualityProfileForModelQuality;
using xjw::core::project::depthQualityProfileId;
using xjw::core::project::depthQualityRank;

} // namespace xjw::gui::project

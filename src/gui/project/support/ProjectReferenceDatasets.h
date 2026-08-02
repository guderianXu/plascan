#pragma once

// Compatibility include for code that still uses the former GUI path.
#include "ReferenceDatasetWorkflow.h"

namespace xjw::gui::project
{
using xjw::core::project::ReferenceDatasetQualityReportResult;
using xjw::core::project::normalizeReferenceDatasetType;
using xjw::core::project::referenceDatasetTypeForPath;
using xjw::core::project::registerReferenceDataset;
using xjw::core::project::writeReferenceDatasetQualityReport;
using xjw::core::project::writeReferenceTerrainPriorPreflightReport;
}

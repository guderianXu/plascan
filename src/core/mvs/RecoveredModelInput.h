#pragma once

#include "metmodel/patchmatch.hpp"

#include <QString>

namespace xjw::mvs
{
    // Versioned, lossless model input. The depths are voting outputs, not PM
    // snapshots and not the public d4 image composed from multiple levels.
    void writeRecoveredModelInput(const QString& directory,
                                  const metmodel::Scene& scene,
                                  const metmodel::RecoveredPatchMatchD4SceneOutput& depth,
                                  bool replaceExisting = false);
    void readRecoveredModelInput(const QString& directory,
                                 metmodel::Scene& scene,
                                 metmodel::RecoveredPatchMatchD4SceneOutput& depth);
} // namespace xjw::mvs

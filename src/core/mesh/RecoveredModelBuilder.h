#pragma once

#include "MeshTypes.h"

#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace xjw::mesh
{
    struct RecoveredModelResult
    {
        TriMesh mesh;
        // Preserve the reference world-coordinate precision at the product boundary.
        // TriMesh remains the existing float representation for GUI consumers.
        std::vector<std::array<double, 3>> precisePositions;
        QJsonObject diagnostics;
    };

    // Produces the reference two-level cadence while ending at the actual
    // balanced-tree maximum. Small trees use one explicit terminal stage.
    std::vector<std::uint32_t> planRecoveredSupportLevels(std::uint32_t maximumLevel,
                                                          const QJsonValue& requestedLevels = {});

    // Throws on unsupported settings, damaged inputs or cancellation. No legacy
    // surface is substituted after the recovered pipeline has been selected.
    RecoveredModelResult buildRecoveredModel(const QString& inputDirectory,
                                             const QJsonObject& settings,
                                             int targetFaces,
                                             const std::function<bool()>& isCancelled,
                                             const std::function<void(const QString&, int)>& progress);

    bool writeRecoveredModelPly(const RecoveredModelResult& model, const QString& path, std::string* error);
} // namespace xjw::mesh

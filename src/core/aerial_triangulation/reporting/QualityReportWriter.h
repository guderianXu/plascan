#pragma once

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"

#include <QJsonArray>
#include <QJsonObject>

namespace xjw
{
class SfmReconstruction;
}

namespace xjw::aerial_triangulation
{

struct SparseQualityReport
{
    QJsonArray points;
    QJsonArray perCameraResiduals;
    QJsonObject qualityMetadata;
    QJsonObject diagnostics;
};

class QualityReportWriter
{
public:
    static SparseQualityReport build(
        const PreparedAerialTriangulationInput &input,
        const xjw::SfmReconstruction &reconstruction,
        const AerialTriangulationReconstructionResult &result);
};

} // namespace xjw::aerial_triangulation

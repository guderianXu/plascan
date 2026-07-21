#pragma once

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"
#include "reconstruction/SfmAttemptRunner.h"

#include <functional>

namespace xjw::aerial_triangulation
{

class AerialTriangulationPipeline
{
public:
    using AttemptRunner = std::function<SfmAttemptExecutionResult(
        const PreparedAerialTriangulationInput &input)>;
    using ResultWriter = std::function<bool(
        const PreparedAerialTriangulationInput &input,
        SfmAttemptExecutionResult *execution,
        QString *errorMessage)>;

    explicit AerialTriangulationPipeline(AttemptRunner attemptRunner = {},
                                         ResultWriter resultWriter = {});

    AerialTriangulationReconstructionResult run(
        const PreparedAerialTriangulationInput &input) const;

private:
    AttemptRunner _attemptRunner;
    ResultWriter _resultWriter;
};

} // namespace xjw::aerial_triangulation

#pragma once

#include "model/AerialTriangulationOptions.h"
#include "reconstruction/SfmAttemptRunner.h"

#include <QString>

namespace xjw::aerial_triangulation
{

class AerialTriangulationResultWriter
{
public:
    bool write(const PreparedAerialTriangulationInput &input,
               SfmAttemptExecutionResult *execution,
               QString *errorMessage = nullptr) const;
};

} // namespace xjw::aerial_triangulation

#pragma once

#include "placamera/linescan_numeric_state.h"

namespace placamera::internal
{

    EvaluationResult<LineScanProjectionDetails> projectLineScanNumericAtLine(const LineScanNumericState& state,
                                                                             const GroundCoordinate& ground,
                                                                             double line,
                                                                             const EvaluationOptions& options);

    EvaluationResult<Projection> projectLineScanNumeric(const LineScanNumericState& state,
                                                        const GroundCoordinate& ground,
                                                        const EvaluationOptions& options);

    EvaluationResult<ImagingLocus> lineScanNumericImagingLocus(const LineScanNumericState& state,
                                                               const ImageCoordinate& image,
                                                               const EvaluationOptions& options);

} // namespace placamera::internal

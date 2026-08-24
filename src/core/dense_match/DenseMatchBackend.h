#pragma once

#include "DenseMatchTypes.h"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace xjw::dense_match
{

    struct DenseMatchConfig;

    [[nodiscard]] const char* denseMatchComputeBackendName(DenseMatchComputeBackend backend);

    // Accepts auto/automatic, cpu, cuda and opencl (case-insensitive).  Invalid
    // names raise std::invalid_argument so configuration mistakes are explicit.
    [[nodiscard]] DenseMatchComputeBackend parseDenseMatchComputeBackend(std::string_view name);

    [[nodiscard]] bool isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend backend, int deviceIndex = 0);

    // Automatic selection is ordered CUDA -> OpenCL -> CPU.  Explicit requests
    // raise std::runtime_error when the backend was not built or the requested
    // device does not exist.
    [[nodiscard]] DenseMatchComputeBackend
    resolveDenseMatchComputeBackend(DenseMatchComputeBackend requested, int cudaDevice = 0, int openClDevice = 0);

    // Resolves the new enum while honoring the legacy useCuda=false CPU override.
    [[nodiscard]] DenseMatchComputeBackend resolveDenseMatchComputeBackend(const DenseMatchConfig& config);

    namespace detail
    {

        using DenseMatchBackendAttempt =
            std::function<bool(DenseMatchComputeBackend backend, std::string* errorMessage)>;

        // Shared retry coordinator kept separate from the algorithm callback so
        // runtime fallback order and reporting can be tested deterministically.
        bool runDenseMatchBackendAttempts(DenseMatchComputeBackend requested,
                                          const std::vector<DenseMatchComputeBackend>& candidates,
                                          std::string initialFallbackReason,
                                          const DenseMatchBackendAttempt& attempt,
                                          DenseMatchExecutionReport* report,
                                          std::string* errorMessage);

    } // namespace detail

} // namespace xjw::dense_match

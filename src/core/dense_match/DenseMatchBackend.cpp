#include "DenseMatchBackend.h"

#include "CostFunctions.h"
#include "DenseMatchConfig.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace xjw::dense_match
{

    const char* denseMatchComputeBackendName(DenseMatchComputeBackend backend)
    {
        switch (backend)
        {
        case DenseMatchComputeBackend::Automatic:
            return "auto";
        case DenseMatchComputeBackend::Cpu:
            return "cpu";
        case DenseMatchComputeBackend::Cuda:
            return "cuda";
        case DenseMatchComputeBackend::OpenCl:
            return "opencl";
        }
        return "unknown";
    }

    DenseMatchComputeBackend parseDenseMatchComputeBackend(std::string_view name)
    {
        const auto first =
            std::find_if_not(name.begin(), name.end(), [](unsigned char value) { return std::isspace(value) != 0; });
        const auto last =
            std::find_if_not(name.rbegin(), name.rend(), [](unsigned char value) { return std::isspace(value) != 0; })
                .base();

        std::string normalized;
        if (first < last)
        {
            normalized.assign(first, last);
        }
        std::transform(normalized.begin(),
                       normalized.end(),
                       normalized.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });

        if (normalized == "auto" || normalized == "automatic")
        {
            return DenseMatchComputeBackend::Automatic;
        }
        if (normalized == "cpu")
        {
            return DenseMatchComputeBackend::Cpu;
        }
        if (normalized == "cuda")
        {
            return DenseMatchComputeBackend::Cuda;
        }
        if (normalized == "opencl" || normalized == "open_cl")
        {
            return DenseMatchComputeBackend::OpenCl;
        }

        throw std::invalid_argument("Unknown dense-match compute backend: " + std::string(name));
    }

    bool isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend backend, int deviceIndex)
    {
        switch (backend)
        {
        case DenseMatchComputeBackend::Automatic:
        case DenseMatchComputeBackend::Cpu:
            return true;
        case DenseMatchComputeBackend::Cuda:
#ifdef DM_ENABLE_CUDA
            return isCostVolumeCUDAAvailable(deviceIndex);
#else
            return false;
#endif
        case DenseMatchComputeBackend::OpenCl:
#ifdef DM_ENABLE_OPENCL
            return isCostVolumeOpenCLAvailable(deviceIndex);
#else
            return false;
#endif
        }
        return false;
    }

    DenseMatchComputeBackend
    resolveDenseMatchComputeBackend(DenseMatchComputeBackend requested, int cudaDevice, int openClDevice)
    {
        if (requested != DenseMatchComputeBackend::Automatic)
        {
            const int deviceIndex = requested == DenseMatchComputeBackend::Cuda     ? cudaDevice
                                    : requested == DenseMatchComputeBackend::OpenCl ? openClDevice
                                                                                    : 0;
            if (!isDenseMatchComputeBackendAvailable(requested, deviceIndex))
            {
                throw std::runtime_error(std::string("Requested dense-match backend is unavailable: ") +
                                         denseMatchComputeBackendName(requested) +
                                         (requested == DenseMatchComputeBackend::Cpu
                                              ? std::string()
                                              : " device " + std::to_string(deviceIndex)));
            }
            return requested;
        }

        if (isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::Cuda, cudaDevice))
        {
            return DenseMatchComputeBackend::Cuda;
        }
        if (isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::OpenCl, openClDevice))
        {
            return DenseMatchComputeBackend::OpenCl;
        }
        return DenseMatchComputeBackend::Cpu;
    }

    DenseMatchComputeBackend resolveDenseMatchComputeBackend(const DenseMatchConfig& config)
    {
        if (config.computeBackend == DenseMatchComputeBackend::Automatic && !config.useCuda)
        {
            return DenseMatchComputeBackend::Cpu;
        }
        return resolveDenseMatchComputeBackend(config.computeBackend, config.cudaDevice, config.openClDevice);
    }

    namespace detail
    {

        bool runDenseMatchBackendAttempts(DenseMatchComputeBackend requested,
                                          const std::vector<DenseMatchComputeBackend>& candidates,
                                          std::string initialFallbackReason,
                                          const DenseMatchBackendAttempt& attempt,
                                          DenseMatchExecutionReport* report,
                                          std::string* errorMessage)
        {
            DenseMatchExecutionReport local_report;
            DenseMatchExecutionReport& execution = report != nullptr ? *report : local_report;
            execution.requestedBackend = requested;
            std::string fallback_reason = std::move(initialFallbackReason);

            for (const DenseMatchComputeBackend backend : candidates)
            {
                execution.actualBackend = backend;
                execution.workSubmitted = true;

                std::string attempt_error;
                if (attempt(backend, &attempt_error))
                {
                    execution.fallbackUsed = !fallback_reason.empty();
                    execution.fallbackReason = std::move(fallback_reason);
                    if (errorMessage != nullptr)
                    {
                        errorMessage->clear();
                    }
                    return true;
                }

                if (attempt_error.empty())
                {
                    attempt_error = std::string(denseMatchComputeBackendName(backend)) + " backend failed";
                }

                if (requested != DenseMatchComputeBackend::Automatic)
                {
                    execution.fallbackUsed = false;
                    execution.fallbackReason.clear();
                    if (errorMessage != nullptr)
                    {
                        *errorMessage = std::move(attempt_error);
                    }
                    return false;
                }

                if (backend == DenseMatchComputeBackend::Cpu)
                {
                    execution.fallbackUsed = !fallback_reason.empty();
                    execution.fallbackReason = fallback_reason;
                    if (errorMessage != nullptr)
                    {
                        *errorMessage = std::move(attempt_error);
                    }
                    return false;
                }

                if (!fallback_reason.empty())
                {
                    fallback_reason += "; ";
                }
                fallback_reason += attempt_error;
            }

            execution.fallbackUsed = !fallback_reason.empty();
            execution.fallbackReason = std::move(fallback_reason);
            if (errorMessage != nullptr)
            {
                *errorMessage = "Dense matching has no compute backend to try";
            }
            return false;
        }

    } // namespace detail

} // namespace xjw::dense_match

#pragma once

#include "CameraCapabilities.h"
#include "../model/CameraInstance.h"

#include <string>
#include <vector>

namespace xjw::camera_core
{

    class CapabilityCheckResult
    {
    public:
        static CapabilityCheckResult success();
        static CapabilityCheckResult failure(const CameraInstance& instance, std::vector<CapabilityKind> missing);

        bool ok() const noexcept;
        const std::vector<CapabilityKind>& missing() const noexcept;
        const std::string& message() const noexcept;

    private:
        CapabilityCheckResult(bool ok, std::vector<CapabilityKind> missing, std::string message);

        bool _ok;
        std::vector<CapabilityKind> _missing;
        std::string _message;
    };

    CapabilityCheckResult requireCapabilities(const CameraInstance& instance, const CapabilitySet& required);

} // namespace xjw::camera_core

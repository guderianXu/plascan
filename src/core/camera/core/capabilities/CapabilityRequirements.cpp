#include "CapabilityRequirements.h"

#include <sstream>
#include <utility>

namespace xjw::camera_core
{

    CapabilityCheckResult CapabilityCheckResult::success()
    {
        return CapabilityCheckResult(true, {}, {});
    }

    CapabilityCheckResult CapabilityCheckResult::failure(const CameraInstance& instance,
                                                         std::vector<CapabilityKind> missing)
    {
        std::ostringstream stream;
        stream << "camera instance " << instance.instanceId().value() << " for image " << instance.imageId().value()
               << " (model " << instance.definition().modelType() << ", definition "
               << instance.definition().definitionId().value() << ") is missing capabilities:";
        for (const CapabilityKind capability : missing)
        {
            stream << ' ' << capabilityName(capability);
        }
        return CapabilityCheckResult(false, std::move(missing), stream.str());
    }

    CapabilityCheckResult::CapabilityCheckResult(bool ok, std::vector<CapabilityKind> missing, std::string message)
        : _ok(ok), _missing(std::move(missing)), _message(std::move(message))
    {
    }

    bool CapabilityCheckResult::ok() const noexcept
    {
        return _ok;
    }

    const std::vector<CapabilityKind>& CapabilityCheckResult::missing() const noexcept
    {
        return _missing;
    }

    const std::string& CapabilityCheckResult::message() const noexcept
    {
        return _message;
    }

    CapabilityCheckResult requireCapabilities(const CameraInstance& instance, const CapabilitySet& required)
    {
        std::vector<CapabilityKind> missing;
        for (const CapabilityKind capability : required.values())
        {
            if (!instance.capabilities().contains(capability) &&
                !instance.definition().capabilities().contains(capability))
            {
                missing.push_back(capability);
            }
        }
        if (missing.empty())
        {
            return CapabilityCheckResult::success();
        }
        return CapabilityCheckResult::failure(instance, std::move(missing));
    }

} // namespace xjw::camera_core

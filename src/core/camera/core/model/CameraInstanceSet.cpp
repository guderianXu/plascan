#include "CameraInstanceSet.h"

#include <unordered_set>
#include <utility>

namespace xjw::camera_core
{

    bool CameraInstanceSet::add(std::shared_ptr<const CameraInstance> instance, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!instance)
        {
            if (error)
            {
                *error = "camera instance set cannot contain a null instance";
            }
            return false;
        }
        for (const std::shared_ptr<const CameraInstance>& existing : _instances)
        {
            if (existing->imageId() == instance->imageId())
            {
                if (error)
                {
                    *error = "image has more than one camera instance: " + instance->imageId().value();
                }
                return false;
            }
        }
        _instances.push_back(std::move(instance));
        return true;
    }

    CameraInstanceLookupResult CameraInstanceSet::forImage(const ImageId& image) const
    {
        for (const std::shared_ptr<const CameraInstance>& instance : _instances)
        {
            if (instance->imageId() == image)
            {
                return CameraInstanceLookupResult{instance, {}};
            }
        }
        return CameraInstanceLookupResult{{}, "camera instance is not bound to image: " + image.value()};
    }

    CameraInstanceSet CameraInstanceSet::select(const std::vector<ImageId>& images, std::string* error) const
    {
        if (error)
        {
            error->clear();
        }

        CameraInstanceSet selected;
        if (images.empty())
        {
            if (error)
            {
                *error = "camera instance selection cannot be empty";
            }
            return selected;
        }

        std::unordered_set<ImageId> seen;
        seen.reserve(images.size());
        for (const ImageId& image : images)
        {
            if (!seen.insert(image).second)
            {
                if (error)
                {
                    *error = "camera instance selection contains duplicate image: " + image.value();
                }
                return CameraInstanceSet{};
            }

            const CameraInstanceLookupResult lookup = forImage(image);
            if (!lookup.ok())
            {
                if (error)
                {
                    *error = lookup.error;
                }
                return CameraInstanceSet{};
            }

            std::string addError;
            if (!selected.add(lookup.instance, &addError))
            {
                if (error)
                {
                    *error = addError;
                }
                return CameraInstanceSet{};
            }
        }
        return selected;
    }

    CameraInstanceSetCapabilityResult CameraInstanceSet::requireCapabilities(const CapabilitySet& required) const
    {
        CameraInstanceSetCapabilityResult result;
        for (const std::shared_ptr<const CameraInstance>& instance : _instances)
        {
            const CapabilityCheckResult check = xjw::camera_core::requireCapabilities(*instance, required);
            if (!check.ok())
            {
                result.failures.push_back(
                    CameraInstanceSetCapabilityResult::Failure{instance->imageId(), check.message()});
            }
        }
        return result;
    }

    CameraInstanceSetFrameResult CameraInstanceSet::requireCommonWorldFrame() const
    {
        CameraInstanceSetFrameResult result;
        if (_instances.empty())
        {
            return result;
        }

        const std::shared_ptr<const CameraInstance>& first = _instances.front();
        result.commonFrame = first->definition().worldFrame();
        const ImageId& firstImage = first->imageId();
        for (std::size_t index = 1; index < _instances.size(); ++index)
        {
            const std::shared_ptr<const CameraInstance>& instance = _instances[index];
            const xjw::coordinate_system::CoordinateFrameId& observedFrame = instance->definition().worldFrame();
            if (observedFrame == *result.commonFrame)
            {
                continue;
            }

            result.failures.push_back(CameraInstanceSetFrameResult::Failure{
                instance->imageId(),
                observedFrame,
                "world frame '" + observedFrame.value() + "' for image '" + instance->imageId().value() +
                    "' differs from image '" + firstImage.value() + "' frame '" + result.commonFrame->value() + "'"});
        }
        return result;
    }

    const std::vector<std::shared_ptr<const CameraInstance>>& CameraInstanceSet::values() const noexcept
    {
        return _instances;
    }

} // namespace xjw::camera_core

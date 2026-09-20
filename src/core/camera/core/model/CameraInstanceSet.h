#pragma once

#include "../capabilities/CapabilityRequirements.h"
#include "CameraInstance.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace xjw::camera_core
{

    struct CameraInstanceLookupResult
    {
        std::shared_ptr<const CameraInstance> instance;
        std::string error;

        bool ok() const noexcept
        {
            return static_cast<bool>(instance) && error.empty();
        }
    };

    struct CameraInstanceSetCapabilityResult
    {
        struct Failure
        {
            ImageId image;
            std::string message;
        };

        std::vector<Failure> failures;

        bool ok() const noexcept
        {
            return failures.empty();
        }
    };

    struct CameraInstanceSetFrameResult
    {
        struct Failure
        {
            ImageId image;
            xjw::coordinate_system::CoordinateFrameId frame;
            std::string reason;
        };

        std::optional<xjw::coordinate_system::CoordinateFrameId> commonFrame;
        std::vector<Failure> failures;

        bool ok() const noexcept
        {
            return failures.empty();
        }
    };

    /**
     * A validated collection of image-bound camera instances.
     *
     * The set is deliberately model-agnostic.  Processing stages look up an
     * image by its stable ImageId and then request the capabilities they need;
     * they never infer a concrete camera model from an array position.
     */
    class CameraInstanceSet
    {
    public:
        bool add(std::shared_ptr<const CameraInstance> instance, std::string* error = nullptr);

        CameraInstanceLookupResult forImage(const ImageId& image) const;

        /**
         * Select a non-empty, ordered subset by canonical ImageId.
         *
         * Selection is identity based and deliberately rejects missing or
         * duplicate ids.  A failed selection returns an empty set and a
         * diagnostic; callers must not continue with the partial subset.
         */
        CameraInstanceSet select(const std::vector<ImageId>& images, std::string* error = nullptr) const;

        CameraInstanceSetCapabilityResult requireCapabilities(const CapabilitySet& required) const;

        CameraInstanceSetFrameResult requireCommonWorldFrame() const;

        const std::vector<std::shared_ptr<const CameraInstance>>& values() const noexcept;

        bool empty() const noexcept
        {
            return _instances.empty();
        }

        std::size_t size() const noexcept
        {
            return _instances.size();
        }

    private:
        std::vector<std::shared_ptr<const CameraInstance>> _instances;
    };

} // namespace xjw::camera_core

#pragma once

#include "placamera/model.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace placamera
{

    struct CameraInstanceSetFailure
    {
        ImageId imageId;
        CameraErrorCode errorCode = CameraErrorCode::None;
        std::string message;
    };

    struct CameraInstanceSetValidation
    {
        std::optional<FrameId> commonGroundFrame;
        std::vector<CameraInstanceSetFailure> failures;

        bool ok() const noexcept
        {
            return failures.empty();
        }
    };

    class CameraInstanceSet
    {
    public:
        using ModelPointer = RasterModelPtr;

        Result<std::size_t> add(ModelPointer model);
        Result<ModelPointer> forImage(const ImageId& imageId) const;
        Result<CameraInstanceSet> select(const std::vector<ImageId>& imageIds) const;

        CameraInstanceSetValidation requireCapabilities(const CapabilitySet& required) const;
        CameraInstanceSetValidation requireCommonGroundFrame() const;

        const std::vector<ModelPointer>& values() const noexcept;
        bool empty() const noexcept;
        std::size_t size() const noexcept;

    private:
        std::vector<ModelPointer> _models;
    };

} // namespace placamera

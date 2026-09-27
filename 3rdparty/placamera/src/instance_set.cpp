#include "placamera/instance_set.h"

#include <algorithm>

namespace placamera
{

    Result<std::size_t> CameraInstanceSet::add(ModelPointer model)
    {
        if (!model)
        {
            return Result<std::size_t>::failure(CameraErrorCode::InvalidArgument,
                                                "camera instance set cannot add a null model");
        }
        const auto duplicate = std::find_if(_models.begin(),
                                            _models.end(),
                                            [&model](const ModelPointer& candidate) {
                                                return candidate->imageId() == model->imageId() ||
                                                       candidate->instanceId() == model->instanceId();
                                            });
        if (duplicate != _models.end())
        {
            return Result<std::size_t>::failure(
                CameraErrorCode::InvalidModelState,
                "camera instance set already contains the image or instance identifier");
        }
        _models.push_back(std::move(model));
        return Result<std::size_t>::success(_models.size() - 1);
    }

    Result<CameraInstanceSet::ModelPointer> CameraInstanceSet::forImage(const ImageId& imageId) const
    {
        const auto found = std::find_if(_models.begin(),
                                        _models.end(),
                                        [&imageId](const ModelPointer& model) { return model->imageId() == imageId; });
        if (found == _models.end())
        {
            return Result<ModelPointer>::failure(CameraErrorCode::InvalidArgument,
                                                 "camera instance set does not contain the requested image");
        }
        return Result<ModelPointer>::success(*found);
    }

    Result<CameraInstanceSet> CameraInstanceSet::select(const std::vector<ImageId>& imageIds) const
    {
        if (imageIds.empty())
        {
            return Result<CameraInstanceSet>::failure(CameraErrorCode::InvalidArgument,
                                                      "camera instance selection must not be empty");
        }
        CameraInstanceSet selected;
        for (const ImageId& image_id : imageIds)
        {
            const auto model = forImage(image_id);
            if (!model)
            {
                return Result<CameraInstanceSet>::failure(model.error());
            }
            const auto added = selected.add(model.value());
            if (!added)
            {
                return Result<CameraInstanceSet>::failure(
                    CameraErrorCode::InvalidArgument,
                    "camera instance selection contains a duplicate image identifier");
            }
        }
        return Result<CameraInstanceSet>::success(std::move(selected));
    }

    CameraInstanceSetValidation CameraInstanceSet::requireCapabilities(const CapabilitySet& required) const
    {
        CameraInstanceSetValidation validation;
        for (const ModelPointer& model : _models)
        {
            const std::uint32_t missing = required.bits() & ~model->capabilities().bits();
            if (missing != 0U)
            {
                validation.failures.push_back({model->imageId(),
                                               CameraErrorCode::UnsupportedModel,
                                               "camera model does not provide all required capabilities"});
            }
        }
        return validation;
    }

    CameraInstanceSetValidation CameraInstanceSet::requireCommonGroundFrame() const
    {
        CameraInstanceSetValidation validation;
        if (_models.empty())
        {
            return validation;
        }
        validation.commonGroundFrame = _models.front()->groundFrame();
        for (const ModelPointer& model : _models)
        {
            if (model->groundFrame() != *validation.commonGroundFrame)
            {
                validation.failures.push_back({model->imageId(),
                                               CameraErrorCode::FrameMismatch,
                                               "camera model ground frame differs from the instance-set frame"});
            }
        }
        if (!validation.failures.empty())
        {
            validation.commonGroundFrame.reset();
        }
        return validation;
    }

    const std::vector<CameraInstanceSet::ModelPointer>& CameraInstanceSet::values() const noexcept
    {
        return _models;
    }

    bool CameraInstanceSet::empty() const noexcept
    {
        return _models.empty();
    }

    std::size_t CameraInstanceSet::size() const noexcept
    {
        return _models.size();
    }

} // namespace placamera

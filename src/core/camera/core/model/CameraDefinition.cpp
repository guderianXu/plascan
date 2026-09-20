#include "CameraDefinition.h"

#include "../types/CameraErrors.h"

#include <utility>

namespace xjw::camera_core
{

    CameraDefinition::CameraDefinition(CameraDefinitionId id,
                                       std::string modelType,
                                       xjw::coordinate_system::CoordinateFrameId worldFrame,
                                       int parameterSchemaVersion,
                                       CapabilitySet capabilities)
        : _definitionId(std::move(id)), _modelType(std::move(modelType)), _worldFrame(std::move(worldFrame)),
          _parameterSchemaVersion(parameterSchemaVersion), _capabilities(std::move(capabilities))
    {
        if (_modelType.empty())
        {
            throw CameraValidationError(CameraErrorCode::InvalidFrame,
                                        "camera definition model type must not be empty");
        }
        if (_parameterSchemaVersion <= 0)
        {
            throw CameraValidationError(CameraErrorCode::InvalidFrame,
                                        "camera definition schema version must be positive");
        }
    }

    const CameraDefinitionId& CameraDefinition::definitionId() const noexcept
    {
        return _definitionId;
    }

    std::string_view CameraDefinition::modelType() const noexcept
    {
        return _modelType;
    }

    const xjw::coordinate_system::CoordinateFrameId& CameraDefinition::worldFrame() const noexcept
    {
        return _worldFrame;
    }

    int CameraDefinition::parameterSchemaVersion() const noexcept
    {
        return _parameterSchemaVersion;
    }

    const CapabilitySet& CameraDefinition::capabilities() const noexcept
    {
        return _capabilities;
    }

} // namespace xjw::camera_core

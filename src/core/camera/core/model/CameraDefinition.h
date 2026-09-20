#pragma once

#include "../capabilities/CameraCapabilities.h"
#include "../types/CameraIds.h"
#include "coordinate_system/types/CoordinateIds.h"

#include <string>
#include <string_view>

namespace xjw::camera_core
{

    class CameraDefinition
    {
    public:
        CameraDefinition(CameraDefinitionId id,
                         std::string modelType,
                         xjw::coordinate_system::CoordinateFrameId worldFrame,
                         int parameterSchemaVersion,
                         CapabilitySet capabilities);
        virtual ~CameraDefinition() = default;

        const CameraDefinitionId& definitionId() const noexcept;
        std::string_view modelType() const noexcept;
        const xjw::coordinate_system::CoordinateFrameId& worldFrame() const noexcept;
        int parameterSchemaVersion() const noexcept;
        const CapabilitySet& capabilities() const noexcept;

    private:
        CameraDefinitionId _definitionId;
        std::string _modelType;
        xjw::coordinate_system::CoordinateFrameId _worldFrame;
        int _parameterSchemaVersion = 0;
        CapabilitySet _capabilities;
    };

} // namespace xjw::camera_core

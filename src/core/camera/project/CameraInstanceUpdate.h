#pragma once

#include "camera/core/types/CameraIds.h"
#include "coordinate_system/types/CoordinateIds.h"

#include <QJsonObject>

#include <vector>

namespace xjw::camera_project
{

    /**
     * A solver result addressed by the canonical image and instance identity.
     *
     * The metadata is still represented as JSON at the project persistence edge,
     * but the key is a stable ImageId.  File paths are deliberately absent: they
     * are locators owned by import/UI code and must not become solver identity.
     * The instance and world-frame fields capture the input snapshot and are
     * checked again before project writeback.
     */
    struct CameraInstanceUpdate
    {
        camera_core::ImageId imageId;
        camera_core::CameraInstanceId instanceId;
        xjw::coordinate_system::CoordinateFrameId worldFrame;
        QJsonObject modelMetadata;
    };

    using CameraImageIds = std::vector<camera_core::ImageId>;
    using CameraInstanceUpdates = std::vector<CameraInstanceUpdate>;

} // namespace xjw::camera_project

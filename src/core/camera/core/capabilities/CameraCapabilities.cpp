#include "CameraCapabilities.h"

namespace xjw::camera_core
{

    const char* capabilityName(CapabilityKind capability) noexcept
    {
        switch (capability)
        {
        case CapabilityKind::Projection:
            return "projection";
        case CapabilityKind::InverseProjection:
            return "inverse_projection";
        case CapabilityKind::Ray:
            return "ray";
        case CapabilityKind::StaticPose:
            return "static_pose";
        case CapabilityKind::Trajectory:
            return "trajectory";
        case CapabilityKind::ImageCorrection:
            return "image_correction";
        case CapabilityKind::Optimization:
            return "optimization";
        }
        return "unknown";
    }

} // namespace xjw::camera_core

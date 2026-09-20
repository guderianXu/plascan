#include "BundleAdjustProjection.h"

namespace xjw::ba
{

    ProjectionCamera makeProjectionCamera(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera)
    {
        ProjectionCamera out;
        const xjw::camera_models::frame_pinhole::FramePinholeNumericState::Intrinsics intrinsics = camera.intrinsics();
        const xjw::camera_models::frame_pinhole::FramePinholeNumericState::Distortion distortion = camera.distortion();
        out.cameraToWorldRotation = camera.cameraToWorldRotation();
        out.cameraCenter = camera.cameraCenter();
        out.focalX = intrinsics.focalX;
        out.focalY = intrinsics.focalY;
        out.principalX = intrinsics.principalX;
        out.principalY = intrinsics.principalY;
        out.radialK1 = distortion.radialK1;
        out.radialK2 = distortion.radialK2;
        out.radialK3 = distortion.radialK3;
        out.tangentialP1 = distortion.tangentialP1;
        out.tangentialP2 = distortion.tangentialP2;
        out.uAxisSign = intrinsics.uAxisSign;
        out.vAxisSign = intrinsics.vAxisSign;
        out.depthAxisFlipped = camera.depthAxisFlipped();
        return out;
}

} // namespace xjw::ba

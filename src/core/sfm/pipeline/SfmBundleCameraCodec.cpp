#include "SfmBundleCameraCodec.h"

#include <exception>
#include <utility>

namespace xjw::sfm_bundle_camera
{

    namespace
    {

        bool fail(std::string* error, const char* message)
        {
            if (error)
            {
                *error = message;
            }
            return false;
        }

    } // namespace

    bool encode(const placamera::FramePinholeNumericState& source, plabundle::FrameCamera* target, std::string* error)
    {
        if (!target)
        {
            return fail(error, "PlaBundle camera output is null");
        }
        if (source.pixelConvention() != placamera::PixelConvention::PixelCenter)
        {
            return fail(error, "PlaBundle requires pixel-center frame calibration");
        }
        const auto& intrinsics = source.intrinsics();
        const auto& distortion = source.distortion();
        plabundle::FrameCamera camera;
        camera.cameraToWorldRotation = source.pose().cameraToWorldRotation;
        camera.cameraCenter = source.pose().center;
        camera.focalXPixels = intrinsics.focalX;
        camera.focalYPixels = intrinsics.focalY;
        camera.principalXPixel = intrinsics.principalX;
        camera.principalYPixel = intrinsics.principalY;
        camera.pixelPitchMillimeters = intrinsics.pixelPitch;
        camera.distortion = {distortion.radialK1,
                             distortion.radialK2,
                             distortion.radialK3,
                             distortion.tangentialP1,
                             distortion.tangentialP2};
        camera.uAxisSign = intrinsics.uAxisSign;
        camera.vAxisSign = intrinsics.vAxisSign;
        camera.depthAxisFlipped = source.depthAxisFlipped();
        camera.imageSize = plabundle::ImageSize{source.imageSize().samples, source.imageSize().lines};
        if (!plabundle::validateFrameCamera(camera, error))
        {
            return false;
        }
        *target = std::move(camera);
        return true;
    }

    bool decode(const plabundle::FrameCamera& source, placamera::FramePinholeNumericState* target, std::string* error)
    {
        if (!target)
        {
            return fail(error, "PlaCamera result target is null");
        }
        if (!plabundle::validateFrameCamera(source, error))
        {
            return false;
        }
        const auto& image_size = target->imageSize();
        if (!source.imageSize || source.imageSize->samples != image_size.samples ||
            source.imageSize->lines != image_size.lines || source.depthAxisFlipped != target->depthAxisFlipped())
        {
            return fail(error, "PlaBundle changed the bound image grid or optical-axis convention");
        }

        auto candidate = *target;
        auto intrinsics = candidate.intrinsics();
        intrinsics.focalX = source.focalXPixels;
        intrinsics.focalY = source.focalYPixels;
        intrinsics.principalX = source.principalXPixel;
        intrinsics.principalY = source.principalYPixel;
        intrinsics.pixelPitch = source.pixelPitchMillimeters;
        intrinsics.uAxisSign = source.uAxisSign;
        intrinsics.vAxisSign = source.vAxisSign;
        const placamera::BrownConradyDistortion distortion{source.distortion.k1,
                                                           source.distortion.k2,
                                                           source.distortion.k3,
                                                           source.distortion.p1,
                                                           source.distortion.p2};
        try
        {
            candidate.setPose(
                placamera::Pose::create(candidate.groundFrame(), source.cameraCenter, source.cameraToWorldRotation));
            candidate.setIntrinsics(intrinsics);
            candidate.setDistortion(distortion);
        }
        catch (const std::exception& exception)
        {
            if (error)
            {
                *error = exception.what();
            }
            return false;
        }
        *target = std::move(candidate);
        return true;
    }

    bool encodeAll(const std::vector<placamera::FramePinholeNumericState>& sources,
                   std::vector<plabundle::FrameCamera>* targets,
                   std::string* error)
    {
        if (!targets)
        {
            return fail(error, "PlaBundle camera list output is null");
        }
        std::vector<plabundle::FrameCamera> converted;
        converted.reserve(sources.size());
        for (const auto& source : sources)
        {
            plabundle::FrameCamera camera;
            if (!encode(source, &camera, error))
            {
                return false;
            }
            converted.push_back(std::move(camera));
        }
        *targets = std::move(converted);
        return true;
    }

    bool decodeAll(const std::vector<plabundle::FrameCamera>& sources,
                   std::vector<placamera::FramePinholeNumericState>* targets,
                   std::string* error)
    {
        if (!targets)
        {
            return fail(error, "PlaCamera result list target is null");
        }
        if (sources.size() != targets->size())
        {
            return fail(error, "PlaBundle camera count does not match the PlaCamera result target count");
        }
        auto converted = *targets;
        for (std::size_t index = 0; index < sources.size(); ++index)
        {
            if (!decode(sources[index], &converted[index], error))
            {
                return false;
            }
        }
        *targets = std::move(converted);
        return true;
    }

} // namespace xjw::sfm_bundle_camera

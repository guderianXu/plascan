#include "SfmBundleCameraCodec.h"

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

    bool encode(const placamera::FramePinholeNumericState& source,
                placamera::FramePinholeNumericState* target,
                std::string* error)
    {
        if (!target)
        {
            return fail(error, "PlaBundle camera output is null");
        }
        if (source.pixelConvention() != placamera::PixelConvention::PixelCenter)
        {
            return fail(error, "PlaBundle requires pixel-center frame calibration");
        }
        *target = source;
        return true;
    }

    bool decode(const placamera::FramePinholeNumericState& source,
                placamera::FramePinholeNumericState* target,
                std::string* error)
    {
        if (!target)
        {
            return fail(error, "PlaCamera result target is null");
        }
        if (source.instanceId() != target->instanceId() || source.imageId() != target->imageId() ||
            source.groundFrame() != target->groundFrame() ||
            source.imageSize().samples != target->imageSize().samples ||
            source.imageSize().lines != target->imageSize().lines ||
            source.depthAxisFlipped() != target->depthAxisFlipped() ||
            source.pixelConvention() != target->pixelConvention())
        {
            return fail(error, "PlaBundle changed camera identity, image grid or optical convention");
        }
        *target = source;
        return true;
    }

    bool encodeAll(const std::vector<placamera::FramePinholeNumericState>& sources,
                   std::vector<placamera::FramePinholeNumericState>* targets,
                   std::string* error)
    {
        if (!targets)
        {
            return fail(error, "PlaBundle camera list output is null");
        }
        std::vector<placamera::FramePinholeNumericState> converted;
        converted.reserve(sources.size());
        for (const auto& source : sources)
        {
            if (source.pixelConvention() != placamera::PixelConvention::PixelCenter)
            {
                return fail(error, "PlaBundle requires pixel-center frame calibration");
            }
            converted.push_back(source);
        }
        *targets = std::move(converted);
        return true;
    }

    bool decodeAll(const std::vector<placamera::FramePinholeNumericState>& sources,
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

#include "engine/PinholeEngine.h"

#include "file/FileIO.h"

#include <optional>
#include <set>
#include <string>

namespace xjw::aerial_triangulation::engine
{

    IncrementalSfmResult runPinhole(const PinholeInput& input)
    {
        const auto fail = [](const std::string& message)
        {
            IncrementalSfmResult result;
            result.summary = message;
            return result;
        };
        if (!input.graph || input.images.size() < 2)
        {
            return fail("空三需要准备好的连接点图和至少两张影像");
        }
        if (input.cancelFlag && input.cancelFlag->load())
        {
            return fail("用户取消");
        }
        std::set<ImageId> image_ids;
        std::set<std::string> canonical_image_ids;
        std::set<std::string> camera_instance_ids;
        std::optional<placoordinate::CoordinateFrameId> common_frame;
        for (const PinholeImage& image : input.images)
        {
            if (image.id == kInvalidImageId || !image_ids.insert(image.id).second)
            {
                return fail("空三影像 ID 重复或无效");
            }
            if (!image.camera || !image.camera->imageSize().isValid() ||
                image.camera->pinholeDefinition().pixelConvention() != placamera::PixelConvention::PixelCenter)
            {
                return fail("空三影像 PlaCamera 相机未准备或不支持 pixel-center 标定");
            }
            if (!canonical_image_ids.insert(image.camera->imageId().value()).second ||
                !camera_instance_ids.insert(image.camera->instanceId().value()).second)
            {
                return fail("空三 PlaCamera 影像或实例身份重复");
            }
            if (!common_frame.has_value())
            {
                common_frame = image.camera->groundFrame();
            }
            else if (*common_frame != image.camera->groundFrame())
            {
                return fail("空三影像相机混用 world frame；必须先显式归一化");
            }
        }
        IncrementalSfm sfm(input.options);
        for (const PinholeImage& image : input.images)
        {
            const auto keypoints = input.graph->keypointsByImage.find(image.id);
            static const std::vector<FeatureKeypoint> empty_keypoints;
            const auto solver_camera = placamera::FramePinholeNumericState::fromModel(*image.camera);
            sfm.addImageWithCamera(image.id,
                                   xjw::common::file::pathToUtf8(image.path),
                                   solver_camera,
                                   keypoints == input.graph->keypointsByImage.end() ? empty_keypoints
                                                                                    : keypoints->second,
                                   image.sensorKey);
        }
        for (const auto& track : input.priorTracks)
        {
            sfm.addPriorTrack(track);
        }
        for (const auto& scale_bar : input.scaleBars)
        {
            sfm.addPriorScaleBar(scale_bar);
        }
        for (const auto& pair : input.graph->matchPairs)
        {
            if (!image_ids.contains(pair.imageA) || !image_ids.contains(pair.imageB))
            {
                return fail("连接点像对引用了未准备的影像");
            }
            sfm.addMatches(pair.imageA, pair.imageB, pair.matches);
        }
        sfm.setInputMultiViewTracks(input.graph->tracks);
        return sfm.run(
            [&input](int registered, int total, const std::string& message)
            {
                if (input.cancelFlag && input.cancelFlag->load())
                {
                    return false;
                }
                return !input.progressFn || input.progressFn(registered, total, message);
            });
    }

} // namespace xjw::aerial_triangulation::engine

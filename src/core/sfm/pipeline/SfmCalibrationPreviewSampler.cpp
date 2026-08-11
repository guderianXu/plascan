#include "SfmCalibrationPreviewSampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xjw::sfm_calibration_preview
{
namespace
{

struct CameraTrack
{
    std::size_t trackIndex = 0;
    double normalizedRadius = 0.0;
};

double normalizedRadius(const Camera &camera, const BAObservation &observation)
{
    const Camera::Intrinsics intrinsics = camera.intrinsics();
    const double focal_x = std::max(1.0, std::abs(intrinsics.focalX));
    const double focal_y = std::max(1.0, std::abs(intrinsics.focalY));
    const double x = (observation.u - intrinsics.principalX) / focal_x;
    const double y = (observation.v - intrinsics.principalY) / focal_y;
    const double radius = std::hypot(x, y);
    return std::isfinite(radius) ? radius : 0.0;
}

} // namespace

std::vector<std::size_t> selectTrackIndices(
    const std::vector<Camera> &cameras,
    const std::vector<BATrack> &tracks,
    std::size_t maximumTrackCount)
{
    if (maximumTrackCount == 0 || tracks.empty() || cameras.empty())
    {
        return {};
    }
    if (tracks.size() <= maximumTrackCount)
    {
        std::vector<std::size_t> all_tracks(tracks.size());
        std::iota(all_tracks.begin(), all_tracks.end(), std::size_t{0});
        return all_tracks;
    }

    std::vector<std::vector<CameraTrack>> camera_tracks(cameras.size());
    for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
    {
        const BAObservation *selected_observation = nullptr;
        std::size_t selected_camera = 0;
        for (const BAObservation &observation : tracks[track_index].observations)
        {
            if (observation.cameraIndex < 0 ||
                observation.cameraIndex >= static_cast<int>(cameras.size()) ||
                !std::isfinite(observation.u) || !std::isfinite(observation.v))
            {
                continue;
            }
            const std::size_t camera_index =
                static_cast<std::size_t>(observation.cameraIndex);
            if (!selected_observation ||
                camera_tracks[camera_index].size() < camera_tracks[selected_camera].size() ||
                (camera_tracks[camera_index].size() == camera_tracks[selected_camera].size() &&
                 camera_index < selected_camera))
            {
                selected_observation = &observation;
                selected_camera = camera_index;
            }
        }
        if (selected_observation)
        {
            camera_tracks[selected_camera].push_back(
                {track_index,
                 normalizedRadius(cameras[selected_camera], *selected_observation)});
        }
    }

    std::vector<std::size_t> quotas(cameras.size(), 0);
    std::size_t selected_count = 0;
    bool quota_changed = true;
    while (selected_count < maximumTrackCount && quota_changed)
    {
        quota_changed = false;
        for (std::size_t camera_index = 0;
             camera_index < cameras.size() && selected_count < maximumTrackCount;
             ++camera_index)
        {
            if (quotas[camera_index] >= camera_tracks[camera_index].size())
            {
                continue;
            }
            ++quotas[camera_index];
            ++selected_count;
            quota_changed = true;
        }
    }

    std::vector<std::size_t> selected_tracks;
    selected_tracks.reserve(selected_count);
    for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
    {
        std::vector<CameraTrack> &bucket = camera_tracks[camera_index];
        std::sort(bucket.begin(), bucket.end(), [](const CameraTrack &left, const CameraTrack &right)
        {
            if (left.normalizedRadius != right.normalizedRadius)
            {
                return left.normalizedRadius < right.normalizedRadius;
            }
            return left.trackIndex < right.trackIndex;
        });
        const std::size_t quota = quotas[camera_index];
        for (std::size_t index = 0; index < quota; ++index)
        {
            const std::size_t rank = std::min(
                bucket.size() - 1,
                ((2 * index + 1) * bucket.size()) / (2 * quota));
            selected_tracks.push_back(bucket[rank].trackIndex);
        }
    }
    std::sort(selected_tracks.begin(), selected_tracks.end());
    return selected_tracks;
}

} // namespace xjw::sfm_calibration_preview

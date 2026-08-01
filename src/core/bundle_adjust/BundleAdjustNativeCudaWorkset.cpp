#include "BundleAdjustNativeCudaWorkset.h"

#include "BundleAdjustProjection.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace xjw::detail::native_cuda
{

namespace
{

bool finitePoint(const std::array<double, 3> &point)
{
    return std::isfinite(point[0]) &&
           std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

bool observationUsable(const BAObservation &observation, int cameraCount)
{
    return observation.cameraIndex >= 0 &&
           observation.cameraIndex < cameraCount &&
           std::isfinite(observation.u) &&
           std::isfinite(observation.v) &&
           std::isfinite(observation.weight) &&
           observation.weight >= 0.0;
}

bool cameraFixed(int cameraIndex, const BAOptions &options)
{
    return std::find(options.fixedCameraIndices.begin(),
                     options.fixedCameraIndices.end(),
                     cameraIndex) != options.fixedCameraIndices.end();
}

HostCamera makeHostCamera(const Camera &camera, int originalIndex, const BAOptions &options)
{
    const xjw::ba::ProjectionCamera projection = xjw::ba::makeProjectionCamera(camera);

    HostCamera host;
    host.cameraToWorldRotation = projection.cameraToWorldRotation;
    host.cameraCenter = projection.cameraCenter;
    host.focalX = projection.focalX;
    host.focalY = projection.focalY;
    host.principalX = projection.principalX;
    host.principalY = projection.principalY;
    host.radialK1 = projection.radialK1;
    host.radialK2 = projection.radialK2;
    host.radialK3 = projection.radialK3;
    host.tangentialP1 = projection.tangentialP1;
    host.tangentialP2 = projection.tangentialP2;
    host.uAxisSign = projection.uAxisSign;
    host.vAxisSign = projection.vAxisSign;
    host.depthAxisFlipped = projection.depthAxisFlipped ? 1 : 0;
    host.fixed = cameraFixed(originalIndex, options) ? 1 : 0;
    host.originalIndex = originalIndex;
    return host;
}

} // namespace

bool hasUnsupportedConstraints(const std::vector<BATrack> &tracks,
                               const BAOptions &options,
                               std::string *message)
{
    if (options.enableLaserPlaneConstraints)
    {
        if (message)
        {
            *message = "native_cuda 首期不支持 LiDAR 点到面约束";
        }
        return true;
    }

    if (options.enableControlPointConstraints)
    {
        if (message)
        {
            *message = "native_cuda 首期不支持控制点约束";
        }
        return true;
    }

    if (options.enableScaleBarConstraints || !options.scaleBarConstraints.empty())
    {
        if (message)
        {
            *message = "native_cuda 首期不支持比例尺约束";
        }
        return true;
    }

    for (const BATrack &track : tracks)
    {
        if (!track.laserPlaneConstraints.empty())
        {
            if (message)
            {
                *message = "native_cuda 首期不支持 track LiDAR 约束";
            }
            return true;
        }

        if (!track.controlPointConstraints.empty())
        {
            if (message)
            {
                *message = "native_cuda 首期不支持 track 控制点约束";
            }
            return true;
        }
    }

    for (const BACameraPosePrior &prior : options.cameraPosePriors)
    {
        if (prior.enabled)
        {
            if (message)
            {
                *message = "native_cuda 首期不支持相机位姿软先验";
            }
            return true;
        }
    }

    return false;
}

WorksetBuildResult buildWorkset(const std::vector<Camera> &cameras,
                                const std::vector<BATrack> &tracks,
                                const BAOptions &options)
{
    WorksetBuildResult result;
    if (hasUnsupportedConstraints(tracks, options, &result.message))
    {
        return result;
    }

    if (cameras.empty())
    {
        result.message = "native_cuda 输入相机为空";
        return result;
    }

    // 相机保持原始顺序，观测中的 cameraIndex 因而无需重映射。
    result.workset.cameras.reserve(cameras.size());
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        result.workset.cameras.push_back(makeHostCamera(cameras[i], static_cast<int>(i), options));
    }

    // 三维点会压缩掉无效 track；originalTrackToPoint 保证下载后仍能构造与输入
    // tracks 等长、索引稳定的 BAResult::points。
    result.workset.originalTrackToPoint.assign(tracks.size(), -1);
    for (size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
        const BATrack &track = tracks[trackIndex];
        if (!finitePoint(track.initialPoint))
        {
            continue;
        }

        std::vector<HostObservation> observations;
        std::set<int> uniqueCameras;
        for (const BAObservation &observation : track.observations)
        {
            if (!observationUsable(observation, static_cast<int>(cameras.size())))
            {
                continue;
            }

            uniqueCameras.insert(observation.cameraIndex);
            observations.push_back({observation.cameraIndex, -1, observation.u, observation.v, observation.weight});
        }

        if (observations.size() < 2 || uniqueCameras.size() < 2)
        {
            continue;
        }

        const int pointIndex = static_cast<int>(result.workset.points.size());

        // 同一点的观测连续追加。CUDA 点块求解可通过 begin/count 顺序扫描，
        // 不需要在设备端建立间接链表或原子聚合。
        HostPoint point;
        point.xyz = track.initialPoint;
        point.originalTrackIndex = static_cast<int>(trackIndex);
        point.observationBegin = static_cast<int>(result.workset.observations.size());
        point.observationCount = static_cast<int>(observations.size());
        result.workset.points.push_back(point);
        result.workset.originalTrackToPoint[trackIndex] = pointIndex;

        for (HostObservation observation : observations)
        {
            observation.pointIndex = pointIndex;
            result.workset.observations.push_back(observation);
        }
    }

    if (result.workset.points.empty())
    {
        result.message = "native_cuda 没有足够有效 track";
        return result;
    }

    result.ok = true;
    return result;
}

} // namespace xjw::detail::native_cuda

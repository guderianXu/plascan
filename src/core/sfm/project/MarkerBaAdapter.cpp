#include "MarkerBaAdapter.h"

#include "geometry/TriangulationQuality.h"
#include "project/ProjectMatchInputReader.h"

#include <QSet>

#include <algorithm>
#include <cmath>

namespace xjw::core::project
{
namespace
{

int cameraIndexForMarkerProjection(const control_points::MarkerProjection &projection,
                                   const MarkerBaInput &input,
                                   const QMap<QString, int> &cameraIndexByPath)
{
    QString path = input.imagePathById.value(projection.imageId);
    if (path.isEmpty())
    {
        path = projection.imagePathSnapshot;
    }
    return cameraIndexForImageToken(path, cameraIndexByPath);
}

bool triangulateMarkerTrack(const xjw::BATrack &track,
                            const std::vector<xjw::Camera> &cameras,
                            std::array<double, 3> *point)
{
    if (!point)
    {
        return false;
    }
    for (std::size_t first = 0; first + 1 < track.observations.size(); ++first)
    {
        const xjw::BAObservation &left = track.observations[first];
        if (left.cameraIndex < 0 || left.cameraIndex >= static_cast<int>(cameras.size()))
        {
            continue;
        }
        for (std::size_t second = first + 1; second < track.observations.size(); ++second)
        {
            const xjw::BAObservation &right = track.observations[second];
            if (right.cameraIndex < 0 || right.cameraIndex >= static_cast<int>(cameras.size())
                || left.cameraIndex == right.cameraIndex)
            {
                continue;
            }
            const xjw::PairIntersectionCandidate candidate =
                xjw::triangulatePairWithDirectionFallback(
                    cameras[static_cast<std::size_t>(left.cameraIndex)], {left.u, left.v},
                    cameras[static_cast<std::size_t>(right.cameraIndex)], {right.u, right.v});
            if (candidate.valid)
            {
                *point = candidate.point;
                return true;
            }
        }
    }
    return false;
}

double referenceSigmaRms(const control_points::ReferenceCoordinate &reference)
{
    return std::sqrt((reference.sigmaX * reference.sigmaX
                    + reference.sigmaY * reference.sigmaY
                    + reference.sigmaZ * reference.sigmaZ) / 3.0);
}

} // namespace

void appendMarkerBaInput(const MarkerBaInput *input,
                         const QMap<QString, int> &cameraIndexByPath,
                         BaInputBuildResult *result)
{
    if (!input || !input->markerSet || !result)
    {
        return;
    }

    QMap<control_points::MarkerId, int> trackIndexByMarker;
    control_points::ControlNetworkInput networkInput;

    // 阶段 1：把每个启用标记的有效投影变成独立多视轨迹。预测投影和禁用投影
    // 不参与平差，同一相机的重复投影只取一个，防止人为重复加权。
    for (const control_points::Marker &marker : input->markerSet->markers())
    {
        if (!marker.enabled)
        {
            ++result->rejectedMarkerTrackCount;
            continue;
        }

        xjw::BATrack track;
        QSet<int> usedCameras;
        for (const control_points::MarkerProjection &projection : marker.projections)
        {
            if (!control_points::projectionParticipatesInAdjustment(projection.state))
            {
                continue;
            }
            const int cameraIndex = cameraIndexForMarkerProjection(
                projection, *input, cameraIndexByPath);
            if (cameraIndex < 0 || usedCameras.contains(cameraIndex))
            {
                continue;
            }
            track.observations.push_back({
                cameraIndex,
                projection.xy.x(),
                projection.xy.y(),
                1.0 / std::max(1.0e-9, projection.sigmaPx * projection.sigmaPx)});
            usedCameras.insert(cameraIndex);
        }
        if (track.observations.size() < 2
            || !triangulateMarkerTrack(track, result->cameras, &track.initialPoint))
        {
            ++result->rejectedMarkerTrackCount;
            continue;
        }

        const int trackIndex = static_cast<int>(result->tracks.size());
        result->tracks.push_back(track);
        trackIndexByMarker.insert(marker.id, trackIndex);
        if (marker.role == control_points::MarkerRole::ControlPoint)
        {
            ++result->markerControlTrackCount;
        }
        else if (marker.role == control_points::MarkerRole::CheckPoint)
        {
            ++result->markerCheckTrackCount;
        }

        if (marker.referenceCoordinate.has_value()
            && marker.referenceCoordinate->referenceUsable
            && marker.role != control_points::MarkerRole::TieMarker)
        {
            const control_points::ReferenceCoordinate &reference = *marker.referenceCoordinate;
            control_points::ControlNetworkPoint point;
            point.markerId = marker.id.toStdString();
            point.role = marker.role;
            point.estimatedPoint = track.initialPoint;
            point.referencePoint = {{reference.x, reference.y, reference.z}};
            point.sigma = {{reference.sigmaX, reference.sigmaY, reference.sigmaZ}};
            networkInput.points.push_back(point);

            BaInputBuildResult::MarkerTrackBinding binding;
            binding.markerId = marker.id;
            binding.role = marker.role;
            binding.trackIndex = trackIndex;
            binding.referencePoint = point.referencePoint;
            binding.sigma = point.sigma;
            result->markerTrackBindings.push_back(binding);
        }
    }

    // 阶段 2：仅控制点参与 Sim(3) 控制网估计；检查点保留独立残差评估语义。
    // 估计失败时不附加错误物方约束，自动连接点仍可用于自由网 BA。
    result->markerControlNetwork = control_points::solveControlNetwork(networkInput);
    if (!result->markerControlNetwork.ok)
    {
        return;
    }
    const control_points::SimilarityTransform3D &transform =
        result->markerControlNetwork.transform;
    for (xjw::Camera &camera : result->cameras)
    {
        camera.setPose(transform.rotate(camera.cameraToWorldRotation()),
                       transform.apply(camera.cameraCenter()));
    }
    for (xjw::BATrack &track : result->tracks)
    {
        track.initialPoint = transform.apply(track.initialPoint);
    }

    // 阶段 3：只把控制网内点写成 BA 控制点约束。离群控制点仍保留轨迹和报告，
    // 但不会把错误物方坐标拉入非线性优化。
    for (const control_points::MarkerResidual &residual :
         result->markerControlNetwork.controlResiduals)
    {
        const QString markerId = QString::fromStdString(residual.markerId);
        if (!residual.inlier || !trackIndexByMarker.contains(markerId))
        {
            continue;
        }
        const control_points::Marker &marker = input->markerSet->marker(markerId);
        if (!marker.referenceCoordinate.has_value())
        {
            continue;
        }
        const control_points::ReferenceCoordinate &reference = *marker.referenceCoordinate;
        xjw::BAControlPointConstraint constraint;
        constraint.point = {{reference.x, reference.y, reference.z}};
        constraint.sigmaMeters = referenceSigmaRms(reference);
        constraint.weight = 1.0;
        constraint.sourceIndex = trackIndexByMarker.value(markerId);
        result->tracks[static_cast<std::size_t>(constraint.sourceIndex)]
            .controlPointConstraints.push_back(constraint);
        for (BaInputBuildResult::MarkerTrackBinding &binding : result->markerTrackBindings)
        {
            if (binding.markerId == markerId)
            {
                binding.usedAsConstraint = true;
            }
        }
        ++result->markerControlPointConstraintCount;
    }

    // 阶段 4：控制标尺进入 BA 消除尺度自由度；检查标尺只建立回写绑定。
    for (int scaleIndex = 0; scaleIndex < input->markerSet->scaleBars().size(); ++scaleIndex)
    {
        const control_points::ScaleBar &scaleBar = input->markerSet->scaleBars()[scaleIndex];
        if (!scaleBar.enabled
            || !trackIndexByMarker.contains(scaleBar.firstMarkerId)
            || !trackIndexByMarker.contains(scaleBar.secondMarkerId))
        {
            ++result->rejectedMarkerScaleBarCount;
            continue;
        }
        if (scaleBar.role == control_points::ScaleBarRole::Check)
        {
            ++result->markerCheckScaleBarCount;
        }
        else
        {
            xjw::BAScaleBarConstraint constraint;
            constraint.trackIndexA = trackIndexByMarker.value(scaleBar.firstMarkerId);
            constraint.trackIndexB = trackIndexByMarker.value(scaleBar.secondMarkerId);
            constraint.measuredDistanceMeters = scaleBar.measuredDistance;
            constraint.sigmaMeters = scaleBar.sigma;
            constraint.weight = 1.0;
            constraint.sourceIndex = scaleIndex;
            result->scaleBarConstraints.push_back(constraint);
            ++result->markerControlScaleBarConstraintCount;
        }

        BaInputBuildResult::MarkerScaleBarBinding binding;
        binding.scaleBarId = scaleBar.id;
        binding.role = scaleBar.role;
        binding.trackIndexA = trackIndexByMarker.value(scaleBar.firstMarkerId);
        binding.trackIndexB = trackIndexByMarker.value(scaleBar.secondMarkerId);
        binding.measuredDistance = scaleBar.measuredDistance;
        result->markerScaleBarBindings.push_back(binding);
    }
}

} // namespace xjw::core::project

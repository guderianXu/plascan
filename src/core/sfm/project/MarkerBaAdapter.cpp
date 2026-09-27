#include "MarkerBaAdapter.h"

#include "geometry/TriangulationQuality.h"
#include "reference/CoordinateReference.h"

#include <placamera/frame_numeric_state.h>

#include <QSet>

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace xjw::core::project
{
    namespace
    {

        int cameraIndexForMarkerProjection(const control_points::MarkerProjection& projection,
                                           const QMap<QString, int>& cameraIndexByImageId)
        {
            const QString image_id = projection.imageId.trimmed();
            if (!image_id.isEmpty())
            {
                const auto by_id = cameraIndexByImageId.constFind(image_id);
                return by_id == cameraIndexByImageId.constEnd() ? -1 : by_id.value();
            }
            return -1;
        }

        void recordControlInputError(BaInputBuildResult* result, const QString& message)
        {
            if (result && result->firstControlInputError.isEmpty())
            {
                result->firstControlInputError = message;
            }
        }

        bool
        triangulateMarkerTrack(const plabundle::Track& track,
                               const std::vector<placamera::FramePinholeNumericState>& cameras,
                               std::array<double, 3>* point)
        {
            if (!point)
            {
                return false;
            }
            for (std::size_t first = 0; first + 1 < track.observations.size(); ++first)
            {
                const plabundle::Observation& left = track.observations[first];
                if (left.cameraIndex < 0 || left.cameraIndex >= static_cast<int>(cameras.size()))
                {
                    continue;
                }
                for (std::size_t second = first + 1; second < track.observations.size(); ++second)
                {
                    const plabundle::Observation& right = track.observations[second];
                    if (right.cameraIndex < 0 || right.cameraIndex >= static_cast<int>(cameras.size()) ||
                        left.cameraIndex == right.cameraIndex)
                    {
                        continue;
                    }
                    const auto candidate = placamera::FramePinholeNumericState::triangulatePair(
                        cameras[static_cast<std::size_t>(left.cameraIndex)],
                        placamera::ImageCoordinate{left.u, left.v},
                        cameras[static_cast<std::size_t>(right.cameraIndex)],
                        placamera::ImageCoordinate{right.u, right.v});
                    if (candidate)
                    {
                        *point = candidate.value().point.position;
                        return true;
                    }
                }
            }
            return false;
        }

        double referenceSigmaRms(const std::array<double, 3>& sigma)
        {
            return std::sqrt((sigma[0] * sigma[0] + sigma[1] * sigma[1] + sigma[2] * sigma[2]) / 3.0);
        }

    } // namespace

    void appendMarkerBaInput(const MarkerBaInput* input,
                             const QMap<QString, int>& cameraIndexByImageId,
                             BaInputBuildResult* result)
    {
        if (!input || !input->markerSet || !result)
        {
            return;
        }

        if (result->cameraInstances.size() != result->imageIdByIndex.size())
        {
            recordControlInputError(result,
                                    QStringLiteral("BA 标记观测的 PlaCamera 实例与 "
                                                   "canonical ImageId 数量不一致"));
            return;
        }
        std::vector<placamera::FramePinholeNumericState> native_cameras;
        native_cameras.reserve(result->cameraInstances.size());
        for (std::size_t index = 0; index < result->cameraInstances.size(); ++index)
        {
            const auto& instance = result->cameraInstances[index];
            if (!instance)
            {
                recordControlInputError(result, QStringLiteral("BA 标记观测包含空 PlaCamera 实例"));
                return;
            }
            if (instance->imageId() != result->imageIdByIndex[index])
            {
                const QString message =
                    QStringLiteral("BA 标记观测的 PlaCamera 实例与 "
                                   "canonical ImageId 不一致（索引 %1）")
                        .arg(static_cast<qulonglong>(index));
                recordControlInputError(result, message);
                return;
            }
            try
            {
                native_cameras.push_back(placamera::FramePinholeNumericState::fromModel(*instance));
            }
            catch (const std::exception& exception)
            {
                recordControlInputError(result,
                                        QStringLiteral("BA 标记相机无法建立 PlaCamera 数值状态：%1")
                                            .arg(QString::fromUtf8(exception.what())));
                return;
            }
        }

        QMap<control_points::MarkerId, int> trackIndexByMarker;
        QMap<control_points::MarkerId, control_points::MetricReferenceCoordinateResult> metricReferenceByMarker;
        std::string commonReferenceKey;
        for (const control_points::Marker& marker : input->markerSet->markers())
        {
            if (!marker.enabled || marker.role == control_points::MarkerRole::TieMarker ||
                !marker.referenceCoordinate.has_value())
            {
                continue;
            }

            const control_points::MetricReferenceCoordinateResult metric_reference =
                control_points::resolveMetricReferenceCoordinate(*marker.referenceCoordinate, input->coordinateContext);
            if (!metric_reference.ok)
            {
                recordControlInputError(
                    result,
                    QStringLiteral("marker %1 (%2) reference coordinate is not solver-ready: %3")
                        .arg(marker.label, marker.id, QString::fromStdString(metric_reference.error)));
                return;
            }
            if (commonReferenceKey.empty())
            {
                commonReferenceKey = metric_reference.referenceKey;
            }
            else if (metric_reference.referenceKey != commonReferenceKey)
            {
                recordControlInputError(
                    result,
                    QStringLiteral("marker %1 (%2) uses a different CRS or vertical reference; "
                                   "all marker references in one BA must share one normalized reference")
                        .arg(marker.label, marker.id));
                return;
            }
            metricReferenceByMarker.insert(marker.id, metric_reference);
        }

        control_points::ControlNetworkInput networkInput;

        // 阶段 1：把每个启用标记的有效投影变成独立多视轨迹。预测投影和禁用投影
        // 不参与平差，同一相机的重复投影只取一个，防止人为重复加权。
        for (const control_points::Marker& marker : input->markerSet->markers())
        {
            if (!marker.enabled)
            {
                ++result->rejectedMarkerTrackCount;
                continue;
            }

            plabundle::Track track;
            QSet<int> usedCameras;
            for (const control_points::MarkerProjection& projection : marker.projections)
            {
                if (!control_points::projectionParticipatesInAdjustment(projection.state))
                {
                    continue;
                }
                const QString image_id = projection.imageId.trimmed();
                if (image_id.isEmpty())
                {
                    recordControlInputError(result,
                                            QStringLiteral("marker %1 has a projection without canonical image_uuid; "
                                                           "image_path_snapshot cannot identify a camera")
                                                .arg(marker.id));
                    continue;
                }
                if (!cameraIndexByImageId.contains(image_id))
                {
                    recordControlInputError(result,
                                            QStringLiteral("marker %1 references unknown canonical image_uuid %2; "
                                                           "image_path_snapshot cannot be used as a fallback")
                                                .arg(marker.id, image_id));
                    continue;
                }
                const int cameraIndex = cameraIndexForMarkerProjection(projection, cameraIndexByImageId);
                if (cameraIndex < 0 || usedCameras.contains(cameraIndex))
                {
                    continue;
                }
                track.observations.push_back({cameraIndex,
                                              projection.xy.x(),
                                              projection.xy.y(),
                                              1.0 / std::max(1.0e-9, projection.sigmaPx * projection.sigmaPx)});
                usedCameras.insert(cameraIndex);
            }
            if (track.observations.size() < 2 ||
                !triangulateMarkerTrack(track, native_cameras, &track.initialPoint))
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

            const auto metricReference = metricReferenceByMarker.constFind(marker.id);
            if (metricReference != metricReferenceByMarker.constEnd())
            {
                control_points::ControlNetworkPoint point;
                point.markerId = marker.id.toStdString();
                point.role = marker.role;
                point.estimatedPoint = track.initialPoint;
                point.referencePoint = metricReference->pointMetres;
                point.sigma = metricReference->sigmaMetres;
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
        const control_points::SimilarityTransform3D& transform = result->markerControlNetwork.transform;
        std::vector<std::shared_ptr<const placamera::FramePinholeModel>> transformed_instances;
        transformed_instances.reserve(result->cameraInstances.size());
        try
        {
            for (std::size_t index = 0; index < result->cameraInstances.size(); ++index)
            {
                const auto& instance = result->cameraInstances[index];
                const placamera::Pose transformed_pose =
                    placamera::Pose::create(instance->groundFrame(),
                                            transform.apply(instance->pose().center),
                                            transform.rotate(instance->pose().cameraToWorldRotation));
                transformed_instances.push_back(std::make_shared<const placamera::FramePinholeModel>(
                    instance->withPose(instance->instanceId(), transformed_pose)));
            }
        }
        catch (const std::exception& exception)
        {
            recordControlInputError(
                result, QStringLiteral("BA 控制网相机位姿更新失败：%1").arg(QString::fromUtf8(exception.what())));
            return;
        }
        result->cameraInstances = std::move(transformed_instances);
        for (plabundle::Track& track : result->tracks)
        {
            track.initialPoint = transform.apply(track.initialPoint);
        }

        // 阶段 3：只把控制网内点写成 BA 控制点约束。离群控制点仍保留轨迹和报告，
        // 但不会把错误物方坐标拉入非线性优化。
        for (const control_points::MarkerResidual& residual : result->markerControlNetwork.controlResiduals)
        {
            const QString markerId = QString::fromStdString(residual.markerId);
            if (!residual.inlier || !trackIndexByMarker.contains(markerId))
            {
                continue;
            }
            const control_points::Marker& marker = input->markerSet->marker(markerId);
            const auto metricReference = metricReferenceByMarker.constFind(markerId);
            if (metricReference == metricReferenceByMarker.constEnd())
            {
                continue;
            }
            plabundle::ControlPointConstraint constraint;
            constraint.point = metricReference->pointMetres;
            constraint.sigmaMeters = referenceSigmaRms(metricReference->sigmaMetres);
            constraint.weight = 1.0;
            constraint.sourceIndex = trackIndexByMarker.value(markerId);
            result->tracks[static_cast<std::size_t>(constraint.sourceIndex)].controlPointConstraints.push_back(
                constraint);
            for (BaInputBuildResult::MarkerTrackBinding& binding : result->markerTrackBindings)
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
            const control_points::ScaleBar& scaleBar = input->markerSet->scaleBars()[scaleIndex];
            if (!scaleBar.enabled || !trackIndexByMarker.contains(scaleBar.firstMarkerId) ||
                !trackIndexByMarker.contains(scaleBar.secondMarkerId))
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
                plabundle::ScaleBarConstraint constraint;
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

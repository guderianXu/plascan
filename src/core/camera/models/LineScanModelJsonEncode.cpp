#include "LineScanModelJson.h"

#include <QJsonArray>

#include <array>
#include <vector>

namespace xjw::camera_models
{
    namespace
    {

        template <std::size_t Size> QJsonArray numberArray(const std::array<double, Size>& values)
        {
            QJsonArray result;
            for (double value : values)
            {
                result.append(value);
            }
            return result;
        }

        QString timeScaleName(xjw::coordinate_system::TimeScale scale)
        {
            switch (scale)
            {
            case xjw::coordinate_system::TimeScale::Utc:
                return QStringLiteral("utc");
            case xjw::coordinate_system::TimeScale::Tai:
                return QStringLiteral("tai");
            case xjw::coordinate_system::TimeScale::Tdb:
                return QStringLiteral("tdb");
            case xjw::coordinate_system::TimeScale::Relative:
                return QStringLiteral("relative");
            }
            return {};
        }

        QJsonObject rotationTrajectoryToJson(const linescan::FrameRotationTrajectory& trajectory)
        {
            QJsonArray samples;
            for (const linescan::QuaternionTrajectorySample& sample : trajectory.samples)
            {
                samples.append(
                    QJsonObject{{QStringLiteral("time_seconds"), sample.time.seconds},
                                {QStringLiteral("quaternion_scalar_first"), numberArray(sample.scalarFirst)}});
            }
            return {{QStringLiteral("constant_rotation"), numberArray(trajectory.constantRotation)},
                    {QStringLiteral("samples"), samples}};
        }

        QJsonObject trajectoryToJson(const linescan::LineScanTrajectory& trajectory)
        {
            QJsonObject result{{QStringLiteral("time_scale"), timeScaleName(trajectory.timeScale())}};
            if (const linescan::FrameComposedTrajectory* composed = trajectory.frameComposed())
            {
                result.insert(QStringLiteral("representation"), QStringLiteral("frame_composed"));
                QJsonArray states;
                for (const linescan::TranslationalStateSample& state : composed->inertialStates)
                {
                    states.append(
                        QJsonObject{{QStringLiteral("time_seconds"), state.time.seconds},
                                    {QStringLiteral("position_m"), numberArray(state.positionMeters)},
                                    {QStringLiteral("velocity_m_per_s"), numberArray(state.velocityMetersPerSecond)}});
                }
                result.insert(QStringLiteral("inertial_states"), states);
                result.insert(QStringLiteral("inertial_to_world"), rotationTrajectoryToJson(composed->inertialToWorld));
                result.insert(QStringLiteral("inertial_to_sensor"),
                              rotationTrajectoryToJson(composed->inertialToSensor));
                return result;
            }

            result.insert(QStringLiteral("representation"), QStringLiteral("direct_pose_samples"));
            QJsonArray samples;
            for (const linescan::TrajectorySample& sample : trajectory.samples())
            {
                samples.append(QJsonObject{
                    {QStringLiteral("time_seconds"), sample.time.seconds},
                    {QStringLiteral("center_m"), numberArray(sample.center)},
                    {QStringLiteral("camera_to_world_rotation"), numberArray(sample.cameraToWorldRotation)}});
            }
            result.insert(QStringLiteral("samples"), samples);
            return result;
        }

        QJsonObject timingToJson(const linescan::LineTiming& timing)
        {
            std::vector<linescan::LineRateSegment> segments = timing.segments;
            if (segments.empty())
            {
                segments.push_back({timing.lineZero, timing.startTimeSeconds, timing.secondsPerLine});
            }
            QJsonArray values;
            for (const linescan::LineRateSegment& segment : segments)
            {
                values.append(QJsonObject{{QStringLiteral("start_line"), segment.startLine},
                                          {QStringLiteral("start_time_seconds"), segment.startTimeSeconds},
                                          {QStringLiteral("seconds_per_line"), segment.secondsPerLine}});
            }
            return {{QStringLiteral("time_scale"), timeScaleName(timing.timeScale)},
                    {QStringLiteral("segments"), values}};
        }

    } // namespace

    QJsonObject lineScanDefinitionParametersToJson(const linescan::LineScanDefinition& definition)
    {
        const linescan::LineScanOptics& source = definition.optics();
        QJsonObject geometry;
        if (source.detectorGeometry)
        {
            const linescan::LineScanDetectorGeometry& detector = *source.detectorGeometry;
            geometry = {{QStringLiteral("type"), QStringLiteral("detector_affine")},
                        {QStringLiteral("detector_sample_summing"), detector.detectorSampleSumming},
                        {QStringLiteral("detector_line_summing"), detector.detectorLineSumming},
                        {QStringLiteral("detector_sample_origin"), detector.detectorSampleOrigin},
                        {QStringLiteral("detector_line_origin"), detector.detectorLineOrigin},
                        {QStringLiteral("starting_detector_sample"), detector.startingDetectorSample},
                        {QStringLiteral("starting_detector_line"), detector.startingDetectorLine},
                        {QStringLiteral("focal_to_pixel_samples"), numberArray(detector.focalToPixelSamples)},
                        {QStringLiteral("focal_to_pixel_lines"), numberArray(detector.focalToPixelLines)}};
        }
        else
        {
            geometry = {{QStringLiteral("type"), QStringLiteral("uniform_pitch")},
                        {QStringLiteral("sample_pitch_mm"), source.samplePitchMillimeters},
                        {QStringLiteral("principal_sample"), source.principalSample}};
        }

        const QString distortion = source.distortionModel == linescan::LineScanDistortionModel::LroNacFocalPlane
                                       ? QStringLiteral("lro_nac_focal_plane")
                                       : QStringLiteral("radial_normalized");
        const QString convention = definition.pixelConvention() == linescan::PixelConvention::ZeroBased
                                       ? QStringLiteral("zero_based")
                                       : QStringLiteral("pixel_center");
        return {{QStringLiteral("optics"),
                 QJsonObject{{QStringLiteral("focal_length_mm"), source.focalLengthMillimeters},
                             {QStringLiteral("distortion_model"), distortion},
                             {QStringLiteral("distortion_k1"), source.distortionK1},
                             {QStringLiteral("sample_geometry"), geometry}}},
                {QStringLiteral("pixel_convention"), convention}};
    }

    QJsonObject lineScanInstanceStateToJson(const linescan::LineScanInstance& instance)
    {
        QJsonObject result{{QStringLiteral("image_size"),
                            QJsonObject{{QStringLiteral("samples"), instance.imageSize().samples},
                                        {QStringLiteral("lines"), instance.imageSize().lines}}},
                           {QStringLiteral("trajectory"), trajectoryToJson(instance.trajectory())},
                           {QStringLiteral("line_timing"), timingToJson(instance.lineTiming())}};
        if (instance.captureTime())
        {
            result.insert(QStringLiteral("capture_time"),
                          QJsonObject{{QStringLiteral("time_scale"), timeScaleName(instance.captureTime()->scale)},
                                      {QStringLiteral("seconds"), instance.captureTime()->seconds}});
        }
        return result;
    }

} // namespace xjw::camera_models

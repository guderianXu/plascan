#include "ProjectCameraStore.h"

#include <placamera/state_codec.h>
#include <placamera/frame_camera.h>
#include <placamera/linescan_camera.h>
#include <placamera/rpc_camera.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

namespace xjw::placamera_runtime
{
    namespace
    {

        using DefinitionStates = std::unordered_map<std::string, placamera::CameraDefinitionState>;

        QJsonObject objectValue(const QJsonObject& object, const QString& key)
        {
            return object.value(key).toObject();
        }

        QString requiredString(const QJsonObject& object, const QString& key)
        {
            return object.value(key).toString().trimmed();
        }

        bool exactSchemaVersion(const QJsonObject& object, int expected)
        {
            const QJsonValue value = object.value(QStringLiteral("schema_version"));
            return value.isDouble() && value.toDouble() == static_cast<double>(expected);
        }

        bool isCentralCameraModel(const QString& model)
        {
            return model == QStringLiteral("frame_pinhole") || model == QStringLiteral("frame_fisheye") ||
                   model == QStringLiteral("frame_equidistant_fisheye") ||
                   model == QStringLiteral("frame_equisolid_fisheye") || model == QStringLiteral("frame_spherical") ||
                   model == QStringLiteral("frame_cylindrical");
        }

        QJsonObject legacyFrameDefinitionParameters(const QJsonObject& parameters)
        {
            const QJsonObject intrinsics = objectValue(parameters, QStringLiteral("intrinsics"));
            const QJsonObject distortion = objectValue(parameters, QStringLiteral("distortion"));
            return {{QStringLiteral("intrinsics"),
                     QJsonObject{{QStringLiteral("focal_x"), intrinsics.value(QStringLiteral("fx_px"))},
                                 {QStringLiteral("focal_y"), intrinsics.value(QStringLiteral("fy_px"))},
                                 {QStringLiteral("principal_x"), intrinsics.value(QStringLiteral("cx_px"))},
                                 {QStringLiteral("principal_y"), intrinsics.value(QStringLiteral("cy_px"))},
                                 {QStringLiteral("pixel_pitch"), intrinsics.value(QStringLiteral("pixel_pitch_mm"))},
                                 {QStringLiteral("u_axis_sign"), intrinsics.value(QStringLiteral("u_axis_sign"))},
                                 {QStringLiteral("v_axis_sign"), intrinsics.value(QStringLiteral("v_axis_sign"))}}},
                    {QStringLiteral("distortion"),
                     QJsonObject{{QStringLiteral("radial_k1"), distortion.value(QStringLiteral("k1"))},
                                 {QStringLiteral("radial_k2"), distortion.value(QStringLiteral("k2"))},
                                 {QStringLiteral("radial_k3"), distortion.value(QStringLiteral("k3"))},
                                 {QStringLiteral("tangential_p1"), distortion.value(QStringLiteral("p1"))},
                                 {QStringLiteral("tangential_p2"), distortion.value(QStringLiteral("p2"))}}},
                    {QStringLiteral("pixel_convention"),
                     parameters.value(QStringLiteral("pixel_convention")) == QStringLiteral("center")
                         ? QJsonValue(QStringLiteral("pixel_center"))
                         : QJsonValue(QStringLiteral("pixel_corner"))},
                    {QStringLiteral("depth_axis_flipped"), parameters.value(QStringLiteral("depth_axis_flipped"))}};
        }

        QJsonObject rpcDefinitionParameters(const QJsonObject& parameters)
        {
            const auto optionalValue = [&parameters](const QString& key)
            {
                const QJsonValue value = parameters.value(key);
                return value.isUndefined() ? QJsonValue(QJsonValue::Null) : value;
            };
            return {
                {QStringLiteral("normalization"),
                 QJsonObject{{QStringLiteral("line_offset"), parameters.value(QStringLiteral("line_offset"))},
                             {QStringLiteral("sample_offset"), parameters.value(QStringLiteral("sample_offset"))},
                             {QStringLiteral("latitude_offset"), parameters.value(QStringLiteral("latitude_offset"))},
                             {QStringLiteral("longitude_offset"), parameters.value(QStringLiteral("longitude_offset"))},
                             {QStringLiteral("height_offset"), parameters.value(QStringLiteral("height_offset"))},
                             {QStringLiteral("line_scale"), parameters.value(QStringLiteral("line_scale"))},
                             {QStringLiteral("sample_scale"), parameters.value(QStringLiteral("sample_scale"))},
                             {QStringLiteral("latitude_scale"), parameters.value(QStringLiteral("latitude_scale"))},
                             {QStringLiteral("longitude_scale"), parameters.value(QStringLiteral("longitude_scale"))},
                             {QStringLiteral("height_scale"), parameters.value(QStringLiteral("height_scale"))}}},
                {QStringLiteral("coefficients"),
                 QJsonObject{
                     {QStringLiteral("line_numerator"), parameters.value(QStringLiteral("line_numerator"))},
                     {QStringLiteral("line_denominator"), parameters.value(QStringLiteral("line_denominator"))},
                     {QStringLiteral("sample_numerator"), parameters.value(QStringLiteral("sample_numerator"))},
                     {QStringLiteral("sample_denominator"), parameters.value(QStringLiteral("sample_denominator"))}}},
                {QStringLiteral("accuracy"),
                 QJsonObject{{QStringLiteral("error_bias_m"), optionalValue(QStringLiteral("error_bias_m"))},
                             {QStringLiteral("error_random_m"), optionalValue(QStringLiteral("error_random_m"))}}},
                {QStringLiteral("ellipsoid"),
                 QJsonObject{{QStringLiteral("semi_major_axis_m"),
                              parameters.value(QStringLiteral("semi_major_axis_m")).isDouble()
                                  ? parameters.value(QStringLiteral("semi_major_axis_m"))
                                  : QJsonValue(6378137.0)},
                             {QStringLiteral("inverse_flattening"),
                              parameters.value(QStringLiteral("inverse_flattening")).isDouble()
                                  ? parameters.value(QStringLiteral("inverse_flattening"))
                                  : QJsonValue(298.257223563)}}}};
        }

        QJsonObject lineDefinitionParameters(const QJsonObject& parameters)
        {
            const QJsonObject source = objectValue(parameters, QStringLiteral("optics"));
            QJsonObject optics{
                {QStringLiteral("focal_length_mm"), source.value(QStringLiteral("focal_length_mm"))},
                {QStringLiteral("sample_pitch_mm"), source.value(QStringLiteral("sample_pitch_mm"))},
                {QStringLiteral("principal_sample"), source.value(QStringLiteral("principal_sample"))},
                {QStringLiteral("distortion_k1"), source.value(QStringLiteral("distortion_k1"))},
                {QStringLiteral("distortion_model"), source.value(QStringLiteral("distortion_model"))},
                {QStringLiteral("detector_geometry"), source.value(QStringLiteral("detector_geometry"))}};
            const QJsonValue complete_calibration = source.value(QStringLiteral("complete_calibration"));
            if (!complete_calibration.isUndefined())
            {
                optics.insert(QStringLiteral("complete_calibration"), complete_calibration);
            }
            if (optics.value(QStringLiteral("sample_pitch_mm")).isUndefined())
            {
                const QJsonObject geometry = objectValue(source, QStringLiteral("sample_geometry"));
                optics.insert(QStringLiteral("sample_pitch_mm"),
                              geometry.value(QStringLiteral("sample_pitch_mm")).isUndefined()
                                  ? QJsonValue(0.0)
                                  : geometry.value(QStringLiteral("sample_pitch_mm")));
                optics.insert(QStringLiteral("principal_sample"),
                              geometry.value(QStringLiteral("principal_sample")).isUndefined()
                                  ? QJsonValue(0.0)
                                  : geometry.value(QStringLiteral("principal_sample")));
                if (geometry.value(QStringLiteral("type")).toString() == QStringLiteral("detector_affine"))
                {
                    optics.insert(
                        QStringLiteral("detector_geometry"),
                        QJsonObject{
                            {QStringLiteral("sample_summing"),
                             geometry.value(QStringLiteral("detector_sample_summing"))},
                            {QStringLiteral("line_summing"), geometry.value(QStringLiteral("detector_line_summing"))},
                            {QStringLiteral("sample_origin"), geometry.value(QStringLiteral("detector_sample_origin"))},
                            {QStringLiteral("line_origin"), geometry.value(QStringLiteral("detector_line_origin"))},
                            {QStringLiteral("starting_sample"),
                             geometry.value(QStringLiteral("starting_detector_sample"))},
                            {QStringLiteral("starting_line"), geometry.value(QStringLiteral("starting_detector_line"))},
                            {QStringLiteral("focal_to_pixel_samples"),
                             geometry.value(QStringLiteral("focal_to_pixel_samples"))},
                            {QStringLiteral("focal_to_pixel_lines"),
                             geometry.value(QStringLiteral("focal_to_pixel_lines"))}});
                }
                else
                {
                    optics.insert(QStringLiteral("detector_geometry"), QJsonValue(QJsonValue::Null));
                }
            }
            return {{QStringLiteral("optics"), optics},
                    {QStringLiteral("pixel_convention"), parameters.value(QStringLiteral("pixel_convention"))}};
        }

        QJsonObject frameInstanceState(const QJsonObject& instance, int schema_version)
        {
            const QJsonObject pose = objectValue(instance, QStringLiteral("pose"));
            QJsonObject state{{QStringLiteral("pose"),
                               QJsonObject{{QStringLiteral("center"), pose.value(QStringLiteral("center_m"))},
                                           {QStringLiteral("camera_to_world_rotation"),
                                            pose.value(QStringLiteral("camera_to_world_rotation"))}}}};
            if (schema_version == placamera::FramePinholeModel::InstanceSchemaVersion)
            {
                state.insert(QStringLiteral("acquisition"), instance.value(QStringLiteral("acquisition")));
            }
            return state;
        }

        QJsonObject rpcInstanceState(const QJsonObject& instance, int schema_version)
        {
            const QJsonObject source =
                objectValue(objectValue(instance, QStringLiteral("state")), QStringLiteral("image_correction"));
            if (schema_version == 1)
            {
                return {
                    {QStringLiteral("image_correction"),
                     QJsonObject{{QStringLiteral("sample_offset"), source.value(QStringLiteral("sample_offset_px"))},
                                 {QStringLiteral("sample_sample"), source.value(QStringLiteral("sample_sample_px"))},
                                 {QStringLiteral("sample_line"), source.value(QStringLiteral("sample_line_px"))},
                                 {QStringLiteral("line_offset"), source.value(QStringLiteral("line_offset_px"))},
                                 {QStringLiteral("line_sample"), source.value(QStringLiteral("line_sample_px"))},
                                 {QStringLiteral("line_line"), source.value(QStringLiteral("line_line_px"))}}}};
            }
            if (source.value(QStringLiteral("model")) == QStringLiteral("affine_normalized_v1"))
            {
                return {
                    {QStringLiteral("correction_domain"), QStringLiteral("normalized_image")},
                    {QStringLiteral("normalized_image_correction"),
                     QJsonObject{{QStringLiteral("sample_offset"), source.value(QStringLiteral("sample_offset_px"))},
                                 {QStringLiteral("sample_sample"), source.value(QStringLiteral("sample_sample_px"))},
                                 {QStringLiteral("sample_line"), source.value(QStringLiteral("sample_line_px"))},
                                 {QStringLiteral("line_offset"), source.value(QStringLiteral("line_offset_px"))},
                                 {QStringLiteral("line_sample"), source.value(QStringLiteral("line_sample_px"))},
                                 {QStringLiteral("line_line"), source.value(QStringLiteral("line_line_px"))}}}};
            }
            if (source.value(QStringLiteral("model")) == QStringLiteral("affine_ground_coordinates_v1"))
            {
                return {
                    {QStringLiteral("correction_domain"), QStringLiteral("ground_coordinates")},
                    {QStringLiteral("ground_correction"),
                     QJsonObject{
                         {QStringLiteral("sample_offset"), source.value(QStringLiteral("sample_offset_px"))},
                         {QStringLiteral("line_offset"), source.value(QStringLiteral("line_offset_px"))},
                         {QStringLiteral("sample_longitude"),
                          source.value(QStringLiteral("sample_longitude_px_per_degree"))},
                         {QStringLiteral("sample_latitude"),
                          source.value(QStringLiteral("sample_latitude_px_per_degree"))},
                         {QStringLiteral("sample_height"), source.value(QStringLiteral("sample_height_px_per_m"))},
                         {QStringLiteral("line_longitude"),
                          source.value(QStringLiteral("line_longitude_px_per_degree"))},
                         {QStringLiteral("line_latitude"), source.value(QStringLiteral("line_latitude_px_per_degree"))},
                         {QStringLiteral("line_height"), source.value(QStringLiteral("line_height_px_per_m"))}}}};
            }
            return {};
        }

        QJsonObject timeReference(const QJsonValue& scale, const QJsonValue& seconds)
        {
            return {{QStringLiteral("scale"), scale}, {QStringLiteral("seconds"), seconds}};
        }

        QJsonObject rotationTrajectory(const QJsonObject& source, const QJsonValue& scale)
        {
            QJsonArray samples;
            for (const QJsonValue& value : source.value(QStringLiteral("samples")).toArray())
            {
                const QJsonObject sample = value.toObject();
                samples.append(QJsonObject{
                    {QStringLiteral("time"), timeReference(scale, sample.value(QStringLiteral("time_seconds")))},
                    {QStringLiteral("quaternion"), sample.value(QStringLiteral("quaternion_scalar_first"))}});
            }
            return {{QStringLiteral("constant_rotation"), source.value(QStringLiteral("constant_rotation"))},
                    {QStringLiteral("samples"), samples}};
        }

        QJsonObject lineTrajectory(const QJsonObject& source, int schema_version)
        {
            const QJsonValue scale = source.value(QStringLiteral("time_scale"));
            if (source.value(QStringLiteral("representation")) == QStringLiteral("frame_composed"))
            {
                QJsonArray states;
                for (const QJsonValue& value : source.value(QStringLiteral("inertial_states")).toArray())
                {
                    const QJsonObject state = value.toObject();
                    states.append(QJsonObject{
                        {QStringLiteral("time"), timeReference(scale, state.value(QStringLiteral("time_seconds")))},
                        {QStringLiteral("position_m"), state.value(QStringLiteral("position_m"))},
                        {QStringLiteral("velocity_mps"), state.value(QStringLiteral("velocity_m_per_s"))}});
                }
                return {{QStringLiteral("kind"), QStringLiteral("frame_composed")},
                        {QStringLiteral("inertial_states"), states},
                        {QStringLiteral("inertial_to_world"),
                         rotationTrajectory(objectValue(source, QStringLiteral("inertial_to_world")), scale)},
                        {QStringLiteral("inertial_to_sensor"),
                         rotationTrajectory(objectValue(source, QStringLiteral("inertial_to_sensor")), scale)}};
            }
            if (source.value(QStringLiteral("representation")) != QStringLiteral("direct_pose_samples"))
            {
                return {{QStringLiteral("kind"), source.value(QStringLiteral("representation"))}};
            }
            QJsonArray samples;
            for (const QJsonValue& value : source.value(QStringLiteral("samples")).toArray())
            {
                const QJsonObject sample = value.toObject();
                QJsonObject canonical{
                    {QStringLiteral("time"), timeReference(scale, sample.value(QStringLiteral("time_seconds")))},
                    {QStringLiteral("center"), sample.value(QStringLiteral("center_m"))},
                    {QStringLiteral("camera_to_world_rotation"),
                     sample.value(QStringLiteral("camera_to_world_rotation"))}};
                if (schema_version == placamera::LineScanModel::InstanceSchemaVersion)
                {
                    canonical.insert(QStringLiteral("constraints"), sample.value(QStringLiteral("constraints")));
                }
                samples.append(canonical);
            }
            return {{QStringLiteral("kind"), QStringLiteral("samples")}, {QStringLiteral("samples"), samples}};
        }

        QJsonObject lineInstanceState(const QJsonObject& instance, int schema_version)
        {
            const QJsonObject source_state = objectValue(instance, QStringLiteral("state"));
            const QJsonObject source_trajectory = objectValue(source_state, QStringLiteral("trajectory"));
            const QJsonObject source_timing = objectValue(source_state, QStringLiteral("line_timing"));
            const QJsonArray segments = source_timing.value(QStringLiteral("segments")).toArray();
            const QJsonObject first = segments.isEmpty() ? QJsonObject{} : segments.at(0).toObject();
            QJsonArray canonical_segments;
            for (const QJsonValue& item : segments)
            {
                const QJsonObject segment = item.toObject();
                canonical_segments.append(QJsonObject{
                    {QStringLiteral("start_line"), segment.value(QStringLiteral("start_line"))},
                    {QStringLiteral("start_time_seconds"), segment.value(QStringLiteral("start_time_seconds"))},
                    {QStringLiteral("seconds_per_line"), segment.value(QStringLiteral("seconds_per_line"))}});
            }
            QJsonObject bias = objectValue(source_state, QStringLiteral("bias"));
            if (bias.isEmpty())
            {
                bias = {{QStringLiteral("translation_m"), QJsonArray{0.0, 0.0, 0.0}},
                        {QStringLiteral("rotation_rad"), QJsonArray{0.0, 0.0, 0.0}},
                        {QStringLiteral("time_offset_seconds"), 0.0}};
            }
            QJsonObject result{
                {QStringLiteral("trajectory"), lineTrajectory(source_trajectory, schema_version)},
                {QStringLiteral("timing"),
                 QJsonObject{{QStringLiteral("line_zero"), first.value(QStringLiteral("start_line"))},
                             {QStringLiteral("start_time_seconds"), first.value(QStringLiteral("start_time_seconds"))},
                             {QStringLiteral("seconds_per_line"), first.value(QStringLiteral("seconds_per_line"))},
                             {QStringLiteral("time_scale"), source_timing.value(QStringLiteral("time_scale"))},
                             {QStringLiteral("segments"), canonical_segments}}},
                {QStringLiteral("bias"), bias}};
            if (schema_version == placamera::LineScanModel::InstanceSchemaVersion)
            {
                result.insert(QStringLiteral("time_offset_prior"),
                              source_state.value(QStringLiteral("time_offset_prior")));
            }
            return result;
        }

        QJsonValue instanceCaptureTime(const QJsonObject& instance)
        {
            const QJsonObject capture =
                objectValue(objectValue(instance, QStringLiteral("state")), QStringLiteral("capture_time"));
            return capture.isEmpty() ? QJsonValue(QJsonValue::Null)
                                     : QJsonValue(timeReference(capture.value(QStringLiteral("time_scale")),
                                                                capture.value(QStringLiteral("seconds"))));
        }

        std::string compactJson(const QJsonObject& object)
        {
            return QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
        }

        QJsonObject definitionEnvelope(const QJsonObject& definition)
        {
            const QString model = requiredString(definition, QStringLiteral("model_type"));
            const QJsonObject parameters = objectValue(definition, QStringLiteral("parameters"));
            const int schema_version = definition.value(QStringLiteral("schema_version")).toInt();
            QJsonObject canonical;
            if (isCentralCameraModel(model))
            {
                canonical = schema_version == 1 ? legacyFrameDefinitionParameters(parameters) : parameters;
            }
            else if (model == QStringLiteral("rpc00b"))
            {
                canonical = rpcDefinitionParameters(parameters);
            }
            else if (model == QStringLiteral("planetary_linescan"))
            {
                canonical = lineDefinitionParameters(parameters);
            }
            else
            {
                return {};
            }
            return {{QStringLiteral("kind"), QStringLiteral("placamera.definition")},
                    {QStringLiteral("envelope_schema"), 1},
                    {QStringLiteral("model_type"), model},
                    {QStringLiteral("definition_id"), definition.value(QStringLiteral("id"))},
                    {QStringLiteral("ground_frame"), definition.value(QStringLiteral("frame"))},
                    {QStringLiteral("parameter_schema"), definition.value(QStringLiteral("schema_version"))},
                    {QStringLiteral("parameters"), canonical}};
        }

        QJsonObject instanceEnvelope(const QJsonObject& instance, const QString& model)
        {
            const QJsonObject size = objectValue(instance, QStringLiteral("image_size"));
            const int schema_version = instance.value(QStringLiteral("schema_version")).toInt();
            QJsonObject state;
            if (isCentralCameraModel(model))
            {
                state = frameInstanceState(instance, schema_version);
            }
            else if (model == QStringLiteral("rpc00b"))
            {
                state = rpcInstanceState(instance, schema_version);
                if (objectValue(objectValue(instance, QStringLiteral("state")), QStringLiteral("image_correction"))
                        .isEmpty())
                {
                    const QJsonObject identity{{QStringLiteral("sample_offset"), 0.0},
                                               {QStringLiteral("sample_sample"), 0.0},
                                               {QStringLiteral("sample_line"), 0.0},
                                               {QStringLiteral("line_offset"), 0.0},
                                               {QStringLiteral("line_sample"), 0.0},
                                               {QStringLiteral("line_line"), 0.0}};
                    state = schema_version == 1
                                ? QJsonObject{{QStringLiteral("image_correction"), identity}}
                                : QJsonObject{{QStringLiteral("correction_domain"), QStringLiteral("normalized_image")},
                                              {QStringLiteral("normalized_image_correction"), identity}};
                }
            }
            else if (model == QStringLiteral("planetary_linescan"))
            {
                state = lineInstanceState(instance, schema_version);
            }
            else
            {
                return {};
            }
            return {{QStringLiteral("kind"), QStringLiteral("placamera.instance")},
                    {QStringLiteral("envelope_schema"), 1},
                    {QStringLiteral("model_type"), model},
                    {QStringLiteral("instance_id"), instance.value(QStringLiteral("id"))},
                    {QStringLiteral("image_id"), instance.value(QStringLiteral("image_uuid"))},
                    {QStringLiteral("definition_id"), instance.value(QStringLiteral("definition_id"))},
                    {QStringLiteral("instance_schema"), instance.value(QStringLiteral("schema_version"))},
                    {QStringLiteral("image_size"), size},
                    {QStringLiteral("capture_time"), instanceCaptureTime(instance)},
                    {QStringLiteral("state"), state}};
        }

        void writeProjectFrameState(QJsonObject* record, const QJsonObject& state, const QString& frame)
        {
            const QJsonObject pose = objectValue(state, QStringLiteral("pose"));
            record->insert(QStringLiteral("pose"),
                           QJsonObject{{QStringLiteral("frame"), frame},
                                       {QStringLiteral("center_m"), pose.value(QStringLiteral("center"))},
                                       {QStringLiteral("camera_to_world_rotation"),
                                        pose.value(QStringLiteral("camera_to_world_rotation"))}});
            record->insert(QStringLiteral("acquisition"), state.value(QStringLiteral("acquisition")));
        }

        QJsonObject projectRotationTrajectory(const QJsonObject& trajectory)
        {
            QJsonArray samples;
            for (const QJsonValue& value : trajectory.value(QStringLiteral("samples")).toArray())
            {
                const QJsonObject sample = value.toObject();
                samples.append(QJsonObject{
                    {QStringLiteral("time_seconds"),
                     objectValue(sample, QStringLiteral("time")).value(QStringLiteral("seconds"))},
                    {QStringLiteral("quaternion_scalar_first"), sample.value(QStringLiteral("quaternion"))}});
            }
            return {{QStringLiteral("constant_rotation"), trajectory.value(QStringLiteral("constant_rotation"))},
                    {QStringLiteral("samples"), samples}};
        }

        QJsonObject projectLineTrajectory(const QJsonObject& trajectory)
        {
            if (trajectory.value(QStringLiteral("kind")) == QStringLiteral("frame_composed"))
            {
                QJsonArray states;
                QString scale;
                for (const QJsonValue& value : trajectory.value(QStringLiteral("inertial_states")).toArray())
                {
                    const QJsonObject state = value.toObject();
                    const QJsonObject time = objectValue(state, QStringLiteral("time"));
                    if (scale.isEmpty())
                    {
                        scale = requiredString(time, QStringLiteral("scale"));
                    }
                    states.append(
                        QJsonObject{{QStringLiteral("time_seconds"), time.value(QStringLiteral("seconds"))},
                                    {QStringLiteral("position_m"), state.value(QStringLiteral("position_m"))},
                                    {QStringLiteral("velocity_m_per_s"), state.value(QStringLiteral("velocity_mps"))}});
                }
                return {{QStringLiteral("time_scale"), scale},
                        {QStringLiteral("representation"), QStringLiteral("frame_composed")},
                        {QStringLiteral("inertial_states"), states},
                        {QStringLiteral("inertial_to_world"),
                         projectRotationTrajectory(objectValue(trajectory, QStringLiteral("inertial_to_world")))},
                        {QStringLiteral("inertial_to_sensor"),
                         projectRotationTrajectory(objectValue(trajectory, QStringLiteral("inertial_to_sensor")))}};
            }
            QJsonArray samples;
            QString scale;
            for (const QJsonValue& value : trajectory.value(QStringLiteral("samples")).toArray())
            {
                const QJsonObject sample = value.toObject();
                const QJsonObject time = objectValue(sample, QStringLiteral("time"));
                if (scale.isEmpty())
                {
                    scale = requiredString(time, QStringLiteral("scale"));
                }
                samples.append(
                    QJsonObject{{QStringLiteral("time_seconds"), time.value(QStringLiteral("seconds"))},
                                {QStringLiteral("center_m"), sample.value(QStringLiteral("center"))},
                                {QStringLiteral("camera_to_world_rotation"),
                                 sample.value(QStringLiteral("camera_to_world_rotation"))},
                                {QStringLiteral("constraints"), sample.value(QStringLiteral("constraints"))}});
            }
            return {{QStringLiteral("time_scale"), scale},
                    {QStringLiteral("representation"), QStringLiteral("direct_pose_samples")},
                    {QStringLiteral("samples"), samples}};
        }

        QJsonObject projectStateForModel(const QJsonObject& state, const QString& model)
        {
            if (model == QStringLiteral("rpc00b"))
            {
                if (state.value(QStringLiteral("correction_domain")) == QStringLiteral("normalized_image"))
                {
                    const QJsonObject correction = objectValue(state, QStringLiteral("normalized_image_correction"));
                    return {
                        {QStringLiteral("image_correction"),
                         QJsonObject{
                             {QStringLiteral("model"), QStringLiteral("affine_normalized_v1")},
                             {QStringLiteral("sample_offset_px"), correction.value(QStringLiteral("sample_offset"))},
                             {QStringLiteral("sample_sample_px"), correction.value(QStringLiteral("sample_sample"))},
                             {QStringLiteral("sample_line_px"), correction.value(QStringLiteral("sample_line"))},
                             {QStringLiteral("line_offset_px"), correction.value(QStringLiteral("line_offset"))},
                             {QStringLiteral("line_sample_px"), correction.value(QStringLiteral("line_sample"))},
                             {QStringLiteral("line_line_px"), correction.value(QStringLiteral("line_line"))}}}};
                }
                const QJsonObject correction = objectValue(state, QStringLiteral("ground_correction"));
                return {
                    {QStringLiteral("image_correction"),
                     QJsonObject{
                         {QStringLiteral("model"), QStringLiteral("affine_ground_coordinates_v1")},
                         {QStringLiteral("sample_offset_px"), correction.value(QStringLiteral("sample_offset"))},
                         {QStringLiteral("line_offset_px"), correction.value(QStringLiteral("line_offset"))},
                         {QStringLiteral("sample_longitude_px_per_degree"),
                          correction.value(QStringLiteral("sample_longitude"))},
                         {QStringLiteral("sample_latitude_px_per_degree"),
                          correction.value(QStringLiteral("sample_latitude"))},
                         {QStringLiteral("sample_height_px_per_m"), correction.value(QStringLiteral("sample_height"))},
                         {QStringLiteral("line_longitude_px_per_degree"),
                          correction.value(QStringLiteral("line_longitude"))},
                         {QStringLiteral("line_latitude_px_per_degree"),
                          correction.value(QStringLiteral("line_latitude"))},
                         {QStringLiteral("line_height_px_per_m"), correction.value(QStringLiteral("line_height"))}}}};
            }
            if (model == QStringLiteral("planetary_linescan"))
            {
                const QJsonObject trajectory = objectValue(state, QStringLiteral("trajectory"));
                const QJsonObject timing = objectValue(state, QStringLiteral("timing"));
                QJsonArray segments;
                for (const QJsonValue& value : timing.value(QStringLiteral("segments")).toArray())
                {
                    const QJsonObject segment = value.toObject();
                    segments.append(QJsonObject{
                        {QStringLiteral("start_line"), segment.value(QStringLiteral("start_line"))},
                        {QStringLiteral("start_time_seconds"), segment.value(QStringLiteral("start_time_seconds"))},
                        {QStringLiteral("seconds_per_line"), segment.value(QStringLiteral("seconds_per_line"))}});
                }
                return {{QStringLiteral("trajectory"), projectLineTrajectory(trajectory)},
                        {QStringLiteral("line_timing"),
                         QJsonObject{{QStringLiteral("time_scale"), timing.value(QStringLiteral("time_scale"))},
                                     {QStringLiteral("segments"), segments}}},
                        {QStringLiteral("bias"), state.value(QStringLiteral("bias"))},
                        {QStringLiteral("time_offset_prior"), state.value(QStringLiteral("time_offset_prior"))}};
            }
            return state;
        }

        QJsonObject projectFrameDefinition(const placamera::FramePinholeDefinition& definition)
        {
            const auto encoded = placamera::encodeCameraDefinitionJson(definition);
            if (!encoded)
            {
                return {};
            }
            const QJsonObject envelope = QJsonDocument::fromJson(QByteArray::fromStdString(encoded.value())).object();
            return {{QStringLiteral("id"), envelope.value(QStringLiteral("definition_id"))},
                    {QStringLiteral("model_type"), envelope.value(QStringLiteral("model_type"))},
                    {QStringLiteral("schema_version"), envelope.value(QStringLiteral("parameter_schema"))},
                    {QStringLiteral("frame"), envelope.value(QStringLiteral("ground_frame"))},
                    {QStringLiteral("parameters"), envelope.value(QStringLiteral("parameters"))}};
        }

        QJsonArray rpcCoefficients(const placamera::RpcCoefficients& values)
        {
            QJsonArray result;
            for (double value : values)
            {
                result.append(value);
            }
            return result;
        }

        QJsonObject projectRpcDefinition(const placamera::RpcDefinition& definition)
        {
            const auto& rpc = definition.parameters();
            const auto& ellipsoid = definition.ellipsoid();
            QJsonObject parameters{{QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                                   {QStringLiteral("line_offset"), rpc.lineOffset},
                                   {QStringLiteral("sample_offset"), rpc.sampleOffset},
                                   {QStringLiteral("latitude_offset"), rpc.latitudeOffset},
                                   {QStringLiteral("longitude_offset"), rpc.longitudeOffset},
                                   {QStringLiteral("height_offset"), rpc.heightOffset},
                                   {QStringLiteral("line_scale"), rpc.lineScale},
                                   {QStringLiteral("sample_scale"), rpc.sampleScale},
                                   {QStringLiteral("latitude_scale"), rpc.latitudeScale},
                                   {QStringLiteral("longitude_scale"), rpc.longitudeScale},
                                   {QStringLiteral("height_scale"), rpc.heightScale},
                                   {QStringLiteral("line_numerator"), rpcCoefficients(rpc.lineNumerator)},
                                   {QStringLiteral("line_denominator"), rpcCoefficients(rpc.lineDenominator)},
                                   {QStringLiteral("sample_numerator"), rpcCoefficients(rpc.sampleNumerator)},
                                   {QStringLiteral("sample_denominator"), rpcCoefficients(rpc.sampleDenominator)},
                                   {QStringLiteral("semi_major_axis_m"), ellipsoid.semiMajorAxisMeters},
                                   {QStringLiteral("inverse_flattening"), ellipsoid.inverseFlattening}};
            if (definition.groundFrame().value() == "EPSG:4978" && ellipsoid.semiMajorAxisMeters == 6378137.0 &&
                ellipsoid.inverseFlattening == 298.257223563)
            {
                parameters.insert(QStringLiteral("ground_crs"), QStringLiteral("EPSG:4979"));
                parameters.insert(QStringLiteral("height_datum"), QStringLiteral("WGS84_ellipsoidal"));
            }
            if (rpc.errorBiasMeters)
            {
                parameters.insert(QStringLiteral("error_bias_m"), *rpc.errorBiasMeters);
            }
            if (rpc.errorRandomMeters)
            {
                parameters.insert(QStringLiteral("error_random_m"), *rpc.errorRandomMeters);
            }
            return {{QStringLiteral("id"), QString::fromStdString(definition.definitionId().value())},
                    {QStringLiteral("model_type"), QStringLiteral("rpc00b")},
                    {QStringLiteral("schema_version"), definition.parameterSchemaVersion()},
                    {QStringLiteral("frame"), QString::fromStdString(definition.groundFrame().value())},
                    {QStringLiteral("parameters"), parameters}};
        }

        QJsonObject projectLineDefinition(const placamera::LineScanDefinition& definition)
        {
            const auto encoded = placamera::encodeCameraDefinitionJson(definition);
            if (!encoded)
            {
                return {};
            }
            const QJsonObject envelope = QJsonDocument::fromJson(QByteArray::fromStdString(encoded.value())).object();
            return {{QStringLiteral("id"), QString::fromStdString(definition.definitionId().value())},
                    {QStringLiteral("model_type"), QStringLiteral("planetary_linescan")},
                    {QStringLiteral("schema_version"), definition.parameterSchemaVersion()},
                    {QStringLiteral("frame"), QString::fromStdString(definition.groundFrame().value())},
                    {QStringLiteral("parameters"), envelope.value(QStringLiteral("parameters"))}};
        }

        bool sameDefinition(const QJsonObject& projectRecord, const placamera::CameraDefinition& definition)
        {
            const auto projectDefinition =
                placamera::decodeCameraDefinitionJson(compactJson(definitionEnvelope(projectRecord)));
            if (!projectDefinition)
            {
                return false;
            }
            placamera::ModelRegistry registry;
            if (!placamera::registerBuiltinJsonModelFactories(registry))
            {
                return false;
            }
            const auto projectModel = registry.createDefinition(projectDefinition.value());
            if (!projectModel)
            {
                return false;
            }
            const auto encodedProject = placamera::encodeCameraDefinitionJson(*projectModel.value());
            const auto encodedUpdate = placamera::encodeCameraDefinitionJson(definition);
            return encodedProject.ok() && encodedUpdate.ok() && encodedProject.value() == encodedUpdate.value();
        }

    } // namespace

    ProjectCameraLoadResult loadProjectCameras(const QJsonObject& projectFiles)
    {
        ProjectCameraLoadResult result;
        for (const QString& key :
             {QStringLiteral("images"), QStringLiteral("camera_definitions"), QStringLiteral("camera_instances")})
        {
            if (!projectFiles.value(key).isArray())
            {
                result.errors.append(QStringLiteral("project_files.%1 must be an array").arg(key));
            }
        }
        if (!result.errors.isEmpty())
        {
            return result;
        }
        const QJsonArray images = projectFiles.value(QStringLiteral("images")).toArray();
        QSet<QString> image_ids;
        for (int index = 0; index < images.size(); ++index)
        {
            if (!images.at(index).isObject())
            {
                result.errors.append(QStringLiteral("images[%1] must be an object").arg(index));
                continue;
            }
            const QJsonObject image = images.at(index).toObject();
            const QString image_id = requiredString(image, QStringLiteral("image_uuid"));
            if (image_id.isEmpty() || image.value(QStringLiteral("image_uuid")).toString() != image_id ||
                image_ids.contains(image_id))
            {
                result.errors.append(
                    QStringLiteral("images[%1] has an empty, untrimmed or duplicate image_uuid").arg(index));
            }
            image_ids.insert(image_id);
            if (image.contains(QStringLiteral("camera")) || image.contains(QStringLiteral("camera_file")))
            {
                result.errors.append(QStringLiteral("images[%1] contains legacy embedded camera fields").arg(index));
            }
        }
        if (!result.errors.isEmpty())
        {
            return result;
        }

        placamera::ModelRegistry registry;
        if (!placamera::registerBuiltinJsonModelFactories(registry))
        {
            result.errors.append(QStringLiteral("failed to register built-in PlaCamera state factories"));
            return result;
        }
        DefinitionStates definitions;
        const QJsonArray definition_records = projectFiles.value(QStringLiteral("camera_definitions")).toArray();
        for (int index = 0; index < definition_records.size(); ++index)
        {
            if (!definition_records.at(index).isObject())
            {
                result.errors.append(QStringLiteral("camera_definitions[%1] must be an object").arg(index));
                return result;
            }
            const QJsonObject record = definition_records.at(index).toObject();
            const QString id = requiredString(record, QStringLiteral("id"));
            if (id.isEmpty() || record.value(QStringLiteral("id")).toString() != id ||
                definitions.contains(id.toStdString()))
            {
                result.errors.append(
                    QStringLiteral("duplicate, empty or untrimmed camera definition id at index %1").arg(index));
                return result;
            }
            if (!record.value(QStringLiteral("parameters")).isObject())
            {
                result.errors.append(QStringLiteral("camera_definitions[%1].parameters must be an object").arg(index));
                return result;
            }
            const QString model_type = requiredString(record, QStringLiteral("model_type"));
            const int schema_version = record.value(QStringLiteral("schema_version")).toInt();
            const bool supported_schema =
                (isCentralCameraModel(model_type) && schema_version >= 1 &&
                 schema_version <= placamera::FramePinholeDefinition::ParameterSchemaVersion &&
                 exactSchemaVersion(record, schema_version) &&
                 (schema_version >= 3 || model_type == QStringLiteral("frame_pinhole"))) ||
                (model_type == QStringLiteral("rpc00b") && exactSchemaVersion(record, 1)) ||
                (model_type == QStringLiteral("planetary_linescan") &&
                 (exactSchemaVersion(record, 2) ||
                  exactSchemaVersion(record, placamera::LineScanDefinition::ParameterSchemaVersion)));
            if (!supported_schema)
            {
                result.errors.append(
                    QStringLiteral("camera_definitions[%1] has an unsupported model or schema_version").arg(index));
                return result;
            }
            const QJsonObject parameters = record.value(QStringLiteral("parameters")).toObject();
            if (model_type == QStringLiteral("frame_pinhole") && schema_version == 1)
            {
                const QJsonValue convention = parameters.value(QStringLiteral("pixel_convention"));
                if (convention != QStringLiteral("center") && convention != QStringLiteral("corner"))
                {
                    result.errors.append(QStringLiteral("camera definition %1 has invalid pixel_convention").arg(id));
                    return result;
                }
            }
            else if (model_type == QStringLiteral("rpc00b"))
            {
                if (parameters.value(QStringLiteral("rpc_spec")) != QStringLiteral("RPC00B"))
                {
                    result.errors.append(QStringLiteral("camera definition %1 requires rpc_spec=RPC00B").arg(id));
                    return result;
                }
                const QJsonValue major_axis = parameters.value(QStringLiteral("semi_major_axis_m"));
                const QJsonValue flattening = parameters.value(QStringLiteral("inverse_flattening"));
                if ((!major_axis.isUndefined() && !major_axis.isDouble()) ||
                    (!flattening.isUndefined() && !flattening.isDouble()))
                {
                    result.errors.append(
                        QStringLiteral("camera definition %1 has invalid ellipsoid parameters").arg(id));
                    return result;
                }
                if (parameters.contains(QStringLiteral("ground_crs")) ||
                    parameters.contains(QStringLiteral("height_datum")))
                {
                    if (parameters.value(QStringLiteral("ground_crs")) != QStringLiteral("EPSG:4979") ||
                        parameters.value(QStringLiteral("height_datum")) != QStringLiteral("WGS84_ellipsoidal") ||
                        record.value(QStringLiteral("frame")) != QStringLiteral("EPSG:4978") ||
                        (major_axis.isDouble() && major_axis.toDouble() != 6378137.0) ||
                        (flattening.isDouble() && flattening.toDouble() != 298.257223563))
                    {
                        result.errors.append(
                            QStringLiteral("camera definition %1 has inconsistent RPC CRS metadata").arg(id));
                        return result;
                    }
                }
            }
            const auto decoded = placamera::decodeCameraDefinitionJson(compactJson(definitionEnvelope(record)));
            if (!decoded)
            {
                result.errors.append(
                    QStringLiteral("camera definition %1: %2").arg(id, QString::fromStdString(decoded.message())));
                return result;
            }
            definitions.emplace(id.toStdString(), decoded.value());
        }

        QSet<QString> instance_ids;
        QSet<QString> bound_images;
        placamera::CameraInstanceSet pending;
        const QJsonArray instance_records = projectFiles.value(QStringLiteral("camera_instances")).toArray();
        for (int index = 0; index < instance_records.size(); ++index)
        {
            if (!instance_records.at(index).isObject())
            {
                result.errors.append(QStringLiteral("camera_instances[%1] must be an object").arg(index));
                return result;
            }
            const QJsonObject record = instance_records.at(index).toObject();
            const QString instance_id = requiredString(record, QStringLiteral("id"));
            const QString image_id = requiredString(record, QStringLiteral("image_uuid"));
            const QString definition_id = requiredString(record, QStringLiteral("definition_id"));
            if ((record.contains(QStringLiteral("state")) && !record.value(QStringLiteral("state")).isObject()) ||
                (record.contains(QStringLiteral("pose")) && !record.value(QStringLiteral("pose")).isObject()) ||
                (record.contains(QStringLiteral("acquisition")) &&
                 !record.value(QStringLiteral("acquisition")).isObject()))
            {
                result.errors.append(
                    QStringLiteral("camera_instances[%1].state, pose or acquisition must be an object").arg(index));
                return result;
            }
            if (instance_id.isEmpty() || record.value(QStringLiteral("id")).toString() != instance_id ||
                instance_ids.contains(instance_id) || image_id.isEmpty() ||
                record.value(QStringLiteral("image_uuid")).toString() != image_id ||
                record.value(QStringLiteral("definition_id")).toString() != definition_id ||
                bound_images.contains(image_id))
            {
                result.errors.append(QStringLiteral("duplicate camera instance identity at index %1").arg(index));
                return result;
            }
            if (!image_ids.contains(image_id) || !definitions.contains(definition_id.toStdString()))
            {
                result.errors.append(
                    QStringLiteral("camera instance %1 references an unknown image or definition").arg(instance_id));
                return result;
            }
            const auto& definition = definitions.at(definition_id.toStdString());
            const QString model_type = QString::fromStdString(definition.parameters.modelType);
            const bool supported_schema =
                (isCentralCameraModel(model_type) &&
                 (exactSchemaVersion(record, 1) ||
                  exactSchemaVersion(record, placamera::FramePinholeModel::InstanceSchemaVersion))) ||
                (model_type == QStringLiteral("rpc00b") &&
                 (exactSchemaVersion(record, 1) ||
                  exactSchemaVersion(record, placamera::RpcModel::InstanceSchemaVersion))) ||
                (model_type == QStringLiteral("planetary_linescan") &&
                 (exactSchemaVersion(record, 1) ||
                  exactSchemaVersion(record, placamera::LineScanModel::InstanceSchemaVersion)));
            if (!supported_schema)
            {
                result.errors.append(
                    QStringLiteral("camera_instances[%1] has an unsupported model or schema_version").arg(index));
                return result;
            }
            if (isCentralCameraModel(model_type))
            {
                const QJsonObject pose = record.value(QStringLiteral("pose")).toObject();
                if (pose.value(QStringLiteral("frame")) != QString::fromStdString(definition.groundFrame.value()))
                {
                    result.errors.append(
                        QStringLiteral("camera instance %1 pose.frame must match definition frame").arg(instance_id));
                    return result;
                }
            }
            const auto decoded = placamera::decodeCameraInstanceJson(compactJson(instanceEnvelope(record, model_type)));
            if (!decoded)
            {
                result.errors.append(QStringLiteral("camera instance %1: %2")
                                         .arg(instance_id, QString::fromStdString(decoded.message())));
                return result;
            }
            const auto definition_model = registry.createDefinition(definition);
            if (!definition_model)
            {
                result.errors.append(QStringLiteral("camera definition %1: %2")
                                         .arg(definition_id, QString::fromStdString(definition_model.message())));
                return result;
            }
            auto created = registry.createInstance(decoded.value(), definition_model.value());
            if (!created)
            {
                result.errors.append(QStringLiteral("camera instance %1: %2")
                                         .arg(instance_id, QString::fromStdString(created.message())));
                return result;
            }
            if (!pending.add(created.value()))
            {
                result.errors.append(QStringLiteral("failed to add camera instance %1").arg(instance_id));
                return result;
            }
            instance_ids.insert(instance_id);
            bound_images.insert(image_id);
        }
        result.instances = std::move(pending);
        return result;
    }

    ProjectCameraWriteResult insertProjectCameras(QJsonObject* projectFiles,
                                                  const placamera::CameraInstanceSet& instances)
    {
        ProjectCameraWriteResult result;
        if (!projectFiles || instances.empty())
        {
            result.errors.append(QStringLiteral("project document is null or camera instance set is empty"));
            return result;
        }
        QJsonObject normalized = *projectFiles;
        for (const QString& key : {QStringLiteral("camera_definitions"), QStringLiteral("camera_instances")})
        {
            if (!normalized.contains(key))
            {
                normalized.insert(key, QJsonArray{});
            }
        }
        const auto current = loadProjectCameras(normalized);
        if (!current.ok())
        {
            result.errors = current.errors;
            return result;
        }

        QJsonArray definitions = normalized.value(QStringLiteral("camera_definitions")).toArray();
        QJsonArray records = normalized.value(QStringLiteral("camera_instances")).toArray();
        const QJsonArray images = normalized.value(QStringLiteral("images")).toArray();
        QSet<QString> instance_ids;
        for (const QJsonValue& value : records)
        {
            instance_ids.insert(requiredString(value.toObject(), QStringLiteral("id")));
        }
        int pending_inserted_count = 0;
        for (const auto& model : instances.values())
        {
            const auto* frame = dynamic_cast<const placamera::FramePinholeModel*>(model.get());
            const auto* rpc = dynamic_cast<const placamera::RpcModel*>(model.get());
            const auto* line = dynamic_cast<const placamera::LineScanModel*>(model.get());
            const QString image_id = QString::fromStdString(model->imageId().value());
            const QString instance_id = QString::fromStdString(model->instanceId().value());
            const QString definition_id = QString::fromStdString(model->definitionId().value());
            if (!frame && !rpc && !line)
            {
                result.errors.append(
                    QStringLiteral("new camera instance %1 has an unsupported model").arg(instance_id));
                return result;
            }
            int image_matches = 0;
            for (const QJsonValue& value : images)
            {
                image_matches += requiredString(value.toObject(), QStringLiteral("image_uuid")) == image_id ? 1 : 0;
            }
            if (image_matches != 1 || current.instances.forImage(model->imageId()).ok() ||
                instance_ids.contains(instance_id))
            {
                result.errors.append(
                    QStringLiteral("new camera instance %1 has a missing or occupied image/instance id")
                        .arg(instance_id));
                return result;
            }
            instance_ids.insert(instance_id);

            bool definition_found = false;
            for (const QJsonValue& value : definitions)
            {
                const QJsonObject record = value.toObject();
                if (requiredString(record, QStringLiteral("id")) != definition_id)
                {
                    continue;
                }
                definition_found = true;
                if (!sameDefinition(record, model->definition()))
                {
                    result.errors.append(
                        QStringLiteral("camera definition %1 conflicts with project calibration").arg(definition_id));
                    return result;
                }
                break;
            }
            if (!definition_found)
            {
                const QJsonObject definition = frame ? projectFrameDefinition(frame->pinholeDefinition())
                                               : rpc ? projectRpcDefinition(rpc->rpcDefinition())
                                                     : projectLineDefinition(line->lineScanDefinition());
                if (definition.isEmpty())
                {
                    result.errors.append(
                        QStringLiteral("camera definition %1 could not be encoded").arg(definition_id));
                    return result;
                }
                definitions.append(definition);
            }

            const auto encoded = placamera::encodeCameraInstanceJson(*model);
            if (!encoded)
            {
                result.errors.append(QStringLiteral("camera instance %1: %2")
                                         .arg(instance_id, QString::fromStdString(encoded.message())));
                return result;
            }
            const QJsonObject envelope = QJsonDocument::fromJson(QByteArray::fromStdString(encoded.value())).object();
            QJsonObject record{{QStringLiteral("id"), instance_id},
                               {QStringLiteral("image_uuid"), image_id},
                               {QStringLiteral("definition_id"), definition_id},
                               {QStringLiteral("schema_version"), envelope.value(QStringLiteral("instance_schema"))},
                               {QStringLiteral("image_size"), envelope.value(QStringLiteral("image_size"))}};
            const QJsonObject state = objectValue(envelope, QStringLiteral("state"));
            if (frame)
            {
                writeProjectFrameState(&record, state, QString::fromStdString(frame->groundFrame().value()));
            }
            else
            {
                record.insert(QStringLiteral("state"),
                              projectStateForModel(state, QString::fromStdString(std::string(model->modelType()))));
            }
            if (envelope.value(QStringLiteral("capture_time")).isObject())
            {
                const QJsonObject capture = envelope.value(QStringLiteral("capture_time")).toObject();
                QJsonObject project_state = objectValue(record, QStringLiteral("state"));
                project_state.insert(
                    QStringLiteral("capture_time"),
                    QJsonObject{{QStringLiteral("time_scale"), capture.value(QStringLiteral("scale"))},
                                {QStringLiteral("seconds"), capture.value(QStringLiteral("seconds"))}});
                record.insert(QStringLiteral("state"), project_state);
            }
            records.append(record);
            ++pending_inserted_count;
        }

        QJsonObject candidate = std::move(normalized);
        candidate.insert(QStringLiteral("camera_definitions"), definitions);
        candidate.insert(QStringLiteral("camera_instances"), records);
        const auto validated = loadProjectCameras(candidate);
        if (!validated.ok())
        {
            result.errors = validated.errors;
            return result;
        }
        *projectFiles = std::move(candidate);
        result.insertedCount = pending_inserted_count;
        return result;
    }

    ProjectCameraWriteResult writeProjectCameras(QJsonObject* projectFiles,
                                                 const placamera::CameraInstanceSet& instances)
    {
        ProjectCameraWriteResult result;
        if (!projectFiles || instances.empty())
        {
            result.errors.append(QStringLiteral("project document is null or camera instance set is empty"));
            return result;
        }
        const auto current = loadProjectCameras(*projectFiles);
        if (!current.ok())
        {
            result.errors = current.errors;
            return result;
        }
        QJsonArray updated_definitions = projectFiles->value(QStringLiteral("camera_definitions")).toArray();
        QJsonArray updated_instances = projectFiles->value(QStringLiteral("camera_instances")).toArray();
        int pending_updated_count = 0;
        for (const auto& model : instances.values())
        {
            const auto previous = current.instances.forImage(model->imageId());
            if (!previous || previous.value()->instanceId() != model->instanceId() ||
                previous.value()->modelType() != model->modelType() ||
                previous.value()->groundFrame() != model->groundFrame())
            {
                result.errors.append(QStringLiteral("camera instance identity, model or frame is stale for image %1")
                                         .arg(QString::fromStdString(model->imageId().value())));
                return result;
            }
            if (previous.value()->imageSize().samples != model->imageSize().samples ||
                previous.value()->imageSize().lines != model->imageSize().lines)
            {
                result.errors.append(QStringLiteral("camera image size is stale for image %1")
                                         .arg(QString::fromStdString(model->imageId().value())));
                return result;
            }
            const QString next_definition_id = QString::fromStdString(model->definitionId().value());
            bool definition_found = false;
            for (const QJsonValue& value : updated_definitions)
            {
                const QJsonObject record = value.toObject();
                if (requiredString(record, QStringLiteral("id")) != next_definition_id)
                {
                    continue;
                }
                definition_found = true;
                if (!sameDefinition(record, model->definition()))
                {
                    result.errors.append(QStringLiteral("camera definition %1 conflicts with project calibration")
                                             .arg(next_definition_id));
                    return result;
                }
                break;
            }
            if (!definition_found)
            {
                const auto* frame_definition =
                    dynamic_cast<const placamera::FramePinholeDefinition*>(&model->definition());
                const auto* rpc_definition = dynamic_cast<const placamera::RpcDefinition*>(&model->definition());
                const auto* line_definition = dynamic_cast<const placamera::LineScanDefinition*>(&model->definition());
                if (!frame_definition && !rpc_definition && !line_definition)
                {
                    result.errors.append(QStringLiteral("camera definition %1 cannot be written for model %2")
                                             .arg(next_definition_id,
                                                  QString::fromUtf8(model->modelType().data(),
                                                                    static_cast<int>(model->modelType().size()))));
                    return result;
                }
                const QJsonObject definition = frame_definition ? projectFrameDefinition(*frame_definition)
                                               : rpc_definition ? projectRpcDefinition(*rpc_definition)
                                                                : projectLineDefinition(*line_definition);
                if (definition.isEmpty())
                {
                    result.errors.append(
                        QStringLiteral("camera definition %1 could not be encoded").arg(next_definition_id));
                    return result;
                }
                updated_definitions.append(definition);
            }
            bool found = false;
            for (int index = 0; index < updated_instances.size(); ++index)
            {
                QJsonObject record = updated_instances.at(index).toObject();
                if (requiredString(record, QStringLiteral("image_uuid")) !=
                    QString::fromStdString(model->imageId().value()))
                {
                    continue;
                }
                if (requiredString(record, QStringLiteral("id")) != QString::fromStdString(model->instanceId().value()))
                {
                    result.errors.append(QStringLiteral("camera instance identity is stale for image %1")
                                             .arg(QString::fromStdString(model->imageId().value())));
                    return result;
                }
                record.insert(QStringLiteral("definition_id"), next_definition_id);
                const auto encoded = placamera::encodeCameraInstanceJson(*model);
                if (!encoded)
                {
                    result.errors.append(QStringLiteral("camera instance %1: %2")
                                             .arg(QString::fromStdString(model->instanceId().value()),
                                                  QString::fromStdString(encoded.message())));
                    return result;
                }
                const QJsonObject envelope =
                    QJsonDocument::fromJson(QByteArray::fromStdString(encoded.value())).object();
                record.insert(QStringLiteral("schema_version"), envelope.value(QStringLiteral("instance_schema")));
                record.insert(QStringLiteral("image_size"), envelope.value(QStringLiteral("image_size")));
                const QJsonObject state = envelope.value(QStringLiteral("state")).toObject();
                const QString model_type =
                    QString::fromUtf8(model->modelType().data(), static_cast<int>(model->modelType().size()));
                const bool central_camera = isCentralCameraModel(model_type);
                QJsonObject project_state = objectValue(record, QStringLiteral("state"));
                if (central_camera)
                {
                    writeProjectFrameState(&record, state, QString::fromStdString(model->groundFrame().value()));
                    project_state.remove(QStringLiteral("pose"));
                    project_state.remove(QStringLiteral("acquisition"));
                }
                else
                {
                    const QJsonObject camera_state = projectStateForModel(state, model_type);
                    for (auto it = camera_state.constBegin(); it != camera_state.constEnd(); ++it)
                    {
                        project_state.insert(it.key(), it.value());
                    }
                }
                if (envelope.value(QStringLiteral("capture_time")).isObject())
                {
                    const QJsonObject capture = envelope.value(QStringLiteral("capture_time")).toObject();
                    project_state.insert(
                        QStringLiteral("capture_time"),
                        QJsonObject{{QStringLiteral("time_scale"), capture.value(QStringLiteral("scale"))},
                                    {QStringLiteral("seconds"), capture.value(QStringLiteral("seconds"))}});
                }
                else
                {
                    project_state.remove(QStringLiteral("capture_time"));
                }
                if (project_state.isEmpty())
                {
                    record.remove(QStringLiteral("state"));
                }
                else
                {
                    record.insert(QStringLiteral("state"), project_state);
                }
                updated_instances.replace(index, record);
                found = true;
                ++pending_updated_count;
                break;
            }
            if (!found)
            {
                result.errors.append(QStringLiteral("camera instance has no project record for image %1")
                                         .arg(QString::fromStdString(model->imageId().value())));
                return result;
            }
        }
        QJsonObject candidate = *projectFiles;
        candidate.insert(QStringLiteral("camera_definitions"), updated_definitions);
        candidate.insert(QStringLiteral("camera_instances"), updated_instances);
        const auto validated = loadProjectCameras(candidate);
        if (!validated.ok())
        {
            result.errors = validated.errors;
            return result;
        }
        *projectFiles = std::move(candidate);
        result.updatedCount = pending_updated_count;
        return result;
    }

    ProjectCameraWriteResult upsertProjectCameras(QJsonObject* projectFiles,
                                                  const placamera::CameraInstanceSet& instances,
                                                  const QMap<QString, QJsonObject>& annotationsByImageId)
    {
        ProjectCameraWriteResult result;
        if (!projectFiles || instances.empty())
        {
            result.errors.append(QStringLiteral("project document is null or camera instance set is empty"));
            return result;
        }
        QJsonObject candidate = *projectFiles;
        for (const QString& key : {QStringLiteral("camera_definitions"), QStringLiteral("camera_instances")})
        {
            if (!candidate.contains(key))
            {
                candidate.insert(key, QJsonArray{});
            }
        }
        const auto current = loadProjectCameras(candidate);
        if (!current.ok())
        {
            result.errors = current.errors;
            return result;
        }

        placamera::CameraInstanceSet updates;
        placamera::CameraInstanceSet inserts;
        QSet<QString> selected_images;
        for (const auto& model : instances.values())
        {
            const QString image_id = QString::fromStdString(model->imageId().value());
            selected_images.insert(image_id);
            auto& destination = current.instances.forImage(model->imageId()).ok() ? updates : inserts;
            const auto added = destination.add(model);
            if (!added)
            {
                result.errors.append(QStringLiteral("camera instance %1: %2")
                                         .arg(QString::fromStdString(model->instanceId().value()),
                                              QString::fromStdString(added.message())));
                return result;
            }
        }
        for (auto it = annotationsByImageId.constBegin(); it != annotationsByImageId.constEnd(); ++it)
        {
            const QJsonObject annotation = it.value();
            bool known_fields = true;
            for (auto field = annotation.constBegin(); field != annotation.constEnd(); ++field)
            {
                known_fields = known_fields &&
                               (field.key() == QStringLiteral("source") || field.key() == QStringLiteral("metadata"));
            }
            if (!selected_images.contains(it.key()) || !known_fields ||
                (annotation.contains(QStringLiteral("source")) &&
                 !annotation.value(QStringLiteral("source")).isString()) ||
                (annotation.contains(QStringLiteral("metadata")) &&
                 !annotation.value(QStringLiteral("metadata")).isObject()))
            {
                result.errors.append(QStringLiteral("invalid camera annotation for image %1").arg(it.key()));
                return result;
            }
        }

        if (!updates.empty())
        {
            const auto written = writeProjectCameras(&candidate, updates);
            if (!written.ok())
            {
                return written;
            }
            result.updatedCount = written.updatedCount;
        }
        if (!inserts.empty())
        {
            const auto written = insertProjectCameras(&candidate, inserts);
            if (!written.ok())
            {
                return written;
            }
            result.insertedCount = written.insertedCount;
        }

        if (!annotationsByImageId.isEmpty())
        {
            QJsonArray records = candidate.value(QStringLiteral("camera_instances")).toArray();
            for (int index = 0; index < records.size(); ++index)
            {
                QJsonObject record = records.at(index).toObject();
                const QJsonObject annotation =
                    annotationsByImageId.value(requiredString(record, QStringLiteral("image_uuid")));
                if (annotation.isEmpty())
                {
                    continue;
                }
                QJsonObject state = objectValue(record, QStringLiteral("state"));
                for (auto it = annotation.constBegin(); it != annotation.constEnd(); ++it)
                {
                    state.insert(it.key(), it.value());
                }
                record.insert(QStringLiteral("state"), state);
                records.replace(index, record);
            }
            candidate.insert(QStringLiteral("camera_instances"), records);
        }
        const auto validated = loadProjectCameras(candidate);
        if (!validated.ok())
        {
            result.errors = validated.errors;
            result.insertedCount = 0;
            result.updatedCount = 0;
            return result;
        }
        *projectFiles = std::move(candidate);
        return result;
    }

    ProjectCameraWriteResult replaceProjectCameras(QJsonObject* projectFiles,
                                                   const std::vector<placamera::ImageId>& targetImageIds,
                                                   const placamera::CameraInstanceSet& instances,
                                                   const QMap<QString, QJsonObject>& annotationsByImageId)
    {
        ProjectCameraWriteResult result;
        if (!projectFiles || targetImageIds.empty())
        {
            result.errors.append(QStringLiteral("project document is null or target image list is empty"));
            return result;
        }
        const auto current = loadProjectCameras(*projectFiles);
        if (!current.ok())
        {
            result.errors = current.errors;
            return result;
        }

        QMap<QString, int> image_counts;
        for (const QJsonValue& value : projectFiles->value(QStringLiteral("images")).toArray())
        {
            ++image_counts[requiredString(value.toObject(), QStringLiteral("image_uuid"))];
        }
        QSet<QString> targets;
        for (const placamera::ImageId& image_id : targetImageIds)
        {
            const QString id = QString::fromStdString(image_id.value());
            if (id.isEmpty() || targets.contains(id) || image_counts.value(id) != 1)
            {
                result.errors.append(QStringLiteral("target image id is empty, duplicated or unknown: %1").arg(id));
                return result;
            }
            targets.insert(id);
        }

        QSet<QString> updated_images;
        for (const auto& model : instances.values())
        {
            const QString image_id = QString::fromStdString(model->imageId().value());
            if (!targets.contains(image_id))
            {
                result.errors.append(QStringLiteral("camera update is outside the selected images: %1").arg(image_id));
                return result;
            }
            const auto previous = current.instances.forImage(model->imageId());
            if (previous.ok() && (previous.value()->instanceId() != model->instanceId() ||
                                  previous.value()->modelType() != model->modelType() ||
                                  previous.value()->groundFrame() != model->groundFrame()))
            {
                result.errors.append(
                    QStringLiteral("camera instance identity, model or frame is stale for image %1").arg(image_id));
                return result;
            }
            for (const auto& old_model : current.instances.values())
            {
                if (old_model->imageId() != model->imageId() && old_model->instanceId() == model->instanceId())
                {
                    result.errors.append(QStringLiteral("camera instance id belongs to another image: %1")
                                             .arg(QString::fromStdString(model->instanceId().value())));
                    return result;
                }
            }
            updated_images.insert(image_id);
        }

        QJsonObject candidate = *projectFiles;
        QJsonArray retained;
        for (const QJsonValue& value : candidate.value(QStringLiteral("camera_instances")).toArray())
        {
            const QJsonObject record = value.toObject();
            const QString image_id = requiredString(record, QStringLiteral("image_uuid"));
            if (targets.contains(image_id) && !updated_images.contains(image_id))
            {
                ++result.clearedCount;
                continue;
            }
            retained.append(record);
        }
        candidate.insert(QStringLiteral("camera_instances"), retained);
        if (!instances.empty())
        {
            const auto written = upsertProjectCameras(&candidate, instances, annotationsByImageId);
            if (!written.ok())
            {
                result.errors = written.errors;
                result.clearedCount = 0;
                return result;
            }
            result.insertedCount = written.insertedCount;
            result.updatedCount = written.updatedCount;
        }
        else if (!annotationsByImageId.isEmpty())
        {
            result.errors.append(QStringLiteral("camera annotations require at least one updated instance"));
            result.clearedCount = 0;
            return result;
        }
        const auto validated = loadProjectCameras(candidate);
        if (!validated.ok())
        {
            result.errors = validated.errors;
            result.clearedCount = 0;
            result.insertedCount = 0;
            result.updatedCount = 0;
            return result;
        }
        *projectFiles = std::move(candidate);
        return result;
    }

} // namespace xjw::placamera_runtime

#include "project/ProjectCameraIO.h"

#include <QDateTime>
#include <QJsonArray>
#include <QtMath>

#include <algorithm>
#include <array>
#include <cmath>

namespace xjw::common::project
{

    namespace
    {

        QJsonObject rotationToYprDegrees(const std::array<double, 9>& rotation)
        {
            const double pitch = std::asin(std::clamp(-rotation[6], -1.0, 1.0));
            double yaw = 0.0;
            double roll = 0.0;
            if (std::abs(std::cos(pitch)) > 1e-8)
            {
                yaw = std::atan2(rotation[3], rotation[0]);
                roll = std::atan2(rotation[7], rotation[8]);
            }
            else
            {
                yaw = std::atan2(-rotation[1], rotation[4]);
            }

            return QJsonObject{{QStringLiteral("yaw_deg"), qRadiansToDegrees(yaw)},
                               {QStringLiteral("pitch_deg"), qRadiansToDegrees(pitch)},
                               {QStringLiteral("roll_deg"), qRadiansToDegrees(roll)}};
        }

    } // namespace

    QJsonObject serializeFramePinholeModel(const placamera::FramePinholeModel& camera)
    {
        const auto& definition = camera.pinholeDefinition();
        const auto& intrinsics = definition.intrinsics();
        const auto& distortion = definition.distortion();
        const auto& center = camera.pose().center;
        const auto& rotation = camera.pose().cameraToWorldRotation;

        QJsonObject result;
        result[QStringLiteral("model")] = QStringLiteral("frame_pinhole");
        result[QStringLiteral("imported_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        result[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
        result[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
        result[QStringLiteral("pixel_convention")] = QStringLiteral("center");
        result[QStringLiteral("pitch")] = intrinsics.pixelPitch;
        result[QStringLiteral("fu")] = intrinsics.focalX * intrinsics.pixelPitch;
        result[QStringLiteral("fv")] = intrinsics.focalY * intrinsics.pixelPitch;
        result[QStringLiteral("cu")] = intrinsics.principalX * intrinsics.pixelPitch;
        result[QStringLiteral("cv")] = intrinsics.principalY * intrinsics.pixelPitch;
        result[QStringLiteral("k1")] = distortion.radialK1;
        result[QStringLiteral("k2")] = distortion.radialK2;
        result[QStringLiteral("k3")] = distortion.radialK3;
        result[QStringLiteral("p1")] = distortion.tangentialP1;
        result[QStringLiteral("p2")] = distortion.tangentialP2;
        result[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
        result[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
        result[QStringLiteral("depth_axis_flipped")] = definition.depthAxisFlipped();
        result[QStringLiteral("world_frame")] = QString::fromStdString(camera.groundFrame().value());
        result[QStringLiteral("image_width")] = camera.imageSize().samples;
        result[QStringLiteral("image_height")] = camera.imageSize().lines;

        QJsonArray center_array;
        for (const double value : center)
        {
            center_array.append(value);
        }
        result[QStringLiteral("C")] = center_array;

        QJsonArray rotation_array;
        for (const double value : rotation)
        {
            rotation_array.append(value);
        }
        result[QStringLiteral("R")] = rotation_array;

        const QJsonObject ypr = rotationToYprDegrees(rotation);
        result[QStringLiteral("yaw_deg")] = ypr.value(QStringLiteral("yaw_deg"));
        result[QStringLiteral("pitch_deg")] = ypr.value(QStringLiteral("pitch_deg"));
        result[QStringLiteral("roll_deg")] = ypr.value(QStringLiteral("roll_deg"));
        return result;
    }


} // namespace xjw::common::project

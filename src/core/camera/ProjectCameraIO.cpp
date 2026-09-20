#include "ProjectCameraIO.h"

#include "FramePinholeTsaiIO.h"

#include "io/PathIO.h"

#include <QDateTime>
#include <QJsonArray>
#include <QtMath>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

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

        bool readFiniteDouble(const QJsonObject& object, const QString& key, double* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonValue value = object.value(key);
            if (!value.isDouble())
            {
                return false;
            }
            const double parsed = value.toDouble();
            if (!std::isfinite(parsed))
            {
                return false;
            }
            *output = parsed;
            return true;
        }

        bool readSignedAxis(const QJsonObject& object, const QString& key, int* output)
        {
            double value = 0.0;
            if (!readFiniteDouble(object, key, &value) || (value != -1.0 && value != 1.0))
            {
                return false;
            }
            *output = static_cast<int>(value);
            return true;
        }

    } // namespace

    QJsonObject
    serializeFramePinholeNumericState(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera)
    {
        const auto intrinsics = camera.intrinsics();
        const auto distortion = camera.distortion();
        const auto center = camera.cameraCenter();
        const auto rotation = camera.cameraToWorldRotation();

        QJsonObject result;
        result[QStringLiteral("model")] = QStringLiteral("frame_pinhole");
        result[QStringLiteral("imported_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        result[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
        result[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
        result[QStringLiteral("pixel_convention")] = QStringLiteral("center");
        result[QStringLiteral("pitch")] = camera.pixelPitch();
        result[QStringLiteral("fu")] = camera.focalXMillimeters();
        result[QStringLiteral("fv")] = camera.focalYMillimeters();
        result[QStringLiteral("cu")] = camera.principalXMillimeters();
        result[QStringLiteral("cv")] = camera.principalYMillimeters();
        result[QStringLiteral("k1")] = distortion.radialK1;
        result[QStringLiteral("k2")] = distortion.radialK2;
        result[QStringLiteral("k3")] = distortion.radialK3;
        result[QStringLiteral("p1")] = distortion.tangentialP1;
        result[QStringLiteral("p2")] = distortion.tangentialP2;
        result[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
        result[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
        result[QStringLiteral("depth_axis_flipped")] = camera.depthAxisFlipped();
        // A standalone Tsai file has no world-frame identifier.  Its import
        // boundary assigns the project's local frame explicitly so every
        // project-facing camera record has the same shape.
        result[QStringLiteral("world_frame")] = camera.hasBoundIdentity()
                                                    ? QString::fromStdString(camera.worldFrame().value())
                                                    : QStringLiteral("project-world");
        if (const std::optional<xjw::camera_core::ImageSize> imageSize = camera.imageSize(); imageSize)
        {
            result[QStringLiteral("image_width")] = imageSize->samples;
            result[QStringLiteral("image_height")] = imageSize->lines;
        }

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

    bool saveFramePinholeNumericState(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                      const std::string& path)
    {
        std::ofstream output = xjw::common::io::openOutputFile(path);
        if (!output)
        {
            return false;
        }

        const auto intrinsics = camera.intrinsics();
        const auto distortion = camera.distortion();
        const auto pose = camera.pose();
        output << std::setprecision(std::numeric_limits<double>::max_digits10);
        output << "fu = " << intrinsics.focalX * intrinsics.pixelPitch << "\n";
        output << "fv = " << intrinsics.focalY * intrinsics.pixelPitch << "\n";
        output << "cu = " << intrinsics.principalX * intrinsics.pixelPitch << "\n";
        output << "cv = " << intrinsics.principalY * intrinsics.pixelPitch << "\n";
        output << "c = " << pose.cameraCenter[0] << " " << pose.cameraCenter[1] << " " << pose.cameraCenter[2] << "\n";
        output << "r = ";
        for (std::size_t index = 0; index < pose.cameraToWorldRotation.size(); ++index)
        {
            if (index != 0)
            {
                output << ' ';
            }
            output << pose.cameraToWorldRotation[index];
        }
        output << "\n";
        output << "k1 = " << distortion.radialK1 << "\n";
        output << "k2 = " << distortion.radialK2 << "\n";
        output << "k3 = " << distortion.radialK3 << "\n";
        output << "p1 = " << distortion.tangentialP1 << "\n";
        output << "p2 = " << distortion.tangentialP2 << "\n";
        output << "pitch = " << intrinsics.pixelPitch << "\n";
        output << "u_direction = " << intrinsics.uAxisSign << "\n";
        output << "v_direction = " << intrinsics.vAxisSign << "\n";
        output << "w_direction = " << (pose.depthAxisFlipped ? -1 : 1) << "\n";
        return output.good();
    }

    bool parseTsaiCamera(const QString& tsai_path, QJsonObject* camera_metadata, QString* error_message)
    {
        if (!camera_metadata)
        {
            if (error_message)
            {
                *error_message = QStringLiteral("相机元数据输出参数为空");
            }
            return false;
        }

        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        std::string parse_error;
        if (!xjw::camera_io::loadFramePinholeNumericStateFromTsaiFile(
                xjw::common::io::toUtf8Path(tsai_path), &camera, &parse_error))
        {
            if (error_message)
            {
                *error_message =
                    QStringLiteral("无法解析相机文件: %1 (%2)").arg(tsai_path, QString::fromStdString(parse_error));
            }
            return false;
        }
        *camera_metadata = serializeFramePinholeNumericState(camera);
        return true;
    }

    bool loadFramePinholeNumericStateFromFile(const QString& path,
                                              xjw::camera_models::frame_pinhole::FramePinholeNumericState* camera,
                                              QString* error_message)
    {
        if (!camera)
        {
            if (error_message)
            {
                *error_message = QStringLiteral("数值相机输出参数为空");
            }
            return false;
        }
        std::string parse_error;
        if (!xjw::camera_io::loadFramePinholeNumericStateFromTsaiFile(
                xjw::common::io::toUtf8Path(path), camera, &parse_error))
        {
            if (error_message)
            {
                *error_message =
                    QStringLiteral("无法解析相机文件: %1 (%2)").arg(path, QString::fromStdString(parse_error));
            }
            return false;
        }
        return true;
    }

    bool decodeFramePinholeNumericState(const QJsonObject& camera_object,
                                        xjw::camera_models::frame_pinhole::FramePinholeNumericState* camera)
    {
        if (!camera || camera_object.isEmpty())
        {
            return false;
        }
        if (camera_object.value(QStringLiteral("model")).toString() != QStringLiteral("frame_pinhole") ||
            camera_object.value(QStringLiteral("intrinsics_unit")).toString() != QStringLiteral("mm") ||
            camera_object.value(QStringLiteral("camera_center_unit")).toString() != QStringLiteral("m") ||
            camera_object.value(QStringLiteral("pixel_convention")).toString() != QStringLiteral("center"))
        {
            return false;
        }

        xjw::camera_models::frame_pinhole::FramePinholeNumericState parsed;

        const QJsonArray center_array = camera_object.value(QStringLiteral("C")).toArray();
        const QJsonArray rotation_array = camera_object.value(QStringLiteral("R")).toArray();
        if (center_array.size() != 3 || rotation_array.size() != 9)
        {
            return false;
        }

        std::array<double, 3> center{};
        for (int index = 0; index < 3; ++index)
        {
            const QJsonValue value = center_array.at(index);
            if (!value.isDouble() || !std::isfinite(value.toDouble()))
            {
                return false;
            }
            center[static_cast<std::size_t>(index)] = value.toDouble();
        }
        std::array<double, 9> rotation{};
        for (int index = 0; index < 9; ++index)
        {
            const QJsonValue value = rotation_array.at(index);
            if (!value.isDouble() || !std::isfinite(value.toDouble()))
            {
                return false;
            }
            rotation[static_cast<std::size_t>(index)] = value.toDouble();
        }

        double pitch = 0.0;
        double fu = 0.0;
        double fv = 0.0;
        double cu = 0.0;
        double cv = 0.0;
        if (!readFiniteDouble(camera_object, QStringLiteral("pitch"), &pitch) ||
            !readFiniteDouble(camera_object, QStringLiteral("fu"), &fu) ||
            !readFiniteDouble(camera_object, QStringLiteral("fv"), &fv) ||
            !readFiniteDouble(camera_object, QStringLiteral("cu"), &cu) ||
            !readFiniteDouble(camera_object, QStringLiteral("cv"), &cv))
        {
            return false;
        }
        if (pitch <= 0.0)
        {
            return false;
        }
        parsed.setIntrinsicsMillimeters(fu, fv, cu, cv, pitch);

        int u_direction = 1;
        int v_direction = 1;
        if (!readSignedAxis(camera_object, QStringLiteral("u_direction"), &u_direction) ||
            !readSignedAxis(camera_object, QStringLiteral("v_direction"), &v_direction))
        {
            return false;
        }
        parsed.setAxisDirections(u_direction, v_direction);
        const QJsonValue depthAxisFlipped = camera_object.value(QStringLiteral("depth_axis_flipped"));
        if (!depthAxisFlipped.isBool())
        {
            return false;
        }
        parsed.setDepthAxisFlipped(depthAxisFlipped.toBool());
        double k1 = 0.0;
        double k2 = 0.0;
        double k3 = 0.0;
        double p1 = 0.0;
        double p2 = 0.0;
        if (!readFiniteDouble(camera_object, QStringLiteral("k1"), &k1) ||
            !readFiniteDouble(camera_object, QStringLiteral("k2"), &k2) ||
            !readFiniteDouble(camera_object, QStringLiteral("k3"), &k3) ||
            !readFiniteDouble(camera_object, QStringLiteral("p1"), &p1) ||
            !readFiniteDouble(camera_object, QStringLiteral("p2"), &p2))
        {
            return false;
        }
        parsed.setDistortion(k1, k2, k3, p1, p2);
        parsed.setPose(rotation, center);

        const bool hasImageWidth = camera_object.contains(QStringLiteral("image_width"));
        const bool hasImageHeight = camera_object.contains(QStringLiteral("image_height"));
        if (hasImageWidth != hasImageHeight)
        {
            return false;
        }
        if (hasImageWidth)
        {
            double imageWidth = 0.0;
            double imageHeight = 0.0;
            if (!readFiniteDouble(camera_object, QStringLiteral("image_width"), &imageWidth) ||
                !readFiniteDouble(camera_object, QStringLiteral("image_height"), &imageHeight) || imageWidth <= 0.0 ||
                imageHeight <= 0.0 || std::floor(imageWidth) != imageWidth || std::floor(imageHeight) != imageHeight ||
                imageWidth > static_cast<double>(std::numeric_limits<int>::max()) ||
                imageHeight > static_cast<double>(std::numeric_limits<int>::max()))
            {
                return false;
            }
            parsed.setImageSize(
                xjw::camera_core::ImageSize{static_cast<int>(imageWidth), static_cast<int>(imageHeight)});
        }

        if (!parsed.validateNumericalState())
        {
            return false;
        }
        *camera = std::move(parsed);
        return true;
    }

} // namespace xjw::common::project

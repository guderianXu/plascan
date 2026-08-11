#include "ProjectCameraIO.h"

#include "io/PathIO.h"

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

QJsonObject rotationToYprDegrees(const std::array<double, 9> &rotation)
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

    return QJsonObject{
        {QStringLiteral("yaw_deg"), qRadiansToDegrees(yaw)},
        {QStringLiteral("pitch_deg"), qRadiansToDegrees(pitch)},
        {QStringLiteral("roll_deg"), qRadiansToDegrees(roll)}};
}

} // namespace

QJsonObject cameraToJson(const xjw::FramePinholeCamera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto distortion = camera.distortion();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonObject result;
    result[QStringLiteral("model")] = QStringLiteral("tsai");
    result[QStringLiteral("imported_at")] =
        QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    result[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    result[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
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

bool parseTsaiCamera(const QString &tsai_path,
                     QJsonObject *camera_metadata,
                     QString *error_message)
{
    if (!camera_metadata)
    {
        if (error_message)
        {
            *error_message = QStringLiteral("相机元数据输出参数为空");
        }
        return false;
    }

    xjw::FramePinholeCamera camera;
    if (!camera.loadFromFile(xjw::common::io::toUtf8Path(tsai_path)) || !camera.isValid())
    {
        if (error_message)
        {
            *error_message = QStringLiteral("无法解析相机文件: %1").arg(tsai_path);
        }
        return false;
    }
    *camera_metadata = cameraToJson(camera);
    return true;
}

bool cameraFromJson(const QJsonObject &camera_object, xjw::FramePinholeCamera *camera)
{
    if (!camera || camera_object.isEmpty())
    {
        return false;
    }

    const QJsonArray center_array = camera_object.value(QStringLiteral("C")).toArray();
    const QJsonArray rotation_array = camera_object.value(QStringLiteral("R")).toArray();
    if (center_array.size() < 3 || rotation_array.size() < 9)
    {
        return false;
    }

    std::array<double, 3> center{{center_array.at(0).toDouble(),
                                  center_array.at(1).toDouble(),
                                  center_array.at(2).toDouble()}};
    if (camera_object.value(QStringLiteral("camera_center_unit"))
            .toString()
            .compare(QStringLiteral("mm"), Qt::CaseInsensitive) == 0)
    {
        center[0] /= 1000.0;
        center[1] /= 1000.0;
        center[2] /= 1000.0;
    }

    std::array<double, 9> rotation{};
    for (int index = 0; index < 9; ++index)
    {
        rotation[index] = rotation_array.at(index).toDouble();
    }

    const double pitch = camera_object.value(QStringLiteral("pitch")).toDouble(1.0);
    const double fu = camera_object.value(QStringLiteral("fu")).toDouble();
    const double fv = camera_object.value(QStringLiteral("fv")).toDouble();
    const double cu = camera_object.value(QStringLiteral("cu")).toDouble();
    const double cv = camera_object.value(QStringLiteral("cv")).toDouble();
    const bool millimeters =
        camera_object.value(QStringLiteral("intrinsics_unit"))
            .toString()
            .compare(QStringLiteral("mm"), Qt::CaseInsensitive) == 0;
    if (millimeters)
    {
        camera->setIntrinsicsMillimeters(fu, fv, cu, cv, pitch);
    }
    else
    {
        camera->setPixelPitch(pitch);
        camera->setIntrinsics(fu, fv, cu, cv);
    }

    camera->setAxisDirections(camera_object.value(QStringLiteral("u_direction")).toInt(1),
                              camera_object.value(QStringLiteral("v_direction")).toInt(1));
    camera->setDepthAxisFlipped(
        camera_object.value(QStringLiteral("depth_axis_flipped")).toBool(false));
    camera->setDistortion(camera_object.value(QStringLiteral("k1")).toDouble(),
                          camera_object.value(QStringLiteral("k2")).toDouble(),
                          camera_object.value(QStringLiteral("k3")).toDouble(),
                          camera_object.value(QStringLiteral("p1")).toDouble(),
                          camera_object.value(QStringLiteral("p2")).toDouble());
    camera->setPose(rotation, center);
    return true;
}

bool imageCameraFromEntry(const QJsonObject &image_object, xjw::FramePinholeCamera *camera)
{
    return cameraFromJson(image_object.value(QStringLiteral("camera")).toObject(), camera);
}

} // namespace xjw::common::project

#include "ProjectCameraIO.h"

#include "RpcRasterIO.h"
#include "io/PathIO.h"

#include <QJsonArray>

namespace xjw::common::project
{
    namespace
    {

        QJsonArray coefficientsToJson(const xjw::camera_models::rpc::RpcDefinition::Coefficients& coefficients)
        {
            QJsonArray result;
            for (double coefficient : coefficients)
            {
                result.append(coefficient);
            }
            return result;
        }

        bool hasImageCorrection(const xjw::camera_models::rpc::ImageCorrection& correction)
        {
            return correction.sampleOffsetPixels != 0.0 || correction.sampleSamplePixels != 0.0 ||
                   correction.sampleLinePixels != 0.0 || correction.lineOffsetPixels != 0.0 ||
                   correction.lineSamplePixels != 0.0 || correction.lineLinePixels != 0.0;
        }

        QJsonObject imageCorrectionToJson(const xjw::camera_models::rpc::ImageCorrection& correction)
        {
            return {{QStringLiteral("model"), QStringLiteral("affine_normalized_v1")},
                    {QStringLiteral("sample_offset_px"), correction.sampleOffsetPixels},
                    {QStringLiteral("sample_sample_px"), correction.sampleSamplePixels},
                    {QStringLiteral("sample_line_px"), correction.sampleLinePixels},
                    {QStringLiteral("line_offset_px"), correction.lineOffsetPixels},
                    {QStringLiteral("line_sample_px"), correction.lineSamplePixels},
                    {QStringLiteral("line_line_px"), correction.lineLinePixels}};
        }

    } // namespace

    QJsonObject serializeRpcInstance(const xjw::camera_models::rpc::RpcInstance& camera)
    {
        const xjw::camera_models::rpc::RpcDefinition::Parameters& parameters =
            camera.rpcDefinition().parameters();
        QJsonObject result{{QStringLiteral("model"), QStringLiteral("rpc00b")},
                           {QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                           {QStringLiteral("ground_crs"), QStringLiteral("EPSG:4979")},
                           {QStringLiteral("world_frame"),
                            QString::fromStdString(camera.rpcDefinition().worldFrame().value())},
                           {QStringLiteral("height_datum"), QStringLiteral("WGS84_ellipsoidal")},
                           {QStringLiteral("pixel_convention"), QStringLiteral("opencv_zero_based_center")},
                           {QStringLiteral("line_off"), parameters.lineOffset},
                           {QStringLiteral("samp_off"), parameters.sampleOffset},
                           {QStringLiteral("lat_off"), parameters.latitudeOffset},
                           {QStringLiteral("long_off"), parameters.longitudeOffset},
                           {QStringLiteral("height_off"), parameters.heightOffset},
                           {QStringLiteral("line_scale"), parameters.lineScale},
                           {QStringLiteral("samp_scale"), parameters.sampleScale},
                           {QStringLiteral("lat_scale"), parameters.latitudeScale},
                           {QStringLiteral("long_scale"), parameters.longitudeScale},
                           {QStringLiteral("height_scale"), parameters.heightScale},
                           {QStringLiteral("line_num_coeff"), coefficientsToJson(parameters.lineNumerator)},
                           {QStringLiteral("line_den_coeff"), coefficientsToJson(parameters.lineDenominator)},
                           {QStringLiteral("samp_num_coeff"), coefficientsToJson(parameters.sampleNumerator)},
                           {QStringLiteral("samp_den_coeff"), coefficientsToJson(parameters.sampleDenominator)},
                           {QStringLiteral("image_samples"), camera.imageSize().samples},
                           {QStringLiteral("image_lines"), camera.imageSize().lines}};
        if (parameters.errorBiasMeters)
        {
            result[QStringLiteral("err_bias_m")] = *parameters.errorBiasMeters;
        }
        if (parameters.errorRandomMeters)
        {
            result[QStringLiteral("err_rand_m")] = *parameters.errorRandomMeters;
        }
        if (hasImageCorrection(camera.imageCorrection()))
        {
            result[QStringLiteral("image_correction")] = imageCorrectionToJson(camera.imageCorrection());
        }
        return result;
    }

    bool parseRpcCameraRaster(const QString& raster_path, QJsonObject* camera_metadata, QString* error_message)
    {
        if (!camera_metadata)
        {
            if (error_message)
            {
                *error_message = QStringLiteral("相机元数据输出参数为空");
            }
            return false;
        }
        *camera_metadata = {};
        std::string error;
        const auto instance = xjw::camera_models::rpc::importRpcRasterInstance(
            xjw::common::io::toUtf8Path(raster_path),
            xjw::camera_core::CameraDefinitionId("rpc-raster-definition"),
            xjw::camera_core::CameraInstanceId("rpc-raster-instance"),
            xjw::camera_core::ImageId("rpc-raster-image"),
            xjw::coordinate_system::CoordinateFrameId("EPSG:4978"),
            &error);
        if (!instance)
        {
            if (error_message)
            {
                *error_message = QString::fromStdString(error);
            }
            return false;
        }
        *camera_metadata = serializeRpcInstance(*instance);
        return true;
    }

} // namespace xjw::common::project

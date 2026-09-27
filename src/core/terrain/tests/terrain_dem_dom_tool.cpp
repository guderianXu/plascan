#include "TerrainPipeline.h"
#include <placamera/tsai.h>

#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QTextStream>

#include <iostream>

namespace
{

    void printUsage(const char* programName)
    {
        std::cerr << "Usage:\n"
                  << "  " << programName
                  << " --list <pointcloud.(obj|ply|xyz)> <image_camera.lis> <output_dir> [dem_resolution]\n"
                  << "  " << programName
                  << " <pointcloud.(obj|ply|xyz)> <output_dir> [dem_resolution] [image1 image2 ...]\n"
                  << "  " << programName << " --dir <obj_dir> <output_dir> [dem_resolution]\n"
                  << "  " << programName << " --obj <textured.obj> <output_dir> [dem_resolution]\n"
                  << "  " << programName << " --asteroid-obj <textured.obj> <output_dir> [dem_resolution]\n";
    }

    bool tryParseResolution(const char* value, double* resolution)
    {
        if (!resolution)
        {
            return false;
        }

        bool ok = false;
        const double parsed = QString::fromLocal8Bit(value).toDouble(&ok);
        if (!ok)
        {
            return false;
        }

        *resolution = parsed;
        return true;
    }

    QJsonObject cameraDefinitionToJson(const placamera::FramePinholeDefinition& camera,
                                       const QString& definitionId,
                                       const QString& frame)
    {
        const auto& intrinsics = camera.intrinsics();
        const auto& distortion = camera.distortion();
        return QJsonObject{{QStringLiteral("id"), definitionId},
                           {QStringLiteral("model_type"), QStringLiteral("frame_pinhole")},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("frame"), frame},
                           {QStringLiteral("parameters"),
                            QJsonObject{{QStringLiteral("intrinsics"),
                                         QJsonObject{{QStringLiteral("fx_px"), intrinsics.focalX},
                                                     {QStringLiteral("fy_px"), intrinsics.focalY},
                                                     {QStringLiteral("cx_px"), intrinsics.principalX},
                                                     {QStringLiteral("cy_px"), intrinsics.principalY},
                                                     {QStringLiteral("pixel_pitch_mm"), intrinsics.pixelPitch},
                                                     {QStringLiteral("u_axis_sign"), intrinsics.uAxisSign},
                                                     {QStringLiteral("v_axis_sign"), intrinsics.vAxisSign}}},
                                        {QStringLiteral("distortion"),
                                         QJsonObject{{QStringLiteral("k1"), distortion.radialK1},
                                                     {QStringLiteral("k2"), distortion.radialK2},
                                                     {QStringLiteral("k3"), distortion.radialK3},
                                                     {QStringLiteral("p1"), distortion.tangentialP1},
                                                     {QStringLiteral("p2"), distortion.tangentialP2}}},
                                        {QStringLiteral("pixel_convention"), QStringLiteral("center")},
                                        {QStringLiteral("depth_axis_flipped"), camera.depthAxisFlipped()}}}};
    }

    QJsonObject cameraInstanceToJson(const placamera::Pose& camera,
                                     const QString& instanceId,
                                     const QString& imageId,
                                     const QString& definitionId,
                                     const QString& frame,
                                     const QSize& imageSize)
    {
        QJsonArray center;
        for (double value : camera.center)
        {
            center.append(value);
        }
        QJsonArray rotation;
        for (double value : camera.cameraToWorldRotation)
        {
            rotation.append(value);
        }
        return QJsonObject{{QStringLiteral("id"), instanceId},
                           {QStringLiteral("image_uuid"), imageId},
                           {QStringLiteral("definition_id"), definitionId},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("image_size"),
                            QJsonObject{{QStringLiteral("samples"), imageSize.width()},
                                        {QStringLiteral("lines"), imageSize.height()}}},
                           {QStringLiteral("pose"),
                            QJsonObject{{QStringLiteral("frame"), frame},
                                        {QStringLiteral("center_m"), center},
                                        {QStringLiteral("camera_to_world_rotation"), rotation}}}};
    }

    bool readImageCameraList(const QString& listPath, QStringList* images, QJsonObject* projectMeta, QString* error)
    {
        if (!images || !projectMeta)
        {
            if (error)
                *error = QStringLiteral("内部错误：列表输出对象为空");
            return false;
        }

        QFile file(listPath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            if (error)
                *error = QStringLiteral("无法打开 image/camera 列表: %1").arg(listPath);
            return false;
        }

        images->clear();
        QJsonArray imageArray;
        QJsonArray definitionArray;
        QJsonArray instanceArray;
        QTextStream stream(&file);
        int lineNumber = 0;
        while (!stream.atEnd())
        {
            ++lineNumber;
            const QString line = stream.readLine().trimmed();
            if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
            {
                continue;
            }

            QStringList parts;
            if (line.contains(QLatin1Char(',')))
            {
                const int comma = line.indexOf(QLatin1Char(','));
                parts << line.left(comma).trimmed() << line.mid(comma + 1).trimmed();
            }
            else
            {
                parts = line.split(QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts);
            }

            if (parts.size() != 2)
            {
                if (error)
                {
                    *error = QStringLiteral("%1:%2 需要 '<image> <camera.tsai>'").arg(listPath).arg(lineNumber);
                }
                return false;
            }

            const QString imagePath = QFileInfo(parts.at(0)).absoluteFilePath();
            const QString cameraPath = QFileInfo(parts.at(1)).absoluteFilePath();
            const QString imageId = QStringLiteral("terrain-tool-image-%1").arg(lineNumber);
            const QString definitionId = QStringLiteral("terrain-tool-definition-%1").arg(lineNumber);
            const QString instanceId = QStringLiteral("terrain-tool-instance-%1").arg(lineNumber);
            const QString frame = QStringLiteral("terrain-tool-frame");
            const auto camera =
                placamera::loadTsaiFramePinhole(cameraPath.toStdString(),
                                                placamera::CameraDefinitionId(definitionId.toStdString()),
                                                placamera::FrameId(frame.toStdString()));
            if (!camera)
            {
                if (error)
                {
                    *error = QStringLiteral("%1:%2 相机文件读取失败: %3；%4")
                                 .arg(listPath)
                                 .arg(lineNumber)
                                 .arg(cameraPath, QString::fromStdString(camera.message()));
                }
                return false;
            }

            QImageReader imageReader(imagePath);
            QSize imageSize = imageReader.size();
            if (!imageSize.isValid())
            {
                imageSize = imageReader.read().size();
            }
            if (!imageSize.isValid())
            {
                if (error)
                {
                    *error = QStringLiteral("%1:%2 无法读取影像尺寸: %3").arg(listPath).arg(lineNumber).arg(imagePath);
                }
                return false;
            }

            QJsonObject imageObject;
            imageObject[QStringLiteral("image_uuid")] = imageId;
            imageObject[QStringLiteral("path")] = imagePath;
            imageObject[QStringLiteral("name")] = QFileInfo(imagePath).fileName();
            imageArray.append(imageObject);
            definitionArray.append(cameraDefinitionToJson(*camera.value().definition, definitionId, frame));
            instanceArray.append(
                cameraInstanceToJson(camera.value().pose, instanceId, imageId, definitionId, frame, imageSize));
            images->push_back(imagePath);
        }

        if (images->isEmpty())
        {
            if (error)
                *error = QStringLiteral("image/camera 列表为空: %1").arg(listPath);
            return false;
        }

        (*projectMeta)[QStringLiteral("images")] = imageArray;
        (*projectMeta)[QStringLiteral("camera_definitions")] = definitionArray;
        (*projectMeta)[QStringLiteral("camera_instances")] = instanceArray;
        return true;
    }

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    const QString firstArg = QString::fromLocal8Bit(argv[1]);

    // --list <pointcloud> <image_camera.lis> <output_dir> [dem_resolution]
    if (firstArg == QStringLiteral("--list"))
    {
        if (argc < 5)
        {
            printUsage(argv[0]);
            return 1;
        }

        const QString pointCloudPath = QString::fromLocal8Bit(argv[2]);
        const QString listPath = QString::fromLocal8Bit(argv[3]);
        const QString outputDir = QString::fromLocal8Bit(argv[4]);
        double demResolution = 0.0;
        if (argc > 5)
        {
            tryParseResolution(argv[5], &demResolution);
        }

        QJsonObject demResult;
        QString error;
        if (!xjw::TerrainPipeline::generateDemProducts(
                pointCloudPath, outputDir, demResolution, QStringLiteral("float32"), true, &demResult, &error))
        {
            std::cerr << "DEM generation failed: " << error.toStdString() << '\n';
            return 2;
        }

        QStringList images;
        QJsonObject projectMeta;
        if (!readImageCameraList(listPath, &images, &projectMeta, &error))
        {
            std::cerr << "Image/camera list parsing failed: " << error.toStdString() << '\n';
            return 3;
        }

        const QString domPath = outputDir + QStringLiteral("/products/dom.png");
        QJsonObject domResult;
        if (!xjw::TerrainPipeline::generateOrthoProduct(images,
                                                        demResult.value(QStringLiteral("dem_path")).toString(),
                                                        domPath,
                                                        demResolution,
                                                        projectMeta,
                                                        &domResult,
                                                        &error))
        {
            std::cerr << "DOM generation failed: " << error.toStdString() << '\n';
            return 4;
        }

        QJsonObject result;
        result[QStringLiteral("dem")] = demResult;
        result[QStringLiteral("dom")] = domResult;
        result[QStringLiteral("image_camera_list")] = QFileInfo(listPath).absoluteFilePath();
        std::cout << QJsonDocument(result).toJson(QJsonDocument::Indented).toStdString() << std::endl;
        return 0;
    }

    // --dir <obj_dir> <output_dir> [dem_resolution]
    if (firstArg == QStringLiteral("--dir"))
    {
        if (argc < 4)
        {
            printUsage(argv[0]);
            return 1;
        }
        const QString dirPath = QString::fromLocal8Bit(argv[2]);
        const QString outputDir = QString::fromLocal8Bit(argv[3]);
        double demResolution = 0.0;
        if (argc > 4)
        {
            tryParseResolution(argv[4], &demResolution);
        }

        QJsonObject result;
        QString error;
        if (!xjw::TerrainPipeline::generateFromObjMtlDir(dirPath, outputDir, demResolution, &result, &error))
        {
            std::cerr << "OBJ directory DEM/DOM generation failed: " << error.toStdString() << '\n';
            return 2;
        }

        std::cout << QJsonDocument(result).toJson(QJsonDocument::Indented).toStdString() << std::endl;
        return 0;
    }

    // --obj <textured.obj> <output_dir> [dem_resolution]
    if (firstArg == QStringLiteral("--obj"))
    {
        if (argc < 4)
        {
            printUsage(argv[0]);
            return 1;
        }
        const QString objPath = QString::fromLocal8Bit(argv[2]);
        const QString outputDir = QString::fromLocal8Bit(argv[3]);
        double demResolution = 0.0;
        if (argc > 4)
        {
            tryParseResolution(argv[4], &demResolution);
        }

        QJsonObject result;
        QString error;
        if (!xjw::TerrainPipeline::generateFromObjMtl(objPath, outputDir, demResolution, &result, &error))
        {
            std::cerr << "OBJ+MTL DEM/DOM generation failed: " << error.toStdString() << '\n';
            return 2;
        }

        std::cout << QJsonDocument(result).toJson(QJsonDocument::Indented).toStdString() << std::endl;
        return 0;
    }

    // --asteroid-obj <textured.obj> <output_dir> [dem_resolution]
    // 生成极射赤平、方位等距、三轴椭球三种投影的 DEM+DOM GeoTIFF 产品
    if (firstArg == QStringLiteral("--asteroid-obj"))
    {
        if (argc < 4)
        {
            printUsage(argv[0]);
            return 1;
        }
        const QString objPath = QString::fromLocal8Bit(argv[2]);
        const QString outputDir = QString::fromLocal8Bit(argv[3]);
        double demResolution = 0.0;
        if (argc > 4)
        {
            tryParseResolution(argv[4], &demResolution);
        }

        QJsonObject result;
        QString error;
        if (!xjw::TerrainPipeline::generateFromObjMtlWithAsteroidProjections(
                objPath, outputDir, demResolution, &result, &error))
        {
            std::cerr << "Asteroid projection DEM/DOM generation failed: " << error.toStdString() << '\n';
            return 2;
        }

        std::cout << QJsonDocument(result).toJson(QJsonDocument::Indented).toStdString() << std::endl;
        return 0;
    }

    // Legacy mode: <pointcloud> <output_dir> [dem_resolution] [image1 image2 ...]
    const QString pointCloudPath = firstArg;
    const QString outputDir = QString::fromLocal8Bit(argv[2]);
    double demResolution = 0.0;
    int firstImageArg = 3;
    if (argc > 3 && tryParseResolution(argv[3], &demResolution))
    {
        firstImageArg = 4;
    }

    QJsonObject demResult;
    QString error;
    if (!xjw::TerrainPipeline::generateDemProducts(
            pointCloudPath, outputDir, demResolution, QStringLiteral("float32"), true, &demResult, &error))
    {
        std::cerr << "DEM generation failed: " << error.toStdString() << '\n';
        return 2;
    }

    std::cout << "DEM generation result:\n"
              << QJsonDocument(demResult).toJson(QJsonDocument::Indented).toStdString() << std::endl;

    if (argc > firstImageArg)
    {
        QStringList images;
        for (int index = firstImageArg; index < argc; ++index)
        {
            images.push_back(QString::fromLocal8Bit(argv[index]));
        }

        const QString domPath = outputDir + QStringLiteral("/products/dom.png");
        const double domResolution = demResolution > 0.0 ? demResolution : 0.0;
        QJsonObject domResult;
        if (!xjw::TerrainPipeline::generateOrthoProduct(images,
                                                        demResult.value(QStringLiteral("dem_path")).toString(),
                                                        domPath,
                                                        domResolution,
                                                        &domResult,
                                                        &error))
        {
            std::cerr << "DOM generation failed: " << error.toStdString() << '\n';
            return 3;
        }

        std::cout << "DOM generation result:\n"
                  << QJsonDocument(domResult).toJson(QJsonDocument::Indented).toStdString() << std::endl;
    }
    else
    {
        std::cout << "No input images provided. DEM products were generated, DOM generation skipped.\n";
        std::cout << "Tip: you can pass an explicit DEM resolution as the third argument.\n";
    }

    return 0;
}

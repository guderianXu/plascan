#include "TerrainPipeline.h"

#include <QFileInfo>
#include <QJsonDocument>

#include <iostream>

namespace
{

void printUsage(const char *programName)
{
    std::cerr << "Usage:\n"
              << "  " << programName << " <pointcloud.(obj|ply|xyz)> <output_dir> [dem_resolution] [image1 image2 ...]\n"
              << "  " << programName << " --dir <obj_dir> <output_dir> [dem_resolution]\n"
              << "  " << programName << " --obj <textured.obj> <output_dir> [dem_resolution]\n"
              << "  " << programName << " --asteroid-obj <textured.obj> <output_dir> [dem_resolution]\n";
}

bool tryParseResolution(const char *value, double *resolution)
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

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    const QString firstArg = QString::fromLocal8Bit(argv[1]);

    // --dir <obj_dir> <output_dir> [dem_resolution]
    if (firstArg == QStringLiteral("--dir"))
    {
        if (argc < 4)
        {
            printUsage(argv[0]);
            return 1;
        }
        const QString dirPath   = QString::fromLocal8Bit(argv[2]);
        const QString outputDir = QString::fromLocal8Bit(argv[3]);
        double demResolution    = 0.0;
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
        const QString objPath   = QString::fromLocal8Bit(argv[2]);
        const QString outputDir = QString::fromLocal8Bit(argv[3]);
        double demResolution    = 0.0;
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
        const QString objPath   = QString::fromLocal8Bit(argv[2]);
        const QString outputDir = QString::fromLocal8Bit(argv[3]);
        double demResolution    = 0.0;
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
    if (!xjw::TerrainPipeline::generateDemProducts(pointCloudPath,
                                                   outputDir,
                                                   demResolution,
                                                   QStringLiteral("float32"),
                                                   true,
                                                   &demResult,
                                                   &error))
    {
        std::cerr << "DEM generation failed: " << error.toStdString() << '\n';
        return 2;
    }

    std::cout << "DEM generation result:\n"
              << QJsonDocument(demResult).toJson(QJsonDocument::Indented).toStdString()
              << std::endl;

    if (argc > firstImageArg)
    {
        QStringList images;
        for (int index = firstImageArg; index < argc; ++index)
        {
            images.push_back(QString::fromLocal8Bit(argv[index]));
        }

        const QString domPath = outputDir + QStringLiteral("/products/dom.png");
        QJsonObject domResult;
        if (!xjw::TerrainPipeline::generateOrthoProduct(images,
                                                        demResult.value(QStringLiteral("dem_tif")).toString(),
                                                        domPath,
                                                        0.5,
                                                        &domResult,
                                                        &error))
        {
            std::cerr << "DOM generation failed: " << error.toStdString() << '\n';
            return 3;
        }

        std::cout << "DOM generation result:\n"
                  << QJsonDocument(domResult).toJson(QJsonDocument::Indented).toStdString()
                  << std::endl;
    }
    else
    {
        std::cout << "No input images provided. DEM products were generated, DOM generation skipped.\n";
        std::cout << "Tip: you can pass an explicit DEM resolution as the third argument.\n";
    }

    return 0;
}

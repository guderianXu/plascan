#include <gtest/gtest.h>

#include "DemGenerator.h"
#include "DomGenerator.h"
#include "io/ObjMtlLoader.h"
#include "TerrainPipeline.h"

#include "data/PointCloud.h"

#include <filesystem>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

using namespace xjw;

namespace
{

class TerrainDemDomTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _tempDir = fs::temp_directory_path() / "plascan_terrain_test";
        std::error_code errorCode;
        fs::remove_all(_tempDir, errorCode);
        fs::create_directories(_tempDir);
    }

    void TearDown() override
    {
        std::error_code errorCode;
        fs::remove_all(_tempDir, errorCode);
    }

    pointcloud::PointCloud makePlaneCloud() const
    {
        pointcloud::PointCloud cloud;
        cloud.addPoint(pointcloud::Point3f{0.0f, 0.0f, 10.0f});
        cloud.addPoint(pointcloud::Point3f{1.0f, 0.0f, 11.0f});
        cloud.addPoint(pointcloud::Point3f{0.0f, 1.0f, 12.0f});
        cloud.addPoint(pointcloud::Point3f{1.0f, 1.0f, 13.0f});
        return cloud;
    }

    fs::path writeObjPointCloud() const
    {
        const fs::path objPath = _tempDir / "input.obj";
        std::ofstream stream(objPath);
        stream << "v 0 0 10\n";
        stream << "v 1 0 11\n";
        stream << "v 0 1 12\n";
        stream << "v 1 1 13\n";
        return objPath;
    }

    fs::path writeImage(const std::string &name, const cv::Scalar &color) const
    {
        const fs::path imagePath = _tempDir / name;
        cv::Mat image(16, 16, CV_8UC3, color);
        cv::imwrite(imagePath.string(), image);
        return imagePath;
    }

    fs::path writeImage(const std::string &name, const cv::Mat &image) const
    {
        const fs::path imagePath = _tempDir / name;
        cv::imwrite(imagePath.string(), image);
        return imagePath;
    }

    static int countValidCells(const cv::Mat &mask)
    {
        return cv::countNonZero(mask);
    }

    // -------------------------------------------------------------------
    // 生成带纹理的最小 OBJ+MTL+PNG（2×2 四边形拆成 2 个三角面）
    // -------------------------------------------------------------------

    /** @brief 写出彩色纹理 PNG（4×4，四象限各一种纯色） */
    fs::path writeColorTexture(const std::string &name) const
    {
        // 左上:红  右上:绿  左下:蓝  右下:黄
        cv::Mat tex(4, 4, CV_8UC3);
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                if (r < 2 && c < 2)       tex.at<cv::Vec3b>(r, c) = {0,   0,   200}; // 红
                else if (r < 2 && c >= 2) tex.at<cv::Vec3b>(r, c) = {0,   200, 0};   // 绿
                else if (r >= 2 && c < 2) tex.at<cv::Vec3b>(r, c) = {200, 0,   0};   // 蓝
                else                      tex.at<cv::Vec3b>(r, c) = {0,   200, 200};  // 黄
            }
        }
        const fs::path texPath = _tempDir / name;
        cv::imwrite(texPath.string(), tex);
        return texPath;
    }

    /**
     * @brief 写出带 UV 和 face 的 OBJ（2×2 平面，高度各不同，UV 映射到全纹理）。
     *
     * 顶点布局（XY 平面，Z 为高度）：
     *   v0(0,0,1)  v1(2,0,2)
     *   v2(0,2,3)  v3(2,2,4)
     * 两个三角面：f v0/v1/v2  f v1/v3/v2
     * UV：v=1 在顶部（OBJ 标准），v0->(0,0),v1->(1,0),v2->(0,1),v3->(1,1)
     */
    fs::path writeTexturedMesh(const std::string &objName,
                               const std::string &mtlName,
                               const std::string &texName) const
    {
        const fs::path mtlPath = _tempDir / mtlName;
        {
            std::ofstream mtl(mtlPath);
            mtl << "newmtl mat0\n"
                << "Ka 1 1 1\n"
                << "Kd 1 1 1\n"
                << "map_Kd " << texName << "\n";
        }

        const fs::path objPath = _tempDir / objName;
        {
            std::ofstream obj(objPath);
            obj << "mtllib " << mtlName << "\n"
                << "v 0.0 0.0 1.0\n"
                << "v 2.0 0.0 2.0\n"
                << "v 0.0 2.0 3.0\n"
                << "v 2.0 2.0 4.0\n"
                << "vt 0.0 0.0\n"  // v0: bottom-left
                << "vt 1.0 0.0\n"  // v1: bottom-right
                << "vt 0.0 1.0\n"  // v2: top-left
                << "vt 1.0 1.0\n"  // v3: top-right
                << "usemtl mat0\n"
                << "f 1/1 2/2 3/3\n"
                << "f 2/2 4/4 3/3\n";
        }

        return objPath;
    }

    /** @brief 写出不带 UV/纹理的纯点云 OBJ（回归测试用） */
    fs::path writeTiledTexturedMesh(const std::string &tilePrefix, int tiles) const
    {
        // 在 _tempDir / tilePrefix_dir/ 下生成 N 个相邻的 OBJ 瓦片
        const fs::path tileDir = _tempDir / (tilePrefix + "_dir");
        fs::create_directories(tileDir);

        for (int t = 0; t < tiles; ++t)
        {
            const float ox = static_cast<float>(t) * 2.0f; // X 方向平铺
            const std::string objName = "tile" + std::to_string(t) + ".obj";
            const std::string mtlName = "tile" + std::to_string(t) + ".mtl";
            const std::string texName = "tile" + std::to_string(t) + ".png";

            // 写纹理（纯色，区分瓦片）
            cv::Mat tex(4, 4, CV_8UC3, cv::Scalar(0, 100 + t * 30, 200 - t * 20));
            cv::imwrite((tileDir / texName).string(), tex);

            // 写 MTL
            {
                std::ofstream mtl(tileDir / mtlName);
                mtl << "newmtl mat0\nmap_Kd " << texName << "\n";
            }

            // 写 OBJ
            {
                std::ofstream obj(tileDir / objName);
                obj << "mtllib " << mtlName << "\n"
                    << "v " << ox       << " 0.0 1.0\n"
                    << "v " << (ox+2.0) << " 0.0 2.0\n"
                    << "v " << ox       << " 2.0 3.0\n"
                    << "v " << (ox+2.0) << " 2.0 4.0\n"
                    << "vt 0.0 0.0\nvt 1.0 0.0\nvt 0.0 1.0\nvt 1.0 1.0\n"
                    << "f 1/1 2/2 3/3\nf 2/2 4/4 3/3\n";
            }
        }

        return tileDir;
    }

    static double laplacianVariance(const cv::Mat &image)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F, 3);

        cv::Scalar meanValue;
        cv::Scalar stdDev;
        cv::meanStdDev(laplacian, meanValue, stdDev);
        return stdDev[0] * stdDev[0];
    }

protected:
    fs::path _tempDir;
};

TEST_F(TerrainDemDomTest, DemGeneratorBuildsRasterAndDenseCloud)
{
    const pointcloud::PointCloud cloud = makePlaneCloud();

    DemGenerationOptions options;
    options.gridResolution = 0.5;
    options.minGridSize = 4;
    options.maxGridSize = 8;

    DemGridData demGrid;
    pointcloud::PointCloud denseCloud;
    QString error;
    ASSERT_TRUE(DemGenerator::generateFromPointCloud(cloud, options, &demGrid, &denseCloud, &error))
        << error.toStdString();

    EXPECT_TRUE(demGrid.isValid());
    EXPECT_EQ(demGrid.width, 3);
    EXPECT_EQ(demGrid.height, 3);
    EXPECT_FALSE(denseCloud.empty());
}

TEST_F(TerrainDemDomTest, TerrainPipelineWritesDemProductsFromObj)
{
    const fs::path objPath = writeObjPointCloud();
    const fs::path outputDir = _tempDir / "terrain_output";

    QJsonObject result;
    QString error;
    ASSERT_TRUE(TerrainPipeline::generateDemProducts(QString::fromStdString(objPath.string()),
                                                     QString::fromStdString(outputDir.string()),
                                                     0.5,
                                                     QStringLiteral("float32"),
                                                     true,
                                                     &result,
                                                     &error))
        << error.toStdString();

    EXPECT_TRUE(fs::exists(result.value(QStringLiteral("dem_tif")).toString().toStdString()));
    EXPECT_TRUE(fs::exists(result.value(QStringLiteral("depth_png")).toString().toStdString()));
    EXPECT_TRUE(fs::exists(result.value(QStringLiteral("dense_cloud_xyz")).toString().toStdString()));
    EXPECT_GE(result.value(QStringLiteral("face_count")).toInt(), 0);
}

TEST_F(TerrainDemDomTest, DemGeneratorSubPixelBilinearSplatIncreasesCoverage)
{
    pointcloud::PointCloud cloud;
    cloud.addPoint(pointcloud::Point3f{0.0f, 0.0f, 1.0f});
    cloud.addPoint(pointcloud::Point3f{2.0f, 2.0f, 1.0f});
    cloud.addPoint(pointcloud::Point3f{0.9f, 0.9f, 10.0f});

    DemGenerationOptions nearestOptions;
    nearestOptions.gridResolution = 1.0;
    nearestOptions.holeFillIterations = 0;
    nearestOptions.useSubPixelBilinearSplat = false;

    DemGridData nearestGrid;
    pointcloud::PointCloud nearestDense;
    QString error;
    ASSERT_TRUE(DemGenerator::generateFromPointCloud(cloud,
                                                     nearestOptions,
                                                     &nearestGrid,
                                                     &nearestDense,
                                                     &error))
        << error.toStdString();

    DemGenerationOptions bilinearOptions = nearestOptions;
    bilinearOptions.useSubPixelBilinearSplat = true;

    DemGridData bilinearGrid;
    pointcloud::PointCloud bilinearDense;
    ASSERT_TRUE(DemGenerator::generateFromPointCloud(cloud,
                                                     bilinearOptions,
                                                     &bilinearGrid,
                                                     &bilinearDense,
                                                     &error))
        << error.toStdString();

    EXPECT_GT(countValidCells(bilinearGrid.validMask), countValidCells(nearestGrid.validMask));
}

TEST_F(TerrainDemDomTest, DomGeneratorSharpnessWeightingImprovesDetailRetention)
{
    DemGridData demGrid;
    demGrid.width = 64;
    demGrid.height = 64;
    demGrid.stepX = 0.5;
    demGrid.stepY = 0.5;
    demGrid.elevation = cv::Mat(demGrid.height, demGrid.width, CV_32F, cv::Scalar(10.0f));
    demGrid.validMask = cv::Mat(demGrid.height, demGrid.width, CV_8U, cv::Scalar(255));

    cv::Mat sharpImage(demGrid.height, demGrid.width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int row = 0; row < sharpImage.rows; ++row)
    {
        for (int col = 0; col < sharpImage.cols; ++col)
        {
            const bool high = ((row / 4) + (col / 4)) % 2 == 0;
            sharpImage.at<cv::Vec3b>(row, col) = high ? cv::Vec3b(250, 250, 250) : cv::Vec3b(5, 5, 5);
        }
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(sharpImage, blurredImage, cv::Size(0, 0), 3.0);

    const fs::path sharpPath = writeImage("sharp.png", sharpImage);
    const fs::path blurPath = writeImage("blur.png", blurredImage);

    const QStringList images{QString::fromStdString(sharpPath.string()),
                             QString::fromStdString(blurPath.string())};

    DomGenerationOptions weightedOptions;
    weightedOptions.enableSharpnessWeighting = true;
    weightedOptions.enableExposureCompensation = false;
    weightedOptions.minBlendWeight = 1e-6;

    cv::Mat weightedDom;
    QString error;
    ASSERT_TRUE(DomGenerator::generateFromImages(demGrid,
                                                 images,
                                                 weightedOptions,
                                                 &weightedDom,
                                                 &error))
        << error.toStdString();

    DomGenerationOptions averageOptions = weightedOptions;
    averageOptions.enableSharpnessWeighting = false;

    cv::Mat averageDom;
    ASSERT_TRUE(DomGenerator::generateFromImages(demGrid,
                                                 images,
                                                 averageOptions,
                                                 &averageDom,
                                                 &error))
        << error.toStdString();

    EXPECT_GT(laplacianVariance(weightedDom), laplacianVariance(averageDom));
}

TEST_F(TerrainDemDomTest, TerrainPipelineGeneratesDomFromDemAndImages)
{
    const fs::path objPath = writeObjPointCloud();
    const fs::path outputDir = _tempDir / "terrain_dom_output";

    QJsonObject demResult;
    QString error;
    ASSERT_TRUE(TerrainPipeline::generateDemProducts(QString::fromStdString(objPath.string()),
                                                     QString::fromStdString(outputDir.string()),
                                                     0.5,
                                                     QStringLiteral("float32"),
                                                     false,
                                                     &demResult,
                                                     &error))
        << error.toStdString();

    const fs::path imageA = writeImage("image_a.png", cv::Scalar(10, 20, 200));
    const fs::path imageB = writeImage("image_b.png", cv::Scalar(200, 80, 10));
    const fs::path domPath = outputDir / "products" / "dom.png";

    QJsonObject domResult;
    ASSERT_TRUE(TerrainPipeline::generateOrthoProduct(
        QStringList{QString::fromStdString(imageA.string()), QString::fromStdString(imageB.string())},
        demResult.value(QStringLiteral("dem_tif")).toString(),
        QString::fromStdString(domPath.string()),
        0.5,
        &domResult,
        &error))
        << error.toStdString();

    EXPECT_TRUE(fs::exists(domPath));
    EXPECT_GT(domResult.value(QStringLiteral("width")).toInt(), 0);
    EXPECT_GT(domResult.value(QStringLiteral("height")).toInt(), 0);
}

// ===========================================================================
// OBJ+MTL+PNG 加载与光栅化测试
// ===========================================================================

TEST_F(TerrainDemDomTest, ObjMtlLoaderLoadsMeshAndTexture)
{
    writeColorTexture("loader_test.png");
    const fs::path objPath = writeTexturedMesh("loader_test.obj", "loader_test.mtl", "loader_test.png");

    TerrainMeshInput meshInput;
    QString error;
    ASSERT_TRUE(pointcloud::ObjMtlLoader::load(QString::fromStdString(objPath.string()), &meshInput, &error))
        << error.toStdString();

    EXPECT_GT(meshInput.mesh.size(), 0u);
    EXPECT_FALSE(meshInput.mesh.faces().empty());
    EXPECT_FALSE(meshInput.mesh.textureCoordinates().empty());
    EXPECT_FALSE(meshInput.texture.empty());

    // 纹理应为 3 通道 BGR
    EXPECT_EQ(meshInput.texture.channels(), 3);
}

TEST_F(TerrainDemDomTest, DomGeneratorTexturedMeshProducesColorOutput)
{
    writeColorTexture("dom_gen_test.png");
    const fs::path objPath = writeTexturedMesh("dom_gen_test.obj", "dom_gen_test.mtl", "dom_gen_test.png");

    TerrainMeshInput meshInput;
    QString loadError;
    ASSERT_TRUE(pointcloud::ObjMtlLoader::load(QString::fromStdString(objPath.string()), &meshInput, &loadError))
        << loadError.toStdString();

    // 构建覆盖整个网格的 DEM 网格
    DemGridData demGrid;
    demGrid.minX    = 0.0;
    demGrid.minY    = 0.0;
    demGrid.stepX   = 0.5;
    demGrid.stepY   = 0.5;
    demGrid.width   = 5;
    demGrid.height  = 5;
    demGrid.elevation = cv::Mat(demGrid.height, demGrid.width, CV_32F, cv::Scalar(2.0f));
    demGrid.validMask = cv::Mat(demGrid.height, demGrid.width, CV_8U,  cv::Scalar(1));

    cv::Mat domImage;
    QString domError;
    ASSERT_TRUE(DomGenerator::generateFromTexturedMesh(meshInput, demGrid, &domImage, &domError))
        << domError.toStdString();

    EXPECT_EQ(domImage.channels(), 3);
    EXPECT_GT(domImage.rows, 0);
    EXPECT_GT(domImage.cols, 0);

    // 至少有部分像素被填充（非全黑）
    cv::Mat gray;
    cv::cvtColor(domImage, gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}

TEST_F(TerrainDemDomTest, TerrainPipelineGeneratesDemDomFromSingleObjMtl)
{
    writeColorTexture("pipeline_single.png");
    const fs::path objPath = writeTexturedMesh("pipeline_single.obj", "pipeline_single.mtl", "pipeline_single.png");
    const fs::path outputDir = _tempDir / "pipeline_single_out";

    QJsonObject result;
    QString error;
    ASSERT_TRUE(TerrainPipeline::generateFromObjMtl(
        QString::fromStdString(objPath.string()),
        QString::fromStdString(outputDir.string()),
        0.5,
        &result,
        &error))
        << error.toStdString();

    // DEM 输出文件必须存在
    const std::string demTif  = result.value(QStringLiteral("dem_tif")).toString().toStdString();
    const std::string depthPng = result.value(QStringLiteral("depth_png")).toString().toStdString();
    EXPECT_TRUE(fs::exists(demTif))  << "dem_tif not found: " << demTif;
    EXPECT_TRUE(fs::exists(depthPng)) << "depth_png not found: " << depthPng;

    // 有纹理时 DOM 也应生成
    const bool hasTexture = result.value(QStringLiteral("has_texture")).toBool();
    if (hasTexture)
    {
        const std::string domPng = result.value(QStringLiteral("dom_png")).toString().toStdString();
        EXPECT_TRUE(fs::exists(domPng)) << "dom_png not found: " << domPng;
    }

    EXPECT_GT(result.value(QStringLiteral("grid_width")).toInt(), 0);
    EXPECT_GT(result.value(QStringLiteral("grid_height")).toInt(), 0);
}

TEST_F(TerrainDemDomTest, TerrainPipelineGeneratesDemDomFromDirectory)
{
    constexpr int kTiles = 2;
    const fs::path tileDir = writeTiledTexturedMesh("multi_tile", kTiles);
    const fs::path outputDir = _tempDir / "multi_tile_out";

    QJsonObject result;
    QString error;
    ASSERT_TRUE(TerrainPipeline::generateFromObjMtlDir(
        QString::fromStdString(tileDir.string()),
        QString::fromStdString(outputDir.string()),
        0.5,
        &result,
        &error))
        << error.toStdString();

    EXPECT_EQ(result.value(QStringLiteral("tile_count")).toInt(), kTiles);

    const std::string demTif  = result.value(QStringLiteral("dem_tif")).toString().toStdString();
    const std::string depthPng = result.value(QStringLiteral("depth_png")).toString().toStdString();
    EXPECT_TRUE(fs::exists(demTif))  << "dem_tif not found: " << demTif;
    EXPECT_TRUE(fs::exists(depthPng)) << "depth_png not found: " << depthPng;

    const std::string domPng = result.value(QStringLiteral("dom_png")).toString().toStdString();
    EXPECT_TRUE(fs::exists(domPng)) << "dom_png not found: " << domPng;
}

} // namespace
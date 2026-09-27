#include <gtest/gtest.h>
#include "CoreImplementationBundles.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QRegularExpression>
#include <QString>

#include <initializer_list>

namespace
{

    QString repoRoot()
    {
        return QStringLiteral(PLASCAN_SOURCE_DIR);
    }

    QString readSourceFile(const QString& relativePath)
    {
        const QString path = QDir(repoRoot()).filePath(relativePath);
        QFile file(path);
        EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(path);
        if (!file.isOpen())
        {
            return QString();
        }
        return QString::fromUtf8(file.readAll());
    }

    bool sourceFileExists(const QString& relativePath)
    {
        return QFileInfo::exists(QDir(repoRoot()).filePath(relativePath));
    }

    QString readMvsPipelineImplementation()
    {
        return xjw::tests::readMvsPipelineImplementation(readSourceFile);
    }

    QString readModelWorkflowImplementation()
    {
        return xjw::tests::readModelWorkflowImplementation(readSourceFile);
    }

    QString readIncrementalSfmImplementation()
    {
        return readSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfm.cpp")) +
               readSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfmDetail.cpp")) +
               readSourceFile(QStringLiteral("src/core/sfm/pipeline/InitialPairInitializer.cpp")) +
               readSourceFile(QStringLiteral("src/core/sfm/pipeline/ImageRegistrationEngine.cpp")) +
               readSourceFile(QStringLiteral("src/core/sfm/pipeline/KnownPoseReconstructor.cpp")) +
               readSourceFile(QStringLiteral("src/core/sfm/pipeline/SfmBundleAdjustCoordinator.cpp"));
    }

    QString utf8(const char* text)
    {
        return QString::fromUtf8(text);
    }

    void expectContainsAll(const QString& text, std::initializer_list<const char*> needles)
    {
        for (const char* needle : needles)
        {
            EXPECT_TRUE(text.contains(utf8(needle))) << needle;
        }
    }

    void expectNotContainsAll(const QString& text, std::initializer_list<const char*> needles)
    {
        for (const char* needle : needles)
        {
            EXPECT_FALSE(text.contains(utf8(needle))) << needle;
        }
    }

    int countOccurrences(const QString& text, const char* needle)
    {
        return text.count(utf8(needle));
    }

    int indexOfOrFail(const QString& text, const char* needle, int from = 0)
    {
        const int index = text.indexOf(utf8(needle), from);
        EXPECT_GE(index, 0) << needle;
        return index;
    }

    QString sectionBetween(const QString& text, const char* startNeedle, const char* endNeedle, int from);

    TEST(SfmModuleContractTest, ModuleOwnedTestsLiveBesideSfm)
    {
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/test/CMakeLists.txt")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("tests/test_sfm_pipeline.cpp")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("tests/test_sparse_point_cloud_processor.cpp")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/aerial_triangulation/tests/test_sfm_pair_planner.cpp")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("tests/test_sfm_pair_planner.cpp")));
    }

    TEST(ProjectCommonModuleContractTest, DeletedGuiProjectIoDirectoryIsNotAnIncludeRoot)
    {
        const QString buildDefinitions = readSourceFile(QStringLiteral("src/cli/CMakeLists.txt")) +
                                         readSourceFile(QStringLiteral("tests/CMakeLists.txt"));

        EXPECT_FALSE(buildDefinitions.contains(QStringLiteral("src/gui/project/io")));
    }

    TEST(CameraModelContractTest, LegacyCameraClassIsRemoved)
    {
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/Camera.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/Camera.cpp")));

        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/FramePinholeCamera.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/FramePinholeCamera.cpp")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/FramePinholeCameraModel.cpp")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/CameraModel.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/CameraModel.cpp")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/PlanetaryLineScanCamera.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/PlanetaryLineScanCamera.cpp")));
        EXPECT_FALSE(
            sourceFileExists(QStringLiteral("src/core/camera/models/frame_pinhole/FramePinholeNumericState.h")));
        const QString numericHeader =
            readSourceFile(QStringLiteral("3rdparty/placamera/include/placamera/frame_numeric_state.h"));
        expectContainsAll(numericHeader,
                          {
                              "class FramePinholeNumericState",
                              "groundToImage",
                              "imageToImagingLocus",
                              "triangulatePair",
                          });
        const QString cameraBuild = readSourceFile(QStringLiteral("src/core/camera/CMakeLists.txt"));
        EXPECT_FALSE(cameraBuild.contains(QStringLiteral("add_subdirectory(models/frame_pinhole)")));
        EXPECT_TRUE(cameraBuild.contains(QStringLiteral("add_library(camera INTERFACE)")));
        EXPECT_TRUE(cameraBuild.contains(QStringLiteral("placamera::gdal")));
        EXPECT_FALSE(cameraBuild.contains(QStringLiteral("add_library(camera STATIC")));
        for (const QString& removedAdapter : {QStringLiteral("TsaiFramePinholeIO.cpp"),
                                              QStringLiteral("TsaiFramePinholeIO.h"),
                                              QStringLiteral("PlanetaryLineScanIsdIO.cpp"),
                                              QStringLiteral("PlanetaryLineScanIsdIO.h"),
                                              QStringLiteral("RpcRasterIO.cpp"),
                                              QStringLiteral("RpcRasterIO.h")})
        {
            EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/") + removedAdapter));
        }
        EXPECT_TRUE(sourceFileExists(QStringLiteral("3rdparty/placamera/include/placamera/tsai.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("3rdparty/placamera/include/placamera/isd.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("3rdparty/placamera/include/placamera/rpc_raster.h")));
    }

    TEST(CameraModelContractTest, CameraReferenceAndProjectImportUsePlaCamera)
    {
        for (const QString& oldDirectory : {QStringLiteral("src/core/camera_core"),
                                            QStringLiteral("src/core/camera/core"),
                                            QStringLiteral("src/core/camera_models"),
                                            QStringLiteral("src/core/camera_project")})
        {
            EXPECT_FALSE(QFileInfo::exists(QDir(repoRoot()).filePath(oldDirectory)));
        }
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/camera/models/frame_pinhole/CMakeLists.txt")));
        EXPECT_FALSE(QFileInfo::exists(QDir(repoRoot()).filePath(QStringLiteral("src/core/camera_reference"))));
        EXPECT_TRUE(sourceFileExists(
            QStringLiteral("3rdparty/placamera/include/placamera/reference/CameraReferenceResolver.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("3rdparty/placamera/include/placamera/project_import.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/common/project/camera_reference/CameraReferenceSetStore.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/gui/project/services/ProjectCameraProjectImport.cpp")));
        const QString projectImport =
            readSourceFile(QStringLiteral("src/gui/project/services/ProjectCameraProjectImport.cpp"));
        EXPECT_TRUE(projectImport.contains(QStringLiteral("placamera::importCameraProject")));
        EXPECT_TRUE(projectImport.contains(QStringLiteral("placamera::makeCentralCameraGeometry")));
        EXPECT_FALSE(projectImport.contains(QStringLiteral("placamera::makeDatasetFramePinhole")));
        EXPECT_FALSE(projectImport.contains(QStringLiteral("placamera::FramePinholeDefinition::create")));
        EXPECT_TRUE(QFileInfo(QDir(repoRoot()).filePath(QStringLiteral("src/core/camera"))).isDir());
        EXPECT_FALSE(QFileInfo::exists(QDir(repoRoot()).filePath(QStringLiteral("src/core/camera_dataset_import"))));
        EXPECT_FALSE(QFileInfo::exists(QDir(repoRoot()).filePath(QStringLiteral("src/core/camera_io"))));
        EXPECT_FALSE(QFileInfo::exists(QDir(repoRoot()).filePath(QStringLiteral("src/cli/camera"))));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/dialogs/camera/CameraConvertDialog.cpp")));

        const QString coreBuild = readSourceFile(QStringLiteral("src/core/CMakeLists.txt"));
        const QString cameraBuild = readSourceFile(QStringLiteral("src/core/camera/CMakeLists.txt"));
        EXPECT_TRUE(
            coreBuild.contains(QStringLiteral("plascan_core_add_optional_module(camera \"Camera Integration\")")));
        EXPECT_FALSE(coreBuild.contains(QStringLiteral("camera_dataset_import")));
        EXPECT_FALSE(coreBuild.contains(
            QStringLiteral("plascan_core_add_optional_module(camera_reference \"Camera Reference\")")));
        EXPECT_FALSE(coreBuild.contains(QStringLiteral("plascan_core_add_optional_module(camera_core")));
        EXPECT_FALSE(cameraBuild.contains(QStringLiteral("add_subdirectory(core)")));
        EXPECT_FALSE(cameraBuild.contains(QStringLiteral("add_subdirectory(models/frame_pinhole)")));
        EXPECT_FALSE(cameraBuild.contains(QStringLiteral("add_subdirectory(project)")));
        EXPECT_FALSE(cameraBuild.contains(QStringLiteral("add_subdirectory(reference)")));
    }

    TEST(CameraModelContractTest, GuiReferenceCameraLookupUsesCanonicalPlaCamera)
    {
        const QString source = readSourceFile(QStringLiteral("src/gui/project/services/ProjectSession.cpp"));
        const int start = source.indexOf(QStringLiteral("ProjectSession::getReferenceCameraGeometriesForImages"));
        ASSERT_GE(start, 0);
        const int end = source.indexOf(QStringLiteral("ProjectSession::getRpcCameraImagePaths"), start);
        ASSERT_GT(end, start);
        const QString getter = source.mid(start, end - start);

        EXPECT_TRUE(getter.contains(QStringLiteral("loadProjectCameras")));
        EXPECT_TRUE(getter.contains(QStringLiteral("loaded.instances.forImage")));
        EXPECT_TRUE(getter.contains(QStringLiteral("placamera::FramePinholeModel")));
        EXPECT_FALSE(source.contains(QStringLiteral("makeSolverNumericState")));
    }

    TEST(CameraModelContractTest, OverlapGeometryUsesPlaCameraImagingLocus)
    {
        const QString header = readSourceFile(QStringLiteral("src/core/overlap/GroundBackProjector.h"));
        const QString source = readSourceFile(QStringLiteral("src/core/overlap/GroundBackProjector.cpp"));
        const QString analyzer = readSourceFile(QStringLiteral("src/core/overlap/OverlapAnalyzer.cpp"));

        EXPECT_TRUE(header.contains(QStringLiteral("placamera::FramePinholeModel")));
        EXPECT_FALSE(header.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_TRUE(source.contains(QStringLiteral("imageToImagingLocus")));
        EXPECT_TRUE(analyzer.contains(QStringLiteral("imageToImagingLocus")));
        EXPECT_FALSE(analyzer.contains(QStringLiteral("uAxisSign")));
        EXPECT_FALSE(analyzer.contains(QStringLiteral("rayCam")));
        EXPECT_FALSE(analyzer.contains(QStringLiteral("radius = 1.0")));
    }

    TEST(CameraModelContractTest, MeshGeometryUsesNumericStateAndQualityRendererUsesPlaCamera)
    {
        const QString sources = readSourceFile(QStringLiteral("src/core/mesh/DepthMapMeshBuilder.h")) +
                                readSourceFile(QStringLiteral("src/core/mesh/DepthRayMetric.h")) +
                                readSourceFile(QStringLiteral("src/core/mesh/DepthTsdfSurfaceBuilder.h")) +
                                readSourceFile(QStringLiteral("src/core/mesh/MeshColorizer.h")) +
                                readSourceFile(QStringLiteral("src/core/mesh/VisualHullReconstructor.h"));
        const QString quality_sources = readSourceFile(QStringLiteral("src/core/qc/ModelImageQualityTypes.h")) +
                                        readSourceFile(QStringLiteral("src/core/qc/ModelMeshRenderer.h"));
        const QString depthFrames = readSourceFile(QStringLiteral("src/core/mesh/DepthMapMeshBuilder.cpp"));

        EXPECT_TRUE(sources.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_TRUE(quality_sources.contains(QStringLiteral("placamera::FramePinholeModel")));
        EXPECT_FALSE(quality_sources.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_FALSE(sources.contains(QStringLiteral("FramePinholeCamera")));
        EXPECT_FALSE(depthFrames.contains(QStringLiteral("attachLegacyReportCameras")));
        EXPECT_FALSE(depthFrames.contains(QStringLiteral("loadFromFile")));
    }

    TEST(CameraModelContractTest, TerrainGeometryUsesPlaCamera)
    {
        const QString ortho_header = readSourceFile(QStringLiteral("src/core/terrain/OrthoProjector.h"));
        const QString sources = readSourceFile(QStringLiteral("src/core/terrain/DemGenerator.h")) +
                                readSourceFile(QStringLiteral("src/core/terrain/TerrainPipeline.h")) +
                                readSourceFile(QStringLiteral("src/core/terrain/DemGeneratorFromDepth.cpp"));
        const QString orthoInput = readSourceFile(QStringLiteral("src/core/terrain/OrthoProjectorGrid.cpp"));
        const QString orthoProjection = readSourceFile(QStringLiteral("src/core/terrain/OrthoProjectorSupport.cpp"));

        EXPECT_TRUE(ortho_header.contains(QStringLiteral("placamera::FramePinholeModel")));
        EXPECT_FALSE(ortho_header.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_TRUE(sources.contains(QStringLiteral("placamera::FramePinholeModel")));
        EXPECT_FALSE(sources.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_TRUE(orthoInput.contains(QStringLiteral("loadProjectCameras")));
        EXPECT_TRUE(orthoInput.contains(QStringLiteral("requireCommonGroundFrame")));
        EXPECT_FALSE(orthoInput.contains(QStringLiteral("makeSolverNumericState")));
        EXPECT_TRUE(orthoProjection.contains(QStringLiteral("groundToImage")));
        EXPECT_FALSE(orthoInput.contains(QStringLiteral("fileName()")));
    }

    TEST(CameraModelContractTest, FinalBaTsaiExportUsesPlaCameraGeometry)
    {
        const QString exporter = readSourceFile(QStringLiteral("src/cli/common/FinalBaCameraExporter.cpp"));
        const QString decoder = readSourceFile(QStringLiteral("src/common/project/ProjectFramePinholeMetadataIO.cpp"));
        const QString writer = readSourceFile(QStringLiteral("3rdparty/placamera/src/tsai.cpp"));
        const QString calibration = readSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraCalibrationData.cpp"));

        EXPECT_TRUE(exporter.contains(QStringLiteral("placamera::CameraInstanceSet")));
        EXPECT_TRUE(exporter.contains(QStringLiteral("finalCameras.forImage(imageId)")));
        EXPECT_FALSE(exporter.contains(QStringLiteral("decodeFramePinholeMetadata")));
        EXPECT_TRUE(decoder.contains(QStringLiteral("placamera::FramePinholeDefinition::create")));
        EXPECT_TRUE(exporter.contains(QStringLiteral("saveTsaiFramePinhole")));
        EXPECT_FALSE(exporter.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_FALSE(decoder.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_FALSE(writer.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_TRUE(calibration.contains(QStringLiteral("decodeFramePinholeMetadata")));
        EXPECT_FALSE(calibration.contains(QStringLiteral("FramePinholeNumericState")));
    }

    TEST(CameraModelContractTest, CliListReaderDoesNotCarryLegacyNumericCamera)
    {
        const QString header = readSourceFile(QStringLiteral("src/cli/common/cli_photogrammetry_common.h"));
        const QString source = readSourceFile(QStringLiteral("src/cli/common/cli_photogrammetry_common.cpp"));
        EXPECT_FALSE(header.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_FALSE(header.contains(QStringLiteral("loadCameras")));
        EXPECT_FALSE(source.contains(QStringLiteral("loadFramePinholeNumericStateFromTsaiFile")));
    }

    TEST(CameraModelContractTest, WorkspaceAndMarkerPredictionLoadCanonicalPlaCameraInstances)
    {
        const QString workspace = readSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
        const QString marker = readSourceFile(QStringLiteral("src/gui/markers/MarkerFocusMeasurementDialog.cpp"));

        for (const QString& source : {workspace, marker})
        {
            EXPECT_TRUE(source.contains(QStringLiteral("loadProjectCameras")));
            EXPECT_TRUE(source.contains(QStringLiteral("FramePinholeModel")));
            EXPECT_FALSE(source.contains(QStringLiteral("FramePinholeNumericState")));
            EXPECT_FALSE(source.contains(QStringLiteral("projectCameraModelParameters")));
        }
        const QString geometry = readSourceFile(QStringLiteral("src/core/control_points/geometry/MarkerGeometry.cpp"));
        const QString predictor =
            readSourceFile(QStringLiteral("src/core/control_points/geometry/MarkerProjectionPredictor.cpp"));
        EXPECT_TRUE(marker.contains(QStringLiteral("MarkerImageView")));
        EXPECT_FALSE(marker.contains(QStringLiteral("normalizedForPositiveDepth")));
        EXPECT_TRUE(geometry.contains(QStringLiteral("imageToImagingLocus")));
        EXPECT_TRUE(geometry.contains(QStringLiteral("groundToImage")));
        EXPECT_TRUE(predictor.contains(QStringLiteral("groundToImage")));
    }

    TEST(GuiStyleContractTest, WorkspaceTreeUsesApplicationColorsInsteadOfSystemPalette)
    {
        const QString treeSource = readSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
        const QString applicationStyle = readSourceFile(QStringLiteral("resources/styles/app.qss"));

        EXPECT_FALSE(treeSource.contains(QStringLiteral("palette(")));
        expectContainsAll(applicationStyle,
                          {
                              "QTreeView,",
                              "background: #ffffff;",
                              "QTreeView::item:selected,",
                              "background: #dbeafe;",
                              "color: #102a43;",
                          });
    }

    TEST(SfmModuleContractTest, ObsoleteFiltersAndCompatibilityAliasesAreRemoved)
    {
        const QString processorHeader =
            readSourceFile(QStringLiteral("src/core/sfm/filtering/SparsePointCloudProcessor.h"));

        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/common/SparsePointCloud.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/common/PhotogrammetryPointAttributes.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/filtering/SfmPointCloudFilter.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/filtering/SfmPointCloudFilter.cpp")));
        EXPECT_FALSE(
            sourceFileExists(QStringLiteral("src/core/sfm/triangulation/InitialSparsePointCloudTriangulator.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/triangulation/InitialSparsePointFilter.h")));
        expectNotContainsAll(processorHeader,
                             {
                                 "SparseCloudLocalOptimOptions",
                                 "SparseCloudLocalOptimResult",
                                 "localOptim(",
                             });
    }

    TEST(SfmModuleContractTest, GenericGraphInfrastructureIsSharedAndPlaPointBacked)
    {
        const QString observationHeader =
            readSourceFile(QStringLiteral("src/core/sfm/graph/ObservationNetworkBuilder.h"));
        const QString observationSource =
            readSourceFile(QStringLiteral("src/core/sfm/graph/ObservationNetworkBuilder.cpp"));
        const QString trackSource = readSourceFile(QStringLiteral("src/core/sfm/tracks/ReferenceTrackBuilder.cpp"));

        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/common/DisjointSet.h")));
        expectNotContainsAll(observationHeader, {"struct KDNode", "struct UnionFind"});
        expectNotContainsAll(observationSource, {"buildKD(", "queryKD(", "UnionFind::"});
        expectNotContainsAll(trackSource, {"class DisjointSet"});
        EXPECT_TRUE(observationSource.contains(QStringLiteral("plapoint::search::SpatialKdTree<2, double>")));
    }

    TEST(SfmModuleContractTest, ProjectionAndTriangulationGeometryHaveSingleOwners)
    {
        const QString initialFilter =
            readSourceFile(QStringLiteral("src/core/sfm/triangulation/InitialSparsePointFilter.cpp"));
        const QString baInputBuilder = readSourceFile(QStringLiteral("src/core/sfm/project/BaTrackBuilder.cpp"));
        const QString pnpSolver = readSourceFile(QStringLiteral("src/core/sfm/pose/PnpSolver.cpp"));
        const QString incrementalSfm = readIncrementalSfmImplementation();

        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/geometry/ProjectionGeometry.h")));
        const QString frameCameraHeader =
            readSourceFile(QStringLiteral("3rdparty/placamera/include/placamera/frame_camera.h"));
        EXPECT_TRUE(frameCameraHeader.contains(QStringLiteral("groundToImageSigned")));
        const QString qualityHeader = readSourceFile(QStringLiteral("src/core/sfm/geometry/TriangulationQuality.h"));
        const QString qualitySource = readSourceFile(QStringLiteral("src/core/sfm/geometry/TriangulationQuality.cpp"));
        EXPECT_TRUE(qualityHeader.contains(QStringLiteral("placamera::FramePinholeModel")));
        EXPECT_TRUE(qualitySource.contains(QStringLiteral("groundToImageSigned")));
        EXPECT_FALSE(qualityHeader.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_FALSE(qualitySource.contains(QStringLiteral("FramePinholeNumericState")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/geometry/OpenCvCameraAdapter.h")));
        expectNotContainsAll(initialFilter,
                             {
                                 "double reprojectionErrorPx(",
                                 "double minimumTriangulationAngleDeg(",
                             });
        expectNotContainsAll(baInputBuilder,
                             {
                                 "PairIntersectionCandidate triangulatePairWithDirectionFallback(",
                                 "double reprojectionErrorPx(",
                             });
        expectNotContainsAll(pnpSolver,
                             {
                                 "cameraToWorldRotationToOpenCvRvec",
                                 "cameraCenterToOpenCvTvec",
                             });
        EXPECT_FALSE(incrementalSfm.contains(QStringLiteral("const double depthSign =")));
    }

    TEST(SfmModuleContractTest, BaInputBuilderOnlyOrchestratesProjectAdapters)
    {
        const QString builder = readSourceFile(QStringLiteral("src/core/sfm/project/BaInputBuilder.cpp"));

        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/BaInputBuilder.cpp")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/BaInputBuilder.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/ProjectMatchInputReader.cpp")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/BaTrackBuilder.cpp")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/SurveyControlBaAdapter.cpp")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/MarkerBaAdapter.cpp")));
        expectNotContainsAll(builder,
                             {
                                 "QFile",
                                 "QJsonArray",
                                 "ReferenceTrackBuilder",
                                 "triangulatePairWithDirectionFallback",
                                 "solveControlNetwork",
                             });
    }

    TEST(SfmModuleContractTest, IncrementalSfmDelegatesMajorResponsibilities)
    {
        const QString incrementalSfm = readSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfm.cpp"));

        for (const QString& component : {
                 QStringLiteral("InitialPairInitializer"),
                 QStringLiteral("ImageRegistrationEngine"),
                 QStringLiteral("KnownPoseReconstructor"),
                 QStringLiteral("SfmBundleAdjustCoordinator"),
             })
        {
            EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/pipeline/%1.h").arg(component)));
            EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/pipeline/%1.cpp").arg(component)));
        }

        expectNotContainsAll(incrementalSfm,
                             {
                                 "IncrementalSfm::initializeFromPair",
                                 "IncrementalSfm::registerImage",
                                 "IncrementalSfm::runKnownCameraPoseReconstruction",
                                 "IncrementalSfm::runBundleAdjust",
                             });
        EXPECT_LT(incrementalSfm.count(QLatin1Char('\n')), 1200);
    }

    TEST(SfmModuleContractTest, QualityMetricsAreQtFreeAndSerializationIsProjectOwned)
    {
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/quality/SfmQualityReport.h")));
        EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/quality/SfmQualityReport.cpp")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/quality/SfmQualityMetrics.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/quality/SfmQualityMetrics.cpp")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/SfmQualityJsonSerializer.h")));
        EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/SfmQualityJsonSerializer.cpp")));

        const QString metricsHeader = readSourceFile(QStringLiteral("src/core/sfm/quality/SfmQualityMetrics.h"));
        const QString metricsSource = readSourceFile(QStringLiteral("src/core/sfm/quality/SfmQualityMetrics.cpp"));
        expectNotContainsAll(metricsHeader + metricsSource,
                             {
                                 "#include <Q",
                                 "QString",
                                 "QJsonObject",
                                 "QJsonArray",
                                 "QObject",
                             });
    }

    TEST(SfmModuleContractTest, CMakeTargetsEnforceAlgorithmDependencyDirection)
    {
        const QString cmake = readSourceFile(QStringLiteral("src/core/sfm/CMakeLists.txt"));
        expectContainsAll(cmake,
                          {
                              "add_library(sfm_core STATIC",
                              "add_library(sfm_postprocess STATIC",
                              "add_library(sfm_project STATIC",
                              "add_library(sfm INTERFACE)",
                              "target_link_libraries(sfm INTERFACE sfm_core sfm_postprocess sfm_project)",
                          });

        const QString coreLinks =
            sectionBetween(cmake, "target_link_libraries(sfm_core", "target_compile_features(sfm_core", 0);
        const QString postprocessLinks = sectionBetween(
            cmake, "target_link_libraries(sfm_postprocess", "target_compile_features(sfm_postprocess", 0);
        expectNotContainsAll(coreLinks + postprocessLinks,
                             {
                                 "Qt6::Core",
                                 "Qt6::Gui",
                             });

        const QString algorithmSources =
            readIncrementalSfmImplementation() +
            readSourceFile(QStringLiteral("src/core/control_points/registration/ControlNetworkSolver.h")) +
            readSourceFile(QStringLiteral("src/core/control_points/registration/ControlNetworkSolver.cpp"));
        expectNotContainsAll(algorithmSources,
                             {
                                 "#include <Q",
                                 "QString",
                                 "QVector",
                             });
    }

    QString sectionBetween(const QString& text, const char* startNeedle, const char* endNeedle, int from = 0)
    {
        const int start = indexOfOrFail(text, startNeedle, from);
        if (start < 0)
        {
            return QString();
        }
        const int end = indexOfOrFail(text, endNeedle, start);
        if (end < 0)
        {
            return QString();
        }
        EXPECT_GT(end, start) << endNeedle;
        return text.mid(start, end - start);
    }

    QString sectionFrom(const QString& text, const char* startNeedle)
    {
        const int start = indexOfOrFail(text, startNeedle);
        if (start < 0)
        {
            return QString();
        }
        return text.mid(start);
    }

    QString functionBody(const QString& source, const char* signature)
    {
        const int start = indexOfOrFail(source, signature);
        if (start < 0)
        {
            return QString();
        }

        const int brace = source.indexOf(QLatin1Char('{'), start);
        EXPECT_GE(brace, 0) << signature;
        if (brace < 0)
        {
            return QString();
        }

        int depth = 0;
        for (int index = brace; index < source.size(); ++index)
        {
            const QChar ch = source.at(index);
            if (ch == QLatin1Char('{'))
            {
                ++depth;
            }
            else if (ch == QLatin1Char('}'))
            {
                --depth;
                if (depth == 0)
                {
                    return source.mid(start, index - start + 1);
                }
            }
        }

        ADD_FAILURE() << "Function body not closed: " << signature;
        return source.mid(start);
    }

    void expectMatches(const QString& text, const char* pattern)
    {
        const QRegularExpression re(utf8(pattern), QRegularExpression::DotMatchesEverythingOption);
        EXPECT_TRUE(re.match(text).hasMatch()) << pattern;
    }

    void expectNotMatches(const QString& text, const char* pattern)
    {
        const QRegularExpression re(utf8(pattern), QRegularExpression::DotMatchesEverythingOption);
        EXPECT_FALSE(re.match(text).hasMatch()) << pattern;
    }

} // namespace

TEST(SfmSourceContractTest, SfmInitialPairSelectionPenalizesNearDuplicateCoverage)
{
    const QString source = readIncrementalSfmImplementation();
    const QString selectionBody = sectionBetween(source,
                                                 "std::vector<std::pair<ImageId, ImageId>> "
                                                 "IncrementalSfm::selectInitialPairCandidates",
                                                 "// COLMAP 式退化对过滤");

    expectContainsAll(selectionBody,
                      {
                          "initialPairScore",
                          "matchCoverage",
                          "std::min(img1.keypoints.size(), img2.keypoints.size())",
                          "coverage > 0.75",
                          "score *= 0.25",
                          "a.initialPairScore > b.initialPairScore",
                          "return a.numMatches > b.numMatches",
                      });
}

TEST(SfmSourceContractTest, SfmInitialPairSelectionUsesGraphConnectivity)
{
    const QString source = readIncrementalSfmImplementation();
    const QString selectionBody = sectionBetween(source,
                                                 "std::vector<std::pair<ImageId, ImageId>> "
                                                 "IncrementalSfm::selectInitialPairCandidates",
                                                 "// COLMAP 式退化对过滤");

    expectContainsAll(selectionBody,
                      {
                          "localGraphReach",
                          "endpointDegree",
                          "initGraph[pair.id1].push_back(pair.id2)",
                          "initGraph[pair.id2].push_back(pair.id1)",
                          "graphConnectivityFactor",
                          "pair.initialPairScore *= graphConnectivityFactor",
                          "容易初始化成功后困在小团里",
                      });
}

TEST(SfmSourceContractTest, SmallNoCameraSfmEvaluatesMultipleInitialPairModels)
{
    const QString header = readSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfm.h"));
    const QString source = readIncrementalSfmImplementation();

    expectContainsAll(header,
                      {
                          "evaluateMultipleInitialPairModels",
                          "multiInitialPairMaxImages",
                      });
    expectContainsAll(source,
                      {
                          "shouldEvaluateMultipleInitialPairModels",
                          "scoreInitialPairTrial",
                          "bestTrialResult",
                          "重置到同一份输入影像/匹配",
                      });
}

TEST(SfmSourceContractTest, SequenceInitialPoseGuessHandlesContiguousMissingRuns)
{
    const QString sfmHeader = readSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfm.h"));
    const QString sfmSource = readIncrementalSfmImplementation();

    expectContainsAll(sfmHeader,
                      {
                          "findRegisteredSequenceNeighbor",
                          "stepsOut",
                          "用最近的已注册序列相机",
                      });

    const QString neighborLookup = sectionBetween(sfmSource,
                                                  "bool IncrementalSfm::findRegisteredSequenceNeighbor",
                                                  "bool IncrementalSfm::hasRegisteredSequenceNeighbor");
    expectContainsAll(neighborLookup,
                      {
                          "direction",
                          "step < imageCount",
                          "hasRegisteredCamera(candidate)",
                          "stepsOut",
                      });

    const QString poseGuess = sectionBetween(
        sfmSource, "bool IncrementalSfm::makeSequenceInitialPoseGuess", "void IncrementalSfm::rebuildVisibilityCache");
    expectContainsAll(poseGuess,
                      {
                          "findRegisteredSequenceNeighbor(imageId, -1",
                          "findRegisteredSequenceNeighbor(imageId, 1",
                          "prevSteps",
                          "nextSteps",
                          "const double t",
                          "连续缺口",
                      });
}

TEST(SfmSourceContractTest, SequenceInitialPoseGuessDoesNotBypassPnpRegistration)
{
    const QString sfmSource = readIncrementalSfmImplementation();
    const QString registration =
        sectionBetween(sfmSource,
                       "IncrementalSfmResult IncrementalSfm::runRegistrationFromCurrentInitialization",
                       "void IncrementalSfm::resetForInitialPairTrial");

    expectContainsAll(registration,
                      {
                          "registerImage(nextId)",
                          "SfmBundleAdjustCoordinator(*this).iterative(true)",
                      });
    EXPECT_FALSE(sfmSource.contains(QStringLiteral("tryRegisterInterpolatedSequenceImages")));
    EXPECT_FALSE(sfmSource.contains(QStringLiteral("Sequence interpolation registered")));
}

TEST(SfmSourceContractTest, SequencePnpRecoveryRunsOnlyAfterStandardPnpFails)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration = sectionBetween(
        source, "bool IncrementalSfm::registerImage", "bool IncrementalSfm::findRegisteredSequenceNeighbor");

    expectContainsAll(registration,
                      {
                          "allowBracketedSequencePnpRelaxation",
                          "findRegisteredSequenceNeighbor(imageId,",
                          "&previousImageId",
                          "&nextImageId",
                          "auto solveSequenceRecovery = [&]",
                          "hasDirectPrevious && hasDirectNext",
                          "hasDirectPrevious != hasDirectNext",
                          "recoveryOptions.allowRelaxedInlierRatio = true",
                          "oneSidedSequencePnpMinInlierRatio",
                          "oneSidedSequencePnpMinInliers",
                          "bracketedSequencePnpMinInlierRatio",
                          "bracketedSequencePnpMinInliers",
                      });

    const int regular_branch = indexOfOrFail(registration, "// 常规增量阶段");
    const int standard_pnp = indexOfOrFail(registration, "pnpResult = PnpSolver::solveCalibrated(", regular_branch);
    const int failure_gate = indexOfOrFail(registration, "if (!pnpResult.success)", standard_pnp);
    const int sequence_recovery = indexOfOrFail(registration, "pnpResult = solveSequenceRecovery();", failure_gate);
    EXPECT_LT(standard_pnp, failure_gate);
    EXPECT_LT(failure_gate, sequence_recovery);
}

TEST(SfmSourceContractTest, ReferenceBatchResectionEvaluatesCandidatesInParallelBeforeCommit)
{
    const QString source = readIncrementalSfmImplementation();
    const QString batch = sectionBetween(
        source, "std::vector<ImageId> IncrementalSfm::registerImageBatch", "bool IncrementalSfm::registerImage");

    expectContainsAll(batch,
                      {
                          "common::concurrency::parallelForIndices",
                          "evaluateImageRegistration(",
                          "candidates[candidateIndex].first, false, false",
                          "std::vector<ImageRegistrationEvaluation> evaluations",
                          "_reconstruction->registerImage(candidate.imageId, candidate.camera)",
                      });
    EXPECT_FALSE(batch.contains(QStringLiteral("registerImage(imageId, false, false")));
    const int parallelEvaluation = indexOfOrFail(batch, "common::concurrency::parallelForIndices");
    const int serialCommit =
        indexOfOrFail(batch, "_reconstruction->registerImage(candidate.imageId", parallelEvaluation);
    EXPECT_LT(parallelEvaluation, serialCommit);
}

TEST(SfmSourceContractTest, IndependentBlockMergeSkipsDuplicateIncrementalRefinement)
{
    const QString source = readSourceFile(QStringLiteral("src/core/sfm/pipeline/IndependentBlockReconstructor.cpp"));
    const QString merge =
        sectionBetween(source, "_owner._reconstruction = std::move(merged)", "result.independentCameraBlocks");

    expectContainsAll(merge,
                      {
                          "runRegistrationFromCurrentInitialization(",
                          "std::move(progressCb), 0, true, false, true, false",
                      });
}

TEST(SfmSourceContractTest, ReferenceResectionConsumesCompleteSelectedTracks)
{
    const QString source = readIncrementalSfmImplementation();
    const QString evaluation = sectionBetween(
        source, "IncrementalSfm::ImageRegistrationEvaluation", "bool IncrementalSfm::findRegisteredSequenceNeighbor");

    expectContainsAll(evaluation,
                      {
                          "_inputTrackIndicesByImage.find(imageId)",
                          "const Track& inputTrack = _inputMultiViewTracks[trackIndex]",
                          "registeredImage.point3DIds[element.featureIdx]",
                          "不要求候选影像与持点影像之间还保留一条 direct pairwise edge",
                      });
}

TEST(SfmSourceContractTest, ReferenceResectionRefreshesAllTracksBeforeBundleAdjustment)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration =
        sectionBetween(source,
                       "IncrementalSfmResult IncrementalSfm::runRegistrationFromCurrentInitialization",
                       "void IncrementalSfm::resetForInitialPairTrial");

    const int referenceBranch = indexOfOrFail(registration, "if (useReferenceBaSchedule)");
    const int fullTrackRefresh = indexOfOrFail(
        registration, "triangulator.triangulateTracks(_inputMultiViewTracks, refreshOptions)", referenceBranch);
    const int structureFilter = indexOfOrFail(registration, "filterReferenceStructurePoints(", fullTrackRefresh);
    const int bundleAdjustment =
        indexOfOrFail(registration, "SfmBundleAdjustCoordinator(*this).iterative(growthStage)", structureFilter);
    EXPECT_LT(fullTrackRefresh, structureFilter);
    EXPECT_LT(structureFilter, bundleAdjustment);
}

TEST(SfmSourceContractTest, MatchGeometryFilteringUsesSeededSerialUsac)
{
    const QString source = readSourceFile(QStringLiteral("src/core/image_matching/geometry/MatchGeometryVerifier.cpp"));

    expectContainsAll(source,
                      {
                          "params.randomGeneratorState = options.randomSeed",
                          "params.isParallel = false",
                      });
}

TEST(SfmSourceContractTest, OptionalFinalGlobalBaRetryRunsBeforePublishingResult)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration =
        sectionBetween(source,
                       "IncrementalSfmResult IncrementalSfm::runRegistrationFromCurrentInitialization",
                       "void IncrementalSfm::resetForInitialPairTrial");

    const int finalBa = indexOfOrFail(registration, "SfmBundleAdjustCoordinator(*this).iterative(true);");
    const int retry = indexOfOrFail(registration, "retryUnregisteredImagesAfterFinalBA", finalBa);
    const int publish = indexOfOrFail(registration, "result.numRegisteredImages", retry);
    EXPECT_LT(finalBa, retry);
    EXPECT_LT(retry, publish);
}

TEST(SfmSourceContractTest, AerialReferencePathDoesNotAutoRunAnotherFinalBaCycle)
{
    const QString source =
        readSourceFile(QStringLiteral("src/core/aerial_triangulation/reconstruction/SfmAttemptRunner.cpp"));

    expectContainsAll(source,
                      {
                          "options->repairParallelAerialPoseOutliers = false;",
                          "正式“对齐照片”路径保持一次最终周期",
                      });
}

TEST(SfmSourceContractTest, ReferenceFinalBaKeepsIncrementalPointNetwork)
{
    const QString source = readSourceFile(QStringLiteral("src/core/sfm/pipeline/SfmBundleAdjustCoordinator.cpp"));
    const QString refinement =
        sectionBetween(source, "void IncrementalSfm::iterativeGlobalBA", "const int registered_count");

    expectContainsAll(refinement,
                      {
                          "finalRefinement && _sfmOptions.executionProfile != SfmExecutionProfile::FullRefinement",
                          "consolidateInputTracksForFinalBa()",
                          "不会先清空所有点再做一次全量重建",
                      });
}

TEST(SfmSourceContractTest, FullReferenceRegistrationUsesReferenceResectionSchedule)
{
    const QString registration = readSourceFile(QStringLiteral("src/core/sfm/pipeline/ImageRegistrationEngine.cpp"));
    const QString solver = readSourceFile(QStringLiteral("src/core/sfm/pose/ReferenceResectionSolver.cpp"));

    expectContainsAll(registration,
                      {
                          "pnpOptions.useReferenceResection = true",
                          "static_cast<double>(0.002F)",
                          "image_size.samples + image_size.lines",
                      });
    expectContainsAll(solver,
                      {
                          "constexpr int ransac_iterations = 500",
                          "constexpr std::size_t level_count = 10",
                          "thresholds[level] = thresholds[level - 1] * 0.75",
                          "for (int outer = 0; outer < 5; ++outer)",
                          "refineReferencePose(*normalized_camera, worldPoints, imagePoints, inliers, &pose, 10)",
                      });
}

TEST(SfmSourceContractTest, SequenceInitialPoseInterpolatesCenterAndRotation)
{
    const QString source = readIncrementalSfmImplementation();
    const QString guess = sectionBetween(
        source, "bool IncrementalSfm::makeSequenceInitialPoseGuess", "void IncrementalSfm::rebuildVisibilityCache");

    expectContainsAll(guess,
                      {
                          "interpolateCameraRotation",
                          "prevSteps",
                          "nextSteps",
                          "rotation = interpolateCameraRotation",
                      });
}

TEST(SfmSourceContractTest, BracketedSequencePnpUsesInitialPoseCorrespondenceGate)
{
    const QString registration = sectionBetween(readIncrementalSfmImplementation(),
                                                "bool IncrementalSfm::registerImage",
                                                "bool IncrementalSfm::findRegisteredSequenceNeighbor");
    const QString pnp = readSourceFile(QStringLiteral("src/core/sfm/pose/PnpSolver.cpp"));

    expectContainsAll(registration,
                      {
                          "recoveryOptions.useInitialPosePrefilter = true",
                          "initialPosePrefilterMaxReprojError",
                      });
    expectContainsAll(pnp,
                      {
                          "cv::projectPoints",
                          "guidedOriginalIndices",
                          "initialPosePrefilterMinCandidates",
                      });
}

TEST(SfmSourceContractTest, AutoSiftExtractionIsPartOfUnifiedImageMatchingModule)
{
    const QString extractor = readSourceFile(QStringLiteral("src/core/image_matching/sift/SiftFeatureExtractor.cpp"));
    const QString backend = readSourceFile(QStringLiteral("src/core/image_matching/sift/SiftComputeBackend.cpp"));
    const QString cmake = readSourceFile(QStringLiteral("src/core/image_matching/CMakeLists.txt"));

    expectContainsAll(extractor, {"extractSiftOnGpu", "SiftComputeBackend", "validMask"});
    expectContainsAll(backend, {"extractMetalSift", "extractOpenClSift", "extractCudaSift"});
    expectContainsAll(cmake,
                      {"SiftFeatureExtractor.cpp",
                       "SiftCudaBackend.cpp",
                       "SiftMetalBackend.mm",
                       "SiftOpenClBackend.cpp",
                       "plascan_cudasift"});
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/feature_extractors")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/feature_match")));
}

TEST(ImageMatchingSourceContractTest, TensorRtCallersUseInferenceModuleDirectly)
{
    const QString runtime = readSourceFile(QStringLiteral("src/core/matchphototask/runtime/MatchPhotosRuntime.cpp"));
    const QString loma = readSourceFile(QStringLiteral("src/core/image_matching/loma_r/LoMaRTensorRtBackend.cpp"));
    const QString cmake = readSourceFile(QStringLiteral("src/core/image_matching/CMakeLists.txt"));

    expectContainsAll(runtime, {"inference/tensorrt/TensorRtEngineBuilder.h", "inference::ensureTensorRtEngine"});
    expectContainsAll(loma, {"inference/tensorrt/TensorRtSession.h", "inference::TensorRtSession"});
    EXPECT_FALSE(cmake.contains(QStringLiteral("tensorrt/TensorRtEngineBuilder.h")));
    EXPECT_FALSE(cmake.contains(QStringLiteral("tensorrt/TensorRtSession.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/image_matching/tensorrt/TensorRtEngineBuilder.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/image_matching/tensorrt/TensorRtSession.h")));
}

TEST(SfmSourceContractTest, LightGlueSiftCarriesScaleAndOrientation)
{
    const QString exportScript = readSourceFile(QStringLiteral("scripts/models/export_lightglue_tensorrt.py"));
    const QString matcher =
        readSourceFile(QStringLiteral("src/core/image_matching/lightglue/TensorRtLightGlueMatcher.cpp"));

    expectContainsAll(exportScript,
                      {
                          R"(LightGlue(features="sift", **common))",
                          "geometry0 = torch.cat([xy0, keypoints0[..., 2:4]], dim=-1)",
                          "geometry1 = torch.cat([xy1, keypoints1[..., 2:4]], dim=-1)",
                          R"("keypoints0")",
                          R"("keypoints1")",
                      });
    expectContainsAll(matcher,
                      {
                          R"(feature.sourceAlgorithm != "sift")",
                          "keypoint.size",
                          "keypoint.angle",
                          "CV_PI",
                      });
}

TEST(BuildDependencyContractTest, ProductionBuildDoesNotDependOnLibTorch)
{
    const QString packages = readSourceFile(QStringLiteral("cmake/PlascanPackages.cmake"));
    const QString dependencyPaths = readSourceFile(QStringLiteral("cmake/PlascanDependencyPaths.cmake"));
    const QString maskCmake = readSourceFile(QStringLiteral("src/core/mask/CMakeLists.txt"));
    const QString guiCmake = readSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));

    for (const QString& source : {packages, dependencyPaths, maskCmake, guiCmake})
    {
        expectNotContainsAll(source,
                             {
                                 "find_package(Torch",
                                 "TORCH_LIBRARIES",
                                 "TORCH_INCLUDE_DIRS",
                                 "Torch_DIR",
                             });
    }
    EXPECT_FALSE(QFileInfo::exists(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("src/core/mask/Sam21MaskGenerator.cpp"))));
}

TEST(GuiAlgorithmAlignmentContractTest, WorkflowAccuracyUsesReferenceFiveLevelDownscale)
{
    const QString guiSource = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString options = readSourceFile(QStringLiteral("src/core/matchphototask/task/MatchPhotosOptions.h"));
    const QString workflow =
        readSourceFile(QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));

    expectContainsAll(options,
                      {
                          "Highest = 0",
                          "High = 1",
                          "Medium = 2",
                          "Low = 4",
                          "Lowest = 8",
                      });
    EXPECT_TRUE(workflow.contains(QStringLiteral("pipeline.quality = 2;")));
    EXPECT_TRUE(workflow.contains(QStringLiteral("tieOptions.accuracy = accuracy;")));
    EXPECT_FALSE(workflow.contains(QStringLiteral("QualityPreset presetForQuality")));
    EXPECT_GE(countOccurrences(guiSource, "workflowOptions.quality ="), 1);
}

TEST(GuiAlgorithmAlignmentContractTest, SequenceReferencePreselectionUsesUnifiedPreselector)
{
    const QString guiSource = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString workflow =
        readSourceFile(QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));
    ASSERT_FALSE(guiSource.isEmpty());
    ASSERT_FALSE(workflow.isEmpty());

    expectContainsAll(guiSource,
                      {
                          "bool shouldUseStoredGeneratedPairConstraints",
                          "reference_preselection_source",
                          "sequence",
                      });

    const QString sequenceHelpers =
        sectionBetween(guiSource, "bool isSequenceReferencePreselection", "QStringList loadGeneratedPairConstraints");
    expectContainsAll(sequenceHelpers,
                      {
                          "reference_preselection_source",
                          "sequence",
                          "shouldUseStoredGeneratedPairConstraints",
                          "return false",
                      });

    const QString resolvedConfig = sectionBetween(workflow,
                                                  "AerialTriangulationWorkflow::resolveConfig",
                                                  "AerialTriangulationResult AerialTriangulationWorkflow::run");
    expectContainsAll(resolvedConfig,
                      {
                          "tieOptions.useReferencePreselection = options.referencePreselection",
                          "tieOptions.referencePreselectionMode = referencePreselectionMode",
                          "matchphotos::PairSelectionMode::ManualOnly",
                      });
    EXPECT_FALSE(resolvedConfig.contains(QStringLiteral("matchphotos::PairSelectionMode::Sequence")));

    const QString unifiedRun = sectionBetween(guiSource,
                                              "void MenuWorkflowController::runUnifiedAerialTriangulation",
                                              "void MenuWorkflowController::openOverlapAnalysisDialog");
    expectContainsAll(unifiedRun,
                      {
                          "shouldUseStoredGeneratedPairConstraints(settings)",
                          "loadGeneratedPairConstraints",
                      });
}

TEST(GuiAlgorithmAlignmentContractTest, AerialTriangulationResetClearsStaleMatchCache)
{
    const QString workflow =
        readSourceFile(QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));
    ASSERT_FALSE(workflow.isEmpty());

    expectContainsAll(workflow,
                      {
                          "clearTiePointCache",
                          "ImageMatchRepository",
                          "repository.clear",
                          "不存在独立特征文件或 JSON sidecar",
                          "QFile::remove",
                          "forceRebuildTiePoints",
                          "TiePointPreparation::run",
                      });

    const int rebuild = workflow.indexOf(QStringLiteral("if (result.config.forceRebuildTiePoints)"));
    const int clear = workflow.indexOf(QStringLiteral("clearTiePointCache(result.config"), rebuild);
    const int prepare = workflow.indexOf(QStringLiteral("TiePointPreparation::run"), rebuild);
    ASSERT_GE(rebuild, 0);
    ASSERT_GT(clear, rebuild);
    ASSERT_GT(prepare, clear);
    EXPECT_LT(clear, prepare) << "重置当前对齐必须先清旧匹配缓存，再运行创建连接点。";
}

TEST(GuiAlgorithmAlignmentContractTest, AerialTriangulationGuiUsesSingleUnifiedWorkflow)
{
    const QString source = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString workflow =
        readSourceFile(QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));

    EXPECT_FALSE(header.contains(QStringLiteral("prepareAerialTriangulationTiePoints")));
    EXPECT_FALSE(header.contains(QStringLiteral("launchAerialTriangulationSfm")));
    EXPECT_FALSE(source.contains(QStringLiteral("MatchPhotosTask task(options)")));

    const QString run = sectionBetween(source,
                                       "void MenuWorkflowController::startAerialTriangulationWorkflow",
                                       "void MenuWorkflowController::openOverlapAnalysisDialog");
    expectContainsAll(run,
                      {
                          "workflowOptions.assetsDir",
                          "workflowOptions.matchDir",
                          "workflowOptions.matchingAlgorithmId",
                          "workflowOptions.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages",
                          "AerialTriangulationWorkflow::run",
                      });
    EXPECT_EQ(countOccurrences(run, "AerialTriangulationWorkflow::run"), 1);
    expectContainsAll(workflow,
                      {
                          "TiePointPreparation::run",
                          "clearTiePointCache",
                      });
}

TEST(SfmPnpObservationContractTest, RegistrationUsesOneThreeDimensionalCandidatePerImageFeature)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration = sectionBetween(
        source, "bool IncrementalSfm::registerImage", "bool IncrementalSfm::findRegisteredSequenceNeighbor");

    expectContainsAll(registration,
                      {
                          "selectUniquePnpCorrespondences(proposals)",
                          "for (const PnpCorrespondenceProposal& proposal : selected_proposals)",
                      });
    expectContainsAll(source,
                      {
                          "std::unordered_set<FeatureIdx> used_features",
                          "used_features.count(proposal.featureIdx)",
                          "used_features.insert(proposal.featureIdx)",
                          "std::unordered_set<Point3DId> used_points",
                      });
}

TEST(AutoSiftContractTest, TiePointThresholdCanReachDenseLowTextureRange)
{
    const QString source = readSourceFile(QStringLiteral("src/core/image_matching/sift/SiftFeatureExtractor.cpp"));

    expectContainsAll(source,
                      {
                          "threshold *= 0.5f",
                          "std::clamp(contrastThreshold, 0.001, 0.20)",
                          "minimumSide < 800",
                          "constexpr double scale = 2.0",
                      });
}

TEST(GuiAlgorithmAlignmentContractTest, ModelGenerationSettingsMigrateToCanonicalV2)
{
    const QString workflow_settings =
        readSourceFile(QStringLiteral("src/gui/dialogs/application/WorkflowSettingsDialog.cpp"));
    const QString cli = readSourceFile(QStringLiteral("src/cli/workflows/cli_mesh_reconstruct.cpp"));
    const QString model_settings = sectionBetween(workflow_settings,
                                                  "QJsonObject WorkflowSettingsDialog::modelGenerationSettings",
                                                  "void WorkflowSettingsDialog::setupUi");

    expectContainsAll(
        model_settings,
        {
            R"(model_settings[QStringLiteral("modelGenerationContractRevision")] = 2)",
            R"(model_settings[QStringLiteral("depthQualityProfile")] =)",
            R"(model_settings[QStringLiteral("interpolation")] =)",
            R"(model_settings[QStringLiteral("surfaceQualityProfile")] = QStringLiteral("recovered_ooc"))",
            R"(model_settings[QStringLiteral("faceCountMode")] = QStringLiteral("high"))",
            R"(model_settings[QStringLiteral("faceCountCustom")] = 200000)",
            R"(source.value(QStringLiteral("targetFaces")))",
            R"(source.value(QStringLiteral("simplifyTargetFaces")))",
            "legacy_faces <= 200000",
            "qBound(1, legacy_faces, 2000000)",
        });
    expectNotContainsAll(model_settings,
                         {
                             "qualityProfile",
                             "modelQualityProfile",
                             "splitIntoBlocks",
                             "blockSizeMeters",
                             "skipBoundaryBlocks",
                             "saveAfterEachStep",
                             "strictVolumetricMasks",
                         });
    expectContainsAll(cli,
                      {
                          R"(settings.remove(QStringLiteral("quality")))",
                          R"(settings.remove(QStringLiteral("qualityProfile")))",
                          R"(settings.remove(QStringLiteral("modelQualityProfile")))",
                          R"(settings.remove(QStringLiteral("targetFaces")))",
                          R"(settings.remove(QStringLiteral("splitIntoBlocks")))",
                          R"(settings.remove(QStringLiteral("blockSizeMeters")))",
                          R"(settings.remove(QStringLiteral("skipBoundaryBlocks")))",
                          R"(settings.remove(QStringLiteral("saveAfterEachStep")))",
                          R"(settings.remove(QStringLiteral("strictVolumetricMasks")))",
                          R"(settings[QStringLiteral("simplifyTargetFaces")] = target_faces)",
                      });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelAcceptsDepthMapsAsMetashapeStyleSource)
{
    const QString controller =
        readSourceFile(QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));

    const QString depthBlock =
        sectionBetween(controller, "const QJsonArray depthResults", "const QJsonArray modelResults");
    expectContainsAll(depthBlock,
                      {
                          R"(QStringLiteral("depth_maps"))",
                          R"(QStringLiteral("深度图"))",
                          "true,",
                          "深度图将作为生成模型入口",
                      });
    expectNotContainsAll(depthBlock,
                         {
                             "当前版本还不能直接从深度图生成模型",
                         });

    const QString meshBlock = sectionBetween(manager,
                                             "ProjectModelManager::startMeshReconstructionAsync",
                                             "bool ProjectModelManager::startTextureMappingAsync");
    expectContainsAll(meshBlock,
                      {
                          "resolveModelSourceForMeshing",
                          "resolvedSource.sourcePointCloudPath",
                          "resolvedSource.outputRoot",
                          "xjw::mesh::workflow::ModelBuildRequest",
                          "xjw::mesh::workflow::buildModel",
                      });
    expectNotContainsAll(meshBlock,
                         {
                             "当前版本还不能直接从深度图生成模型",
                         });

    expectContainsAll(
        dialog,
        {
            R"(_reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested))",
            R"(settings[QStringLiteral("depthMapSourcePath")] = sourcePath)",
            R"(settings[QStringLiteral("modelGenerationContractRevision")] = 2)",
            R"(settings[QStringLiteral("depthQualityProfile")] =)",
            R"(settings[QStringLiteral("interpolation")] = _interpolationCombo->currentData().toString())",
            R"(settings[QStringLiteral("surfaceQualityProfile")] =)",
        });
    expectNotContainsAll(dialog,
                         {
                             "splitIntoBlocks",
                             "blockSizeMeters",
                             R"(settings[QStringLiteral("strictVolumetricMasks")])",
                         });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelDepthMapsUseDirectMeshWorkflow)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString workflow = readModelWorkflowImplementation();

    expectContainsAll(dialog,
                      {
                          R"(settings[QStringLiteral("depthMapSourcePath")] = sourcePath)",
                          R"(settings[QStringLiteral("reuseDepthMaps")] =)",
                          R"(_hasReusableDepthMaps && selectedDepthBatchCompatible)",
                          R"(settings[QStringLiteral("force_depth_recompute")] =)",
                      });

    expectContainsAll(manager,
                      {
                          "xjw::mesh::workflow::ModelBuildRequest request",
                          "request.depthMapSourcePath",
                          "xjw::mesh::workflow::buildModel(request)",
                          "effectiveSettings.value(QStringLiteral(\"source_data\"))",
                      });

    expectContainsAll(workflow,
                      {
                          "WorkflowResult buildModel",
                          "WorkflowResult buildMeshFromDepthMaps",
                          "DepthMapMeshBuilder",
                          "request.depthMapSourcePath",
                      });

    expectNotContainsAll(manager,
                         {
                             "深度图源需要先融合为密集点云，但未找到可复用的 dense_cloud.ply",
                         });
}

TEST(GuiAlgorithmAlignmentContractTest, CanonicalFaceDiagnosticsKeepModeTargetsAndActualOutputDistinct)
{
    const QString workflow = readModelWorkflowImplementation();
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString contract = sectionBetween(workflow, "bool resolveModelGenerationContract", "QString sha256ForFile");
    const QString build_model =
        sectionBetween(workflow, "WorkflowResult buildModel", "WorkflowResult buildTextureOnly");

    expectContainsAll(contract,
                      {
                          R"(QStringLiteral("requestedTargetFaces"), target_faces)",
                          R"(QStringLiteral("effectiveTargetFaces"), target_faces)",
                      });
    expectContainsAll(build_model,
                      {
                          R"(QStringLiteral("face_count_mode"))",
                          R"(QStringLiteral("requested_target_faces")] = contract.targetFaces)",
                          R"(QStringLiteral("effective_target_faces")] = contract.targetFaces)",
                          R"(QStringLiteral("requested_face_count")] = contract.targetFaces)",
                          R"(QStringLiteral("effective_face_count")] = contract.targetFaces)",
                          R"(QStringLiteral("actual_output_face_count"))",
                          R"(QStringLiteral("face_count"))",
                      });
    expectContainsAll(manager,
                      {
                          R"(modelRecord[QStringLiteral("requested_target_faces")])",
                          R"(modelRecord[QStringLiteral("effective_target_faces")])",
                          R"(modelRecord[QStringLiteral("actual_output_face_count")])",
                          R"(reconstruction_parameters[QStringLiteral("requested_target_faces")])",
                          R"(reconstruction_parameters[QStringLiteral("effective_target_faces")])",
                          R"(reconstruction_parameters[QStringLiteral("actual_output_face_count")])",
                      });
    expectNotContainsAll(
        manager,
        {
            R"(taskResult.value(QStringLiteral("face_count")).toInt(settings.value(QStringLiteral("simplifyTargetFaces")).toInt()))",
        });
}

TEST(GuiAlgorithmAlignmentContractTest, DepthProductsRejectLegacyScaffoldFallback)
{
    const QString workflow = readSourceFile(QStringLiteral("src/core/mesh/workflow/DepthModelWorkflow.cpp"));
    expectContainsAll(workflow,
                      {
                          R"(mode != QStringLiteral("recovered_ooc"))",
                          "请迁移项目设置",
                          "return buildRecoveredDepthModel",
                      });
    expectNotContainsAll(workflow,
                         {
                             "buildLegacyDepthModelForValidation",
                             "sparse_scaffold_completion_requested",
                             "SparseScaffoldSurfaceBuilder",
                         });
    EXPECT_LT(indexOfOrFail(workflow, R"(mode != QStringLiteral("recovered_ooc"))"),
              indexOfOrFail(workflow, "return buildRecoveredDepthModel"));
}

TEST(GuiAlgorithmAlignmentContractTest, RecoveredModelIsBuiltBeforePublicationWithoutLegacyColorizer)
{
    const QString workflow = readSourceFile(QStringLiteral("src/core/mesh/workflow/RecoveredDepthModelStage.cpp"));
    expectContainsAll(workflow,
                      {
                          "buildRecoveredModel(recovered_input",
                          "saveMeshAndOptionalTexture(recovered.mesh",
                          "mergePayload(recovered.diagnostics",
                          "cancellationRequested(request.isCancelled)",
                      });
    expectNotContainsAll(workflow, {"MeshColorizer::colorize", "sparse_scaffold_completion"});
    EXPECT_LT(indexOfOrFail(workflow, "buildRecoveredModel(recovered_input"),
              indexOfOrFail(workflow, "saveMeshAndOptionalTexture(recovered.mesh"));
}

TEST(GuiAlgorithmAlignmentContractTest, TextureOnlyUsesTemporaryVertexColoredMeshWithoutRewritingOriginal)
{
    const QString workflow = readModelWorkflowImplementation();
    const QString texture_block =
        sectionBetween(workflow, "WorkflowResult buildTextureOnly", "} // namespace xjw::mesh::workflow");

    expectContainsAll(texture_block,
                      {
                          "TriMesh::loadPLY",
                          "!source_mesh.hasVertexColors",
                          "MeshColorizer::colorize",
                          R"(QStringLiteral(".texture_source_colored.ply"))",
                          "texture_mesh_path = temporary_colored_mesh_path",
                          "generateCameraTexturedModelFromMeshFile",
                          "QFile::remove(temporary_colored_mesh_path)",
                      });
    expectNotContainsAll(texture_block,
                         {
                             "source_mesh.savePLY(xjw::common::io::toUtf8Path(request.meshPath)",
                         });
}

TEST(GuiAlgorithmAlignmentContractTest, RecoveredProductionRequiresCompletedPrimaryFrames)
{
    const QString workflow = readSourceFile(QStringLiteral("src/core/mesh/workflow/RecoveredDepthModelStage.cpp"));
    expectContainsAll(workflow,
                      {
                          "artifacts.isEmpty()",
                          R"(artifact.status != QStringLiteral("completed"))",
                          "!xjw::mvs::isPrimaryFusionFrame(artifact.role)",
                          "return result",
                          "buildRecoveredModel(recovered_input",
                      });
    expectNotContainsAll(workflow,
                         {
                             "sparse_scaffold_low_primary_bypass_applied",
                             "allow_sparse_scaffold_fallback",
                         });
    EXPECT_LT(indexOfOrFail(workflow, "artifacts.isEmpty()"),
              indexOfOrFail(workflow, "buildRecoveredModel(recovered_input"));
}

TEST(GuiAlgorithmAlignmentContractTest, RecoveredProductionDoesNotApplyLegacyCarrierDenoising)
{
    const QString workflow = readSourceFile(QStringLiteral("src/core/mesh/workflow/RecoveredDepthModelStage.cpp"));
    expectContainsAll(workflow, {"buildRecoveredModel", "&recovered", "saveMeshAndOptionalTexture"});
    expectNotContainsAll(workflow,
                         {
                             "applyTopologyGuardedFinalSurfaceDenoising",
                             "allow_carrier_area_relaxation",
                             "direct_visibility_occupancy_output",
                         });
}

TEST(GuiAlgorithmAlignmentContractTest, ModelManagerUsesSharedModelWorkflowEntry)
{
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString mesh_block = sectionBetween(manager,
                                              "ProjectModelManager::startMeshReconstructionAsync",
                                              "bool ProjectModelManager::startTextureMappingAsync");

    expectContainsAll(mesh_block,
                      {
                          "xjw::mesh::workflow::ModelBuildRequest",
                          "xjw::mesh::workflow::buildModel",
                      });
    expectNotContainsAll(mesh_block,
                         {
                             "xjw::mesh::workflow::buildMeshFromDepthMaps",
                             "xjw::mesh::workflow::buildMeshAndOptionalTexture",
                             "reconstructionConfigFromModelSettings",
                         });
}

TEST(GuiAlgorithmAlignmentContractTest, ReconstructionStagesRouteToDedicatedManagers)
{
    const QString controller =
        readSourceFile(QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    const QString project_manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString task_orchestrator_header =
        readSourceFile(QStringLiteral("src/gui/project/tasks/ProjectTaskOrchestrator.h"));
    const QString task_orchestrator =
        readSourceFile(QStringLiteral("src/gui/project/tasks/ProjectTaskOrchestrator.cpp"));

    expectContainsAll(controller,
                      {
                          "_tasks->startCreatePointCloudAsync(settings)",
                          "_tasks->startGenerateModelAsync(settings)",
                      });
    expectContainsAll(task_orchestrator_header,
                      {
                          "void startCreatePointCloudAsync(const QJsonObject& settings)",
                          "void startGenerateModelAsync(const QJsonObject& settings)",
                      });

    const QString generate_block =
        sectionBetween(task_orchestrator,
                       "void ProjectTaskOrchestrator::startGenerateModelAsync(const QJsonObject& settings)",
                       "void ProjectTaskOrchestrator::startCreatePointCloudAsync");
    expectContainsAll(generate_block,
                      {
                          "startAutomaticModelDepth(settings)",
                          R"(settings.value(QStringLiteral("force_depth_recompute")))",
                          R"(settings.value(QStringLiteral("reuseDepthMaps")))",
                          "prepare_depth_maps",
                      });
    const QString automatic_depth_block = sectionBetween(task_orchestrator,
                                                         "bool ProjectTaskOrchestrator::startAutomaticModelDepth",
                                                         "void ProjectTaskOrchestrator::startAutomaticModelBuild");
    expectContainsAll(automatic_depth_block,
                      {
                          "_pointCloudWorkflow->startDepthMapsOnlyAsync(depth_settings, shared_context)",
                      });
    expectNotContainsAll(project_manager,
                         {
                             "ProjectModelGenerationWorkflow",
                             "ProjectReconstructionManager",
                             "ProjectTaskDispatcher",
                         });

    EXPECT_FALSE(project_manager.contains(QStringLiteral("startMeshReconstructionAsync")));
    const QString orchestrator_mesh_block = sectionBetween(task_orchestrator,
                                                           "void ProjectTaskOrchestrator::startMeshReconstructionAsync",
                                                           "void ProjectTaskOrchestrator::startTextureMappingAsync");
    expectContainsAll(orchestrator_mesh_block,
                      {
                          "_modelManager->startMeshReconstructionAsync(settings, start_context)",
                      });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelDialogOffersAutomaticDepthMaps)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));
    ASSERT_FALSE(dialog.isEmpty());

    expectContainsAll(dialog,
                      {
                          "重用深度图",
                          "自动生成深度图",
                          "缺少深度图时将自动估计深度图",
                          "automatic_depth_maps",
                          R"(settings[QStringLiteral("reuseDepthMaps")])",
                          R"(settings[QStringLiteral("depthMapSourcePath")] = sourcePath)",
                      });
}

TEST(GuiAlgorithmAlignmentContractTest, AutomaticModelDepthPreparationUsesSingleStatusTask)
{
    const QString source = readSourceFile(QStringLiteral("src/gui/project/tasks/ProjectTaskOrchestrator.cpp"));

    // Keep this contract at the workflow boundary.  Automatic depth generation
    // temporarily routes the point-cloud task into the model progress channel
    // and must restore the normal point-cloud status path when it finishes.
    expectContainsAll(source,
                      {
                          "void ProjectTaskOrchestrator::startGenerateModelAsync",
                          "startDepthMapsOnlyAsync",
                          "startMeshReconstructionAsync",
                          "emit meshProgressChanged(QStringLiteral(\"",
                          "std::clamp(percent * 3 / 5, 0, 59)",
                      });
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectPointCloudWorkflowController::pointCloudProgressChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectPointCloudWorkflowController::pointCloudProgressFinished")));
    EXPECT_TRUE(source.contains(QStringLiteral("context.cancelFlag")));
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelUsesCanonicalFaceCountControls)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));

    expectContainsAll(dialog,
                      {
                          "BuildModelDialog",
                          "workflowGeneralGroup",
                          "groupBlocks",
                          "workflowAdvancedGroup",
                          "checkSplitInBlocks",
                          "checkVertexColors",
                          "checkStrictVolumetricMasks",
                          "comboSourceData",
                          "comboFilterMode",
                          "comboFaceCount",
                          "editCustomFaceCount",
                          "QStringLiteral(\"low\")",
                          "QStringLiteral(\"medium\")",
                          "QStringLiteral(\"high\")",
                          "QStringLiteral(\"custom\")",
                          "_customFaceCountSpin->setRange(1, 2000000000)",
                          R"(faceCountMode == QStringLiteral("custom") ? customFaces : 0)",
                          R"(settings[QStringLiteral("simplifyTargetFaces")] = targetFaces)",
                          R"(settings[QStringLiteral("saveAfterEachStep")])",
                      });
    expectNotContainsAll(dialog,
                         {
                             "_splitRegionCheck",
                             "splitIntoBlocks",
                             "blockSizeMeters",
                             R"(settings[QStringLiteral("strictVolumetricMasks")])",
                             "? 20000",
                             "? 100000",
                             "? 200000",
                         });
}

TEST(MvsSchedulerContractTest, RecoveredProductionUsesTrackRankedSceneSelection)
{
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/MvsPipelineService.h"));
    const QString generator = readMvsPipelineImplementation();
    const QString adapter = readSourceFile(QStringLiteral("src/core/mvs/RecoveredDepthScene.cpp"));
    const QString selector = readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/neighbor_selection.cpp"));
    const QString orchestrator =
        readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/patchmatch_orchestrator.cpp"));
    const QString run =
        sectionBetween(generator, "void MvsPipelineService::runInBackgroundImpl()", "} // namespace xjw::mvs");

    expectNotContainsAll(header,
                         {
                             "FrameMvsCache",
                             "prepareFrameCaches",
                             "_visibilityBits",
                             "computeDepthForView",
                         });
    expectContainsAll(run,
                      {
                          "_effectiveDepthFilterMode = DepthFilterMode::Mild",
                          "_configuredSourceViewCount = 16",
                          "runRecoveredDepthScene(",
                          "recovered_workspace_root",
                          "frame.initialQualityAcceptanceAvailable = false",
                          "QStringLiteral(\"recovered三层投票\")",
                      });
    expectNotContainsAll(run,
                         {
                             "prepareFrameCaches();",
                             "selectMvsSourceViewIndices(_views, _sparse, refIdx, numSrc)",
                             "crossCheckDepthConsistencyStreaming()",
                             "runDepthPoseRefinementCandidateStage(",
                         });
    expectContainsAll(adapter,
                      {
                          "views.size() < 2",
                          "select_recovered_neighbors(scene, 16)",
                          "neighbors[index].empty() || neighbors[index].size() > 16",
                      });
    expectContainsAll(selector,
                      {
                          "common_track_count",
                          "pair_score",
                          "records.front().common_count / 10U",
                      });
    expectContainsAll(orchestrator,
                      {
                          "neighbor_count == 0U || neighbor_count > 16U",
                          "evidence-backed 1..16-neighbor domain",
                      });
}

TEST(MvsSchedulerContractTest, RecoveredVotingUsesBoundedFileBackedBatches)
{
    const QString adapter = readSourceFile(QStringLiteral("src/core/mvs/RecoveredDepthScene.cpp"));
    const QString orchestrator =
        readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/patchmatch_orchestrator.cpp"));
    const QString api = readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/include/metmodel/patchmatch.hpp"));

    expectContainsAll(adapter,
                      {
                          "uniquePatchMatchStoreRoot",
                          "QUuid::createUuid()",
                          "qScopeGuard",
                          "false,",
                          "patchmatch_store_root,",
                          "read_recovered_patchmatch_store_camera(",
                          "std::filesystem::remove_all(patchmatch_store_root",
                          "writeRecoveredModelInput(model_root, scene, recovered, true)",
                      });
    expectContainsAll(api,
                      {
                          "std::size_t voting_batch_size = 16U",
                      });
    expectContainsAll(orchestrator,
                      {
                          "result.voting_batch_size =",
                          "patchmatch_store_root.empty() ? reference_camera_indices.size() : voting_batch_size",
                          "plan_recovered_patchmatch_store_batches(",
                          "for (const auto& voting_batch : voting_batches)",
                      });
}

TEST(MvsDepthArtifactContractTest, RequiredArtifactsFailClosedBeforePublication)
{
    const QString source = readMvsPipelineImplementation();
    const QString saveBlock = sectionBetween(
        source, "bool MvsPipelineService::saveDepthFrameArtifacts", "void MvsPipelineService::captureStageSnapshot");
    const QString photometricFailureBlock =
        sectionBetween(saveBlock, "if (saveRawDepth && !photometricSourceMaskSaved)", "bool geometrySourceMaskSaved");
    expectContainsAll(photometricFailureBlock,
                      {
                          "markManifestFrameFailed(frameIndex, message)",
                          "errorOccurred(message)",
                          "return false;",
                      });
    const QString geometrySourceContractBlock =
        sectionBetween(saveBlock, "if (geometry_source_contract.persistMask)", "const bool inverseDepthMeanSaved");
    EXPECT_EQ(countOccurrences(geometrySourceContractBlock, "markManifestFrameFailed(frameIndex, message)"), 2);
    const QString finalConsistencyArtifactsBlock = sectionBetween(
        saveBlock, "const bool final_consistency_artifacts =", "if (!missing_required_artifacts.empty())");
    expectContainsAll(finalConsistencyArtifactsBlock,
                      {
                          "final_artifacts && consistency_publication_completed",
                          "if (final_consistency_artifacts)",
                          "require_artifact(geometrySupportSaved",
                          "require_artifact(inverseDepthSpreadSaved",
                          "if (final_consistency_artifacts &&",
                          "require_artifact(inverseDepthMeanSaved",
                          "require_artifact(crossViewRepairedMaskSaved",
                      });
    EXPECT_LT(indexOfOrFail(finalConsistencyArtifactsBlock, "final_artifacts && consistency_publication_completed"),
              indexOfOrFail(finalConsistencyArtifactsBlock, "require_artifact(geometrySupportSaved"));

    expectContainsAll(saveBlock,
                      {
                          "QStringList missing_required_artifacts",
                          "require_artifact(rawSaved",
                          "require_artifact(confidenceSaved",
                          "require_artifact(maskSaved",
                          "require_artifact(supportMaskSaved",
                          "require_artifact(depthProvenanceSaved",
                          "require_artifact(missingReasonSaved",
                          "const bool final_artifacts = stageLabel != QStringLiteral(\"初始\")",
                          "consistency_publication_expected != consistency_publication_completed",
                          "已在写盘和公共发布前失败关闭",
                          "const bool final_consistency_artifacts =",
                          "final_artifacts && consistency_publication_completed",
                          "if (final_consistency_artifacts)",
                          "require_artifact(geometrySupportSaved",
                          "require_artifact(inverseDepthSpreadSaved",
                          "if (final_consistency_artifacts &&",
                          "_effectiveSceneProfile == MvsSceneProfile::OrbitalObject",
                          "require_artifact(inverseDepthMeanSaved",
                          "require_artifact(crossViewRepairedMaskSaved",
                          "require_artifact(adaptiveGeometrySupportWeightSaved",
                          "require_artifact(adaptiveGeometryEffectiveViewCountSaved",
                          "require_artifact(adaptiveGeometryConflictRatioSaved",
                          "markManifestFrameFailed(frameIndex, message)",
                          "errorOccurred(message)",
                      });

    const int failClosedGate = indexOfOrFail(saveBlock, "if (!missing_required_artifacts.empty())");
    const int publication = indexOfOrFail(saveBlock, "if (previewSaved && rawSaved && savePreviewPng)");
    const QString failClosedBlock = sectionBetween(
        saveBlock, "if (!missing_required_artifacts.empty())", "if (previewSaved && rawSaved && savePreviewPng)");
    expectContainsAll(failClosedBlock,
                      {
                          "markManifestFrameFailed(frameIndex, message)",
                          "errorOccurred(message)",
                          "return false;",
                      });
    const int markCompleted = indexOfOrFail(saveBlock, "_workspaceManifest.markCompleted(record)");
    const int emitArtifact = indexOfOrFail(saveBlock, "depthMapArtifactSaved(artifact)");
    const int emitDepthSaved = indexOfOrFail(saveBlock, "depthMapSaved(");
    const int manifestPersist = indexOfOrFail(saveBlock, "persistWorkspaceManifest(&manifestError)");
    const QString publicationTail =
        sectionBetween(saveBlock, "MvsDepthFrameRecord record;", "return previewSaved && rawSaved;");
    EXPECT_EQ(countOccurrences(publicationTail, "if (final_artifacts)"), 2);
    expectContainsAll(publicationTail,
                      {
                          "record.status = final_artifacts",
                          "_workspaceManifest.upsertFrame(record)",
                          "_workspaceManifest.markCompleted(record)",
                          "depthMapArtifactSaved(artifact)",
                      });
    EXPECT_LT(indexOfOrFail(publicationTail, "if (final_artifacts)"),
              indexOfOrFail(publicationTail, "_workspaceManifest.markCompleted(record)"));
    EXPECT_LT(failClosedGate, publication);
    EXPECT_LT(failClosedGate, markCompleted);
    EXPECT_LT(failClosedGate, emitArtifact);
    EXPECT_LT(manifestPersist, emitDepthSaved);
    EXPECT_LT(manifestPersist, emitArtifact);
}

TEST(MvsDepthArtifactContractTest, RecoveredPublicationPreservesPhotometricEvidence)
{
    const QString generator = readMvsPipelineImplementation();
    const QString adapter = readSourceFile(QStringLiteral("src/core/mvs/RecoveredDepthScene.cpp"));
    const QString run =
        sectionBetween(generator, "void MvsPipelineService::runInBackgroundImpl()", "} // namespace xjw::mvs");

    expectContainsAll(adapter,
                      {
                          "frame.geometrySupportCount = cv::Mat::zeros(height, width, CV_16U)",
                          "frame.geometrySupportCount.setTo(2, frame.validMask)",
                          "frame.inverseDepthRelativeSpread = cv::Mat::zeros(height, width, CV_32F)",
                          "std::numeric_limits<float>::quiet_NaN(), frame.validMask",
                          "frame.photometricSourceMask = cv::Mat::zeros(height, width, CV_32S)",
                          "unpack_recovered_patchmatch_inlier_mask(",
                          "destination[column] |= source_bit",
                          "frame.photometricSourceMask.setTo(0, frame.validMask == 0)",
                      });
    expectContainsAll(run,
                      {
                          "frame.photometricSourceMask =",
                          "std::move(recovered_frame.photometricSourceMask)",
                          "frame.geometrySupportCount =",
                          "std::move(recovered_frame.geometrySupportCount)",
                          "frame.inverseDepthRelativeSpread =",
                          "std::move(recovered_frame.inverseDepthRelativeSpread)",
                          "adaptive_geometry_fallback_to_discrete_core",
                          "recovered_voting_geometry_proxy",
                          "saveDepthFrameArtifacts(",
                          "QStringLiteral(\"recovered三层投票\")",
                          "markManifestFrameRunning(frame_index)",
                      });

    const QString artifactSaver = sectionBetween(
        generator, "bool MvsPipelineService::saveDepthFrameArtifacts", "void MvsPipelineService::captureStageSnapshot");
    expectContainsAll(artifactSaver,
                      {
                          "photometricSourceMaskSaved",
                          "markManifestFrameFailed(frameIndex, message)",
                          "_workspaceManifest.markCompleted(record)",
                          "depthMapArtifactSaved(artifact)",
                      });
}

TEST(MvsSchedulerContractTest, StandaloneSparseHintsUseProjectedSamplesAndPrescaledPatchMatchInputs)
{
    const QString cameraHeader =
        readSourceFile(QStringLiteral("3rdparty/placamera/include/placamera/frame_numeric_state.h"));
    const QString cameraSource = readSourceFile(QStringLiteral("3rdparty/placamera/src/frame_numeric_state.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/MvsPipelineService.h"));
    const QString scheduler = readMvsPipelineImplementation();
    const QString pyramid = readSourceFile(QStringLiteral("src/core/mvs/DepthPyramidEstimator.cpp"));
    const QString cuda = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCUDA.cu"));
    const QString cpu = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCPU.cpp"));

    expectContainsAll(header,
                      {
                          "ProjectedSparseDepthSample",
                          "buildSparseSeedDepthFromProjectedSamples",
                      });
    const QString projectedBlock = sectionBetween(scheduler,
                                                  "MvsPipelineService::collectProjectedSparseDepthSamples(",
                                                  "cv::Mat MvsPipelineService::buildHintDepthFromProjectedSamples");
    expectContainsAll(scheduler, {"kMaxProjectedDepthQuantileSamples"});
    expectContainsAll(projectedBlock,
                      {
                          "depthQuantileSamples",
                          "std::nth_element",
                          "projectedCandidates",
                          "camera.groundToImage",
                          "candidate.depth",
                      });
    expectNotContainsAll(projectedBlock,
                         {
                             "std::sort(allZc",
                             "allZc.reserve(visiblePointIndices.size())",
                             "float Zc = cam.R_cw[6]*pt[0]",
                             "camera.projectWorldPoint(world",
                         });
    EXPECT_EQ(countOccurrences(projectedBlock, "for (size_t pointIndex : visiblePointIndices)"), 1);

    expectContainsAll(cameraHeader, {"groundToImage", "signedDepth"});
    expectContainsAll(cameraSource,
                      {
                          "FramePinholeNumericState::groundToImage",
                          "internal::projectFrame(",
                          "FramePinholeNumericState::signedDepth",
                      });

    expectContainsAll(pyramid,
                      {
                          "prior = propagateDepthPrior(",
                          "request.referenceImage.size())",
                          "mergeSparseHint(prior, request.sparseDepthHints[index], target_size)",
                          "hint.copyTo(prior.center, sparse_mask)",
                      });

    const QString hintBody = sectionBetween(scheduler,
                                            "cv::Mat MvsPipelineService::buildHintDepthFromProjectedSamples",
                                            "cv::Mat MvsPipelineService::buildSparseSeedDepthFromProjectedSamples");
    expectContainsAll(hintBody,
                      {
                          "cv::distanceTransform",
                          "DIST_LABEL_PIXEL",
                          "maxHintRadius",
                      });
    expectNotContainsAll(hintBody,
                         {
                             "cv::Mat distMap",
                             "INT_MAX / 2",
                         });

    expectContainsAll(cuda,
                      {
                          "hintDepth->cols == sW && hintDepth->rows == sH",
                          "hintScaled = *hintDepth",
                      });
    expectContainsAll(cpu,
                      {
                          "hintDepth->cols == W && hintDepth->rows == H",
                      });
    const QString gpuBody = sectionBetween(
        cuda, "bool PatchMatchDepthEstimator::estimateGPU", "bool PatchMatchDepthEstimator::isCudaAvailable");
    expectContainsAll(gpuBody,
                      {
                          "const int sW = std::max(1, refW / ds);",
                          "const int sH = std::max(1, refH / ds);",
                          "getOrUploadGrayImageGpu(refGray",
                          "image_upload_lane.stream",
                      });
    expectNotContainsAll(gpuBody, {"cv::resize(refGray, refScaled"});
}

TEST(MvsSchedulerContractTest, PatchMatchRequiresRobustMultiViewPhotometricSupport)
{
    const QString policy = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchPhotometricCost.h"));
    const QString cuda = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCUDA.cu"));
    const QString cpu = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCPU.cpp"));

    expectContainsAll(policy,
                      {
                          "robustMultiSourceNcc",
                          "requiredPhotometricSupport",
                          "JointViewSelection",
                          "selectJointSourceViews",
                      });
    expectContainsAll(cpu,
                      {
                          "cpuEvalHypCost",
                          "const JointViewSelection selection = selectJointSourceViews(",
                          "result.photometricNcc = selection.photometricScore",
                          "result.sourceMask = selection.sourceMask",
                      });
    expectContainsAll(cuda,
                      {
                          "evalHypCost",
                          "const JointViewSelection selection = selectJointSourceViews(",
                          "*selectedMask = selection.sourceMask",
                          "*photometricNcc = selection.photometricScore",
                      });
    expectNotContainsAll(cuda,
                         {
                             "if (ncc > 0.05f) { sumScore += ncc; ++goodSrc; }",
                             "sumScore / static_cast<float>(goodSrc)",
                             "sumScore / goodSrc",
                         });
}

TEST(MvsSchedulerContractTest, FinalPyramidLevelKeepsConfiguredIterationBudget)
{
    const QString policy = readSourceFile(QStringLiteral("src/core/mvs/DepthPyramidPolicy.cpp"));

    expectContainsAll(policy,
                      {
                          "result.patchMatch = base_config",
                          "result.minSupportViews = 3",
                          "result.radiusScale = 1.0f",
                      });
    expectNotContainsAll(policy,
                         {
                             "base_config.numIterations - 1",
                             "hintCoverage",
                         });
}

TEST(MvsHeterogeneousSchedulingContractTest, RecoveredProductionIsStrictlyCudaOnly)
{
    const QString generator = readMvsPipelineImplementation();
    const QString run =
        sectionBetween(generator, "void MvsPipelineService::runInBackgroundImpl()", "} // namespace xjw::mvs");

    expectContainsAll(run,
                      {
                          "configuredBackend == PatchMatchBackend::Cpu ||",
                          "configuredBackend == PatchMatchBackend::OpenCl",
                          "仅支持 CUDA；CPU/OpenCL 不会回退到旧 PatchMatch",
                          "_config.patchMatch.backend = PatchMatchBackend::Cuda",
                          "recovered 多视深度需要可用 CUDA 设备；未执行旧算法回退",
                          "runRecoveredDepthScene(",
                      });
    expectNotContainsAll(run,
                         {
                             "DepthComputeScheduler computeScheduler",
                             "computeDepthForView(",
                         });
}

TEST(MvsHeterogeneousSchedulingContractTest, RecoveredCudaKeepsReferenceFloatingPointSemantics)
{
    const QString cmake = readSourceFile(QStringLiteral("src/core/mvs/CMakeLists.txt"));
    const QString generator = readMvsPipelineImplementation();
    const QString cudaSource =
        readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/recovered_cuda_source.cu"));
    const QString orchestrator =
        readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/patchmatch_orchestrator.cpp"));
    const QString run =
        sectionBetween(generator, "void MvsPipelineService::runInBackgroundImpl()", "} // namespace xjw::mvs");

    expectContainsAll(cmake,
                      {
                          "recovered_depth/src/gpu_cuda.cu",
                          "recovered_depth/src/recovered_cuda_source.cu",
                          "--ftz=false;--prec-div=true;--prec-sqrt=true",
                      });
    expectContainsAll(cudaSource,
                      {
                          "template <bool AffineTransform>",
                          "camera.transform[15]) == 0x3f800000U",
                          "launch_recovered_patchmatch_filter_speckles_edges_typed_source<0U, true>",
                      });
    expectContainsAll(orchestrator,
                      {
                          "const std::size_t checker_items = ((width + 1U) / 2U) * height;",
                          "propagation.global_work_items = checker_items;",
                      });
    EXPECT_FALSE(orchestrator.contains(QStringLiteral("(pixels + 1U) / 2U")));
    expectContainsAll(run,
                      {
                          "for (RecoveredDepthFrame& recovered_frame : recovered_result.frames)",
                          "saveDepthFrameArtifacts(",
                          "depthMapReady(_depthFrames",
                      });
    expectNotContainsAll(run,
                         {
                             "DepthFrameArtifactSaveQueue",
                             "saveQueue.reserveProducer",
                             "computeDepthForView(",
                         });
}

TEST(MvsHeterogeneousSchedulingContractTest, RecoveredPropagationPassesRotationByValue)
{
    const QString cudaSource =
        readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/recovered_cuda_source.cu"));
    const QString perspectiveKernel = sectionBetween(cudaSource,
                                                     "patchmatch_propagation_u8_perspective_kernel(",
                                                     "__device__ __forceinline__ float recovered_propagation_sample");
    const QString extendedKernel = sectionBetween(
        cudaSource, "patchmatch_propagation_extended_kernel(", "patchmatch_final_refinement_extended_kernel(");

    expectContainsAll(cudaSource,
                      {
                          "struct alignas(16) RecoveredPropagationMatrix3x3f",
                          "static_assert(sizeof(RecoveredPropagationMatrix3x3f) == 48U)",
                          "make_recovered_propagation_rotation(rotation_to_local)",
                      });
    expectContainsAll(perspectiveKernel,
                      {
                          "RecoveredPropagationMatrix3x3f rotation_to_local",
                          "rotation_to_local.values[base + 2U]",
                      });
    expectContainsAll(extendedKernel,
                      {
                          "RecoveredPropagationMatrix3x3f rotation_to_local",
                          "rotation_to_local.values[base + 2U]",
                      });
    EXPECT_FALSE(perspectiveKernel.contains(QStringLiteral("const float* rotation_to_local")));
    EXPECT_FALSE(extendedKernel.contains(QStringLiteral("const float* rotation_to_local")));
}

TEST(MvsHeterogeneousSchedulingContractTest, RecoveredUsesReferenceParallelHotPaths)
{
    const QString cmake = readSourceFile(QStringLiteral("src/core/mvs/CMakeLists.txt"));
    const QString patchmatchHeader =
        readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/include/metmodel/patchmatch.hpp"));
    const QString patchmatch = readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/patchmatch.cpp"));
    const QString orchestrator =
        readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/patchmatch_orchestrator.cpp"));
    const QString octree = readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/octree_prepare.cpp"));
    const QString neighbors = readSourceFile(QStringLiteral("src/core/mvs/recovered_depth/src/ooc_neighbors_cuda.cu"));

    expectContainsAll(cmake, {"recovered_depth/src/ooc_neighbors_cuda.cu"});
    expectContainsAll(patchmatchHeader,
                      {"std::span<const float> depth_view;",
                       "std::span<const std::uint8_t> normal_view;",
                       "std::span<const float> cost_view;",
                       "std::uint64_t set_device_nanoseconds = 0;",
                       "std::uint64_t cuda_free_nanoseconds = 0;",
                       "std::uint64_t cost_phase_nanoseconds = 0;",
                       "std::uint64_t voting_phase_nanoseconds = 0;"});
    expectContainsAll(patchmatch,
                      {"parallel_for_recovered_patchmatch_rows(",
                       "pixels < 131072U",
                       "std::max(1U, logical_cpus / 2U)",
                       "METMODEL_PM_HOST_TIMING",
                       "PATCHMATCH_HOST_SPECKLES",
                       "PATCHMATCH_HOST_CROSS_LEVEL"});
    expectContainsAll(orchestrator,
                      {"c2p.depth_view = state.depth;",
                       "c2p.normal_view = state.normal;",
                       "c2p.cost_view = state.cost;",
                       "patchmatch_finer_level_speckle_component_threshold(downscale)",
                       "const std::size_t x16_width = (camera.image.width + 15U) / 16U;",
                       "const std::size_t x16_height = (camera.image.height + 15U) / 16U;"});
    EXPECT_FALSE(orchestrator.contains(QStringLiteral("target_level ? 30U : 6U")));
    expectContainsAll(octree,
                      {"run_recovered_ooc_neighbors_cuda_source(",
                       "bool same_index_space = records.size() == balanced_records.size();",
                       "std::lower_bound(nodes.begin(), nodes.end(), child_key, node_less)",
                       "std::vector<std::optional<LocalMarchingRawPart>> raw_parts"});
    expectContainsAll(neighbors,
                      {"neighbors_binary_search_kernel<<<blocks, threads, 0U, stream>>>",
                       "constexpr std::size_t batch_capacity = 1'000'000U;"});
}

TEST(MvsAdaptivePatchMatchContractTest, AuxiliaryEvidenceCrossesEveryAdaptiveBackendBranch)
{
    const QString generator = readMvsPipelineImplementation();
    const QString adaptive_helper = readSourceFile(QStringLiteral("src/core/mvs/pipeline/FrameBackend.cpp"));
    const QString adaptive_declarations =
        readSourceFile(QStringLiteral("src/core/mvs/pipeline/MvsPipelineInternals.h"));
    const QString adaptive_backend =
        sectionBetween(generator, "class AdaptivePatchMatchBackend final", "} // namespace");

    expectContainsAll(adaptive_declarations,
                      {
                          "const PatchMatchAuxiliaryInput* auxiliaryInput = nullptr",
                          "PatchMatchAuxiliaryOutput* auxiliaryOutput = nullptr",
                      });
    EXPECT_EQ(countOccurrences(adaptive_helper, "PatchMatchDepthEstimator::estimate("), 3);
    EXPECT_EQ(countOccurrences(adaptive_helper, "auxiliaryInput"), 4);
    EXPECT_EQ(countOccurrences(adaptive_helper, "auxiliaryOutput"), 4);

    expectContainsAll(adaptive_backend,
                      {
                          "auxiliary_input.sourceDepthMaps =",
                          "request.sourceDepthMaps.empty() ? nullptr : &request.sourceDepthMaps;",
                          "auxiliary_output.photometricSourceMask =",
                          "&result.photometricSourceMask",
                          "&auxiliary_input",
                          "&auxiliary_output",
                          "selectedSourceCount(",
                      });
    EXPECT_FALSE(adaptive_backend.contains(
        QStringLiteral("supportCount.setTo(\n"
                       "            cv::Scalar(static_cast<int>(request.sourceImages.size()))")));
}

TEST(MeshReconstructionContractTest, ClosedSurfaceNormalsUseNeighborhoodConsistency)
{
    const QString source = readSourceFile(QStringLiteral("src/core/mesh/SurfaceReconstructor.cpp"));
    const QString orientation =
        sectionBetween(source, "void orientNormalsOutwardFromCentroid", "PlaPointCloud pointXYZRGBToCloud");

    expectContainsAll(orientation,
                      {
                          "plapoint::search::KdTree",
                          "nearestKSearch",
                          "#pragma omp parallel for",
                          "component_outward_score",
                      });
    expectNotContainsAll(orientation,
                         {
                             "if (outward_dot < 0.0)",
                         });
}

TEST(GuiDialogLayoutContractTest, DialogSourcesAreGroupedByDomain)
{
    const QString gui_sources = readSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    const QString dialog_sources = readSourceFile(QStringLiteral("src/gui/cmake/GuiDialogSources.cmake"));
    const QString layout_readme = readSourceFile(QStringLiteral("src/gui/dialogs/README.md"));
    const QString workspace_ui = readSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.ui"));

    expectContainsAll(gui_sources,
                      {
                          "include(${CMAKE_CURRENT_LIST_DIR}/GuiDialogSources.cmake)",
                          "views/CameraSceneWidget.cpp",
                          "views/CameraSceneViewMath.cpp",
                          "views/ObjRenderPreparation.cpp",
                      });
    expectNotContainsAll(gui_sources,
                         {
                             "dialogs/application/AboutDialog.cpp",
                             "dialogs/reconstruction/GenerateModelDialog.cpp",
                             "dialogs/tie_points/MatchViewerDialog.cpp",
                         });
    expectContainsAll(dialog_sources,
                      {
                          "GUI_APPLICATION_DIALOG_SOURCES",
                          "GUI_CAMERA_DIALOG_SOURCES",
                          "GUI_IMAGE_DIALOG_SOURCES",
                          "GUI_RECONSTRUCTION_DIALOG_SOURCES",
                          "GUI_TIE_POINT_DIALOG_SOURCES",
                          "GUI_SHARED_DIALOG_SOURCES",
                          "dialogs/application/AboutDialog.cpp",
                          "dialogs/image/GenerateMaskDialog.cpp",
                          "dialogs/reconstruction/CreatePointCloudDialog.cpp",
                          "dialogs/reconstruction/GenerateModelDialog.cpp",
                          "dialogs/tie_points/MatchViewerDialog.cpp",
                          "dialogs/shared/WorkflowParameterDialogStyle.cpp",
                      });
    expectContainsAll(layout_readme,
                      {
                          "`application/`",
                          "`camera/`",
                          "`image/`",
                          "`reconstruction/`",
                          "`tie_points/`",
                          "`shared/`",
                      });
    EXPECT_TRUE(workspace_ui.contains(QStringLiteral("<header>CameraSceneWidget.h</header>")));
}

TEST(GuiArchitectureContractTest, PointCloudWorkflowControllerOnlyCoordinatesCoreMvs)
{
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/gui/project/manager/ProjectPointCloudWorkflowController.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/gui/project/manager/ProjectPointCloudWorkflowController.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.cpp")));

    const QString guiSources = readSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("ProjectPointCloudWorkflowController")));
    const QString pointCloudController =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectPointCloudWorkflowController.cpp"));
    expectContainsAll(pointCloudController,
                      {
                          "xjw::gui::tasks::DepthMapTask",
                          "xjw::mvs::fuseDepthMapsStreaming",
                          "xjw::gui::tasks::runGuardedWithOutcome",
                          "PinholeModelsByImageId",
                          "view.camera = std::move(camera)",
                          "影像尺寸与 PlaCamera 相机不一致",
                      });
    expectNotContainsAll(pointCloudController,
                         {
                             "FramePinholeNumericState",
                             "toModel()",
                         });
    const QString dense_publication =
        sectionBetween(pointCloudController, "const QJsonObject processing =", "manager->finishTask(true);");
    expectContainsAll(dense_publication,
                      {
                          "bool published = false",
                          "published = manager->_session->replaceResultRecordWithLatest",
                          "published = manager->_session->upsertResultRecordByPath",
                          "if (!published)",
                          "manager->failTask",
                      });
    EXPECT_LT(dense_publication.indexOf(QStringLiteral("if (!published)")),
              dense_publication.indexOf(QStringLiteral("emit manager->pointCloudResultReady")));
    expectNotContainsAll(pointCloudController,
                         {
                             "PatchMatchCPU",
                             "PatchMatchCuda",
                         });
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/manager/ProjectTaskDispatcher.cpp")));
}

TEST(MvsArchitectureContractTest, ObsoleteStereoDenseCloudPipelineIsRemoved)
{
    for (const char* path : {
             "src/core/mvs/StereoDenseCloudPipeline.h",
             "src/core/mvs/StereoDenseCloudPipeline.cpp",
             "src/core/mvs/StereoDenseCloudPipelineOutput.h",
             "src/core/mvs/StereoDenseCloudPipelineOutput.cpp",
             "src/core/mvs/StereoDenseCloudPipelinePaths.h",
             "src/core/mvs/StereoDenseCloudPipelinePaths.cpp",
             "src/core/mvs/AspPointCloudMetrics.h",
             "src/core/mvs/AspPointCloudMetrics.cpp",
             "src/core/mvs/DisparityFilter.h",
             "src/core/mvs/DisparityFilter.cpp",
             "src/core/mvs/PointCloudTifIO.h",
             "src/core/mvs/PointCloudTifIO.cpp",
             "src/core/mvs/SubpixelRefiner.h",
             "src/core/mvs/SubpixelRefiner.cpp",
             "src/core/terrain/tests/terrain_stereo_test.cpp",
             "src/core/terrain/tests/stereo_pipeline_test.cpp",
             "src/core/terrain/tests/stereo_pipeline_benchmark.cpp",
         })
    {
        EXPECT_FALSE(sourceFileExists(utf8(path))) << path;
    }

    const QString mvsCmake = readSourceFile(QStringLiteral("src/core/mvs/CMakeLists.txt"));
    EXPECT_FALSE(mvsCmake.contains(QStringLiteral("StereoDenseCloudPipeline")));
}

TEST(ProjectWorkflowContractTest, LegacyDenseWorkflowConfigShimsAreRemoved)
{
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/project_workflows/ProjectDenseWorkflowConfig.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.h")));

    const QString cliCmake = readSourceFile(QStringLiteral("src/cli/workflows/CMakeLists.txt"));
    EXPECT_FALSE(cliCmake.contains(QStringLiteral("src/gui")));
}

TEST(GuiArchitectureContractTest, DenseMatchDiagnosticLibraryIsNotLinkedIntoGui)
{
    const QString guiCmake = readSourceFile(QStringLiteral("src/gui/CMakeLists.txt")) +
                             readSourceFile(QStringLiteral("src/gui/cmake/GuiCoreLinking.cmake"));

    EXPECT_FALSE(guiCmake.contains(QStringLiteral("dense_match")));
}

TEST(GuiArchitectureContractTest, GuiTestsReuseProductionRuntimeLibrary)
{
    const QString guiCmake = readSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));
    const QString testsCmake = readSourceFile(QStringLiteral("tests/CMakeLists.txt"));
    const QString testSources = sectionBetween(
        testsCmake, "add_executable(test_gui_project_utils", "set_target_properties(test_gui_project_utils");
    const QString testLinks = sectionBetween(testsCmake,
                                             "target_link_libraries(test_gui_project_utils PRIVATE",
                                             "target_compile_definitions(test_gui_project_utils");

    expectContainsAll(guiCmake,
                      {
                          "add_library(gui_runtime STATIC",
                          "list(REMOVE_ITEM GUI_RUNTIME_SOURCES main.cpp)",
                          "target_link_libraries(plascan_gui PRIVATE",
                          "gui_runtime",
                      });
    EXPECT_FALSE(testSources.contains(QStringLiteral("${CMAKE_SOURCE_DIR}/src/gui/")));
    EXPECT_TRUE(testLinks.contains(QStringLiteral("gui_runtime")));
}

TEST(CoreArchitectureContractTest, LegacyIntersectionTargetIsRemoved)
{
    const QString coreCmake = readSourceFile(QStringLiteral("src/core/CMakeLists.txt"));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/intersection/CMakeLists.txt")));
    EXPECT_FALSE(coreCmake.contains(QStringLiteral("plascan_core_add_optional_module(intersection")));
}

TEST(GuiArchitectureContractTest, AsyncTasksExposeSharedCancellationVocabulary)
{
    const QString runner = readSourceFile(QStringLiteral("src/gui/tasks/GuiTaskRunner.h"));
    const QString maskController =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectMaskWorkflowController.cpp"));
    const QString maskHeader =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectMaskWorkflowController.h"));
    const QString orchestrator = readSourceFile(QStringLiteral("src/gui/project/tasks/ProjectTaskOrchestrator.cpp"));

    expectContainsAll(runner,
                      {
                          "class TaskCancellationToken",
                          "class TaskCancellationSource",
                          "isCancellationRequested",
                          "requestCancellation",
                      });
    expectContainsAll(maskController,
                      {
                          "context.cancelFlag->load",
                          "target.stagingPath",
                          "publishImageMaskRecords",
                          "QSaveFile output(destinationPath)",
                          "copyFileAtomically(artifact.stagingPath, artifact.finalPath, &publish_error)",
                      });
    expectContainsAll(orchestrator,
                      {
                          "MaskCancelReason::User",
                          "MaskCancelReason::SessionChanged",
                          "MaskCancelReason::Destroying",
                          "_maskContext.cancelFlag->store",
                      });
    EXPECT_FALSE(maskHeader.contains(QStringLiteral("ProjectData")));
    EXPECT_FALSE(maskController.contains(QStringLiteral("QMessageBox")));
    EXPECT_FALSE(maskController.contains(QStringLiteral("GenerateMaskDialog")));
}

TEST(GuiArchitectureContractTest, ProjectPersistenceAndWorkflowAlgorithmsLiveOutsideGui)
{
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/common/project/ProjectSessionModel.cpp")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/common/project/ProjectDocumentModel.cpp")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/project_workflows/ProjectWorkflowOperations.cpp")));

    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/data/ProjectData.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/data/ProjectFilesManager.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/support/ProjectWorkflowUtils.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/data/ProjectData.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/data/ProjectFilesManager.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/services/ProjectResourceCleanupService.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/support/ProjectReferenceDatasets.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/gui/project/support/ProjectWorkflowUtils.h")));

    const QString guiSources = readSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    EXPECT_FALSE(guiSources.contains(QStringLiteral("project/data/ProjectData.cpp")));
    EXPECT_FALSE(guiSources.contains(QStringLiteral("project/support/ProjectWorkflowUtils.cpp")));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("main_window/MainWindowProjectLifecycle.cpp")));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("widgets/DataTreePopulation.cpp")));
}

#include <gtest/gtest.h>

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

QString readSourceFile(const QString &relativePath)
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

bool sourceFileExists(const QString &relativePath)
{
    return QFileInfo::exists(QDir(repoRoot()).filePath(relativePath));
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

QString utf8(const char *text)
{
    return QString::fromUtf8(text);
}

void expectContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_TRUE(text.contains(utf8(needle))) << needle;
    }
}

void expectNotContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_FALSE(text.contains(utf8(needle))) << needle;
    }
}

int countOccurrences(const QString &text, const char *needle)
{
    return text.count(utf8(needle));
}

int indexOfOrFail(const QString &text, const char *needle, int from = 0)
{
    const int index = text.indexOf(utf8(needle), from);
    EXPECT_GE(index, 0) << needle;
    return index;
}

QString sectionBetween(const QString &text, const char *startNeedle, const char *endNeedle, int from);

TEST(SfmModuleContractTest, ModuleOwnedTestsLiveBesideSfm)
{
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/test/CMakeLists.txt")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("tests/test_sfm_pipeline.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("tests/test_sparse_point_cloud_processor.cpp")));
    EXPECT_TRUE(sourceFileExists(
        QStringLiteral("src/core/aerial_triangulation/tests/test_sfm_pair_planner.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("tests/test_sfm_pair_planner.cpp")));
}

TEST(ProjectCommonModuleContractTest, DeletedGuiProjectIoDirectoryIsNotAnIncludeRoot)
{
    const QString buildDefinitions =
        readSourceFile(QStringLiteral("src/cli/CMakeLists.txt")) +
        readSourceFile(QStringLiteral("tests/CMakeLists.txt"));

    EXPECT_FALSE(buildDefinitions.contains(QStringLiteral("src/gui/project/io")));
}

TEST(GuiStyleContractTest, WorkspaceTreeUsesApplicationColorsInsteadOfSystemPalette)
{
    const QString treeSource =
        readSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    const QString applicationStyle =
        readSourceFile(QStringLiteral("resources/styles/app.qss"));

    EXPECT_FALSE(treeSource.contains(QStringLiteral("palette(")));
    expectContainsAll(applicationStyle, {
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
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/core/sfm/common/PhotogrammetryPointAttributes.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/filtering/SfmPointCloudFilter.h")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/filtering/SfmPointCloudFilter.cpp")));
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/core/sfm/triangulation/InitialSparsePointCloudTriangulator.h")));
    EXPECT_TRUE(sourceFileExists(
        QStringLiteral("src/core/sfm/triangulation/InitialSparsePointFilter.h")));
    expectNotContainsAll(processorHeader, {
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
    const QString trackSource =
        readSourceFile(QStringLiteral("src/core/sfm/tracks/MultiViewTrackBuilder.cpp"));

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
    const QString baInputBuilder =
        readSourceFile(QStringLiteral("src/core/sfm/project/BaTrackBuilder.cpp"));
    const QString pnpSolver =
        readSourceFile(QStringLiteral("src/core/sfm/pose/PnpSolver.cpp"));
    const QString incrementalSfm = readIncrementalSfmImplementation();

    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/geometry/ProjectionGeometry.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/geometry/TriangulationQuality.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/geometry/OpenCvCameraAdapter.h")));
    expectNotContainsAll(initialFilter, {
        "double reprojectionErrorPx(",
        "double minimumTriangulationAngleDeg(",
    });
    expectNotContainsAll(baInputBuilder, {
        "PairIntersectionCandidate triangulatePairWithDirectionFallback(",
        "double reprojectionErrorPx(",
    });
    expectNotContainsAll(pnpSolver, {
        "cameraToWorldRotationToOpenCvRvec",
        "cameraCenterToOpenCvTvec",
    });
    EXPECT_FALSE(incrementalSfm.contains(QStringLiteral("const double depthSign =")));
}

TEST(SfmModuleContractTest, BaInputBuilderOnlyOrchestratesProjectAdapters)
{
    const QString builder =
        readSourceFile(QStringLiteral("src/core/sfm/project/BaInputBuilder.cpp"));

    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/BaInputBuilder.cpp")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/sfm/BaInputBuilder.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/ProjectMatchInputReader.cpp")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/BaTrackBuilder.cpp")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/SurveyControlBaAdapter.cpp")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/MarkerBaAdapter.cpp")));
    expectNotContainsAll(builder, {
        "QFile",
        "QJsonArray",
        "MultiViewTrackBuilder",
        "triangulatePairWithDirectionFallback",
        "solveControlNetwork",
    });
}

TEST(SfmModuleContractTest, IncrementalSfmDelegatesMajorResponsibilities)
{
    const QString incrementalSfm =
        readSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfm.cpp"));

    for (const QString &component : {
             QStringLiteral("InitialPairInitializer"),
             QStringLiteral("ImageRegistrationEngine"),
             QStringLiteral("KnownPoseReconstructor"),
             QStringLiteral("SfmBundleAdjustCoordinator"),
         })
    {
        EXPECT_TRUE(sourceFileExists(
            QStringLiteral("src/core/sfm/pipeline/%1.h").arg(component)));
        EXPECT_TRUE(sourceFileExists(
            QStringLiteral("src/core/sfm/pipeline/%1.cpp").arg(component)));
    }

    expectNotContainsAll(incrementalSfm, {
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
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/quality/SfmError.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/SfmQualityJsonSerializer.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/sfm/project/SfmQualityJsonSerializer.cpp")));

    const QString metricsHeader =
        readSourceFile(QStringLiteral("src/core/sfm/quality/SfmQualityMetrics.h"));
    const QString metricsSource =
        readSourceFile(QStringLiteral("src/core/sfm/quality/SfmQualityMetrics.cpp"));
    expectNotContainsAll(metricsHeader + metricsSource, {
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
    expectContainsAll(cmake, {
        "add_library(sfm_core STATIC",
        "add_library(sfm_postprocess STATIC",
        "add_library(sfm_project STATIC",
        "add_library(sfm INTERFACE)",
        "target_link_libraries(sfm INTERFACE sfm_core sfm_postprocess sfm_project)",
    });

    const QString coreLinks = sectionBetween(cmake,
                                             "target_link_libraries(sfm_core",
                                             "target_compile_features(sfm_core",
                                             0);
    const QString postprocessLinks = sectionBetween(cmake,
                                                    "target_link_libraries(sfm_postprocess",
                                                    "target_compile_features(sfm_postprocess",
                                                    0);
    expectNotContainsAll(coreLinks + postprocessLinks, {
        "Qt6::Core",
        "Qt6::Gui",
    });

    const QString algorithmSources =
        readIncrementalSfmImplementation() +
        readSourceFile(QStringLiteral("src/core/control_points/registration/ControlNetworkSolver.h")) +
        readSourceFile(QStringLiteral("src/core/control_points/registration/ControlNetworkSolver.cpp"));
    expectNotContainsAll(algorithmSources, {
        "#include <Q",
        "QString",
        "QVector",
    });
}

QString sectionBetween(const QString &text, const char *startNeedle, const char *endNeedle, int from = 0)
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

QString sectionFrom(const QString &text, const char *startNeedle)
{
    const int start = indexOfOrFail(text, startNeedle);
    if (start < 0)
    {
        return QString();
    }
    return text.mid(start);
}

QString functionBody(const QString &source, const char *signature)
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

void expectMatches(const QString &text, const char *pattern)
{
    const QRegularExpression re(utf8(pattern), QRegularExpression::DotMatchesEverythingOption);
    EXPECT_TRUE(re.match(text).hasMatch()) << pattern;
}

void expectNotMatches(const QString &text, const char *pattern)
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

    expectContainsAll(selectionBody, {
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

    expectContainsAll(selectionBody, {
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

    expectContainsAll(header, {
        "evaluateMultipleInitialPairModels",
        "multiInitialPairMaxImages",
    });
    expectContainsAll(source, {
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

    expectContainsAll(sfmHeader, {
        "findRegisteredSequenceNeighbor",
        "stepsOut",
        "用最近的已注册序列相机",
    });

    const QString neighborLookup = sectionBetween(sfmSource,
                                                  "bool IncrementalSfm::findRegisteredSequenceNeighbor",
                                                  "bool IncrementalSfm::hasRegisteredSequenceNeighbor");
    expectContainsAll(neighborLookup, {
        "direction",
        "step < imageCount",
        "hasRegisteredCamera(candidate)",
        "stepsOut",
    });

    const QString poseGuess = sectionBetween(sfmSource,
                                             "bool IncrementalSfm::makeSequenceInitialPoseGuess",
                                             "void IncrementalSfm::runBundleAdjust");
    expectContainsAll(poseGuess, {
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
    const QString registration = sectionBetween(sfmSource,
                                                "IncrementalSfmResult IncrementalSfm::runRegistrationFromCurrentInitialization",
                                                "void IncrementalSfm::resetForInitialPairTrial");

    expectContainsAll(registration, {
        "registerImage(nextId)",
        "SfmBundleAdjustCoordinator(*this).iterative(true)",
    });
    EXPECT_FALSE(sfmSource.contains(QStringLiteral("tryRegisterInterpolatedSequenceImages")));
    EXPECT_FALSE(sfmSource.contains(QStringLiteral("Sequence interpolation registered")));
}

TEST(SfmSourceContractTest, SequencePnpRecoveryRunsOnlyAfterStandardPnpFails)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration = sectionBetween(source,
                                                "bool IncrementalSfm::registerImage",
                                                "bool IncrementalSfm::findRegisteredSequenceNeighbor");

    expectContainsAll(registration, {
        "allowBracketedSequencePnpRelaxation",
        "findRegisteredSequenceNeighbor(imageId,",
        "&previousImageId",
        "&nextImageId",
        "if (!pnpResult.success && makeSequenceInitialPoseGuess",
        "hasDirectPrevious && hasDirectNext",
        "hasDirectPrevious != hasDirectNext",
        "recoveryOptions.allowRelaxedInlierRatio = true",
        "oneSidedSequencePnpMinInlierRatio",
        "oneSidedSequencePnpMinInliers",
        "bracketedSequencePnpMinInlierRatio",
        "bracketedSequencePnpMinInliers",
    });
}

TEST(SfmSourceContractTest, MatchGeometryFilteringUsesSeededSerialUsac)
{
    const QString source = readSourceFile(QStringLiteral(
        "src/core/image_matching/geometry/MatchGeometryVerifier.cpp"));

    expectContainsAll(source, {
        "params.randomGeneratorState = options.randomSeed",
        "params.isParallel = false",
    });
}

TEST(SfmSourceContractTest, FinalGlobalBaRetriesUnregisteredImagesBeforePublishingResult)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration = sectionBetween(
        source,
        "IncrementalSfmResult IncrementalSfm::runRegistrationFromCurrentInitialization",
        "void IncrementalSfm::resetForInitialPairTrial");

    const int finalBa = indexOfOrFail(
        registration,
        "SfmBundleAdjustCoordinator(*this).iterative(true);");
    const int retry = indexOfOrFail(registration, "retryUnregisteredImagesAfterFinalBA", finalBa);
    const int publish = indexOfOrFail(registration, "result.numRegisteredImages", retry);
    EXPECT_LT(finalBa, retry);
    EXPECT_LT(retry, publish);
}

TEST(SfmSourceContractTest, SequenceInitialPoseInterpolatesCenterAndRotation)
{
    const QString source = readIncrementalSfmImplementation();
    const QString guess = sectionBetween(
        source,
        "bool IncrementalSfm::makeSequenceInitialPoseGuess",
        "// ============================================================\n// 内部：光束法平差");

    expectContainsAll(guess, {
        "interpolateCameraRotation",
        "prevSteps",
        "nextSteps",
        "rotation = interpolateCameraRotation",
    });
}

TEST(SfmSourceContractTest, BracketedSequencePnpUsesInitialPoseCorrespondenceGate)
{
    const QString registration = sectionBetween(
        readIncrementalSfmImplementation(),
        "bool IncrementalSfm::registerImage",
        "bool IncrementalSfm::findRegisteredSequenceNeighbor");
    const QString pnp = readSourceFile(QStringLiteral("src/core/sfm/pose/PnpSolver.cpp"));

    expectContainsAll(registration, {
        "recoveryOptions.useInitialPosePrefilter = true",
        "initialPosePrefilterMaxReprojError",
    });
    expectContainsAll(pnp, {
        "cv::projectPoints",
        "guidedOriginalIndices",
        "initialPosePrefilterMinCandidates",
    });
}

TEST(SfmSourceContractTest, CudaSiftExtractionIsPartOfUnifiedImageMatchingModule)
{
    const QString extractor = readSourceFile(
        QStringLiteral("src/core/image_matching/sift/SiftFeatureExtractor.cpp"));
    const QString cmake = readSourceFile(
        QStringLiteral("src/core/image_matching/CMakeLists.txt"));

    expectContainsAll(extractor, {"ExtractSift", "PLASCAN_HAS_CUDA_SIFT", "validMask"});
    expectContainsAll(cmake, {"SiftFeatureExtractor.cpp", "plascan_cudasift"});
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/feature_extractors")));
    EXPECT_FALSE(sourceFileExists(QStringLiteral("src/core/feature_match")));
}

TEST(SfmSourceContractTest, LightGlueSiftCarriesScaleAndOrientation)
{
    const QString exportScript = readSourceFile(
        QStringLiteral("scripts/models/export_lightglue_tensorrt.py"));
    const QString matcher = readSourceFile(
        QStringLiteral("src/core/image_matching/lightglue/TensorRtLightGlueMatcher.cpp"));

    expectContainsAll(exportScript, {
        R"(LightGlue(features="sift", **common))",
        "geometry0 = torch.cat([xy0, keypoints0[..., 2:4]], dim=-1)",
        "geometry1 = torch.cat([xy1, keypoints1[..., 2:4]], dim=-1)",
        R"("keypoints0")",
        R"("keypoints1")",
    });
    expectContainsAll(matcher, {
        R"(feature.sourceAlgorithm != "sift")",
        "keypoint.size",
        "keypoint.angle",
        "CV_PI",
    });
}

TEST(BuildDependencyContractTest, ProductionBuildDoesNotDependOnLibTorch)
{
    const QString packages = readSourceFile(QStringLiteral("cmake/PlascanPackages.cmake"));
    const QString dependencyPaths = readSourceFile(
        QStringLiteral("cmake/PlascanDependencyPaths.cmake"));
    const QString maskCmake = readSourceFile(QStringLiteral("src/core/mask/CMakeLists.txt"));
    const QString guiCmake = readSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));

    for (const QString &source : {packages, dependencyPaths, maskCmake, guiCmake})
    {
        expectNotContainsAll(source, {
            "find_package(Torch",
            "TORCH_LIBRARIES",
            "TORCH_INCLUDE_DIRS",
            "Torch_DIR",
        });
    }
    EXPECT_FALSE(QFileInfo::exists(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR))
            .filePath(QStringLiteral("src/core/mask/Sam21MaskGenerator.cpp"))));
}

TEST(GuiAlgorithmAlignmentContractTest, WorkflowQualityMappingCoversFastStandardQuality)
{
    const QString guiSource = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString workflow = readSourceFile(
        QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));
    const QString helper = functionBody(workflow, "QualityPreset presetForQuality");

    expectContainsAll(helper, {
        R"(QStringLiteral("highest"))",
        R"(QStringLiteral("medium"))",
        R"(QStringLiteral("low"))",
        R"(QStringLiteral("fast"))",
    });
    EXPECT_GE(countOccurrences(guiSource, "workflowOptions.quality ="), 1);
}

TEST(GuiAlgorithmAlignmentContractTest, SequenceReferencePreselectionOwnsPairPlanning)
{
    const QString guiSource = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString workflow = readSourceFile(
        QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));
    ASSERT_FALSE(guiSource.isEmpty());
    ASSERT_FALSE(workflow.isEmpty());

    expectContainsAll(guiSource, {
        "bool shouldUseStoredGeneratedPairConstraints",
        "reference_preselection_source",
        "sequence",
    });

    const QString sequenceHelpers = sectionBetween(guiSource,
                                                   "bool isSequenceReferencePreselection",
                                                   "QStringList loadGeneratedPairConstraints");
    expectContainsAll(sequenceHelpers, {
        "reference_preselection_source",
        "sequence",
        "shouldUseStoredGeneratedPairConstraints",
        "return false",
    });

    const QString resolvedConfig = sectionBetween(
        workflow,
        "AerialTriangulationResolvedConfig AerialTriangulationWorkflow::resolveConfig",
        "AerialTriangulationResult AerialTriangulationWorkflow::run");
    expectContainsAll(resolvedConfig, {
        "matchphotos::PairSelectionMode::Sequence",
        "tieOptions.pairPolicy.sequenceWindow",
        "tieOptions.useReferencePreselection = false",
    });

    const QString unifiedRun = sectionBetween(guiSource,
                                              "void MenuWorkflowController::runUnifiedAerialTriangulation",
                                              "void MenuWorkflowController::openOverlapAnalysisDialog");
    expectContainsAll(unifiedRun, {
        "shouldUseStoredGeneratedPairConstraints(settings)",
        "loadGeneratedPairConstraints",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, AerialTriangulationResetClearsStaleMatchCache)
{
    const QString workflow = readSourceFile(
        QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));
    ASSERT_FALSE(workflow.isEmpty());

    expectContainsAll(workflow, {
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
    EXPECT_LT(clear, prepare)
        << "重置当前对齐必须先清旧匹配缓存，再运行创建连接点。";
}

TEST(GuiAlgorithmAlignmentContractTest, AerialTriangulationGuiUsesSingleUnifiedWorkflow)
{
    const QString source = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString workflow = readSourceFile(
        QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));

    EXPECT_FALSE(header.contains(QStringLiteral("prepareAerialTriangulationTiePoints")));
    EXPECT_FALSE(header.contains(QStringLiteral("launchAerialTriangulationSfm")));
    EXPECT_FALSE(source.contains(QStringLiteral("MatchPhotosTask task(options)")));

    const QString run = sectionBetween(source,
                                       "void MenuWorkflowController::startAerialTriangulationWorkflow",
                                       "void MenuWorkflowController::openOverlapAnalysisDialog");
    expectContainsAll(run, {
        "workflowOptions.assetsDir",
        "workflowOptions.matchDir",
        "workflowOptions.matchingAlgorithmId",
        "workflowOptions.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages",
        "AerialTriangulationWorkflow::run",
    });
    EXPECT_EQ(countOccurrences(run, "AerialTriangulationWorkflow::run"), 1);
    expectContainsAll(workflow, {
        "TiePointPreparation::run",
        "clearTiePointCache",
    });
}

TEST(SfmPnpObservationContractTest, RegistrationUsesOneThreeDimensionalCandidatePerImageFeature)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration = sectionBetween(source,
                                                 "bool IncrementalSfm::registerImage",
                                                 "bool IncrementalSfm::findRegisteredSequenceNeighbor");

    expectContainsAll(registration, {
        "std::unordered_set<FeatureIdx> addedFeatures",
        "addedFeatures.count(myFeat)",
        "addedFeatures.insert(myFeat)",
    });
}

TEST(CudaSiftContractTest, TiePointThresholdCanReachDenseLowTextureRange)
{
    const QString source = readSourceFile(
        QStringLiteral("src/core/image_matching/sift/SiftFeatureExtractor.cpp"));

    EXPECT_TRUE(source.contains(QStringLiteral("0.1f")));
    EXPECT_TRUE(source.contains(QStringLiteral("20.0f")));
}

TEST(GuiAlgorithmAlignmentContractTest, MeshDecimationReachesReconstructionConfig)
{
    const QString workflow = readSourceFile(QStringLiteral("src/core/mesh/ModelWorkflowService.cpp"));

    expectContainsAll(workflow, {
        R"(settings.value(QStringLiteral("decimate")).toBool(false))",
        R"(settings.value(QStringLiteral("decimateRatio"))",
        "config.simplifyTargetFaces =",
        "std::lround(config.simplifyTargetFaces * decimateRatio)",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelAcceptsDepthMapsAsMetashapeStyleSource)
{
    const QString controller =
        readSourceFile(QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));

    const QString depthBlock =
        sectionBetween(controller,
                       "const QJsonArray depthResults",
                       "const QJsonArray modelResults");
    expectContainsAll(depthBlock, {
        R"(QStringLiteral("depth_maps"))",
        R"(QStringLiteral("深度图"))",
        "true,",
        "深度图将作为生成模型入口",
    });
    expectNotContainsAll(depthBlock, {
        "当前版本还不能直接从深度图生成模型",
    });

    const QString meshBlock =
        sectionBetween(manager,
                       "ProjectModelManager::startMeshReconstructionAsync",
                       "void ProjectModelManager::startTextureMappingAsync");
    expectContainsAll(meshBlock, {
        "resolveModelSourceForMeshing",
        "resolvedSource.sourcePointCloudPath",
        "resolvedSource.outputRoot",
        "xjw::mesh::workflow::ModelBuildRequest",
        "xjw::mesh::workflow::buildModel",
    });
    expectNotContainsAll(meshBlock, {
        "当前版本还不能直接从深度图生成模型",
    });

    expectContainsAll(dialog, {
        R"(_reuseDepthMapsCheck->setChecked(true))",
        R"(settings[QStringLiteral("depthMapSourcePath")] = sourcePath)",
        "使用严格的体积掩模",
    });
    expectNotContainsAll(dialog, {
        "使用严格的体积掩摸",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelDepthMapsUseDirectMeshWorkflow)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));
    const QString manager =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString workflow = readSourceFile(QStringLiteral("src/core/mesh/ModelWorkflowService.cpp"));

    expectContainsAll(dialog, {
        R"(settings[QStringLiteral("depthMapSourcePath")] = sourcePath)",
        R"(settings[QStringLiteral("reuseDepthMaps")] =)",
        R"(_hasReusableDepthMaps && selected_depth_batch_compatible)",
        R"(settings[QStringLiteral("force_depth_recompute")] =)",
    });

    expectContainsAll(manager, {
        "xjw::mesh::workflow::ModelBuildRequest request",
        "request.depthMapSourcePath",
        "xjw::mesh::workflow::buildModel(request)",
        "effectiveSettings.value(QStringLiteral(\"source_data\"))",
    });

    expectContainsAll(workflow, {
        "WorkflowResult buildModel",
        "WorkflowResult buildMeshFromDepthMaps",
        "DepthMapMeshBuilder",
        "request.depthMapSourcePath",
    });

    expectNotContainsAll(manager, {
        "深度图源需要先融合为密集点云，但未找到可复用的 dense_cloud.ply",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, ModelManagerUsesSharedModelWorkflowEntry)
{
    const QString manager =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString mesh_block =
        sectionBetween(manager,
                       "ProjectModelManager::startMeshReconstructionAsync",
                       "void ProjectModelManager::startTextureMappingAsync");

    expectContainsAll(mesh_block, {
        "xjw::mesh::workflow::ModelBuildRequest",
        "xjw::mesh::workflow::buildModel",
    });
    expectNotContainsAll(mesh_block, {
        "xjw::mesh::workflow::buildMeshFromDepthMaps",
        "xjw::mesh::workflow::buildMeshAndOptionalTexture",
        "reconstructionConfigFromModelSettings",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, ReconstructionStagesRouteToDedicatedManagers)
{
    const QString controller = readSourceFile(
        QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    const QString project_manager_header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString project_manager = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));

    expectContainsAll(controller, {
        "_projectManager->startCreatePointCloudAsync(settings)",
        "_projectManager->startGenerateModelAsync(settings)",
    });
    expectContainsAll(project_manager_header, {
        "void startCreatePointCloudAsync(const QJsonObject &settings)",
        "void startGenerateModelAsync(const QJsonObject &settings)",
    });

    const QString generate_block = sectionBetween(
        project_manager,
        "void ProjectManager::startGenerateModelAsync(const QJsonObject &settings)",
        "void ProjectManager::startMeshReconstructionAsync");
    expectContainsAll(generate_block, {
        "_pointCloudWorkflowController->startCreatePointCloudAsync(settings)",
        "_modelManager->startMeshReconstructionAsync(settings)",
        R"(settings.value(QStringLiteral("force_depth_recompute")))",
        R"(settings.value(QStringLiteral("reuseDepthMaps")))",
        "prepare_depth_maps",
    });
    expectNotContainsAll(project_manager, {
        "ProjectModelGenerationWorkflow",
        "ProjectReconstructionManager",
        "ProjectTaskDispatcher",
    });

    const QString mesh_block = sectionBetween(
        project_manager,
        "void ProjectManager::startMeshReconstructionAsync",
        "void ProjectManager::startTextureMappingAsync");
    expectContainsAll(mesh_block, {
        "_modelManager->startMeshReconstructionAsync(settings)",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelDialogOffersAutomaticDepthMaps)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));
    ASSERT_FALSE(dialog.isEmpty());

    expectContainsAll(dialog, {
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
    const QString header =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString source =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));

    expectContainsAll(header, {
        "_automaticModelDepthPreparationActive",
    });
    expectContainsAll(source, {
        "if (_automaticModelDepthPreparationActive)",
        R"(emit meshProgressChanged()",
        R"(emit pointCloudProgressChanged(stage, percent))",
        R"(_automaticModelDepthPreparationActive = true)",
        R"(_automaticModelDepthPreparationActive = false)",
    });
    expectNotContainsAll(source, {
        R"(this, &ProjectManager::pointCloudProgressChanged)",
        R"(this, &ProjectManager::pointCloudProgressFinished)",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelBlockControlsAreBoundToSettings)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));

    expectContainsAll(dialog, {
        "_splitRegionCheck",
        "_blockSizeSpin",
        R"(settings[QStringLiteral("splitIntoBlocks")] = _splitRegionCheck->isChecked())",
        R"(settings[QStringLiteral("blockSizeMeters")] = _blockSizeSpin->value())",
        "updateBlockControlsAvailability",
    });
}

TEST(MvsSchedulerContractTest, UsesDedicatedVisibilityBuilderAndBudget)
{
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString visibilityHeader = readSourceFile(
        QStringLiteral("src/core/mvs/MvsVisibilityGraphBuilder.h"));
    const QString visibilityBuilder = readSourceFile(
        QStringLiteral("src/core/mvs/MvsVisibilityGraphBuilder.cpp"));
    const QString memoryPolicy = readSourceFile(
        QStringLiteral("src/core/mvs/DepthMemoryPolicy.cpp"));

    expectContainsAll(header, {
        "FrameMvsCache",
        "prepareFrameCaches",
        "_visibilityBits",
        "sourceSharedPointIndices",
    });
    expectContainsAll(scheduler, {
        "prepareFrameCaches();",
        "MvsVisibilityGraphBuilder::build",
        "estimateMvsVisibilityGraphMemory",
        "plannedVisibilityMemory",
        "sourceViewIndicesForFrame",
        "visibleSparsePointIndicesForFrame",
        "sourceSharedPointIndices.reserve",
        "sourceSharedPointIndices.push_back",
        "sourceIndicesMatchCachedPrefix",
    });
    expectNotContainsAll(scheduler, {"selectMvsSourceViewIndices(_views, _sparse, refIdx, numSrc)"});

    expectContainsAll(visibilityHeader, {
        "MvsVisibilityGraphBuildOptions",
        "requiredPairs",
        "geometryPreferredPeersByView",
        "cancelFlag",
        "cooperativeCheckpointHook",
        "buildGeometryPeerShortlist",
    });
    expectContainsAll(visibilityBuilder, {
        "buildCompletePairs = viewCount <= fullPairLimit",
        "cancellationRequested(options)",
        "(pointIndex & 255U)",
        "(viewIndex & 255U)",
        "(word & 255U)",
        "std::popcount",
        "buildGeometryPeerShortlist",
    });
    expectNotContainsAll(visibilityBuilder, {
        "visibleViews.size() <= fullPairLimit",
    });
    expectContainsAll(memoryPolicy, {
        "estimateMvsVisibilityGraphMemory",
        "visibilityEstimate.totalBytes",
        "estimate.visibility = visibilityEstimate",
    });

    const QString visibleBlock =
        sectionBetween(scheduler, "std::vector<size_t> DepthMapGenerator::visibleSparsePointIndicesForFrame", "// =============================================================================");
    expectContainsAll(visibleBlock, {"return cache.sourceSharedPointIndices;"});
    EXPECT_LT(indexOfOrFail(visibleBlock, "return cache.sourceSharedPointIndices;"),
              indexOfOrFail(visibleBlock, "std::vector<size_t> filtered;"));

    const QString frameCacheBlock =
        sectionBetween(scheduler, "void DepthMapGenerator::prepareFrameCaches()", "std::vector<int> DepthMapGenerator::sourceViewIndicesForFrame");
    expectContainsAll(frameCacheBlock, {
        "visibilityOptions.cancelFlag = &_cancelled",
        "MvsVisibilityGraphBuilder::build",
        "visibilityGraph.cancelled",
        "visibilityGraph.neighborsByView",
        "rankedSourceCandidates",
        "desiredSourceCount",
    });
}

TEST(MvsSchedulerContractTest, LargeHybridBatchKeepsBoundedOpenClFullFrame)
{
    const QString scheduler =
        readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    expectContainsAll(scheduler, {
        "recommendedOpenClFullFrameFloorPerDevice(",
        "schedulingPolicy.guaranteedOpenClFullFramesPerDevice",
        "schedulingPolicy.maximumOpenClInFlightTasksPerDevice",
        "!claim.requiresFullFrame",
        "OpenCL完整帧 %1/%2",
    });
}

TEST(MvsDepthArtifactContractTest, OptionalTargetedRecoveryMaskIsCheckedBeforeRead)
{
    const QString source =
        readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString targetedRecoveryLoad = sectionBetween(
        source,
        "const QString targeted_recovered_path = storage_dir.filePath(",
        "const QString provenance_path = storage_dir.filePath(");

    expectContainsAll(targetedRecoveryLoad, {
        "QFileInfo::exists(targeted_recovered_path)",
        "xjw::common::io::readImage(",
    });
    EXPECT_LT(indexOfOrFail(targetedRecoveryLoad,
                            "QFileInfo::exists(targeted_recovered_path)"),
              indexOfOrFail(targetedRecoveryLoad,
                            "xjw::common::io::readImage("));
}

TEST(MvsSchedulerContractTest, SparseHintsUseProjectedSamplesAndPrescaledPatchMatchInputs)
{
    const QString cameraHeader = readSourceFile(QStringLiteral("src/core/camera/Camera.h"));
    const QString cameraSource = readSourceFile(QStringLiteral("src/core/camera/Camera.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString pyramid = readSourceFile(QStringLiteral("src/core/mvs/DepthPyramidEstimator.cpp"));
    const QString cuda = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCUDA.cu"));
    const QString cpu = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCPU.cpp"));

    expectContainsAll(scheduler, {
        "estimateDepthRangeFromVisiblePoints",
        "buildHintDepthFromVisiblePoints",
        "buildSparseSupportMaskFromVisiblePoints",
        "const std::vector<size_t> visibleSparsePointIndices",
        "visibleSparsePointIndices)",
        "patchMatchWorkSize",
        "collectProjectedSparseDepthSamples",
        "buildHintDepthFromProjectedSamples",
        "makeDepthPyramidConfig",
        "pyramid_sparse_hints",
        "const cv::Size hint_size = patchMatchWorkSize",
        "hint_size.width",
        "hint_size.height",
        "workRefSparseSamples",
        "buildHintDepthFromProjectedSamples(refIdx",
        "buildSparseSupportMaskFromProjectedSamples(refIdx",
    });
    expectContainsAll(header, {
        "ProjectedSparseDepthSample",
        "buildSparseSeedDepthFromProjectedSamples",
    });
    expectNotContainsAll(scheduler, {
        "cv::Mat hintDepth = buildHintDepthFromVisiblePoints(refIdx, W, H, visibleSparsePointIndices);",
        "buildHintDepthForCamera(refIdx,\n                                                 coarseHintCam",
        "buildHintDepthForCamera(refIdx,\n                                                         fineHintCam",
    });

    const QString supportBlock =
        sectionBetween(scheduler, "const std::vector<ProjectedSparseDepthSample> workRefSparseSamples", "timing.hintMs = elapsedMs");
    expectContainsAll(supportBlock, {
        "support_mask_config",
        "pyramid_config.levels[pyramid_config.activeLevelCount - 1].patchMatch",
        "const cv::Size supportMaskSize = patchMatchWorkSize(refImg, support_mask_config);",
    });
    expectNotContainsAll(supportBlock, {"patchMatchWorkSize(refImg, pmCfg)"});

    const QString projectedBlock =
        sectionBetween(scheduler,
                       "std::vector<ProjectedSparseDepthSample> DepthMapGenerator::collectProjectedSparseDepthSamples",
                       "cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples");
    expectContainsAll(scheduler, {"kMaxProjectedDepthQuantileSamples"});
    expectContainsAll(projectedBlock, {
        "depthQuantileSamples",
        "std::nth_element",
        "projectedCandidates",
        "camera.projectWorldPointWithDepth",
        "candidate.depth",
    });
    expectNotContainsAll(projectedBlock, {
        "std::sort(allZc",
        "allZc.reserve(visiblePointIndices.size())",
        "float Zc = cam.R_cw[6]*pt[0]",
        "camera.projectWorldPoint(world",
    });
    EXPECT_EQ(countOccurrences(projectedBlock, "for (size_t pointIndex : visiblePointIndices)"), 1);

    expectContainsAll(cameraHeader, {"projectWorldPointWithDepth"});
    expectContainsAll(cameraSource, {
        "Camera::projectWorldPointWithDepth",
        "worldToCameraFromCameraToWorldPose(world, camera_point)",
        "applyTsaiDistortion(",
    });
    expectNotContainsAll(cameraSource, {"return projectWorldPoint(world, pixel)"});

    expectContainsAll(pyramid, {
        "prior = propagateDepthPrior(",
        "request.referenceImage.size())",
        "mergeSparseHint(prior, request.sparseDepthHints[index], target_size)",
        "hint.copyTo(prior.center, sparse_mask)",
    });

    const QString hintBody =
        sectionBetween(scheduler, "cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples", "cv::Mat DepthMapGenerator::buildSparseSeedDepthFromProjectedSamples");
    expectContainsAll(hintBody, {
        "cv::distanceTransform",
        "DIST_LABEL_PIXEL",
        "maxHintRadius",
    });
    expectNotContainsAll(hintBody, {
        "cv::Mat distMap",
        "INT_MAX / 2",
    });

    expectContainsAll(cuda, {
        "hintDepth->cols == sW && hintDepth->rows == sH",
        "hintScaled = *hintDepth",
    });
    expectContainsAll(cpu, {
        "hintDepth->cols == W && hintDepth->rows == H",
    });
    const QString gpuBody = sectionBetween(cuda,
                                           "bool PatchMatchDepthEstimator::estimateGPU",
                                           "bool PatchMatchDepthEstimator::isCudaAvailable");
    expectContainsAll(gpuBody, {
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
    const QString implementation = policy + cuda + cpu;

    expectContainsAll(implementation, {
        "robustMultiSourceNcc",
        "requiredPhotometricSupport",
        "cpuEvalHypCost",
        "evalHypCost",
    });
    EXPECT_GE(countOccurrences(implementation, "robustMultiSourceNcc("), 3);
    expectNotContainsAll(cuda, {
        "if (ncc > 0.05f) { sumScore += ncc; ++goodSrc; }",
        "sumScore / static_cast<float>(goodSrc)",
        "sumScore / goodSrc",
    });
}

TEST(MvsSchedulerContractTest, FinalPyramidLevelKeepsConfiguredIterationBudget)
{
    const QString policy = readSourceFile(QStringLiteral("src/core/mvs/DepthPyramidPolicy.cpp"));

    expectContainsAll(policy, {
        "result.patchMatch = base_config",
        "result.minSupportViews = 3",
        "result.radiusScale = 1.0f",
    });
    expectNotContainsAll(policy, {
        "base_config.numIterations - 1",
        "hintCoverage",
    });
}

TEST(MvsHeterogeneousSchedulingContractTest,
     AutoCombinesDistinctCudaAndOpenClButKeepsExplicitRequestsStrict)
{
    const QString generator =
        readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString scheduler_header =
        readSourceFile(QStringLiteral("src/core/mvs/DepthComputeScheduler.h"));
    const QString scheduler_source =
        readSourceFile(QStringLiteral("src/core/mvs/DepthComputeScheduler.cpp"));
    const QString opencl_source =
        readSourceFile(QStringLiteral("src/core/mvs/PatchMatchOpenCL.cpp"));
    const QString gui_main =
        readSourceFile(QStringLiteral("src/gui/main.cpp"));

    expectContainsAll(generator, {
        "const bool probeOpenCl = configuredBackend == PatchMatchBackend::OpenCl ||",
        "configuredBackend == PatchMatchBackend::OpenCl &&",
        "isNvidiaOpenClVendor(device.vendor)",
        "selectedPhysicalDeviceIdentities.contains(descriptor.physicalIdentity)",
        "shouldSkipUnstableOpenClCudaAlias",
        "const bool heterogeneousAuto = configuredBackend == PatchMatchBackend::Auto",
        "_config.patchMatch.backend = heterogeneousAuto",
        "_config.patchMatch.cudaFallbackToCpu = false",
        "_config.patchMatch.openClFallbackToCpu = false",
    });
    expectNotContainsAll(generator, {
        "(automaticAcceleration && !cudaAvailable)",
        "Auto resolves to one backend before this pool is built",
    });
    EXPECT_GE(countOccurrences(
                  generator,
                  "selectedPhysicalDeviceIdentities.insert(descriptor.physicalIdentity)"),
              2);

    const QString scheduler_wiring = sectionBetween(
        generator,
        "const std::vector<DepthComputeWorker> acceleratorWorkers",
        "const int gpuFrameWorkers");
    expectContainsAll(scheduler_wiring, {
        "const bool benefitAwareScheduling = heterogeneousAuto",
        "DepthComputeScheduler computeScheduler",
        "acceleratorWorkers,\n        schedulingPolicy",
    });

    expectContainsAll(scheduler_header + scheduler_source, {
        "enableBenefitAwareScheduling",
        "emaElapsedMilliseconds",
        "DepthTaskClaimStatus",
        "DepthTaskClaim",
        "claimNext",
        "waitForStateChange",
        "shouldPauseAtQueueTail",
        "DepthTaskCompletionResult",
        "retryScheduled",
        "findPendingCrossBackendRetry",
    });
    expectContainsAll(opencl_source, {
        "cudaPhysicalIdentityForOpenClName",
        "normalizedGpuDeviceName",
        "fallbackGpuPhysicalIdentity",
    });
    expectContainsAll(gui_main, {
        "configureOpenClDevicePolicy",
        "PLAMATRIX_OPENCL_DEVICE_INDEX",
        "isNvidiaOpenClVendor(device.vendor)",
    });
}

TEST(MvsHeterogeneousSchedulingContractTest,
     StreamingHybridUsesByteBoundedParallelArtifactSaving)
{
    const QString generator =
        readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    expectContainsAll(generator, {
        "depthFrameResultResidentBytes",
        "_maxResidentTasks",
        "_maxResidentBytes",
        "ProducerReservation",
        "reserveProducer(",
        "_producerReservations",
        "_peakResidentTasks",
        "_maxEnqueueWait",
        "const size_t saveWorkerCount = !retainDepthFrames",
        "physicalGpuCount >= 2",
        "深度产物保存队列统计",
    });

    expectContainsAll(generator, {
        "activeFrameWorkerCount",
        "minimumCpuThreadsPerWorker",
        "cpuThreadRemainder",
        "assigned_cpu_threads",
        "cpu_thread_budget=%9",
        "workerConfig.cpuWorkerCount = std::max(1, assignedCpuThreadCount)",
        "omp_set_num_threads(std::max(1, assignedCpuThreadCount))",
        "fusionCfg.workerCount    = resolvedTotalCpuThreadBudget(_config)",
    });

    const QString capacity_policy = sectionBetween(
        generator,
        "uint64_t adaptiveSaveQueueResidentByteCapacity",
        "int preloadImagesWorkerCount");
    expectContainsAll(capacity_policy, {
        "estimatedSaveQueueProducerBytes(largestFrameBytes)",
        "if (snapshot.valid)",
        "byteCapacity = std::min(byteCapacity, memoryBudget)",
    });
    expectNotContainsAll(capacity_policy, {
        "if (memoryBudget > 0)",
    });

    const QString save_queue = sectionBetween(
        generator,
        "class DepthFrameArtifactSaveQueue",
        "bool writeFastDepthMatStorage");
    expectContainsAll(save_queue, {
        "canAcceptLocked(_producerReservationBytes)",
        "++_producerReservations",
        "++_residentTasks",
        "_residentBytes += _producerReservationBytes",
        "return taskResidentBytes <= _maxResidentBytes - _residentBytes;",
        "releaseProducerReservationLocked",
        "resident_bytes > reserved_bytes",
        "catch (const std::exception &exception)",
        "saved = _saveFn(task.frameIndex, task.result, task.stageLabel)",
        "save_exception_occurred",
        "waitUntilIdle(const std::atomic<bool> *cancelFlag",
    });
    const int queue_insert = save_queue.indexOf(
        QStringLiteral("_tasks.push_back(std::move(task))"));
    const int reservation_transfer = save_queue.indexOf(
        QStringLiteral("reservation.disarm()"), queue_insert);
    ASSERT_GE(queue_insert, 0);
    ASSERT_GE(reservation_transfer, 0);
    EXPECT_LT(queue_insert, reservation_transfer)
        << "Queue insertion must commit before producer reservation accounting transfers.";
    const int save_call = save_queue.indexOf(QStringLiteral("saved = _saveFn("));
    const int active_release = save_queue.indexOf(
        QStringLiteral("--_activeTasks"), save_call);
    const int exception_log = save_queue.indexOf(
        QStringLiteral("if (save_exception_occurred)"), active_release);
    ASSERT_GE(save_call, 0);
    ASSERT_GE(active_release, 0);
    ASSERT_GE(exception_log, 0);
    EXPECT_LT(active_release, exception_log)
        << "Saver accounting must complete before exception diagnostics are formatted.";

    const QString worker = sectionBetween(
        generator,
        "auto workerFunc =",
        "std::vector<std::thread> workers");
    const int reservation = worker.indexOf(QStringLiteral(
        "saveQueue.reserveProducer(&_cancelled)"));
    const int compute = worker.indexOf(QStringLiteral(
        "DepthFrameResult res = computeDepthForView("));
    const int enqueue = worker.indexOf(QStringLiteral(
        "std::move(saveReservation), i, res, QStringLiteral(\"初始\")"));
    ASSERT_GE(reservation, 0);
    ASSERT_GE(compute, 0);
    ASSERT_GE(enqueue, 0);
    EXPECT_LT(reservation, compute)
        << "A complete producer result must be reserved before depth computation allocates it.";
    EXPECT_LT(compute, enqueue);

    expectContainsAll(worker, {
        "catch (const std::exception &exception)",
        "computeScheduler.complete(",
        "saveQueue.cancel()",
        "workerExceptionReported.exchange(true)",
    });
    expectContainsAll(generator, {
        "!saveQueue.waitUntilIdle(&_cancelled)",
        "allOk = !anyFailure.load() && !_cancelled.load()",
        "fusionCfg.cancelFlag = std::shared_ptr<std::atomic_bool>",
        "[MVS][深度融合] 收到取消请求",
    });
}

TEST(MeshReconstructionContractTest, ClosedSurfaceNormalsUseNeighborhoodConsistency)
{
    const QString source = readSourceFile(QStringLiteral("src/core/mesh/SurfaceReconstructor.cpp"));
    const QString orientation = sectionBetween(source,
                                               "void orientNormalsOutwardFromCentroid",
                                               "PlaPointCloud pointXYZRGBToCloud");

    expectContainsAll(orientation, {
        "plapoint::search::KdTree",
        "nearestKSearch",
        "#pragma omp parallel for",
        "component_outward_score",
    });
    expectNotContainsAll(orientation, {
        "if (outward_dot < 0.0)",
    });
}

TEST(GuiDialogLayoutContractTest, DialogSourcesAreGroupedByDomain)
{
    const QString gui_sources =
        readSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    const QString dialog_sources =
        readSourceFile(QStringLiteral("src/gui/cmake/GuiDialogSources.cmake"));
    const QString layout_readme =
        readSourceFile(QStringLiteral("src/gui/dialogs/README.md"));
    const QString workspace_ui =
        readSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.ui"));

    expectContainsAll(gui_sources, {
        "include(${CMAKE_CURRENT_LIST_DIR}/GuiDialogSources.cmake)",
        "views/CameraSceneWidget.cpp",
        "views/CameraSceneViewMath.cpp",
        "views/ObjRenderPreparation.cpp",
    });
    expectNotContainsAll(gui_sources, {
        "dialogs/application/AboutDialog.cpp",
        "dialogs/camera/CameraModel3DDialog.cpp",
        "dialogs/reconstruction/GenerateModelDialog.cpp",
        "dialogs/tie_points/MatchViewerDialog.cpp",
    });
    expectContainsAll(dialog_sources, {
        "GUI_APPLICATION_DIALOG_SOURCES",
        "GUI_CAMERA_DIALOG_SOURCES",
        "GUI_IMAGE_DIALOG_SOURCES",
        "GUI_RECONSTRUCTION_DIALOG_SOURCES",
        "GUI_TIE_POINT_DIALOG_SOURCES",
        "GUI_SHARED_DIALOG_SOURCES",
        "dialogs/application/AboutDialog.cpp",
        "dialogs/camera/CameraModel3DDialog.cpp",
        "dialogs/image/GenerateMaskDialog.cpp",
        "dialogs/reconstruction/CreatePointCloudDialog.cpp",
        "dialogs/reconstruction/GenerateModelDialog.cpp",
        "dialogs/tie_points/MatchViewerDialog.cpp",
        "dialogs/shared/WorkflowParameterDialogStyle.cpp",
    });
    expectContainsAll(layout_readme, {
        "`application/`",
        "`camera/`",
        "`image/`",
        "`reconstruction/`",
        "`tie_points/`",
        "`shared/`",
    });
    EXPECT_TRUE(workspace_ui.contains(
        QStringLiteral("<header>CameraSceneWidget.h</header>")));
}

TEST(GuiArchitectureContractTest, PointCloudWorkflowControllerOnlyCoordinatesCoreMvs)
{
    EXPECT_TRUE(sourceFileExists(
        QStringLiteral("src/gui/project/manager/ProjectPointCloudWorkflowController.h")));
    EXPECT_TRUE(sourceFileExists(
        QStringLiteral("src/gui/project/manager/ProjectPointCloudWorkflowController.cpp")));
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")));
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.h")));
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.cpp")));

    const QString guiSources = readSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("ProjectPointCloudWorkflowController")));
    const QString pointCloudController = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectPointCloudWorkflowController.cpp"));
    expectContainsAll(pointCloudController, {
        "xjw::mvs::DepthMapGenerator",
        "xjw::mvs::fuseDepthMapsStreaming",
        "xjw::gui::tasks::runGuardedWithOutcome",
    });
    expectNotContainsAll(pointCloudController, {
        "PatchMatchCPU",
        "PatchMatchCuda",
    });
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp")));
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/manager/ProjectTaskDispatcher.cpp")));
}

TEST(GuiArchitectureContractTest, AsyncTasksExposeSharedCancellationVocabulary)
{
    const QString runner = readSourceFile(QStringLiteral("src/gui/tasks/GuiTaskRunner.h"));
    const QString maskController = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectMaskWorkflowController.cpp"));

    expectContainsAll(runner, {
        "class TaskCancellationToken",
        "class TaskCancellationSource",
        "isCancellationRequested",
        "requestCancellation",
    });
    expectContainsAll(maskController, {
        "_cancellation.reset()",
        "cancellation.isCancellationRequested()",
        "_cancellation.requestCancellation()",
    });
}

TEST(GuiArchitectureContractTest, ProjectPersistenceAndWorkflowAlgorithmsLiveOutsideGui)
{
    EXPECT_TRUE(sourceFileExists(
        QStringLiteral("src/common/project/ProjectSessionModel.cpp")));
    EXPECT_TRUE(sourceFileExists(
        QStringLiteral("src/common/project/ProjectDocumentModel.cpp")));
    EXPECT_TRUE(sourceFileExists(
        QStringLiteral("src/core/project_workflows/ProjectWorkflowOperations.cpp")));

    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/data/ProjectData.cpp")));
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/data/ProjectFilesManager.cpp")));
    EXPECT_FALSE(sourceFileExists(
        QStringLiteral("src/gui/project/support/ProjectWorkflowUtils.cpp")));

    const QString guiSources = readSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    EXPECT_FALSE(guiSources.contains(QStringLiteral("project/data/ProjectData.cpp")));
    EXPECT_FALSE(guiSources.contains(QStringLiteral("project/support/ProjectWorkflowUtils.cpp")));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("main_window/MainWindowProjectLifecycle.cpp")));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("widgets/DataTreePopulation.cpp")));
}

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

TEST(CommonPathIoContractTest, FallsBackToGdalWhenOpenCvCannotDecodeImage)
{
    const QString source = readSourceFile(QStringLiteral("src/common/io/PathIO.cpp"));
    const QString cmake = readSourceFile(QStringLiteral("src/common/CMakeLists.txt"));

    expectContainsAll(source, {
        "#include <gdal_priv.h>",
        "cv::Mat readImageWithGdal(const QString &path, int flags)",
        "if (!decoded.empty())",
        "return readImageWithGdal(path, flags)",
    });

    expectContainsAll(cmake, {
        "GDAL_INCLUDE_DIRS",
        "${PLASCAN_GDAL_TARGET}",
    });
}

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
        "SfmBundleAdjustCoordinator(*this).iterative()",
    });
    EXPECT_FALSE(sfmSource.contains(QStringLiteral("tryRegisterInterpolatedSequenceImages")));
    EXPECT_FALSE(sfmSource.contains(QStringLiteral("Sequence interpolation registered")));
}

TEST(SfmSourceContractTest, SequencePnpRelaxationRequiresRegisteredCamerasOnBothSides)
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
        "previousSteps == 1 && nextSteps == 1",
        "pnpOptions.allowRelaxedInlierRatio = true",
        "bracketedSequencePnpMinInlierRatio",
        "bracketedSequencePnpMinInliers",
    });
}

TEST(SfmSourceContractTest, MatchGeometryFilteringUsesSeededSerialUsac)
{
    const QString source = readSourceFile(QStringLiteral(
        "src/core/feature_match/MatchGeometryFilter.cpp"));

    expectContainsAll(source, {
        "usacParams.randomGeneratorState = config.randomSeed",
        "usacParams.isParallel = false",
    });
}

TEST(SfmSourceContractTest, FinalGlobalBaRetriesUnregisteredImagesBeforePublishingResult)
{
    const QString source = readIncrementalSfmImplementation();
    const QString registration = sectionBetween(
        source,
        "IncrementalSfmResult IncrementalSfm::runRegistrationFromCurrentInitialization",
        "void IncrementalSfm::resetForInitialPairTrial");

    const int finalBa = indexOfOrFail(registration, "SfmBundleAdjustCoordinator(*this).iterative();");
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
        "pnpOptions.useInitialPosePrefilter = true",
        "initialPosePrefilterMaxReprojError",
    });
    expectContainsAll(pnp, {
        "cv::projectPoints",
        "guidedOriginalIndices",
        "initialPosePrefilterMinCandidates",
    });
}

TEST(SfmSourceContractTest, CudaSiftMatchingIsDedicatedModule)
{
    const QString header = readSourceFile(QStringLiteral("src/core/feature_match/tradition/CudaSiftMatcher.h"));
    const QString cmake = readSourceFile(QStringLiteral("src/core/feature_match/tradition/CMakeLists.txt"));
    const QString matcher = readSourceFile(QStringLiteral("src/core/feature_match/tradition/TraditionalFeatureMatcher.cpp"));

    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/feature_match/tradition/CudaSiftMatcher.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/feature_match/tradition/CudaSiftMatcher.cpp")));
    expectContainsAll(header, {"class CudaSiftMatcher"});
    expectContainsAll(cmake, {"CudaSiftMatcher.cpp"});
    expectContainsAll(matcher, {R"(#include "CudaSiftMatcher.h")"});
    expectNotContainsAll(matcher, {"knnMatchL2Cuda"});
}

TEST(PythonRuntimeSourceContractTest, RuntimeLaunchersPreferRepoLocalVenv)
{
    const QStringList runtimeSources{
        QStringLiteral("src/gui/project/manager/ProjectManager.cpp"),
        QStringLiteral("src/core/feature_match/MatcherFactory.cpp"),
        QStringLiteral("src/cli/features/cli_feature_extract.cpp"),
    };

    for (const QString &relativePath : runtimeSources)
    {
        const QString source = readSourceFile(relativePath);
        ASSERT_FALSE(source.isEmpty()) << qPrintable(relativePath);
        expectContainsAll(source, {
            "PLASCAN_PYTHON_EXECUTABLE",
            "PLASCAN_PYTHON",
            ".venv/Scripts/python.exe",
            ".venv/bin/python",
        });
    }
}

TEST(SfmSourceContractTest, LightGlueSiftCarriesScaleAndOrientation)
{
    const QString exportScript = readSourceFile(QStringLiteral("scripts/models/export_lightglue_torchscript.py"));
    const QString matcher = readSourceFile(QStringLiteral("src/core/feature_match/lightglue/LightGlueMatcher.cpp"));

    expectContainsAll(exportScript, {
        "self.model.conf.add_scale_ori",
        "xy0 = normalize_keypoints(kpts0[..., :2], image_size0).clone()",
        "torch.cat([xy0, scales0, oris0], dim=-1)",
        "torch.cat([xy1, scales1, oris1], dim=-1)",
        R"(feature == "sift")",
        "kpt_dim = 4",
    });
    expectContainsAll(matcher, {
        R"(fd.sourceAlgorithm == "sift")",
        "keypoint.size",
        "keypoint.angle",
        "CV_PI",
    });
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
        "MatchResultCatalog",
        "group.variants",
        "variant.matchFilePath",
        "variant.sidecarPath",
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
        "workflowOptions.featureDir",
        "workflowOptions.matchDir",
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
        QStringLiteral("src/core/feature_extractors/tradition/CudaSiftFeatureExtractor.cpp"));

    EXPECT_TRUE(source.contains(QStringLiteral("0.1f")));
    EXPECT_FALSE(source.contains(QStringLiteral("0.5f, 20.0f")));
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
        R"(settings[QStringLiteral("reuseDepthMaps")] = _reuseDepthMapsCheck->isChecked())",
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

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelUsesMetashapeStyleWorkflowOrchestrator)
{
    const QString workflow = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.cpp"));
    const QString reconstruction_manager = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp"));
    const QString gui_sources = readSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));

    ASSERT_FALSE(workflow.isEmpty());
    expectContainsAll(workflow, {
        "decideModelGenerationWorkflow",
        "startEstimateDepthMapsAsync",
        "depthMapBatchReady",
        "startMeshReconstructionAsync",
        "_expectedDepthOutputDir",
        R"(QStringLiteral("depth_maps"))",
    });

    expectContainsAll(reconstruction_manager, {
        "ProjectModelGenerationWorkflow",
        "_modelWorkflow->start(settings)",
    });
    expectContainsAll(gui_sources, {
        "project/manager/ProjectModelGenerationWorkflow.cpp",
        "project/manager/ProjectModelGenerationWorkflow.h",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelWorkflowHandlesSynchronousStartRejection)
{
    const QString workflow = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.cpp"));
    const QString dense_header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.h"));
    const QString model_header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelManager.h"));

    expectContainsAll(dense_header, {
        "bool startFuseDepthMapsAsync",
        "bool startGenerateDenseCloudAsync",
        "bool isMvsRunning() const",
    });
    expectContainsAll(model_header, {
        "bool startMeshReconstructionAsync",
    });
    expectContainsAll(workflow, {
        "if (_denseManager->startEstimateDepthMapsAsync(depth_settings_with_runtime))",
        "else",
        "if (!_modelManager->startMeshReconstructionAsync(settings))",
        "finish(false)",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, DensePipelinePropagatesFusionStartFailure)
{
    const QString manager = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    expectContainsAll(manager, {
        "if (!self->startFuseDepthMapsAsync(settings))",
        "emit self->mvsProgressFinished(false);",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, DenseAndModelTasksGuardProjectAndRejectOverlap)
{
    const QString dense = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString model = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString model_header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelManager.h"));
    const QString reconstruction = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp"));

    expectContainsAll(dense, {
        "if (isMvsRunning())",
        "owner->currentProjectPath() != projectPath",
        "source_depth_map_dir",
        "batch_frame_count",
    });
    expectContainsAll(model, {
        "if (_isRunning)",
        "_isRunning = true",
        "self->_isRunning = false",
    });
    expectContainsAll(model_header, {
        "bool isRunning() const",
    });
    expectContainsAll(reconstruction, {
        "_modelManager->isRunning()",
        "_modelWorkflow->isRunning()",
        "_denseManager->isMvsRunning()",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, DensePipelineGuardsQueuedFusionAndProjectSwitchCompletion)
{
    const QString header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.h"));
    const QString manager = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString model = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));

    expectContainsAll(header, {
        "bool _mvsTransitionPending = false;",
    });
    expectContainsAll(manager, {
        "_mvsTransitionPending = true",
        "if (!self || !self->_mvsTransitionPending)",
        "_mvsTransitionPending = false",
        "return !_activeMvsGenerator.isNull() || static_cast<bool>(_activeMvsCancelFlag) ||",
        "emit self->mvsProgressFinished(false);",
    });
    EXPECT_GE(countOccurrences(model, "emit self->meshProgressFinished(false);"), 2);
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelKeepsIndependentMeshToolSeparate)
{
    const QString controller = readSourceFile(
        QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    const QString project_manager_header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString reconstruction_manager = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp"));

    expectContainsAll(controller, {
        "_projectManager->startGenerateModelAsync(settings)",
    });
    expectContainsAll(project_manager_header, {
        "void startGenerateModelAsync(const QJsonObject &settings)",
    });

    const QString generate_block = sectionBetween(
        reconstruction_manager,
        "case Task::GenerateModel:",
        "case Task::MeshReconstruction:");
    expectContainsAll(generate_block, {
        "_modelWorkflow->start(settings)",
    });

    const QString mesh_block = sectionBetween(
        reconstruction_manager,
        "case Task::MeshReconstruction:",
        "case Task::TextureMapping:");
    expectContainsAll(mesh_block, {
        "_modelManager->startMeshReconstructionAsync(settings)",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelContinuationChecksExpectedDenseOutput)
{
    const QString workflow = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.cpp"));
    const QString workflow_header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelGenerationWorkflow.h"));

    expectContainsAll(workflow_header, {
        "QString _expectedDepthOutputDir;",
        "bool isPreparingDenseCloud() const;",
    });
    expectContainsAll(workflow, {
        "_expectedDepthOutputDir",
        "QDir::cleanPath(output_directory)",
        "result_directory.compare(_expectedDepthOutputDir",
        "depthMapBatchReady",
        "Qt::CaseInsensitive",
        "_denseManager->isMvsRunning()",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, GenerateModelDialogExplainsAutomaticDepthMapGeneration)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/GenerateModelDialog.cpp"));
    ASSERT_FALSE(dialog.isEmpty());

    expectContainsAll(dialog, {
        "缺少深度图时将自动估计深度图",
        "重用深度图",
        R"(settings[QStringLiteral("reuseDepthMaps")])",
        R"(settings[QStringLiteral("depthMapSourcePath")] = sourcePath)",
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

TEST(MvsSchedulerContractTest, FrameWorkerControlsAndSchedulerBasics)
{
    const QString types = readSourceFile(QStringLiteral("src/core/mvs/MvsTypes.h"));
    const QString configH = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.h"));
    const QString configCpp = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.cpp"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));

    expectContainsAll(types, {
        "gpuFrameWorkerCount",
        "cpuFrameWorkerCount",
        "cudaUseParallelSweep",
    });
    expectContainsAll(configH, {
        "gpuFrameWorkers",
        "cpuFrameWorkers",
    });
    expectContainsAll(configCpp, {
        "gpu_frame_workers",
        "cpu_frame_workers",
        "config.gpuFrameWorkerCount",
        "config.cpuFrameWorkerCount",
        "autoGpuFrameWorkers",
        "settings.gpuFrameWorkers",
        "return std::clamp",
    });
    const QString autoGpuWorkers = sectionBetween(configCpp,
                                                  "int autoGpuFrameWorkers",
                                                  "int autoCpuFrameWorkers");
    expectNotContainsAll(autoGpuWorkers, {
        "return 1;",
        "Q_UNUSED(threads)",
    });
    expectContainsAll(scheduler, {
        "gpuFrameWorkers",
        "workerIndex < gpuFrameWorkers",
        "cpuFrameWorkers",
        "DepthFrameArtifactSaveQueue",
        R"(saveQueue.enqueue(i, res, QStringLiteral("初始")))",
        "saveQueue.waitUntilIdle()",
        "saveQueue.stop()",
        "preloadImagesWorkerCount",
        "std::atomic<int> nextImage",
        "preloadWorkers.emplace_back",
        "preloadImages(): workers=",
    });
    expectNotContainsAll(scheduler, {
        "const int cpuWorkers = cudaAvailable ? 0 : 1;",
        R"(if (!saveDepthFrameArtifacts(i, res, QStringLiteral("初始"))))",
    });
    expectContainsAll(header, {"void releasePixelStorage()"});
}

TEST(MvsSchedulerContractTest, DepthCacheAndArtifactLifecycleContracts)
{
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString helper = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    expectContainsAll(scheduler, {
        "shouldRetainAllDepthFramesInMemory",
        "std::atomic<bool> keepDepthFramesInMemory",
        "DepthFrameResult storedResult = res;",
        "if (!keepDepthFramesInMemory.load())",
        "storedResult.releasePixelStorage();",
        "_depthFrames[i] = storedResult;",
        "if (keepDepthFramesInMemory.load() && NV >= 2)",
        "SystemMemorySnapshot",
        "querySystemMemorySnapshot",
        "estimateDepthFrameCacheBytes",
        "retainedDepthMemoryBudgetBytes",
        "_config.maxDepthCacheRamFraction",
        "_config.minFreeRamBytes",
        "深度图内存策略",
        "refreshViewImageDimensionsFromCache",
        "refreshViewImageDimensionsFromCache();",
        "无有效影像尺寸，采用保守流式模式",
        "memoryPressureRequiresStreaming",
        "keepDepthFramesInMemory.compare_exchange_strong",
        "releaseStoredDepthFramePixelStorage",
        "内存压力升高，切换为流式保存",
        "无法继续本次内存融合",
    });
    expectNotContainsAll(scheduler, {
        "_depthFrames[i] = res;",
        R"(QStringLiteral("无有效影像尺寸");
        }
        return true;)",
    });

    expectContainsAll(manager, {"applyImageSizeToMvsView"});
    EXPECT_GE(countOccurrences(manager, "applyImageSizeToMvsView(imgPath, &view)"), 2);
    expectNotContainsAll(manager, {
        "cv::imread(imagePath.toStdString()",
        "view.imageWidth = 0;",
        "view.imageHeight = 0;",
    });
    expectContainsAll(helper, {
        "if (!rawConfidencePath.isEmpty())",
        "loadDepthMatStorage(rawConfidencePath, &result.frame.confidence)",
        "result.frame.confidence.release();",
    });
    expectNotContainsAll(helper, {"cv::Mat filteredDepth = result.frame.depthMap.clone();"});
}

TEST(MvsSchedulerContractTest, QueueCancelPostprocessAndProgressContracts)
{
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    const QString fusionBody =
        sectionBetween(scheduler, "FusionFrameInput DepthMapGenerator::buildFusionFrame", "void DepthMapGenerator::crossCheckDepthConsistency");
    expectContainsAll(fusionBody, {"frame.confidence.release();"});

    const QString fuseBody = sectionBetween(manager,
                                            "ProjectDenseReconstructionManager::startFuseDepthMapsAsync",
                                            "void ProjectDenseReconstructionManager::startDenseCloudRefineAsync");
    expectContainsAll(fuseBody, {
        "正在加载深度图 %1/%2",
        "流式深度图融合 %1/%2",
        "mvsProgressChanged",
    });

    const QString queueBlock = sectionBetween(scheduler, "class DepthFrameArtifactSaveQueue", "// =============================================================================");
    expectContainsAll(queueBlock, {
        "maxBufferedTasks",
        "m_capacityCv.wait",
        "m_tasks.size() < m_maxBufferedTasks",
        "m_capacityCv.notify_one()",
        "void cancel()",
        "m_dropPendingTasks",
        "m_tasks.clear()",
    });

    const QString postWorkerBlock = sectionBetween(scheduler, "for (std::thread &worker : workers)", "// 释放图像缓存");
    expectContainsAll(postWorkerBlock, {
        "if (_cancelled.load())",
        "saveQueue.cancel()",
    });
    EXPECT_LT(indexOfOrFail(postWorkerBlock, "if (_cancelled.load())"),
              indexOfOrFail(postWorkerBlock, "saveQueue.waitUntilIdle()"));

    const QString crossBlock = sectionBetween(scheduler,
                                              "void DepthMapGenerator::crossCheckDepthConsistency()",
                                              "bool DepthMapGenerator::saveDepthFrameArtifacts");
    expectContainsAll(crossBlock, {"if (_cancelled.load())"});

    const int filteredSavePos = indexOfOrFail(scheduler, R"(saveQueue.enqueue(i, res, QStringLiteral("过滤后")))");
    const QString filteredBlock = scheduler.mid(qMax(0, filteredSavePos - 400), 600);
    expectContainsAll(filteredBlock, {"if (_cancelled.load())"});
}

TEST(MvsSchedulerContractTest, DenseSparsePreloadAndDepthArtifactSavingAreCancelableAndFast)
{
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    EXPECT_GE(countOccurrences(manager, "QPointer<DepthMapGenerator> genSelf(gen)"), 2);
    EXPECT_GE(countOccurrences(manager, "const QString projectPath = _owner ? _owner->currentProjectPath() : QString()"), 2);
    EXPECT_GE(countOccurrences(manager, "QtConcurrent::run([self, genSelf, sparseXyz, views, request, projectPath]()"), 2);
    EXPECT_GE(countOccurrences(manager, "self->_owner->currentProjectPath() != projectPath"), 2);
    EXPECT_GE(countOccurrences(manager, "if (genSelf->isCancelled())"), 2);
    EXPECT_GE(countOccurrences(manager, "return;"), 2);
    EXPECT_GE(countOccurrences(manager, R"(QMetaObject::invokeMethod(genSelf.data(), "finished")"), 2);
    EXPECT_GE(countOccurrences(manager, "QMetaObject::invokeMethod(genSelf.data(), [self, genSelf, sparseCloud, projectPath]()"), 2);
    expectContainsAll(manager, {"Q_ARG(bool, false)"});
    expectNotContainsAll(manager, {
        "QtConcurrent::run([gen, sparseXyz, views, request]()",
        "QtConcurrent::run([genSelf, sparseXyz, views, request]()",
        "gen->setSparseCloud(sparse)",
        R"(QMetaObject::invokeMethod(gen, "start")",
    });

    expectContainsAll(scheduler, {
        "writeFastDepthMatStorage",
        "saveDepthPreviewPng",
        "maxPreviewDimension",
        "保存%1深度产物耗时",
        "xjw::common::io::writeImage(path, colorVis)",
    });
    expectNotContainsAll(scheduler, {
        "FileStorage storage(path, cv::FileStorage::WRITE)",
        ".yml.gz",
    });
}

TEST(MvsSchedulerContractTest, VisibilityAndSourceViewCachesAvoidRedundantWork)
{
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    expectContainsAll(header, {
        "FrameMvsCache",
        "prepareFrameCaches",
        "_visibilityBits",
        "_pairCommonCounts",
        "sourceSharedPointIndices",
    });
    expectContainsAll(scheduler, {
        "prepareFrameCaches();",
        "sourceViewIndicesForFrame",
        "visibleSparsePointIndicesForFrame",
        "sourceSharedPointIndices.reserve",
        "sourceSharedPointIndices.push_back",
        "sourceIndicesMatchCachedPrefix",
    });
    expectNotContainsAll(scheduler, {"selectMvsSourceViewIndices(_views, _sparse, refIdx, numSrc)"});

    const QString visibleBlock =
        sectionBetween(scheduler, "std::vector<size_t> DepthMapGenerator::visibleSparsePointIndicesForFrame", "// =============================================================================");
    expectContainsAll(visibleBlock, {"return cache.sourceSharedPointIndices;"});
    EXPECT_LT(indexOfOrFail(visibleBlock, "return cache.sourceSharedPointIndices;"),
              indexOfOrFail(visibleBlock, "std::vector<size_t> filtered;"));

    const QString frameCacheBlock =
        sectionBetween(scheduler, "void DepthMapGenerator::prepareFrameCaches()", "std::vector<int> DepthMapGenerator::sourceViewIndicesForFrame");
    expectContainsAll(frameCacheBlock, {
        "VisibilityCacheShard",
        "visibilityWorkerCount",
        "#pragma omp parallel",
        "#pragma omp for",
        "shard.visiblePointIndicesByView",
        "shard.pairCommonCounts",
        "mergeVisibilityCacheShards",
        "buildVisibilityBitsFromFrameCaches",
        "rankedSourceCandidates",
        "desiredSourceCount",
        "currentSourceScoreCutoff",
        "remaining candidates are sorted by common count",
        "candidate.commonVisiblePoints <= currentSourceScoreCutoff",
    });
    expectNotContainsAll(frameCacheBlock, {
        "_frameCaches[static_cast<size_t>(viewIdx)].visiblePointIndices.push_back(pointIndex);",
    });
    EXPECT_LT(indexOfOrFail(frameCacheBlock, "candidate.commonVisiblePoints <= currentSourceScoreCutoff"),
              indexOfOrFail(frameCacheBlock, "sampledMedianAngle(refIdx, candidate.viewIndex)"));
}

TEST(MvsSchedulerContractTest, SparseHintsUseProjectedSamplesAndPrescaledPatchMatchInputs)
{
    const QString cameraHeader = readSourceFile(QStringLiteral("src/core/camera/Camera.h"));
    const QString cameraSource = readSourceFile(QStringLiteral("src/core/camera/Camera.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString pyramid = readSourceFile(QStringLiteral("src/core/mvs/DepthPyramidEstimator.cpp"));
    const QString cuda = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCUDA.cu"));

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
        "return projectWorldPoint(world, pixel)",
    });

    expectContainsAll(pyramid, {
        "propagateDepthPrior(parent, guide, target_size)",
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
        "hintDepth->cols == W && hintDepth->rows == H",
        "hintScaled = *hintDepth",
    });
    const QString gpuBody = sectionBetween(cuda, "bool PatchMatchDepthEstimator::estimateGPU", "bool PatchMatchDepthEstimator::estimateCPU");
    expectContainsAll(gpuBody, {
        "const int sW = std::max(1, refW / ds);",
        "const int sH = std::max(1, refH / ds);",
        "getOrUploadGrayImageGpu(refGray, sW, sH, ds",
    });
    expectNotContainsAll(gpuBody, {"cv::resize(refGray, refScaled"});
}

TEST(MvsSchedulerContractTest, PatchMatchRequiresRobustMultiViewPhotometricSupport)
{
    const QString policy = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchPhotometricCost.h"));
    const QString cuda = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCUDA.cu"));
    const QString implementation = policy + cuda;

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

TEST(MvsSchedulerContractTest, StoredDepthFramesUseSharedPostprocessBeforeFusion)
{
    const QString depth_utils = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString manager =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString stored_frame = sectionBetween(depth_utils,
                                                "FusionFrameBuildResult buildStoredFusionFrame",
                                                "} // namespace xjw::core::project");

    expectContainsAll(stored_frame, {
        "const xjw::mvs::FusionConfig &fusionConfig",
        "DepthMapGenerator::postprocessFusionDepthMap",
        "result.frame.depthPostprocess",
    });
    expectContainsAll(manager, {
        "buildDepthGenConfig(request, totalFrames).fusion",
        "const xjw::mvs::FusionConfig &fusionConfig",
    });
    expectContainsAll(scheduler, {
        "#pragma omp parallel for schedule(static)",
        "removeLocalDepthOutliers",
        "postprocessFusionDepthMap",
    });
}

TEST(MvsSchedulerContractTest, StoredDepthFusionUsesBoundedParallelPrefetch)
{
    const QString manager = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString depth_utils = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));

    ASSERT_FALSE(manager.isEmpty());
    ASSERT_FALSE(depth_utils.isEmpty());
    EXPECT_TRUE(manager.contains(QStringLiteral("recommendedDepthFrameLoadWorkers")));
    EXPECT_TRUE(manager.contains(QStringLiteral("std::condition_variable")));
    EXPECT_TRUE(manager.contains(QStringLiteral("depthFrameCache.prefetch(next_window)")));
    EXPECT_TRUE(manager.contains(QStringLiteral("准备深度图：已加载 %1/%2")));
    EXPECT_TRUE(manager.contains(QStringLiteral("readTotal=%9 ms postTotal=%10 ms resizeTotal=%11 ms")));
    EXPECT_TRUE(depth_utils.contains(QStringLiteral("estimateFusionFrameWorkingSetBytes")));
    EXPECT_TRUE(depth_utils.contains(QStringLiteral("downsampleFusionFrameForMaxDimension")));
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

TEST(MvsSchedulerContractTest, DepthPostprocessAndFusionContracts)
{
    const QString types = readSourceFile(QStringLiteral("src/core/mvs/MvsTypes.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString cli =
        readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"))
        + readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.cpp"))
        + readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.h"));
    const QString streamingFusion =
        readSourceFile(QStringLiteral("src/core/mvs/StreamingDepthFusionService.cpp"));
    const QString fusionH = readSourceFile(QStringLiteral("src/core/mvs/DepthMapFusion.h"));
    const QString fusion = readSourceFile(QStringLiteral("src/core/mvs/DepthMapFusion.cpp"));
    const QString depthUtilsH = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.h"));
    const QString depthUtils = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    expectContainsAll(scheduler, {
        "if (seedHintCnt <= 0)",
        "没有可用 hint seed",
        "FrameTiming",
        "耗时统计",
        "source=",
        "patchmatch=",
        "filter=",
        "applySparseSupportPrior(depthMap, confMap, supportMask, refIdx)",
        "稀疏支撑软约束",
        "removeLocalDepthOutliers",
        "postprocessFusionDepthMap(filteredDepth",
    });
    expectNotContainsAll(scheduler, {
        "depthMap.setTo(0, supportMask == 0)",
        "confMap.setTo(0, supportMask == 0)",
    });
    expectContainsAll(types, {
        "DepthPostProcessStats",
        "enableLocalDepthOutlierFilter",
        "localDepthOutlierRelThresh",
        "maxLocalDepthOutlierRemovalRatio",
    });
    expectContainsAll(cli, {
        "postprocessFusionDepthMap",
        "depth_postprocess",
        "local_depth_outlier_removed",
        "speckle_removed",
        "edge_confidence_removed",
        "geom_consistency_removed",
        "rawDepthStoragePath(depthPngPath)",
        "loadDepthMatStorage",
        "--mvs-res-scale",
        "--mvs-iterations",
        "--mvs-confidence",
        "--mvs-gpu-frame-workers",
        "--mvs-cpu-frame-workers",
        "--mvs-max-frames",
        "denseSettings.resScale = mvsResScale;",
        "denseSettings.iterations = mvsIterations;",
        "denseSettings.patchMatchConfidence = mvsConfidence;",
        "denseSettings.fusionMinConfidence = mvsFusionConfidence;",
        "denseSettings.gpuFrameWorkers = mvsGpuFrameWorkers;",
        "denseSettings.cpuFrameWorkers = mvsCpuFrameWorkers;",
        "limitMvsInputsForRegression",
    });
    expectMatches(scheduler,
                  R"(if\s*\(\s*keepDepthFramesInMemory\.load\(\)\s*&&\s*\(\s*savePreviewPng\s*\|\|\s*saveRawDepth\s*\)\s*\))");

    expectContainsAll(cli, {
        "QJsonArray depthArtifacts;",
        "&xjw::mvs::DepthMapGenerator::depthMapArtifactSaved",
        "depthArtifacts.append(artifact)",
        R"(denseReport[QStringLiteral("depth_maps")] = depthArtifacts;)",
        R"(denseReport[QStringLiteral("mvs_settings")])",
        R"(denseReport[QStringLiteral("mvs_depth_config")])",
        "bool mvsDepthOnly = false;",
        "--mvs-depth-only",
        R"(denseReport[QStringLiteral("status")] = QStringLiteral("depth_only");)",
        R"(report[QStringLiteral("stop_stage")] = QStringLiteral("mvs_depth");)",
        R"(markSkippedStage(QStringLiteral("mvs_fusion"), depthOnlyReason);)",
        R"(markSkippedStage(QStringLiteral("mesh"), depthOnlyReason);)",
        R"(markSkippedStage(QStringLiteral("terrain"), depthOnlyReason);)",
        "const int depthMapCount = static_cast<int>(depthArtifacts.size());",
        R"(std::fprintf(stdout, "depth_maps=%d\n", depthMapCount);)",
        "fuseDepthMapsStreamingFromDisk",
        "voxelDownsampleFusedPointsToTarget",
        "--mvs-fusion-max-image-dim",
        "mvsFusionMaxImageDim",
        "loadFusionFrameFromDepthMap(",
        "fusionMaxImageDim",
        R"(settings[QStringLiteral("fusion_max_image_dim")])",
    });
    expectContainsAll(streamingFusion, {
        "streamingFusionWindowIndices",
        "config.cacheFrameLimit",
        "const bool cacheFrames",
        "std::vector<FusionFrameInput> cachedFrames",
        "if (cacheFrames)",
        "fusionConfig.fuseOnlyFirstFrame = true",
        "frameLoader",
        "MVS 流式融合没有生成有效稠密点",
    });
    expectContainsAll(depthUtilsH, {"int fusionMaxImageDim"});
    expectContainsAll(depthUtils, {
        "frame->cameraModel.scaledIntrinsics",
        "downsampleFusionFrameForMaxDimension",
        "cv::resize(frame->depthMap",
        "cv::resize(frame->confidence",
    });
    expectContainsAll(fusionH, {
        "bool  useColor",
        "int   colorCacheCapacity",
        "bool  fuseOnlyFirstFrame",
        "fuseFirstFrameObservationsFast",
    });
    expectContainsAll(fusion, {
        "class ColorImageCache",
        "ColorImageCache colorCache",
        "colorCache.get",
        "const int fusionStartFrame",
        "const int fusionEndFrame",
        "m_frames[frameIdx].imgW",
        "m_frames[frameIdx].imgH",
        "cv::resize(image, image",
        "fuseFirstFrameObservationsFast",
        "使用已过滤深度图快速反投影",
        "_config.fuseOnlyFirstFrame",
        "resolveFusionWorkerCount",
    });
    expectNotContainsAll(fusion, {
        "std::vector<cv::Mat> colorImages(NF)",
        "colorImages[fi] = cv::imread",
    });
    expectContainsAll(manager, {
        "class DepthFrameLruCache",
        "nearestFusionWindowIndices",
        "streamFusionWindowSize",
        "depthFrameCache.get",
        "fusionCfg.fuseOnlyFirstFrame = true",
        "流式深度图融合",
        "fusionMaxImageDim",
        "_fusionMaxImageDim",
        "request.fusionMaxImageDim",
        "_fusionMaxImageDim)",
    });
    expectNotContainsAll(manager, {
        "frames.reserve(storedFrames.size());\n        for (const auto &stored : storedFrames)",
    });
}

TEST(MvsSchedulerContractTest, SuccessfulDepthGenerationDoesNotOpenCompletionDialog)
{
    const QString manager = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    EXPECT_FALSE(manager.contains(QStringLiteral("深度图估计完成。")));
    EXPECT_TRUE(manager.contains(QStringLiteral("深度图估计失败或被取消。")));
}

TEST(MvsSchedulerContractTest, DepthConsistencyMetadataAndReuseContracts)
{
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString generator = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString tree = readSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    const QString depthUtils = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));
    const QString terrain = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    const QString model = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString heatmapH = readSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.h"));
    const QString heatmap = readSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.cpp"));

    expectContainsAll(header, {"sourceViewIndices", "void depthMapArtifactSaved(QJsonObject artifact)"});
    expectContainsAll(generator, {
        "std::vector<int> sourceIndices",
        "sourceIndices.push_back(si)",
        "result.sourceViewIndices = sourceIndices",
        "consistencySourceIndicesForFrame",
        "const std::vector<int> consistencySources",
        "saveDepthPreviewPng",
        "xjw::common::io::writeImage(path, colorVis)",
        "saveDepthFrameArtifacts",
        "emit depthMapSaved",
        R"(artifact[QStringLiteral("source_images")])",
        R"(artifact[QStringLiteral("source_indices")])",
        R"(artifact[QStringLiteral("valid_mask_path")])",
        R"(artifact[QStringLiteral("device")])",
        R"(artifact[QStringLiteral("elapsed_ms")])",
        "emit depthMapArtifactSaved(artifact)",
        R"(cancelled("深度范围估计后"))",
        R"(cancelled("极线校正后"))",
        R"(cancelled("构建三级深度先验后"))",
        R"(cancelled("三级 PatchMatch 后"))",
        R"(cancelled("深度后处理后"))",
    });
    EXPECT_GT(indexOfOrFail(generator, R"(saveQueue.enqueue(i, res, QStringLiteral("过滤后")))"),
              indexOfOrFail(generator, "crossCheckDepthConsistency();"));
    EXPECT_GT(indexOfOrFail(generator, "emit depthMapSaved"), 0);

    expectContainsAll(manager, {
        "makeProjectDepthRecordFromArtifact",
        "&DepthMapGenerator::depthMapArtifactSaved",
        R"(depthResult[QStringLiteral("mvs_output_dir")])",
        "depthFrameArtifactsExist(pngPath)",
        "cameraForImagePath(camMap, imgPath",
        "cameraForImagePath(_cameraMap, _records[index].refImage",
        "if (request.pipelineMode)",
        "genCfg.runFusion = false",
        "shouldStartFusion = success && (continueMissingMode || pipelineMode)",
        "startFuseDepthMapsAsync(settings)",
    });
    expectNotContainsAll(manager, {
        "camMap.value(imgPath)",
        "camMap.value(stored.refImage)",
    });
    expectContainsAll(tree, {
        "depthRecordPrimaryPath",
        R"(normalized.value("depth_map_results").toArray())",
        "depthFrameKeys",
        "depth_preview_png",
    });
    expectContainsAll(depthUtils, {
        R"(record.value(QStringLiteral("depth_png")))",
        "depthFrameArtifactsExist(frame)",
        "QFileInfo::exists(frame.depthPng)",
        "QFileInfo::exists(frame.rawDepthPath)",
    });

    expectNotContainsAll(terrain, {
        R"(upsertMetaArrayRecordByPath(&metaUpdated, QStringLiteral("depth_map_results"))",
    });
    expectNotContainsAll(model, {"depth_map_results"});
    expectContainsAll(terrain, {"collectLatestStoredDepthFrames"});

    expectContainsAll(heatmapH, {"QImage heatmapImage() const"});
    expectContainsAll(heatmap, {
        "QImage::Format_RGBA8888",
        "alphaRow[col] = validRow[col] ? 255 : 0",
        "if (_showInvalid)",
    });
}

TEST(MvsSchedulerContractTest, DenseCloudRefineFilteringAndCancelContracts)
{
    const QString preprocessor = readSourceFile(QStringLiteral("src/core/mvs/SparseCloudPreprocessor.cpp"));
    const QString generator = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.h"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString configH = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.h"));
    const QString configCpp = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.cpp"));

    expectContainsAll(preprocessor, {
        "kMaxMedianSpacingSamples",
        "sampleCount",
        "#pragma omp parallel for",
    });
    expectNotContainsAll(preprocessor, {"distances.reserve(cloud.size());"});
    expectContainsAll(generator, {
        "kMaxInlineDenseFilterPoints",
        "initialCount <= kMaxInlineDenseFilterPoints",
        "跳过内联稠密点云过滤",
    });
    expectNotContainsAll(generator, {"\"SOR-2\""});

    expectContainsAll(manager, {
        "统计离群点移除 (SOR)",
        "半径离群点移除",
    });
    expectNotContainsAll(manager, {
        "离群点二次清理",
        "strictSorReport",
    });

    expectContainsAll(header, {
        "std::shared_ptr<std::atomic_bool> _activeMvsCancelFlag",
        "createActiveMvsCancelFlag",
        "clearActiveMvsCancelFlag",
    });
    const QString cancelBody = sectionFrom(manager, "void ProjectDenseReconstructionManager::cancelMvs()");
    expectContainsAll(cancelBody, {
        "_activeMvsCancelFlag->store(true",
        "gen->requestCancel()",
        "if (!requestedCancel)",
        "emit mvsProgressFinished(false);",
    });
    EXPECT_LT(indexOfOrFail(cancelBody, "if (!requestedCancel)"),
              indexOfOrFail(cancelBody, R"(qDebug() << "[MVS] 已请求取消")"));

    const QString fuseBody = sectionBetween(manager,
                                            "ProjectDenseReconstructionManager::startFuseDepthMapsAsync",
                                            "void ProjectDenseReconstructionManager::startDenseCloudRefineAsync");
    const QString refineBody = sectionFrom(manager, "void ProjectDenseReconstructionManager::startDenseCloudRefineAsync");
    expectContainsAll(fuseBody, {
        "const auto cancelFlag = createActiveMvsCancelFlag();",
        "cancelFlag",
        "cancelFlag->load",
        "深度图融合已取消",
        "clearActiveMvsCancelFlag(cancelFlag)",
    });
    expectContainsAll(manager, {
        "kMaxDenseRefineFilterInputPoints",
        "preconditionDenseRefineCloudForFilters",
        "点云过大，先进行预降采样",
        "kStreamingDenseRefineMinPoints",
        "shouldUseStreamingDenseRefine",
        "runStreamingDenseCloudRefineCli",
        "dense_cloud_refine_cli",
        "parseBinaryPlyVertexStreamHeader",
        "QProcess process",
        "--terrain-filter-passes",
    });
    expectContainsAll(refineBody, {
        "const auto cancelFlag = createActiveMvsCancelFlag();",
        "cancelFlag",
        "cancelFlag->load",
        "密集点云后处理已取消",
        "clearActiveMvsCancelFlag(cancelFlag)",
        "!precondition.consumedRequestedVoxel",
        "shouldUseStreamingDenseRefine",
        "runStreamingDenseCloudRefineCli",
    });
    EXPECT_LT(indexOfOrFail(refineBody, "preconditionDenseRefineCloudForFilters"),
              indexOfOrFail(refineBody, "sorFilter(cloud"));
    EXPECT_LT(indexOfOrFail(refineBody, "preconditionDenseRefineCloudForFilters"),
              indexOfOrFail(refineBody, "estimateNormals(cloud"));
    EXPECT_LT(indexOfOrFail(refineBody, "runStreamingDenseCloudRefineCli"),
              indexOfOrFail(refineBody, "readPointCloudPly(inputPly"));

    expectContainsAll(configH, {
        "int terrainSpikeGridResolution = 260;",
        "int terrainSpikeMinCellPoints = 32;",
        "double terrainSpikeMinHeightThreshold = 0.25;",
        "double terrainSpikeMadMultiplier = 3.0;",
        "bool terrainLocalPlaneFilterEnabled = true;",
        "int terrainLocalPlaneMinPoints = 12;",
        "double terrainLocalPlaneMinResidualThreshold = 0.12;",
        "double terrainLocalPlaneMadMultiplier = 4.0;",
        "int terrainFilterPasses = 2;",
    });
    expectContainsAll(configCpp, {
        R"(settings.value(QStringLiteral("terrainSpikeGridResolution")).toInt(260))",
        R"(settings.value(QStringLiteral("terrainSpikeMinCellPoints")).toInt(32))",
        R"(settings.value(QStringLiteral("terrainSpikeMinHeightThreshold")).toDouble(0.25))",
        R"(settings.value(QStringLiteral("terrainSpikeMadMultiplier")).toDouble(3.0))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneFilterEnabled")).toBool(true))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneMinPoints")).toInt(12))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneMinResidualThreshold")).toDouble(0.12))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneMadMultiplier")).toDouble(4.0))",
        R"(settings.value(QStringLiteral("terrainFilterPasses")).toInt(2))",
    });
    expectContainsAll(manager, {
        "options.localPlaneFilterEnabled = request.terrainLocalPlaneFilterEnabled",
        "options.localPlaneMinPoints = request.terrainLocalPlaneMinPoints",
        "options.localPlaneMinResidualThreshold",
        "options.localPlaneMadMultiplier",
        "local_plane_removed_points",
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
        QStringLiteral("<header>camera/CameraModel3DDialog.h</header>")));
}

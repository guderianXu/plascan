#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"
#include "reconstruction/SfmAttemptRunner.h"
#include "search/SfmSearchPolicy.h"
#include "workflow/AerialTriangulationPipeline.h"

#include "Camera.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

namespace
{

void writeKnownPoseTiePoints(const QString &path,
                             const QString &imageA,
                             const QString &imageB,
                             const xjw::Camera &cameraA,
                             const xjw::Camera &cameraB)
{
    QJsonArray tracks;
    for (int featureIndex = 0; featureIndex < 30; ++featureIndex)
    {
        const std::array<double, 3> point{
            (featureIndex % 6 - 2.5) * 0.16,
            (featureIndex / 6 - 2.0) * 0.14,
            5.0 + 0.05 * (featureIndex % 3)};
        double pixelA[2]{};
        double pixelB[2]{};
        ASSERT_TRUE(cameraA.projectWorldPoint(point.data(), pixelA));
        ASSERT_TRUE(cameraB.projectWorldPoint(point.data(), pixelB));
        tracks.append(QJsonObject{
            {QStringLiteral("confidence"), 1.0},
            {QStringLiteral("observations"),
             QJsonArray{
                 QJsonObject{{QStringLiteral("image_id"), 0},
                             {QStringLiteral("feature_idx"), featureIndex},
                             {QStringLiteral("xy"), QJsonArray{pixelA[0], pixelA[1]}}},
                 QJsonObject{{QStringLiteral("image_id"), 1},
                             {QStringLiteral("feature_idx"), featureIndex},
                             {QStringLiteral("xy"), QJsonArray{pixelB[0], pixelB[1]}}},
             }},
        });
    }

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(QJsonDocument(QJsonObject{
        {QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
        {QStringLiteral("format_version"), 1},
        {QStringLiteral("images"),
         QJsonArray{
             QJsonObject{{QStringLiteral("image_id"), 0}, {QStringLiteral("path"), imageA}},
             QJsonObject{{QStringLiteral("image_id"), 1}, {QStringLiteral("path"), imageB}},
         }},
        {QStringLiteral("tracks"), tracks},
    }).toJson(QJsonDocument::Compact));
}

} // namespace

TEST(AerialTriangulationPipelineTest, MissingPreparedTiePointFileFailsWithoutFrontendFallback)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};
    input.tiePointPath = QStringLiteral("missing/latest_tie_points.json");
    input.outputDir = QStringLiteral("output");

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline().run(input);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("连接点")));
}

TEST(AerialTriangulationPipelineTest, ReplaysBestCoarseFocalCandidateWhenBaseCoverageIsIncomplete)
{
    QVector<double> attemptedScales;
    const auto attemptRunner = [&attemptedScales](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.meanReprojError = 0.6;
        execution.result.numRegisteredImages = 9;
        execution.result.numPoints3D = 500;
        if (std::abs(input.estimatedFocalScale - 0.85) < 1.0e-9)
        {
            execution.result.numRegisteredImages = 16;
            execution.result.numPoints3D = 1200;
            execution.result.meanReprojError = 0.4;
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 1;
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numRegisteredImages, 16);
    ASSERT_GE(attemptedScales.size(), 3);
    EXPECT_DOUBLE_EQ(attemptedScales.front(), 1.2);
    EXPECT_DOUBLE_EQ(attemptedScales.back(), 0.85);
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     0.85);
}

TEST(AerialTriangulationPipelineTest, SearchesNarrowFieldFocalCandidatesEvenWhenBaseRegistersAllImages)
{
    QVector<double> attemptedScales;
    const auto attemptRunner = [&attemptedScales](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 1000;
        execution.result.meanReprojError = 0.7;
        if (std::abs(input.estimatedFocalScale - 5.2) < 1.0e-9)
        {
            execution.result.numPoints3D = 2400;
            execution.result.meanReprojError = 0.4;
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 1;
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_NE(std::find(attemptedScales.cbegin(), attemptedScales.cend(), 5.2),
              attemptedScales.cend());
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     5.2);
}

TEST(AerialTriangulationPipelineTest, ParallelizesCoarseFocalSearchWithinThreadBudget)
{
    std::atomic<int> activeAttempts{0};
    std::atomic<int> maximumConcurrentAttempts{0};
    const auto attemptRunner = [&activeAttempts, &maximumConcurrentAttempts](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &)
    {
        const int active = activeAttempts.fetch_add(1) + 1;
        int observedMaximum = maximumConcurrentAttempts.load();
        while (active > observedMaximum &&
               !maximumConcurrentAttempts.compare_exchange_weak(observedMaximum, active))
        {
        }
        // 给所有显式 worker 足够的重叠窗口；真实焦距候选通常运行数秒以上。
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        activeAttempts.fetch_sub(1);

        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 1000;
        execution.result.meanReprojError = 0.5;
        execution.result.sfmDiagnostics.insert(
            QStringLiteral("sparse_quality"),
            QJsonObject{{QStringLiteral("quality_gate"),
                         QJsonObject{{QStringLiteral("acceptable_for_mvs"), true}}}});
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 32;
    input.adaptiveCameraModelFitting = false;

    const int expectedWorkers = std::min(
        static_cast<int>(
            xjw::aerial_triangulation::adaptiveFocalCoarseScaleCandidates().size()) + 1,
        xjw::aerial_triangulation::resolveSfmThreadBudget(input.threads));

    const auto result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(
            attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(maximumConcurrentAttempts.load(), expectedWorkers);
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_worker_count")).toInt(), expectedWorkers);
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_thread_budget")).toInt(),
        xjw::aerial_triangulation::resolveSfmThreadBudget(input.threads));
}

TEST(AerialTriangulationPipelineTest, CompleteCoarseModelOnlyEvaluatesTopSeedNeighborhoods)
{
    std::atomic<int> attemptCount{0};
    const auto attemptRunner = [&attemptCount](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &)
    {
        attemptCount.fetch_add(1);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 1000;
        execution.result.meanReprojError = 0.5;
        execution.result.sfmDiagnostics.insert(
            QStringLiteral("sparse_quality"),
            QJsonObject{{QStringLiteral("quality_gate"),
                         QJsonObject{{QStringLiteral("acceptable_for_mvs"), true}}}});
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 4;
    input.adaptiveCameraModelFitting = false;

    const auto result = xjw::aerial_triangulation::AerialTriangulationPipeline(
        attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_exhaustive_fallback")).toBool());
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_coarse_candidate_count")).toInt(), 6);
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_refinement_candidate_count")).toInt(), 3);
    EXPECT_EQ(attemptCount.load(), 9);
    EXPECT_LT(attemptCount.load(),
              static_cast<int>(
                  xjw::aerial_triangulation::adaptiveFocalScaleCandidates().size()));
}

TEST(AerialTriangulationPipelineTest, CompleteButPoorCoarseModelFallsBackToFullFocalRange)
{
    std::atomic<int> attemptCount{0};
    const auto attemptRunner = [&attemptCount](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &)
    {
        attemptCount.fetch_add(1);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 1000;
        execution.result.meanReprojError = 0.5;
        execution.result.sfmDiagnostics.insert(
            QStringLiteral("sparse_quality"),
            QJsonObject{{QStringLiteral("quality_gate"),
                         QJsonObject{{QStringLiteral("acceptable_for_mvs"), false}}}});
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 2;
    input.adaptiveCameraModelFitting = false;

    const auto result = xjw::aerial_triangulation::AerialTriangulationPipeline(
        attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_exhaustive_fallback")).toBool());
    EXPECT_EQ(attemptCount.load(), static_cast<int>(
        xjw::aerial_triangulation::adaptiveFocalScaleCandidates().size()));
}

TEST(AerialTriangulationPipelineTest, IncompleteCoarseSearchFallsBackToFullFocalRange)
{
    QVector<double> attemptedScales;
    const auto attemptRunner = [&attemptedScales](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = std::abs(input.estimatedFocalScale - 10.0) < 1.0e-9;
        execution.result.numRegisteredImages = execution.result.success ? 16 : 4;
        execution.result.numPoints3D = execution.result.success ? 1200 : 0;
        execution.result.meanReprojError = execution.result.success ? 0.4 : 2.0;
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 1;
    input.adaptiveCameraModelFitting = false;

    const auto result = xjw::aerial_triangulation::AerialTriangulationPipeline(
        attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_exhaustive_fallback")).toBool());
    EXPECT_GT(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_fallback_candidate_count")).toInt(), 0);
    EXPECT_NE(std::find(attemptedScales.cbegin(), attemptedScales.cend(), 10.0),
              attemptedScales.cend());
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_focal_scale")).toDouble(), 10.0);
}

TEST(AerialTriangulationPipelineTest, LargeDatasetProbesCandidatesAndReplaysOnlyWinnerAtFullScale)
{
    struct AttemptRecord
    {
        double focalScale = 0.0;
        bool coarse = false;
        int maxRegisteredImages = 0;
    };

    QVector<AttemptRecord> attempts;
    const auto attemptRunner = [&attempts](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attempts.append({input.estimatedFocalScale,
                         input.coarseFocalEvaluation,
                         input.maxRegisteredImages});
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = input.coarseFocalEvaluation ? 24 : 444;
        execution.result.numPoints3D =
            std::abs(input.estimatedFocalScale - 5.2) < 1.0e-9 ? 2400 : 1000;
        execution.result.meanReprojError =
            std::abs(input.estimatedFocalScale - 5.2) < 1.0e-9 ? 0.4 : 0.8;
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 444; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    // 单 worker 使记录顺序确定；并行预算由单独策略测试覆盖。
    input.threads = 1;
    input.adaptiveCameraModelFitting = false;

    const auto result = xjw::aerial_triangulation::AerialTriangulationPipeline(
        attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numRegisteredImages, 444);
    ASSERT_GE(attempts.size(), 2);
    for (int index = 0; index + 1 < attempts.size(); ++index)
    {
        EXPECT_TRUE(attempts.at(index).coarse);
        EXPECT_EQ(attempts.at(index).maxRegisteredImages, 24);
    }
    EXPECT_FALSE(attempts.back().coarse);
    EXPECT_EQ(attempts.back().maxRegisteredImages, 0);
    EXPECT_DOUBLE_EQ(attempts.back().focalScale, 5.2);
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("focal_probe_registration_limit")).toInt(), 24);
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("focal_probe_full_replay")).toBool());
}

TEST(AerialTriangulationPipelineTest, InitializesUnknownFocalEvenWhenAdaptiveModelFittingIsDisabled)
{
    QVector<double> attemptedScales;
    QVector<bool> attemptedAdaptiveFlags;
    const auto attemptRunner = [&attemptedScales, &attemptedAdaptiveFlags](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        attemptedAdaptiveFlags.append(input.adaptiveCameraModelFitting);

        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 900;
        execution.result.meanReprojError = 0.8;
        if (std::abs(input.estimatedFocalScale - 5.2) < 1.0e-9)
        {
            execution.result.numPoints3D = 2400;
            execution.result.meanReprojError = 0.4;
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 1;
    input.adaptiveCameraModelFitting = false;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_NE(std::find(attemptedScales.cbegin(), attemptedScales.cend(), 5.2),
              attemptedScales.cend());
    EXPECT_TRUE(std::all_of(attemptedAdaptiveFlags.cbegin(),
                            attemptedAdaptiveFlags.cend(),
                            [](bool enabled) { return !enabled; }));
    EXPECT_TRUE(result.sfmDiagnostics.value(QStringLiteral("focal_initialization_search")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     5.2);
}

TEST(AerialTriangulationPipelineTest, RejectsAdaptiveRefinementWhenItLosesRegisteredImages)
{
    QVector<double> attemptedScales;
    QVector<bool> attemptedAdaptiveFlags;
    const auto attemptRunner = [&attemptedScales, &attemptedAdaptiveFlags](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        attemptedAdaptiveFlags.append(input.adaptiveCameraModelFitting);

        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 9;
        execution.result.numPoints3D = 700;
        execution.result.meanReprojError = 0.6;
        if (std::abs(input.estimatedFocalScale - 0.85) < 1.0e-9)
        {
            execution.result.numRegisteredImages = input.adaptiveCameraModelFitting ? 9 : 16;
            execution.result.numPoints3D = input.adaptiveCameraModelFitting ? 900 : 1800;
            execution.result.meanReprojError = input.adaptiveCameraModelFitting ? 0.3 : 0.45;
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 1;
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numRegisteredImages, 16);
    EXPECT_TRUE(std::all_of(attemptedAdaptiveFlags.cbegin(),
                            attemptedAdaptiveFlags.cend() - 1,
                            [](bool enabled) { return !enabled; }));
    ASSERT_FALSE(attemptedAdaptiveFlags.isEmpty());
    EXPECT_TRUE(attemptedAdaptiveFlags.back());
    EXPECT_DOUBLE_EQ(attemptedScales.back(), 0.85);
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_refinement_accepted")).toBool());
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_scheduled")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_effective")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_applied")).toBool());
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_skip_reason")).toString(),
        QStringLiteral("refinement_rejected"));
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_focal_seed_scale")).toDouble(), 0.85);
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("camera_self_calibration_status")).toString(),
        QStringLiteral("coarse_seed_only"));
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("camera_self_calibration_requires_review")).toBool());
}

TEST(AerialTriangulationPipelineTest, ReportsAcceptedAdaptiveFocalRefinement)
{
    const auto attemptRunner = [](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 1000;
        execution.result.meanReprojError = 0.8;
        if (std::abs(input.estimatedFocalScale - 4.0) < 1.0e-9)
        {
            execution.result.numPoints3D = input.adaptiveCameraModelFitting ? 2300 : 2000;
            execution.result.meanReprojError = input.adaptiveCameraModelFitting ? 0.3 : 0.5;
            if (input.adaptiveCameraModelFitting)
            {
                execution.result.sfmDiagnostics.insert(
                    QStringLiteral("final_camera_focal_median_px"), 9063.9);
                execution.result.sfmDiagnostics.insert(
                    QStringLiteral(
                        "ba_adaptive_camera_model_fitting_evaluated"), true);
                execution.result.sfmDiagnostics.insert(
                    QStringLiteral(
                        "ba_adaptive_camera_model_fitting_applied"), true);
                execution.result.sfmDiagnostics.insert(
                    QStringLiteral("ba_adaptive_camera_model"),
                    QStringLiteral("f+k1"));
            }
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.threads = 1;
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_refinement_accepted")).toBool());
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_focal_seed_scale")).toDouble(), 4.0);
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("camera_self_calibration_status")).toString(),
        QStringLiteral("refined"));
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("camera_self_calibration_requires_review")).toBool());
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(
        QStringLiteral("final_camera_focal_median_px")).toDouble(), 9063.9);
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_requested")).toBool());
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_scheduled")).toBool());
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_effective")).toBool());
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_applied")).toBool());
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_skip_reason")).toString().isEmpty());
}

TEST(AerialTriangulationPipelineTest, DefaultEstimatedFocalScaleUsesLongestImageDimensionRatio)
{
    const xjw::aerial_triangulation::PreparedAerialTriangulationInput input;

    EXPECT_DOUBLE_EQ(input.estimatedFocalScale, 1.2);
}

TEST(AerialTriangulationPipelineTest, LegacySfmCameraMetadataDoesNotSuppressFocalSearch)
{
    QVector<double> attemptedScales;
    const auto attemptRunner = [&attemptedScales](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = input.images.size();
        execution.result.numPoints3D = 100;
        execution.result.meanReprojError = 0.5;
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    QJsonArray images;
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 3; ++index)
    {
        const QString path = QStringLiteral("legacy_%1.tif").arg(index);
        input.images.append(path);
        images.append(QJsonObject{
            {QStringLiteral("path"), path},
            {QStringLiteral("camera"),
             QJsonObject{
                 {QStringLiteral("fu"), 430.0},
                 {QStringLiteral("fv"), 430.0},
                 {QStringLiteral("cu"), 390.0},
                 {QStringLiteral("cv"), 360.0},
                 {QStringLiteral("pitch"), 1.0},
                 {QStringLiteral("C"), QJsonArray{0.0, 0.0, 0.0}},
                 {QStringLiteral("R"), QJsonArray{1.0, 0.0, 0.0,
                                                  0.0, 1.0, 0.0,
                                                  0.0, 0.0, 1.0}},
             }},
        });
    }
    input.projectMeta.insert(QStringLiteral("images"), images);
    input.useProjectCameraIntrinsics = true;
    input.threads = 1;

    const auto result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_GT(attemptedScales.size(), 1);
    EXPECT_TRUE(result.sfmDiagnostics.value(QStringLiteral("focal_initialization_search")).toBool());
}

TEST(AerialTriangulationPipelineTest, KeepsAdaptiveCalibrationFixedForCompleteProjectPoses)
{
    QVector<bool> attemptedAdaptiveFlags;
    const auto attemptRunner = [&attemptedAdaptiveFlags](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedAdaptiveFlags.append(input.adaptiveCameraModelFitting);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = input.images.size();
        execution.result.numPoints3D = 100;
        execution.result.meanReprojError = 0.5;
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    QJsonArray images;
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 3; ++index)
    {
        const QString path = QStringLiteral("known_pose_%1.tif").arg(index);
        input.images.append(path);
        images.append(QJsonObject{
            {QStringLiteral("path"), path},
            {QStringLiteral("camera"),
             QJsonObject{
                 {QStringLiteral("fu"), 430.0},
                 {QStringLiteral("fv"), 430.0},
                 {QStringLiteral("cu"), 390.0},
                 {QStringLiteral("cv"), 360.0},
                 {QStringLiteral("pitch"), 1.0},
                 {QStringLiteral("intrinsic_source"), QStringLiteral("sfm_estimated")},
                 {QStringLiteral("pose_initialized_as_identity"), false},
                 {QStringLiteral("C"), QJsonArray{static_cast<double>(index), 0.0, 0.0}},
                 {QStringLiteral("R"), QJsonArray{1.0, 0.0, 0.0,
                                                  0.0, 1.0, 0.0,
                                                  0.0, 0.0, 1.0}},
             }},
        });
    }
    input.projectMeta.insert(QStringLiteral("images"), images);
    input.useProjectCameraIntrinsics = true;
    input.useProjectCameraPoses = true;
    input.adaptiveCameraModelFitting = true;
    input.threads = 1;

    const auto result = xjw::aerial_triangulation::AerialTriangulationPipeline(
        attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    ASSERT_EQ(attemptedAdaptiveFlags.size(), 1);
    EXPECT_FALSE(attemptedAdaptiveFlags.front());
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_requested")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_scheduled")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_effective")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_applied")).toBool());
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_skip_reason")).toString(),
        QStringLiteral("known_pose_input"));
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("camera_self_calibration_status")).toString(),
        QStringLiteral("known_pose_fixed_calibration"));
}

TEST(AerialTriangulationPipelineTest, DoesNotTrustMalformedExternalCameraFiles)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    std::atomic<int> attemptCount{0};
    const auto attemptRunner = [&attemptCount](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        ++attemptCount;
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = input.images.size();
        execution.result.numPoints3D = 100;
        execution.result.meanReprojError = 0.5;
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 3; ++index)
    {
        input.images.append(QStringLiteral("invalid_camera_image_%1.tif").arg(index));
        const QString cameraPath = QDir(tempDir.path()).filePath(
            QStringLiteral("invalid_%1.tsai").arg(index));
        QFile cameraFile(cameraPath);
        ASSERT_TRUE(cameraFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
        cameraFile.write("not a camera\n");
        cameraFile.close();
        input.cameraPaths.append(cameraPath);
    }
    input.adaptiveCameraModelFitting = false;
    input.threads = 1;

    const auto result = xjw::aerial_triangulation::AerialTriangulationPipeline(
        attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_GT(attemptCount.load(), 1);
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("focal_initialization_search")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_scheduled")).toBool());
    EXPECT_EQ(result.sfmDiagnostics.value(
        QStringLiteral("adaptive_camera_model_fitting_skip_reason")).toString(),
        QStringLiteral("not_requested"));
}

TEST(AerialTriangulationPipelineTest, PrefersRigidPhotogrammetricNetworkOverMoreWeakPoints)
{
    const auto sparseQuality = [](int pointCount,
                                  int twoViewTrackCount,
                                  double medianAngleDeg,
                                  double gridCoverage)
    {
        return QJsonObject{
            {QStringLiteral("point_count"), pointCount},
            {QStringLiteral("two_view_track_count"), twoViewTrackCount},
            {QStringLiteral("triangulation_angle"),
             QJsonObject{{QStringLiteral("count"), pointCount},
                         {QStringLiteral("p50"), medianAngleDeg}}},
            {QStringLiteral("observation_grid_coverage"),
             QJsonObject{{QStringLiteral("mean"), gridCoverage}}},
        };
    };
    const auto attemptRunner = [sparseQuality](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 8;
        execution.result.numPoints3D = 400;
        execution.result.meanReprojError = 0.8;
        if (std::abs(input.estimatedFocalScale - 1.2) < 1.0e-9)
        {
            execution.result.numRegisteredImages = 16;
            execution.result.numPoints3D = 1800;
            execution.result.meanReprojError = 0.35;
            execution.result.sfmDiagnostics.insert(
                QStringLiteral("sparse_quality"), sparseQuality(1800, 1500, 2.0, 0.05));
        }
        else if (std::abs(input.estimatedFocalScale - 2.4) < 1.0e-9)
        {
            execution.result.numRegisteredImages = 16;
            execution.result.numPoints3D = 1300;
            execution.result.meanReprojError = 0.55;
            execution.result.sfmDiagnostics.insert(
                QStringLiteral("sparse_quality"), sparseQuality(1300, 300, 12.0, 0.22));
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     2.4);
}

TEST(AerialTriangulationPipelineTest, RunsSfmAndWritesPreparedReconstruction)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    ASSERT_TRUE(QImage(640, 480, QImage::Format_Grayscale8).save(imageA));
    ASSERT_TRUE(QImage(640, 480, QImage::Format_Grayscale8).save(imageB));

    xjw::Camera cameraA;
    cameraA.setIntrinsics(700.0, 700.0, 320.0, 240.0);
    cameraA.setPose({1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0},
                    {-0.5, 0.0, 0.0});
    xjw::Camera cameraB = cameraA;
    cameraB.setCameraCenter({0.5, 0.0, 0.0});
    const QString cameraAPath = QDir(tempDir.path()).filePath(QStringLiteral("a.tsai"));
    const QString cameraBPath = QDir(tempDir.path()).filePath(QStringLiteral("b.tsai"));
    ASSERT_TRUE(cameraA.saveToFile(cameraAPath.toStdString()));
    ASSERT_TRUE(cameraB.saveToFile(cameraBPath.toStdString()));

    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("tie_points.json"));
    writeKnownPoseTiePoints(tiePointPath, imageA, imageB, cameraA, cameraB);

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {imageA, imageB};
    input.cameraPaths = {cameraAPath, cameraBPath};
    input.tiePointPath = tiePointPath;
    input.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("output"));

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline().run(input);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.numRegisteredImages, 2);
    EXPECT_GE(result.numPoints3D, 20);
    EXPECT_TRUE(result.sfmDiagnostics.value(
        QStringLiteral("focal_search_shared_tie_point_graph")).toBool());
    EXPECT_GE(result.sfmDiagnostics.value(
        QStringLiteral("tie_point_graph_prepare_seconds")).toDouble(), 0.0);
    EXPECT_GT(result.sfmDiagnostics.value(
        QStringLiteral("tie_point_graph_track_count")).toInt(), 0);
    EXPECT_GT(result.sfmDiagnostics.value(
        QStringLiteral("tie_point_graph_pair_count")).toInt(), 0);
    EXPECT_TRUE(QFile::exists(result.sparseCloudPath));
}

TEST(AerialTriangulationPipelineTest, FlagsParallelCurvedBlockWithoutAbsoluteControl)
{
    using xjw::aerial_triangulation::AerialTriangulationPipeline;
    EXPECT_TRUE(AerialTriangulationPipeline::shouldFlagAerialDomingRisk(
        true, 0.9706, 0.0740, false));
    EXPECT_FALSE(AerialTriangulationPipeline::shouldFlagAerialDomingRisk(
        true, 0.9706, 0.0740, true));
    EXPECT_FALSE(AerialTriangulationPipeline::shouldFlagAerialDomingRisk(
        true, 0.75, 0.0740, false));
    EXPECT_FALSE(AerialTriangulationPipeline::shouldFlagAerialDomingRisk(
        true, 0.9706, 0.02, false));
    EXPECT_FALSE(AerialTriangulationPipeline::shouldFlagAerialDomingRisk(
        false, 0.9706, 0.0740, false));
}

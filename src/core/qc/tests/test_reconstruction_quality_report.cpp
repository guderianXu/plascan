#include "ReconstructionQualityReport.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <gtest/gtest.h>

using xjw::qc::ReconstructionQualityReport;

namespace
{

QJsonObject makeImage(const QString &path, bool registered)
{
    QJsonObject image;
    image[QStringLiteral("path")] = path;
    image[QStringLiteral("registered")] = registered;
    return image;
}

} // namespace

TEST(ReconstructionQualityReport, BuildsSummaryFromProjectMetadata)
{
    QJsonObject sparseQuality;
    sparseQuality[QStringLiteral("registered_image_count")] = 3;
    sparseQuality[QStringLiteral("total_image_count")] = 4;
    sparseQuality[QStringLiteral("point_count")] = 1200;
    sparseQuality[QStringLiteral("mean_reprojection_error_px")] = 0.42;
    sparseQuality[QStringLiteral("track_length_histogram")] = QJsonObject{
        {QStringLiteral("2"), 300},
        {QStringLiteral("3"), 900}
    };
    sparseQuality[QStringLiteral("reprojection_error")] = QJsonObject{
        {QStringLiteral("median"), 0.35},
        {QStringLiteral("p95"), 0.9}
    };
    sparseQuality[QStringLiteral("triangulation_angle")] = QJsonObject{
        {QStringLiteral("median"), 8.5},
        {QStringLiteral("p95"), 15.0}
    };

    QJsonObject sfmDiagnostics;
    sfmDiagnostics[QStringLiteral("sparse_quality")] = sparseQuality;
    sfmDiagnostics[QStringLiteral("ba_summary")] = QJsonObject{
        {QStringLiteral("rms_before_px"), 1.8},
        {QStringLiteral("rms_after_px"), 0.6}
    };

    QJsonObject atResult;
    atResult[QStringLiteral("path")] = QStringLiteral("E:/tmp/sfm_sparse.ply");
    atResult[QStringLiteral("sfm_diagnostics")] = sfmDiagnostics;

    QJsonObject depth0;
    depth0[QStringLiteral("status")] = QStringLiteral("completed");
    depth0[QStringLiteral("valid_ratio")] = 0.75;
    depth0[QStringLiteral("ref_image")] = QStringLiteral("img_001.jpg");

    QJsonObject depth1;
    depth1[QStringLiteral("status")] = QStringLiteral("failed");
    depth1[QStringLiteral("valid_ratio")] = 0.0;
    depth1[QStringLiteral("ref_image")] = QStringLiteral("img_002.jpg");

    QJsonObject demResult;
    demResult[QStringLiteral("path")] = QStringLiteral("E:/tmp/dem.tif");
    demResult[QStringLiteral("coverage_ratio")] = 0.81;

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        makeImage(QStringLiteral("img_001.jpg"), true),
        makeImage(QStringLiteral("img_002.jpg"), true),
        makeImage(QStringLiteral("img_003.jpg"), true),
        makeImage(QStringLiteral("img_004.jpg"), false)
    };
    meta[QStringLiteral("at_results")] = QJsonArray{atResult};
    meta[QStringLiteral("depth_map_results")] = QJsonArray{depth0, depth1};
    meta[QStringLiteral("dem_results")] = QJsonArray{demResult};
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("point_count"), 4567}}
    };

    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);

    EXPECT_EQ(report.value(QStringLiteral("total_image_count")).toInt(), 4);
    EXPECT_EQ(report.value(QStringLiteral("registered_image_count")).toInt(), 3);
    EXPECT_EQ(report.value(QStringLiteral("unregistered_image_count")).toInt(), 1);
    EXPECT_EQ(report.value(QStringLiteral("sparse_point_count")).toInt(), 1200);
    EXPECT_EQ(report.value(QStringLiteral("dense_point_count")).toInt(), 4567);
    EXPECT_NEAR(report.value(QStringLiteral("mean_reprojection_error_px")).toDouble(), 0.42, 1e-9);
    EXPECT_NEAR(report.value(QStringLiteral("mvs_valid_coverage")).toDouble(), 0.75, 1e-9);
    EXPECT_NEAR(report.value(QStringLiteral("dem_coverage")).toDouble(), 0.81, 1e-9);
    EXPECT_TRUE(report.value(QStringLiteral("track_length_histogram")).isObject());
    EXPECT_TRUE(report.value(QStringLiteral("ba_summary")).isObject());
}

TEST(ReconstructionQualityReport, ReadsCanonicalNestedLegacyAndComputedDepthCoverage)
{
    const QJsonObject canonical{
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("valid_coverage"), 0.8}};
    const QJsonObject nested{
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("depth_quality"),
         QJsonObject{{QStringLiteral("valid_coverage"), 0.6}}}};
    const QJsonObject legacy{
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("valid_ratio"), 0.4}};
    const QJsonObject computed{
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("valid_pixel_count"), 25},
        {QStringLiteral("grid_width"), 10},
        {QStringLiteral("grid_height"), 10}};
    const QJsonObject meta{
        {QStringLiteral("depth_map_results"), QJsonArray{canonical, nested, legacy, computed}}};

    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);
    EXPECT_NEAR(report.value(QStringLiteral("mvs_valid_coverage")).toDouble(), 0.5125, 1e-9);
}

TEST(ReconstructionQualityReport, LeavesCoverageUnavailableWhenNoFrameHasAMeasurement)
{
    const QJsonObject meta{
        {QStringLiteral("depth_map_results"),
         QJsonArray{QJsonObject{{QStringLiteral("status"), QStringLiteral("completed")}}}}};

    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);
    EXPECT_FALSE(report.contains(QStringLiteral("mvs_valid_coverage")));
}

TEST(ReconstructionQualityReport, UsesCurrentWorkflowArtifactsInsteadOfStaleReportRecords)
{
    QJsonObject staleReport;
    staleReport[QStringLiteral("type")] = QStringLiteral("reconstruction_quality");
    staleReport[QStringLiteral("registered_image_count")] = 0;
    staleReport[QStringLiteral("sparse_point_count")] = 0;
    staleReport[QStringLiteral("dense_point_count")] = 0;

    QJsonObject sparseQuality;
    sparseQuality[QStringLiteral("registered_image_count")] = 444;
    sparseQuality[QStringLiteral("total_image_count")] = 444;
    sparseQuality[QStringLiteral("point_count")] = 588257;
    sparseQuality[QStringLiteral("two_view_ratio")] = 0.23;

    QJsonObject sfmDiagnostics;
    sfmDiagnostics[QStringLiteral("sparse_quality")] = sparseQuality;
    sfmDiagnostics[QStringLiteral("ba_summary")] = QJsonObject{
        {QStringLiteral("rms_after_px"), 0.74}
    };

    QJsonObject currentSparse;
    currentSparse[QStringLiteral("result_kind")] = QStringLiteral("sfm_sparse_reconstruction");
    currentSparse[QStringLiteral("sparse_point_count")] = 588257;
    currentSparse[QStringLiteral("sfm_diagnostics")] = sfmDiagnostics;

    QJsonObject meta;
    meta[QStringLiteral("project_files")] = QJsonObject{
        {QStringLiteral("images"), QJsonArray{
             makeImage(QStringLiteral("img_001.jpg"), true),
             makeImage(QStringLiteral("img_002.jpg"), true),
             makeImage(QStringLiteral("img_003.jpg"), true)
         }}
    };
    meta[QStringLiteral("report_results")] = QJsonArray{staleReport};
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{currentSparse};
    meta[QStringLiteral("dense_cloud_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("point_count"), 1058511291}}};

    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);

    EXPECT_EQ(report.value(QStringLiteral("total_image_count")).toInt(), 444);
    EXPECT_EQ(report.value(QStringLiteral("registered_image_count")).toInt(), 444);
    EXPECT_EQ(report.value(QStringLiteral("sparse_point_count")).toInt(), 588257);
    EXPECT_EQ(report.value(QStringLiteral("dense_point_count")).toInt(), 1058511291);
    EXPECT_NEAR(report.value(QStringLiteral("mean_reprojection_error_px")).toDouble(), 0.0, 1e-9);
    EXPECT_EQ(report.value(QStringLiteral("ba_summary")).toObject().value(QStringLiteral("rms_after_px")).toDouble(),
              0.74);
}

TEST(ReconstructionQualityReport, SummarizesSurveyControlResiduals)
{
    QJsonObject control0;
    control0[QStringLiteral("id")] = QStringLiteral("GCP001");
    control0[QStringLiteral("enabled")] = true;
    control0[QStringLiteral("residual")] = QJsonObject{
        {QStringLiteral("horizontal_m"), 0.03},
        {QStringLiteral("vertical_m"), 0.04}
    };

    QJsonObject control1;
    control1[QStringLiteral("id")] = QStringLiteral("GCP002");
    control1[QStringLiteral("enabled")] = true;
    control1[QStringLiteral("residual")] = QJsonObject{
        {QStringLiteral("total_m"), 0.05}
    };

    QJsonObject check0;
    check0[QStringLiteral("id")] = QStringLiteral("CHK001");
    check0[QStringLiteral("enabled")] = true;
    check0[QStringLiteral("residual")] = QJsonObject{
        {QStringLiteral("total_m"), 0.12}
    };

    QJsonObject scaleBar0;
    scaleBar0[QStringLiteral("id")] = QStringLiteral("SB001");
    scaleBar0[QStringLiteral("enabled")] = true;
    scaleBar0[QStringLiteral("residual_m")] = -0.02;

    QJsonObject thresholds;
    thresholds[QStringLiteral("checkpoint_rmse_warn_m")] = 0.10;
    thresholds[QStringLiteral("scale_bar_rmse_warn_m")] = 0.05;

    QJsonObject survey;
    survey[QStringLiteral("control_points")] = QJsonArray{control0, control1};
    survey[QStringLiteral("check_points")] = QJsonArray{check0};
    survey[QStringLiteral("scale_bars")] = QJsonArray{scaleBar0};
    survey[QStringLiteral("quality_thresholds")] = thresholds;

    QJsonObject meta;
    meta[QStringLiteral("survey_control")] = survey;

    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);
    const QJsonObject surveyReport = report.value(QStringLiteral("survey_control")).toObject();

    EXPECT_EQ(surveyReport.value(QStringLiteral("control_point_count")).toInt(), 2);
    EXPECT_EQ(surveyReport.value(QStringLiteral("check_point_count")).toInt(), 1);
    EXPECT_EQ(surveyReport.value(QStringLiteral("scale_bar_count")).toInt(), 1);
    EXPECT_NEAR(surveyReport.value(QStringLiteral("control_point_rmse_m")).toDouble(), 0.05, 1e-9);
    EXPECT_NEAR(surveyReport.value(QStringLiteral("check_point_rmse_m")).toDouble(), 0.12, 1e-9);
    EXPECT_NEAR(surveyReport.value(QStringLiteral("scale_bar_rmse_m")).toDouble(), 0.02, 1e-9);
    EXPECT_EQ(surveyReport.value(QStringLiteral("status")).toString(), QStringLiteral("warn"));

    EXPECT_EQ(report.value(QStringLiteral("control_point_count")).toInt(), 2);
    EXPECT_EQ(report.value(QStringLiteral("check_point_count")).toInt(), 1);
    EXPECT_EQ(report.value(QStringLiteral("scale_bar_count")).toInt(), 1);
}

TEST(ReconstructionQualityReport, WritesJsonAndCsvReport)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        makeImage(QStringLiteral("img_001.jpg"), true),
        makeImage(QStringLiteral("img_002.jpg"), false)
    };

    const auto result = ReconstructionQualityReport::writeFromProjectMeta(
        meta,
        tempDir.path(),
        QStringLiteral("quality_report"));

    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_TRUE(QFile::exists(result.jsonPath));
    EXPECT_TRUE(QFile::exists(result.csvPath));

    QFile jsonFile(result.jsonPath);
    ASSERT_TRUE(jsonFile.open(QIODevice::ReadOnly));
    const QJsonObject saved = QJsonDocument::fromJson(jsonFile.readAll()).object();
    EXPECT_EQ(saved.value(QStringLiteral("total_image_count")).toInt(), 2);

    QFile csvFile(result.csvPath);
    ASSERT_TRUE(csvFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString csv = QString::fromUtf8(csvFile.readAll());
    EXPECT_TRUE(csv.contains(QStringLiteral("metric,value")));
    EXPECT_TRUE(csv.contains(QStringLiteral("total_image_count,2")));
}

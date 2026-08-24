#include "RpcStereoDemGenerator.h"

#include "RpcGeospatialSupport.h"

#include "DemDomIO.h"
#include "DemGenerator.h"
#include "DemGridAggregator.h"
#include "RpcCameraIO.h"
#include "RpcStereoIntersection.h"
#include "io/ImageIO.h"
#include "io/PathIO.h"

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

#include <opencv2/imgcodecs.hpp>

#include <plamatrix/dense/dense_matrix.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace xjw
{
    namespace
    {

        struct StereoPoint
        {
            RpcCameraModel::GeodeticCoordinate geodetic{};
            double reprojectionErrorPixels = 0.0;
            float intensity = 0.0f;
        };

        void
        reportProgress(const RpcStereoDemGenerator::ProgressCallback& callback, const QString& message, int percent)
        {
            if (callback)
            {
                callback(message, percent);
            }
        }

        bool cancellationRequested(const std::atomic_bool* cancelFlag, QString* errorMessage)
        {
            if (!cancelFlag || !cancelFlag->load(std::memory_order_relaxed))
            {
                return false;
            }
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("RPC 立体 DEM 生成已取消");
            }
            return true;
        }

        double median(std::vector<double> values)
        {
            if (values.empty())
            {
                return 0.0;
            }
            const std::size_t middle = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + middle, values.end());
            const double upper = values[middle];
            if (values.size() % 2 != 0)
            {
                return upper;
            }
            std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
            return 0.5 * (upper + values[middle - 1]);
        }

        bool validateOptions(const RpcStereoDemOptions& options, QString* errorMessage)
        {
            if (options.maximumFeatures < 100 || options.descriptorRatio <= 0.0 || options.descriptorRatio >= 1.0 ||
                options.fundamentalRansacThresholdPixels <= 0.0 || options.maximumReprojectionErrorPixels <= 0.0 ||
                options.minimumAcceptedPoints < 20 || options.gridResolutionMeters <= 0.0)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("RPC 立体 DEM 参数无效");
                }
                return false;
            }
            return true;
        }

        bool loadCamera(const QString& path, RpcCameraModel* camera, QString* errorMessage)
        {
            std::string error;
            if (!loadRpcCameraFromRaster(common::io::toUtf8Path(path), camera, &error))
            {
                if (errorMessage)
                {
                    *errorMessage =
                        QStringLiteral("读取 RPC 相机失败 (%1): %2").arg(path, QString::fromUtf8(error.c_str()));
                }
                return false;
            }
            return true;
        }

        std::vector<cv::DMatch>
        mutualRatioMatches(const cv::Mat& leftDescriptors, const cv::Mat& rightDescriptors, double ratio)
        {
            cv::BFMatcher matcher(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> forward;
            std::vector<std::vector<cv::DMatch>> reverse;
            matcher.knnMatch(leftDescriptors, rightDescriptors, forward, 2);
            matcher.knnMatch(rightDescriptors, leftDescriptors, reverse, 2);

            std::vector<int> reverseBest(static_cast<std::size_t>(rightDescriptors.rows), -1);
            for (const auto& candidates : reverse)
            {
                if (candidates.size() >= 2 && candidates[0].distance < ratio * candidates[1].distance)
                {
                    reverseBest[static_cast<std::size_t>(candidates[0].queryIdx)] = candidates[0].trainIdx;
                }
            }

            std::vector<cv::DMatch> matches;
            matches.reserve(forward.size());
            for (const auto& candidates : forward)
            {
                if (candidates.size() < 2 || candidates[0].distance >= ratio * candidates[1].distance)
                {
                    continue;
                }
                const cv::DMatch& match = candidates[0];
                if (match.trainIdx >= 0 && match.trainIdx < static_cast<int>(reverseBest.size()) &&
                    reverseBest[static_cast<std::size_t>(match.trainIdx)] == match.queryIdx)
                {
                    matches.push_back(match);
                }
            }
            return matches;
        }

        bool writeJson(const QString& path, const QJsonObject& object, QString* errorMessage)
        {
            QSaveFile file(path);
            if (!file.open(QIODevice::WriteOnly) ||
                file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) < 0 || !file.commit())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("写入立体 DEM 报告失败: %1").arg(path);
                }
                return false;
            }
            return true;
        }

    } // namespace

    bool RpcStereoDemGenerator::generate(const QString& leftImagePath,
                                         const QString& rightImagePath,
                                         const QString& outputDirectory,
                                         const RpcStereoDemOptions& options,
                                         QJsonObject* result,
                                         QString* errorMessage,
                                         const ProgressCallback& progress,
                                         const std::atomic_bool* cancelFlag)
    {
        if (result)
        {
            *result = {};
        }
        if (!validateOptions(options, errorMessage) || !QFileInfo::exists(leftImagePath) ||
            !QFileInfo::exists(rightImagePath) || !QDir().mkpath(outputDirectory))
        {
            if (errorMessage && errorMessage->isEmpty())
            {
                *errorMessage = QStringLiteral("RPC 立体影像或输出目录无效");
            }
            return false;
        }

        if (cancellationRequested(cancelFlag, errorMessage))
        {
            return false;
        }
        reportProgress(progress, QStringLiteral("读取 RPC 相机与影像"), 5);
        RpcCameraModel leftCamera;
        RpcCameraModel rightCamera;
        if (!loadCamera(leftImagePath, &leftCamera, errorMessage) ||
            !loadCamera(rightImagePath, &rightCamera, errorMessage))
        {
            return false;
        }
        QString imageError;
        const cv::Mat leftImage = common::io::readImage(leftImagePath, cv::IMREAD_GRAYSCALE, &imageError);
        const cv::Mat rightImage = common::io::readImage(rightImagePath, cv::IMREAD_GRAYSCALE, &imageError);
        if (leftImage.empty() || rightImage.empty() || leftImage.depth() != CV_8U || rightImage.depth() != CV_8U)
        {
            if (errorMessage)
            {
                *errorMessage =
                    imageError.isEmpty() ? QStringLiteral("RPC 立体影像必须可读取为 8 位灰度图") : imageError;
            }
            return false;
        }

        reportProgress(progress, QStringLiteral("提取并匹配 SIFT 同名点"), 15);
        const cv::Ptr<cv::SIFT> sift = cv::SIFT::create(options.maximumFeatures);
        std::vector<cv::KeyPoint> leftKeypoints;
        std::vector<cv::KeyPoint> rightKeypoints;
        cv::Mat leftDescriptors;
        cv::Mat rightDescriptors;
        sift->detectAndCompute(leftImage, cv::noArray(), leftKeypoints, leftDescriptors);
        sift->detectAndCompute(rightImage, cv::noArray(), rightKeypoints, rightDescriptors);
        if (cancellationRequested(cancelFlag, errorMessage))
        {
            return false;
        }
        if (leftDescriptors.empty() || rightDescriptors.empty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("影像纹理不足，无法提取 SIFT 描述子");
            }
            return false;
        }

        const std::vector<cv::DMatch> matches =
            mutualRatioMatches(leftDescriptors, rightDescriptors, options.descriptorRatio);
        if (matches.size() < static_cast<std::size_t>(options.minimumAcceptedPoints))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("通过双向比值检验的同名点不足: %1").arg(matches.size());
            }
            return false;
        }

        std::vector<cv::Point2f> leftPoints;
        std::vector<cv::Point2f> rightPoints;
        leftPoints.reserve(matches.size());
        rightPoints.reserve(matches.size());
        for (const cv::DMatch& match : matches)
        {
            leftPoints.push_back(leftKeypoints[static_cast<std::size_t>(match.queryIdx)].pt);
            rightPoints.push_back(rightKeypoints[static_cast<std::size_t>(match.trainIdx)].pt);
        }
        cv::Mat inlierMask;
        cv::findFundamentalMat(
            leftPoints, rightPoints, cv::FM_RANSAC, options.fundamentalRansacThresholdPixels, 0.999, inlierMask);
        if (inlierMask.empty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("基础矩阵估计失败");
            }
            return false;
        }

        reportProgress(progress, QStringLiteral("执行 RPC 双像前方交会"), 35);
        RpcStereoIntersectionOptions intersectionOptions;
        intersectionOptions.pixelTolerance = 1.0e-4;
        intersectionOptions.positionToleranceMeters = 1.0e-3;
        intersectionOptions.maximumIterations = 40;
        std::vector<StereoPoint> stereoPoints;
        stereoPoints.reserve(matches.size());
        int fundamentalInliers = 0;
        for (std::size_t index = 0; index < matches.size(); ++index)
        {
            if (index % 64 == 0 && cancellationRequested(cancelFlag, errorMessage))
            {
                return false;
            }
            if (inlierMask.at<uchar>(static_cast<int>(index)) == 0)
            {
                continue;
            }
            ++fundamentalInliers;
            const cv::Point2f& left = leftPoints[index];
            const cv::Point2f& right = rightPoints[index];
            RpcStereoIntersectionResult intersection;
            const bool converged = intersectRpcObservations(
                leftCamera, {left.x, left.y}, rightCamera, {right.x, right.y}, &intersection, intersectionOptions);
            if ((!converged && intersection.iterations <= 0) || !std::isfinite(intersection.reprojectionRmsPixels) ||
                intersection.reprojectionRmsPixels > options.maximumReprojectionErrorPixels ||
                std::hypot(intersection.ecefMeters[0],
                           std::hypot(intersection.ecefMeters[1], intersection.ecefMeters[2])) < 1.0e6)
            {
                continue;
            }
            StereoPoint point;
            point.geodetic = intersection.geodetic;
            point.reprojectionErrorPixels = intersection.reprojectionRmsPixels;
            const int intensityRow = std::clamp(cvRound(left.y), 0, leftImage.rows - 1);
            const int intensityCol = std::clamp(cvRound(left.x), 0, leftImage.cols - 1);
            point.intensity = leftImage.at<uchar>(intensityRow, intensityCol);
            stereoPoints.push_back(point);
        }

        if (stereoPoints.size() < static_cast<std::size_t>(options.minimumAcceptedPoints))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("RPC 前方交会有效点不足: %1").arg(stereoPoints.size());
            }
            return false;
        }

        std::vector<double> heights;
        heights.reserve(stereoPoints.size());
        for (const StereoPoint& point : stereoPoints)
        {
            heights.push_back(point.geodetic[2]);
        }
        const double medianHeight = median(heights);
        std::vector<double> deviations;
        deviations.reserve(heights.size());
        for (double height : heights)
        {
            deviations.push_back(std::abs(height - medianHeight));
        }
        const double normalizedMad = 1.4826 * median(deviations);
        const double heightTolerance = std::max(50.0, 6.0 * normalizedMad);
        std::erase_if(stereoPoints,
                      [&](const StereoPoint& point)
                      {
                          return !std::isfinite(point.geodetic[0]) || !std::isfinite(point.geodetic[1]) ||
                                 !std::isfinite(point.geodetic[2]) ||
                                 std::abs(point.geodetic[2] - medianHeight) > heightTolerance;
                      });
        if (stereoPoints.size() < static_cast<std::size_t>(options.minimumAcceptedPoints))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("高程鲁棒过滤后有效点不足: %1").arg(stereoPoints.size());
            }
            return false;
        }

        std::vector<double> longitudes;
        std::vector<double> latitudes;
        std::vector<RpcCameraModel::GeodeticCoordinate> geodetic;
        longitudes.reserve(stereoPoints.size());
        latitudes.reserve(stereoPoints.size());
        geodetic.reserve(stereoPoints.size());
        for (const StereoPoint& point : stereoPoints)
        {
            longitudes.push_back(point.geodetic[0]);
            latitudes.push_back(point.geodetic[1]);
            geodetic.push_back(point.geodetic);
        }

        stereo_dem::ProjectedCoordinateSystem coordinateSystem;
        if (!stereo_dem::createLocalUtm(median(longitudes), median(latitudes), &coordinateSystem, errorMessage))
        {
            return false;
        }
        std::vector<std::array<double, 3>> projected;
        if (!stereo_dem::geodeticToProjected(geodetic, coordinateSystem, &projected, errorMessage))
        {
            return false;
        }

        if (cancellationRequested(cancelFlag, errorMessage))
        {
            return false;
        }
        reportProgress(progress, QStringLiteral("栅格化摄影测量点云"), 70);
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> coordinates(projected.size(), 3);
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(projected.size(), 3);
        for (std::size_t index = 0; index < projected.size(); ++index)
        {
            coordinates(index, 0) = static_cast<float>(projected[index][0]);
            coordinates(index, 1) = static_cast<float>(projected[index][1]);
            coordinates(index, 2) = static_cast<float>(projected[index][2]);
            const uint8_t value = cv::saturate_cast<uint8_t>(stereoPoints[index].intensity);
            colors(index, 0) = value;
            colors(index, 1) = value;
            colors(index, 2) = value;
        }
        PlaPointCloud cloud(std::move(coordinates));
        cloud.setColors(std::move(colors));

        DemGenerationOptions demOptions;
        demOptions.gridResolution = options.gridResolutionMeters;
        demOptions.holeFillIterations = std::max(0, options.holeFillIterations);
        demOptions.holeFillMinNeighbors = 3;
        demOptions.holeFillSearchRadius = 6;
        demOptions.elevationAggregation = DemGenerationOptions::ElevationAggregation::WeightedAverage;
        demOptions.generateDenseCloud = false;
        demOptions.generateMesh = false;
        demOptions.projection.coordinateSystem = coordinateSystem.name;
        demOptions.projection.projectionWkt = coordinateSystem.wkt;
        demOptions.projection.metadata.insert(QStringLiteral("BAND_UNIT"), QStringLiteral("m"));
        demOptions.projection.metadata.insert(QStringLiteral("VERTICAL_REFERENCE"),
                                              QStringLiteral("WGS84 ellipsoidal height"));
        demOptions.projection.metadata.insert(QStringLiteral("SOURCE_CAMERA_MODEL"), QStringLiteral("RPC00B"));

        DemGridData dem;
        if (!DemGenerator::generateFromPointCloud(cloud, demOptions, &dem, nullptr, errorMessage))
        {
            return false;
        }

        std::vector<DemGridSample> samples;
        samples.reserve(projected.size());
        for (std::size_t index = 0; index < projected.size(); ++index)
        {
            DemGridSample sample;
            sample.col = qRound((projected[index][0] - dem.minX) / dem.stepX);
            sample.row = qRound((projected[index][1] - dem.minY) / dem.stepY);
            sample.elevation = static_cast<float>(projected[index][2]);
            sample.triangulationError = static_cast<float>(stereoPoints[index].reprojectionErrorPixels);
            sample.confidence =
                static_cast<float>(1.0 / (1.0 + std::pow(stereoPoints[index].reprojectionErrorPixels, 2.0)));
            samples.push_back(sample);
        }
        DemGridData quality;
        if (DemGridAggregator::aggregateSamples(dem.width,
                                                dem.height,
                                                samples,
                                                DemGenerationOptions::ElevationAggregation::WeightedAverage,
                                                &quality,
                                                errorMessage))
        {
            dem.pointCount = quality.pointCount;
            dem.confidence = quality.confidence;
            dem.triangulationError = quality.triangulationError;
            dem.coverageMask = quality.validMask;
            const cv::Mat noDirectObservation = quality.validMask == 0;
            dem.confidence.setTo(-1.0f, noDirectObservation);
            dem.triangulationError.setTo(-9999.0f, noDirectObservation);
        }

        if (cancellationRequested(cancelFlag, errorMessage))
        {
            return false;
        }
        reportProgress(progress, QStringLiteral("写出 DEM 与质量栅格"), 88);
        const QDir outputDir(outputDirectory);
        const QString demPath = outputDir.filePath(QStringLiteral("dem.tif"));
        const QString previewPath = outputDir.filePath(QStringLiteral("dem_preview.png"));
        const QString cloudPath = outputDir.filePath(QStringLiteral("stereo_points.ply"));
        int cloudPointCount = 0;
        DemGridData cloudDem = dem;
        cloudDem.triangulationError.release();
        if (!DemDomIO::writeDenseCloudPly(cloudDem, cloudPath, &cloudPointCount, errorMessage) ||
            !DemDomIO::writeDemPreviewPng(dem, previewPath, errorMessage))
        {
            return false;
        }
        DemGridData rasterDem = dem;
        rasterDem.worldX.release();
        rasterDem.worldY.release();
        if (!DemDomIO::writeDemRaster(rasterDem, demPath, DemRasterFormat::Float32Tiff, errorMessage))
        {
            return false;
        }
        DemQualityArtifacts artifacts;
        if (!DemDomIO::writeDemQualityRasters(dem, outputDirectory, &artifacts, errorMessage))
        {
            return false;
        }

        std::vector<double> acceptedErrors;
        acceptedErrors.reserve(stereoPoints.size());
        for (const StereoPoint& point : stereoPoints)
        {
            acceptedErrors.push_back(point.reprojectionErrorPixels);
        }
        const int directObservationCells = dem.coverageMask.empty() ? 0 : cv::countNonZero(dem.coverageMask);
        QJsonObject output{{QStringLiteral("schema"), QStringLiteral("plascan.rpc_stereo_dem.v1")},
                           {QStringLiteral("created_at"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
                           {QStringLiteral("left_image"), QFileInfo(leftImagePath).absoluteFilePath()},
                           {QStringLiteral("right_image"), QFileInfo(rightImagePath).absoluteFilePath()},
                           {QStringLiteral("dem_path"), QFileInfo(demPath).absoluteFilePath()},
                           {QStringLiteral("preview_path"), QFileInfo(previewPath).absoluteFilePath()},
                           {QStringLiteral("point_cloud_path"), QFileInfo(cloudPath).absoluteFilePath()},
                           {QStringLiteral("error_path"), artifacts.errorPath},
                           {QStringLiteral("count_path"), artifacts.countPath},
                           {QStringLiteral("confidence_path"), artifacts.confidencePath},
                           {QStringLiteral("coverage_path"), artifacts.coveragePath},
                           {QStringLiteral("coordinate_system"), coordinateSystem.name},
                           {QStringLiteral("grid_resolution_m"), dem.stepX},
                           {QStringLiteral("width"), dem.width},
                           {QStringLiteral("height"), dem.height},
                           {QStringLiteral("left_keypoints"), static_cast<qint64>(leftKeypoints.size())},
                           {QStringLiteral("right_keypoints"), static_cast<qint64>(rightKeypoints.size())},
                           {QStringLiteral("mutual_ratio_matches"), static_cast<qint64>(matches.size())},
                           {QStringLiteral("fundamental_inliers"), fundamentalInliers},
                           {QStringLiteral("accepted_stereo_points"), static_cast<qint64>(stereoPoints.size())},
                           {QStringLiteral("median_reprojection_error_px"), median(acceptedErrors)},
                           {QStringLiteral("direct_observation_cells"), directObservationCells},
                           {QStringLiteral("direct_coverage_fraction"),
                            static_cast<double>(directObservationCells) / static_cast<double>(dem.width * dem.height)},
                           {QStringLiteral("raster_point_count"), cloudPointCount},
                           {QStringLiteral("median_ellipsoidal_height_m"), medianHeight},
                           {QStringLiteral("height_nmad_m"), normalizedMad}};
        const QString reportPath = outputDir.filePath(QStringLiteral("stereo_dem_report.json"));
        output.insert(QStringLiteral("report_path"), QFileInfo(reportPath).absoluteFilePath());
        if (!writeJson(reportPath, output, errorMessage))
        {
            return false;
        }
        if (result)
        {
            *result = output;
        }
        reportProgress(progress, QStringLiteral("RPC 立体 DEM 生成完成"), 100);
        return true;
    }

} // namespace xjw

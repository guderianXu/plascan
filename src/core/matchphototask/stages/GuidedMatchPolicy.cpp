#include "GuidedMatchPolicy.h"

#include "GeometryVerifyStage.h"
#include "MatchGeometryVerifier.h"
#include "ReferencePoseEpipolarGeometry.h"

#include <QDir>
#include <QFileInfo>

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw::matchphotos
{
    namespace
    {

        QString normalizedPath(const QString& path)
        {
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
        }

        const FramePinholeCamera* findReferenceCamera(const GuidedMatchPolicyCache& cache, const QString& imagePath)
        {
            auto exact = cache.referenceCamerasByPath.constFind(imagePath);
            if (exact != cache.referenceCamerasByPath.cend())
            {
                return &exact.value();
            }
            const auto normalized = cache.referenceCamerasByPath.constFind(normalizedPath(imagePath));
            return normalized == cache.referenceCamerasByPath.cend() ? nullptr : &normalized.value();
        }

        bool validFundamental(const std::array<double, 9>& values)
        {
            double squaredNorm = 0.0;
            for (const double value : values)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
                squaredNorm += value * value;
            }
            if (squaredNorm <= 1.0e-24)
            {
                return false;
            }
            const cv::Matx33d fundamental(
                values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]);
            const double norm = std::sqrt(squaredNorm);
            return std::abs(cv::determinant(fundamental)) / (norm * norm * norm) <= 1.0e-3;
        }

        struct ResidualSummary
        {
            std::size_t count = 0;
            double median = -1.0;
            double p90 = -1.0;
        };

        ResidualSummary summarizeResiduals(std::vector<double> values)
        {
            ResidualSummary summary;
            summary.count = values.size();
            if (values.empty())
            {
                return summary;
            }
            const std::size_t median_index =
                static_cast<std::size_t>(std::llround(0.5 * static_cast<double>(values.size() - 1)));
            std::nth_element(values.begin(), values.begin() + median_index, values.end());
            summary.median = values[median_index];
            const std::size_t p90_index =
                static_cast<std::size_t>(std::llround(0.9 * static_cast<double>(values.size() - 1)));
            std::nth_element(values.begin(), values.begin() + p90_index, values.end());
            summary.p90 = values[p90_index];
            return summary;
        }

        std::vector<double> residualsForFundamental(const image_matching::PairMatchData& pair,
                                                    const std::array<double, 9>& fundamental,
                                                    bool geometryInliersOnly)
        {
            std::vector<double> residuals;
            residuals.reserve(pair.correspondences.size());
            for (const auto& correspondence : pair.correspondences)
            {
                if (geometryInliersOnly &&
                    !image_matching::hasFlag(correspondence.flags, image_matching::MatchRecordFlag::GeometryInlier))
                {
                    continue;
                }
                const double residual = epipolarSampsonDistance(fundamental,
                                                                correspondence.observation0.x,
                                                                correspondence.observation0.y,
                                                                correspondence.observation1.x,
                                                                correspondence.observation1.y);
                if (std::isfinite(residual))
                {
                    residuals.push_back(residual);
                }
            }
            return residuals;
        }

        bool isHealthyPair(const MatchPhotosContext& context,
                           const MatchPhotosOptions& options,
                           const MatchPhotosMatchRecord& record)
        {
            if (!record.pairData || !record.passedGeometry)
            {
                return false;
            }
            const bool has_cached_metrics = record.settings.contains(QStringLiteral("geometry_inlier_ratio")) &&
                                            record.settings.contains(QStringLiteral("geometry_grid_coverage_image0")) &&
                                            record.settings.contains(QStringLiteral("geometry_grid_coverage_image1"));
            GeometryQualityMetrics metrics;
            if (has_cached_metrics)
            {
                metrics.inlierCount = record.geometricInlierCount;
                metrics.inlierRatio = record.settings.value(QStringLiteral("geometry_inlier_ratio")).toDouble();
                metrics.image0GridCoverage =
                    record.settings.value(QStringLiteral("geometry_grid_coverage_image0")).toDouble();
                metrics.image1GridCoverage =
                    record.settings.value(QStringLiteral("geometry_grid_coverage_image1")).toDouble();
            }
            else
            {
                const bool adjacent = areSequenceAdjacent(context, options, record.image0Path, record.image1Path);
                metrics = measureGeometryQuality(*record.pairData, options, adjacent);
            }
            const double coverage = std::min(metrics.image0GridCoverage, metrics.image1GridCoverage);
            const int healthyInliers = std::max(80, options.geometryMinInliers * 4);
            const double healthyRatio = std::max(0.45, options.geometryMinInlierRatio * 1.75);
            const double healthyCoverage = std::max(0.25, options.geometryMinGridCoverage * 1.75);
            return metrics.inlierCount >= healthyInliers && metrics.inlierRatio >= healthyRatio &&
                   coverage >= healthyCoverage;
        }

        image_matching::MatchResult matchResultForPair(const image_matching::PairMatchData& pair)
        {
            image_matching::MatchResult matches;
            matches.cvMatches.reserve(pair.correspondences.size());
            for (const auto& correspondence : pair.correspondences)
            {
                matches.cvMatches.emplace_back(static_cast<int>(correspondence.observation0.featureId),
                                               static_cast<int>(correspondence.observation1.featureId),
                                               1.0f - correspondence.confidence);
            }
            matches.numMatches = static_cast<int>(matches.cvMatches.size());
            return matches;
        }

        bool homographyDominates(const MatchPhotosOptions& options,
                                 const MatchPhotosMatchRecord& record,
                                 const image_matching::FeatureSet& features0,
                                 const image_matching::FeatureSet& features1)
        {
            if (!record.pairData || record.geometricInlierCount <= 0 ||
                record.geometricInlierCount >= std::max(100, options.geometryMinInliers * 5))
            {
                return false;
            }
            image_matching::MatchGeometryOptions homographyOptions;
            homographyOptions.model = image_matching::GeometryModel::Homography;
            homographyOptions.reprojectionThresholdPixels = options.geometryReprojThreshold;
            homographyOptions.minimumInliers = 4;
            homographyOptions.maximumIterations = std::max(100, options.geometryMaxIterations);
            homographyOptions.confidence = 0.9999;
            const image_matching::MatchGeometryResult homography = image_matching::MatchGeometryVerifier::verify(
                matchResultForPair(*record.pairData), features0, features1, homographyOptions);
            return homography.modelEstimated && static_cast<double>(homography.inlierCount) >=
                                                    0.95 * static_cast<double>(record.geometricInlierCount);
        }

    } // namespace

    GuidedMatchPolicyCache buildGuidedMatchPolicyCache(const MatchPhotosContext& context)
    {
        GuidedMatchPolicyCache cache;
        cache.referenceCamerasByPath.reserve(context.referenceCameras.size() * 2);
        for (auto it = context.referenceCameras.cbegin(); it != context.referenceCameras.cend(); ++it)
        {
            cache.referenceCamerasByPath.insert(it.key(), it.value());
            cache.referenceCamerasByPath.insert(normalizedPath(it.key()), it.value());
        }
        return cache;
    }

    GuidedMatchGeometryChoice chooseGuidedMatchGeometry(const MatchPhotosContext& context,
                                                        const MatchPhotosOptions& options,
                                                        const GuidedMatchPolicyCache& cache,
                                                        const MatchPhotosMatchRecord& record,
                                                        const image_matching::FeatureSet& features0,
                                                        const image_matching::FeatureSet& features1)
    {
        GuidedMatchGeometryChoice choice;
        if (!record.pairData)
        {
            choice.skipReason = QStringLiteral("pair_data_missing");
            return choice;
        }

        const bool estimatedReliable = record.passedGeometry &&
                                       record.pairData->geometryModel == image_matching::GeometryModel::Fundamental &&
                                       validFundamental(record.pairData->geometryMatrix);
        ResidualSummary estimated_summary;
        if (estimatedReliable)
        {
            estimated_summary =
                summarizeResiduals(residualsForFundamental(*record.pairData, record.pairData->geometryMatrix, true));
            choice.estimatedMedianResidualPixels = estimated_summary.median;
        }

        ReferencePoseEpipolarGeometry referenceGeometry;
        if (options.guidedUseReferenceCameraPoses)
        {
            const FramePinholeCamera* camera0 = findReferenceCamera(cache, record.image0Path);
            const FramePinholeCamera* camera1 = findReferenceCamera(cache, record.image1Path);
            if (camera0 && camera1)
            {
                referenceGeometry = fundamentalFromReferenceCameras(*camera0, *camera1);
            }
        }

        bool referenceConsistent = referenceGeometry.valid;
        ResidualSummary reference_summary;
        if (referenceGeometry.valid && record.pairData->correspondences.size() >= 8)
        {
            reference_summary = summarizeResiduals(
                residualsForFundamental(*record.pairData, referenceGeometry.fundamental, estimatedReliable));
            choice.referenceMedianResidualPixels = reference_summary.median;
            referenceConsistent =
                reference_summary.count >= 8 &&
                choice.referenceMedianResidualPixels <= std::max(2.0, options.geometryReprojThreshold * 2.0);
        }

        if (estimatedReliable && referenceGeometry.valid && !referenceConsistent)
        {
            choice.referenceConflict = true;
        }

        if (referenceConsistent &&
            (!estimatedReliable || choice.estimatedMedianResidualPixels < 0.0 ||
             choice.referenceMedianResidualPixels < 0.0 ||
             choice.referenceMedianResidualPixels <= choice.estimatedMedianResidualPixels * 1.25 + 0.25))
        {
            choice.fundamental = referenceGeometry.fundamental;
            choice.geometrySource = QStringLiteral("reference_pose");
        }
        else if (estimatedReliable)
        {
            choice.fundamental = record.pairData->geometryMatrix;
            choice.geometrySource = QStringLiteral("estimated_fundamental");
        }
        else
        {
            choice.skipReason = referenceGeometry.valid ? QStringLiteral("reference_pose_inconsistent")
                                                        : QStringLiteral("fundamental_unavailable");
            return choice;
        }

        if (options.guidedMatchingMode == GuidedMatchingMode::Automatic)
        {
            if (isHealthyPair(context, options, record))
            {
                choice.autoSkippedHealthy = true;
                choice.skipReason = QStringLiteral("healthy_pair");
                return choice;
            }
            if (choice.geometrySource == QLatin1String("estimated_fundamental") &&
                homographyDominates(options, record, features0, features1))
            {
                choice.homographyDominant = true;
                choice.skipReason = QStringLiteral("homography_dominant");
                return choice;
            }
        }

        const double residualP90 =
            choice.geometrySource == QLatin1String("reference_pose") ? reference_summary.p90 : estimated_summary.p90;
        const double adaptiveThreshold = residualP90 >= 0.0 ? 2.5 * residualP90 : 2.0 * options.geometryReprojThreshold;
        choice.epipolarThresholdPixels = std::clamp(std::max(2.0, adaptiveThreshold), 2.0, 8.0);
        choice.eligible = true;
        return choice;
    }

} // namespace xjw::matchphotos

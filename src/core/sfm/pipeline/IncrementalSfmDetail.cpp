#include "IncrementalSfmDetail.h"

#include "Intersection.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace xjw::incremental_sfm_detail
{

bool shouldEvaluateMultipleInitialPairModels(const IncrementalSfmOptions &options,
                                             int totalImages,
                                             std::size_t candidateCount)
{
    return options.autoSelectInitPair &&
           options.evaluateMultipleInitialPairModels &&
           candidateCount > 1 &&
           totalImages >= 3 &&
           totalImages <= std::max(2, options.multiInitialPairMaxImages);
}

double scoreInitialPairTrial(const IncrementalSfmResult &result, int totalImages)
{
    const int registered = std::max(0, result.numRegisteredImages);
    const int missing = std::max(0, totalImages - registered);
    const double reprojPenalty = std::isfinite(result.meanReprojError)
        ? result.meanReprojError * 1000.0
        : 1000000.0;
    return static_cast<double>(registered) * 1000000000.0 -
           static_cast<double>(missing) * 1000000.0 +
           static_cast<double>(std::max(0, result.numPoints3D)) -
           reprojPenalty;
}

int effectivePnpMinTrackLength(const IncrementalSfmOptions &options, std::size_t registeredImageCount)
{
    if (registeredImageCount < 3)
    {
        return 2;
    }
    return std::max(2, options.pnpMinTrackLength);
}

bool pointUsableForPnp(const SfmReconstruction &reconstruction,
                       Point3DId pointId,
                       int minTrackLength)
{
    if (pointId == kInvalidPoint3DId || !reconstruction.hasPoint3D(pointId))
    {
        return false;
    }
    const ScenePoint3D &point = reconstruction.point3D(pointId);
    return point.track.length() >= static_cast<std::size_t>(std::max(2, minTrackLength));
}

double distance3d(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double percentile(std::vector<double> values, double ratio)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const double clampedRatio = std::max(0.0, std::min(1.0, ratio));
    const auto index = static_cast<size_t>(std::round(clampedRatio * static_cast<double>(values.size() - 1)));
    return values[index];
}

std::array<double, 4> rotationToQuaternion(const std::array<double, 9> &rotation)
{
    std::array<double, 4> quaternion{}; // w, x, y, z
    const double trace = rotation[0] + rotation[4] + rotation[8];
    if (trace > 0.0)
    {
        const double scale = std::sqrt(trace + 1.0) * 2.0;
        quaternion = {{0.25 * scale,
                       (rotation[7] - rotation[5]) / scale,
                       (rotation[2] - rotation[6]) / scale,
                       (rotation[3] - rotation[1]) / scale}};
    }
    else if (rotation[0] > rotation[4] && rotation[0] > rotation[8])
    {
        const double scale = std::sqrt(1.0 + rotation[0] - rotation[4] - rotation[8]) * 2.0;
        quaternion = {{(rotation[7] - rotation[5]) / scale,
                       0.25 * scale,
                       (rotation[1] + rotation[3]) / scale,
                       (rotation[2] + rotation[6]) / scale}};
    }
    else if (rotation[4] > rotation[8])
    {
        const double scale = std::sqrt(1.0 + rotation[4] - rotation[0] - rotation[8]) * 2.0;
        quaternion = {{(rotation[2] - rotation[6]) / scale,
                       (rotation[1] + rotation[3]) / scale,
                       0.25 * scale,
                       (rotation[5] + rotation[7]) / scale}};
    }
    else
    {
        const double scale = std::sqrt(1.0 + rotation[8] - rotation[0] - rotation[4]) * 2.0;
        quaternion = {{(rotation[3] - rotation[1]) / scale,
                       (rotation[2] + rotation[6]) / scale,
                       (rotation[5] + rotation[7]) / scale,
                       0.25 * scale}};
    }

    const double norm = std::sqrt(std::inner_product(quaternion.begin(),
                                                     quaternion.end(),
                                                     quaternion.begin(),
                                                     0.0));
    if (norm > 1e-12)
    {
        for (double &value : quaternion)
        {
            value /= norm;
        }
    }
    return quaternion;
}

std::array<double, 9> quaternionToRotation(const std::array<double, 4> &quaternion)
{
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    return {{1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
             2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
             2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)}};
}

std::array<double, 9> interpolateCameraRotation(const std::array<double, 9> &rotationA,
                                                const std::array<double, 9> &rotationB,
                                                double ratio)
{
    std::array<double, 4> quaternionA = rotationToQuaternion(rotationA);
    std::array<double, 4> quaternionB = rotationToQuaternion(rotationB);
    double dot = std::inner_product(quaternionA.begin(),
                                    quaternionA.end(),
                                    quaternionB.begin(),
                                    0.0);
    if (dot < 0.0)
    {
        dot = -dot;
        for (double &value : quaternionB)
        {
            value = -value;
        }
    }

    // 区间内用于 SLERP；单侧序列恢复允许最多再外推一个帧间旋转。
    const double t = std::clamp(ratio, -2.0, 2.0);
    std::array<double, 4> interpolated{};
    if (dot > 0.9995)
    {
        for (std::size_t index = 0; index < interpolated.size(); ++index)
        {
            interpolated[index] = quaternionA[index] + t * (quaternionB[index] - quaternionA[index]);
        }
    }
    else
    {
        const double theta = std::acos(std::clamp(dot, -1.0, 1.0));
        const double sinTheta = std::sin(theta);
        const double weightA = std::sin((1.0 - t) * theta) / sinTheta;
        const double weightB = std::sin(t * theta) / sinTheta;
        for (std::size_t index = 0; index < interpolated.size(); ++index)
        {
            interpolated[index] = weightA * quaternionA[index] + weightB * quaternionB[index];
        }
    }

    const double norm = std::sqrt(std::inner_product(interpolated.begin(),
                                                     interpolated.end(),
                                                     interpolated.begin(),
                                                     0.0));
    if (norm > 1e-12)
    {
        for (double &value : interpolated)
        {
            value /= norm;
        }
    }
    return quaternionToRotation(interpolated);
}

KnownPoseTriangulationPolicy resolveKnownPoseTriangulationPolicy(
    const std::shared_ptr<SfmReconstruction> &reconstruction,
    const CorrespondenceGraph &correspondenceGraph,
    const std::vector<ImageId> &imageIds,
    const IncrementalSfmOptions &options)
{
    KnownPoseTriangulationPolicy policy;
    policy.triangulatorOptions = options.triangulatorOptions;
    policy.filterMinTriAngle = options.filterMinTriAngle;
    policy.chosenMinTriAngle = options.triangulatorOptions.minTriAngle;

    constexpr int minUsefulTriangulations = 20;
    constexpr double relaxedMinTriAngle = 0.1;
    constexpr double absoluteMinTriAngle = 0.05;

    std::vector<double> validAngles;
    validAngles.reserve(1024);

    for (ImageId imageId : imageIds)
    {
        if (!reconstruction->isRegistered(imageId) || !reconstruction->hasCamera(imageId))
        {
            continue;
        }

        const ImageData &image = reconstruction->image(imageId);
        const Camera &camera = reconstruction->camera(imageId);

        for (FeatureIdx featureIdx = 0; featureIdx < static_cast<FeatureIdx>(image.keypoints.size()); ++featureIdx)
        {
            const auto correspondences = correspondenceGraph.findCorrespondences(imageId, featureIdx);
            for (const auto &correspondence : correspondences)
            {
                if (correspondence.imageId <= imageId ||
                    !reconstruction->isRegistered(correspondence.imageId) ||
                    !reconstruction->hasCamera(correspondence.imageId))
                {
                    continue;
                }

                const ImageData &otherImage = reconstruction->image(correspondence.imageId);
                if (correspondence.featureIdx >= otherImage.keypoints.size())
                {
                    continue;
                }

                const Camera &otherCamera = reconstruction->camera(correspondence.imageId);
                const auto &keypoint = image.keypoints[featureIdx];
                const auto &otherKeypoint = otherImage.keypoints[correspondence.featureIdx];
                const auto triResult = Intersection::intersectPair(camera,
                                                                    keypoint.x,
                                                                    keypoint.y,
                                                                    otherCamera,
                                                                    otherKeypoint.x,
                                                                    otherKeypoint.y);
                if (!triResult.valid ||
                    !std::isfinite(triResult.angle_deg) ||
                    !std::isfinite(triResult.reproj_error_rms) ||
                    triResult.reproj_error_rms > options.triangulatorOptions.maxReprojError)
                {
                    continue;
                }

                validAngles.push_back(triResult.angle_deg);
            }
        }
    }

    policy.validCandidates = static_cast<int>(validAngles.size());
    policy.acceptedWithDefault = static_cast<int>(std::count_if(validAngles.begin(),
                                                                validAngles.end(),
                                                                [&](double angle)
                                                                {
                                                                    return angle >= options.triangulatorOptions.minTriAngle;
                                                                }));

    if (policy.acceptedWithDefault >= minUsefulTriangulations ||
        policy.validCandidates < minUsefulTriangulations)
    {
        policy.acceptedWithAdapted = policy.acceptedWithDefault;
        return policy;
    }

    int acceptedWithRelaxed = static_cast<int>(std::count_if(validAngles.begin(),
                                                            validAngles.end(),
                                                            [](double angle)
                                                            {
                                                                return angle >= relaxedMinTriAngle;
                                                            }));

    double adaptedMinTriAngle = relaxedMinTriAngle;
    if (acceptedWithRelaxed < minUsefulTriangulations)
    {
        adaptedMinTriAngle = std::max(absoluteMinTriAngle, percentile(validAngles, 0.20) * 0.8);
        adaptedMinTriAngle = std::min(adaptedMinTriAngle, relaxedMinTriAngle);
    }

    policy.triangulatorOptions.minTriAngle = std::min(options.triangulatorOptions.minTriAngle, adaptedMinTriAngle);
    policy.filterMinTriAngle = std::min(options.filterMinTriAngle, policy.triangulatorOptions.minTriAngle);
    policy.chosenMinTriAngle = policy.triangulatorOptions.minTriAngle;
    policy.adapted = policy.chosenMinTriAngle < options.triangulatorOptions.minTriAngle;
    policy.acceptedWithAdapted = static_cast<int>(std::count_if(validAngles.begin(),
                                                                validAngles.end(),
                                                                [&](double angle)
                                                                {
                                                                    return angle >= policy.chosenMinTriAngle;
                                                                }));

    return policy;
}

bool knownPoseMatchPassesGeometry(const SfmReconstruction &reconstruction,
                                  ImageId imageId,
                                  ImageId otherImageId,
                                  const FeatureMatch &match,
                                  const TriangulatorOptions &options)
{
    if (!reconstruction.hasCamera(imageId) ||
        !reconstruction.hasCamera(otherImageId) ||
        !reconstruction.hasImage(imageId) ||
        !reconstruction.hasImage(otherImageId))
    {
        return false;
    }

    const ImageData &image = reconstruction.image(imageId);
    const ImageData &otherImage = reconstruction.image(otherImageId);
    if (match.idx1 >= image.keypoints.size() || match.idx2 >= otherImage.keypoints.size())
    {
        return false;
    }

    const Camera &camera = reconstruction.camera(imageId);
    const Camera &otherCamera = reconstruction.camera(otherImageId);
    const FeatureKeypoint &keypoint = image.keypoints[match.idx1];
    const FeatureKeypoint &otherKeypoint = otherImage.keypoints[match.idx2];
    const auto triResult = Intersection::intersectPair(camera,
                                                       keypoint.x,
                                                       keypoint.y,
                                                       otherCamera,
                                                       otherKeypoint.x,
                                                       otherKeypoint.y);
    if (!triResult.valid ||
        !std::isfinite(triResult.angle_deg) ||
        !std::isfinite(triResult.reproj_error_rms))
    {
        return false;
    }

    return triResult.angle_deg >= options.minTriAngle &&
           triResult.reproj_error_rms <= options.maxReprojError;
}

std::array<double, 3> transformPoint(const SimilarityTransform3d &transform,
                                     const std::array<double, 3> &point)
{
    std::array<double, 3> out{};
    for (int r = 0; r < 3; ++r)
    {
        out[static_cast<size_t>(r)] = transform.translation[static_cast<size_t>(r)];
        for (int c = 0; c < 3; ++c)
        {
            out[static_cast<size_t>(r)] +=
                transform.scale *
                transform.rotation[static_cast<size_t>(r * 3 + c)] *
                point[static_cast<size_t>(c)];
        }
    }
    return out;
}

std::array<double, 9> multiplyRotation(const std::array<double, 9> &left,
                                       const std::array<double, 9> &right)
{
    std::array<double, 9> result{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            double value = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                value += left[static_cast<size_t>(r * 3 + k)] *
                         right[static_cast<size_t>(k * 3 + c)];
            }
            result[static_cast<size_t>(r * 3 + c)] = value;
        }
    }
    return result;
}

double pointDistance(const std::array<double, 3> &a,
                     const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double centerExtent(const std::vector<std::array<double, 3>> &points)
{
    double extent = 0.0;
    for (size_t i = 0; i < points.size(); ++i)
    {
        for (size_t j = i + 1; j < points.size(); ++j)
        {
            extent = std::max(extent, pointDistance(points[i], points[j]));
        }
    }
    return extent;
}

SimilarityTransform3d estimateSimilarityUmeyama(const std::vector<std::array<double, 3>> &source,
                                                const std::vector<std::array<double, 3>> &target)
{
    SimilarityTransform3d transform;
    if (source.size() != target.size() || source.size() < 3)
    {
        return transform;
    }

    std::array<double, 3> sourceMean{{0.0, 0.0, 0.0}};
    std::array<double, 3> targetMean{{0.0, 0.0, 0.0}};
    for (size_t i = 0; i < source.size(); ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            sourceMean[static_cast<size_t>(k)] += source[i][static_cast<size_t>(k)];
            targetMean[static_cast<size_t>(k)] += target[i][static_cast<size_t>(k)];
        }
    }
    for (int k = 0; k < 3; ++k)
    {
        sourceMean[static_cast<size_t>(k)] /= static_cast<double>(source.size());
        targetMean[static_cast<size_t>(k)] /= static_cast<double>(target.size());
    }

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    double sourceVariance = 0.0;
    for (size_t i = 0; i < source.size(); ++i)
    {
        std::array<double, 3> srcCentered{};
        std::array<double, 3> dstCentered{};
        for (int k = 0; k < 3; ++k)
        {
            srcCentered[static_cast<size_t>(k)] =
                source[i][static_cast<size_t>(k)] - sourceMean[static_cast<size_t>(k)];
            dstCentered[static_cast<size_t>(k)] =
                target[i][static_cast<size_t>(k)] - targetMean[static_cast<size_t>(k)];
            sourceVariance += srcCentered[static_cast<size_t>(k)] * srcCentered[static_cast<size_t>(k)];
        }

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                covariance.at<double>(r, c) +=
                    dstCentered[static_cast<size_t>(r)] * srcCentered[static_cast<size_t>(c)];
            }
        }
    }

    sourceVariance /= static_cast<double>(source.size());
    if (!(sourceVariance > 1e-18))
    {
        return transform;
    }
    covariance /= static_cast<double>(source.size());

    cv::SVD svd(covariance, cv::SVD::FULL_UV);
    cv::Mat sign = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat rotation = svd.u * svd.vt;
    if (cv::determinant(rotation) < 0.0)
    {
        sign.at<double>(2, 2) = -1.0;
        rotation = svd.u * sign * svd.vt;
    }

    double scaleNumerator = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        scaleNumerator += svd.w.at<double>(i) * sign.at<double>(i, i);
    }
    const double scale = scaleNumerator / sourceVariance;
    if (!(scale > 0.0) || !std::isfinite(scale))
    {
        return transform;
    }

    transform.valid = true;
    transform.scale = scale;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            transform.rotation[static_cast<size_t>(r * 3 + c)] = rotation.at<double>(r, c);
        }
    }

    const auto mappedSourceMean = transformPoint(transform, sourceMean);
    for (int k = 0; k < 3; ++k)
    {
        transform.translation[static_cast<size_t>(k)] +=
            targetMean[static_cast<size_t>(k)] - mappedSourceMean[static_cast<size_t>(k)];
    }

    double sum2 = 0.0;
    for (size_t i = 0; i < source.size(); ++i)
    {
        const double residual = pointDistance(transformPoint(transform, source[i]), target[i]);
        sum2 += residual * residual;
    }
    transform.inlierCount = static_cast<int>(source.size());
    transform.rmse = std::sqrt(sum2 / static_cast<double>(source.size()));
    return transform;
}

SimilarityTransform3d estimateRobustCameraCenterSimilarity(
    const std::vector<std::array<double, 3>> &source,
    const std::vector<std::array<double, 3>> &target)
{
    SimilarityTransform3d best;
    if (source.size() != target.size() || source.size() < 3)
    {
        return best;
    }

    const double extent = std::max(centerExtent(source), centerExtent(target));
    const double inlierThreshold = std::max(1e-4, extent * 0.03);
    constexpr int maxSamples = 256;
    int sampleCount = 0;

    for (size_t i = 0; i + 2 < source.size() && sampleCount < maxSamples; ++i)
    {
        for (size_t j = i + 1; j + 1 < source.size() && sampleCount < maxSamples; ++j)
        {
            for (size_t k = j + 1; k < source.size() && sampleCount < maxSamples; ++k)
            {
                ++sampleCount;
                std::vector<std::array<double, 3>> sampleSource{source[i], source[j], source[k]};
                std::vector<std::array<double, 3>> sampleTarget{target[i], target[j], target[k]};
                const SimilarityTransform3d candidate = estimateSimilarityUmeyama(sampleSource, sampleTarget);
                if (!candidate.valid)
                {
                    continue;
                }

                std::vector<std::array<double, 3>> inlierSource;
                std::vector<std::array<double, 3>> inlierTarget;
                inlierSource.reserve(source.size());
                inlierTarget.reserve(target.size());
                double sum2 = 0.0;
                for (size_t idx = 0; idx < source.size(); ++idx)
                {
                    const double residual = pointDistance(transformPoint(candidate, source[idx]), target[idx]);
                    if (residual <= inlierThreshold)
                    {
                        inlierSource.push_back(source[idx]);
                        inlierTarget.push_back(target[idx]);
                        sum2 += residual * residual;
                    }
                }

                if (inlierSource.size() < 3)
                {
                    continue;
                }

                const double rmse = std::sqrt(sum2 / static_cast<double>(inlierSource.size()));
                if (!best.valid ||
                    inlierSource.size() > static_cast<size_t>(best.inlierCount) ||
                    (inlierSource.size() == static_cast<size_t>(best.inlierCount) && rmse < best.rmse))
                {
                    best = estimateSimilarityUmeyama(inlierSource, inlierTarget);
                    best.inlierCount = static_cast<int>(inlierSource.size());
                    best.rmse = rmse;
                }
            }
        }
    }

    if (!best.valid)
    {
        best = estimateSimilarityUmeyama(source, target);
    }
    return best;
}


} // namespace xjw::incremental_sfm_detail

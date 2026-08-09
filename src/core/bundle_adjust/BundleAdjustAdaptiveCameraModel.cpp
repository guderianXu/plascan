#include "BundleAdjustAdaptiveCameraModel.h"
#include "BundleAdjustValidation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

namespace xjw
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr int kRadiusHistogramBins = 160;
constexpr double kMaximumHistogramRadius = 2.0;

using InformationMatrix = std::array<
    std::array<double, kBAIntrinsicParameterCount>,
    kBAIntrinsicParameterCount>;
using CalibrationPointMatrix = std::array<
    std::array<double, 3>,
    kBAIntrinsicParameterCount>;
using PointInformationMatrix = std::array<std::array<double, 3>, 3>;

std::size_t parameterIndex(BAIntrinsicParameter parameter)
{
    return static_cast<std::size_t>(parameter);
}

double clampScore(double value)
{
    return std::clamp(std::isfinite(value) ? value : 0.0, 0.0, 1.0);
}

double balancedRatio(int negativeCount, int positiveCount)
{
    const int maximum = std::max(negativeCount, positiveCount);
    if (maximum <= 0)
    {
        return 0.0;
    }
    return static_cast<double>(std::min(negativeCount, positiveCount)) /
           static_cast<double>(maximum);
}

bool invertRegularizedPointInformation(
    PointInformationMatrix matrix,
    PointInformationMatrix *inverse)
{
    if (!inverse)
    {
        return false;
    }
    const double scale = std::max({
        std::abs(matrix[0][0]),
        std::abs(matrix[1][1]),
        std::abs(matrix[2][2]),
        1.0e-12});
    const double regularization = scale * 1.0e-10;
    for (int axis = 0; axis < 3; ++axis)
    {
        matrix[static_cast<std::size_t>(axis)]
              [static_cast<std::size_t>(axis)] += regularization;
    }

    const double a = matrix[0][0];
    const double b = matrix[0][1];
    const double c = matrix[0][2];
    const double d = matrix[1][1];
    const double e = matrix[1][2];
    const double f = matrix[2][2];
    const double determinant =
        a * (d * f - e * e) -
        b * (b * f - c * e) +
        c * (b * e - c * d);
    if (!std::isfinite(determinant) ||
        determinant <= scale * scale * scale * 1.0e-18)
    {
        return false;
    }
    const double reciprocal = 1.0 / determinant;
    (*inverse)[0] = {{
        (d * f - e * e) * reciprocal,
        (c * e - b * f) * reciprocal,
        (b * e - c * d) * reciprocal}};
    (*inverse)[1] = {{
        (*inverse)[0][1],
        (a * f - c * c) * reciprocal,
        (b * c - a * e) * reciprocal}};
    (*inverse)[2] = {{
        (*inverse)[0][2],
        (*inverse)[1][2],
        (a * d - b * b) * reciprocal}};
    return true;
}

bool pointProjectionJacobian(
    const Camera &camera,
    const double *cameraPoint,
    double x,
    double y,
    std::array<double, 3> *derivativeU,
    std::array<double, 3> *derivativeV)
{
    if (!cameraPoint || !derivativeU || !derivativeV ||
        !std::isfinite(cameraPoint[2]) ||
        std::abs(cameraPoint[2]) <= 1.0e-12)
    {
        return false;
    }
    const Camera::Distortion distortion = camera.distortion();
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double radial = 1.0 + distortion.radialK1 * r2 +
                          distortion.radialK2 * r4 +
                          distortion.radialK3 * r4 * r2;
    const double radialSlope = distortion.radialK1 +
                               2.0 * distortion.radialK2 * r2 +
                               3.0 * distortion.radialK3 * r4;
    const double radialX = 2.0 * x * radialSlope;
    const double radialY = 2.0 * y * radialSlope;
    const double distortedXByX =
        radial + x * radialX + 2.0 * distortion.tangentialP1 * y +
        6.0 * distortion.tangentialP2 * x;
    const double distortedXByY =
        x * radialY + 2.0 * distortion.tangentialP1 * x +
        2.0 * distortion.tangentialP2 * y;
    const double distortedYByX =
        y * radialX + 2.0 * distortion.tangentialP1 * x +
        2.0 * distortion.tangentialP2 * y;
    const double distortedYByY =
        radial + y * radialY + 6.0 * distortion.tangentialP1 * y +
        2.0 * distortion.tangentialP2 * x;
    const double inverseDepth = 1.0 / cameraPoint[2];
    const std::array<double, 3> normalizedXByCameraPoint{{
        inverseDepth, 0.0, -x * inverseDepth}};
    const std::array<double, 3> normalizedYByCameraPoint{{
        0.0, inverseDepth, -y * inverseDepth}};
    const auto rotation = camera.cameraToWorldRotation();
    for (int worldAxis = 0; worldAxis < 3; ++worldAxis)
    {
        double normalizedXByWorld = 0.0;
        double normalizedYByWorld = 0.0;
        for (int cameraAxis = 0; cameraAxis < 3; ++cameraAxis)
        {
            const double cameraByWorld = rotation[
                static_cast<std::size_t>(worldAxis * 3 + cameraAxis)];
            normalizedXByWorld += normalizedXByCameraPoint[
                static_cast<std::size_t>(cameraAxis)] * cameraByWorld;
            normalizedYByWorld += normalizedYByCameraPoint[
                static_cast<std::size_t>(cameraAxis)] * cameraByWorld;
        }
        (*derivativeU)[static_cast<std::size_t>(worldAxis)] =
            static_cast<double>(camera.uAxisSign()) * camera.focalX() *
            (distortedXByX * normalizedXByWorld +
             distortedXByY * normalizedYByWorld);
        (*derivativeV)[static_cast<std::size_t>(worldAxis)] =
            static_cast<double>(camera.vAxisSign()) * camera.focalY() *
            (distortedYByX * normalizedXByWorld +
             distortedYByY * normalizedYByWorld);
    }
    return true;
}

double finiteMedian(std::vector<double> values)
{
    values.erase(
        std::remove_if(values.begin(), values.end(), [](double value)
        {
            return !std::isfinite(value);
        }),
        values.end());
    if (values.empty())
    {
        return 0.0;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2 != 0)
    {
        return values[middle];
    }
    const double upper = values[middle];
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return 0.5 * (values[middle - 1] + upper);
}

double radiusQuantile(
    const std::array<int, kRadiusHistogramBins> &histogram,
    int sampleCount,
    double quantile)
{
    if (sampleCount <= 0)
    {
        return 0.0;
    }
    const int target = std::max(
        1,
        static_cast<int>(std::ceil(clampScore(quantile) * sampleCount)));
    int accumulated = 0;
    for (int bin = 0; bin < kRadiusHistogramBins; ++bin)
    {
        accumulated += histogram[static_cast<std::size_t>(bin)];
        if (accumulated >= target)
        {
            return (static_cast<double>(bin) + 0.5) *
                   kMaximumHistogramRadius /
                   static_cast<double>(kRadiusHistogramBins);
        }
    }
    return kMaximumHistogramRadius;
}

bool solveSmallLinearSystem(
    InformationMatrix matrix,
    std::array<double, kBAIntrinsicParameterCount> rightHandSide,
    int size,
    std::array<double, kBAIntrinsicParameterCount> *solution)
{
    if (!solution || size <= 0 ||
        size > static_cast<int>(kBAIntrinsicParameterCount))
    {
        return false;
    }
    solution->fill(0.0);
    for (int column = 0; column < size; ++column)
    {
        int pivot = column;
        double pivotMagnitude = std::abs(
            matrix[static_cast<std::size_t>(column)]
                  [static_cast<std::size_t>(column)]);
        for (int row = column + 1; row < size; ++row)
        {
            const double magnitude = std::abs(
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(column)]);
            if (magnitude > pivotMagnitude)
            {
                pivot = row;
                pivotMagnitude = magnitude;
            }
        }
        if (!std::isfinite(pivotMagnitude) || pivotMagnitude <= 1.0e-12)
        {
            return false;
        }
        if (pivot != column)
        {
            std::swap(matrix[static_cast<std::size_t>(pivot)],
                      matrix[static_cast<std::size_t>(column)]);
            std::swap(rightHandSide[static_cast<std::size_t>(pivot)],
                      rightHandSide[static_cast<std::size_t>(column)]);
        }
        const double diagonal =
            matrix[static_cast<std::size_t>(column)]
                  [static_cast<std::size_t>(column)];
        for (int entry = column; entry < size; ++entry)
        {
            matrix[static_cast<std::size_t>(column)]
                  [static_cast<std::size_t>(entry)] /= diagonal;
        }
        rightHandSide[static_cast<std::size_t>(column)] /= diagonal;
        for (int row = 0; row < size; ++row)
        {
            if (row == column)
            {
                continue;
            }
            const double factor =
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(column)];
            for (int entry = column; entry < size; ++entry)
            {
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(entry)] -=
                    factor * matrix[static_cast<std::size_t>(column)]
                                   [static_cast<std::size_t>(entry)];
            }
            rightHandSide[static_cast<std::size_t>(row)] -=
                factor * rightHandSide[static_cast<std::size_t>(column)];
        }
    }
    for (int index = 0; index < size; ++index)
    {
        (*solution)[static_cast<std::size_t>(index)] =
            rightHandSide[static_cast<std::size_t>(index)];
    }
    return true;
}

std::array<double, kBAIntrinsicParameterCount> incrementalInformationScore(
    const InformationMatrix &normalMatrix)
{
    InformationMatrix correlation{};
    for (std::size_t row = 0; row < kBAIntrinsicParameterCount; ++row)
    {
        for (std::size_t column = 0; column < kBAIntrinsicParameterCount; ++column)
        {
            const double denominator = std::sqrt(
                std::max(0.0, normalMatrix[row][row]) *
                std::max(0.0, normalMatrix[column][column]));
            correlation[row][column] = denominator > 1.0e-15
                ? normalMatrix[row][column] / denominator
                : 0.0;
        }
    }

    // 低阶、稳定参数先进入条件信息矩阵。后续参数只有提供此前模型不能解释的
    // 独立像面形变时才获得高分，避免 k2/k3 在窄半径覆盖下同时漂移。
    constexpr std::array<BAIntrinsicParameter, kBAIntrinsicParameterCount> order{{
        BAIntrinsicParameter::FocalLength,
        BAIntrinsicParameter::RadialK1,
        BAIntrinsicParameter::FocalAspectRatio,
        BAIntrinsicParameter::PrincipalPointX,
        BAIntrinsicParameter::PrincipalPointY,
        BAIntrinsicParameter::RadialK2,
        BAIntrinsicParameter::TangentialP1,
        BAIntrinsicParameter::TangentialP2,
        BAIntrinsicParameter::RadialK3,
    }};

    std::array<double, kBAIntrinsicParameterCount> result{};
    for (std::size_t position = 0; position < order.size(); ++position)
    {
        const std::size_t current = parameterIndex(order[position]);
        if (normalMatrix[current][current] <= 1.0e-15)
        {
            result[current] = 0.0;
            continue;
        }
        if (position == 0)
        {
            result[current] = 1.0;
            continue;
        }

        InformationMatrix previousCorrelation{};
        std::array<double, kBAIntrinsicParameterCount> cross{};
        for (std::size_t row = 0; row < position; ++row)
        {
            const std::size_t rowParameter = parameterIndex(order[row]);
            cross[row] = correlation[rowParameter][current];
            for (std::size_t column = 0; column < position; ++column)
            {
                const std::size_t columnParameter =
                    parameterIndex(order[column]);
                previousCorrelation[row][column] =
                    correlation[rowParameter][columnParameter];
            }
            previousCorrelation[row][row] += 1.0e-6;
        }
        std::array<double, kBAIntrinsicParameterCount> solved{};
        if (!solveSmallLinearSystem(
                previousCorrelation,
                cross,
                static_cast<int>(position),
                &solved))
        {
            result[current] = 0.0;
            continue;
        }
        double explained = 0.0;
        for (std::size_t index = 0; index < position; ++index)
        {
            explained += cross[index] * solved[index];
        }
        result[current] = clampScore(1.0 - explained);
    }
    return result;
}

bool requestedByLegacyFlags(
    const BAOptions &options,
    BAIntrinsicParameter parameter)
{
    switch (parameter)
    {
    case BAIntrinsicParameter::FocalLength:
        return options.refineSharedFocalLength;
    case BAIntrinsicParameter::FocalAspectRatio:
        return options.refineSharedFocalAspectRatio;
    case BAIntrinsicParameter::PrincipalPointX:
    case BAIntrinsicParameter::PrincipalPointY:
        return options.refineSharedPrincipalPoint;
    case BAIntrinsicParameter::RadialK1:
        return options.refineSharedRadialDistortion;
    case BAIntrinsicParameter::RadialK2:
    case BAIntrinsicParameter::RadialK3:
    case BAIntrinsicParameter::TangentialP1:
    case BAIntrinsicParameter::TangentialP2:
        return options.refineSharedRadialDistortion &&
               options.refineSharedHighOrderDistortion;
    case BAIntrinsicParameter::Count:
        return false;
    }
    return false;
}

} // namespace

const char *baIntrinsicParameterName(BAIntrinsicParameter parameter)
{
    switch (parameter)
    {
    case BAIntrinsicParameter::FocalLength:
        return "f";
    case BAIntrinsicParameter::FocalAspectRatio:
        return "aspect";
    case BAIntrinsicParameter::PrincipalPointX:
        return "cx";
    case BAIntrinsicParameter::PrincipalPointY:
        return "cy";
    case BAIntrinsicParameter::RadialK1:
        return "k1";
    case BAIntrinsicParameter::RadialK2:
        return "k2";
    case BAIntrinsicParameter::RadialK3:
        return "k3";
    case BAIntrinsicParameter::TangentialP1:
        return "p1";
    case BAIntrinsicParameter::TangentialP2:
        return "p2";
    case BAIntrinsicParameter::Count:
        return "unknown";
    }
    return "unknown";
}

int enabledIntrinsicParameterCount(const BAIntrinsicParameterMask &mask)
{
    return static_cast<int>(std::count(mask.begin(), mask.end(), true));
}

std::string adaptiveCameraModelName(const BAIntrinsicParameterMask &mask)
{
    std::ostringstream stream;
    bool first = true;
    for (std::size_t index = 0; index < mask.size(); ++index)
    {
        if (!mask[index])
        {
            continue;
        }
        if (!first)
        {
            stream << '+';
        }
        first = false;
        stream << baIntrinsicParameterName(
            static_cast<BAIntrinsicParameter>(index));
    }
    return first ? "fixed" : stream.str();
}

BAAdaptiveCameraModelAssessment assessAdaptiveCameraModel(
    const std::vector<Camera> &cameras,
    const std::vector<BATrack> &tracks,
    const BAOptions *options)
{
    BAAdaptiveCameraModelAssessment result;
    result.cameraCount = static_cast<int>(cameras.size());
    result.trackCount = static_cast<int>(tracks.size());
    if (cameras.size() < 3 || tracks.empty())
    {
        result.reason = "insufficient_camera_or_track_support";
        return result;
    }
    if (options &&
        options->cameraCalibrationGroupIds.size() == cameras.size())
    {
        std::vector<int> groupIds = options->cameraCalibrationGroupIds;
        std::sort(groupIds.begin(), groupIds.end());
        groupIds.erase(
            std::unique(groupIds.begin(), groupIds.end()),
            groupIds.end());
        if (groupIds.size() > 1)
        {
            BAOptions ungroupedOptions = *options;
            ungroupedOptions.cameraCalibrationGroupIds.clear();
            BAAdaptiveCameraModelAssessment combined =
                assessAdaptiveCameraModel(
                    cameras, tracks, &ungroupedOptions);
            if (!combined.valid)
            {
                return combined;
            }
            for (const int groupId : groupIds)
            {
                std::vector<int> cameraRemap(cameras.size(), -1);
                std::vector<Camera> groupCameras;
                for (std::size_t cameraIndex = 0;
                     cameraIndex < cameras.size();
                     ++cameraIndex)
                {
                    if (options->cameraCalibrationGroupIds[cameraIndex] ==
                        groupId)
                    {
                        cameraRemap[cameraIndex] =
                            static_cast<int>(groupCameras.size());
                        groupCameras.push_back(cameras[cameraIndex]);
                    }
                }
                std::vector<BATrack> groupTracks;
                groupTracks.reserve(tracks.size());
                for (const BATrack &track : tracks)
                {
                    BATrack groupTrack = track;
                    groupTrack.observations.clear();
                    for (const BAObservation &observation : track.observations)
                    {
                        if (observation.cameraIndex < 0 ||
                            observation.cameraIndex >=
                                static_cast<int>(cameraRemap.size()))
                        {
                            continue;
                        }
                        const int remappedCamera = cameraRemap[
                            static_cast<std::size_t>(observation.cameraIndex)];
                        if (remappedCamera < 0)
                        {
                            continue;
                        }
                        BAObservation remappedObservation = observation;
                        remappedObservation.cameraIndex = remappedCamera;
                        groupTrack.observations.push_back(remappedObservation);
                    }
                    if (groupTrack.observations.size() >= 2)
                    {
                        groupTracks.push_back(std::move(groupTrack));
                    }
                }
                BAAdaptiveCameraModelAssessment groupAssessment =
                    assessAdaptiveCameraModel(
                        groupCameras, groupTracks, &ungroupedOptions);
                for (std::size_t index = 0;
                     index < kBAIntrinsicParameterCount;
                     ++index)
                {
                    combined.enabled[index] = combined.enabled[index] &&
                                              groupAssessment.enabled[index];
                    if (groupAssessment.valid ||
                        groupAssessment.enabled[index])
                    {
                        combined.reliability[index] = std::min(
                            combined.reliability[index],
                            groupAssessment.reliability[index]);
                    }
                    else
                    {
                        combined.reliability[index] = 0.0;
                    }
                }
            }
            combined.modelName = adaptiveCameraModelName(combined.enabled);
            combined.reason = "calibration_group_conservative_intersection";
            return combined;
        }
    }

    std::vector<Camera> normalizedCameras;
    normalizedCameras.reserve(cameras.size());
    std::vector<double> focalSamples;
    focalSamples.reserve(cameras.size());
    int validCameraCount = 0;
    for (const Camera &sourceCamera : cameras)
    {
        const Camera camera = sourceCamera.normalizedForPositiveDepth();
        normalizedCameras.push_back(camera);
        if (!camera.isValid() || !std::isfinite(camera.focalX()) ||
            !std::isfinite(camera.focalY()) || camera.focalX() <= 1.0e-9 ||
            camera.focalY() <= 1.0e-9)
        {
            continue;
        }
        focalSamples.push_back(0.5 * (camera.focalX() + camera.focalY()));
        ++validCameraCount;
    }
    if (validCameraCount < 3)
    {
        result.reason = "insufficient_valid_cameras";
        return result;
    }
    const double referenceFocal = std::max(1.0, finiteMedian(focalSamples));
    const std::array<double, kBAIntrinsicParameterCount> canonicalSteps{{
        0.01,
        0.01,
        referenceFocal * 0.01,
        referenceFocal * 0.01,
        0.01,
        0.02,
        0.05,
        0.002,
        0.002,
    }};

    InformationMatrix normalMatrix{};
    double totalWeight = 0.0;
    int centralObservationCount = 0;
    int peripheralObservationCount = 0;
    int leftCount = 0;
    int rightCount = 0;
    int upperCount = 0;
    int lowerCount = 0;
    std::array<bool, 8> occupiedSectors{};
    std::array<int, kRadiusHistogramBins> radiusHistogram{};
    std::vector<bool> activeCameras(cameras.size(), false);
    std::vector<double> triangulationAngles;
    triangulationAngles.reserve(tracks.size());
    int activeTrackCount = 0;

    for (const BATrack &track : tracks)
    {
        if (!std::all_of(
                track.initialPoint.begin(),
                track.initialPoint.end(),
                [](double value) { return std::isfinite(value); }))
        {
            continue;
        }
        int firstCameraIndex = -1;
        int usableObservationCount = 0;
        int initialResidualCount = 0;
        bool hasSecondDistinctCamera = false;
        bool initialPointHasPositiveDepth = true;
        double initialResidualSquared = 0.0;
        const double world[3] = {
            track.initialPoint[0],
            track.initialPoint[1],
            track.initialPoint[2]};
        for (const BAObservation &observation : track.observations)
        {
            if (!detail::observationIsUsable(
                    observation, normalizedCameras.size()))
            {
                continue;
            }
            const Camera &camera = normalizedCameras[
                static_cast<std::size_t>(observation.cameraIndex)];
            double pixel[2] = {0.0, 0.0};
            if (!camera.isValid() ||
                !camera.projectWorldPoint(world, pixel))
            {
                initialPointHasPositiveDepth = false;
                break;
            }
            ++usableObservationCount;
            if (firstCameraIndex < 0)
            {
                firstCameraIndex = observation.cameraIndex;
            }
            else if (observation.cameraIndex != firstCameraIndex)
            {
                hasSecondDistinctCamera = true;
            }
            const double deltaU = pixel[0] - observation.u;
            const double deltaV = pixel[1] - observation.v;
            initialResidualSquared += detail::sanitizedObservationWeight(
                observation) * (deltaU * deltaU + deltaV * deltaV);
            initialResidualCount += 2;
        }
        const double initialTrackRms = initialResidualCount > 0
            ? std::sqrt(initialResidualSquared /
                        static_cast<double>(initialResidualCount))
            : std::numeric_limits<double>::infinity();
        if (usableObservationCount < 2 ||
            !hasSecondDistinctCamera ||
            !initialPointHasPositiveDepth ||
            (options && options->maxCeresInitialTrackRms > 0.0 &&
             initialTrackRms > options->maxCeresInitialTrackRms))
        {
            continue;
        }
        ++activeTrackCount;

        std::array<std::array<double, 3>, 12> viewingRays{};
        int viewingRayCount = 0;
        int validTrackObservationCount = 0;
        InformationMatrix trackCalibrationInformation{};
        CalibrationPointMatrix calibrationPointInformation{};
        PointInformationMatrix pointInformation{};
        for (const BAObservation &observation : track.observations)
        {
            if (!detail::observationIsUsable(
                    observation, normalizedCameras.size()))
            {
                continue;
            }
            const Camera &camera = normalizedCameras[
                static_cast<std::size_t>(observation.cameraIndex)];
            if (!camera.isValid() || camera.focalX() <= 1.0e-9 ||
                camera.focalY() <= 1.0e-9)
            {
                continue;
            }
            double cameraPoint[3] = {0.0, 0.0, 0.0};
            camera.worldToCamera(world, cameraPoint);
            if (!std::isfinite(cameraPoint[0]) ||
                !std::isfinite(cameraPoint[1]) ||
                !std::isfinite(cameraPoint[2]) || cameraPoint[2] <= 1.0e-9)
            {
                continue;
            }
            activeCameras[static_cast<std::size_t>(
                observation.cameraIndex)] = true;
            const double x = cameraPoint[0] / cameraPoint[2];
            const double y = cameraPoint[1] / cameraPoint[2];
            const double r2 = x * x + y * y;
            const double radius = std::sqrt(r2);
            if (!std::isfinite(radius) || radius > 4.0)
            {
                continue;
            }
            const Camera::Distortion distortion = camera.distortion();
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double radial = 1.0 + distortion.radialK1 * r2 +
                                  distortion.radialK2 * r4 +
                                  distortion.radialK3 * r6;
            const double twoXY = 2.0 * x * y;
            const double distortedX =
                x * radial + distortion.tangentialP1 * twoXY +
                distortion.tangentialP2 * (r2 + 2.0 * x * x);
            const double distortedY =
                y * radial +
                distortion.tangentialP1 * (r2 + 2.0 * y * y) +
                distortion.tangentialP2 * twoXY;
            const double focalX = camera.focalX();
            const double focalY = camera.focalY();
            const double signU = static_cast<double>(camera.uAxisSign());
            const double signV = static_cast<double>(camera.vAxisSign());
            std::array<double, kBAIntrinsicParameterCount> derivativeU{{
                signU * focalX * distortedX,
                0.0,
                1.0,
                0.0,
                signU * focalX * x * r2,
                signU * focalX * x * r4,
                signU * focalX * x * r6,
                signU * focalX * twoXY,
                signU * focalX * (r2 + 2.0 * x * x),
            }};
            std::array<double, kBAIntrinsicParameterCount> derivativeV{{
                signV * focalY * distortedY,
                signV * focalY * distortedY,
                0.0,
                1.0,
                signV * focalY * y * r2,
                signV * focalY * y * r4,
                signV * focalY * y * r6,
                signV * focalY * (r2 + 2.0 * y * y),
                signV * focalY * twoXY,
            }};
            const double weight = detail::sanitizedObservationWeight(observation);
            std::array<double, 3> pointDerivativeU{};
            std::array<double, 3> pointDerivativeV{};
            if (weight <= 0.0 ||
                !pointProjectionJacobian(
                    camera,
                    cameraPoint,
                    x,
                    y,
                    &pointDerivativeU,
                    &pointDerivativeV))
            {
                continue;
            }
            for (std::size_t row = 0; row < kBAIntrinsicParameterCount; ++row)
            {
                for (std::size_t column = 0;
                     column < kBAIntrinsicParameterCount;
                     ++column)
                {
                    trackCalibrationInformation[row][column] += weight *
                        (derivativeU[row] * derivativeU[column] +
                         derivativeV[row] * derivativeV[column]);
                }
                for (std::size_t pointAxis = 0; pointAxis < 3; ++pointAxis)
                {
                    calibrationPointInformation[row][pointAxis] += weight *
                        (derivativeU[row] * pointDerivativeU[pointAxis] +
                         derivativeV[row] * pointDerivativeV[pointAxis]);
                }
            }
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    pointInformation[row][column] += weight *
                        (pointDerivativeU[row] * pointDerivativeU[column] +
                         pointDerivativeV[row] * pointDerivativeV[column]);
                }
            }
            totalWeight += weight;
            ++result.observationCount;
            ++validTrackObservationCount;
            result.maximumNormalizedRadius =
                std::max(result.maximumNormalizedRadius, radius);
            const int histogramBin = std::clamp(
                static_cast<int>(
                    radius / kMaximumHistogramRadius *
                    kRadiusHistogramBins),
                0,
                kRadiusHistogramBins - 1);
            ++radiusHistogram[static_cast<std::size_t>(histogramBin)];
            if (radius <= 0.15)
            {
                ++centralObservationCount;
            }
            if (radius >= 0.30)
            {
                ++peripheralObservationCount;
                double angle = std::atan2(y, x);
                if (angle < 0.0)
                {
                    angle += 2.0 * kPi;
                }
                const int sector = std::clamp(
                    static_cast<int>(angle * 8.0 / (2.0 * kPi)),
                    0,
                    7);
                occupiedSectors[static_cast<std::size_t>(sector)] = true;
            }
            if (x <= -0.15)
            {
                ++leftCount;
            }
            else if (x >= 0.15)
            {
                ++rightCount;
            }
            if (y <= -0.15)
            {
                ++upperCount;
            }
            else if (y >= 0.15)
            {
                ++lowerCount;
            }

            if (viewingRayCount < static_cast<int>(viewingRays.size()))
            {
                const auto center = camera.cameraCenter();
                std::array<double, 3> ray{{
                    track.initialPoint[0] - center[0],
                    track.initialPoint[1] - center[1],
                    track.initialPoint[2] - center[2]}};
                const double length = std::sqrt(
                    ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
                if (std::isfinite(length) && length > 1.0e-12)
                {
                    for (double &component : ray)
                    {
                        component /= length;
                    }
                    viewingRays[static_cast<std::size_t>(viewingRayCount++)] = ray;
                }
            }
        }
        if (options && options->enableControlPointConstraints)
        {
            for (const BAControlPointConstraint &constraint :
                 track.controlPointConstraints)
            {
                if (!std::isfinite(constraint.sigmaMeters) ||
                    constraint.sigmaMeters <= 0.0 ||
                    !std::isfinite(constraint.weight) ||
                    constraint.weight <= 0.0)
                {
                    continue;
                }
                const double information =
                    std::max(0.0, options->controlPointWeight * constraint.weight) /
                    (constraint.sigmaMeters * constraint.sigmaMeters);
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    pointInformation[axis][axis] += information;
                }
            }
        }
        PointInformationMatrix inversePointInformation{};
        if (validTrackObservationCount >= 2 &&
            invertRegularizedPointInformation(
                pointInformation, &inversePointInformation))
        {
            for (std::size_t row = 0;
                 row < kBAIntrinsicParameterCount;
                 ++row)
            {
                for (std::size_t column = 0;
                     column < kBAIntrinsicParameterCount;
                     ++column)
                {
                    double structureExplanation = 0.0;
                    for (std::size_t firstAxis = 0;
                         firstAxis < 3;
                         ++firstAxis)
                    {
                        for (std::size_t secondAxis = 0;
                             secondAxis < 3;
                             ++secondAxis)
                        {
                            structureExplanation +=
                                calibrationPointInformation[row][firstAxis] *
                                inversePointInformation[firstAxis][secondAxis] *
                                calibrationPointInformation[column][secondAxis];
                        }
                    }
                    normalMatrix[row][column] +=
                        trackCalibrationInformation[row][column] -
                        structureExplanation;
                }
            }
        }
        if (validTrackObservationCount >= 3)
        {
            ++result.multiViewTrackCount;
        }
        if (viewingRayCount >= 2)
        {
            double maximumAngle = 0.0;
            for (int first = 0; first < viewingRayCount; ++first)
            {
                for (int second = first + 1; second < viewingRayCount; ++second)
                {
                    const auto &lhs = viewingRays[static_cast<std::size_t>(first)];
                    const auto &rhs = viewingRays[static_cast<std::size_t>(second)];
                    const double dot = std::clamp(
                        lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2],
                        -1.0,
                        1.0);
                    const double crossX = lhs[1] * rhs[2] - lhs[2] * rhs[1];
                    const double crossY = lhs[2] * rhs[0] - lhs[0] * rhs[2];
                    const double crossZ = lhs[0] * rhs[1] - lhs[1] * rhs[0];
                    const double crossLength = std::sqrt(
                        crossX * crossX + crossY * crossY + crossZ * crossZ);
                    maximumAngle = std::max(
                        maximumAngle,
                        std::atan2(crossLength, std::abs(dot)) * 180.0 / kPi);
                }
            }
            triangulationAngles.push_back(maximumAngle);
        }
    }

    result.occupiedPeripheralSectors = static_cast<int>(std::count(
        occupiedSectors.begin(), occupiedSectors.end(), true));
    result.normalizedRadiusP90 = radiusQuantile(
        radiusHistogram, result.observationCount, 0.90);
    result.medianTriangulationAngleDegrees =
        finiteMedian(std::move(triangulationAngles));
    result.multiViewTrackRatio = activeTrackCount <= 0
        ? 0.0
        : static_cast<double>(result.multiViewTrackCount) /
              static_cast<double>(activeTrackCount);

    std::array<double, 3> meanOpticalAxis{{0.0, 0.0, 0.0}};
    for (std::size_t cameraIndex = 0;
         cameraIndex < normalizedCameras.size();
         ++cameraIndex)
    {
        if (!activeCameras[cameraIndex] ||
            !normalizedCameras[cameraIndex].isValid())
        {
            continue;
        }
        const auto rotation =
            normalizedCameras[cameraIndex].cameraToWorldRotation();
        meanOpticalAxis[0] += rotation[2];
        meanOpticalAxis[1] += rotation[5];
        meanOpticalAxis[2] += rotation[8];
        ++result.activeCameraCount;
    }
    if (result.activeCameraCount < 3)
    {
        result.reason = "insufficient_active_cameras";
        return result;
    }
    for (double &component : meanOpticalAxis)
    {
        component /= static_cast<double>(result.activeCameraCount);
    }
    result.opticalAxisConcentration = clampScore(std::sqrt(
        meanOpticalAxis[0] * meanOpticalAxis[0] +
        meanOpticalAxis[1] * meanOpticalAxis[1] +
        meanOpticalAxis[2] * meanOpticalAxis[2]));

    if (result.observationCount < 100 || totalWeight <= 1.0e-12)
    {
        result.reason = "insufficient_valid_observations";
        return result;
    }
    result.valid = true;
    result.incrementalInformationScore =
        incrementalInformationScore(normalMatrix);
    for (std::size_t index = 0; index < kBAIntrinsicParameterCount; ++index)
    {
        const double response = std::sqrt(
            std::max(0.0, normalMatrix[index][index]) / totalWeight) *
            canonicalSteps[index];
        result.sensitivity[index] = clampScore(
            1.0 - std::exp(-response / 0.35));
    }

    const double directionDiversity = clampScore(
        (1.0 - result.opticalAxisConcentration) / 0.35);
    const double angleScore = clampScore(
        (result.medianTriangulationAngleDegrees - 1.0) / 14.0);
    const double coupledGeometry = std::sqrt(
        std::max(0.0, directionDiversity * angleScore));
    result.geometryStrength = clampScore(
        0.70 * coupledGeometry +
        0.20 * angleScore +
        0.10 * result.multiViewTrackRatio);
    const double support = clampScore(
        static_cast<double>(result.observationCount) /
        static_cast<double>(std::max(500, result.activeCameraCount * 40)));
    const double peripheralRatio =
        static_cast<double>(peripheralObservationCount) /
        static_cast<double>(result.observationCount);
    const double centralRatio =
        static_cast<double>(centralObservationCount) /
        static_cast<double>(result.observationCount);
    const double peripheralCoverage = clampScore(peripheralRatio / 0.12);
    const double centralCoverage = clampScore(centralRatio / 0.03);
    const double sectorCoverage = clampScore(
        static_cast<double>(result.occupiedPeripheralSectors) / 8.0);
    const double horizontalBalance = balancedRatio(leftCount, rightCount);
    const double verticalBalance = balancedRatio(upperCount, lowerCount);
    const double axisBalance = std::sqrt(
        std::max(0.0, horizontalBalance * verticalBalance));
    result.observationSupport = support;
    result.peripheralCoverage = peripheralCoverage;
    result.sectorCoverage = sectorCoverage;
    result.imageAxisBalance = axisBalance;
    const double broadRadius = clampScore(
        (result.normalizedRadiusP90 - 0.18) / 0.24);
    const double highOrderRadius = clampScore(
        (result.normalizedRadiusP90 - 0.30) / 0.30);
    const double extremeRadius = clampScore(
        (result.normalizedRadiusP90 - 0.42) / 0.30);

    const auto score = [&](BAIntrinsicParameter parameter,
                           double spatialCoverage,
                           double geometryFactor)
    {
        const std::size_t index = parameterIndex(parameter);
        return clampScore(
            0.15 * support +
            0.22 * result.sensitivity[index] +
            0.20 * std::sqrt(result.incrementalInformationScore[index]) +
            0.20 * clampScore(spatialCoverage) +
            0.23 * clampScore(geometryFactor));
    };

    result.reliability[parameterIndex(BAIntrinsicParameter::FocalLength)] =
        score(BAIntrinsicParameter::FocalLength,
              0.65 * broadRadius + 0.35 * centralCoverage,
              0.55 + 0.45 * result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::FocalAspectRatio)] =
        score(BAIntrinsicParameter::FocalAspectRatio,
              axisBalance * broadRadius,
              result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::PrincipalPointX)] =
        score(BAIntrinsicParameter::PrincipalPointX,
              horizontalBalance * sectorCoverage,
              result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::PrincipalPointY)] =
        score(BAIntrinsicParameter::PrincipalPointY,
              verticalBalance * sectorCoverage,
              result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::RadialK1)] =
        score(BAIntrinsicParameter::RadialK1,
              0.55 * peripheralCoverage + 0.45 * broadRadius,
              0.45 + 0.55 * result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::RadialK2)] =
        score(BAIntrinsicParameter::RadialK2,
              0.45 * highOrderRadius + 0.30 * peripheralCoverage +
                  0.25 * sectorCoverage,
              0.20 + 0.80 * result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::RadialK3)] =
        score(BAIntrinsicParameter::RadialK3,
              0.55 * extremeRadius + 0.25 * peripheralCoverage +
                  0.20 * sectorCoverage,
              result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::TangentialP1)] =
        score(BAIntrinsicParameter::TangentialP1,
              axisBalance * sectorCoverage,
              result.geometryStrength);
    result.reliability[parameterIndex(BAIntrinsicParameter::TangentialP2)] =
        score(BAIntrinsicParameter::TangentialP2,
              axisBalance * sectorCoverage,
              result.geometryStrength);

    const auto enable = [&](BAIntrinsicParameter parameter,
                            double threshold,
                            bool prerequisites)
    {
        const std::size_t index = parameterIndex(parameter);
        result.enabled[index] = prerequisites &&
            result.reliability[index] >= threshold;
    };
    enable(BAIntrinsicParameter::FocalLength, 0.52, true);
    enable(BAIntrinsicParameter::RadialK1,
           0.53,
           result.normalizedRadiusP90 >= 0.22 &&
               result.occupiedPeripheralSectors >= 4);
    enable(BAIntrinsicParameter::FocalAspectRatio,
           0.61,
           result.geometryStrength >= 0.40 && axisBalance >= 0.45);
    enable(BAIntrinsicParameter::PrincipalPointX,
           0.63,
           result.geometryStrength >= 0.50 &&
               horizontalBalance >= 0.50 &&
               result.occupiedPeripheralSectors >= 6);
    enable(BAIntrinsicParameter::PrincipalPointY,
           0.63,
           result.geometryStrength >= 0.50 &&
               verticalBalance >= 0.50 &&
               result.occupiedPeripheralSectors >= 6);
    enable(BAIntrinsicParameter::RadialK2,
           0.62,
           result.geometryStrength >= 0.45 &&
               result.normalizedRadiusP90 >= 0.34 &&
               result.occupiedPeripheralSectors >= 6);
    enable(BAIntrinsicParameter::TangentialP1,
           0.65,
           result.geometryStrength >= 0.55 && axisBalance >= 0.55 &&
               result.occupiedPeripheralSectors >= 6);
    enable(BAIntrinsicParameter::TangentialP2,
           0.65,
           result.geometryStrength >= 0.55 && axisBalance >= 0.55 &&
               result.occupiedPeripheralSectors >= 6);
    enable(BAIntrinsicParameter::RadialK3,
           0.70,
           result.geometryStrength >= 0.65 &&
               result.normalizedRadiusP90 >= 0.46 &&
               result.occupiedPeripheralSectors >= 7);

    // 自标定模型只要释放任何畸变/主点参数，就必须保留焦距共同吸收一阶尺度。
    if (enabledIntrinsicParameterCount(result.enabled) > 0)
    {
        result.enabled[parameterIndex(BAIntrinsicParameter::FocalLength)] = true;
    }
    result.modelName = adaptiveCameraModelName(result.enabled);
    result.reason = result.opticalAxisConcentration >= 0.90
        ? "weak_parallel_geometry_parameter_reliability"
        : "convergent_geometry_parameter_reliability";
    return result;
}

bool applyAdaptiveCameraModel(
    const BAAdaptiveCameraModelAssessment &assessment,
    BAOptions *options)
{
    if (!options)
    {
        return false;
    }
    const bool callerUsesMask = options->useSharedIntrinsicParameterMask;
    const BAIntrinsicParameterMask callerMask =
        options->sharedIntrinsicParameterMask;
    const auto callerAllows = [&](std::size_t index)
    {
        return !callerUsesMask || callerMask[index];
    };

    BAIntrinsicParameterMask effective{};
    for (std::size_t index = 0; index < effective.size(); ++index)
    {
        const auto parameter = static_cast<BAIntrinsicParameter>(index);
        effective[index] = assessment.valid && assessment.enabled[index] &&
                           requestedByLegacyFlags(*options, parameter) &&
                           callerAllows(index);
    }
    const bool anyExtended = std::any_of(
        effective.begin() + 1, effective.end(), [](bool enabled)
        {
            return enabled;
        });
    if (anyExtended)
    {
        const std::size_t focalIndex =
            parameterIndex(BAIntrinsicParameter::FocalLength);
        if (!effective[focalIndex])
        {
            // 扩展参数必须和焦距共同估计；调用方显式冻结焦距时不能绕过其掩码。
            effective.fill(false);
        }
    }

    options->useSharedIntrinsicParameterMask = true;
    options->sharedIntrinsicParameterMask = effective;
    options->refineSharedFocalLength =
        effective[parameterIndex(BAIntrinsicParameter::FocalLength)];
    options->refineSharedFocalAspectRatio =
        effective[parameterIndex(BAIntrinsicParameter::FocalAspectRatio)];
    options->refineSharedPrincipalPoint =
        effective[parameterIndex(BAIntrinsicParameter::PrincipalPointX)] ||
        effective[parameterIndex(BAIntrinsicParameter::PrincipalPointY)];
    options->refineSharedRadialDistortion =
        effective[parameterIndex(BAIntrinsicParameter::RadialK1)] ||
        effective[parameterIndex(BAIntrinsicParameter::RadialK2)] ||
        effective[parameterIndex(BAIntrinsicParameter::RadialK3)] ||
        effective[parameterIndex(BAIntrinsicParameter::TangentialP1)] ||
        effective[parameterIndex(BAIntrinsicParameter::TangentialP2)];
    options->refineSharedHighOrderDistortion =
        effective[parameterIndex(BAIntrinsicParameter::RadialK2)] ||
        effective[parameterIndex(BAIntrinsicParameter::RadialK3)] ||
        effective[parameterIndex(BAIntrinsicParameter::TangentialP1)] ||
        effective[parameterIndex(BAIntrinsicParameter::TangentialP2)];

    if (assessment.valid && assessment.opticalAxisConcentration >= 0.90)
    {
        options->sharedFocalPriorSigma = std::min(
            options->sharedFocalPriorSigma, 0.04);
        options->sharedRadialK1PriorSigma = std::min(
            options->sharedRadialK1PriorSigma, 0.05);
    }
    return enabledIntrinsicParameterCount(effective) > 0;
}

} // namespace xjw

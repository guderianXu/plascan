#include "PlanetaryLaserBaAdapter.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace xjw
{
namespace lidar
{
namespace
{

constexpr int kAmbiguousCamera = -2;

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

std::string trim(std::string value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char character)
    {
        return std::isspace(character) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char character)
    {
        return std::isspace(character) != 0;
    }).base();
    if (first >= last)
    {
        return {};
    }
    return std::string(first, last);
}

std::vector<std::string> imageIdVariants(const std::string &input)
{
    std::string normalized = trim(input);
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    while (normalized.starts_with("./"))
    {
        normalized.erase(0, 2);
    }

    std::vector<std::string> variants;
    const auto appendUnique = [&variants](const std::string &value)
    {
        if (!value.empty() &&
            std::find(variants.begin(), variants.end(), value) == variants.end())
        {
            variants.push_back(value);
        }
    };
    appendUnique(normalized);

    const std::size_t slash = normalized.find_last_of('/');
    const std::string fileName = slash == std::string::npos
        ? normalized
        : normalized.substr(slash + 1);
    appendUnique(fileName);
    const std::size_t dot = fileName.find_last_of('.');
    if (dot != std::string::npos && dot > 0)
    {
        appendUnique(fileName.substr(0, dot));
    }
    return variants;
}

class CameraAliasIndex
{
public:
    explicit CameraAliasIndex(
        const std::vector<std::vector<std::string>> &aliasesByCamera)
    {
        for (std::size_t cameraIndex = 0; cameraIndex < aliasesByCamera.size(); ++cameraIndex)
        {
            for (const std::string &alias : aliasesByCamera[cameraIndex])
            {
                for (const std::string &variant : imageIdVariants(alias))
                {
                    auto [iterator, inserted] = _cameraByAlias.emplace(
                        variant, static_cast<int>(cameraIndex));
                    if (!inserted && iterator->second != static_cast<int>(cameraIndex))
                    {
                        iterator->second = kAmbiguousCamera;
                    }
                }
            }
        }
    }

    int resolve(const std::string &imageId) const
    {
        for (const std::string &variant : imageIdVariants(imageId))
        {
            const auto iterator = _cameraByAlias.find(variant);
            if (iterator == _cameraByAlias.end())
            {
                continue;
            }
            if (iterator->second == kAmbiguousCamera)
            {
                return kAmbiguousCamera;
            }
            return iterator->second;
        }
        return -1;
    }

private:
    std::unordered_map<std::string, int> _cameraByAlias;
};

bool finiteMatrix(const std::array<double, 9> &matrix)
{
    return std::all_of(matrix.begin(), matrix.end(), [](double value)
    {
        return std::isfinite(value);
    });
}

bool covarianceToSqrtInformation(const std::array<double, 9> &covariance,
                                 std::array<double, 9> *sqrtInformation)
{
    if (!sqrtInformation || !finiteMatrix(covariance))
    {
        return false;
    }

    const double scale = std::max(
        {1.0, std::abs(covariance[0]), std::abs(covariance[4]),
         std::abs(covariance[8])});
    const double tolerance = 1.0e-14 * scale;
    const double c00 = covariance[0];
    const double c10 = 0.5 * (covariance[3] + covariance[1]);
    const double c20 = 0.5 * (covariance[6] + covariance[2]);
    const double c11 = covariance[4];
    const double c21 = 0.5 * (covariance[7] + covariance[5]);
    const double c22 = covariance[8];
    if (!(c00 > tolerance))
    {
        return false;
    }

    const double l00 = std::sqrt(c00);
    const double l10 = c10 / l00;
    const double l20 = c20 / l00;
    const double pivot11 = c11 - l10 * l10;
    if (!(pivot11 > tolerance))
    {
        return false;
    }
    const double l11 = std::sqrt(pivot11);
    const double l21 = (c21 - l20 * l10) / l11;
    const double pivot22 = c22 - l20 * l20 - l21 * l21;
    if (!(pivot22 > tolerance))
    {
        return false;
    }
    const double l22 = std::sqrt(pivot22);

    // W = inverse(L), where covariance = L * L^T. Then
    // ||W * delta||^2 = delta^T * covariance^-1 * delta.
    *sqrtInformation = {{
        1.0 / l00, 0.0, 0.0,
        -l10 / (l00 * l11), 1.0 / l11, 0.0,
        (l10 * l21 - l20 * l11) / (l00 * l11 * l22),
        -l21 / (l11 * l22),
        1.0 / l22,
    }};
    return finiteMatrix(*sqrtInformation);
}

double observationWeight(const PlanetaryLaserImageMeasure &measure)
{
    if (!measure.covariancePixelsSquared)
    {
        return 1.0;
    }
    const auto &covariance = *measure.covariancePixelsSquared;
    const double scale = std::max({1.0, std::abs(covariance[0]),
                                   std::abs(covariance[3])});
    const double tolerance = 1.0e-12 * scale;
    if (std::abs(covariance[1]) > tolerance ||
        std::abs(covariance[2]) > tolerance ||
        std::abs(covariance[0] - covariance[3]) > tolerance)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::isfinite(covariance[0]) && covariance[0] > 0.0
        ? 1.0 / covariance[0]
        : 0.0;
}

bool hasNonZeroLeverArm(const std::array<double, 3> &leverArm)
{
    return std::hypot(std::hypot(leverArm[0], leverArm[1]), leverArm[2]) > 0.0;
}

bool validateDatasetMode(const PlanetaryLaserDataset &dataset,
                         const PlanetaryLaserBaAdapterOptions &options,
                         std::string *errorMessage)
{
    if (dataset.sensorModel == PlanetaryLaserSensorModel::LineScan)
    {
        setError(errorMessage,
                 "行星激光数据声明为 line_scan；当前 PlaScan 仅支持静态 frame camera，"
                 "不能把逐行时变轨迹当成单个位姿");
        return false;
    }
    if (dataset.sensorModel == PlanetaryLaserSensorModel::Unknown &&
        !options.confirmUnknownSensorModelIsFrame)
    {
        setError(errorMessage,
                 "行星激光 sensor_model 未知；必须明确声明 frame，或由调用方显式确认"
                 "按静态 frame camera 处理");
        return false;
    }
    if (dataset.rangeType == PlanetaryLaserRangeType::RoundTrip)
    {
        setError(errorMessage,
                 "行星激光 range_type=round_trip；请先按产品定义转换为单程几何距离");
        return false;
    }
    if (dataset.rangeType == PlanetaryLaserRangeType::Unknown &&
        !options.confirmUnknownRangeTypeIsOneWay)
    {
        setError(errorMessage,
                 "行星激光 range_type 未知；必须明确声明 one_way，或由调用方显式确认"
                 "输入已经是单程几何距离");
        return false;
    }
    if (options.cameraCoordinateFrame.empty() ||
        options.cameraCoordinateFrame != dataset.reference.bodyFixedFrame)
    {
        setError(errorMessage,
                 "相机求解坐标系与激光落点 body_fixed_frame 不一致；当前适配器不执行"
                 "隐式坐标转换");
        return false;
    }
    return true;
}

} // namespace

bool buildPlanetaryLaserRangeConstraints(
    const PlanetaryLaserDataset &dataset,
    const PlanetaryLaserBaAdapterOptions &options,
    std::vector<BALaserRangeConstraint> *constraints,
    PlanetaryLaserBaAdapterSummary *summary,
    std::string *errorMessage)
{
    if (!constraints)
    {
        setError(errorMessage, "行星激光 BA 输出 constraints 指针为空");
        return false;
    }
    constraints->clear();
    PlanetaryLaserBaAdapterSummary localSummary;
    localSummary.totalShots = static_cast<int>(dataset.shots.size());

    std::string validationError;
    if (!dataset.validate(&validationError))
    {
        setError(errorMessage, "行星激光数据校验失败: " + validationError);
        return false;
    }
    if (!validateDatasetMode(dataset, options, errorMessage))
    {
        return false;
    }
    if (options.imageAliasesByCameraIndex.empty())
    {
        setError(errorMessage, "行星激光 BA 没有可用于 shot 关联的相机别名");
        return false;
    }

    const CameraAliasIndex aliasIndex(options.imageAliasesByCameraIndex);
    constraints->reserve(dataset.shots.size());
    for (std::size_t shotIndex = 0; shotIndex < dataset.shots.size(); ++shotIndex)
    {
        const PlanetaryLaserShot &shot = dataset.shots[shotIndex];
        std::set<int> simultaneousCameras;
        bool ambiguousSimultaneousImage = false;
        for (const std::string &imageId : shot.simultaneousImageIds)
        {
            const int cameraIndex = aliasIndex.resolve(imageId);
            if (cameraIndex == kAmbiguousCamera)
            {
                ambiguousSimultaneousImage = true;
                break;
            }
            if (cameraIndex >= 0)
            {
                simultaneousCameras.insert(cameraIndex);
            }
        }
        if (ambiguousSimultaneousImage || simultaneousCameras.size() > 1)
        {
            setError(errorMessage,
                     "Shot '" + shot.id +
                         "' 的 simultaneous image 映射到多个/歧义相机；frame MVP 要求唯一相机");
            constraints->clear();
            return false;
        }
        if (simultaneousCameras.empty())
        {
            if (options.allowUnmappedShots)
            {
                ++localSummary.skippedUnmappedShots;
                continue;
            }
            setError(errorMessage,
                     "Shot '" + shot.id +
                         "' 的 simultaneous image 未映射到当前 BA 相机；请提供工程影像别名");
            constraints->clear();
            return false;
        }

        if (hasNonZeroLeverArm(shot.leverArmSensorMeters) &&
            (options.cameraSensorFrame.empty() ||
             options.cameraSensorFrame != dataset.reference.laserFrame))
        {
            setError(errorMessage,
                     "Shot '" + shot.id +
                         "' 的非零 lever arm 不在相机坐标系表达；当前 MVP 不提供传感器框架旋转");
            constraints->clear();
            return false;
        }

        BALaserRangeConstraint constraint;
        constraint.cameraIndex = *simultaneousCameras.begin();
        constraint.initialPoint = shot.pointBodyFixedMeters;
        constraint.observedRangeMeters = shot.observedRangeMeters;
        constraint.sigmaRangeMeters = shot.rangeSigmaMeters;
        constraint.weight = 1.0;
        constraint.leverArmCameraMeters = shot.leverArmSensorMeters;
        constraint.shotId = shot.id;
        constraint.ephemerisTimeSeconds = shot.ephemerisTimeSeconds;
        constraint.sourceIndex = static_cast<int>(shotIndex);

        switch (shot.pointMode)
        {
        case PlanetaryLaserPointMode::Fixed:
            constraint.pointMode = BALaserPointMode::Fixed;
            ++localSummary.fixedPointShots;
            break;
        case PlanetaryLaserPointMode::Constrained:
            constraint.pointMode = BALaserPointMode::Constrained;
            constraint.pointPrior = shot.pointBodyFixedMeters;
            if (!shot.pointCovarianceBodyFixedMetersSquared ||
                !covarianceToSqrtInformation(
                    *shot.pointCovarianceBodyFixedMetersSquared,
                    &constraint.pointPriorSqrtInformation))
            {
                setError(errorMessage,
                         "Shot '" + shot.id +
                             "' 的 3x3 落点协方差不是可逆正定矩阵，无法建立白化先验");
                constraints->clear();
                return false;
            }
            ++localSummary.constrainedPointShots;
            break;
        case PlanetaryLaserPointMode::Free:
            constraint.pointMode = BALaserPointMode::Free;
            ++localSummary.freePointShots;
            break;
        default:
            setError(errorMessage,
                     "Shot '" + shot.id + "' 的 point mode 非法");
            constraints->clear();
            return false;
        }

        std::set<int> measuredCameras;
        for (const PlanetaryLaserImageMeasure &measure : shot.imageMeasures)
        {
            if (measure.kind == PlanetaryLaserImageMeasureKind::ProjectedVirtual)
            {
                ++localSummary.ignoredProjectedMeasures;
                continue;
            }
            const int cameraIndex = aliasIndex.resolve(measure.imageId);
            if (cameraIndex == kAmbiguousCamera)
            {
                setError(errorMessage,
                         "Shot '" + shot.id + "' 的 measured image '" + measure.imageId +
                             "' 映射歧义");
                constraints->clear();
                return false;
            }
            if (cameraIndex < 0)
            {
                if (!options.allowUnmappedMeasuredImages)
                {
                    setError(errorMessage,
                             "Shot '" + shot.id + "' 的真实 measured image '" +
                                 measure.imageId +
                                 "' 未映射到 BA 相机；不能静默丢弃真实像点");
                    constraints->clear();
                    return false;
                }
                ++localSummary.ignoredUnmappedMeasuredImages;
                continue;
            }
            if (!measuredCameras.insert(cameraIndex).second)
            {
                setError(errorMessage,
                         "Shot '" + shot.id + "' 在同一相机中包含重复 measured 像点");
                constraints->clear();
                return false;
            }
            const double weight = observationWeight(measure);
            if (!std::isfinite(weight))
            {
                setError(errorMessage,
                         "Shot '" + shot.id +
                             "' 的 measured image covariance 不是各向同性 sigma^2*I；"
                             "当前标量 BAObservation 不允许静默丢失相关性");
                constraints->clear();
                return false;
            }
            if (!(weight > 0.0))
            {
                setError(errorMessage,
                         "Shot '" + shot.id + "' 的 measured image covariance 无法形成正权重");
                constraints->clear();
                return false;
            }
            constraint.measuredImageObservations.push_back({
                cameraIndex,
                measure.samplePixels,
                measure.linePixels,
                weight,
            });
            ++localSummary.measuredImageObservations;
        }
        if (constraint.pointMode == BALaserPointMode::Free && measuredCameras.size() < 2)
        {
            setError(errorMessage,
                     "Shot '" + shot.id +
                         "' 是 Free 落点，但当前 BA 相机中不足两幅真实 measured 像点");
            constraints->clear();
            return false;
        }

        constraints->push_back(std::move(constraint));
        ++localSummary.acceptedShots;
    }

    if (constraints->empty())
    {
        setError(errorMessage, "行星激光数据没有任何 shot 能映射到当前 BA 相机");
        return false;
    }
    if (summary)
    {
        *summary = localSummary;
    }
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

} // namespace lidar
} // namespace xjw

// ============================================================
// 文件：BundleAdjust.cpp
// 功能：实现光束法平差（Bundle Adjustment）核心算法。
//
// 算法设计要点：
//   1. 交替优化策略：先固定相机优化三维点，再固定点优化相机，循环多轮
//   2. 鲁棒正则化：采用高斯牛顿 + LM 风格阻尼
//   3. Huber 权重：残差范数 > huberDelta 时降低对应观测的权重，抹除粗差
//   4. 数值雅可比：对点 2×3、相机 2×6 残差块采用中央有限差分
//   5. SO(3) 旋转更新：使用 Rodrigues 公式 exp(ω) 进行旋转增量扰动
//   6. 系统求解：自实现带部分主元高斯-约尔当消元，避免外部线性代数依赖
// ============================================================

#include "BundleAdjust.h"

#include "BundleAdjustCeres.h"
#include "BundleAdjustNativeCuda.h"
#include "BundleAdjustQuality.h"
#include "BundleAdjustValidation.h"
#include "log/Logger.h"

#include <plamatrix/ops/small_matrix.h>
#include <plamatrix/ops/statistics.h>
#include <plamatrix/ops/vector.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace xjw
{
namespace
{

// ================================================================
// 说明：
// 该文件实现“手写”的光束法平差核心，采用交替优化策略：
// 1) 固定相机，优化三维点；
// 2) 固定三维点，优化相机位姿（6自由度增量）；
// 3) 重复若干轮。
//
// 数值部分采用高斯牛顿 + 阻尼（LM 风格），雅可比通过中央有限差分近似。
// 为增强鲁棒性，残差使用 Huber 权重抑制粗差影响。
// ================================================================


/**
 * @brief 计算 Huber 损失函数对应的加权系数。
 *
 * Huber 权重定义：
 *   w = 1                 若 ||r|| <= delta
 *   w = delta / ||r||    若 ||r|| >  delta
 *
 * 小残差保留原始平方损失，大残差降为绝对值损失，实现鲁棒优化。
 *
 * @param residualNorm  残差向量的范数
 * @param delta         Huber 阈值（像素）
 * @return              Huber 加权系数 w
 */
inline double huberWeight(double residualNorm, double delta)
{
    if (!(delta > 0.0) || !std::isfinite(residualNorm)) return 1.0;
    if (residualNorm <= delta) return 1.0;
    return delta / std::max(residualNorm, 1e-12);
}

inline double huberLoss(double residualNorm, double delta)
{
    if (!std::isfinite(residualNorm))
    {
        return std::numeric_limits<double>::infinity();
    }
    if (!(delta > 0.0) || residualNorm <= delta)
    {
        return 0.5 * residualNorm * residualNorm;
    }
    return delta * (residualNorm - 0.5 * delta);
}

inline double observationWeight(const BAObservation &observation)
{
    if (!std::isfinite(observation.weight))
    {
        return 1.0;
    }
    return std::max(0.0, observation.weight);
}

/**
 * @brief 封装 Camera::project，将 std::array<double,3> 格式的三维点投影到图像坐标。
 * @param cam  相机对象
 * @param X    三维点（数组格式）
 * @param uv   输出像素坐标
 * @return 投影成功返回 true
 */
bool projectPoint(const Camera &cam, const std::array<double, 3> &X, double uv[2])
{
    const double world[3] = {X[0], X[1], X[2]};
    return cam.projectWorldPoint(world, uv);
}

std::array<double, 9> relativeRotationCurrentToPrior(const std::array<double, 9> &current,
                                                     const std::array<double, 9> &prior)
{
    std::array<double, 9> relative{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                sum += current[r * 3 + k] * prior[c * 3 + k];
            }
            relative[r * 3 + c] = sum;
        }
    }
    return relative;
}

std::array<double, 3> rotationLogVector(const std::array<double, 9> &rotation)
{
    const double trace = rotation[0] + rotation[4] + rotation[8];
    const double cosAngle = std::clamp(0.5 * (trace - 1.0), -1.0, 1.0);
    const double angle = std::acos(cosAngle);
    const std::array<double, 3> vee{{
        rotation[7] - rotation[5],
        rotation[2] - rotation[6],
        rotation[3] - rotation[1],
    }};

    if (angle < 1e-10)
    {
        return {{0.5 * vee[0], 0.5 * vee[1], 0.5 * vee[2]}};
    }

    const double sinAngle = std::sin(angle);
    if (std::abs(sinAngle) < 1e-10)
    {
        return {{0.0, 0.0, 0.0}};
    }

    const double scale = angle / (2.0 * sinAngle);
    return {{scale * vee[0], scale * vee[1], scale * vee[2]}};
}

const BACameraPosePrior *cameraPosePriorForIndex(const BAOptions &opt, int cameraIndex)
{
    if (cameraIndex < 0 || cameraIndex >= static_cast<int>(opt.cameraPosePriors.size()))
    {
        return nullptr;
    }
    const BACameraPosePrior &prior = opt.cameraPosePriors[static_cast<size_t>(cameraIndex)];
    return prior.enabled ? &prior : nullptr;
}

std::array<double, 6> cameraPosePriorResidual(const Camera &cam,
                                              const BACameraPosePrior &prior)
{
    std::array<double, 6> residual{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    const double rotationSigmaRad = prior.rotationSigmaDegrees * 3.14159265358979323846 / 180.0;
    if (rotationSigmaRad > 1e-12)
    {
        const auto relativeRotation =
            relativeRotationCurrentToPrior(cam.cameraToWorldRotation(), prior.cameraToWorldRotation);
        const auto rotationResidual = rotationLogVector(relativeRotation);
        residual[0] = rotationResidual[0] / rotationSigmaRad;
        residual[1] = rotationResidual[1] / rotationSigmaRad;
        residual[2] = rotationResidual[2] / rotationSigmaRad;
    }

    if (prior.positionSigmaMeters > 1e-12)
    {
        const auto center = cam.cameraCenter();
        residual[3] = (center[0] - prior.cameraCenter[0]) / prior.positionSigmaMeters;
        residual[4] = (center[1] - prior.cameraCenter[1]) / prior.positionSigmaMeters;
        residual[5] = (center[2] - prior.cameraCenter[2]) / prior.positionSigmaMeters;
    }
    return residual;
}

double cameraPosePriorResidualNorm(const Camera &cam,
                                   const BACameraPosePrior &prior)
{
    const auto residual = cameraPosePriorResidual(cam, prior);
    double sum2 = 0.0;
    for (const double value : residual)
    {
        sum2 += value * value;
    }
    return std::sqrt(sum2);
}

/**
 * @brief 自实现带部分主元高斯-约尔当消元，求解小型线性系统 Ax = b。
 *
 * 采用高斯（Jordan）全列截消元 + 部分主元，即每列最大绝对值元素优先交换行，
 * 改善病态时逐步累走导致的数值不稳定。
 *
 * @param A  n×n 系数矩阵（行优先，则内部修改）
 * @param b  n 元素右端向量（则内部修改）
 * @param n  方程组阶数
 * @param x  输出解向量
 * @return   求解成功返回 true；遇到奇异矩阵或解非有限时返回 false
 */
template <std::size_t N>
bool solveFixedLinearSystem(const std::vector<double> &matrix,
                            const std::vector<double> &rhs,
                            std::vector<double> *solution)
{
    if (!solution || matrix.size() != N * N || rhs.size() != N)
    {
        return false;
    }

    std::array<double, N * N> fixedMatrix{};
    std::array<double, N> fixedRhs{};
    std::copy(matrix.begin(), matrix.end(), fixedMatrix.begin());
    std::copy(rhs.begin(), rhs.end(), fixedRhs.begin());
    std::array<double, N> fixedSolution{};
    if (!plamatrix::solveSmallLinearSystem(
            fixedMatrix, fixedRhs, &fixedSolution))
    {
        return false;
    }

    solution->assign(fixedSolution.begin(), fixedSolution.end());
    return true;
}

bool solveLinearSystem(const std::vector<double> &matrix,
                       const std::vector<double> &rhs,
                       int order,
                       std::vector<double> *solution)
{
    if (order == 3)
    {
        return solveFixedLinearSystem<3>(matrix, rhs, solution);
    }
    if (order == 6)
    {
        return solveFixedLinearSystem<6>(matrix, rhs, solution);
    }
    return false;
}

/**
 * @brief 计算单条轨迹在当前三维点 X 下的重投影 RMS 误差。
 *
 * RMS = sqrt( sum_i( (u_i - u_hat_i)^2 + (v_i - v_hat_i)^2 ) / (2 * n_obs) )
 *
 * @param cams   相机列表
 * @param track  包含观测的轨迹
 * @param X      当前三维点坐标
 * @return       RMS 误差（像素），无有效观测时返回 NaN
 */
double computeTrackRms(const std::vector<Camera> &cams, const BATrack &track, const std::array<double, 3> &X)
{
    double sum2 = 0.0;
    int cnt = 0;
    for (const BAObservation &obs : track.observations) {
        if (obs.cameraIndex < 0 || obs.cameraIndex >= static_cast<int>(cams.size())) continue;
        double uv[2] = {0.0, 0.0};
        if (!projectPoint(cams[obs.cameraIndex], X, uv)) continue;
        const double du = uv[0] - obs.u;
        const double dv = uv[1] - obs.v;
        const double w = observationWeight(obs);
        sum2 += w * (du * du + dv * dv);
        cnt += 2;
    }
    if (cnt <= 0) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(sum2 / static_cast<double>(cnt));
}

double laserSignedDistance(const BALaserPlaneConstraint &constraint, const std::array<double, 3> &X)
{
    return (X[0] - constraint.point[0]) * constraint.normal[0]
        + (X[1] - constraint.point[1]) * constraint.normal[1]
        + (X[2] - constraint.point[2]) * constraint.normal[2];
}

double controlPointDistance(const BAControlPointConstraint &constraint, const std::array<double, 3> &X)
{
    const double dx = X[0] - constraint.point[0];
    const double dy = X[1] - constraint.point[1];
    const double dz = X[2] - constraint.point[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool scaleBarConstraintIsUsable(const BAScaleBarConstraint &constraint, std::size_t trackCount)
{
    return constraint.trackIndexA >= 0
        && constraint.trackIndexB >= 0
        && constraint.trackIndexA != constraint.trackIndexB
        && static_cast<std::size_t>(constraint.trackIndexA) < trackCount
        && static_cast<std::size_t>(constraint.trackIndexB) < trackCount
        && std::isfinite(constraint.measuredDistanceMeters)
        && constraint.measuredDistanceMeters > 0.0;
}

bool pointFromRefinedOrInitial(const std::vector<BATrack> &tracks,
                               const std::vector<BARefinedPoint> *points,
                               int trackIndex,
                               std::array<double, 3> *point)
{
    if (!point || trackIndex < 0 || static_cast<std::size_t>(trackIndex) >= tracks.size())
    {
        return false;
    }

    if (points)
    {
        if (static_cast<std::size_t>(trackIndex) >= points->size()
            || !(*points)[static_cast<std::size_t>(trackIndex)].valid)
        {
            return false;
        }
        *point = (*points)[static_cast<std::size_t>(trackIndex)].point;
    }
    else
    {
        *point = tracks[static_cast<std::size_t>(trackIndex)].initialPoint;
    }
    return plamatrix::isFinite(plamatrix::Vec3<double>(*point));
}

double pointDistance(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double scaleBarResidual(const BAScaleBarConstraint &constraint,
                        const std::array<double, 3> &a,
                        const std::array<double, 3> &b)
{
    return pointDistance(a, b) - constraint.measuredDistanceMeters;
}

bool scaleBarOtherEndpoint(const BAScaleBarConstraint &constraint,
                           int trackIndex,
                           const std::vector<BARefinedPoint> &pointSnapshot,
                           std::array<double, 3> *other)
{
    if (!other)
    {
        return false;
    }

    const int otherIndex = (trackIndex == constraint.trackIndexA)
        ? constraint.trackIndexB
        : ((trackIndex == constraint.trackIndexB) ? constraint.trackIndexA : -1);
    if (otherIndex < 0 || static_cast<std::size_t>(otherIndex) >= pointSnapshot.size())
    {
        return false;
    }

    const BARefinedPoint &otherPoint = pointSnapshot[static_cast<std::size_t>(otherIndex)];
    if (!otherPoint.valid || !plamatrix::isFinite(plamatrix::Vec3<double>(otherPoint.point)))
    {
        return false;
    }

    *other = otherPoint.point;
    return true;
}

double computeTrackLaserRms(const BATrack &track, const std::array<double, 3> &X)
{
    if (track.laserPlaneConstraints.empty())
    {
        return 0.0;
    }

    double sum2 = 0.0;
    int count = 0;
    for (const BALaserPlaneConstraint &constraint : track.laserPlaneConstraints)
    {
        const double distance = laserSignedDistance(constraint, X);
        if (!std::isfinite(distance))
        {
            continue;
        }
        sum2 += distance * distance;
        ++count;
    }
    return count > 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
}

double computeTrackControlPointRms(const BATrack &track, const std::array<double, 3> &X)
{
    if (track.controlPointConstraints.empty())
    {
        return 0.0;
    }

    double sum2 = 0.0;
    int count = 0;
    for (const BAControlPointConstraint &constraint : track.controlPointConstraints)
    {
        const double distance = controlPointDistance(constraint, X);
        if (!std::isfinite(distance))
        {
            continue;
        }
        sum2 += distance * distance;
        ++count;
    }
    return count > 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
}

double computeTrackScaleBarRms(const BAOptions &opt,
                               int trackIndex,
                               const std::array<double, 3> &X,
                               const std::vector<BARefinedPoint> *pointSnapshot)
{
    if (!pointSnapshot || opt.scaleBarConstraints.empty())
    {
        return 0.0;
    }

    double sum2 = 0.0;
    int count = 0;
    for (const BAScaleBarConstraint &constraint : opt.scaleBarConstraints)
    {
        if (constraint.trackIndexA != trackIndex && constraint.trackIndexB != trackIndex)
        {
            continue;
        }
        if (!scaleBarConstraintIsUsable(constraint, pointSnapshot->size()))
        {
            continue;
        }

        std::array<double, 3> other{};
        if (!scaleBarOtherEndpoint(constraint, trackIndex, *pointSnapshot, &other))
        {
            continue;
        }

        const double residual = pointDistance(X, other) - constraint.measuredDistanceMeters;
        if (!std::isfinite(residual))
        {
            continue;
        }
        sum2 += residual * residual;
        ++count;
    }
    return count > 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
}

double computeTrackAcceptanceCost(const std::vector<Camera> &cams,
                                  const BATrack &track,
                                  const std::array<double, 3> &X,
                                  const BAOptions &opt,
                                  int trackIndex,
                                  const std::vector<BARefinedPoint> *pointSnapshot)
{
    double cost = 0.0;
    int residualCount = 0;
    for (const BAObservation &observation : track.observations)
    {
        if (observation.cameraIndex < 0 ||
            observation.cameraIndex >= static_cast<int>(cams.size()))
        {
            continue;
        }
        double pixel[2] = {0.0, 0.0};
        if (!projectPoint(cams[static_cast<size_t>(observation.cameraIndex)], X, pixel))
        {
            continue;
        }
        const double du = pixel[0] - observation.u;
        const double dv = pixel[1] - observation.v;
        const double residualNorm = std::sqrt(du * du + dv * dv);
        const double contribution =
            observationWeight(observation) * huberLoss(residualNorm, opt.huberDelta);
        if (!std::isfinite(contribution))
        {
            continue;
        }
        cost += contribution;
        ++residualCount;
    }
    if (opt.enableLaserPlaneConstraints && !track.laserPlaneConstraints.empty())
    {
        for (const BALaserPlaneConstraint &constraint : track.laserPlaneConstraints)
        {
            const double residual = laserSignedDistance(constraint, X);
            const double contribution =
                std::max(0.0, opt.laserPlaneWeight) *
                std::max(0.0, constraint.weight) *
                huberLoss(std::abs(residual), opt.laserHuberDeltaMeters);
            if (std::isfinite(contribution))
            {
                cost += contribution;
                ++residualCount;
            }
        }
    }
    if (opt.enableControlPointConstraints && !track.controlPointConstraints.empty())
    {
        for (const BAControlPointConstraint &constraint : track.controlPointConstraints)
        {
            const double residual = controlPointDistance(constraint, X);
            const double sigma =
                std::isfinite(constraint.sigmaMeters) && std::abs(constraint.sigmaMeters) > 1e-12
                    ? std::abs(constraint.sigmaMeters)
                    : 1.0;
            const double contribution =
                std::max(0.0, opt.controlPointWeight) *
                std::max(0.0, constraint.weight) /
                (sigma * sigma) *
                huberLoss(residual, opt.controlPointHuberDeltaMeters);
            if (std::isfinite(contribution))
            {
                cost += contribution;
                ++residualCount;
            }
        }
    }
    if (opt.enableScaleBarConstraints && pointSnapshot && !opt.scaleBarConstraints.empty())
    {
        for (const BAScaleBarConstraint &constraint : opt.scaleBarConstraints)
        {
            if ((constraint.trackIndexA != trackIndex &&
                 constraint.trackIndexB != trackIndex) ||
                !scaleBarConstraintIsUsable(constraint, pointSnapshot->size()))
            {
                continue;
            }
            std::array<double, 3> other{};
            if (!scaleBarOtherEndpoint(constraint, trackIndex, *pointSnapshot, &other))
            {
                continue;
            }
            const double residual =
                std::abs(pointDistance(X, other) - constraint.measuredDistanceMeters);
            const double sigma =
                std::isfinite(constraint.sigmaMeters) && std::abs(constraint.sigmaMeters) > 1e-12
                    ? std::abs(constraint.sigmaMeters)
                    : 1.0;
            const double contribution =
                std::max(0.0, opt.scaleBarWeight) *
                std::max(0.0, constraint.weight) /
                (sigma * sigma) *
                huberLoss(residual, opt.scaleBarHuberDeltaMeters);
            if (std::isfinite(contribution))
            {
                cost += contribution;
                ++residualCount;
            }
        }
    }
    return residualCount > 0 ? cost : std::numeric_limits<double>::infinity();
}

struct LaserDistanceStats
{
    int count = 0;
    double rms = 0.0;
    double median = 0.0;
};

struct ControlPointDistanceStats
{
    int count = 0;
    double rms = 0.0;
};

struct ScaleBarDistanceStats
{
    int count = 0;
    double rms = 0.0;
};

std::array<double, 3> pointForConstraintStats(
    const std::vector<BATrack> &tracks,
    const std::vector<BARefinedPoint> *points,
    std::size_t trackIndex)
{
    if (points &&
        trackIndex < points->size() &&
        (*points)[trackIndex].valid &&
        plamatrix::isFinite(
            plamatrix::Vec3<double>((*points)[trackIndex].point)))
    {
        return (*points)[trackIndex].point;
    }

    // 被正深度或重投影门控拒绝的点不能从约束统计中消失，否则会把
    // “优化后没有可评估样本”误报成 0 RMS。此时保守沿用该 track 初值。
    return tracks[trackIndex].initialPoint;
}

LaserDistanceStats computeLaserStatsForPoints(const std::vector<BATrack> &tracks,
                                              const std::vector<BARefinedPoint> *points)
{
    LaserDistanceStats stats;
    std::vector<double> distances;

    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
        const BATrack &track = tracks[trackIndex];
        const std::array<double, 3> X =
            pointForConstraintStats(tracks, points, trackIndex);

        for (const BALaserPlaneConstraint &constraint : track.laserPlaneConstraints)
        {
            const double distance = std::abs(laserSignedDistance(constraint, X));
            if (!std::isfinite(distance))
            {
                continue;
            }
            distances.push_back(distance);
        }
    }

    stats.count = static_cast<int>(distances.size());
    if (distances.empty())
    {
        return stats;
    }

    double sum2 = 0.0;
    for (double distance : distances)
    {
        sum2 += distance * distance;
    }
    stats.rms = std::sqrt(sum2 / static_cast<double>(distances.size()));

    stats.median =
        plamatrix::finiteMedian(std::move(distances)).value_or(0.0);
    return stats;
}

ControlPointDistanceStats computeControlPointStatsForPoints(const std::vector<BATrack> &tracks,
                                                            const std::vector<BARefinedPoint> *points)
{
    ControlPointDistanceStats stats;
    double sum2 = 0.0;

    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
        const BATrack &track = tracks[trackIndex];
        const std::array<double, 3> X =
            pointForConstraintStats(tracks, points, trackIndex);

        for (const BAControlPointConstraint &constraint : track.controlPointConstraints)
        {
            const double distance = controlPointDistance(constraint, X);
            if (!std::isfinite(distance))
            {
                continue;
            }
            sum2 += distance * distance;
            ++stats.count;
        }
    }

    if (stats.count > 0)
    {
        stats.rms = std::sqrt(sum2 / static_cast<double>(stats.count));
    }
    return stats;
}

ScaleBarDistanceStats computeScaleBarStatsForPoints(const std::vector<BATrack> &tracks,
                                                    const std::vector<BARefinedPoint> *points,
                                                    const std::vector<BAScaleBarConstraint> &constraints)
{
    ScaleBarDistanceStats stats;
    double sum2 = 0.0;

    for (const BAScaleBarConstraint &constraint : constraints)
    {
        if (!scaleBarConstraintIsUsable(constraint, tracks.size()))
        {
            continue;
        }

        const std::array<double, 3> a =
            pointForConstraintStats(
                tracks,
                points,
                static_cast<std::size_t>(constraint.trackIndexA));
        const std::array<double, 3> b =
            pointForConstraintStats(
                tracks,
                points,
                static_cast<std::size_t>(constraint.trackIndexB));
        if (!plamatrix::isFinite(plamatrix::Vec3<double>(a)) ||
            !plamatrix::isFinite(plamatrix::Vec3<double>(b)))
        {
            continue;
        }

        const double residual = scaleBarResidual(constraint, a, b);
        if (!std::isfinite(residual))
        {
            continue;
        }

        sum2 += residual * residual;
        ++stats.count;
    }

    if (stats.count > 0)
    {
        stats.rms = std::sqrt(sum2 / static_cast<double>(stats.count));
    }
    return stats;
}

/**
 * @brief 对单条轨迹优化三维点坐标（固定所有相机）。
 *
 * 优化目标：min_X sum_i w_i * || proj(cam_i, X) - obs_i ||^2
 * 其中 w_i 为 Huber 加权，雅可比通过对 X 每个分量做有限差分得到。
 * 正规方程 H*dx = -g，加 damping 阻尼后求解点坐标增量 dx。
 *
 * @param cams   相机列表（优化期间固定不变）
 * @param track  待优化的轨迹
 * @param opt    优化选项
 * @return       点优化结果（BARefinedPoint）
 */
BARefinedPoint optimizeOnePoint(const std::vector<Camera> &cams,
                                const BATrack &track,
                                const BAOptions &opt,
                                int trackIndex,
                                const std::vector<BARefinedPoint> *pointSnapshot)
{
    BARefinedPoint out;
    out.point = track.initialPoint;
    if (!plamatrix::isFinite(plamatrix::Vec3<double>(out.point)) || track.observations.size() < 2) return out;

    out.rmsBefore = computeTrackRms(cams, track, out.point);
    std::array<double, 3> X = out.point;
    double currentCost = computeTrackAcceptanceCost(cams, track, X, opt, trackIndex, pointSnapshot);
    double lambda = opt.damping;  // 自适应 LM 阻尼

    for (int iter = 0; iter < std::max(1, opt.maxPointIterations); ++iter)
    {
        std::vector<double> H(9, 0.0);
        std::vector<double> g(3, 0.0);
        int used = 0;

        for (const BAObservation &obs : track.observations)
        {
            if (obs.cameraIndex < 0 || obs.cameraIndex >= static_cast<int>(cams.size()))
            {
                continue;
            }
            const Camera &cam = cams[obs.cameraIndex];

            double uv[2] = {0.0, 0.0};
            if (!projectPoint(cam, X, uv))
            {
                continue;
            }

            const double r[2] = {uv[0] - obs.u, uv[1] - obs.v};
            const double w = observationWeight(obs)
                * huberWeight(std::sqrt(r[0] * r[0] + r[1] * r[1]), opt.huberDelta);
            if (!(w > 0.0))
            {
                continue;
            }

            // 对三维点坐标 (X,Y,Z) 做中央差分，近似雅可比 (精度 O(h²))。
            double J[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
            const double eps = opt.finiteDiffEps;
            for (int k = 0; k < 3; ++k)
            {
                std::array<double, 3> Xp = X, Xm = X;
                Xp[k] += eps;
                Xm[k] -= eps;
                double uvp[2] = {0.0, 0.0}, uvm[2] = {0.0, 0.0};
                if (!projectPoint(cam, Xp, uvp) || !projectPoint(cam, Xm, uvm))
                {
                    continue;
                }
                J[0][k] = (uvp[0] - uvm[0]) / (2.0 * eps);
                J[1][k] = (uvp[1] - uvm[1]) / (2.0 * eps);
            }

            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    H[a * 3 + b] += w * (J[0][a] * J[0][b] + J[1][a] * J[1][b]);
                }
                g[a] += w * (J[0][a] * r[0] + J[1][a] * r[1]);
            }
            ++used;
        }

        if (opt.enableLaserPlaneConstraints && !track.laserPlaneConstraints.empty())
        {
            for (const BALaserPlaneConstraint &constraint : track.laserPlaneConstraints)
            {
                const double r = laserSignedDistance(constraint, X);
                if (!std::isfinite(r))
                {
                    continue;
                }

                const double robustWeight = huberWeight(std::abs(r), opt.laserHuberDeltaMeters);
                const double w = robustWeight * std::max(0.0, opt.laserPlaneWeight)
                    * std::max(0.0, constraint.weight);
                if (!(w > 0.0))
                {
                    continue;
                }

                const double J[3] = {constraint.normal[0], constraint.normal[1], constraint.normal[2]};
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        H[a * 3 + b] += w * J[a] * J[b];
                    }
                    g[a] += w * J[a] * r;
                }
            }
        }

        if (opt.enableControlPointConstraints && !track.controlPointConstraints.empty())
        {
            for (const BAControlPointConstraint &constraint : track.controlPointConstraints)
            {
                const double r[3] = {
                    X[0] - constraint.point[0],
                    X[1] - constraint.point[1],
                    X[2] - constraint.point[2],
                };
                const double norm = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                if (!std::isfinite(norm))
                {
                    continue;
                }

                const double sigma = (std::isfinite(constraint.sigmaMeters) && std::abs(constraint.sigmaMeters) > 1e-12)
                    ? std::abs(constraint.sigmaMeters)
                    : 1.0;
                const double robustWeight = huberWeight(norm, opt.controlPointHuberDeltaMeters);
                const double w = robustWeight * std::max(0.0, opt.controlPointWeight)
                    * std::max(0.0, constraint.weight) / (sigma * sigma);
                if (!(w > 0.0))
                {
                    continue;
                }

                for (int a = 0; a < 3; ++a)
                {
                    H[a * 3 + a] += w;
                    g[a] += w * r[a];
                }
                used += 3;
            }
        }

        if (opt.enableScaleBarConstraints && pointSnapshot && !opt.scaleBarConstraints.empty())
        {
            for (const BAScaleBarConstraint &constraint : opt.scaleBarConstraints)
            {
                if (constraint.trackIndexA != trackIndex && constraint.trackIndexB != trackIndex)
                {
                    continue;
                }
                if (!scaleBarConstraintIsUsable(constraint, pointSnapshot->size()))
                {
                    continue;
                }

                std::array<double, 3> other{};
                if (!scaleBarOtherEndpoint(constraint, trackIndex, *pointSnapshot, &other))
                {
                    continue;
                }

                const double delta[3] = {
                    X[0] - other[0],
                    X[1] - other[1],
                    X[2] - other[2],
                };
                const double distance =
                    std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
                if (!std::isfinite(distance) || distance <= 1e-12)
                {
                    continue;
                }

                const double residual = distance - constraint.measuredDistanceMeters;
                const double sigma = (std::isfinite(constraint.sigmaMeters) && std::abs(constraint.sigmaMeters) > 1e-12)
                    ? std::abs(constraint.sigmaMeters)
                    : 1.0;
                const double robustWeight = huberWeight(std::abs(residual), opt.scaleBarHuberDeltaMeters);
                const double w = robustWeight * std::max(0.0, opt.scaleBarWeight)
                    * std::max(0.0, constraint.weight) / (sigma * sigma);
                if (!(w > 0.0))
                {
                    continue;
                }

                const double J[3] = {
                    delta[0] / distance,
                    delta[1] / distance,
                    delta[2] / distance,
                };
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        H[a * 3 + b] += w * J[a] * J[b];
                    }
                    g[a] += w * J[a] * residual;
                }
                ++used;
            }
        }

        if (used < 2)
        {
            break;
        }

        // LM 阻尼：对角线加 lambda * diag(H)
        for (int d = 0; d < 3; ++d)
        {
            H[d * 3 + d] += lambda * std::max(H[d * 3 + d], 1e-10);
        }

        std::vector<double> rhs = {-g[0], -g[1], -g[2]};
        std::vector<double> dx;
        if (!solveLinearSystem(H, rhs, 3, &dx))
        {
            break;
        }

        // 候选点
        std::array<double, 3> Xnew = {X[0] + dx[0], X[1] + dx[1], X[2] + dx[2]};
        double newCost = computeTrackAcceptanceCost(cams, track, Xnew, opt, trackIndex, pointSnapshot);

        // 步长接受检查：仅当 cost 减小（或首次有限cost）时接受步长
        if (std::isfinite(newCost) && (newCost < currentCost + 1e-12 || !std::isfinite(currentCost)))
        {
            X = Xnew;
            currentCost = newCost;
            lambda = std::max(lambda * 0.5, 1e-10);  // 成功 → 减小阻尼
        } else {
            lambda *= 5.0;  // 失败 → 增大阻尼
            if (lambda > 1e8) break;  // 阻尼过大，放弃
        }
        out.iterations = iter + 1;

        const double step = plamatrix::norm(plamatrix::Vec3<double>{dx[0], dx[1], dx[2]});
        if (step < opt.stepTolerance) {
            out.converged = true;
            break;
        }
    }

    out.point = X;
    out.rmsAfter = computeTrackRms(cams, track, X);
    out.valid = plamatrix::isFinite(plamatrix::Vec3<double>(X)) && std::isfinite(out.rmsAfter);
    return out;
}

/**
 * @brief 对单台相机优化 6-DOF 外参（固定所有三维点）。
 *
 * 优化目标：min_{delta} sum_{ti,obs} w * || proj(cam+delta, X_ti) - obs ||^2
 * 雅可比对 6 个参数 [rx, ry, rz, tx, ty, tz] 做有限差分得到 2×6 雅可比矩阵。
 * 使用带阻尼高斯牛顿迭代求解增量 dx，然后通过 applyDeltaPose 更新相机。
 *
 * @param cam          待优化的相机（改变会直接写入）
 * @param cameraIndex  相机在全局列表中的索引
 * @param tracks       全部轨迹列表
 * @param points       已优化的三维点列表（与 tracks 索引对应）
 * @param opt          优化选项
 * @return             至少更新了一次相机位姿时返回 true
 */
/// @brief 计算单台相机在所有观测上的重投影 RMS。
static double computeCameraCost(const Camera &cam, int cameraIndex,
                               const std::vector<BATrack> &tracks,
                               const std::vector<BARefinedPoint> &points,
                               const BAOptions &opt)
{
    double cost = 0.0;
    int residualCount = 0;
    for (size_t ti = 0; ti < tracks.size() && ti < points.size(); ++ti) {
        if (!points[ti].valid) continue;
        for (const BAObservation &obs : tracks[ti].observations) {
            if (obs.cameraIndex != cameraIndex) continue;
            double uv[2];
            if (!projectPoint(cam, points[ti].point, uv)) continue;
            double du = uv[0] - obs.u, dv = uv[1] - obs.v;
            const double residualNorm = std::sqrt(du * du + dv * dv);
            cost += observationWeight(obs) * huberLoss(residualNorm, opt.huberDelta);
            ++residualCount;
        }
    }

    if (const BACameraPosePrior *prior = cameraPosePriorForIndex(opt, cameraIndex))
    {
        const double norm = cameraPosePriorResidualNorm(cam, *prior);
        cost += std::max(0.0, opt.cameraPosePriorWeight) *
                huberLoss(norm, opt.cameraPosePriorHuberDelta);
        ++residualCount;
    }
    return residualCount > 0 ? cost : std::numeric_limits<double>::infinity();
}

bool optimizeOneCamera(Camera *cam,
                       int cameraIndex,
                       const std::vector<BATrack> &tracks,
                       const std::vector<BARefinedPoint> &points,
                       const BAOptions &opt)
{
    if (!cam) return false;

    bool anyUpdated = false;
    double lambda = opt.damping;  // 自适应 LM 阻尼
    double currentCost = computeCameraCost(*cam, cameraIndex, tracks, points, opt);
    const double eps = std::max(opt.finiteDiffEps, 1e-7);

    for (int iter = 0; iter < std::max(1, opt.maxCameraIterations); ++iter) {
        std::vector<double> H(36, 0.0);
        std::vector<double> g(6, 0.0);
        int used = 0;

        for (size_t ti = 0; ti < tracks.size() && ti < points.size(); ++ti) {
            const BARefinedPoint &pt = points[ti];
            if (!pt.valid) continue;

            const BATrack &track = tracks[ti];
            for (const BAObservation &obs : track.observations) {
                if (obs.cameraIndex != cameraIndex) continue;

                double uv[2] = {0.0, 0.0};
                if (!projectPoint(*cam, pt.point, uv)) continue;

                const double r[2] = {uv[0] - obs.u, uv[1] - obs.v};
                const double w = observationWeight(obs)
                    * huberWeight(std::sqrt(r[0] * r[0] + r[1] * r[1]), opt.huberDelta);
                if (!(w > 0.0))
                {
                    continue;
                }

                // 对 [rx, ry, rz, tx, ty, tz] 做中央差分，构造 2x6 雅可比（精度 O(h²)）。
                double J[2][6] = {{0,0,0,0,0,0},{0,0,0,0,0,0}};
                for (int k = 0; k < 6; ++k) {
                    Camera tmpP = *cam, tmpM = *cam;
                    double deltaP[6] = {0,0,0,0,0,0};
                    double deltaM[6] = {0,0,0,0,0,0};
                    deltaP[k] = +eps;
                    deltaM[k] = -eps;
                    tmpP.applyDeltaPose(deltaP);
                    tmpM.applyDeltaPose(deltaM);

                    double uvp[2] = {0,0}, uvm[2] = {0,0};
                    if (!projectPoint(tmpP, pt.point, uvp) ||
                        !projectPoint(tmpM, pt.point, uvm)) continue;
                    J[0][k] = (uvp[0] - uvm[0]) / (2.0 * eps);
                    J[1][k] = (uvp[1] - uvm[1]) / (2.0 * eps);
                }

                for (int a = 0; a < 6; ++a) {
                    for (int b = 0; b < 6; ++b) {
                        H[a * 6 + b] += w * (J[0][a] * J[0][b] + J[1][a] * J[1][b]);
                    }
                    g[a] += w * (J[0][a] * r[0] + J[1][a] * r[1]);
                }
                ++used;
            }
        }

        if (const BACameraPosePrior *prior = cameraPosePriorForIndex(opt, cameraIndex))
        {
            const auto residual = cameraPosePriorResidual(*cam, *prior);
            const double residualNorm = cameraPosePriorResidualNorm(*cam, *prior);
            const double w = std::max(0.0, opt.cameraPosePriorWeight)
                * huberWeight(residualNorm, opt.cameraPosePriorHuberDelta);

            if (w > 0.0)
            {
                double J[6][6] = {};
                for (int k = 0; k < 6; ++k)
                {
                    Camera tmpP = *cam;
                    Camera tmpM = *cam;
                    double deltaP[6] = {0, 0, 0, 0, 0, 0};
                    double deltaM[6] = {0, 0, 0, 0, 0, 0};
                    deltaP[k] = +eps;
                    deltaM[k] = -eps;
                    tmpP.applyDeltaPose(deltaP);
                    tmpM.applyDeltaPose(deltaM);
                    const auto rp = cameraPosePriorResidual(tmpP, *prior);
                    const auto rm = cameraPosePriorResidual(tmpM, *prior);
                    for (int row = 0; row < 6; ++row)
                    {
                        J[row][k] = (rp[static_cast<size_t>(row)] - rm[static_cast<size_t>(row)]) / (2.0 * eps);
                    }
                }

                for (int a = 0; a < 6; ++a)
                {
                    for (int b = 0; b < 6; ++b)
                    {
                        double value = 0.0;
                        for (int row = 0; row < 6; ++row)
                        {
                            value += J[row][a] * J[row][b];
                        }
                        H[a * 6 + b] += w * value;
                    }
                    double grad = 0.0;
                    for (int row = 0; row < 6; ++row)
                    {
                        grad += J[row][a] * residual[static_cast<size_t>(row)];
                    }
                    g[a] += w * grad;
                }
                used += 3;
            }
        }

        if (used < 3) break;

        // LM 阻尼：加 lambda * diag(H)
        for (int d = 0; d < 6; ++d)
            H[d * 6 + d] += lambda * std::max(H[d * 6 + d], 1e-10);

        std::vector<double> rhs(6);
        for (int i = 0; i < 6; ++i) rhs[i] = -g[i];

        std::vector<double> dx;
        if (!solveLinearSystem(H, rhs, 6, &dx)) break;

        // 试探性应用步长
        Camera tmpCam = *cam;
        double delta[6] = {dx[0], dx[1], dx[2], dx[3], dx[4], dx[5]};
        tmpCam.applyDeltaPose(delta);

        double newCost = computeCameraCost(tmpCam, cameraIndex, tracks, points, opt);

        // 步长接受检查
        if (std::isfinite(newCost) && (newCost < currentCost + 1e-12 || !std::isfinite(currentCost))) {
            *cam = tmpCam;
            currentCost = newCost;
            lambda = std::max(lambda * 0.33, 1e-10);  // 成功 → 减小阻尼
            anyUpdated = true;
        } else {
            lambda *= 5.0;  // 失败 → 增大阻尼
            if (lambda > 1e8) break;
            continue;  // 重新计算 Jacobian
        }

        const double step = std::sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] +
                                      dx[3]*dx[3] + dx[4]*dx[4] + dx[5]*dx[5]);
        if (step < opt.stepTolerance) break;
    }

    return anyUpdated;
}

double averageSharedFocalScale(const std::vector<Camera> &cameras,
                               const std::vector<Camera> &referenceCameras)
{
    double sum = 0.0;
    int count = 0;
    const size_t n = std::min(cameras.size(), referenceCameras.size());
    for (size_t i = 0; i < n; ++i)
    {
        const Camera &camera = cameras[i];
        const Camera &reference = referenceCameras[i];
        if (std::isfinite(camera.focalX()) && std::isfinite(reference.focalX()) &&
            std::abs(reference.focalX()) > 1e-12)
        {
            sum += camera.focalX() / reference.focalX();
            ++count;
        }
        if (std::isfinite(camera.focalY()) && std::isfinite(reference.focalY()) &&
            std::abs(reference.focalY()) > 1e-12)
        {
            sum += camera.focalY() / reference.focalY();
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 1.0;
}

void applySharedFocalScale(std::vector<Camera> *cameras, double scale)
{
    if (!cameras || !(scale > 0.0) || !std::isfinite(scale))
    {
        return;
    }

    for (Camera &camera : *cameras)
    {
        camera.setIntrinsics(camera.focalX() * scale,
                             camera.focalY() * scale,
                             camera.principalX(),
                             camera.principalY());
    }
}

double computeSharedFocalAcceptanceCost(const std::vector<Camera> &cameras,
                                        const std::vector<BATrack> &tracks,
                                        const std::vector<BARefinedPoint> &points,
                                        double huberDelta)
{
    double cost = 0.0;
    int count = 0;
    for (size_t ti = 0; ti < tracks.size() && ti < points.size(); ++ti)
    {
        const BARefinedPoint &point = points[ti];
        if (!point.valid)
        {
            continue;
        }

        const BATrack &track = tracks[ti];
        for (const BAObservation &observation : track.observations)
        {
            if (observation.cameraIndex < 0 ||
                observation.cameraIndex >= static_cast<int>(cameras.size()))
            {
                continue;
            }
            double uv[2] = {0.0, 0.0};
            if (!projectPoint(cameras[static_cast<size_t>(observation.cameraIndex)],
                              point.point,
                              uv))
            {
                continue;
            }

            const double du = uv[0] - observation.u;
            const double dv = uv[1] - observation.v;
            const double residualNorm = std::sqrt(du * du + dv * dv);
            cost += observationWeight(observation) *
                    huberLoss(residualNorm, huberDelta);
            ++count;
        }
    }
    return count > 0 ? cost : std::numeric_limits<double>::infinity();
}

void recomputePointRmsForCurrentCameras(const std::vector<Camera> &cameras,
                                        const std::vector<BATrack> &tracks,
                                        std::vector<BARefinedPoint> *points)
{
    if (!points)
    {
        return;
    }

    const size_t n = std::min(tracks.size(), points->size());
    for (size_t i = 0; i < n; ++i)
    {
        BARefinedPoint &point = (*points)[i];
        if (!point.valid)
        {
            continue;
        }
        point.rmsAfter = computeTrackRms(cameras, tracks[i], point.point);
        point.valid = plamatrix::isFinite(plamatrix::Vec3<double>(point.point)) && std::isfinite(point.rmsAfter);
    }
}

struct SharedFocalOptimizationResult
{
    bool updated = false;
    int updatedCameraCount = 0;
    double sharedScale = 1.0;
};

SharedFocalOptimizationResult optimizeSharedFocalScale(std::vector<Camera> *cameras,
                                                       const std::vector<Camera> &referenceCameras,
                                                       const std::vector<BATrack> &tracks,
                                                       std::vector<BARefinedPoint> *points,
                                                       const BAOptions &opt)
{
    SharedFocalOptimizationResult result;
    if (!cameras || !points || !opt.refineSharedFocalLength ||
        cameras->size() < 2 || tracks.empty())
    {
        return result;
    }

    const double minScale = std::max(1e-3, opt.minSharedFocalScale);
    const double maxScale = std::max(minScale, opt.maxSharedFocalScale);
    const double maxStepScale = std::max(1.01, opt.maxSharedFocalStepScale);
    const double maxLogStep = std::log(maxStepScale);

    double currentScale = std::clamp(averageSharedFocalScale(*cameras, referenceCameras),
                                     minScale,
                                     maxScale);
    double currentCost =
        computeSharedFocalAcceptanceCost(*cameras, tracks, *points, opt.huberDelta);
    if (!std::isfinite(currentCost))
    {
        return result;
    }

    double lambda = std::max(opt.damping, 1e-12);
    for (int iter = 0; iter < std::max(1, opt.maxSharedFocalIterations); ++iter)
    {
        double H = 0.0;
        double g = 0.0;
        int used = 0;

        for (size_t ti = 0; ti < tracks.size() && ti < points->size(); ++ti)
        {
            const BARefinedPoint &point = (*points)[ti];
            if (!point.valid)
            {
                continue;
            }

            const BATrack &track = tracks[ti];
            for (const BAObservation &observation : track.observations)
            {
                if (observation.cameraIndex < 0 ||
                    observation.cameraIndex >= static_cast<int>(cameras->size()))
                {
                    continue;
                }

                const Camera &camera = (*cameras)[static_cast<size_t>(observation.cameraIndex)];
                double uv[2] = {0.0, 0.0};
                if (!projectPoint(camera, point.point, uv))
                {
                    continue;
                }

                const double residualU = uv[0] - observation.u;
                const double residualV = uv[1] - observation.v;
                const double residualNorm = std::sqrt(residualU * residualU + residualV * residualV);
                const double weight = observationWeight(observation) *
                    huberWeight(residualNorm, opt.huberDelta);
                if (!(weight > 0.0))
                {
                    continue;
                }

                // 用 log(scale) 做 1D 参数。对当前投影，d(u)/dlogf = u - cu，
                // d(v)/dlogf = v - cv；主点和畸变保持固定。
                const double jacU = uv[0] - camera.principalX();
                const double jacV = uv[1] - camera.principalY();
                H += weight * (jacU * jacU + jacV * jacV);
                g += weight * (jacU * residualU + jacV * residualV);
                ++used;
            }
        }

        if (used < 3 || !(H > 1e-12))
        {
            break;
        }

        double delta = -g / (H + lambda * std::max(H, 1e-12));
        delta = std::clamp(delta, -maxLogStep, maxLogStep);
        if (currentScale * std::exp(delta) < minScale)
        {
            delta = std::log(minScale / std::max(currentScale, 1e-12));
        }
        if (currentScale * std::exp(delta) > maxScale)
        {
            delta = std::log(maxScale / std::max(currentScale, 1e-12));
        }
        if (std::abs(delta) < opt.stepTolerance)
        {
            break;
        }

        std::vector<Camera> trialCameras = *cameras;
        const double trialStepScale = std::exp(delta);
        applySharedFocalScale(&trialCameras, trialStepScale);
        const double trialCost =
            computeSharedFocalAcceptanceCost(trialCameras, tracks, *points, opt.huberDelta);
        if (std::isfinite(trialCost) && trialCost < currentCost)
        {
            *cameras = std::move(trialCameras);
            currentScale *= trialStepScale;
            currentCost = trialCost;
            lambda = std::max(lambda * 0.33, 1e-12);
            result.updated = true;
        }
        else
        {
            lambda *= 5.0;
            if (lambda > 1e8)
            {
                break;
            }
        }
    }

    if (result.updated)
    {
        recomputePointRmsForCurrentCameras(*cameras, tracks, points);
        result.updatedCameraCount = static_cast<int>(cameras->size());
    }
    result.sharedScale = averageSharedFocalScale(*cameras, referenceCameras);
    return result;
}

} // namespace

// ---- 辅助：判断相机是否被 gauge 固定 ----
static bool isCameraFixed(int ci, const BAOptions &options)
{
    for (int fx : options.fixedCameraIndices) {
        if (fx == ci) return true;
    }
    return false;
}

static bool isCancelled(const BAOptions &options)
{
    return options.cancelFlag && options.cancelFlag->load(std::memory_order_relaxed);
}

int countObservations(const std::vector<BATrack> &tracks)
{
    int count = 0;
    for (const BATrack &track : tracks)
    {
        count += static_cast<int>(track.observations.size());
    }
    return count;
}

int calibrationGroupCount(const BAOptions &options, std::size_t cameraCount)
{
    if (cameraCount == 0 || options.cameraCalibrationGroupIds.empty())
    {
        return cameraCount == 0 ? 0 : 1;
    }
    std::vector<int> groups = options.cameraCalibrationGroupIds;
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    return static_cast<int>(groups.size());
}

void updateDerivedResultStats(BAResult &result)
{
    result.validTrackRatio = result.totalTracks > 0
                                 ? static_cast<double>(result.optimizedTracks) /
                                       static_cast<double>(result.totalTracks)
                                 : 0.0;
}

bool resultFailsQualityGate(const BAResult &result,
                            const BAOptions &options,
                            std::string *message)
{
    if (!options.enableBackendQualityGate)
    {
        return false;
    }

    if (!result.solutionUsable)
    {
        if (message)
        {
            *message = "质量门控拒绝: BA 求解状态不可写回";
        }
        return true;
    }

    if (!std::isfinite(result.meanRmsAfter))
    {
        if (message)
        {
            *message = "质量门控拒绝: BA 后 RMS 非有限";
        }
        return true;
    }

    if (result.totalTracks > 0 &&
        result.validTrackRatio < std::max(0.0, options.minAcceptedValidTrackRatio))
    {
        if (message)
        {
            *message = "质量门控拒绝: 有效 track 比例过低";
        }
        return true;
    }

    const double maxGrowth = std::max(0.0, options.maxAcceptedRmsGrowth);
    if (maxGrowth > 0.0 && std::isfinite(result.meanRmsBefore))
    {
        if (result.meanRmsBefore > 1e-12)
        {
            const double maxAccepted = result.meanRmsBefore * maxGrowth;
            if (result.meanRmsAfter > maxAccepted)
            {
                if (message)
                {
                    *message = "质量门控拒绝: BA 后 RMS 增长超过阈值";
                }
                return true;
            }
        }
        else if (result.meanRmsAfter > 1e-9)
        {
            if (message)
            {
                *message = "质量门控拒绝: 零残差输入被优化为非零残差";
            }
            return true;
        }
    }

    const double maxConstraintGrowth =
        std::max(1.0, options.maxAcceptedConstraintRmsGrowth);
    if (!detail::constraintRmsPassesQualityGate(
            result.laserConstraintCount,
            result.laserRmsBeforeMeters,
            result.laserRmsAfterMeters,
            maxConstraintGrowth,
            "LiDAR 点到面约束",
            message) ||
        !detail::constraintRmsPassesQualityGate(
            result.controlPointConstraintCount,
            result.controlPointRmsBeforeMeters,
            result.controlPointRmsAfterMeters,
            maxConstraintGrowth,
            "控制点约束",
            message) ||
        !detail::constraintRmsPassesQualityGate(
            result.scaleBarConstraintCount,
            result.scaleBarRmsBeforeMeters,
            result.scaleBarRmsAfterMeters,
            maxConstraintGrowth,
            "比例尺约束",
            message))
    {
        return true;
    }

    return false;
}

bool legacyIsBetterThanCandidate(const BAResult &candidate,
                                 const BAResult &legacy,
                                 const BAOptions &options,
                                 std::string *message)
{
    if (!options.enableBackendQualityGate ||
        !options.compareAutoBackendWithLegacy ||
        legacy.totalTracks <= 0 ||
        legacy.optimizedTracks <= 0 ||
        !std::isfinite(legacy.meanRmsAfter))
    {
        return false;
    }

    const double maxGrowth = std::max(1.0, options.maxAcceptedRmsGrowth);
    if (std::isfinite(candidate.meanRmsAfter) &&
        candidate.meanRmsAfter > legacy.meanRmsAfter * maxGrowth)
    {
        if (message)
        {
            *message = "质量门控拒绝: 候选后端 RMS 明显差于 legacy_cpu";
        }
        return true;
    }

    if (candidate.validTrackRatio + 1e-12 < legacy.validTrackRatio &&
        candidate.validTrackRatio < std::max(0.0, options.minAcceptedValidTrackRatio))
    {
        if (message)
        {
            *message = "质量门控拒绝: 候选后端有效 track 比例低于 legacy_cpu";
        }
        return true;
    }

    return false;
}

BAResult optimizePointsLegacyImpl(const std::vector<Camera> &cameras,
                                  const std::vector<BATrack> &tracks,
                                  const BAOptions &options)
{
    const auto totalStart = std::chrono::steady_clock::now();
    BAResult result;
    result.totalTracks = static_cast<int>(tracks.size());
    result.observationCount = countObservations(tracks);
    result.ceresLinearSolverName = "none";
    // 将相机列表拷贝到结果中，在优化过程中来回修改
    result.refinedCameras = cameras;
    result.points.resize(tracks.size());
    bool callbackAborted = false;

    auto finishResult = [&]() -> BAResult {
        detail::finalizeBundleAdjustResult(cameras, tracks, options, &result);
        if (result.solveStatus == BASolveStatus::NotRun)
        {
            if (isCancelled(options) || callbackAborted)
            {
                result.solveStatus = BASolveStatus::Cancelled;
                result.solutionUsable = false;
                if (result.backendMessage.empty())
                {
                    result.backendMessage = "legacy_cpu BA 已取消，结果不会写回";
                }
            }
            else if (result.optimizedTracks > 0)
            {
                result.solveStatus = BASolveStatus::Success;
                result.solutionUsable = true;
            }
            else
            {
                result.solveStatus = BASolveStatus::NumericalFailure;
                result.solutionUsable = false;
                if (result.backendMessage.empty())
                {
                    result.backendMessage = "legacy_cpu BA 未生成可用 track";
                }
            }
        }
        const auto totalEnd = std::chrono::steady_clock::now();
        result.totalSeconds = std::chrono::duration<double>(totalEnd - totalStart).count();
        result.solveSeconds = result.totalSeconds;
        updateDerivedResultStats(result);
        return result;
    };

    if (cameras.empty() || tracks.empty())
    {
        result.solveStatus = BASolveStatus::InvalidInput;
        result.backendMessage = "legacy_cpu BA 输入相机或 track 为空";
        return finishResult();
    }
    if (isCancelled(options))
    {
        result.solveStatus = BASolveStatus::Cancelled;
        result.backendMessage = "legacy_cpu BA 在启动前被取消";
        return finishResult();
    }

    if (options.enableLaserPlaneConstraints)
    {
        const LaserDistanceStats beforeLaser = computeLaserStatsForPoints(tracks, nullptr);
        result.laserConstraintCount = beforeLaser.count;
        result.laserRmsBeforeMeters = beforeLaser.rms;
        result.laserMedianBeforeMeters = beforeLaser.median;
    }
    if (options.enableControlPointConstraints)
    {
        const ControlPointDistanceStats beforeControl = computeControlPointStatsForPoints(tracks, nullptr);
        result.controlPointConstraintCount = beforeControl.count;
        result.controlPointRmsBeforeMeters = beforeControl.rms;
    }
    if (options.enableScaleBarConstraints)
    {
        const ScaleBarDistanceStats beforeScaleBars =
            computeScaleBarStatsForPoints(tracks, nullptr, options.scaleBarConstraints);
        result.scaleBarConstraintCount = beforeScaleBars.count;
        result.scaleBarRmsBeforeMeters = beforeScaleBars.rms;
    }

    // 配置 OpenMP 线程数
#ifdef _OPENMP
    if (options.numThreads > 0)
        omp_set_num_threads(options.numThreads);
#endif

    const int numTracks  = static_cast<int>(tracks.size());
    const int numCameras = static_cast<int>(cameras.size());

    // 加载初始点坐标并计算优化前重投影误差，用于后续对比
#pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < numTracks; ++i)
    {
        if (isCancelled(options)) continue;
        result.points[static_cast<size_t>(i)].point  = tracks[static_cast<size_t>(i)].initialPoint;
        result.points[static_cast<size_t>(i)].valid = plamatrix::isFinite(
            plamatrix::Vec3<double>(tracks[static_cast<size_t>(i)].initialPoint));
        result.points[static_cast<size_t>(i)].rmsBefore = computeTrackRms(
            result.refinedCameras, tracks[static_cast<size_t>(i)],
            tracks[static_cast<size_t>(i)].initialPoint);
    }
    if (isCancelled(options))
    {
        Logger::instance()->info("[BA] 已请求取消，跳过光束法平差优化");
        return finishResult();
    }

    // 外层交替优化：先点后相机，重复迭代直到达到最大次数或收敛
    double prevTotalCost = std::numeric_limits<double>::quiet_NaN();
    const int maxOuterIterations = std::max(1, options.maxIterations);

    for (int outer = 0; outer < maxOuterIterations; ++outer)
    {
        if (isCancelled(options))
        {
            Logger::instance()->info("[BA] 已请求取消，终止外层迭代");
            break;
        }

        // 阶段一：固定相机，并行优化每条轨迹的三维点坐标
        // 每个轨迹独立写入 result.points[i]，无读写竞争
        const std::vector<BARefinedPoint> pointSnapshot = result.points;
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < numTracks; ++i)
        {
            if (isCancelled(options)) continue;
            const size_t si = static_cast<size_t>(i);
            if (!result.points[si].valid) continue;   // 离群点跳过
            BARefinedPoint p = optimizeOnePoint(result.refinedCameras, tracks[si], options, i, &pointSnapshot);
            // 优化失败时回落到初始坐标
            if (!p.valid && plamatrix::isFinite(plamatrix::Vec3<double>(tracks[si].initialPoint)))
            {
                p.point    = tracks[si].initialPoint;
                p.rmsAfter = p.rmsBefore;
                p.valid    = true;
            }
            result.points[si] = p;
        }
        if (isCancelled(options))
        {
            Logger::instance()->info("[BA] 已请求取消，点优化阶段后终止");
            break;
        }

        // 阶段二：固定三维点，并行优化每台相机的 6-DOF 位姿
        // 每个相机独立写入 result.refinedCameras[ci]，无读写竞争
        if (options.refineCameraPose) {
            int refinedCnt = 0;
#pragma omp parallel for schedule(static) reduction(+:refinedCnt)
            for (int ci = 0; ci < numCameras; ++ci)
            {
                if (isCancelled(options)) continue;
                if (isCameraFixed(ci, options)) continue;   // gauge 固定
                if (optimizeOneCamera(&result.refinedCameras[static_cast<size_t>(ci)],
                                      ci, tracks, result.points, options))
                    ++refinedCnt;
            }
            result.refinedCameraCount = refinedCnt;
        }

        // 阶段二补充：无相机参数空三可只释放所有影像共享的焦距尺度。
        // 这里不优化主点/畸变，避免转台或弱基线数据把几何误差吸收到过多内参自由度里。
        if (options.refineSharedFocalLength)
        {
            const SharedFocalOptimizationResult focalResult =
                optimizeSharedFocalScale(&result.refinedCameras,
                                         cameras,
                                         tracks,
                                         &result.points,
                                         options);
            if (focalResult.updated)
            {
                result.refinedIntrinsicCount = focalResult.updatedCameraCount;
                result.refinedSharedFocalScale = focalResult.sharedScale;
            }
        }
        if (isCancelled(options))
        {
            Logger::instance()->info("[BA] 已请求取消，相机优化阶段后终止");
            break;
        }

        // 阶段三（可选）：离群点过滤——每轮结束后根据 rmsAfter 过滤高误差点
        if (options.enablePointFilter)
        {
            std::vector<double> currentRms;
            currentRms.reserve(result.points.size());
            for (const BARefinedPoint &point : result.points)
            {
                if (point.valid)
                {
                    currentRms.push_back(point.rmsAfter);
                }
            }
            const double adaptThresh =
                detail::adaptivePointFilterThreshold(
                    currentRms,
                    options.filterMaxReprojError,
                    options.filterSigmaFactor);

#pragma omp parallel for schedule(static)
            for (int i = 0; i < numTracks; ++i)
            {
                if (isCancelled(options)) continue;
                auto &p = result.points[static_cast<size_t>(i)];
                if (!p.valid) continue;
                if (std::isfinite(p.rmsAfter) && p.rmsAfter > adaptThresh)
                {
                    p.valid = false;   // 标记为离群点
                }
            }
        }

        // ── 外层收敛检查：全局 cost 变化 < 1% 则提前终止 ──
        {
            double totalCost = 0.0;
            int costCnt = 0;
            for (int i = 0; i < numTracks; ++i)
            {
                const auto &p = result.points[static_cast<size_t>(i)];
                if (p.valid && std::isfinite(p.rmsAfter))
                {
                    totalCost += p.rmsAfter;
                    ++costCnt;
                }
            }
            double avgCost = (costCnt > 0) ? totalCost / costCnt : 0.0;
            double relChange = std::numeric_limits<double>::infinity();
            if (std::isfinite(prevTotalCost))
            {
                const double costScale = std::max(
                    {std::fabs(prevTotalCost), std::fabs(avgCost), 1e-12});
                relChange = std::fabs(prevTotalCost - avgCost) / costScale;
            }

            if (options.logIterationProgress)
            {
                Logger::instance()->infof("[BA] iter %d/%d: avgRMS=%.6f, relChange=%.6f, validPts=%d", outer + 1,
                                           maxOuterIterations, avgCost, relChange, costCnt);
            }

            if (options.progressCallback &&
                !options.progressCallback(outer + 1, maxOuterIterations, avgCost, costCnt))
            {
                Logger::instance()->info("[BA] 进度回调请求终止光束法平差");
                callbackAborted = true;
                break;
            }

            if (outer >= 2 && relChange < 1e-4) {
                Logger::instance()->infof("[BA] 收敛: relChange=%.2e < 1e-4, 终止", relChange);
                break;
            }
            prevTotalCost = avgCost;
        }
    }

    // 汇总全局统计数据：均值 RMS、有效轨迹数
    double sumBefore = 0.0;
    double sumAfter  = 0.0;
    int cntBefore = 0;
    int cntAfter  = 0;

    for (size_t i = 0; i < tracks.size(); ++i) {
        BARefinedPoint &p = result.points[i];
        if (std::isfinite(p.rmsBefore)) {
            sumBefore += p.rmsBefore;
            ++cntBefore;
        }

        // 用最终相机位姿重新计算 rmsAfter，最终正深度和离群点判定由统一终结器完成。
        p.rmsAfter = computeTrackRms(result.refinedCameras, tracks[i], p.point);
        p.valid = plamatrix::isFinite(plamatrix::Vec3<double>(p.point)) && std::isfinite(p.rmsAfter);
        if (p.valid) {
            ++result.optimizedTracks;
            sumAfter += p.rmsAfter;
            ++cntAfter;
        }
    }

    result.meanRmsBefore = (cntBefore > 0) ? (sumBefore / cntBefore) : 0.0;
    result.meanRmsAfter  = (cntAfter  > 0) ? (sumAfter  / cntAfter)  : 0.0;
    result.refinedSharedFocalScale = averageSharedFocalScale(result.refinedCameras, cameras);
    if (options.refineSharedFocalLength &&
        std::abs(result.refinedSharedFocalScale - 1.0) > 1e-6 &&
        result.refinedIntrinsicCount == 0)
    {
        result.refinedIntrinsicCount = static_cast<int>(result.refinedCameras.size());
    }
    if (options.refineSharedFocalLength)
    {
        result.refinedCalibrationGroupCount = 1;
        result.selfCalibrationStagesRun = 1;
    }

    if (options.enableLaserPlaneConstraints)
    {
        const LaserDistanceStats afterLaser = computeLaserStatsForPoints(tracks, &result.points);
        result.laserConstraintCount = afterLaser.count;
        result.laserRmsAfterMeters = afterLaser.rms;
        result.laserMedianAfterMeters = afterLaser.median;
    }
    if (options.enableControlPointConstraints)
    {
        const ControlPointDistanceStats afterControl = computeControlPointStatsForPoints(tracks, &result.points);
        result.controlPointConstraintCount = afterControl.count;
        result.controlPointRmsAfterMeters = afterControl.rms;
    }
    if (options.enableScaleBarConstraints)
    {
        const ScaleBarDistanceStats afterScaleBars =
            computeScaleBarStatsForPoints(tracks, &result.points, options.scaleBarConstraints);
        result.scaleBarConstraintCount = afterScaleBars.count;
        result.scaleBarRmsAfterMeters = afterScaleBars.rms;
    }
    return finishResult();
}

const char *BundleAdjust::backendName(BABackend backend)
{
    switch (backend)
    {
    case BABackend::Auto:
        return "auto";
    case BABackend::LegacyCpu:
        return "legacy_cpu";
    case BABackend::CeresCpu:
        return "ceres_cpu";
    case BABackend::CeresCuda:
        return "ceres_cuda";
    case BABackend::NativeCuda:
        return "native_cuda";
    }
    return "unknown";
}

const char *BundleAdjust::solveStatusName(BASolveStatus status)
{
    switch (status)
    {
    case BASolveStatus::NotRun:
        return "not_run";
    case BASolveStatus::Success:
        return "success";
    case BASolveStatus::NoConvergence:
        return "no_convergence";
    case BASolveStatus::Cancelled:
        return "cancelled";
    case BASolveStatus::InvalidInput:
        return "invalid_input";
    case BASolveStatus::UnsupportedConfiguration:
        return "unsupported_configuration";
    case BASolveStatus::BackendUnavailable:
        return "backend_unavailable";
    case BASolveStatus::NumericalFailure:
        return "numerical_failure";
    }
    return "unknown";
}

BABackendCapabilities BundleAdjust::backendCapabilities(BABackend backend)
{
    switch (backend)
    {
    case BABackend::Auto:
        return {true, true, true, true, true, true, true};
    case BABackend::LegacyCpu:
        return {true, true, true, false, false, false, true};
    case BABackend::CeresCpu:
    case BABackend::CeresCuda:
        return {true, true, true, true, true, true, true};
    case BABackend::NativeCuda:
        // 当前 native CUDA kernel 只联合调度各 track 的三维点更新，尚未求解相机块。
        return {true, false, false, false, false, false, false};
    }
    return {};
}

bool BundleAdjust::isBackendAvailable(BABackend backend)
{
    switch (backend)
    {
    case BABackend::Auto:
        return true;
    case BABackend::LegacyCpu:
        return true;
    case BABackend::CeresCpu:
        return detail::isCeresBackendCompiled();
    case BABackend::CeresCuda:
        return detail::isCeresCudaBackendCompiled();
    case BABackend::NativeCuda:
    {
        std::string message;
        return detail::isNativeCudaRuntimeAvailable(0, &message);
    }
    }
    return false;
}

BAProblemStats BundleAdjust::summarizeProblem(const std::vector<Camera> &cameras,
                                              const std::vector<BATrack> &tracks)
{
    BAProblemStats stats;
    stats.cameraCount = static_cast<int>(cameras.size());
    stats.trackCount = static_cast<int>(tracks.size());
    stats.observationCount = countObservations(tracks);
    return stats;
}

BABackendDecision BundleAdjust::decideBackendForProblem(const BAProblemStats &stats,
                                                        const BAOptions &options)
{
    if (options.backend != BABackend::Auto)
    {
        return {options.backend, "explicit_backend"};
    }
    if (options.cameraPlaneConstraint.enabled &&
        isBackendAvailable(BABackend::CeresCpu))
    {
        return {BABackend::CeresCpu, "camera_plane_constraint_requires_ceres"};
    }
    const bool refineExtendedIntrinsics =
        options.refineSharedFocalAspectRatio ||
        options.refineSharedPrincipalPoint ||
        options.refineSharedRadialDistortion;
    if (refineExtendedIntrinsics)
    {
        if (options.refineCameraPose &&
            isBackendAvailable(BABackend::CeresCuda) &&
            stats.cameraCount >= std::max(1, options.minCeresCudaCameras) &&
            stats.observationCount >= std::max(1, options.minCeresCudaObservations))
        {
            return {BABackend::CeresCuda,
                    "large_joint_shared_intrinsics_uses_ceres_cuda"};
        }
        if (isBackendAvailable(BABackend::CeresCpu))
        {
            return {BABackend::CeresCpu, "joint_shared_intrinsics_requires_ceres"};
        }
    }
    if (!options.refineCameraPose)
    {
        return {BABackend::LegacyCpu, "point_only_prefers_legacy_cpu"};
    }
    if (isBackendAvailable(BABackend::CeresCuda) &&
        stats.cameraCount >= std::max(1, options.minCeresCudaCameras) &&
        stats.observationCount >= std::max(1, options.minCeresCudaObservations))
    {
        return {BABackend::CeresCuda, "ceres_cuda_problem_size"};
    }
    if (options.refineSharedFocalLength &&
        isBackendAvailable(BABackend::CeresCpu))
    {
        // 共享焦距必须和相机、三维点放在同一个非线性问题中求解。
        // 即使问题较小，也优先使用联合 Ceres，而不是退回 Legacy 分块交替更新。
        return {BABackend::CeresCpu, "joint_shared_focal_requires_ceres"};
    }
    if (isBackendAvailable(BABackend::CeresCpu) &&
        stats.observationCount >= std::max(1, options.minCeresCpuObservations))
    {
        return {BABackend::CeresCpu, "ceres_cpu_problem_size"};
    }
    return {BABackend::LegacyCpu, "below_accelerated_problem_size"};
}

BABackend BundleAdjust::selectBackendForProblem(const BAProblemStats &stats,
                                                 const BAOptions &options)
{
    return decideBackendForProblem(stats, options).backend;
}

BAResult BundleAdjust::optimizePoints(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &requestedOptions)
{
    BAOptions normalizedOptions;
    const detail::BundleAdjustValidationResult validation =
        detail::validateAndNormalizeBundleAdjustOptions(
            cameras, tracks, requestedOptions, &normalizedOptions);
    if (!validation.ok)
    {
        BAResult result;
        result.requestedBackend = requestedOptions.backend;
        result.usedBackend = requestedOptions.backend;
        result.solveStatus = validation.status;
        result.solutionUsable = false;
        result.backendMessage = validation.message;
        result.totalTracks = static_cast<int>(tracks.size());
        result.observationCount = countObservations(tracks);
        result.refinedCameras = cameras;
        result.points.resize(tracks.size());
        updateDerivedResultStats(result);
        return result;
    }
    const BAOptions &options = normalizedOptions;

    auto runLegacy = [&](const std::string &fallbackMessage) {
        if (options.cameraPlaneConstraint.enabled)
        {
            BAResult result;
            result.requestedBackend = options.backend;
            result.usedBackend = BABackend::LegacyCpu;
            result.solveStatus = BASolveStatus::UnsupportedConfiguration;
            result.solutionUsable = false;
            result.backendFallback = options.backend != BABackend::LegacyCpu;
            result.backendMessage = fallbackMessage +
                                    "；legacy_cpu 不支持相机中心平面约束，未执行无约束回退";
            result.backendSelectionReason = fallbackMessage;
            result.totalTracks = static_cast<int>(tracks.size());
            result.observationCount = countObservations(tracks);
            result.refinedCameras = cameras;
            result.points.resize(tracks.size());
            updateDerivedResultStats(result);
            return result;
        }
        BAResult result = optimizePointsLegacyImpl(cameras, tracks, options);
        result.requestedBackend = options.backend;
        result.usedBackend = BABackend::LegacyCpu;
        result.usedGpu = false;
        result.backendFallback = options.backend != BABackend::LegacyCpu;
        result.backendMessage = fallbackMessage;
        result.backendSelectionReason = fallbackMessage;
        updateDerivedResultStats(result);
        return result;
    };

    auto pointOnlyCeresTooLarge = [&]() {
        if (options.refineCameraPose)
        {
            return false;
        }
        const BAProblemStats stats = summarizeProblem(cameras, tracks);
        return stats.observationCount > std::max(1, options.maxCeresPointOnlyObservations);
    };

    if (options.backend == BABackend::Auto)
    {
        const BAProblemStats stats = summarizeProblem(cameras, tracks);
        BAOptions selectedOptions = options;
        const BABackendDecision decision = decideBackendForProblem(stats, options);
        selectedOptions.backend = decision.backend;
        const std::string selectedName = backendName(selectedOptions.backend);

        if (selectedOptions.backend == BABackend::LegacyCpu)
        {
            BAResult result = optimizePointsLegacyImpl(cameras, tracks, selectedOptions);
            result.requestedBackend = BABackend::Auto;
            result.usedBackend = BABackend::LegacyCpu;
            result.usedGpu = false;
            result.backendFallback = false;
            result.backendSelectionReason = "自动选择 legacy_cpu: " + decision.reason;
            result.backendMessage = result.backendSelectionReason;
            updateDerivedResultStats(result);
            return result;
        }

        BAResult candidate = optimizePoints(cameras, tracks, selectedOptions);
        candidate.requestedBackend = BABackend::Auto;
        updateDerivedResultStats(candidate);

        std::string qualityMessage;
        bool rejectCandidate = resultFailsQualityGate(candidate, options, &qualityMessage);
        BAResult legacy;
        bool comparedWithLegacy = false;
        if (!rejectCandidate &&
            options.enableBackendQualityGate &&
            options.compareAutoBackendWithLegacy &&
            !options.cameraPlaneConstraint.enabled)
        {
            BAOptions legacyOptions = options;
            legacyOptions.backend = BABackend::LegacyCpu;
            legacy = optimizePointsLegacyImpl(cameras, tracks, legacyOptions);
            legacy.requestedBackend = BABackend::Auto;
            legacy.usedBackend = BABackend::LegacyCpu;
            legacy.usedGpu = false;
            updateDerivedResultStats(legacy);
            comparedWithLegacy = true;
            rejectCandidate = legacyIsBetterThanCandidate(candidate, legacy, options, &qualityMessage);
        }

        if (rejectCandidate)
        {
            if (legacy.totalTracks <= 0)
            {
                BAOptions legacyOptions = options;
                legacyOptions.backend = BABackend::LegacyCpu;
                legacy = optimizePointsLegacyImpl(cameras, tracks, legacyOptions);
                legacy.requestedBackend = BABackend::Auto;
                legacy.usedBackend = BABackend::LegacyCpu;
                legacy.usedGpu = false;
                updateDerivedResultStats(legacy);
            }
            legacy.setupSeconds += candidate.setupSeconds;
            legacy.solveSeconds += candidate.solveSeconds;
            legacy.totalSeconds += candidate.totalSeconds;
            legacy.backendFallback = true;
            legacy.qualityGateRejected = true;
            legacy.qualityGateMessage = qualityMessage;
            legacy.backendSelectionReason = "自动候选 " + selectedName +
                                            " 被质量门控拒绝，回退 legacy_cpu";
            legacy.backendMessage = legacy.backendSelectionReason + "；" + qualityMessage;
            return legacy;
        }

        candidate.backendSelectionReason = "自动选择 " + selectedName +
                                           ": 通过 BA 质量门控";
        if (comparedWithLegacy)
        {
            candidate.setupSeconds += legacy.setupSeconds;
            candidate.solveSeconds += legacy.solveSeconds;
            candidate.totalSeconds += legacy.totalSeconds;
        }
        if (!candidate.backendMessage.empty())
        {
            candidate.backendMessage = candidate.backendSelectionReason + "；" + candidate.backendMessage;
        }
        else
        {
            candidate.backendMessage = candidate.backendSelectionReason;
        }
        return candidate;
    }

    if (options.backend == BABackend::LegacyCpu)
    {
        const bool requiresCeresFeatures =
            options.refineSharedFocalAspectRatio ||
            options.refineSharedPrincipalPoint ||
            options.refineSharedRadialDistortion ||
            options.cameraPlaneConstraint.enabled ||
            (options.refineSharedFocalLength &&
             calibrationGroupCount(options, cameras.size()) > 1);
        if (requiresCeresFeatures)
        {
            const std::string message =
                options.cameraPlaneConstraint.enabled
                    ? "legacy_cpu 不支持相机中心平面约束，"
                    : "legacy_cpu 不支持请求的联合共享内参优化，";
            if (detail::isCeresBackendCompiled() &&
                options.allowBackendFallback)
            {
                BAOptions ceresOptions = options;
                ceresOptions.backend = BABackend::CeresCpu;
                BAResult result =
                    optimizePoints(cameras, tracks, ceresOptions);
                result.requestedBackend = BABackend::LegacyCpu;
                result.backendFallback = true;
                result.backendMessage =
                    message + "已切换到 ceres_cpu；" +
                    result.backendMessage;
                return result;
            }

            BAResult result;
            result.requestedBackend = BABackend::LegacyCpu;
            result.usedBackend = BABackend::LegacyCpu;
            result.solveStatus =
                BASolveStatus::UnsupportedConfiguration;
            result.solutionUsable = false;
            result.backendMessage =
                message + "当前无法回退到 ceres_cpu";
            result.totalTracks = static_cast<int>(tracks.size());
            result.observationCount = countObservations(tracks);
            result.refinedCameras = cameras;
            result.points.resize(tracks.size());
            return result;
        }
        BAResult result = optimizePointsLegacyImpl(cameras, tracks, options);
        result.requestedBackend = BABackend::LegacyCpu;
        result.usedBackend = BABackend::LegacyCpu;
        result.usedGpu = false;
        updateDerivedResultStats(result);
        return result;
    }

    if (options.backend == BABackend::NativeCuda)
    {
        const BABackendCapabilities capabilities = backendCapabilities(BABackend::NativeCuda);
        const bool unsupportedConfiguration =
            (options.refineCameraPose && !capabilities.refinesCameraPose) ||
            (options.refineSharedFocalLength && !capabilities.refinesSharedFocalLength) ||
            (options.refineSharedFocalAspectRatio &&
             !capabilities.refinesSharedFocalAspectRatio) ||
            (options.refineSharedPrincipalPoint &&
             !capabilities.refinesSharedPrincipalPoint) ||
            (options.refineSharedRadialDistortion &&
             !capabilities.refinesSharedRadialDistortion) ||
            ((options.enableLaserPlaneConstraints ||
              options.enableControlPointConstraints ||
              options.enableScaleBarConstraints ||
              options.cameraPlaneConstraint.enabled ||
              !options.cameraPosePriors.empty()) &&
             !capabilities.supportsSoftConstraints);
        if (unsupportedConfiguration)
        {
            const std::string message =
                "native_cuda 当前仅支持固定相机的三维点优化，不能执行请求的联合 BA";
            if (!options.allowBackendFallback)
            {
                BAResult result;
                result.requestedBackend = BABackend::NativeCuda;
                result.usedBackend = BABackend::NativeCuda;
                result.solveStatus = BASolveStatus::UnsupportedConfiguration;
                result.backendMessage = message + "，且当前禁止回退";
                result.totalTracks = static_cast<int>(tracks.size());
                result.refinedCameras = cameras;
                updateDerivedResultStats(result);
                return result;
            }
            return runLegacy(message + "，已回退到 legacy_cpu");
        }

        std::string message;
        if (!detail::isNativeCudaRuntimeAvailable(options.nativeCudaDevice, &message))
        {
            if (!options.allowBackendFallback)
            {
                BAResult result;
                result.requestedBackend = BABackend::NativeCuda;
                result.usedBackend = BABackend::NativeCuda;
                result.solveStatus = BASolveStatus::BackendUnavailable;
                result.backendMessage = message + "，且禁止回退";
                updateDerivedResultStats(result);
                return result;
            }
            return runLegacy(message);
        }

        BAResult result = detail::optimizePointsWithNativeCuda(cameras, tracks, options);
        result.requestedBackend = BABackend::NativeCuda;
        updateDerivedResultStats(result);
        const bool nativeCudaFailed =
            !result.solutionUsable ||
            !result.usedGpu ||
            result.optimizedTracks == 0 ||
            !std::isfinite(result.meanRmsAfter);
        if (nativeCudaFailed &&
            result.solveStatus != BASolveStatus::Cancelled &&
            options.allowBackendFallback)
        {
            const std::string message = result.backendMessage.empty()
                                            ? "native_cuda 求解失败，已回退到 legacy_cpu"
                                            : result.backendMessage + "，已回退到 legacy_cpu";
            BAResult fallback = runLegacy(message);
            fallback.setupSeconds += result.setupSeconds;
            fallback.solveSeconds += result.solveSeconds;
            fallback.totalSeconds += result.totalSeconds;
            return fallback;
        }
        if (nativeCudaFailed && !options.allowBackendFallback && result.backendMessage.empty())
        {
            result.backendMessage = "native_cuda 求解失败，且当前禁止回退";
        }
        return result;
    }

    if (options.backend == BABackend::CeresCpu)
    {
        if (pointOnlyCeresTooLarge())
        {
            const std::string message =
                "point-only Ceres BA 观测数超过安全阈值，已回退到 legacy_cpu";
            if (!options.allowBackendFallback)
            {
                BAResult result;
                result.requestedBackend = BABackend::CeresCpu;
                result.usedBackend = BABackend::CeresCpu;
                result.solveStatus = BASolveStatus::UnsupportedConfiguration;
                result.backendMessage = message + "，但当前禁止回退";
                updateDerivedResultStats(result);
                return result;
            }
            return runLegacy(message);
        }
        if (detail::isCeresBackendCompiled())
        {
            BAResult result = detail::optimizePointsWithCeres(cameras, tracks, options, false);
            result.requestedBackend = BABackend::CeresCpu;
            result.usedBackend = BABackend::CeresCpu;
            result.usedGpu = false;
            updateDerivedResultStats(result);
            if (!result.solutionUsable &&
                result.solveStatus != BASolveStatus::Cancelled &&
                options.allowBackendFallback)
            {
                const std::string message = result.backendMessage.empty()
                                                ? "Ceres CPU 求解失败，已回退到 legacy_cpu"
                                                : result.backendMessage + "，已回退到 legacy_cpu";
                BAResult fallback = runLegacy(message);
                fallback.setupSeconds += result.setupSeconds;
                fallback.solveSeconds += result.solveSeconds;
                fallback.totalSeconds += result.totalSeconds;
                return fallback;
            }
            return result;
        }
        if (!options.allowBackendFallback)
        {
            BAResult result;
            result.requestedBackend = BABackend::CeresCpu;
            result.usedBackend = BABackend::CeresCpu;
            result.solveStatus = BASolveStatus::BackendUnavailable;
            result.backendMessage = "Ceres CPU 后端不可用，且禁止回退";
            updateDerivedResultStats(result);
            return result;
        }
        return runLegacy("Ceres CPU 后端不可用，已回退到 legacy_cpu");
    }

    if (options.backend == BABackend::CeresCuda)
    {
        if (pointOnlyCeresTooLarge())
        {
            const std::string message =
                "point-only Ceres CUDA BA 观测数超过安全阈值，已回退到 legacy_cpu";
            if (!options.allowBackendFallback)
            {
                BAResult result;
                result.requestedBackend = BABackend::CeresCuda;
                result.usedBackend = BABackend::CeresCuda;
                result.solveStatus = BASolveStatus::UnsupportedConfiguration;
                result.backendMessage = message + "，但当前禁止回退";
                updateDerivedResultStats(result);
                return result;
            }
            return runLegacy(message);
        }
        if (detail::isCeresCudaBackendCompiled() &&
            static_cast<int>(cameras.size()) >= std::max(1, options.minCeresCudaCameras))
        {
            BAResult result = detail::optimizePointsWithCeres(cameras, tracks, options, true);
            result.requestedBackend = BABackend::CeresCuda;
            updateDerivedResultStats(result);
            if (!result.solutionUsable &&
                result.solveStatus != BASolveStatus::Cancelled &&
                options.allowBackendFallback)
            {
                const std::string message = result.backendMessage.empty()
                                                ? "Ceres CUDA 求解失败，已回退到 legacy_cpu"
                                                : result.backendMessage + "，已回退到 legacy_cpu";
                BAResult fallback = runLegacy(message);
                fallback.setupSeconds += result.setupSeconds;
                fallback.solveSeconds += result.solveSeconds;
                fallback.totalSeconds += result.totalSeconds;
                return fallback;
            }
            return result;
        }

        if (!options.allowBackendFallback)
        {
            BAResult result;
            result.requestedBackend = BABackend::CeresCuda;
            result.usedBackend = BABackend::CeresCuda;
            result.solveStatus = BASolveStatus::BackendUnavailable;
            result.backendMessage = "Ceres CUDA 后端不可用或未达到 GPU 求解阈值，且禁止回退";
            updateDerivedResultStats(result);
            return result;
        }

        if (detail::isCeresBackendCompiled())
        {
            BAOptions cpuOptions = options;
            cpuOptions.backend = BABackend::CeresCpu;
            BAResult result = detail::optimizePointsWithCeres(cameras, tracks, cpuOptions, false);
            result.requestedBackend = BABackend::CeresCuda;
            result.usedBackend = BABackend::CeresCpu;
            result.usedGpu = false;
            result.backendFallback = true;
            if (!detail::isCeresCudaBackendCompiled())
            {
                result.backendMessage = "Ceres 未启用 CUDA，已回退到 ceres_cpu；" + result.backendMessage;
            }
            else
            {
                result.backendMessage = "相机数量低于 Ceres CUDA 阈值，已回退到 ceres_cpu；" + result.backendMessage;
            }
            updateDerivedResultStats(result);
            if (!result.solutionUsable &&
                result.solveStatus != BASolveStatus::Cancelled &&
                options.allowBackendFallback)
            {
                const std::string message = result.backendMessage.empty()
                                                ? "Ceres CPU 回退求解失败，已继续回退到 legacy_cpu"
                                                : result.backendMessage + "，已继续回退到 legacy_cpu";
                BAResult fallback = runLegacy(message);
                fallback.setupSeconds += result.setupSeconds;
                fallback.solveSeconds += result.solveSeconds;
                fallback.totalSeconds += result.totalSeconds;
                return fallback;
            }
            return result;
        }
        return runLegacy("Ceres CUDA/CPU 后端均不可用，已回退到 legacy_cpu");
    }

    return runLegacy("未知 BA 后端，已回退到 legacy_cpu");
}

} // namespace xjw

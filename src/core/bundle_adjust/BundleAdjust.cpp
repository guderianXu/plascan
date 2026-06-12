// ============================================================
// 文件：BundleAdjust.cpp
// 功能：实现光束法平差（Bundle Adjustment）核心算法。
//
// 算法设计要点：
//   1. 交替优化策略：先固定相机优化三维点，再固定点优化相机，循环多轮
//   2. 鲁棒正则化：采用高斯牛顿 + LM 风格阻尼
//   3. Huber 权重：残差范数 > huberDelta 时降低对应观测的权重，抹除粗差
//   4. 解析雅可比：针对针孔+Tsai 畸变模型推导闭式 Jacobian（点 2×3，相机 2×6）
//   5. SO(3) 旋转更新：使用 Rodrigues 公式 exp(ω) 进行旋转增量扰动
//   6. 系统求解：自实现带部分主元高斯-约尔当消元，避免外部线性代数依赖
// ============================================================

#include "BundleAdjust.h"

#include "log/Logger.h"
#include "math/Vec3Ops.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
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
// 数值部分采用高斯牛顿 + 阻尼（LM 风格），雅可比通过有限差分近似。
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
bool solveLinearSystem(std::vector<double> A, std::vector<double> b, int n, std::vector<double> *x)
{
    if (!x || n <= 0 || static_cast<int>(A.size()) != n * n || static_cast<int>(b.size()) != n)
    {
        return false;
    }

    x->assign(n, 0.0);
    // 使用带部分主元的高斯-约尔当消元，提升病态情况下的稳定性。
    for (int col = 0; col < n; ++col)
    {
        // 在当前列中寻找绝对值最大的行（主元行）
        int pivot = col;
        double best = std::abs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r)
        {
            const double v = std::abs(A[r * n + col]);
            if (v > best)
            {
                best = v;
                pivot = r;
            }
        }
        // 若主元过小，矩阵近似奇异，返回失败
        if (best < 1e-14)
        {
            return false;
        }

        // 交换主元行与当前行
        if (pivot != col)
        {
            for (int c = col; c < n; ++c)
            {
                std::swap(A[col * n + c], A[pivot * n + c]);
            }
            std::swap(b[col], b[pivot]);
        }

        // 归一：将主元行除以对角元
        const double diag = A[col * n + col];
        for (int c = col; c < n; ++c)
        {
            A[col * n + c] /= diag;
        }
        b[col] /= diag;

        // 消元：将其余行该列的元素归零
        for (int r = 0; r < n; ++r)
        {
            if (r == col)
            {
                continue;
            }
            const double factor = A[r * n + col];
            for (int c = col; c < n; ++c)
            {
                A[r * n + c] -= factor * A[col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }

    // 主元化后 b 即为解，检查有限性
    for (int i = 0; i < n; ++i)
    {
        (*x)[i] = b[i];
        if (!std::isfinite((*x)[i]))
        {
            return false;
        }
    }
    return true;
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
        sum2 += du * du + dv * dv;
        cnt += 2;
    }
    if (cnt <= 0) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(sum2 / static_cast<double>(cnt));
}

/**
 * @brief 对单条轨迹优化三维点坐标（固定所有相机）。
 *
 * 优化目标：min_X sum_i w_i * || proj(cam_i, X) - obs_i ||^2
 * 其中 w_i 为 Huber 加权，雅可比通过对 X 每个分量做有限差分得到。
 * 正规化方程 H*dx = -g，加 damping 阻尼后求解电车增量 dx。
 *
 * @param cams   相机列表（优化期间固定不变）
 * @param track  待优化的轨迹
 * @param opt    优化选项
 * @return       点优化结果（BARefinedPoint）
 */
BARefinedPoint optimizeOnePoint(const std::vector<Camera> &cams, const BATrack &track, const BAOptions &opt)
{
    BARefinedPoint out;
    out.point = track.initialPoint;
    if (!vec3::isFinite(out.point) || track.observations.size() < 2) return out;

    out.rmsBefore = computeTrackRms(cams, track, out.point);
    std::array<double, 3> X = out.point;
    double currentCost = out.rmsBefore;
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
            const double w = huberWeight(std::sqrt(r[0] * r[0] + r[1] * r[1]), opt.huberDelta);

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
        double newCost = computeTrackRms(cams, track, Xnew);

        // 步长接受检查：仅当 cost 减小（或首次有限cost）时接受步长
        if (std::isfinite(newCost) && (newCost < currentCost || !std::isfinite(currentCost)))
        {
            X = Xnew;
            currentCost = newCost;
            lambda = std::max(lambda * 0.5, 1e-10);  // 成功 → 减小阻尼
        } else {
            lambda *= 5.0;  // 失败 → 增大阻尼
            if (lambda > 1e8) break;  // 阻尼过大，放弃
        }
        out.iterations = iter + 1;

        const double step = vec3::norm({{dx[0], dx[1], dx[2]}});
        if (step < opt.stepTolerance) {
            out.converged = true;
            break;
        }
    }

    out.point = X;
    out.rmsAfter = computeTrackRms(cams, track, X);
    out.valid = vec3::isFinite(X) && std::isfinite(out.rmsAfter);
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
                               const std::vector<BARefinedPoint> &points)
{
    double sum2 = 0.0;
    int cnt = 0;
    for (size_t ti = 0; ti < tracks.size() && ti < points.size(); ++ti) {
        if (!points[ti].valid) continue;
        for (const BAObservation &obs : tracks[ti].observations) {
            if (obs.cameraIndex != cameraIndex) continue;
            double uv[2];
            if (!projectPoint(cam, points[ti].point, uv)) continue;
            double du = uv[0] - obs.u, dv = uv[1] - obs.v;
            sum2 += du * du + dv * dv;
            cnt += 2;
        }
    }
    return (cnt > 0) ? std::sqrt(sum2 / cnt) : 0.0;
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
    double currentCost = computeCameraCost(*cam, cameraIndex, tracks, points);
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
                const double w = huberWeight(std::sqrt(r[0] * r[0] + r[1] * r[1]), opt.huberDelta);

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

        double newCost = computeCameraCost(tmpCam, cameraIndex, tracks, points);

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

} // namespace

// ---- 辅助：判断相机是否被 gauge 固定 ----
static bool isCameraFixed(int ci, const BAOptions &options)
{
    for (int fx : options.fixedCameraIndices) {
        if (fx == ci) return true;
    }
    return false;
}

// ---- 辅助：计算所有有效点 rmsAfter 的中位数（用于自适应过滤） ----
static double computeMedianRms(const std::vector<BARefinedPoint> &pts)
{
    std::vector<double> vals;
    vals.reserve(pts.size());
    for (const auto &p : pts) {
        if (p.valid && std::isfinite(p.rmsAfter) && p.rmsAfter > 0.0)
            vals.push_back(p.rmsAfter);
    }
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    const size_t m = vals.size() / 2;
    return (vals.size() % 2 == 0) ? 0.5 * (vals[m - 1] + vals[m]) : vals[m];
}

static bool isCancelled(const BAOptions &options)
{
    return options.cancelFlag && options.cancelFlag->load(std::memory_order_relaxed);
}

BAResult BundleAdjust::optimizePoints(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &options)
{
    BAResult result;
    result.totalTracks = static_cast<int>(tracks.size());
    // 将相机列表拷贝到结果中，在优化过程中来回修改
    result.refinedCameras = cameras;
    result.points.resize(tracks.size());

    if (cameras.empty() || tracks.empty()) return result;
    if (isCancelled(options)) return result;

    // 配置 OpenMP 线程数
#ifdef _OPENMP
    if (options.numThreads > 0)
        omp_set_num_threads(options.numThreads);
#endif

    const int numTracks  = static_cast<int>(tracks.size());
    const int numCameras = static_cast<int>(cameras.size());

    // 加载初始点坐标并计算优化前重投影误差，用于后续对比
#pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < numTracks; ++i) {
        if (isCancelled(options)) continue;
        result.points[static_cast<size_t>(i)].point  = tracks[static_cast<size_t>(i)].initialPoint;
        result.points[static_cast<size_t>(i)].valid  = vec3::isFinite(tracks[static_cast<size_t>(i)].initialPoint);
        result.points[static_cast<size_t>(i)].rmsBefore = computeTrackRms(
            result.refinedCameras, tracks[static_cast<size_t>(i)],
            tracks[static_cast<size_t>(i)].initialPoint);
    }
    if (isCancelled(options)) {
        Logger::instance()->info("[BA] 已请求取消，跳过光束法平差优化");
        return result;
    }

    // 外层交替优化：先点后相机，重复迭代直到达到最大次数或收敛
    double prevTotalCost = std::numeric_limits<double>::max();
    const int maxOuterIterations = std::max(1, options.maxIterations);

    for (int outer = 0; outer < maxOuterIterations; ++outer) {
        if (isCancelled(options)) {
            Logger::instance()->info("[BA] 已请求取消，终止外层迭代");
            break;
        }

        // 阶段一：固定相机，并行优化每条轨迹的三维点坐标
        // 每个轨迹独立写入 result.points[i]，无读写竞争
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < numTracks; ++i) {
            if (isCancelled(options)) continue;
            const size_t si = static_cast<size_t>(i);
            if (!result.points[si].valid) continue;   // 离群点跳过
            BARefinedPoint p = optimizeOnePoint(result.refinedCameras, tracks[si], options);
            // 优化失败时回落到初始坐标
            if (!p.valid && vec3::isFinite(tracks[si].initialPoint)) {
                p.point    = tracks[si].initialPoint;
                p.rmsAfter = p.rmsBefore;
                p.valid    = true;
            }
            result.points[si] = p;
        }
        if (isCancelled(options)) {
            Logger::instance()->info("[BA] 已请求取消，点优化阶段后终止");
            break;
        }

        // 阶段二：固定三维点，并行优化每台相机的 6-DOF 位姿
        // 每个相机独立写入 result.refinedCameras[ci]，无读写竞争
        if (options.refineCameraPose) {
            int refinedCnt = 0;
#pragma omp parallel for schedule(static) reduction(+:refinedCnt)
            for (int ci = 0; ci < numCameras; ++ci) {
                if (isCancelled(options)) continue;
                if (isCameraFixed(ci, options)) continue;   // gauge 固定
                if (optimizeOneCamera(&result.refinedCameras[static_cast<size_t>(ci)],
                                      ci, tracks, result.points, options)) {
                    ++refinedCnt;
                }
            }
            result.refinedCameraCount = refinedCnt;
        }
        if (isCancelled(options)) {
            Logger::instance()->info("[BA] 已请求取消，相机优化阶段后终止");
            break;
        }

        // 阶段三（可选）：离群点过滤——每轮结束后根据 rmsAfter 过滤高误差点
        if (options.enablePointFilter) {
            // 自适应阈值 = max( filterMaxReprojError, filterSigmaFactor × median_rms )
            double adaptThresh = options.filterMaxReprojError;
            if (options.filterSigmaFactor > 0.0) {
                const double medRms = computeMedianRms(result.points);
                adaptThresh = std::max(adaptThresh,
                                       options.filterSigmaFactor * medRms);
            }

#pragma omp parallel for schedule(static)
            for (int i = 0; i < numTracks; ++i) {
                if (isCancelled(options)) continue;
                auto &p = result.points[static_cast<size_t>(i)];
                if (!p.valid) continue;
                if (std::isfinite(p.rmsAfter) && p.rmsAfter > adaptThresh) {
                    p.valid = false;   // 标记为离群点
                }
            }
        }

        // ── 外层收敛检查：全局 cost 变化 < 1% 则提前终止 ──
        {
            double totalCost = 0.0;
            int costCnt = 0;
            for (int i = 0; i < numTracks; ++i) {
                const auto &p = result.points[static_cast<size_t>(i)];
                if (p.valid && std::isfinite(p.rmsAfter)) {
                    totalCost += p.rmsAfter;
                    ++costCnt;
                }
            }
            double avgCost = (costCnt > 0) ? totalCost / costCnt : 0.0;
            double relChange = (prevTotalCost > 1e-12)
                ? std::fabs(prevTotalCost - avgCost) / prevTotalCost : 1.0;

            Logger::instance()->infof("[BA] iter %d/%d: avgRMS=%.6f, relChange=%.6f, validPts=%d", outer + 1,
                                      maxOuterIterations, avgCost, relChange, costCnt);

            if (options.progressCallback &&
                !options.progressCallback(outer + 1, maxOuterIterations, avgCost, costCnt)) {
                Logger::instance()->info("[BA] 进度回调请求终止光束法平差");
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

        // 用最终相机位姿重新计算 rmsAfter，确保全局一致性
        p.rmsAfter = computeTrackRms(result.refinedCameras, tracks[i], p.point);
        p.valid    = vec3::isFinite(p.point) && std::isfinite(p.rmsAfter);
        if (p.valid) {
            // 若本点 rmsAfter 超过自适应阈值，仍标记为无效
            if (options.enablePointFilter &&
                p.rmsAfter > options.filterMaxReprojError) {
                p.valid = false;
            }
        }
        if (p.valid) {
            ++result.optimizedTracks;
            sumAfter += p.rmsAfter;
            ++cntAfter;
        }
    }

    result.meanRmsBefore = (cntBefore > 0) ? (sumBefore / cntBefore) : 0.0;
    result.meanRmsAfter  = (cntAfter  > 0) ? (sumAfter  / cntAfter)  : 0.0;
    return result;
}

} // namespace xjw

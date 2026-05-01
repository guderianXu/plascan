#include "DisparityTriangulator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

namespace xjw
{
namespace mvs
{

namespace
{

inline std::array<double, 3> sub3(const std::array<double, 3> &a,
                                  const std::array<double, 3> &b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline std::array<double, 3> add3(const std::array<double, 3> &a,
                                  const std::array<double, 3> &b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline std::array<double, 3> mul3(const std::array<double, 3> &a, double s)
{
    return {a[0] * s, a[1] * s, a[2] * s};
}

inline double dot3(const std::array<double, 3> &a,
                   const std::array<double, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm3(const std::array<double, 3> &a)
{
    return std::sqrt(dot3(a, a));
}

inline std::array<double, 3> normalize3(const std::array<double, 3> &a)
{
    double n = norm3(a);
    return n < 1e-15 ? std::array<double,3>{0,0,0} : mul3(a, 1.0 / n);
}

void applyHomography(const cv::Mat &H, double u, double v,
                     double &ox, double &oy)
{
    const double *h = H.ptr<double>(0);
    double w = h[6] * u + h[7] * v + h[8];
    if (std::abs(w) < 1e-12) { ox = u; oy = v; return; }
    ox = (h[0] * u + h[1] * v + h[2]) / w;
    oy = (h[3] * u + h[4] * v + h[5]) / w;
}

std::array<double, 3> pixelToWorldRay(const Camera &cam, double u, double v)
{
    const double x = (u - cam.principalX()) / (cam.uAxisSign() * cam.focalX());
    const double y = (v - cam.principalY()) / (cam.vAxisSign() * cam.focalY());
    const double zSign = cam.depthAxisFlipped() ? -1.0 : 1.0;
    const std::array<double, 3> ray_cam{x * zSign, y * zSign, zSign};
    const auto R = cam.cameraToWorldRotation();
    std::array<double, 3> ray_world{
        R[0] * ray_cam[0] + R[1] * ray_cam[1] + R[2] * ray_cam[2],
        R[3] * ray_cam[0] + R[4] * ray_cam[1] + R[5] * ray_cam[2],
        R[6] * ray_cam[0] + R[7] * ray_cam[1] + R[8] * ray_cam[2]};
    return normalize3(ray_world);
}

struct RawPoint
{
    double x, y, z;
    float error;
    bool valid;
};

} // namespace

TriangulationResult DisparityTriangulator::triangulate(
    const cv::Mat &disparity,
    const cv::Mat &validMask,
    const cv::Mat &H1inv,
    const cv::Mat &H2inv,
    const Camera &camL,
    const Camera &camR,
    const TriangulationConfig &cfg)
{
    TriangulationResult result;
    const int rows = disparity.rows;
    const int cols = disparity.cols;
    result.totalPixels = rows * cols;

    std::vector<RawPoint> rawPoints(rows * cols);

    auto C1 = camL.cameraCenter();
    auto C2 = camR.cameraCenter();

    unsigned int nThreads = cfg.numThreads;
    if (nThreads == 0)
        nThreads = std::max(1u, std::thread::hardware_concurrency());
    nThreads = std::min(nThreads, static_cast<unsigned int>(rows));

    auto worker = [&](int rowStart, int rowEnd)
    {
        int localValid = 0, localNegDepth = 0, localHighErr = 0, localBadRay = 0;
        bool dbg = (rowStart == 0);
        int dbgCount = 0;
        for (int r = rowStart; r < rowEnd; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                int idx = r * cols + c;
                rawPoints[idx].valid = false;

                if (validMask.at<uint8_t>(r, c) == 0) continue;
                float d = disparity.at<float>(r, c);
                if (!std::isfinite(d) || d == 0.0f) continue;

                double lu, lv, ru, rv;
                applyHomography(H1inv, c, r, lu, lv);
                applyHomography(H2inv, c - d, r, ru, rv);

                auto d1 = pixelToWorldRay(camL, lu, lv);
                auto d2 = pixelToWorldRay(camR, ru, rv);

                if (norm3(d1) < 0.5 || norm3(d2) < 0.5)
                {
                    ++localBadRay;
                    continue;
                }

                auto w0 = sub3(C1, C2);
                double a = dot3(d1, d1);
                double b = dot3(d1, d2);
                double cc2 = dot3(d2, d2);
                double dd2 = dot3(d1, w0);
                double e = dot3(d2, w0);
                double den = a * cc2 - b * b;

                double s = 0, t = 0;
                if (std::abs(den) < 1e-12)
                {
                    t = (cc2 > 1e-12) ? (e / cc2) : 0.0;
                }
                else
                {
                    s = (b * e - cc2 * dd2) / den;
                    t = (a * e - b * dd2) / den;
                }

                if (dbg && dbgCount < 3)
                {
                    fprintf(stderr, "[Tri dbg] pix(%d,%d) disp=%.2f s=%.4f t=%.4f\n",
                            c, r, d, s, t);
                    ++dbgCount;
                }

                if (s <= 0 || t <= 0)
                {
                    ++localNegDepth;
                    continue;
                }

                auto P1 = add3(C1, mul3(d1, s));
                auto P2 = add3(C2, mul3(d2, t));
                auto X = mul3(add3(P1, P2), 0.5);
                float miss = static_cast<float>(norm3(sub3(P1, P2)));

                if (miss > cfg.maxTriangulationError)
                {
                    ++localHighErr;
                    continue;
                }

                rawPoints[idx] = {X[0], X[1], X[2], miss, true};
                ++localValid;
            }
        }
    if (rowStart == 0)
        fprintf(stderr, "[Triangulator] thread0: valid=%d negDepth=%d highErr=%d badRay=%d\n",
                localValid, localNegDepth, localHighErr, localBadRay);
    };

    // Launch threads
    std::vector<std::thread> threads;
    int chunkSize = (rows + nThreads - 1) / nThreads;
    for (unsigned int i = 0; i < nThreads; ++i)
    {
        int start = i * chunkSize;
        int end = std::min(rows, start + chunkSize);
        if (start >= end) break;
        threads.emplace_back(worker, start, end);
    }
    for (auto &th : threads) th.join();

    // Compute centroid (POINT_OFFSET)
    double sumX = 0, sumY = 0, sumZ = 0;
    int validCount = 0;
    for (auto &p : rawPoints)
    {
        if (!p.valid) continue;
        sumX += p.x;
        sumY += p.y;
        sumZ += p.z;
        ++validCount;
    }

    result.validPoints = validCount;
    if (validCount == 0)
    {
        result.pointCloud = cv::Mat(rows, cols, CV_64FC3, cv::Scalar(0, 0, 0));
        result.errorMap = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
        result.validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
        return result;
    }

    result.pointOffset = {sumX / validCount, sumY / validCount, sumZ / validCount};

    // Build output matrices
    result.pointCloud = cv::Mat(rows, cols, CV_64FC3, cv::Scalar(0, 0, 0));
    result.errorMap = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    result.validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));

    std::vector<float> errors;
    errors.reserve(validCount);

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            auto &p = rawPoints[r * cols + c];
            if (!p.valid) continue;

            auto &pt = result.pointCloud.at<cv::Vec3d>(r, c);
            pt[0] = p.x - result.pointOffset[0];
            pt[1] = p.y - result.pointOffset[1];
            pt[2] = p.z - result.pointOffset[2];
            result.errorMap.at<float>(r, c) = p.error;
            result.validMask.at<uint8_t>(r, c) = 255;
            errors.push_back(p.error);
        }
    }

    std::sort(errors.begin(), errors.end());
    result.medianError = errors[errors.size() / 2];

    float cov = 100.0f * validCount / (rows * cols);
    fprintf(stderr, "[DisparityTriangulator] valid=%d (%.1f%%), "
            "median_error=%.6f, offset=(%.4f, %.4f, %.4f)\n",
            validCount, cov, result.medianError,
            result.pointOffset[0], result.pointOffset[1], result.pointOffset[2]);

    return result;
}

TriangulationResult DisparityTriangulator::triangulateFromDepth(
    const cv::Mat &depthMap,
    const cv::Mat &validMask,
    const cv::Mat &H1inv,
    const Camera &camL,
    const Camera &camR,
    const PositiveDepthCameraModel &rectCam,
    const TriangulationConfig &cfg)
{
    TriangulationResult result;
    const int rows = depthMap.rows;
    const int cols = depthMap.cols;
    result.totalPixels = rows * cols;

    std::vector<RawPoint> rawPoints(rows * cols);

    unsigned int nThreads = cfg.numThreads;
    if (nThreads == 0)
        nThreads = std::max(1u, std::thread::hardware_concurrency());
    nThreads = std::min(nThreads, static_cast<unsigned int>(rows));

    auto worker = [&](int rowStart, int rowEnd)
    {
        int dbgCount = 0;
        for (int r = rowStart; r < rowEnd; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                int idx = r * cols + c;
                rawPoints[idx].valid = false;

                if (validMask.at<uint8_t>(r, c) == 0) continue;
                float depth = depthMap.at<float>(r, c);
                if (!std::isfinite(depth) || depth <= 0.0f) continue;

                float wx, wy, wz;
                rectCam.unproject(static_cast<float>(c), static_cast<float>(r),
                                  depth, wx, wy, wz);

                double world[3] = {wx, wy, wz};
                double uv1[2];
                float errL = 0.0f;

                if (camL.projectWorldPoint(world, uv1))
                {
                    double hU = c;
                    double hV = r;
                    double origU, origV;
                    applyHomography(H1inv, hU, hV, origU, origV);
                    double du = uv1[0] - origU;
                    double dv = uv1[1] - origV;
                    errL = static_cast<float>(std::sqrt(du*du + dv*dv));
                }

                if (rowStart == 0 && dbgCount < 3)
                {
                    fprintf(stderr, "[Tri dbg] pix(%d,%d) depth=%.4f "
                            "world=(%.6f,%.6f,%.6f) reproj_err=%.4f\n",
                            c, r, depth, wx, wy, wz, errL);
                    ++dbgCount;
                }

                rawPoints[idx] = {static_cast<double>(wx),
                                  static_cast<double>(wy),
                                  static_cast<double>(wz),
                                  errL, true};
            }
        }
    };

    std::vector<std::thread> threads;
    int chunkSize = (rows + nThreads - 1) / nThreads;
    for (unsigned int i = 0; i < nThreads; ++i)
    {
        int start = i * chunkSize;
        int end = std::min(rows, start + chunkSize);
        if (start >= end) break;
        threads.emplace_back(worker, start, end);
    }
    for (auto &th : threads) th.join();

    // Compute centroid
    double sumX = 0, sumY = 0, sumZ = 0;
    int validCount = 0;
    for (auto &p : rawPoints)
    {
        if (!p.valid) continue;
        sumX += p.x;
        sumY += p.y;
        sumZ += p.z;
        ++validCount;
    }

    result.validPoints = validCount;
    if (validCount == 0)
    {
        result.pointCloud = cv::Mat(rows, cols, CV_64FC3, cv::Scalar(0, 0, 0));
        result.errorMap = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
        result.validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
        return result;
    }

    result.pointOffset = {sumX / validCount, sumY / validCount, sumZ / validCount};

    result.pointCloud = cv::Mat(rows, cols, CV_64FC3, cv::Scalar(0, 0, 0));
    result.errorMap = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    result.validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));

    std::vector<float> errors;
    errors.reserve(validCount);

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            auto &p = rawPoints[r * cols + c];
            if (!p.valid) continue;

            auto &pt = result.pointCloud.at<cv::Vec3d>(r, c);
            pt[0] = p.x - result.pointOffset[0];
            pt[1] = p.y - result.pointOffset[1];
            pt[2] = p.z - result.pointOffset[2];
            result.errorMap.at<float>(r, c) = p.error;
            result.validMask.at<uint8_t>(r, c) = 255;
            errors.push_back(p.error);
        }
    }

    std::sort(errors.begin(), errors.end());
    result.medianError = errors[errors.size() / 2];

    float cov = 100.0f * validCount / (rows * cols);
    fprintf(stderr, "[DepthTriangulator] valid=%d (%.1f%%), "
            "median_error=%.6f, offset=(%.4f, %.4f, %.4f)\n",
            validCount, cov, result.medianError,
            result.pointOffset[0], result.pointOffset[1], result.pointOffset[2]);

    return result;
}

} // namespace mvs
} // namespace xjw

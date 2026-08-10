#include "DisparityTriangulator.h"

#include "concurrency/SafeWorkerGroup.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
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

inline bool finite3(const std::array<double, 3> &value)
{
    return std::isfinite(value[0])
        && std::isfinite(value[1])
        && std::isfinite(value[2]);
}

inline std::array<double, 3> normalize3(const std::array<double, 3> &a)
{
    double n = norm3(a);
    return n < 1e-15 ? std::array<double,3>{0,0,0} : mul3(a, 1.0 / n);
}

bool applyHomography(const cv::Mat &H, double u, double v,
                     double &ox, double &oy)
{
    const double *h = H.ptr<double>(0);
    const double w = h[6] * u + h[7] * v + h[8];
    if (!std::isfinite(w) || std::abs(w) < 1e-12)
    {
        return false;
    }
    ox = (h[0] * u + h[1] * v + h[2]) / w;
    oy = (h[3] * u + h[4] * v + h[5]) / w;
    return std::isfinite(ox) && std::isfinite(oy);
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
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    float error = 0.0f;
    bool valid = false;
};

bool validateMapAndMask(const cv::Mat &map,
                        const char *mapName,
                        const cv::Mat &validMask,
                        std::size_t &pixelCount,
                        std::string &error)
{
    if (map.empty())
    {
        error = std::string(mapName) + "为空";
        return false;
    }
    if (map.dims != 2 || map.type() != CV_32FC1)
    {
        error = std::string(mapName) + "必须是二维 CV_32FC1";
        return false;
    }
    if (validMask.empty())
    {
        error = "有效掩码为空";
        return false;
    }
    if (validMask.dims != 2 || validMask.type() != CV_8UC1)
    {
        error = "有效掩码必须是二维 CV_8UC1";
        return false;
    }
    if (validMask.size() != map.size())
    {
        error = "有效掩码尺寸与输入图不一致";
        return false;
    }

    const std::size_t rows = static_cast<std::size_t>(map.rows);
    const std::size_t cols = static_cast<std::size_t>(map.cols);
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols)
    {
        error = std::string(mapName) + "像素数溢出";
        return false;
    }
    pixelCount = rows * cols;
    if (pixelCount > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        error = std::string(mapName) + "像素数超过结果格式上限";
        return false;
    }
    return true;
}

bool prepareHomography(const cv::Mat &input,
                       const char *name,
                       cv::Mat &prepared,
                       std::string &error)
{
    if (input.dims != 2 || input.rows != 3 || input.cols != 3)
    {
        error = std::string(name) + "必须是 3x3 矩阵";
        return false;
    }
    if (input.type() != CV_64FC1)
    {
        error = std::string(name) + "必须是 CV_64FC1";
        return false;
    }

    prepared = input.isContinuous() ? input : input.clone();
    const double *values = prepared.ptr<double>(0);
    for (int index = 0; index < 9; ++index)
    {
        if (!std::isfinite(values[index]))
        {
            error = std::string(name) + "包含非有限值";
            return false;
        }
    }
    return true;
}

bool validateCamera(const Camera &camera, const char *name, std::string &error)
{
    if (!camera.isValid()
        || !std::isfinite(camera.focalX())
        || !std::isfinite(camera.focalY())
        || !std::isfinite(camera.principalX())
        || !std::isfinite(camera.principalY())
        || camera.focalX() <= 0.0
        || camera.focalY() <= 0.0
        || (camera.uAxisSign() != -1 && camera.uAxisSign() != 1)
        || (camera.vAxisSign() != -1 && camera.vAxisSign() != 1))
    {
        error = std::string(name) + "内参无效";
        return false;
    }

    for (double value : camera.cameraCenter())
    {
        if (!std::isfinite(value))
        {
            error = std::string(name) + "光心包含非有限值";
            return false;
        }
    }

    const std::array<double, 9> rotation = camera.cameraToWorldRotation();
    for (double value : rotation)
    {
        if (!std::isfinite(value))
        {
            error = std::string(name) + "旋转矩阵包含非有限值";
            return false;
        }
    }
    const double determinant =
        rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
        - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
        + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-12)
    {
        error = std::string(name) + "旋转矩阵退化";
        return false;
    }
    return true;
}

bool validateConfig(const TriangulationConfig &config, std::string &error)
{
    if (!std::isfinite(config.maxTriangulationError)
        || config.maxTriangulationError < 0.0f)
    {
        error = "最大三角化误差必须是非负有限值";
        return false;
    }
    if (config.numThreads < 0)
    {
        error = "线程数不能为负数";
        return false;
    }
    return true;
}

bool validateStereoBaseline(const Camera &left,
                            const Camera &right,
                            std::string &error)
{
    const double baseline = norm3(sub3(left.cameraCenter(), right.cameraCenter()));
    if (!std::isfinite(baseline) || baseline <= 1e-12)
    {
        error = "左右相机基线为零";
        return false;
    }
    return true;
}

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
    std::size_t pixelCount = 0;
    if (!validateMapAndMask(
            disparity,
            "视差图",
            validMask,
            pixelCount,
            result.errorMessage)
        || !validateConfig(cfg, result.errorMessage)
        || !validateCamera(camL, "左相机", result.errorMessage)
        || !validateCamera(camR, "右相机", result.errorMessage)
        || !validateStereoBaseline(camL, camR, result.errorMessage))
    {
        return result;
    }

    cv::Mat preparedH1;
    cv::Mat preparedH2;
    if (!prepareHomography(H1inv, "H1inv", preparedH1, result.errorMessage)
        || !prepareHomography(H2inv, "H2inv", preparedH2, result.errorMessage))
    {
        return result;
    }

    const int rows = disparity.rows;
    const int cols = disparity.cols;
    result.totalPixels = static_cast<int>(pixelCount);
    std::vector<RawPoint> rawPoints;
    try
    {
        rawPoints.resize(pixelCount);
    }
    catch (const std::exception &error)
    {
        result.errorMessage = "视差三角化缓冲区分配失败: " + std::string(error.what());
        return result;
    }

    auto C1 = camL.cameraCenter();
    auto C2 = camR.cameraCenter();

    unsigned int nThreads = cfg.numThreads;
    if (nThreads == 0)
        nThreads = std::max(1u, std::thread::hardware_concurrency());
    nThreads = std::min(nThreads, static_cast<unsigned int>(rows));

    auto worker = [&](int rowStart, int rowEnd, std::stop_token stopToken)
    {
        int localValid = 0, localNegDepth = 0, localHighErr = 0, localBadRay = 0;
        bool dbg = (rowStart == 0);
        int dbgCount = 0;
        for (int r = rowStart; r < rowEnd; ++r)
        {
            if (stopToken.stop_requested())
            {
                break;
            }
            for (int c = 0; c < cols; ++c)
            {
                const std::size_t idx = static_cast<std::size_t>(r)
                    * static_cast<std::size_t>(cols)
                    + static_cast<std::size_t>(c);
                rawPoints[idx].valid = false;

                if (validMask.at<uint8_t>(r, c) == 0) continue;
                float d = disparity.at<float>(r, c);
                if (!std::isfinite(d) || d == 0.0f) continue;

                double lu, lv, ru, rv;
                if (!applyHomography(preparedH1, c, r, lu, lv)
                    || !applyHomography(preparedH2, c - d, r, ru, rv))
                {
                    ++localBadRay;
                    continue;
                }

                auto d1 = pixelToWorldRay(camL, lu, lv);
                auto d2 = pixelToWorldRay(camR, ru, rv);

                const double rayNorm1 = norm3(d1);
                const double rayNorm2 = norm3(d2);
                if (!finite3(d1) || !finite3(d2)
                    || !std::isfinite(rayNorm1) || !std::isfinite(rayNorm2)
                    || rayNorm1 < 0.5 || rayNorm2 < 0.5)
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

                if (!std::isfinite(s) || !std::isfinite(t) || s <= 0 || t <= 0)
                {
                    ++localNegDepth;
                    continue;
                }

                auto P1 = add3(C1, mul3(d1, s));
                auto P2 = add3(C2, mul3(d2, t));
                auto X = mul3(add3(P1, P2), 0.5);
                float miss = static_cast<float>(norm3(sub3(P1, P2)));

                if (!finite3(X) || !std::isfinite(miss)
                    || miss > cfg.maxTriangulationError)
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

    const int threadCount = static_cast<int>(nThreads);
    const int chunkSize = rows / threadCount + (rows % threadCount != 0 ? 1 : 0);
    try
    {
        xjw::common::concurrency::runWorkerGroup(
            static_cast<std::size_t>(nThreads),
            [worker, chunkSize, rows](std::size_t workerIndex,
                                      std::stop_token stopToken) mutable
        {
            const int start = static_cast<int>(workerIndex) * chunkSize;
            const int end = std::min(rows, start + chunkSize);
            if (start < end)
            {
                worker(start, end, stopToken);
            }
        });
    }
    catch (const std::exception &error)
    {
        result.errorMessage = "视差三角化 worker 异常: " + std::string(error.what());
        return result;
    }
    catch (...)
    {
        result.errorMessage = "视差三角化 worker 发生未知异常";
        return result;
    }

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
            const std::size_t index = static_cast<std::size_t>(r)
                * static_cast<std::size_t>(cols)
                + static_cast<std::size_t>(c);
            auto &p = rawPoints[index];
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

    const float cov = 100.0f * static_cast<float>(validCount)
        / static_cast<float>(pixelCount);
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
    const Camera &rectCam,
    const TriangulationConfig &cfg)
{
    TriangulationResult result;
    std::size_t pixelCount = 0;
    if (!validateMapAndMask(
            depthMap,
            "深度图",
            validMask,
            pixelCount,
            result.errorMessage)
        || !validateConfig(cfg, result.errorMessage)
        || !validateCamera(camL, "左相机", result.errorMessage)
        || !validateCamera(camR, "右相机", result.errorMessage)
        || !validateCamera(rectCam, "校正左相机", result.errorMessage)
        || !validateStereoBaseline(camL, camR, result.errorMessage))
    {
        return result;
    }

    cv::Mat preparedH1;
    if (!prepareHomography(H1inv, "H1inv", preparedH1, result.errorMessage))
    {
        return result;
    }

    const int rows = depthMap.rows;
    const int cols = depthMap.cols;
    result.totalPixels = static_cast<int>(pixelCount);
    std::vector<RawPoint> rawPoints;
    try
    {
        rawPoints.resize(pixelCount);
    }
    catch (const std::exception &error)
    {
        result.errorMessage = "深度三角化缓冲区分配失败: " + std::string(error.what());
        return result;
    }

    unsigned int nThreads = cfg.numThreads;
    if (nThreads == 0)
        nThreads = std::max(1u, std::thread::hardware_concurrency());
    nThreads = std::min(nThreads, static_cast<unsigned int>(rows));

    auto worker = [&](int rowStart, int rowEnd, std::stop_token stopToken)
    {
        int dbgCount = 0;
        for (int r = rowStart; r < rowEnd; ++r)
        {
            if (stopToken.stop_requested())
            {
                break;
            }
            for (int c = 0; c < cols; ++c)
            {
                const std::size_t idx = static_cast<std::size_t>(r)
                    * static_cast<std::size_t>(cols)
                    + static_cast<std::size_t>(c);
                rawPoints[idx].valid = false;

                if (validMask.at<uint8_t>(r, c) == 0) continue;
                float depth = depthMap.at<float>(r, c);
                if (!std::isfinite(depth) || depth <= 0.0f) continue;

                const double pixel[2] = {static_cast<double>(c), static_cast<double>(r)};
                double world[3] = {0.0, 0.0, 0.0};
                if (!rectCam.unprojectPixel(pixel, static_cast<double>(depth), world))
                {
                    continue;
                }
                if (!std::isfinite(world[0])
                    || !std::isfinite(world[1])
                    || !std::isfinite(world[2]))
                {
                    continue;
                }
                double uv1[2];
                float errL = 0.0f;

                if (!camL.projectWorldPoint(world, uv1))
                {
                    continue;
                }
                double origU, origV;
                if (!applyHomography(preparedH1, c, r, origU, origV))
                {
                    continue;
                }
                const double du = uv1[0] - origU;
                const double dv = uv1[1] - origV;
                errL = static_cast<float>(std::sqrt(du*du + dv*dv));

                if (rowStart == 0 && dbgCount < 3)
                {
                    fprintf(stderr, "[Tri dbg] pix(%d,%d) depth=%.4f "
                            "world=(%.6f,%.6f,%.6f) reproj_err=%.4f\n",
                            c, r, depth, world[0], world[1], world[2], errL);
                    ++dbgCount;
                }

                rawPoints[idx] = {world[0],
                                  world[1],
                                  world[2],
                                  errL, true};
            }
        }
    };

    const int threadCount = static_cast<int>(nThreads);
    const int chunkSize = rows / threadCount + (rows % threadCount != 0 ? 1 : 0);
    try
    {
        xjw::common::concurrency::runWorkerGroup(
            static_cast<std::size_t>(nThreads),
            [worker, chunkSize, rows](std::size_t workerIndex,
                                      std::stop_token stopToken) mutable
        {
            const int start = static_cast<int>(workerIndex) * chunkSize;
            const int end = std::min(rows, start + chunkSize);
            if (start < end)
            {
                worker(start, end, stopToken);
            }
        });
    }
    catch (const std::exception &error)
    {
        result.errorMessage = "深度三角化 worker 异常: " + std::string(error.what());
        return result;
    }
    catch (...)
    {
        result.errorMessage = "深度三角化 worker 发生未知异常";
        return result;
    }

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
            const std::size_t index = static_cast<std::size_t>(r)
                * static_cast<std::size_t>(cols)
                + static_cast<std::size_t>(c);
            auto &p = rawPoints[index];
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

    const float cov = 100.0f * static_cast<float>(validCount)
        / static_cast<float>(pixelCount);
    fprintf(stderr, "[DepthTriangulator] valid=%d (%.1f%%), "
            "median_error=%.6f, offset=(%.4f, %.4f, %.4f)\n",
            validCount, cov, result.medianError,
            result.pointOffset[0], result.pointOffset[1], result.pointOffset[2]);

    return result;
}

} // namespace mvs
} // namespace xjw

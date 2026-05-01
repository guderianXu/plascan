#include "DemGenerator.h"

#include <QtGlobal>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw
{

static float readDepth(const cv::Mat &depthMap, int row, int col)
{
    if (depthMap.type() == CV_32FC1)
        return depthMap.at<float>(row, col);
    if (depthMap.type() == CV_16UC1)
        return static_cast<float>(depthMap.at<uint16_t>(row, col));
    return 0.0f;
}

static void unprojectPixel(int col, int row, double depth,
                           double fx, double fy, double cx, double cy,
                           const std::array<double,9> &R,
                           const std::array<double,3> &C,
                           double &wx, double &wy, double &wz)
{
    const double xn = (col - cx) / fx;
    const double yn = (row - cy) / fy;
    const double d = depth;
    double cp[3] = {xn * d, yn * d, d};
    wx = R[0]*cp[0] + R[1]*cp[1] + R[2]*cp[2] + C[0];
    wy = R[3]*cp[0] + R[4]*cp[1] + R[5]*cp[2] + C[1];
    wz = R[6]*cp[0] + R[7]*cp[1] + R[8]*cp[2] + C[2];
}

bool DemGenerator::generateFromDepthMaps(const std::vector<cv::Mat> &depthMaps,
                                         const std::vector<Camera> &cameras,
                                         const DemGenerationOptions &options,
                                         DemGridData *demGrid,
                                         QString *errorMsg)
{
    if (!demGrid)
    {
        if (errorMsg) *errorMsg = "demGrid is null";
        return false;
    }
    if (depthMaps.empty() || cameras.empty())
    {
        if (errorMsg) *errorMsg = "depthMaps or cameras is empty";
        return false;
    }
    if (depthMaps.size() != cameras.size())
    {
        if (errorMsg)
            *errorMsg = QString("depthMaps size (%1) != cameras size (%2)")
                            .arg(depthMaps.size()).arg(cameras.size());
        return false;
    }

    const cv::Mat &refDepth = depthMaps[0];
    const Camera &refCam = cameras[0];
    if (refDepth.empty())
    {
        if (errorMsg) *errorMsg = "Reference depth map is empty";
        return false;
    }

    const int imgH = refDepth.rows;
    const int imgW = refDepth.cols;
    const auto R = refCam.cameraToWorldRotation();
    const auto C = refCam.cameraCenter();
    const double fx = refCam.focalX();
    const double fy = refCam.focalY();
    const double cx = refCam.principalX();
    const double cy = refCam.principalY();

    // --- Step 0: Compute depth statistics for outlier filtering ---
    // For planetary/asteroid imaging, surface is near origin, camera is far away.
    // Use tight depth filter: median ± 5% covers the expected surface depth range.
    std::vector<float> allDepths;
    allDepths.reserve(imgH * imgW / 4);
    for (int row = 0; row < imgH; row += 2)
        for (int col = 0; col < imgW; col += 2)
        {
            float d = readDepth(refDepth, row, col);
            if (d > 0.0f && std::isfinite(d))
                allDepths.push_back(d);
        }

    float depthLo = 0.0f, depthHi = 1e30f;
    if (allDepths.size() > 100)
    {
        std::sort(allDepths.begin(), allDepths.end());
        size_t n = allDepths.size();
        float median = allDepths[n / 2];
        float tolerance = median * 0.05f;
        depthLo = median - tolerance;
        depthHi = median + tolerance;
    }

    // --- Step 1: Unproject reference depth map to 3D ---
    cv::Mat wX(imgH, imgW, CV_64FC1, cv::Scalar(0));
    cv::Mat wY(imgH, imgW, CV_64FC1, cv::Scalar(0));
    cv::Mat wZ(imgH, imgW, CV_64FC1, cv::Scalar(0));
    cv::Mat triErr(imgH, imgW, CV_64FC1, cv::Scalar(0));
    cv::Mat valid(imgH, imgW, CV_8UC1, cv::Scalar(0));
    int validCount = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:validCount)
    for (int row = 0; row < imgH; ++row)
    {
        for (int col = 0; col < imgW; ++col)
        {
            float depth = readDepth(refDepth, row, col);
            if (depth <= 0.0f || !std::isfinite(depth))
                continue;
            if (depth < depthLo || depth > depthHi)
                continue;

            double wx, wy, wz;
            unprojectPixel(col, row, depth, fx, fy, cx, cy, R, C, wx, wy, wz);

            wX.at<double>(row, col) = wx;
            wY.at<double>(row, col) = wy;
            wZ.at<double>(row, col) = wz;
            valid.at<uchar>(row, col) = 255;
            ++validCount;
        }
    }

    // Merge secondary depth maps and compute triangulation error
    for (size_t vi = 1; vi < depthMaps.size(); ++vi)
    {
        if (depthMaps[vi].empty()) continue;
        const cv::Mat &secDepth = depthMaps[vi];
        const Camera &secCam = cameras[vi];
        const auto R2 = secCam.cameraToWorldRotation();
        const auto C2 = secCam.cameraCenter();
        const double fx2 = secCam.focalX();
        const double fy2 = secCam.focalY();
        const double cx2 = secCam.principalX();
        const double cy2 = secCam.principalY();

        for (int row = 0; row < secDepth.rows; ++row)
        {
            for (int col = 0; col < secDepth.cols; ++col)
            {
                float depth = readDepth(secDepth, row, col);
                if (depth <= 0.0f || !std::isfinite(depth))
                    continue;
                if (depth < depthLo || depth > depthHi)
                    continue;

                double wx2, wy2, wz2;
                unprojectPixel(col, row, depth, fx2, fy2, cx2, cy2, R2, C2, wx2, wy2, wz2);

                double pixel[2];
                if (!refCam.projectWorldPoint(&wx2, pixel))
                    continue;
                int refCol = static_cast<int>(std::round(pixel[0]));
                int refRow = static_cast<int>(std::round(pixel[1]));
                if (refCol < 0 || refCol >= imgW || refRow < 0 || refRow >= imgH)
                    continue;

                if (valid.at<uchar>(refRow, refCol) == 0)
                {
                    wX.at<double>(refRow, refCol) = wx2;
                    wY.at<double>(refRow, refCol) = wy2;
                    wZ.at<double>(refRow, refCol) = wz2;
                    valid.at<uchar>(refRow, refCol) = 255;
                    ++validCount;
                }
                else
                {
                    // Compute triangulation error: distance between 3D points
                    // from reference and secondary cameras
                    double dx = wX.at<double>(refRow, refCol) - wx2;
                    double dy = wY.at<double>(refRow, refCol) - wy2;
                    double dz = wZ.at<double>(refRow, refCol) - wz2;
                    double err = std::sqrt(dx*dx + dy*dy + dz*dz);
                    triErr.at<double>(refRow, refCol) = err;
                }
            }
        }
    }

    if (validCount == 0)
    {
        if (errorMsg) *errorMsg = "No valid 3D points generated from depth maps";
        return false;
    }

    // --- Step 2: Radius-based outlier rejection ---
    std::vector<double> radii;
    radii.reserve(validCount);
    for (int row = 0; row < imgH; ++row)
        for (int col = 0; col < imgW; ++col)
            if (valid.at<uchar>(row, col))
            {
                double x = wX.at<double>(row, col);
                double y = wY.at<double>(row, col);
                double z = wZ.at<double>(row, col);
                radii.push_back(std::sqrt(x*x + y*y + z*z));
            }

    std::sort(radii.begin(), radii.end());
    double rMedian = radii[radii.size() / 2];
    double rP25 = radii[radii.size() * 25 / 100];
    double rP75 = radii[radii.size() * 75 / 100];
    double rIqr = rP75 - rP25;
    double rLo  = rP25 - 1.5 * rIqr;
    double rHi  = rP75 + 1.5 * rIqr;
    if (rLo < 0.0) rLo = 0.0;

    int outlierCount = 0;
    for (int row = 0; row < imgH; ++row)
        for (int col = 0; col < imgW; ++col)
            if (valid.at<uchar>(row, col))
            {
                double x = wX.at<double>(row, col);
                double y = wY.at<double>(row, col);
                double z = wZ.at<double>(row, col);
                double r = std::sqrt(x*x + y*y + z*z);
                if (r < rLo || r > rHi)
                {
                    valid.at<uchar>(row, col) = 0;
                    ++outlierCount;
                    --validCount;
                }
            }

    if (validCount == 0)
    {
        if (errorMsg) *errorMsg = "All points rejected as outliers";
        return false;
    }

    // --- Step 3: Compute robust geo bounds ---
    double gMinX = 1e30, gMaxX = -1e30;
    double gMinY = 1e30, gMaxY = -1e30;
    {
        std::vector<double> xs, ys;
        xs.reserve(validCount);
        ys.reserve(validCount);
        for (int row = 0; row < imgH; ++row)
            for (int col = 0; col < imgW; ++col)
                if (valid.at<uchar>(row, col))
                {
                    xs.push_back(wX.at<double>(row, col));
                    ys.push_back(wY.at<double>(row, col));
                }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());
        size_t n = xs.size();
        if (n > 100)
        {
            size_t lo = n / 200, hi = n - 1 - lo;
            double padX = (xs[hi] - xs[lo]) * 0.02;
            double padY = (ys[hi] - ys[lo]) * 0.02;
            gMinX = xs[lo] - padX;
            gMaxX = xs[hi] + padX;
            gMinY = ys[lo] - padY;
            gMaxY = ys[hi] + padY;
        }
        else
        {
            for (double v : xs) { gMinX = std::min(gMinX, v); gMaxX = std::max(gMaxX, v); }
            for (double v : ys) { gMinY = std::min(gMinY, v); gMaxY = std::max(gMaxY, v); }
        }
    }

    // --- Step 4: Build DEM grid (image-space, 3-band XYZ) ---
    const int demW = imgW;
    const int demH = imgH;

    demGrid->width = demW;
    demGrid->height = demH;
    demGrid->minX = gMinX;
    demGrid->minY = gMinY;
    demGrid->stepX = (gMaxX - gMinX) / std::max(1, demW - 1);
    demGrid->stepY = (gMaxY - gMinY) / std::max(1, demH - 1);
    demGrid->elevation = cv::Mat(demH, demW, CV_32FC1, cv::Scalar(0.0f));
    demGrid->worldX    = cv::Mat(demH, demW, CV_32FC1, cv::Scalar(0.0f));
    demGrid->worldY    = cv::Mat(demH, demW, CV_32FC1, cv::Scalar(0.0f));
    demGrid->triangulationError = cv::Mat(demH, demW, CV_32FC1, cv::Scalar(0.0f));
    demGrid->validMask = cv::Mat(demH, demW, CV_8UC1, cv::Scalar(0));

    for (int row = 0; row < demH; ++row)
        for (int col = 0; col < demW; ++col)
            if (valid.at<uchar>(row, col))
            {
                demGrid->worldX.at<float>(row, col) =
                    static_cast<float>(wX.at<double>(row, col));
                demGrid->worldY.at<float>(row, col) =
                    static_cast<float>(wY.at<double>(row, col));
                demGrid->elevation.at<float>(row, col) =
                    static_cast<float>(wZ.at<double>(row, col));
                demGrid->triangulationError.at<float>(row, col) =
                    static_cast<float>(triErr.at<double>(row, col));
                demGrid->validMask.at<uchar>(row, col) = 255;
            }

    // --- Step 5: Hole filling (all 3 bands) ---
    const int holeFillRadius = std::max(1, options.holeFillSearchRadius);
    for (int iteration = 0; iteration < options.holeFillIterations; ++iteration)
    {
        cv::Mat xCopy = demGrid->worldX.clone();
        cv::Mat yCopy = demGrid->worldY.clone();
        cv::Mat zCopy = demGrid->elevation.clone();
        cv::Mat maskCopy = demGrid->validMask.clone();

        for (int row = 0; row < demH; ++row)
        {
            for (int col = 0; col < demW; ++col)
            {
                if (demGrid->validMask.at<uchar>(row, col) != 0)
                    continue;

                double wSum = 0.0, wxSum = 0.0, wySum = 0.0, wzSum = 0.0;
                int nCount = 0;
                const int r0 = std::max(0, row - holeFillRadius);
                const int r1 = std::min(demH - 1, row + holeFillRadius);
                const int c0 = std::max(0, col - holeFillRadius);
                const int c1 = std::min(demW - 1, col + holeFillRadius);

                for (int nr = r0; nr <= r1; ++nr)
                    for (int nc = c0; nc <= c1; ++nc)
                    {
                        if (nr == row && nc == col) continue;
                        if (demGrid->validMask.at<uchar>(nr, nc) == 0) continue;
                        int dx = nc - col, dy = nr - row;
                        double dist = std::sqrt(static_cast<double>(dx*dx + dy*dy));
                        double w = 1.0 / std::max(1e-6, dist);
                        wxSum += static_cast<double>(demGrid->worldX.at<float>(nr, nc)) * w;
                        wySum += static_cast<double>(demGrid->worldY.at<float>(nr, nc)) * w;
                        wzSum += static_cast<double>(demGrid->elevation.at<float>(nr, nc)) * w;
                        wSum += w;
                        ++nCount;
                    }

                if (nCount >= options.holeFillMinNeighbors && wSum > 0.0)
                {
                    xCopy.at<float>(row, col) = static_cast<float>(wxSum / wSum);
                    yCopy.at<float>(row, col) = static_cast<float>(wySum / wSum);
                    zCopy.at<float>(row, col) = static_cast<float>(wzSum / wSum);
                    maskCopy.at<uchar>(row, col) = 255;
                }
            }
        }
        demGrid->worldX = xCopy;
        demGrid->worldY = yCopy;
        demGrid->elevation = zCopy;
        demGrid->validMask = maskCopy;
    }

    return demGrid->isValid();
}

} // namespace xjw

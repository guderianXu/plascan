#include "SfmPointCloudFilter.h"

#include "SparsePointCloudWorkspace.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xjw
{

// ============================================================
// SfmPointCloudFilterResult
// ============================================================

std::string SfmPointCloudFilterResult::summary() const
{
    std::ostringstream oss;
    oss << "Point cloud filter result:\n"
        << "  Before: " << pointsBefore << " points\n"
        << "  After:  " << pointsAfter  << " points\n"
        << "  Removed total: " << (pointsBefore - pointsAfter) << "\n";
    if (removedByReprojError > 0)
        oss << "    - By reproj error: " << removedByReprojError << "\n";
    if (removedByTrackLen > 0)
        oss << "    - By track length: " << removedByTrackLen << "\n";
    if (removedByTriAngle > 0)
        oss << "    - By tri angle:    " << removedByTriAngle << "\n";
    if (removedByStatistical > 0)
        oss << "    - By statistical:  " << removedByStatistical << "\n";
    return oss.str();
}

// ============================================================
// SfmPointCloudFilter
// ============================================================

SfmPointCloudFilter::SfmPointCloudFilter(SfmReconstruction &reconstruction)
    : reconstruction_(reconstruction)
{
}

/**
 * @brief 构造函数。
 *
 * 绑定到给定的 `SfmReconstruction`，随后调用 `run()` 执行过滤。
 */

SfmPointCloudFilterResult SfmPointCloudFilter::run(
    const SfmPointCloudFilterOptions &options,
    FilterProgressCallback progressCb)
{
    SfmPointCloudFilterResult result;
    result.pointsBefore = static_cast<int>(reconstruction_.numPoints3D());

    int step = 0;
    const int totalSteps = (options.filterByReprojError ? 1 : 0)
                         + (options.filterByTrackLen    ? 1 : 0)
                         + (options.filterByTriAngle    ? 1 : 0)
                         + (options.filterByStatistical ? 1 : 0);

    auto report = [&](const std::string &msg) -> bool
    {
        if (!progressCb)
        {
            return true;
        }
        int pct = totalSteps > 0 ? (step * 100 / totalSteps) : 100;
        return progressCb(msg, pct);
    };

    // 1. 重投影误差过滤
    if (options.filterByReprojError)
    {
        if (!report("Filtering by reprojection error..."))
        {
            result.pointsAfter = static_cast<int>(reconstruction_.numPoints3D());
            return result;
        }
        result.removedByReprojError = filterByReprojError(options.maxReprojError);
        ++step;
    }

    // 2. 轨迹长度过滤
    if (options.filterByTrackLen)
    {
        if (!report("Filtering by track length..."))
        {
            result.pointsAfter = static_cast<int>(reconstruction_.numPoints3D());
            return result;
        }
        result.removedByTrackLen = filterByTrackLen(options.minTrackLen);
        ++step;
    }

    // 3. 三角化角度过滤
    if (options.filterByTriAngle)
    {
        if (!report("Filtering by triangulation angle..."))
        {
            result.pointsAfter = static_cast<int>(reconstruction_.numPoints3D());
            return result;
        }
        result.removedByTriAngle = filterByTriAngle(options.minTriAngleDeg);
        ++step;
    }

    // 4. 统计离群过滤
    if (options.filterByStatistical)
    {
        if (!report("Statistical outlier removal..."))
        {
            result.pointsAfter = static_cast<int>(reconstruction_.numPoints3D());
            return result;
        }
        result.removedByStatistical = filterByStatistical(options.statK,
                                                          options.statStdDevMul,
                                                          options.processingDevice);
        ++step;
    }

    result.pointsAfter = static_cast<int>(reconstruction_.numPoints3D());
    return result;
}

// ============================================================
// 过滤实现
// ============================================================

int SfmPointCloudFilter::filterByReprojError(double maxError)
{
    int removed = 0;
    auto allIds = reconstruction_.allPoint3DIds();

    for (Point3DId pid : allIds)
    {
        if (!reconstruction_.hasPoint3D(pid))
        {
            continue;
        }
        const auto &pt = reconstruction_.point3D(pid);

        // 使用存储的平均重投影误差
        if (pt.error > maxError)
        {
            reconstruction_.deletePoint3D(pid);
            ++removed;
        }
    }
    return removed;
}

int SfmPointCloudFilter::filterByTrackLen(int minLen)
{
    int removed = 0;
    auto allIds = reconstruction_.allPoint3DIds();

    for (Point3DId pid : allIds)
    {
        if (!reconstruction_.hasPoint3D(pid))
        {
            continue;
        }
        const auto &pt = reconstruction_.point3D(pid);

        // 统计有效观测（已注册图像的观测）
        int validObs = 0;
        for (const auto &elem : pt.track.elements)
        {
            if (reconstruction_.isRegistered(elem.imageId))
            {
                ++validObs;
            }
        }

        if (validObs < minLen)
        {
            reconstruction_.deletePoint3D(pid);
            ++removed;
        }
    }
    return removed;
}

int SfmPointCloudFilter::filterByTriAngle(double minAngleDeg)
{
    int removed = 0;
    auto allIds = reconstruction_.allPoint3DIds();

    for (Point3DId pid : allIds)
    {
        if (!reconstruction_.hasPoint3D(pid))
        {
            continue;
        }
        const auto &pt = reconstruction_.point3D(pid);

        if (pt.track.length() < 2)
        {
            reconstruction_.deletePoint3D(pid);
            ++removed;
            continue;
        }

        // 计算所有观测相机对中的最大三角化角
        double maxAngle = 0.0;
        const auto &elems = pt.track.elements;

        for (size_t i = 0; i < elems.size() && maxAngle < minAngleDeg; ++i)
        {
            if (!reconstruction_.hasCamera(elems[i].imageId))
            {
                continue;
            }
            const Camera &ci = reconstruction_.camera(elems[i].imageId);
            auto Ci = ci.cameraCenter();

            for (size_t j = i + 1; j < elems.size(); ++j)
            {
                if (!reconstruction_.hasCamera(elems[j].imageId))
                {
                    continue;
                }
                const Camera &cj = reconstruction_.camera(elems[j].imageId);
                auto Cj = cj.cameraCenter();

                // 向量从相机中心到 3D 点
                double v0[3] = {pt.xyz[0] - Ci[0], pt.xyz[1] - Ci[1], pt.xyz[2] - Ci[2]};
                double v1[3] = {pt.xyz[0] - Cj[0], pt.xyz[1] - Cj[1], pt.xyz[2] - Cj[2]};
                double len0 = std::sqrt(v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]);
                double len1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);

                if (len0 > 1e-9 && len1 > 1e-9)
                {
                    double cosA = (v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2]) / (len0 * len1);
                    cosA = std::max(-1.0, std::min(1.0, cosA));
                    double angle = std::acos(cosA) * 180.0 / M_PI;
                    if (angle > maxAngle)
                    {
                        maxAngle = angle;
                    }
                }
            }
        }

        if (maxAngle < minAngleDeg)
        {
            reconstruction_.deletePoint3D(pid);
            ++removed;
        }
    }
    return removed;
}

int SfmPointCloudFilter::filterByStatistical(int k,
                                             double stdDevMul,
                                             plapoint::ProcessingDevice processingDevice)
{
    const SparsePointCloudWorkspace workspace =
        SparsePointCloudWorkspace::fromReconstruction(reconstruction_);
    if (workspace.size() <= static_cast<std::size_t>(std::max(0, k)))
    {
        return 0;
    }

    const std::vector<int> removedIndices =
        workspace.statisticalOutlierIndices(k, stdDevMul, processingDevice);

    int removed = 0;
    for (int index : removedIndices)
    {
        if (index >= 0 && static_cast<std::size_t>(index) < workspace.pointIds().size())
        {
            reconstruction_.deletePoint3D(workspace.pointIds()[static_cast<std::size_t>(index)]);
            ++removed;
        }
    }

    return removed;
}

} // namespace xjw

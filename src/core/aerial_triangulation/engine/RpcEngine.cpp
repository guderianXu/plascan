#include "engine/RpcEngineInternals.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xjw::aerial_triangulation::engine
{
    RpcResult runRpc(const RpcInput& input)
    {
        RpcResult result;
        const auto fail = [&](const std::string& error)
        {
            result.error = error;
            return std::move(result);
        };
        const auto canceled = [&input]()
        { return input.cancelFlag && input.cancelFlag->load(std::memory_order_relaxed); };
        if (canceled())
        {
            return fail("用户取消");
        }
        if (!input.graph || input.cameras.size() < 2 || !std::isfinite(input.maximumRmsPixels) ||
            input.maximumRmsPixels <= 0.0)
        {
            return fail("RPC 求解需要连接点图、至少两台相机和正的残差门限");
        }
        const auto tracks = detail::buildRpcTracks(*input.graph);
        result.inputTrackCount = tracks.size();
        if (tracks.empty())
        {
            return fail("RPC 连接点图中没有可交会的多视轨迹");
        }
        result.points.reserve(tracks.size());
        if (input.progressFn)
        {
            input.progressFn("执行 RPC 多视前方交会", 10);
        }
        for (std::size_t index = 0; index < tracks.size(); ++index)
        {
            if (index % 64 == 0)
            {
                if (canceled())
                {
                    return fail("用户取消");
                }
                if (input.progressFn)
                {
                    input.progressFn("执行 RPC 多视前方交会 " + std::to_string(index) + "/" +
                                         std::to_string(tracks.size()),
                                     10 + static_cast<int>(70 * index / tracks.size()));
                }
            }
            RpcPoint point;
            if (detail::intersectRpcTrack(*input.graph,
                                          input.cameras,
                                          tracks[index],
                                          input.maximumRmsPixels,
                                          &point,
                                          &result.cameraResiduals))
            {
                result.points.push_back(std::move(point));
            }
        }
        constexpr int minimumAcceptedPoints = 3;
        if (result.points.size() < minimumAcceptedPoints)
        {
            return fail("RPC 前方交会有效点不足: " + std::to_string(result.points.size()) + "/" +
                        std::to_string(tracks.size()) + "（至少需要 3 个）");
        }
        if (canceled())
        {
            return fail("用户取消");
        }
        result.origin = detail::assignRpcLocalEnu(&result.points);
        result.success = true;
        return result;
    }
} // namespace xjw::aerial_triangulation::engine

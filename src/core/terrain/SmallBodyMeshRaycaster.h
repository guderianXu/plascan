#pragma once

#include "DemDomTypes.h"

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>

namespace xjw
{

/**
 * @brief Accelerated radial ray queries against a small-body triangle mesh.
 *
 * initialize() copies the geometry and sampling attributes it needs, so the
 * source TerrainMeshInput may be released or changed after initialization.
 * Concurrent intersect() calls are safe after initialization has completed.
 */
class SmallBodyMeshRaycaster final
{
public:
    struct Hit
    {
        double radius = 0.0;
        std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
        cv::Vec3d barycentric = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3b colorBgr = cv::Vec3b(128, 128, 128);
        double reliability = 0.0;
        bool ambiguous = false;
    };

    SmallBodyMeshRaycaster();
    SmallBodyMeshRaycaster(const TerrainMeshInput &input,
                           const cv::Vec3d &bodyCenter,
                           QString *errorMsg = nullptr);
    ~SmallBodyMeshRaycaster();

    SmallBodyMeshRaycaster(const SmallBodyMeshRaycaster &) = delete;
    SmallBodyMeshRaycaster &operator=(const SmallBodyMeshRaycaster &) = delete;
    SmallBodyMeshRaycaster(SmallBodyMeshRaycaster &&) noexcept;
    SmallBodyMeshRaycaster &operator=(SmallBodyMeshRaycaster &&) noexcept;

    /**
     * @brief Validate and cache a triangular mesh, then build its BVH.
     * @return false for invalid centers, indices, vertices, UV data, textures,
     *         or degenerate faces. A failed call leaves the previous BVH intact.
     */
    [[nodiscard]] bool initialize(const TerrainMeshInput &input,
                                  const cv::Vec3d &bodyCenter,
                                  QString *errorMsg = nullptr,
                                  const std::atomic_bool *cancelFlag = nullptr);

    [[nodiscard]] bool isInitialized() const noexcept;

    /**
     * @brief Intersect bodyCenter + radius * unitDirection with the mesh.
     *
     * The nearest strictly positive hit is returned. ambiguous is true when
     * another geometrically distinct positive hit exists. The output is not
     * modified on failure. unitDirection must be finite and unit length.
     */
    [[nodiscard]] bool intersect(const cv::Vec3d &unitDirection, Hit *hit) const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace xjw

#include "metmodel/neighbor_selection.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

// Supplied depth.cpp tracked-scene selector. Keep scoring and region gates together.
namespace metmodel
{
    namespace
    {
        struct NeighborRecord
        {
            std::uint32_t common_count = 0;
            std::size_t camera_id = 0;
        };

        std::uint32_t common_track_count(const std::vector<std::uint32_t>& first,
                                         const std::vector<std::uint32_t>& second)
        {
            std::size_t i = 0, j = 0;
            std::uint32_t count = 0;
            while (i < first.size() && j < second.size())
            {
                if (first[i] < second[j])
                    ++i;
                else if (second[j] < first[i])
                    ++j;
                else
                {
                    ++count;
                    ++i;
                    ++j;
                }
            }
            return count;
        }

        bool target_neighbor_graph_point_selected(const SparsePoint& point, const ReconstructionRegion& region)
        {
            if (!region.specified || (region.size.x == 0.0 && region.size.y == 0.0 && region.size.z == 0.0))
                return true;

            // sub_1CC0A10 consumes float3 point records, then evaluates the chunk's
            // 15-double oriented region with inclusive column-axis bounds.
            const double dx = static_cast<double>(static_cast<float>(point.position.x)) - region.center.x;
            const double dy = static_cast<double>(static_cast<float>(point.position.y)) - region.center.y;
            const double dz = static_cast<double>(static_cast<float>(point.position.z)) - region.center.z;
            const double axis0 = region.rotation[0] * dx + region.rotation[3] * dy + region.rotation[6] * dz;
            const double axis1 = region.rotation[1] * dx + region.rotation[4] * dy + region.rotation[7] * dz;
            const double axis2 = region.rotation[2] * dx + region.rotation[5] * dy + region.rotation[8] * dz;
            return axis0 >= -0.5 * region.size.x && axis0 <= 0.5 * region.size.x && axis1 >= -0.5 * region.size.y &&
                   axis1 <= 0.5 * region.size.y && axis2 >= -0.5 * region.size.z && axis2 <= 0.5 * region.size.z;
        }

        std::vector<std::vector<std::uint32_t>>
        make_target_neighbor_graph_tracks(const Scene& scene,
                                          const std::unordered_map<std::uint32_t, const SparsePoint*>& points)
        {
            std::vector<std::vector<std::uint32_t>> result(scene.cameras.size());
            for (std::size_t camera_id = 0; camera_id < scene.cameras.size(); ++camera_id)
            {
                const Camera& camera = scene.cameras[camera_id];
                auto& selected = result[camera_id];
                selected.reserve(camera.track_ids.size());
                for (const std::uint32_t track_id : camera.track_ids)
                {
                    const auto found = points.find(track_id);
                    if (found != points.end() && target_neighbor_graph_point_selected(*found->second, scene.region))
                        selected.push_back(track_id);
                }
            }
            return result;
        }

        bool projection_jacobian(const Camera& camera, metalign::Vec3 point, std::array<double, 6>& jacobian)
        {
            const metalign::Vec3 local = camera.pose.rotation * point + camera.pose.translation;
            if (!(local.z > 1.0e-15) || !std::isfinite(local.z))
                return false;
            const double x = local.x / local.z;
            const double y = local.y / local.z;
            const double r2 = x * x + y * y;
            const auto& c = camera.model;
            const double radial = 1.0 + r2 * (c.k1 + r2 * (c.k2 + r2 * (c.k3 + r2 * c.k4)));
            const double radial_slope = c.k1 + r2 * (2.0 * c.k2 + r2 * (3.0 * c.k3 + r2 * 4.0 * c.k4));
            const double drdx = 2.0 * x * radial_slope;
            const double drdy = 2.0 * y * radial_slope;
            const double dtxdx = 6.0 * c.p1 * x + 2.0 * c.p2 * y + 2.0 * c.p3 * x + 4.0 * c.p4 * x * r2;
            const double dtxdy = 2.0 * c.p1 * y + 2.0 * c.p2 * x + 2.0 * c.p3 * y + 4.0 * c.p4 * y * r2;
            const double dtydx = 2.0 * c.p1 * y + 2.0 * c.p2 * x;
            const double dtydy = 2.0 * c.p1 * x + 6.0 * c.p2 * y;
            const double dxdx = radial + x * drdx + dtxdx;
            const double dxdy = x * drdy + dtxdy;
            const double dydx = y * drdx + dtydx;
            const double dydy = radial + y * drdy + dtydy;
            const double dudx = (c.f + c.b1) * dxdx + c.b2 * dydx;
            const double dudy = (c.f + c.b1) * dxdy + c.b2 * dydy;
            const double dvdx = c.f * dydx;
            const double dvdy = c.f * dydy;
            const std::array<double, 3> uq{dudx / local.z, dudy / local.z, -(dudx * x + dudy * y) / local.z};
            const std::array<double, 3> vq{dvdx / local.z, dvdy / local.z, -(dvdx * x + dvdy * y) / local.z};
            for (std::size_t column = 0; column < 3; ++column)
            {
                jacobian[column] = uq[0] * camera.pose.rotation(0, column) + uq[1] * camera.pose.rotation(1, column) +
                                   uq[2] * camera.pose.rotation(2, column);
                jacobian[3 + column] = vq[0] * camera.pose.rotation(0, column) +
                                       vq[1] * camera.pose.rotation(1, column) +
                                       vq[2] * camera.pose.rotation(2, column);
            }
            return std::all_of(jacobian.begin(), jacobian.end(), [](double value) { return std::isfinite(value); });
        }

        double reciprocal_condition(const std::array<double, 12>& stacked)
        {
            std::array<double, 9> a{};
            for (std::size_t row = 0; row < 4; ++row)
                for (std::size_t i = 0; i < 3; ++i)
                    for (std::size_t j = 0; j < 3; ++j)
                        a[i * 3 + j] += stacked[row * 3 + i] * stacked[row * 3 + j];
            const double q = (a[0] + a[4] + a[8]) / 3.0;
            const double p2 = (a[0] - q) * (a[0] - q) + (a[4] - q) * (a[4] - q) + (a[8] - q) * (a[8] - q) +
                              2.0 * (a[1] * a[1] + a[2] * a[2] + a[5] * a[5]);
            if (!(p2 > 0.0))
                return 0.001;
            const double p = std::sqrt(p2 / 6.0);
            std::array<double, 9> b = a;
            b[0] -= q;
            b[4] -= q;
            b[8] -= q;
            for (double& value : b)
                value /= p;
            const double determinant = b[0] * (b[4] * b[8] - b[5] * b[7]) - b[1] * (b[3] * b[8] - b[5] * b[6]) +
                                       b[2] * (b[3] * b[7] - b[4] * b[6]);
            const double phi = std::acos(std::clamp(determinant * 0.5, -1.0, 1.0)) / 3.0;
            constexpr double two_pi_over_three = 2.0943951023931954923;
            const double largest = q + 2.0 * p * std::cos(phi);
            const double smallest = q + 2.0 * p * std::cos(phi + two_pi_over_three);
            if (!(smallest > 0.0) || !(largest > 0.0))
                return 0.001;
            const double value = std::sqrt(smallest / largest);
            return value < 0.001 ? 0.001 : value;
        }

        double pair_score(const Scene& scene,
                          std::size_t first_id,
                          std::size_t second_id,
                          double lower_cos,
                          double upper_cos,
                          const std::unordered_map<std::uint32_t, const SparsePoint*>& points)
        {
            const Camera& first = scene.cameras[first_id];
            const Camera& second = scene.cameras[second_id];
            std::size_t i = 0, j = 0;
            double score = 0.0;
            while (i < first.track_ids.size() && j < second.track_ids.size())
            {
                if (first.track_ids[i] < second.track_ids[j])
                {
                    ++i;
                    continue;
                }
                if (second.track_ids[j] < first.track_ids[i])
                {
                    ++j;
                    continue;
                }
                const auto found = points.find(first.track_ids[i]);
                ++i;
                ++j;
                if (found == points.end())
                    continue;
                const SparsePoint& point = *found->second;
                const metalign::Vec3 first_ray = metalign::normalized(point.position - first.center);
                const metalign::Vec3 second_ray = metalign::normalized(point.position - second.center);
                const double angle_cos = metalign::dot(first_ray, second_ray);
                if (angle_cos < lower_cos || angle_cos > upper_cos)
                    continue;
                if (point.homogeneous_w == 0.0F)
                {
                    score += 1.0;
                    continue;
                }
                std::array<double, 6> first_j{}, second_j{};
                if (!projection_jacobian(first, point.position, first_j) ||
                    !projection_jacobian(second, point.position, second_j))
                    continue;
                std::array<double, 12> stacked{};
                std::copy(first_j.begin(), first_j.end(), stacked.begin());
                std::copy(second_j.begin(), second_j.end(), stacked.begin() + 6);
                score += reciprocal_condition(stacked);
            }
            return score;
        }

    } // namespace

    std::vector<std::vector<std::size_t>> select_recovered_neighbors(const Scene& scene, std::size_t max_neighbors)
    {
        std::vector<std::vector<std::size_t>> result(scene.cameras.size());
        const bool tracked = std::any_of(
            scene.cameras.begin(), scene.cameras.end(), [](const Camera& camera) { return !camera.track_ids.empty(); });
        if (tracked)
        {
            std::unordered_map<std::uint32_t, const SparsePoint*> points;
            points.reserve(scene.sparse_points.size());
            for (const SparsePoint& point : scene.sparse_points)
                if (point.track_id != std::numeric_limits<std::uint32_t>::max())
                    points[point.track_id] = &point;
            const auto graph_tracks = make_target_neighbor_graph_tracks(scene, points);
            std::vector<std::vector<NeighborRecord>> graph(scene.cameras.size());
            for (std::size_t first = 0; first < scene.cameras.size(); ++first)
            {
                if (!scene.cameras[first].aligned)
                    continue;
                for (std::size_t second = first + 1; second < scene.cameras.size(); ++second)
                {
                    if (!scene.cameras[second].aligned)
                        continue;
                    const std::uint32_t common = common_track_count(graph_tracks[first], graph_tracks[second]);
                    if (common == 0)
                        continue;
                    graph[first].push_back({common, second});
                    graph[second].push_back({common, first});
                }
            }
            // Exact float cosine table passed by 0x1CFB150 to 0x12D4A80.
            constexpr std::array<float, 7> angle_cosines{std::bit_cast<float>(0x3F800000U),
                                                         std::bit_cast<float>(0x3F7F069EU),
                                                         std::bit_cast<float>(0x3F7C1C5CU),
                                                         std::bit_cast<float>(0x3F708FB2U),
                                                         std::bit_cast<float>(0x3F441B7DU),
                                                         std::bit_cast<float>(0x3F000000U),
                                                         std::bit_cast<float>(0xBF800000U)};
            for (std::size_t reference = 0; reference < graph.size(); ++reference)
            {
                auto& records = graph[reference];
                std::sort(records.begin(),
                          records.end(),
                          [](const NeighborRecord& a, const NeighborRecord& b) {
                              return a.common_count != b.common_count ? a.common_count > b.common_count
                                                                      : a.camera_id > b.camera_id;
                          });
                if (records.empty())
                    continue;
                const std::uint32_t threshold = std::max<std::uint32_t>(
                    static_cast<std::uint32_t>(scene.neighbor_common_threshold), records.front().common_count / 10U);
                records.erase(std::find_if(records.begin(),
                                           records.end(),
                                           [threshold](const NeighborRecord& record)
                                           { return record.common_count < threshold; }),
                              records.end());
                std::vector<std::size_t> remaining;
                remaining.reserve(records.size());
                for (const NeighborRecord& record : records)
                    remaining.push_back(record.camera_id);
                // The target materializes the surviving candidate set in ascending
                // camera-ID order before every score pass.
                std::sort(remaining.begin(), remaining.end());
                auto choose = [&](double lower_cos, double upper_cos, std::size_t requested)
                {
                    if (requested == 0 || remaining.empty())
                        return;
                    std::vector<std::pair<double, std::size_t>> scored;
                    scored.reserve(remaining.size());
                    for (const std::size_t candidate : remaining)
                    {
                        const double score = pair_score(scene, reference, candidate, lower_cos, upper_cos, points);
                        scored.emplace_back(score, candidate);
                    }
                    std::sort(
                        scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    const std::size_t take = std::min(requested, scored.size());
                    for (std::size_t index = 0; index < take; ++index)
                    {
                        result[reference].push_back(scored[index].second);
                        std::erase(remaining, scored[index].second);
                    }
                };
                if (max_neighbors > 5)
                {
                    const std::size_t quota = max_neighbors / 6;
                    for (std::size_t band = 0; band < 6; ++band)
                    {
                        const double lower_cos = angle_cosines[band + 1];
                        const double upper_cos = angle_cosines[band];
                        choose(lower_cos, upper_cos, quota);
                    }
                }
                if (result[reference].size() < max_neighbors)
                    choose(-1.0, 1.0, max_neighbors - result[reference].size());
                std::sort(result[reference].begin(),
                          result[reference].end(),
                          [&](std::size_t a, std::size_t b)
                          {
                              const auto find_count = [&](std::size_t id)
                              {
                                  return std::find_if(records.begin(),
                                                      records.end(),
                                                      [id](const NeighborRecord& r) { return r.camera_id == id; })
                                      ->common_count;
                              };
                              const auto ac = find_count(a), bc = find_count(b);
                              return ac != bc ? ac > bc : a > b;
                          });
            }
            return result;
        }
        throw std::invalid_argument("Recovered neighbor selection requires sparse track observations");
    }
} // namespace metmodel

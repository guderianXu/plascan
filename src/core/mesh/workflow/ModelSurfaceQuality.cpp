#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow::workflow_detail
{
    using namespace workflow_detail;

    bool hasSameFaceIndexBuffer(const TriMesh& first, const TriMesh& second)
    {
        if (first.faces.size() != second.faces.size())
        {
            return false;
        }
        for (std::size_t index = 0; index < first.faces.size(); ++index)
        {
            for (int corner = 0; corner < 3; ++corner)
            {
                if (first.faces[index].v[corner] != second.faces[index].v[corner])
                {
                    return false;
                }
            }
        }
        return true;
    }

    QJsonArray eulerCharacteristicsToJson(const std::vector<int>& values)
    {
        QJsonArray json;
        for (const int value : values)
        {
            json.append(value);
        }
        return json;
    }

    double meshSurfaceArea(const TriMesh& mesh)
    {
        double area = 0.0;
        for (const Triangle& face : mesh.faces)
        {
            const MeshVertex& first = mesh.vertices[static_cast<std::size_t>(face.v[0])];
            const MeshVertex& second = mesh.vertices[static_cast<std::size_t>(face.v[1])];
            const MeshVertex& third = mesh.vertices[static_cast<std::size_t>(face.v[2])];
            const double ab_x = second.x - first.x;
            const double ab_y = second.y - first.y;
            const double ab_z = second.z - first.z;
            const double ac_x = third.x - first.x;
            const double ac_y = third.y - first.y;
            const double ac_z = third.z - first.z;
            const double cross_x = ab_y * ac_z - ab_z * ac_y;
            const double cross_y = ab_z * ac_x - ab_x * ac_z;
            const double cross_z = ab_x * ac_y - ab_y * ac_x;
            area += 0.5 * std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
        }
        return area;
    }

    double meshAbsoluteOrientedVolume(const TriMesh& mesh)
    {
        if (mesh.vertices.empty())
        {
            return 0.0;
        }

        const MeshVertex& origin = mesh.vertices.front();
        double signed_volume_six = 0.0;
        for (const Triangle& face : mesh.faces)
        {
            const MeshVertex& first = mesh.vertices[static_cast<std::size_t>(face.v[0])];
            const MeshVertex& second = mesh.vertices[static_cast<std::size_t>(face.v[1])];
            const MeshVertex& third = mesh.vertices[static_cast<std::size_t>(face.v[2])];
            const double first_x = first.x - origin.x;
            const double first_y = first.y - origin.y;
            const double first_z = first.z - origin.z;
            const double second_x = second.x - origin.x;
            const double second_y = second.y - origin.y;
            const double second_z = second.z - origin.z;
            const double third_x = third.x - origin.x;
            const double third_y = third.y - origin.y;
            const double third_z = third.z - origin.z;
            const double cross_x = second_y * third_z - second_z * third_y;
            const double cross_y = second_z * third_x - second_x * third_z;
            const double cross_z = second_x * third_y - second_y * third_x;
            signed_volume_six += first_x * cross_x + first_y * cross_y + first_z * cross_z;
        }
        return std::abs(signed_volume_six) / 6.0;
    }

    double meshMeanNormalVariation(const TriMesh& mesh)
    {
        double variation_sum = 0.0;
        std::uint64_t sample_count = 0;
        for (const Triangle& face : mesh.faces)
        {
            for (int corner = 0; corner < 3; ++corner)
            {
                const MeshVertex& first = mesh.vertices[static_cast<std::size_t>(face.v[corner])];
                const MeshVertex& second = mesh.vertices[static_cast<std::size_t>(face.v[(corner + 1) % 3])];
                const double dot = std::clamp(
                    static_cast<double>(first.nx * second.nx + first.ny * second.ny + first.nz * second.nz), -1.0, 1.0);
                variation_sum += 1.0 - dot;
                ++sample_count;
            }
        }
        return sample_count > 0 ? variation_sum / static_cast<double>(sample_count) : 0.0;
    }

    RefinementQualityGuardResult limitRefinementRoughness(const TriMesh& original,
                                                          TriMesh* refined,
                                                          double maximumAreaGrowth,
                                                          double maximumNormalVariationGrowth)
    {
        RefinementQualityGuardResult result;
        if (refined == nullptr || original.vertices.size() != refined->vertices.size() ||
            original.faces.size() != refined->faces.size())
        {
            return result;
        }

        result.applied = true;
        result.areaBefore = meshSurfaceArea(original);
        result.normalVariationBefore = meshMeanNormalVariation(original);
        const auto acceptable = [&](const TriMesh& candidate)
        {
            const double candidate_area = meshSurfaceArea(candidate);
            const double candidate_variation = meshMeanNormalVariation(candidate);
            const double area_limit = result.areaBefore * maximumAreaGrowth;
            const double variation_limit = std::max(result.normalVariationBefore * maximumNormalVariationGrowth,
                                                    result.normalVariationBefore + 1.0e-6);
            return candidate_area <= area_limit && candidate_variation <= variation_limit;
        };
        if (!acceptable(*refined))
        {
            const TriMesh full_refinement = *refined;
            bool accepted = false;
            for (const float blend : {0.75f, 0.50f, 0.25f})
            {
                TriMesh candidate = original;
                for (std::size_t index = 0; index < candidate.vertices.size(); ++index)
                {
                    MeshVertex& vertex = candidate.vertices[index];
                    const MeshVertex& target = full_refinement.vertices[index];
                    vertex.x += blend * (target.x - vertex.x);
                    vertex.y += blend * (target.y - vertex.y);
                    vertex.z += blend * (target.z - vertex.z);
                }
                detail::recomputeNormals(&candidate);
                if (acceptable(candidate))
                {
                    *refined = std::move(candidate);
                    result.acceptedBlend = blend;
                    accepted = true;
                    break;
                }
            }
            if (!accepted)
            {
                *refined = original;
                result.acceptedBlend = 0.0f;
            }
            result.limited = true;
        }
        result.areaAfter = meshSurfaceArea(*refined);
        result.normalVariationAfter = meshMeanNormalVariation(*refined);
        return result;
    }
} // namespace xjw::mesh::workflow::workflow_detail

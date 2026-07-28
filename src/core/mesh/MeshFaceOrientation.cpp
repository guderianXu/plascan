#include "MeshFaceOrientation.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace xjw::mesh
{
namespace
{

struct EdgeRecord
{
    std::uint64_t key = 0;
    int face = -1;
    bool forward = false;
};

struct FaceConstraint
{
    int firstFace = -1;
    int secondFace = -1;
    bool differentFlip = false;
};

std::uint64_t edgeKey(int first, int second)
{
    const std::uint32_t low = static_cast<std::uint32_t>(
        std::min(first, second));
    const std::uint32_t high = static_cast<std::uint32_t>(
        std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

class ParityDisjointSet
{
public:
    explicit ParityDisjointSet(int size)
        : _parent(static_cast<std::size_t>(size)),
          _rank(static_cast<std::size_t>(size), 0),
          _parityToParent(static_cast<std::size_t>(size), 0)
    {
        std::iota(_parent.begin(), _parent.end(), 0);
    }

    std::pair<int, std::uint8_t> find(int value)
    {
        const std::size_t index = static_cast<std::size_t>(value);
        if (_parent[index] == value)
        {
            return {value, 0};
        }
        const auto parent_result = find(_parent[index]);
        _parityToParent[index] ^= parent_result.second;
        _parent[index] = parent_result.first;
        return {_parent[index], _parityToParent[index]};
    }

    bool unite(int first, int second, bool differentFlip)
    {
        auto first_result = find(first);
        auto second_result = find(second);
        const std::uint8_t required_parity =
            differentFlip ? std::uint8_t{1} : std::uint8_t{0};
        if (first_result.first == second_result.first)
        {
            return (first_result.second ^ second_result.second) ==
                required_parity;
        }

        if (_rank[static_cast<std::size_t>(first_result.first)] <
            _rank[static_cast<std::size_t>(second_result.first)])
        {
            std::swap(first_result, second_result);
        }
        _parent[static_cast<std::size_t>(second_result.first)] =
            first_result.first;
        _parityToParent[static_cast<std::size_t>(second_result.first)] =
            first_result.second ^ second_result.second ^ required_parity;
        if (_rank[static_cast<std::size_t>(first_result.first)] ==
            _rank[static_cast<std::size_t>(second_result.first)])
        {
            ++_rank[static_cast<std::size_t>(first_result.first)];
        }
        return true;
    }

private:
    std::vector<int> _parent;
    std::vector<std::uint8_t> _rank;
    std::vector<std::uint8_t> _parityToParent;
};

} // namespace

MeshFaceOrientationStatistics repairMeshFaceOrientation(TriMesh *mesh)
{
    MeshFaceOrientationStatistics statistics;
    if (mesh == nullptr || mesh->faces.empty())
    {
        return statistics;
    }

    std::vector<EdgeRecord> edges;
    edges.reserve(mesh->faces.size() * 3);
    for (int face_index = 0;
         face_index < static_cast<int>(mesh->faces.size());
         ++face_index)
    {
        const Triangle &face =
            mesh->faces[static_cast<std::size_t>(face_index)];
        for (int corner = 0; corner < 3; ++corner)
        {
            const int first = face.v[corner];
            const int second = face.v[(corner + 1) % 3];
            if (first == second || first < 0 || second < 0)
            {
                continue;
            }
            edges.push_back({
                edgeKey(first, second),
                face_index,
                first < second});
        }
    }
    std::sort(
        edges.begin(),
        edges.end(),
        [](const EdgeRecord &first, const EdgeRecord &second)
        {
            return first.key < second.key;
        });

    std::vector<FaceConstraint> constraints;
    constraints.reserve(edges.size() / 2);
    for (std::size_t begin = 0; begin < edges.size();)
    {
        std::size_t end = begin + 1;
        while (end < edges.size() && edges[end].key == edges[begin].key)
        {
            ++end;
        }
        const std::size_t count = end - begin;
        if (count == 2)
        {
            ++statistics.sharedEdgeCount;
            const bool same_direction =
                edges[begin].forward == edges[begin + 1].forward;
            statistics.inconsistentSharedEdgeCountBefore +=
                same_direction ? 1 : 0;
            constraints.push_back({
                edges[begin].face,
                edges[begin + 1].face,
                same_direction});
        }
        else if (count > 2)
        {
            ++statistics.nonManifoldEdgeCount;
        }
        begin = end;
    }

    if (statistics.nonManifoldEdgeCount > 0)
    {
        return statistics;
    }

    ParityDisjointSet sets(static_cast<int>(mesh->faces.size()));
    std::vector<bool> contradictory_faces(mesh->faces.size(), false);
    for (const FaceConstraint &constraint : constraints)
    {
        if (!sets.unite(
                constraint.firstFace,
                constraint.secondFace,
                constraint.differentFlip))
        {
            ++statistics.orientationConflictCount;
            contradictory_faces[
                static_cast<std::size_t>(constraint.secondFace)] = true;
        }
    }
    if (statistics.orientationConflictCount > 0)
    {
        const std::size_t face_count_before = mesh->faces.size();
        int face_index = 0;
        mesh->faces.erase(
            std::remove_if(
                mesh->faces.begin(),
                mesh->faces.end(),
                [&contradictory_faces, &face_index](const Triangle &)
                {
                    const bool remove = contradictory_faces[
                        static_cast<std::size_t>(face_index)];
                    ++face_index;
                    return remove;
                }),
            mesh->faces.end());
        statistics.removedContradictoryFaceCount = static_cast<int>(
            face_count_before - mesh->faces.size());
        if (statistics.removedContradictoryFaceCount <= 0 ||
            mesh->faces.empty())
        {
            return statistics;
        }

        const MeshFaceOrientationStatistics repaired =
            repairMeshFaceOrientation(mesh);
        statistics.inconsistentSharedEdgeCountAfter =
            repaired.inconsistentSharedEdgeCountAfter;
        statistics.flippedFaceCount += repaired.flippedFaceCount;
        statistics.removedContradictoryFaceCount +=
            repaired.removedContradictoryFaceCount;
        statistics.nonManifoldEdgeCount += repaired.nonManifoldEdgeCount;
        statistics.orientationConflictCount +=
            repaired.orientationConflictCount;
        statistics.succeeded = repaired.succeeded;
        return statistics;
    }

    std::vector<int> root_by_face(mesh->faces.size(), -1);
    std::vector<std::uint8_t> parity_by_face(mesh->faces.size(), 0);
    std::vector<int> zero_count(mesh->faces.size(), 0);
    std::vector<int> one_count(mesh->faces.size(), 0);
    for (int face_index = 0;
         face_index < static_cast<int>(mesh->faces.size());
         ++face_index)
    {
        const auto result = sets.find(face_index);
        root_by_face[static_cast<std::size_t>(face_index)] = result.first;
        parity_by_face[static_cast<std::size_t>(face_index)] = result.second;
        if (result.second == 0)
        {
            ++zero_count[static_cast<std::size_t>(result.first)];
        }
        else
        {
            ++one_count[static_cast<std::size_t>(result.first)];
        }
    }

    for (int face_index = 0;
         face_index < static_cast<int>(mesh->faces.size());
         ++face_index)
    {
        const int root = root_by_face[static_cast<std::size_t>(face_index)];
        const bool invert_component =
            one_count[static_cast<std::size_t>(root)] >
            zero_count[static_cast<std::size_t>(root)];
        const bool flip =
            (parity_by_face[static_cast<std::size_t>(face_index)] != 0) ^
            invert_component;
        if (flip)
        {
            Triangle &face =
                mesh->faces[static_cast<std::size_t>(face_index)];
            std::swap(face.v[1], face.v[2]);
            ++statistics.flippedFaceCount;
        }
    }

    for (const FaceConstraint &constraint : constraints)
    {
        const bool first_flipped =
            (parity_by_face[static_cast<std::size_t>(constraint.firstFace)] != 0) ^
            (one_count[static_cast<std::size_t>(
                root_by_face[static_cast<std::size_t>(constraint.firstFace)])] >
             zero_count[static_cast<std::size_t>(
                root_by_face[static_cast<std::size_t>(constraint.firstFace)])]);
        const bool second_flipped =
            (parity_by_face[static_cast<std::size_t>(constraint.secondFace)] != 0) ^
            (one_count[static_cast<std::size_t>(
                root_by_face[static_cast<std::size_t>(constraint.secondFace)])] >
             zero_count[static_cast<std::size_t>(
                root_by_face[static_cast<std::size_t>(constraint.secondFace)])]);
        const bool satisfied =
            (first_flipped ^ second_flipped) ==
            constraint.differentFlip;
        statistics.inconsistentSharedEdgeCountAfter += satisfied ? 0 : 1;
    }
    statistics.succeeded =
        statistics.inconsistentSharedEdgeCountAfter == 0;
    return statistics;
}

} // namespace xjw::mesh

#include "MeshAcquisitionGapReport.h"

#include <QJsonArray>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace xjw::mesh
{
namespace
{

constexpr int kAzimuthSectorCount = 12;
constexpr int kElevationBandCount = 3;
constexpr int kOrientationBinCount = kAzimuthSectorCount * kElevationBandCount;
constexpr int kReasonCount = 9;
constexpr double kPi = 3.14159265358979323846;

struct OrientationBin
{
    std::uint64_t edgeCount = 0;
    double boundaryLength = 0.0;
    double severity = 0.0;
    std::array<std::uint64_t, kReasonCount> reasons{};
    std::array<std::uint64_t, 4> sourceCountHistogram{};
    std::array<double, 3> normalSum{};
    std::uint64_t validNormalCount = 0;
};

struct FrameCoverage
{
    std::uint64_t participatingEdgeCount = 0;
    std::uint64_t exclusiveEdgeCount = 0;
    std::uint64_t sharedEdgeCount = 0;
    std::array<std::uint64_t, kOrientationBinCount> orientationCounts{};
};

int reasonIndex(MeshBoundaryAttributionReason reason)
{
    return std::clamp(static_cast<int>(reason), 0, kReasonCount - 1);
}

QString reasonName(MeshBoundaryAttributionReason reason)
{
    switch (reason)
    {
    case MeshBoundaryAttributionReason::ExtractionOrPostprocess:
        return QStringLiteral("extraction_or_postprocess");
    case MeshBoundaryAttributionReason::SupportGateRejected:
        return QStringLiteral("support_gate");
    case MeshBoundaryAttributionReason::AbsoluteTsdfRejected:
        return QStringLiteral("absolute_tsdf");
    case MeshBoundaryAttributionReason::SurfaceWeightRejected:
        return QStringLiteral("surface_weight");
    case MeshBoundaryAttributionReason::DepthSpreadRejected:
        return QStringLiteral("depth_spread");
    case MeshBoundaryAttributionReason::InsufficientSource:
        return QStringLiteral("insufficient_source");
    case MeshBoundaryAttributionReason::NoObservation:
        return QStringLiteral("no_observation");
    case MeshBoundaryAttributionReason::Unclassified:
        return QStringLiteral("unclassified");
    case MeshBoundaryAttributionReason::None:
    default:
        return QStringLiteral("none");
    }
}

double reasonSeverity(MeshBoundaryAttributionReason reason)
{
    switch (reason)
    {
    case MeshBoundaryAttributionReason::NoObservation:
        return 4.0;
    case MeshBoundaryAttributionReason::InsufficientSource:
        return 3.0;
    case MeshBoundaryAttributionReason::DepthSpreadRejected:
        return 2.5;
    case MeshBoundaryAttributionReason::SurfaceWeightRejected:
        return 2.0;
    case MeshBoundaryAttributionReason::AbsoluteTsdfRejected:
        return 1.5;
    case MeshBoundaryAttributionReason::SupportGateRejected:
        return 1.25;
    case MeshBoundaryAttributionReason::ExtractionOrPostprocess:
        return 0.5;
    case MeshBoundaryAttributionReason::Unclassified:
        return 1.0;
    case MeshBoundaryAttributionReason::None:
    default:
        return 0.0;
    }
}

QString recommendation(MeshBoundaryAttributionReason reason)
{
    switch (reason)
    {
    case MeshBoundaryAttributionReason::NoObservation:
        return QStringLiteral("沿该表面法向补拍，并增加略高和略低俯仰角绕开遮挡");
    case MeshBoundaryAttributionReason::InsufficientSource:
        return QStringLiteral("在相邻机位间增加约15至30度基线，避免重复同一方向");
    case MeshBoundaryAttributionReason::DepthSpreadRejected:
    case MeshBoundaryAttributionReason::SurfaceWeightRejected:
    case MeshBoundaryAttributionReason::AbsoluteTsdfRejected:
        return QStringLiteral("先改善焦距、曝光、纹理或拍摄距离，再做该扇区局部密集匹配");
    case MeshBoundaryAttributionReason::SupportGateRejected:
        return QStringLiteral("现有观测接近可用，优先按该扇区做严格多基线局部重算");
    case MeshBoundaryAttributionReason::ExtractionOrPostprocess:
        return QStringLiteral("观测已支持，优先检查等值面提取和网格后处理，不建议补拍");
    case MeshBoundaryAttributionReason::Unclassified:
        return QStringLiteral("保留调试样本并人工复核，暂不自动给出补拍结论");
    case MeshBoundaryAttributionReason::None:
    default:
        return QStringLiteral("无建议");
    }
}

int orientationBin(const MeshBoundaryEdgeAttribution &edge,
                   const std::array<double, 3> &center)
{
    const double x = static_cast<double>(edge.midpoint[0]) - center[0];
    const double y = static_cast<double>(edge.midpoint[1]) - center[1];
    const double z = static_cast<double>(edge.midpoint[2]) - center[2];
    const double length = std::sqrt(x * x + y * y + z * z);
    if (!std::isfinite(length) || length <= std::numeric_limits<double>::epsilon())
    {
        return -1;
    }
    double azimuth = std::atan2(y, x);
    if (azimuth < 0.0)
    {
        azimuth += 2.0 * kPi;
    }
    const int azimuth_sector = std::clamp(
        static_cast<int>(std::floor(
            azimuth / (2.0 * kPi) * kAzimuthSectorCount)),
        0,
        kAzimuthSectorCount - 1);
    const double vertical = z / length;
    const int elevation_band = vertical < -0.35 ? 0 : (vertical > 0.35 ? 2 : 1);
    return elevation_band * kAzimuthSectorCount + azimuth_sector;
}

MeshBoundaryAttributionReason dominantReason(const OrientationBin &bin)
{
    int best_index = 0;
    std::uint64_t best_count = 0;
    double best_severity = -1.0;
    for (int index = 0; index < kReasonCount; ++index)
    {
        const auto reason = static_cast<MeshBoundaryAttributionReason>(index);
        const std::uint64_t count = bin.reasons[static_cast<std::size_t>(index)];
        const double severity = reasonSeverity(reason);
        if (count > best_count || (count == best_count && severity > best_severity))
        {
            best_index = index;
            best_count = count;
            best_severity = severity;
        }
    }
    return static_cast<MeshBoundaryAttributionReason>(best_index);
}

QJsonObject reasonCountsToJson(
    const std::array<std::uint64_t, kReasonCount> &counts)
{
    QJsonObject json;
    for (int index = 0; index < kReasonCount; ++index)
    {
        json[reasonName(static_cast<MeshBoundaryAttributionReason>(index))] =
            static_cast<double>(counts[static_cast<std::size_t>(index)]);
    }
    return json;
}

QJsonObject orientationBinToJson(int index, const OrientationBin &bin)
{
    const int elevation = index / kAzimuthSectorCount;
    const int azimuth = index % kAzimuthSectorCount;
    const double sector_degrees = 360.0 / kAzimuthSectorCount;
    const MeshBoundaryAttributionReason dominant = dominantReason(bin);
    QJsonArray recommended_direction;
    const double normal_length = std::sqrt(
        bin.normalSum[0] * bin.normalSum[0] +
        bin.normalSum[1] * bin.normalSum[1] +
        bin.normalSum[2] * bin.normalSum[2]);
    if (bin.validNormalCount > 0 && normal_length > 1.0e-12)
    {
        recommended_direction.append(bin.normalSum[0] / normal_length);
        recommended_direction.append(bin.normalSum[1] / normal_length);
        recommended_direction.append(bin.normalSum[2] / normal_length);
    }
    return {
        {QStringLiteral("index"), index},
        {QStringLiteral("azimuth_sector"), azimuth},
        {QStringLiteral("azimuth_start_degrees"), azimuth * sector_degrees},
        {QStringLiteral("azimuth_end_degrees"), (azimuth + 1) * sector_degrees},
        {QStringLiteral("elevation_band"),
         elevation == 0 ? QStringLiteral("lower")
                        : (elevation == 1 ? QStringLiteral("middle")
                                          : QStringLiteral("upper"))},
        {QStringLiteral("edge_count"), static_cast<double>(bin.edgeCount)},
        {QStringLiteral("boundary_length"), bin.boundaryLength},
        {QStringLiteral("severity"), bin.severity},
        {QStringLiteral("reason_counts"), reasonCountsToJson(bin.reasons)},
        {QStringLiteral("source_count_0"),
         static_cast<double>(bin.sourceCountHistogram[0])},
        {QStringLiteral("source_count_1"),
         static_cast<double>(bin.sourceCountHistogram[1])},
        {QStringLiteral("source_count_2"),
         static_cast<double>(bin.sourceCountHistogram[2])},
        {QStringLiteral("source_count_3_plus"),
         static_cast<double>(bin.sourceCountHistogram[3])},
        {QStringLiteral("dominant_reason"), reasonName(dominant)},
        {QStringLiteral("recommendation"), recommendation(dominant)},
        {QStringLiteral("recommended_observation_direction"),
         recommended_direction}};
}

} // namespace

QJsonObject buildMeshAcquisitionGapReport(
    const std::vector<MeshBoundaryEdgeAttribution> &edges,
    const DepthTsdfLayout &layout,
    const QStringList &frameLabels,
    int inputFrameCount)
{
    std::array<OrientationBin, kOrientationBinCount> bins{};
    OrientationBin unknown_bin;
    std::array<std::uint64_t, kReasonCount> primary_reasons{};
    std::array<std::uint64_t, kReasonCount> root_causes{};
    std::array<FrameCoverage, 16> frame_coverage{};
    const std::array<double, 3> center{{
        0.5 * (layout.boundsMin[0] + layout.boundsMax[0]),
        0.5 * (layout.boundsMin[1] + layout.boundsMax[1]),
        0.5 * (layout.boundsMin[2] + layout.boundsMax[2])}};

    for (const MeshBoundaryEdgeAttribution &edge : edges)
    {
        const int primary_index = reasonIndex(edge.reason);
        const int cause_index = reasonIndex(edge.evidenceReason);
        ++primary_reasons[static_cast<std::size_t>(primary_index)];
        ++root_causes[static_cast<std::size_t>(cause_index)];
        const int bin_index = orientationBin(edge, center);
        OrientationBin &bin = bin_index >= 0
            ? bins[static_cast<std::size_t>(bin_index)]
            : unknown_bin;
        ++bin.edgeCount;
        bin.boundaryLength += edge.length;
        bin.severity += reasonSeverity(edge.evidenceReason);
        ++bin.reasons[static_cast<std::size_t>(cause_index)];
        const int source_count = std::popcount(edge.sourceMask);
        ++bin.sourceCountHistogram[static_cast<std::size_t>(
            std::min(source_count, 3))];
        const double normal_length = std::sqrt(
            static_cast<double>(edge.normal[0]) * edge.normal[0] +
            static_cast<double>(edge.normal[1]) * edge.normal[1] +
            static_cast<double>(edge.normal[2]) * edge.normal[2]);
        if (std::isfinite(normal_length) && normal_length > 1.0e-8)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                bin.normalSum[static_cast<std::size_t>(axis)] +=
                    edge.normal[static_cast<std::size_t>(axis)] / normal_length;
            }
            ++bin.validNormalCount;
        }
        for (int frame = 0; frame < 16; ++frame)
        {
            if ((edge.sourceMask & (static_cast<std::uint16_t>(1U) << frame)) == 0)
            {
                continue;
            }
            FrameCoverage &coverage = frame_coverage[static_cast<std::size_t>(frame)];
            ++coverage.participatingEdgeCount;
            if (source_count == 1)
            {
                ++coverage.exclusiveEdgeCount;
            }
            else
            {
                ++coverage.sharedEdgeCount;
            }
            if (bin_index >= 0)
            {
                ++coverage.orientationCounts[static_cast<std::size_t>(bin_index)];
            }
        }
    }

    QJsonArray orientation_bins;
    for (int index = 0; index < kOrientationBinCount; ++index)
    {
        orientation_bins.append(orientationBinToJson(
            index, bins[static_cast<std::size_t>(index)]));
    }
    std::array<int, kOrientationBinCount> ranked_indices{};
    std::iota(ranked_indices.begin(), ranked_indices.end(), 0);
    std::stable_sort(
        ranked_indices.begin(),
        ranked_indices.end(),
        [&bins](int lhs, int rhs)
        {
            const OrientationBin &left = bins[static_cast<std::size_t>(lhs)];
            const OrientationBin &right = bins[static_cast<std::size_t>(rhs)];
            if (left.severity != right.severity)
            {
                return left.severity > right.severity;
            }
            return left.edgeCount > right.edgeCount;
        });
    QJsonArray highest_risk_sectors;
    for (int rank = 0; rank < 3; ++rank)
    {
        highest_risk_sectors.append(orientationBinToJson(
            ranked_indices[static_cast<std::size_t>(rank)],
            bins[static_cast<std::size_t>(
                ranked_indices[static_cast<std::size_t>(rank)])]));
    }

    QJsonArray frames;
    const int reported_frame_count = std::min(std::max(inputFrameCount, 0), 16);
    for (int frame = 0; frame < reported_frame_count; ++frame)
    {
        const FrameCoverage &coverage = frame_coverage[static_cast<std::size_t>(frame)];
        const auto strongest = std::max_element(
            coverage.orientationCounts.begin(), coverage.orientationCounts.end());
        const int strongest_sector = coverage.participatingEdgeCount > 0
            ? static_cast<int>(std::distance(
                  coverage.orientationCounts.begin(), strongest))
            : -1;
        frames.append(QJsonObject{
            {QStringLiteral("frame_index"), frame},
            {QStringLiteral("frame_label"),
             frame < frameLabels.size() ? frameLabels[frame] : QString{}},
            {QStringLiteral("participating_edge_count"),
             static_cast<double>(coverage.participatingEdgeCount)},
            {QStringLiteral("exclusive_edge_count"),
             static_cast<double>(coverage.exclusiveEdgeCount)},
            {QStringLiteral("shared_edge_count"),
             static_cast<double>(coverage.sharedEdgeCount)},
            {QStringLiteral("strongest_orientation_sector"), strongest_sector}});
    }

    const std::uint64_t acquisition_edges =
        root_causes[reasonIndex(MeshBoundaryAttributionReason::NoObservation)] +
        root_causes[reasonIndex(MeshBoundaryAttributionReason::InsufficientSource)];
    const std::uint64_t rematch_edges =
        root_causes[reasonIndex(MeshBoundaryAttributionReason::DepthSpreadRejected)] +
        root_causes[reasonIndex(MeshBoundaryAttributionReason::SurfaceWeightRejected)] +
        root_causes[reasonIndex(MeshBoundaryAttributionReason::AbsoluteTsdfRejected)] +
        root_causes[reasonIndex(MeshBoundaryAttributionReason::SupportGateRejected)];
    const std::uint64_t backend_edges =
        root_causes[reasonIndex(
            MeshBoundaryAttributionReason::ExtractionOrPostprocess)];
    QString conclusion = QStringLiteral("existing_images_local_rematch");
    if (acquisition_edges > rematch_edges && acquisition_edges > backend_edges)
    {
        conclusion = QStringLiteral("additional_capture_required");
    }
    else if (backend_edges > acquisition_edges && backend_edges > rematch_edges)
    {
        conclusion = QStringLiteral("mesh_backend_diagnosis");
    }

    return {
        {QStringLiteral("schema"),
         QStringLiteral("plascan.mesh.acquisition_gap_report.v1")},
        {QStringLiteral("boundary_edge_count"),
         static_cast<double>(edges.size())},
        {QStringLiteral("records_complete"), true},
        {QStringLiteral("orientation_bin_count"), kOrientationBinCount},
        {QStringLiteral("unknown_orientation_edge_count"),
         static_cast<double>(unknown_bin.edgeCount)},
        {QStringLiteral("primary_reason_counts"),
         reasonCountsToJson(primary_reasons)},
        {QStringLiteral("root_cause_counts"), reasonCountsToJson(root_causes)},
        {QStringLiteral("orientation_bins"), orientation_bins},
        {QStringLiteral("unknown_orientation"),
         orientationBinToJson(-1, unknown_bin)},
        {QStringLiteral("highest_risk_sectors"), highest_risk_sectors},
        {QStringLiteral("input_frame_count"), inputFrameCount},
        {QStringLiteral("source_mask_bit_width"), 16},
        {QStringLiteral("source_mask_frame_mapping_complete"),
         inputFrameCount <= 16},
        {QStringLiteral("frames"), frames},
        {QStringLiteral("existing_image_rematch_edge_count"),
         static_cast<double>(rematch_edges)},
        {QStringLiteral("additional_capture_edge_count"),
         static_cast<double>(acquisition_edges)},
        {QStringLiteral("mesh_backend_edge_count"),
         static_cast<double>(backend_edges)},
        {QStringLiteral("recommended_next_action"), conclusion}};
}

} // namespace xjw::mesh

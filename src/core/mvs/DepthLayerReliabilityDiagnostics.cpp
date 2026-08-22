#include "DepthLayerReliability.h"

#include <QJsonArray>

namespace xjw::mvs
{
namespace
{

QString classId(DepthLayerReliabilityClass value)
{
    switch (value)
    {
    case DepthLayerReliabilityClass::Reliable:
        return QStringLiteral("reliable");
    case DepthLayerReliabilityClass::AmbiguousLowTexture:
        return QStringLiteral("ambiguous_low_texture");
    case DepthLayerReliabilityClass::RejectedLayer:
        return QStringLiteral("rejected_layer");
    case DepthLayerReliabilityClass::Unobservable:
    default:
        return QStringLiteral("unobservable");
    }
}

} // namespace

QJsonObject depthLayerReliabilityDiagnosticsToJson(
    const DepthLayerReliabilityResult &result,
    const DepthLayerReliabilityOptions &options)
{
    QJsonArray components;
    for (const DepthLayerReliabilityComponent &component : result.components)
    {
        components.append(QJsonObject{
            {QStringLiteral("class"), classId(component.reliabilityClass)},
            {QStringLiteral("x"), component.bounds.x},
            {QStringLiteral("y"), component.bounds.y},
            {QStringLiteral("width"), component.bounds.width},
            {QStringLiteral("height"), component.bounds.height},
            {QStringLiteral("pixel_count"), component.pixelCount},
            {QStringLiteral("boundary_anchor_count"),
             component.boundaryAnchorCount},
            {QStringLiteral("boundary_surface_model"),
             QStringLiteral("quadratic_inverse_depth")},
            {QStringLiteral("boundary_surface_fit_p90"),
             component.boundarySurfaceFitP90},
            {QStringLiteral("signed_relative_residual_median"),
             component.signedRelativeResidualMedian},
            {QStringLiteral("absolute_relative_residual_median"),
             component.absoluteRelativeResidualMedian}});
    }
    return QJsonObject{
        {QStringLiteral("schema"),
         QStringLiteral("plascan.mvs.depth_layer_reliability.v2")},
        {QStringLiteral("classification_read_only"), true},
        {QStringLiteral("valid_inputs"), result.validInputs},
        {QStringLiteral("error"), QString::fromStdString(result.errorMessage)},
        {QStringLiteral("valid_pixel_count"), result.validPixelCount},
        {QStringLiteral("low_texture_pixel_count"), result.lowTexturePixelCount},
        {QStringLiteral("weak_geometry_pixel_count"), result.weakGeometryPixelCount},
        {QStringLiteral("candidate_pixel_count"), result.candidatePixelCount},
        {QStringLiteral("reliable_pixel_count"), result.reliablePixelCount},
        {QStringLiteral("ambiguous_pixel_count"), result.ambiguousPixelCount},
        {QStringLiteral("rejected_layer_pixel_count"),
         result.rejectedLayerPixelCount},
        {QStringLiteral("rejected_layer_component_count"),
         result.rejectedLayerComponentCount},
        {QStringLiteral("options"),
         QJsonObject{
             {QStringLiteral("texture_radius_pixels"),
              options.textureRadiusPixels},
             {QStringLiteral("maximum_low_texture_standard_deviation"),
              options.maximumLowTextureStandardDeviation},
             {QStringLiteral("maximum_weak_effective_view_count"),
              options.maximumWeakEffectiveViewCount},
             {QStringLiteral("minimum_weak_conflict_ratio"),
              options.minimumWeakConflictRatio},
             {QStringLiteral("minimum_weak_inverse_depth_spread"),
              options.minimumWeakInverseDepthSpread},
             {QStringLiteral("minimum_boundary_effective_view_count"),
              options.minimumBoundaryEffectiveViewCount},
             {QStringLiteral("maximum_boundary_conflict_ratio"),
              options.maximumBoundaryConflictRatio},
             {QStringLiteral("maximum_boundary_inverse_depth_spread"),
              options.maximumBoundaryInverseDepthSpread},
             {QStringLiteral("minimum_component_area"),
              options.minimumComponentArea},
             {QStringLiteral("boundary_ring_radius_pixels"),
              options.boundaryRingRadiusPixels},
             {QStringLiteral("minimum_boundary_anchor_count"),
              options.minimumBoundaryAnchorCount},
             {QStringLiteral("minimum_rejected_layer_relative_residual"),
              options.minimumRejectedLayerRelativeResidual},
             {QStringLiteral("maximum_boundary_surface_fit_p90"),
              options.maximumBoundarySurfaceFitP90}}},
        {QStringLiteral("components"), components}};
}

} // namespace xjw::mvs

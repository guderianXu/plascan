#include <placoordinate/presets/ProjectionPresets.h>

int main()
{
    using namespace placoordinate;

    const GdalSpatialReferenceResult geographic =
        makeGeographicSpatialReference(SpatialReferenceId("moon-geographic"),
                                       CoordinateFrameId("moon-geodetic-frame"),
                                       ReferenceEllipsoid::moonMeanSphere(),
                                       VerticalReference::PlanetaryRadius);
    const GdalSpatialReferenceResult projected =
        makeLunarEquirectangularSpatialReference(SpatialReferenceId("moon-eq"), CoordinateFrameId("moon-eq-frame"));
    if (!geographic.ok() || !projected.ok())
    {
        return 1;
    }

    const GdalCoordinateTransformResult coordinate = transformGdalCoordinate(
        {10.0, 5.0, 0.0}, *geographic.reference, CoordinateAxisOrder::LongitudeLatitude, *projected.reference);
    return coordinate.ok ? 0 : 1;
}

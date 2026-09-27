#include <placoordinate/placoordinate.h>

#include <array>

int main()
{
    using namespace placoordinate;

    CoordinateTransformService transforms;
    transforms.registerFrame(CoordinateFrame::create(CoordinateFrameId("metres"),
                                                     CoordinateFrameKind::LocalCartesian,
                                                     LinearUnit::Metre,
                                                     AngleUnit::Radian,
                                                     std::nullopt,
                                                     RigidTransform::identity()));
    transforms.registerFrame(CoordinateFrame::create(CoordinateFrameId("kilometres"),
                                                     CoordinateFrameKind::LocalCartesian,
                                                     LinearUnit::Kilometre,
                                                     AngleUnit::Radian,
                                                     CoordinateFrameId("metres"),
                                                     RigidTransform::identity()));

    const std::array<double, 3> converted =
        transforms.transformPoint(CoordinateFrameId("kilometres"), CoordinateFrameId("metres"), {1.0, 2.0, 3.0});
    return converted == std::array<double, 3>{1000.0, 2000.0, 3000.0} ? 0 : 1;
}

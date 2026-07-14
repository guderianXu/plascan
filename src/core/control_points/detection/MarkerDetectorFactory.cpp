#include "MarkerDetectorFactory.h"

#include "AprilTagDetector.h"
#include "NonCodedTargetDetector.h"

#include <array>
#include <stdexcept>

namespace xjw::control_points
{

namespace
{

AprilTagFamily aprilTagFamily(MarkerTargetFamily family)
{
    switch (family)
    {
    case MarkerTargetFamily::AprilTag16h5: return AprilTagFamily::Tag16h5;
    case MarkerTargetFamily::AprilTag25h9: return AprilTagFamily::Tag25h9;
    case MarkerTargetFamily::AprilTag36h10: return AprilTagFamily::Tag36h10;
    case MarkerTargetFamily::AprilTag36h11: return AprilTagFamily::Tag36h11;
    case MarkerTargetFamily::AprilTagCircle21h7: return AprilTagFamily::Circle21h7;
    case MarkerTargetFamily::AprilTagStandard41h12: return AprilTagFamily::Standard41h12;
    case MarkerTargetFamily::AprilTagStandard52h13: return AprilTagFamily::Standard52h13;
    default: throw std::invalid_argument("Target family is not an AprilTag family");
    }
}

bool isAprilTag(MarkerTargetFamily family)
{
    return family >= MarkerTargetFamily::AprilTag16h5
        && family <= MarkerTargetFamily::AprilTagStandard52h13;
}

bool isCircularCoded(MarkerTargetFamily family)
{
    return family >= MarkerTargetFamily::Circular12Bit
        && family <= MarkerTargetFamily::Circular20Bit;
}

} // namespace

std::unique_ptr<MarkerDetector> MarkerDetectorFactory::create(MarkerTargetFamily family)
{
    if (isAprilTag(family))
    {
        return std::make_unique<AprilTagDetector>(aprilTagFamily(family));
    }
    if (family == MarkerTargetFamily::NonCodedCircle)
    {
        return std::make_unique<NonCodedTargetDetector>(NonCodedTargetType::Circle);
    }
    if (family == MarkerTargetFamily::NonCodedFourQuadrant)
    {
        return std::make_unique<NonCodedTargetDetector>(NonCodedTargetType::FourQuadrant);
    }
    if (isCircularCoded(family))
    {
        throw std::runtime_error(
            "Metashape circular coded target corpus has not been installed");
    }
    throw std::invalid_argument("Unsupported marker target family");
}

std::optional<MarkerTargetFamily> MarkerDetectorFactory::parseFamily(const QString &name)
{
    const QString normalized = name.trimmed().toLower();
    const std::array<MarkerTargetFamily, 13> families = {
        MarkerTargetFamily::AprilTag16h5,
        MarkerTargetFamily::AprilTag25h9,
        MarkerTargetFamily::AprilTag36h10,
        MarkerTargetFamily::AprilTag36h11,
        MarkerTargetFamily::AprilTagCircle21h7,
        MarkerTargetFamily::AprilTagStandard41h12,
        MarkerTargetFamily::AprilTagStandard52h13,
        MarkerTargetFamily::Circular12Bit,
        MarkerTargetFamily::Circular14Bit,
        MarkerTargetFamily::Circular16Bit,
        MarkerTargetFamily::Circular20Bit,
        MarkerTargetFamily::NonCodedCircle,
        MarkerTargetFamily::NonCodedFourQuadrant,
    };
    for (const MarkerTargetFamily family : families)
    {
        if (markerTargetFamilyName(family).toLower() == normalized)
        {
            return family;
        }
    }
    return std::nullopt;
}

} // namespace xjw::control_points

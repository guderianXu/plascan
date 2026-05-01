#include "data/PointCloudPoint.h"

namespace xjw::pointcloud
{

void PointCloudPoint::setTextureCoordinate(const Point2f &value)
{
    textureCoordinate = value;
    hasTextureCoordinate = true;
}

void PointCloudPoint::setNormal(const Point3f &value)
{
    normal = value;
    hasNormal = true;
}

void PointCloudPoint::setColor(const ColorRGBA &value)
{
    color = value;
    hasColor = true;
}

void PointCloudPoint::setPhotogrammetry(const PhotogrammetryPointAttributes &value)
{
    photogrammetry = value;
    hasPhotogrammetry = true;
}

} // namespace xjw::pointcloud

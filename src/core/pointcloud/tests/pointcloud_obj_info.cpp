#include "data/PointCloud.h"
#include "io/PointCloudIO.h"

#include <iomanip>
#include <iostream>
#include <string>

using namespace xjw::pointcloud;

namespace
{

void printPoint(const std::string &label, const Point3f &point)
{
    std::cout << label << ": ("
              << point.x << ", "
              << point.y << ", "
              << point.z << ")\n";
}

void printUsage(const char *programName)
{
    std::cerr << "Usage: " << programName << " <path/to/file.obj>\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::string inputPath = argv[1];

    PointCloud pointCloud;
    PointCloudIOResult ioResult;
    if (!readObjPointCloud(inputPath, &pointCloud, &ioResult))
    {
        std::cerr << "Failed to read OBJ point cloud: " << ioResult.errorMessage << "\n";
        return 2;
    }

    const PointCloudBounds bounds = pointCloud.computeBounds();
    const Point3f centroid = pointCloud.computeCentroid();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Point Cloud Summary\n";
    std::cout << "===================\n";
    std::cout << "Input Path: " << inputPath << "\n";
    std::cout << "Point Count: " << pointCloud.size() << "\n";
    std::cout << "Has Normals: " << (pointCloud.hasNormals() ? "true" : "false") << "\n";
    std::cout << "Has Colors: " << (pointCloud.hasColors() ? "true" : "false") << "\n";
    std::cout << "Has Photogrammetry Attributes: "
              << (pointCloud.hasPhotogrammetryAttributes() ? "true" : "false") << "\n";
    std::cout << "Consistent: " << (pointCloud.isConsistent() ? "true" : "false") << "\n";

    if (bounds.valid)
    {
        printPoint("Bounds Min", bounds.minCorner);
        printPoint("Bounds Max", bounds.maxCorner);
    }
    else
    {
        std::cout << "Bounds: invalid\n";
    }

    printPoint("Centroid", centroid);

    if (!pointCloud.empty())
    {
        printPoint("First Point", pointCloud.positions().front());
        if (const Point3f *firstNormal = pointCloud.normalAt(0))
        {
            printPoint("First Normal", *firstNormal);
        }
        if (const ColorRGBA *firstColor = pointCloud.colorAt(0))
        {
            std::cout << "First Color: ("
                      << static_cast<int>(firstColor->r) << ", "
                      << static_cast<int>(firstColor->g) << ", "
                      << static_cast<int>(firstColor->b) << ", "
                      << static_cast<int>(firstColor->a) << ")\n";
        }
    }

    const PointCloudMetadata &metadata = pointCloud.metadata();
    std::cout << "Metadata Name: " << metadata.name << "\n";
    std::cout << "Metadata Source Path: " << metadata.sourcePath << "\n";
    std::cout << "Metadata Registered: " << (metadata.isRegistered ? "true" : "false") << "\n";
    std::cout << "Metadata Frame: " << static_cast<int>(metadata.coordinateFrame) << "\n";

    return 0;
}
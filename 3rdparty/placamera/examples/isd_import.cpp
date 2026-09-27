#include <placamera/isd.h>

#include <filesystem>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: placamera_isd_import <camera.isd>\n";
        return 1;
    }

    const auto imported = placamera::importPlanetaryLineScanIsd(std::filesystem::path(argv[1]),
                                                                placamera::CameraDefinitionId("isd-definition"),
                                                                placamera::CameraInstanceId("isd-instance"),
                                                                placamera::ImageId("isd-image"));
    if (!imported)
    {
        std::cerr << imported.error().source << ": " << imported.message() << '\n';
        return 2;
    }

    std::cout << imported->metadata.platformName << " / " << imported->metadata.sensorName << " / "
              << imported->instance->imageSize().samples << 'x' << imported->instance->imageSize().lines << '\n';
    return 0;
}

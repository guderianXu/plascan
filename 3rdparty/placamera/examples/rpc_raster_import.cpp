#include <placamera/rpc_raster.h>

#include <filesystem>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: placamera_rpc_raster_import <raster>\n";
        return 1;
    }

    const auto model = placamera::importRpcRasterModel(std::filesystem::path(argv[1]),
                                                       placamera::CameraDefinitionId("rpc-definition"),
                                                       placamera::CameraInstanceId("rpc-instance"),
                                                       placamera::ImageId("rpc-image"),
                                                       placamera::FrameId("EPSG:4978"));
    if (!model)
    {
        std::cerr << model.error().source << ": " << model.message() << '\n';
        return 2;
    }

    std::cout << model.value()->imageSize().samples << 'x' << model.value()->imageSize().lines << '\n';
    return 0;
}

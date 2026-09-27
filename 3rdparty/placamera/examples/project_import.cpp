#include <placamera/formats.h>
#include <placamera/instance_set.h>

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: placamera_project_import <project> <fallback-width> <fallback-height>\n";
        return 1;
    }

    const int fallback_width = std::stoi(argv[2]);
    const int fallback_height = std::stoi(argv[3]);
    const auto project = placamera::importCameraProject(std::filesystem::path(argv[1]));
    if (!project)
    {
        std::cerr << project.error().source << ": " << project.message() << '\n';
        return 2;
    }

    placamera::CameraInstanceSet models;
    std::size_t skipped = 0;
    for (std::size_t index = 0; index < project->cameras.size(); ++index)
    {
        const auto& imported = project->cameras[index];
        if (!imported.compatibility.isExactlyRepresentable())
        {
            ++skipped;
            std::cerr << "skip " << imported.imageName << ": "
                      << imported.compatibility.unsupportedReason.value_or("source calibration has unsupported terms")
                      << '\n';
            continue;
        }

        const std::string suffix = std::to_string(index);
        auto geometry =
            placamera::makeCentralCameraGeometry(imported,
                                                 placamera::CameraDefinitionId("import-definition-" + suffix),
                                                 placamera::FrameId("project-world"));
        if (!geometry)
        {
            std::cerr << imported.imageName << ": " << geometry.message() << '\n';
            return 3;
        }

        placamera::ImageSize size{fallback_width, fallback_height};
        if (imported.sourceColmapCamera)
        {
            size = {imported.sourceColmapCamera->width, imported.sourceColmapCamera->height};
        }
        auto model = placamera::bindCentralCamera(geometry.takeValue(),
                                                  {placamera::CameraInstanceId("import-instance-" + suffix),
                                                   placamera::ImageId(imported.imageName),
                                                   size,
                                                   std::nullopt,
                                                   imported.acquisition});
        if (!model)
        {
            std::cerr << imported.imageName << ": " << model.message() << '\n';
            return 4;
        }
        const auto added = models.add(model.takeValue());
        if (!added)
        {
            std::cerr << imported.imageName << ": " << added.message() << '\n';
            return 5;
        }
    }

    std::cout << "imported " << models.size() << " frame cameras; skipped " << skipped << '\n';
    return 0;
}

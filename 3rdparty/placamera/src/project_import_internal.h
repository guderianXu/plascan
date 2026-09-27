#pragma once

#include "placamera/project_import.h"

namespace placamera::internal
{

    CameraProjectImportResult importMetashapeReferenceFiles(const std::filesystem::path& cameraPath,
                                                            const std::filesystem::path& gnssOffsetPath);

} // namespace placamera::internal

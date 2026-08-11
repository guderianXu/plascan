#pragma once

#include <QString>

#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include <plapoint/core/point_cloud.h>

using StreamingObjCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using ObjLoadProgressCallback = std::function<void(int, const QString &)>;

// Reads OBJ data without retaining a second, file-sized byte buffer. The file
// is scanned once for exact allocation sizes and parsed during a second pass.
std::shared_ptr<StreamingObjCloud> readObjStreaming(
    const std::string &path,
    const ObjLoadProgressCallback &progress = {},
    const std::atomic_bool *cancellationFlag = nullptr);

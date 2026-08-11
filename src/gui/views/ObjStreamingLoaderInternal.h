#pragma once

#include "ObjStreamingLoader.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xjw::gui::obj_streaming
{

struct ObjAssemblyInput
{
    std::vector<float> vx;
    std::vector<float> vy;
    std::vector<float> vz;
    std::vector<float> nx;
    std::vector<float> ny;
    std::vector<float> nz;
    std::vector<float> tx;
    std::vector<float> ty;
    std::vector<std::uint8_t> vr;
    std::vector<std::uint8_t> vg;
    std::vector<std::uint8_t> vb;
    std::vector<bool> hasVertexColor;
    std::vector<std::array<int, 3>> faceVertices;
    std::vector<std::array<int, 3>> faceTextures;
    std::string materialLibrary;
    bool faceTexturesComplete = false;
};

std::shared_ptr<StreamingObjCloud> assembleObjCloud(
    ObjAssemblyInput input,
    const ObjLoadProgressCallback &progress,
    const std::atomic_bool *cancellationFlag);

} // namespace xjw::gui::obj_streaming

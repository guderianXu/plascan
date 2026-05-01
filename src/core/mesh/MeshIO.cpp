#include "MeshTypes.h"

#include <fstream>
#include <filesystem>

namespace xjw 
{
namespace mesh 
{

bool TriMesh::savePLY(const std::string &path, std::string *errorMsg) const
{
    if (empty()) {
        if (errorMsg) *errorMsg = "mesh 为空";
        return false;
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        if (errorMsg) *errorMsg = "无法创建文件: " + path;
        return false;
    }

    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << vertices.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property float nx\n";
    ofs << "property float ny\n";
    ofs << "property float nz\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "element face " << faces.size() << "\n";
    ofs << "property list uchar int vertex_indices\n";
    ofs << "end_header\n";

#pragma pack(push, 1)
    struct VOut 
    {
        float x, y, z;
        float nx, ny, nz;
        uint8_t r, g, b;
    };
#pragma pack(pop)

    for (const auto &v : vertices) 
    {
        const VOut out{v.x, v.y, v.z, v.nx, v.ny, v.nz, v.r, v.g, v.b};
        ofs.write(reinterpret_cast<const char*>(&out), sizeof(VOut));
    }

    for (const auto &f : faces) {
        const uint8_t n = 3;
        ofs.write(reinterpret_cast<const char*>(&n), sizeof(uint8_t));
        ofs.write(reinterpret_cast<const char*>(&f.v[0]), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&f.v[1]), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&f.v[2]), sizeof(int));
    }

    if (!ofs.good()) {
        if (errorMsg) *errorMsg = "写入 PLY 失败: " + path;
        return false;
    }
    return true;
}

bool TriMesh::saveOBJ(const std::string &path, std::string *errorMsg) const
{
    if (empty()) {
        if (errorMsg) *errorMsg = "mesh 为空";
        return false;
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream ofs(path);
    if (!ofs) {
        if (errorMsg) *errorMsg = "无法创建文件: " + path;
        return false;
    }

    for (const auto &v : vertices) {
        ofs << "v " << v.x << ' ' << v.y << ' ' << v.z << "\n";
    }
    for (const auto &v : vertices) {
        ofs << "vn " << v.nx << ' ' << v.ny << ' ' << v.nz << "\n";
    }
    for (const auto &f : faces) {
        const int a = f.v[0] + 1;
        const int b = f.v[1] + 1;
        const int c = f.v[2] + 1;
        ofs << "f " << a << "//" << a << ' ' << b << "//" << b << ' ' << c << "//" << c << "\n";
    }

    if (!ofs.good()) {
        if (errorMsg) *errorMsg = "写入 OBJ 失败: " + path;
        return false;
    }
    return true;
}

} // namespace mesh
} // namespace xjw

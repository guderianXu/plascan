#include "MeshTypes.h"
#include "io/PathIO.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <fstream>
#include <filesystem>

namespace xjw 
{
namespace mesh 
{

bool TriMesh::loadPLY(const std::string &path, TriMesh *mesh, std::string *errorMsg)
{
    if (!mesh)
    {
        if (errorMsg)
        {
            *errorMsg = "mesh 输出为空";
        }
        return false;
    }

    mesh->vertices.clear();
    mesh->faces.clear();
    try
    {
        const auto cloud = plapoint::io::readPly<float>(
            xjw::common::io::toNativeNarrowPath(path));
        if (!cloud || cloud->size() == 0 || !cloud->hasFaces())
        {
            if (errorMsg)
            {
                *errorMsg = "PLY 不包含有效三角网格";
            }
            return false;
        }

        mesh->hasVertexColors = cloud->hasColors();

        mesh->vertices.resize(cloud->size());
        for (std::size_t index = 0; index < cloud->size(); ++index)
        {
            const auto row = static_cast<plamatrix::Index>(index);
            MeshVertex &vertex = mesh->vertices[index];
            vertex.x = cloud->points().getValue(row, 0);
            vertex.y = cloud->points().getValue(row, 1);
            vertex.z = cloud->points().getValue(row, 2);
            if (cloud->hasNormals())
            {
                vertex.nx = cloud->normals()->getValue(row, 0);
                vertex.ny = cloud->normals()->getValue(row, 1);
                vertex.nz = cloud->normals()->getValue(row, 2);
            }
            if (cloud->hasColors())
            {
                vertex.r = cloud->colors()->getValue(row, 0);
                vertex.g = cloud->colors()->getValue(row, 1);
                vertex.b = cloud->colors()->getValue(row, 2);
            }
        }

        const auto *faces = cloud->faces();
        if (!faces || faces->cols() != 3)
        {
            if (errorMsg)
            {
                *errorMsg = "PLY 面不是三角形";
            }
            mesh->vertices.clear();
            return false;
        }

        mesh->faces.resize(static_cast<std::size_t>(faces->rows()));
        for (plamatrix::Index row = 0; row < faces->rows(); ++row)
        {
            Triangle &triangle = mesh->faces[static_cast<std::size_t>(row)];
            for (int column = 0; column < 3; ++column)
            {
                const int vertex_index = faces->getValue(row, column);
                if (vertex_index < 0 ||
                    static_cast<std::size_t>(vertex_index) >= mesh->vertices.size())
                {
                    if (errorMsg)
                    {
                        *errorMsg = "PLY 面索引越界";
                    }
                    mesh->vertices.clear();
                    mesh->faces.clear();
                    return false;
                }
                triangle.v[column] = vertex_index;
            }
        }
        return !mesh->empty();
    }
    catch (const std::exception &exception)
    {
        mesh->vertices.clear();
        mesh->faces.clear();
        if (errorMsg)
        {
            *errorMsg = exception.what();
        }
        return false;
    }
}

bool TriMesh::savePLY(const std::string &path, std::string *errorMsg) const
{
    if (empty()) {
        if (errorMsg) *errorMsg = "mesh 为空";
        return false;
    }

    std::filesystem::create_directories(
        xjw::common::io::toFilesystemPath(xjw::common::io::fromUtf8Path(path)).parent_path());
    try
    {
        using PlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(vertices.size(), 3);
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(vertices.size(), 3);
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(vertices.size(), 3);
        for (std::size_t i = 0; i < vertices.size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            const MeshVertex &vertex = vertices[i];
            points(row, 0) = vertex.x;
            points(row, 1) = vertex.y;
            points(row, 2) = vertex.z;
            normals(row, 0) = vertex.nx;
            normals(row, 1) = vertex.ny;
            normals(row, 2) = vertex.nz;
            colors(row, 0) = vertex.r;
            colors(row, 1) = vertex.g;
            colors(row, 2) = vertex.b;
        }

        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faceMatrix(faces.size(), 3);
        for (std::size_t i = 0; i < faces.size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            faceMatrix(row, 0) = faces[i].v[0];
            faceMatrix(row, 1) = faces[i].v[1];
            faceMatrix(row, 2) = faces[i].v[2];
        }

        PlaCloud cloud(std::move(points));
        cloud.setNormals(std::move(normals));
        cloud.setColors(std::move(colors));
        cloud.setFaces(std::move(faceMatrix));
        plapoint::io::writePly(
            xjw::common::io::toNativeNarrowPath(path), cloud, plapoint::io::PlyFormat::BinaryLE);
        return true;
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = e.what();
        return false;
    }
}

bool TriMesh::saveOBJ(const std::string &path, std::string *errorMsg) const
{
    if (empty()) {
        if (errorMsg) *errorMsg = "mesh 为空";
        return false;
    }

    std::filesystem::create_directories(
        xjw::common::io::toFilesystemPath(xjw::common::io::fromUtf8Path(path)).parent_path());
    std::ofstream ofs = xjw::common::io::openOutputFile(path, std::ios::out | std::ios::trunc);
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

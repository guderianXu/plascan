#include "TextureMappingV4Internal.h"

#include "io/PathIO.h"

#include <plapoint/io/obj_io.h>
#include <plapoint/io/ply_io.h>

#include <opencv2/imgproc.hpp>

#include <QFileInfo>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace xjw::mesh::texture_v4
{
namespace
{

std::shared_ptr<PlaPointCloud> loadMesh(const std::string &mesh_path)
{
    const QString suffix =
        QFileInfo(xjw::common::io::fromUtf8Path(mesh_path)).suffix().toLower();
    if (suffix == QStringLiteral("ply"))
    {
        return plapoint::io::readPly<float>(
            xjw::common::io::toNativeNarrowPath(mesh_path));
    }
    return plapoint::io::readObj<float>(
        xjw::common::io::toNativeNarrowPath(mesh_path));
}

std::array<float, 3> faceNormal(const std::array<double, 3> &first,
                                const std::array<double, 3> &second,
                                const std::array<double, 3> &third)
{
    const double ux = second[0] - first[0];
    const double uy = second[1] - first[1];
    const double uz = second[2] - first[2];
    const double vx = third[0] - first[0];
    const double vy = third[1] - first[1];
    const double vz = third[2] - first[2];
    std::array<float, 3> normal{
        static_cast<float>(uy * vz - uz * vy),
        static_cast<float>(uz * vx - ux * vz),
        static_cast<float>(ux * vy - uy * vx)};
    const float length = std::sqrt(
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (length > 1.0e-12f)
    {
        for (float &value : normal)
        {
            value /= length;
        }
    }
    return normal;
}

double distance(const std::array<double, 3> &left,
                const std::array<double, 3> &right)
{
    const double dx = left[0] - right[0];
    const double dy = left[1] - right[1];
    const double dz = left[2] - right[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::uint64_t edgeKey(int first, int second)
{
    const std::uint32_t low = static_cast<std::uint32_t>(std::min(first, second));
    const std::uint32_t high = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

float imageSharpness(const cv::Mat &gray)
{
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_32FC1, 3);
    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(laplacian, mean, deviation);
    return static_cast<float>(deviation[0] * deviation[0]);
}

bool cancelled(const TextureMappingConfig &config)
{
    return config.isCancelled && config.isCancelled();
}

bool hasNonzeroDistortion(const FramePinholeCamera &camera)
{
    const FramePinholeCamera::Distortion distortion = camera.distortion();
    return std::fabs(distortion.radialK1) > 1.0e-15 ||
        std::fabs(distortion.radialK2) > 1.0e-15 ||
        std::fabs(distortion.radialK3) > 1.0e-15 ||
        std::fabs(distortion.tangentialP1) > 1.0e-15 ||
        std::fabs(distortion.tangentialP2) > 1.0e-15;
}

} // namespace

bool prepareInputs(const std::string &meshPath,
                   const QVector<MeshColorView> &views,
                   const TextureMappingConfig &config,
                   PipelineData *data,
                   TextureMappingResult *result,
                   std::string *errorMsg)
{
    if (!data || !result)
    {
        if (errorMsg)
        {
            *errorMsg = "纹理 v4 输出参数为空";
        }
        return false;
    }
    if (config.mappingMode != TextureMappingMode::AutoProjective)
    {
        if (errorMsg)
        {
            *errorMsg = "纹理 v4 当前仅支持自动相机投影 UV";
        }
        return false;
    }
    if (cancelled(config))
    {
        result->cancelled = true;
        if (errorMsg)
        {
            *errorMsg = "纹理映射已取消";
        }
        return false;
    }
    if (config.progressFn)
    {
        config.progressFn("正在检查纹理输入...", 2);
    }

    data->mesh = loadMesh(meshPath);
    if (!data->mesh || !data->mesh->hasFaces() || data->mesh->size() == 0)
    {
        if (errorMsg)
        {
            *errorMsg = "纹理 v4 无法读取有效网格面";
        }
        return false;
    }

    data->views.clear();
    QVector<float> sharpness_values;
    for (int index = 0; index < views.size(); ++index)
    {
        const MeshColorView &source = views[index];
        if (!source.camera.isValid() || source.colorBgr.type() != CV_8UC3 ||
            source.depth.type() != CV_32FC1 ||
            source.confidence.type() != CV_32FC1 ||
            source.depthValidMask.type() != CV_8UC1 ||
            source.supportMask.type() != CV_8UC1 ||
            source.depth.size() != source.confidence.size() ||
            source.depth.size() != source.depthValidMask.size() ||
            source.depth.size() != source.supportMask.size())
        {
            continue;
        }

        const FramePinholeCamera &source_color_camera =
            source.colorCamera.isValid() ? source.colorCamera : source.camera;
        if (hasNonzeroDistortion(source.camera) ||
            hasNonzeroDistortion(source_color_camera))
        {
            if (errorMsg)
            {
                *errorMsg =
                    "纹理 v4 需要预去畸变的零畸变深度与彩色相机；"
                    "请重新生成 revision 39 或更新版本的 MVS workspace";
            }
            return false;
        }

        PreparedView prepared;
        prepared.sourceIndex = index;
        prepared.evidenceCamera = source.camera;
        prepared.colorCamera = source.colorCamera.isValid()
            ? source.colorCamera
            : source.camera.scaledIntrinsics(
                  static_cast<double>(source.colorBgr.cols) / source.depth.cols,
                  static_cast<double>(source.colorBgr.rows) / source.depth.rows);
        const int downscale = std::clamp(config.imageDownscale, 1, 8);
        if (downscale > 1)
        {
            const cv::Size work_size(
                std::max(1, source.colorBgr.cols / downscale),
                std::max(1, source.colorBgr.rows / downscale));
            cv::resize(source.colorBgr,
                       prepared.colorBgr,
                       work_size,
                       0.0,
                       0.0,
                       cv::INTER_AREA);
            prepared.colorCamera = prepared.colorCamera.scaledIntrinsics(
                static_cast<double>(work_size.width) / source.colorBgr.cols,
                static_cast<double>(work_size.height) / source.colorBgr.rows);
        }
        else
        {
            prepared.colorBgr = source.colorBgr;
        }
        cv::cvtColor(prepared.colorBgr, prepared.gray, cv::COLOR_BGR2GRAY);
        cv::Mat color_support;
        cv::resize(source.supportMask,
                   color_support,
                   prepared.colorBgr.size(),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);
        cv::distanceTransform(color_support,
                              prepared.supportDistance,
                              cv::DIST_L2,
                              cv::DIST_MASK_PRECISE);
        prepared.qualityWeight = std::max(0.01f, source.qualityWeight);
        prepared.depth = &source.depth;
        prepared.confidence = &source.confidence;
        prepared.depthValidMask = &source.depthValidMask;
        prepared.supportMask = &source.supportMask;
        const float sharpness = imageSharpness(prepared.gray);
        sharpness_values.push_back(sharpness);
        prepared.sharpnessWeight = sharpness;
        data->views.push_back(std::move(prepared));
    }
    if (data->views.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = "纹理 v4 没有尺寸和类型均有效的相机影像证据";
        }
        return false;
    }

    QVector<float> sorted_sharpness = sharpness_values;
    std::sort(sorted_sharpness.begin(), sorted_sharpness.end());
    const float median_sharpness =
        sorted_sharpness[sorted_sharpness.size() / 2];
    for (PreparedView &view : data->views)
    {
        view.sharpnessWeight = std::clamp(
            view.sharpnessWeight / std::max(median_sharpness, 1.0f),
            0.20f,
            2.0f);
    }

    auto *faces = data->mesh->faces();
    data->geometry.resize(static_cast<int>(faces->rows()));
    std::vector<double> edge_lengths;
    edge_lengths.reserve(static_cast<std::size_t>(faces->rows()) * 3);
    std::unordered_map<std::uint64_t, int> first_face_by_edge;
    for (int face_index = 0; face_index < faces->rows(); ++face_index)
    {
        FaceGeometry &face = data->geometry[face_index];
        for (int corner = 0; corner < 3; ++corner)
        {
            const int vertex_index = faces->getValue(face_index, corner);
            if (vertex_index < 0 ||
                static_cast<std::size_t>(vertex_index) >= data->mesh->size())
            {
                if (errorMsg)
                {
                    *errorMsg = "纹理 v4 网格包含越界面索引";
                }
                return false;
            }
            face.vertexIndices[corner] = vertex_index;
            face.vertices[corner] = {
                data->mesh->points().getValue(vertex_index, 0),
                data->mesh->points().getValue(vertex_index, 1),
                data->mesh->points().getValue(vertex_index, 2)};
        }
        for (int axis = 0; axis < 3; ++axis)
        {
            face.centroid[axis] =
                (face.vertices[0][axis] + face.vertices[1][axis] +
                 face.vertices[2][axis]) / 3.0;
        }
        face.normal = faceNormal(face.vertices[0], face.vertices[1], face.vertices[2]);
        const double first_edge = distance(face.vertices[0], face.vertices[1]);
        const double second_edge = distance(face.vertices[1], face.vertices[2]);
        const double third_edge = distance(face.vertices[2], face.vertices[0]);
        face.meanEdgeLength = (first_edge + second_edge + third_edge) / 3.0;
        const double cross_length = std::sqrt(
            std::pow(face.normal[0], 2.0) +
            std::pow(face.normal[1], 2.0) +
            std::pow(face.normal[2], 2.0));
        face.area = cross_length > 0.0
            ? 0.25 * std::sqrt(
                  std::max(0.0,
                           (first_edge + second_edge + third_edge) *
                           (-first_edge + second_edge + third_edge) *
                           (first_edge - second_edge + third_edge) *
                           (first_edge + second_edge - third_edge)))
            : 0.0;
        edge_lengths.push_back(first_edge);
        edge_lengths.push_back(second_edge);
        edge_lengths.push_back(third_edge);

        for (int edge = 0; edge < 3; ++edge)
        {
            const std::uint64_t key = edgeKey(
                face.vertexIndices[edge], face.vertexIndices[(edge + 1) % 3]);
            const auto existing = first_face_by_edge.find(key);
            if (existing == first_face_by_edge.end())
            {
                first_face_by_edge.emplace(key, face_index);
            }
            else
            {
                face.neighbors.push_back(existing->second);
                data->geometry[existing->second].neighbors.push_back(face_index);
            }
        }
    }
    std::nth_element(edge_lengths.begin(),
                     edge_lengths.begin() + edge_lengths.size() / 2,
                     edge_lengths.end());
    data->medianEdgeLength = edge_lengths[edge_lengths.size() / 2];
    data->assignments.resize(data->geometry.size());
    result->sourceViewCount = data->views.size();
    result->peakMemoryEstimateMiB =
        static_cast<double>(config.textureSize) * config.textureSize * 20.0 /
        (1024.0 * 1024.0);
    if (result->peakMemoryEstimateMiB > 3072.0)
    {
        if (errorMsg)
        {
            *errorMsg = "纹理大小需要超过 2 GiB 的估算工作内存，请降低纹理大小";
        }
        return false;
    }
    return true;
}

} // namespace xjw::mesh::texture_v4

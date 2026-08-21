#pragma once

#include "MeshColorizer.h"
#include "TextureMapper.h"

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>

#include <QPointF>
#include <QRect>
#include <QVector>

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace xjw::mesh::texture_v4
{

using PlaPointCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

// The left atlas strip stores the constant emergency sample and compact
// per-face vertex-colour tiles for geometry with no valid camera observation.
inline constexpr int kFallbackAtlasWidth = 130;
inline constexpr int kFallbackTileSize = 4;

struct PreparedView
{
    int sourceIndex = -1;
    FramePinholeCamera evidenceCamera;
    FramePinholeCamera colorCamera;
    cv::Mat colorBgr;
    cv::Mat gray;
    cv::Mat supportDistance;
    cv::Mat finalMeshFaceIds;
    std::vector<std::uint8_t> finalMeshVisibleFaces;
    float qualityWeight = 1.0f;
    float sharpnessWeight = 1.0f;
    float exposureGain = 1.0f;
    const cv::Mat *depth = nullptr;
    const cv::Mat *confidence = nullptr;
    const cv::Mat *depthValidMask = nullptr;
    const cv::Mat *supportMask = nullptr;
};

struct FaceGeometry
{
    std::array<int, 3> vertexIndices{{-1, -1, -1}};
    std::array<std::array<double, 3>, 3> vertices{};
    std::array<double, 3> centroid{};
    std::array<float, 3> normal{};
    double area = 0.0;
    double meanEdgeLength = 0.0;
    QVector<int> neighbors;
};

struct FaceCandidate
{
    int viewIndex = -1;
    float score = -1.0f;
    float angleScore = 0.0f;
    float resolutionScore = 0.0f;
    bool strict = false;
    bool finalMeshVisibilityRequired = false;
};

struct FaceAssignment
{
    QVector<FaceCandidate> candidates;
    int primaryView = -1;
    float primaryScore = -1.0f;
    bool relaxed = false;
    bool optimized = false;
};

bool passesUnaryQualityFloor(const FaceAssignment &assignment,
                             const FaceCandidate &candidate,
                             float replacement_ratio);

struct TextureChart
{
    int index = -1;
    int primaryView = -1;
    QVector<int> faces;
    QRect sourceBounds;
    QRect atlasBounds;
    QRect atlasContentBounds;
    float atlasScale = 1.0f;
};

struct PipelineData
{
    std::shared_ptr<PlaPointCloud> mesh;
    QVector<PreparedView> views;
    QVector<FaceGeometry> geometry;
    QVector<FaceAssignment> assignments;
    QVector<TextureChart> charts;
    double medianEdgeLength = 0.0;
    double atlasOccupancy = 0.0;
};

bool prepareInputs(const std::string &meshPath,
                   const QVector<MeshColorView> &views,
                   const TextureMappingConfig &config,
                   PipelineData *data,
                   TextureMappingResult *result,
                   std::string *errorMsg);

bool selectTextureViews(const TextureMappingConfig &config,
                        PipelineData *data,
                        TextureMappingResult *result,
                        std::string *errorMsg);

bool buildFinalMeshVisibility(const TextureMappingConfig &config,
                              PipelineData *data,
                              TextureMappingResult *result,
                              std::string *errorMsg);

bool isFinalMeshFaceVisible(const PreparedView &view,
                            int face_index,
                            const std::array<double, 3> &world);

bool isFinalMeshFaceVisibleSomewhere(const PreparedView &view,
                                     int face_index);

bool perspectiveCorrectBarycentricWeights(
    const std::array<double, 3> &affine_weights,
    const std::array<double, 3> &camera_depths,
    std::array<double, 3> *corrected_weights);

bool buildAndPackCharts(const TextureMappingConfig &config,
                        PipelineData *data,
                        TextureMappingResult *result,
                        std::string *errorMsg);

bool bakeAndExport(const std::string &productsDir,
                   const TextureMappingConfig &config,
                   PipelineData *data,
                   TextureMappingResult *result,
                   std::string *errorMsg);

} // namespace xjw::mesh::texture_v4

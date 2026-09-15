#pragma once
// Private workflow stage interfaces; no GUI or project session ownership.
#include "../ModelWorkflowService.h"
#include "RecoveredModelBuilder.h"
#include "RpcPlaneSweepModelBuilder.h"

#include "DepthMapMeshBuilder.h"
#include "DepthConstrainedSurfaceRefiner.h"
#include "DepthFrameQualificationPolicy.h"
#include "MvsWorkspaceManifest.h"
#include "DepthMeshCompleteness.h"
#include "DepthTsdfFinalQualityGate.h"
#include "DepthTsdfSurfaceBuilder.h"
#include "Mc33IsoSurfaceExtractor.h"
#include "MeshColorizer.h"
#include "MeshIsotropicRemesher.h"
#include "MeshQuadricSimplifier.h"
#include "MeshTopologyQuality.h"
#include "NativeMeshSimplifier.h"
#include "OrbitalSparseScaffoldSurfaceBuilder.h"
#include "SurfaceReconstructor.h"
#include "SurfaceReconstructorPostprocess.h"
#include "VisualHullDepthRefiner.h"
#include "VisibilityOccupancyCarrierFairer.h"
#include "VisibilityOccupancyCarrierFieldProjector.h"
#include "VisibilityOccupancyCarrierSubdivider.h"
#include "io/PathIO.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QScopeGuard>
#include <QTextStream>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <vector>

namespace xjw::mesh::workflow::workflow_detail
{
    // Internal conversion to algorithm callbacks, not a second public request API.
    template <class Request> struct BoundWorkflowRequest : Request
    {
        std::function<bool()> isCancelled;
        std::function<void(const QString&, int)> progress;
    };

    struct ModelGenerationContract
    {
        QString surfaceProfile;
        int targetFaces = 0;
        QJsonObject requested;
        QJsonObject effective;
    };

    struct RefinementQualityGuardResult
    {
        bool applied = false;
        bool limited = false;
        float acceptedBlend = 1.0f;
        double areaBefore = 0.0;
        double areaAfter = 0.0;
        double normalVariationBefore = 0.0;
        double normalVariationAfter = 0.0;
    };

    struct FinalSurfaceDenoisingResult
    {
        bool attempted = false;
        bool accepted = false;
        bool carrierAreaRelaxationEnabled = false;
        bool carrierAreaRelaxationApplied = false;
        int movedVertexCount = 0;
        double areaBefore = 0.0;
        double areaAfter = 0.0;
        double areaRatio = 0.0;
        double absoluteVolumeBefore = 0.0;
        double absoluteVolumeAfter = 0.0;
        double absoluteVolumeRatio = 0.0;
        double effectiveMinimumAreaRatio = 0.96;
        MeshTopologyQualityStatistics qualityBefore;
        MeshTopologyQualityStatistics qualityAfter;
    };
    bool resolveModelGenerationContract(const QJsonObject& settings, ModelGenerationContract* contract, QString* error);
    QString sha256ForFile(const QString& path, QString* error_message);
    bool validateUsableDepthSceneProfileBatch(const QVector<DepthFrameArtifact>& artifacts,
                                              QString* canonical_scene_profile,
                                              int* invalid_ref_index,
                                              QString* invalid_scene_profile);
    bool cancellationRequested(const std::function<bool()>& isCancelled);
    WorkflowResult cancelledWorkflowResult();
    bool isNonEmptyFile(const QString& path);
    bool validateModelRunArtifacts(const QJsonObject& payload, QString* errorMessage);
    bool writeRunDiagnostics(WorkflowResult* result, const QString& diagnosticsPath, const QString& diagnosticsType);
    bool finalizeModelRun(WorkflowResult* result,
                          ModelOutputPolicy policy,
                          const QString& runId,
                          const QString& runOutputRoot);
    bool finalizeTextureRun(WorkflowResult* result, const QString& runId, const QString& runOutputRoot);
    bool hasSameFaceIndexBuffer(const TriMesh& first, const TriMesh& second);
    QJsonArray eulerCharacteristicsToJson(const std::vector<int>& values);
    double meshSurfaceArea(const TriMesh& mesh);
    double meshAbsoluteOrientedVolume(const TriMesh& mesh);
    double meshMeanNormalVariation(const TriMesh& mesh);
    RefinementQualityGuardResult limitRefinementRoughness(const TriMesh& original,
                                                          TriMesh* refined,
                                                          double maximumAreaGrowth,
                                                          double maximumNormalVariationGrowth);
    QJsonObject textureResultToJson(const xjw::mesh::TextureMappingResult& result,
                                    const xjw::mesh::TextureMappingConfig* config = nullptr);
    MeshColorView textureViewFromFrame(const DepthTsdfFrame& frame);
    MeshColorView vertexColorViewFromFrame(const DepthTsdfFrame& frame);
    void addFinalMeshColorStatistics(const MeshColorStatistics& statistics, QJsonObject* payload);
    void assignFinalModelFields(QJsonObject* result, bool exportObj);
    WorkflowResult saveMeshAndOptionalTexture(const xjw::mesh::TriMesh& mesh,
                                              const std::string& mesh_algorithm,
                                              const QString& output_root,
                                              bool export_obj,
                                              const xjw::mesh::TextureMappingConfig& texture,
                                              const std::function<void(const QString&, int)>& progress,
                                              const std::function<bool()>& isCancelled,
                                              const QVector<MeshColorView>* camera_views = nullptr,
                                              const RecoveredModelResult* recovered_model = nullptr);
    bool interpolationIsDisabled(const QJsonObject& settings);
    void enforceNoDepthInterpolationPolicy(const QJsonObject& settings,
                                           xjw::mesh::DepthTsdfOptions* options,
                                           bool preserveVisibilityTopologyCarrier = false);
    xjw::mesh::DepthTsdfOptions makeDepthTsdfOptions(const QJsonObject& settings, int requested_resolution);
    float visibilityOccupancyMedianVoxelStep(const xjw::mesh::DepthTsdfLayout& layout, int occupancy_resolution);
    xjw::mesh::DepthConstrainedSurfaceRefineOptions
    makeVisibilityOccupancyDepthRefineOptions(const QJsonObject& settings,
                                              const xjw::mesh::DepthTsdfLayout& layout,
                                              int occupancy_resolution,
                                              bool orbital_workspace);
    void mergePayload(const QJsonObject& source, QJsonObject* target);
    int maximumReliableOrbitalResolution(const QVector<DepthTsdfFrame>& frames, int requestedResolution);
    WorkflowResult buildRecoveredDepthModel(const BoundWorkflowRequest<DepthMapMeshBuildRequest>&,
                                            WorkflowResult,
                                            const QString&,
                                            const QString&);

} // namespace xjw::mesh::workflow::workflow_detail
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;
    FinalSurfaceDenoisingResult applyTopologyGuardedFinalSurfaceDenoising(TriMesh* mesh,
                                                                          int iterations,
                                                                          float lambda,
                                                                          float mu,
                                                                          float maximumDisplacement,
                                                                          float featureAngleDegrees,
                                                                          int boundaryProtectionRings,
                                                                          bool allowCarrierAreaRelaxation);
    QString orbitalRoleForDepthFrame(const QJsonArray& roles, int refIndex);
    QStringList worstDepthCompletenessLabels(const DepthMeshCompletenessStatistics& completeness,
                                             const QJsonArray& orbitalRoles,
                                             int maximumCount = 3);
    void addDepthCompletenessPayload(const DepthMeshCompletenessStatistics& completeness,
                                     const QString& prefix,
                                     QJsonObject* payload);
    DepthMeshCompletenessStatistics evaluateDepthCompleteness(const TriMesh& mesh,
                                                              const QVector<DepthTsdfFrame>& frames,
                                                              const DepthTsdfOptions& options,
                                                              const DepthTsdfLayout& layout,
                                                              const QVector<int>& excludedRefIndices);
} // namespace xjw::mesh::workflow

namespace xjw::mesh::workflow::workflow_detail
{
    template <class Request> BoundWorkflowRequest<Request> bindWorkflowCallbacks(const Request& request)
    {
        BoundWorkflowRequest<Request> bound;
        static_cast<Request&>(bound) = request;
        const auto control = request.execution;
        bound.isCancelled = [control]() { return control.isCancelled(); };
        if (control.progress)
        {
            bound.progress = [control](const QString& stage, int percent)
            { control.reportProgress(stage.toUtf8().toStdString(), percent / 100.0); };
        }
        return bound;
    }
} // namespace xjw::mesh::workflow::workflow_detail

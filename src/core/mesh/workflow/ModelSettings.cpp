#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    int meshResolutionFromSettings(const QJsonObject& settings)
    {
        const double requestedResolution = settings.value(QStringLiteral("meshResolution")).toDouble(0.0);
        if (requestedResolution > 0.0)
        {
            return qBound(64, static_cast<int>(std::lround(requestedResolution)), 1024);
        }

        const int octreeDepth = settings.value(QStringLiteral("octreeDepth")).toInt(10);
        const int depth = qBound(4, octreeDepth, 14);
        return qBound(64, 1 << (depth - 2), 1024);
    }

    QString depthReconstructionModeFromSettings(const QJsonObject& settings)
    {
        const QString requested = settings.value(QStringLiteral("reconstruction_mode")).toString().trimmed().toLower();
        if (!requested.isEmpty())
        {
            return requested;
        }

        return settings.value(QStringLiteral("surface_type")).toString() == QStringLiteral("height_field")
                   ? QStringLiteral("poisson_legacy")
                   : QStringLiteral("depth_tsdf");
    }

    xjw::mesh::ReconstructionConfig reconstructionConfigFromModelSettings(const QJsonObject& settings)
    {
        xjw::mesh::ReconstructionConfig config;

        QString compute_mode = settings.value(QStringLiteral("compute_mode")).toString().trimmed().toLower();
        if (compute_mode.isEmpty())
        {
            compute_mode = settings.value(QStringLiteral("processingDevice")).toString().trimmed().toLower();
        }
        if (compute_mode == QStringLiteral("cuda"))
        {
            config.preprocessingDevice = plapoint::ProcessingDevice::CUDA;
            config.poissonSolverDevice = plapoint::ProcessingDevice::CUDA;
        }
        else if (compute_mode == QStringLiteral("opencl"))
        {
            config.preprocessingDevice = plapoint::ProcessingDevice::OpenCL;
            config.poissonSolverDevice = plapoint::ProcessingDevice::OpenCL;
        }
        else if (compute_mode == QStringLiteral("hybrid"))
        {
            // Split independent model stages so both physical GPUs stay useful:
            // OpenCL handles point preprocessing/height grids while CUDA handles
            // the Poisson solve. Automatic depth generation independently uses
            // PatchMatch Auto, which schedules CUDA and OpenCL frame workers.
            config.preprocessingDevice = plapoint::ProcessingDevice::OpenCL;
            config.poissonSolverDevice = plapoint::ProcessingDevice::CUDA;
        }

        const QString surfaceType =
            settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("arbitrary_3d"));
        config.resolution = meshResolutionFromSettings(settings);
        config.smoothIterations = qBound(0, settings.value(QStringLiteral("smoothIter")).toInt(3), 50);
        config.smoothLambda = 0.5f;
        config.padding = 0.05f;
        config.forcePoisson = surfaceType != QStringLiteral("height_field") &&
                              settings.value(QStringLiteral("method"))
                                  .toString(QStringLiteral("Poisson Surface"))
                                  .contains(QStringLiteral("Poisson"), Qt::CaseInsensitive);
        config.allowHeightGridFallback = surfaceType == QStringLiteral("height_field");
        config.orientNormalsForClosedSurface = surfaceType == QStringLiteral("arbitrary_3d");
        config.poissonDepth = qBound(7, settings.value(QStringLiteral("octreeDepth")).toInt(10), 12);
        config.poissonSolverIterations =
            qBound(1, settings.value(QStringLiteral("poissonSolverIterations")).toInt(200), 2000);
        config.poissonSolverTolerance =
            std::clamp(settings.value(QStringLiteral("poissonSolverTolerance")).toDouble(config.poissonSolverTolerance),
                       1.0e-8,
                       1.0);
        config.poissonThreads = qBound(1, settings.value(QStringLiteral("threads")).toInt(8), 128);

        const double pointWeight =
            settings.value(QStringLiteral("poissonPointWeight")).toDouble(config.poissonPointWeight);
        config.poissonPointWeight = std::clamp(static_cast<float>(pointWeight), 0.0f, 8.0f);

        const double poissonTrim = settings.value(QStringLiteral("poissonTrim")).toDouble(config.poissonTrim);
        config.poissonTrim = std::clamp(static_cast<float>(poissonTrim), 0.0f, 12.0f);

        const QString interpolation =
            settings.value(QStringLiteral("interpolation")).toString(QStringLiteral("enabled"));
        config.fillHoles =
            interpolation != QStringLiteral("disabled") && settings.value(QStringLiteral("holeFill")).toBool(true);
        if (interpolation == QStringLiteral("extrapolated"))
        {
            config.holeFillPasses =
                std::max(16, holeFillPassesFromArea(settings.value(QStringLiteral("maxHoleSize")).toDouble(400.0)));
        }
        else
        {
            config.holeFillPasses =
                holeFillPassesFromArea(settings.value(QStringLiteral("maxHoleSize")).toDouble(100.0));
        }

        config.cleanSmallComponents = settings.value(QStringLiteral("cleanSmall")).toBool(true);
        config.minComponentFaces = qBound(2, settings.value(QStringLiteral("minFaces")).toInt(100), 100000);

        const QString qualityProfile = settings.contains(QStringLiteral("qualityProfile"))
                                           ? settings.value(QStringLiteral("qualityProfile")).toString()
                                           : QStringLiteral("balanced");
        const QString quality = settings.value(QStringLiteral("quality")).toString();
        const QString voxelDensity = settings.value(QStringLiteral("voxelDensity")).toString(QStringLiteral("medium"));

        if (qualityProfile == QStringLiteral("detail"))
        {
            config.resolution = std::max(config.resolution, quality == QStringLiteral("ultra") ? 384 : 320);
            config.poissonDepth = std::max(config.poissonDepth, quality == QStringLiteral("ultra") ? 11 : 10);
            config.poissonPointWeight = std::max(config.poissonPointWeight, 4.8f);
            config.poissonTrim = std::max(config.poissonTrim, 9.0f);
            config.simplifyTargetFaces = std::max(config.simplifyTargetFaces, 65000);
            config.enableDownsample = false;
            config.voxelSimplifyFactor = 1.15f;
            config.kNormals = std::max(config.kNormals, 18);
            config.smoothIterations = std::max(config.smoothIterations, 4);
            config.smoothLambda = 0.36f;
        }
        else if (qualityProfile == QStringLiteral("lite"))
        {
            config.resolution = std::min(config.resolution, 224);
            config.poissonDepth = std::min(config.poissonDepth, 9);
            config.poissonPointWeight = std::min(config.poissonPointWeight, 3.2f);
            config.poissonTrim = std::min(config.poissonTrim, 8.4f);
            config.simplifyTargetFaces = 16000;
            config.enableDownsample = true;
            config.downsampleVoxelScale = 1.0f;
            config.voxelSimplifyFactor = 2.5f;
            config.smoothIterations = std::min(3, config.smoothIterations + 1);
            config.smoothLambda = 0.55f;
        }
        else if (voxelDensity == QStringLiteral("coarse"))
        {
            config.voxelSimplifyFactor = 2.6f;
            config.enableDownsample = true;
            config.downsampleVoxelScale = 1.0f;
            config.poissonPointWeight = std::min(config.poissonPointWeight, 3.6f);
            config.simplifyTargetFaces = 14000;
        }
        else if (voxelDensity == QStringLiteral("fine"))
        {
            config.voxelSimplifyFactor = 1.25f;
            config.enableDownsample = false;
            config.denoiseStdMul = 1.8f;
            config.kNormals = 18;
            config.poissonPointWeight = std::max(config.poissonPointWeight, 4.6f);
            config.poissonTrim = std::max(8.0f, config.poissonTrim);
            config.simplifyTargetFaces = 60000;
        }
        else
        {
            config.voxelSimplifyFactor = 1.8f;
            config.enableDownsample = true;
            config.downsampleVoxelScale = 0.8f;
            config.simplifyTargetFaces = 28000;
        }

        const int targetFaces = settings.value(QStringLiteral("simplifyTargetFaces"))
                                    .toInt(settings.value(QStringLiteral("targetFaces")).toInt(0));
        if (targetFaces > 0)
        {
            config.simplifyTargetFaces = qBound(1000, targetFaces, 2000000);
        }
        else if (settings.contains(QStringLiteral("targetFaces")))
        {
            config.simplifyTargetFaces = 0;
        }

        if (config.forcePoisson)
        {
            // PlaPoint's current octree solver is capped at depth 8. Feeding tens of
            // millions of nearly redundant samples only increases solve time and
            // memory pressure; it cannot raise the octree resolution. Keep enough
            // samples to support the requested face budget while preserving a
            // bounded production runtime.
            if (config.simplifyTargetFaces > 0)
            {
                const long long requested_samples = static_cast<long long>(config.simplifyTargetFaces) * 5LL / 2LL;
                config.maxInputPointsForMeshing = static_cast<int>(std::clamp(requested_samples, 250000LL, 400000LL));
            }
            else
            {
                config.maxInputPointsForMeshing = 400000;
            }
        }

        const QString depthFiltering =
            settings.value(QStringLiteral("depthFiltering")).toString(QStringLiteral("moderate"));
        if (depthFiltering == QStringLiteral("disabled"))
        {
            config.enableDenoise = false;
        }
        else if (depthFiltering == QStringLiteral("mild"))
        {
            config.enableDenoise = true;
            config.denoiseK = 16;
            config.denoiseStdMul = std::max(config.denoiseStdMul, 1.6f);
        }
        else if (depthFiltering == QStringLiteral("aggressive"))
        {
            config.enableDenoise = true;
            config.denoiseK = std::max(config.denoiseK, 28);
            config.denoiseStdMul = std::min(config.denoiseStdMul, 0.95f);
        }
        else
        {
            config.enableDenoise = true;
            config.denoiseK = std::max(config.denoiseK, 20);
            config.denoiseStdMul = std::min(config.denoiseStdMul, 1.25f);
        }

        const bool decimate = settings.value(QStringLiteral("decimate")).toBool(false);
        if (decimate && config.simplifyTargetFaces > 0)
        {
            const double decimateRatio =
                std::clamp(settings.value(QStringLiteral("decimateRatio")).toDouble(0.5), 0.05, 1.0);
            config.simplifyTargetFaces =
                std::max(1000, static_cast<int>(std::lround(config.simplifyTargetFaces * decimateRatio)));
            config.enableDownsample = true;
            config.voxelSimplifyFactor = std::max(config.voxelSimplifyFactor, static_cast<float>(1.0 / decimateRatio));
        }

        config.verbose = false;
        return config;
    }

    xjw::mesh::ReconstructionConfig
    reconstructionConfigForDenseScene(int requestedResolution, bool aerialTerrain, bool preserveDetail)
    {
        xjw::mesh::ReconstructionConfig config;
        config.resolution = qBound(64, requestedResolution, 1024);
        config.poissonDepth = 9;
        config.simplifyTargetFaces = 28000;
        config.forcePoisson = !aerialTerrain;
        config.allowHeightGridFallback = aerialTerrain;
        config.orientNormalsForClosedSurface = !aerialTerrain;

        if (aerialTerrain && preserveDetail)
        {
            config.resolution = std::max(config.resolution, 320);
            config.enableDownsample = false;
            config.simplifyTargetFaces = 65000;
            config.smoothIterations = std::max(config.smoothIterations, 4);
            config.smoothLambda = 0.36f;
        }

        return config;
    }

    int holeFillPassesFromArea(double maxHoleArea)
    {
        if (maxHoleArea <= 0.0)
        {
            return 0;
        }

        return qBound(1, static_cast<int>(std::ceil(std::sqrt(maxHoleArea * 0.5))), 64);
    }

    xjw::mesh::TextureMappingConfig defaultTextureConfig()
    {
        xjw::mesh::TextureMappingConfig config;
        return config;
    }

    xjw::mesh::TextureMappingConfig textureConfigFromSettings(const QJsonObject& settings)
    {
        xjw::mesh::TextureMappingConfig config = defaultTextureConfig();
        if (!settings.isEmpty())
        {
            config.textureSize = settings.value(QStringLiteral("textureSize")).toInt(config.textureSize);
            config.imageDownscale = settings.value(QStringLiteral("imageDownscale")).toInt(config.imageDownscale);
            config.antiAliasing = settings.value(QStringLiteral("antiAliasing")).toInt(config.antiAliasing);
            config.atlasUpscaleLimit = static_cast<float>(
                settings.value(QStringLiteral("atlasUpscaleLimit")).toDouble(config.atlasUpscaleLimit));
            config.padding = settings.value(QStringLiteral("padding")).toInt(config.padding);
            config.maximumCandidateViews =
                settings.value(QStringLiteral("maximumCandidateViews")).toInt(config.maximumCandidateViews);
            config.maximumBlendedViews =
                settings.value(QStringLiteral("maximumBlendedViews")).toInt(config.maximumBlendedViews);
            config.keepUnmapped = settings.value(QStringLiteral("keepUnmapped")).toBool(config.keepUnmapped);
            config.enableGhostFilter = settings.value(QStringLiteral("ghostFilter")).toBool(config.enableGhostFilter);
            config.enableOutOfFocusFilter =
                settings.value(QStringLiteral("outOfFocusFilter")).toBool(config.enableOutOfFocusFilter);
            config.enableColorCorrection =
                settings.value(QStringLiteral("colorCorrection")).toBool(config.enableColorCorrection);
            config.enableFinalMeshVisibility =
                settings.value(QStringLiteral("finalMeshVisibility")).toBool(config.enableFinalMeshVisibility);
            config.enableSeamLeveling =
                settings.value(QStringLiteral("seamLeveling")).toBool(config.enableSeamLeveling);
            config.seamBorderBlendRadiusPixels =
                settings.value(QStringLiteral("seamBorderBlendRadiusPixels")).toInt(config.seamBorderBlendRadiusPixels);
            config.seamMaximumLinearCorrection =
                static_cast<float>(settings.value(QStringLiteral("seamMaximumLinearCorrection"))
                                       .toDouble(config.seamMaximumLinearCorrection));
            config.seamGlobalCorrectionStrength =
                static_cast<float>(settings.value(QStringLiteral("seamGlobalCorrectionStrength"))
                                       .toDouble(config.seamGlobalCorrectionStrength));
            config.sharpeningStrength = static_cast<float>(
                settings.value(QStringLiteral("sharpeningStrength")).toDouble(config.sharpeningStrength));
            const QString hole_fill_mode = settings.value(QStringLiteral("holeFillMode")).toString();
            if (hole_fill_mode == QStringLiteral("disabled"))
            {
                config.holeFillMode = xjw::mesh::TextureHoleFillMode::Disabled;
            }
            else if (hole_fill_mode == QStringLiteral("neighbor_view_recovery"))
            {
                config.holeFillMode = xjw::mesh::TextureHoleFillMode::NeighborViewRecovery;
            }
            else
            {
                config.holeFillMode = settings.value(QStringLiteral("holeFill")).toBool(true)
                                          ? xjw::mesh::TextureHoleFillMode::TextureSpaceSmallHoles
                                          : xjw::mesh::TextureHoleFillMode::Disabled;
            }

            const QString blendMethod = settings.value(QStringLiteral("blendMode"))
                                            .toString(settings.value(QStringLiteral("blendMethod")).toString())
                                            .trimmed()
                                            .toLower();
            if (!blendMethod.isEmpty())
            {
                config.blendMethod = blendMethod.toStdString();
                if (blendMethod == QStringLiteral("best_view") || blendMethod.contains(QStringLiteral("最佳")))
                {
                    config.blendMode = xjw::mesh::TextureBlendMode::BestView;
                }
                else if (blendMethod == QStringLiteral("weighted_average") ||
                         blendMethod.contains(QStringLiteral("加权")))
                {
                    config.blendMode = xjw::mesh::TextureBlendMode::WeightedAverage;
                }
                else
                {
                    config.blendMode = xjw::mesh::TextureBlendMode::Natural;
                }
            }

            const QString uvMethod = settings.value(QStringLiteral("mappingMode"))
                                         .toString(settings.value(QStringLiteral("uvMethod")).toString())
                                         .trimmed()
                                         .toLower();
            if (!uvMethod.isEmpty())
            {
                config.uvMethod = uvMethod.toStdString();
                config.mappingMode = uvMethod == QStringLiteral("keep_existing_uv")
                                         ? xjw::mesh::TextureMappingMode::KeepExistingUv
                                         : xjw::mesh::TextureMappingMode::AutoProjective;
            }
        }

        return config;
    }
} // namespace xjw::mesh::workflow

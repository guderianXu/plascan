#include "BaBenchmarkRealDataset.h"
#include "BundleAdjustAdaptiveCameraModel.h"
#include "io/PathIO.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct BenchmarkSettings
    {
        int threads = 32;
        int iterations = 8;
        int repetitions = 1;
        int maxDenseSchurCameras = 200;
        bool refinePose = false;
        bool isolatedBackend = false;
        bool referenceOnlineSchur = false;
        double loadSeconds = 0.0;
        std::string cameraModel = "fixed";
    };

    struct CameraModelRunInfo
    {
        std::string effectiveModel = "fixed";
        bool assessmentValid = false;
        int activeCameraCount = 0;
        double assessmentSeconds = 0.0;
        double policySeconds = 0.0;
        xjw::BAIntrinsicParameterMask enabled{};
        std::array<double, xjw::kBAIntrinsicParameterCount> reliability{};
        std::string assessmentReason = "not_requested";
    };
    struct RealDatasetOptions
    {
        std::filesystem::path datasetJson;
        std::filesystem::path cameraList;
        std::string backends = "plamatrix_cpu";
        BenchmarkSettings settings;
    };
    xjw::FramePinholeCamera makeCamera(double cx, double cy, double cz)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
        camera.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{cx, cy, cz}});
        return camera;
    }
    std::vector<xjw::FramePinholeCamera> makeCameras(int count)
    {
        std::vector<xjw::FramePinholeCamera> cameras;
        cameras.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i)
        {
            const double t = (static_cast<double>(i) / std::max(1, count - 1) - 0.5) * 18.0;
            cameras.push_back(makeCamera(t, std::sin(i * 0.45) * 3.0, 0.0));
        }
        return cameras;
    }
    std::vector<xjw::BATrack>
    makeTracks(const std::vector<xjw::FramePinholeCamera>& cameras, int trackCount, int viewsPerTrack)
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> xy(-4.0, 4.0);
        std::uniform_real_distribution<double> z(32.0, 52.0);
        std::normal_distribution<double> initNoise(0.0, 0.30);
        std::normal_distribution<double> imageNoise(0.0, 0.05);

        std::vector<xjw::BATrack> tracks;
        tracks.reserve(static_cast<std::size_t>(trackCount));
        for (int i = 0; i < trackCount; ++i)
        {
            const std::array<double, 3> truth{{xy(rng), xy(rng), z(rng)}};
            xjw::BATrack track;
            track.initialPoint = {{truth[0] + initNoise(rng), truth[1] + initNoise(rng), truth[2] + initNoise(rng)}};
            const int start = i % static_cast<int>(cameras.size());
            for (int k = 0; k < viewsPerTrack; ++k)
            {
                const int ci = (start + k * 3) % static_cast<int>(cameras.size());
                const double world[3] = {truth[0], truth[1], truth[2]};
                double pixel[2] = {0.0, 0.0};
                if (cameras[static_cast<std::size_t>(ci)].projectWorldPoint(world, pixel))
                {
                    track.observations.push_back(
                        xjw::BAObservation{ci, pixel[0] + imageNoise(rng), pixel[1] + imageNoise(rng), 1.0});
                }
            }
            if (track.observations.size() >= 2)
            {
                tracks.push_back(std::move(track));
            }
        }
        return tracks;
    }
    std::string sanitizeField(std::string value)
    {
        std::replace(value.begin(), value.end(), ',', ';');
        std::replace(value.begin(), value.end(), '\n', ' ');
        std::replace(value.begin(), value.end(), '\r', ' ');
        return value;
    }
    std::vector<std::string> splitBackends(const std::string& raw)
    {
        std::vector<std::string> names;
        std::stringstream stream(raw);
        std::string item;
        while (std::getline(stream, item, ','))
        {
            item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }),
                       item.end());
            if (!item.empty())
            {
                names.push_back(item);
            }
        }
        return names;
    }
    bool parseBackend(const std::string& name, xjw::BABackend* backend)
    {
        if (name == "legacy_cpu")
        {
            *backend = xjw::BABackend::PlaMatrixCpu;
        }
        else if (name == "plamatrix_cpu")
        {
            *backend = xjw::BABackend::PlaMatrixCpu;
        }
        else if (name == "plamatrix_cuda")
        {
            *backend = xjw::BABackend::PlaMatrixCuda;
        }
        else if (name == "plamatrix_opencl")
        {
            *backend = xjw::BABackend::PlaMatrixOpenCl;
        }
        else if (name == "auto")
        {
            *backend = xjw::BABackend::Auto;
        }
        else
        {
            return false;
        }
        return true;
    }

    void enableFullSharedCameraModel(xjw::BAOptions* options)
    {
        options->refineSharedFocalLength = true;
        options->refineSharedFocalAspectRatio = true;
        options->refineSharedPrincipalPoint = true;
        options->refineSharedRadialDistortion = true;
        options->refineSharedHighOrderDistortion = true;
    }

    CameraModelRunInfo configureCameraModel(const BenchmarkSettings& settings,
                                            const std::vector<xjw::FramePinholeCamera>& cameras,
                                            const std::vector<xjw::BATrack>& tracks,
                                            xjw::BAOptions* options)
    {
        CameraModelRunInfo info;
        if (settings.cameraModel == "fixed")
        {
            return info;
        }

        enableFullSharedCameraModel(options);
        if (settings.cameraModel == "full")
        {
            info.enabled.fill(true);
            info.effectiveModel = xjw::adaptiveCameraModelName(info.enabled);
            return info;
        }

        const auto policyStarted = std::chrono::steady_clock::now();
        const auto assessmentStarted = std::chrono::steady_clock::now();
        const xjw::BAAdaptiveCameraModelAssessment assessment =
            xjw::assessAdaptiveCameraModel(cameras, tracks, options);
        info.assessmentSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - assessmentStarted).count();
        xjw::applyAdaptiveCameraModel(assessment, options);
        info.policySeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - policyStarted).count();
        info.assessmentValid = assessment.valid;
        info.activeCameraCount = assessment.activeCameraCount;
        info.enabled = options->sharedIntrinsicParameterMask;
        info.reliability = assessment.reliability;
        info.effectiveModel = xjw::adaptiveCameraModelName(info.enabled);
        info.assessmentReason = assessment.reason;
        return info;
    }

    void appendCameraModelMetrics(std::ostream& stream,
                                  const BenchmarkSettings& settings,
                                  const CameraModelRunInfo& info,
                                  const xjw::BAResult& result,
                                  double apiWallSeconds)
    {
        const bool modelApplied = settings.cameraModel != "fixed" && result.solutionUsable &&
                                  !result.qualityGateRejected && result.refinedIntrinsicCount > 0;
        stream << ",camera_model_requested=" << settings.cameraModel
               << ",camera_model_effective=" << info.effectiveModel
               << ",camera_model_applied=" << (modelApplied ? "true" : "false")
               << ",camera_model_assessment_valid=" << (info.assessmentValid ? "true" : "false")
               << ",camera_model_assessment_seconds=" << info.assessmentSeconds
               << ",camera_model_policy_seconds=" << info.policySeconds
               << ",end_to_end_seconds=" << apiWallSeconds + info.policySeconds
               << ",camera_model_assessment_reason=" << sanitizeField(info.assessmentReason)
               << ",camera_model_active_cameras=" << info.activeCameraCount;
        for (std::size_t index = 0; index < xjw::kBAIntrinsicParameterCount; ++index)
        {
            const auto parameter = static_cast<xjw::BAIntrinsicParameter>(index);
            const char* parameterName = xjw::baIntrinsicParameterName(parameter);
            stream << ",intrinsic_" << parameterName << "_enabled=" << (info.enabled[index] ? "true" : "false")
                   << ",intrinsic_" << parameterName << "_reliability=" << info.reliability[index];
        }
    }

    void runCase(const std::string& name,
                 xjw::BABackend backend,
                 const xjw::ba_benchmark::BenchmarkDataset& dataset,
                 const BenchmarkSettings& settings)
    {
        for (int repetition = 1; repetition <= settings.repetitions; ++repetition)
        {
            std::vector<xjw::FramePinholeCamera> cameras = dataset.cameras;
            std::vector<xjw::BATrack> tracks = dataset.tracks;
            xjw::BAOptions options;
            options.backend = backend;
            options.numThreads = settings.threads;
            options.maxIterations = settings.iterations;
            options.refineCameraPose = settings.refinePose;
            options.enablePointFilter = false;
            options.maxDenseSchurCameras = settings.maxDenseSchurCameras;
            options.allowBackendFallback = !settings.isolatedBackend;
            options.enableBackendQualityGate = !settings.isolatedBackend;
            options.compareAutoBackendWithLegacy = !settings.isolatedBackend;
            options.logIterationProgress = !settings.isolatedBackend;
            options.useReferenceOnlineSchur = settings.referenceOnlineSchur;
            if (settings.refinePose)
            {
                options.fixedCameraIndices.push_back(0);
                const int scale_camera_index = static_cast<int>(cameras.size()) - 1;
                const auto anchor = cameras.front().cameraCenter();
                const auto scale = cameras.back().cameraCenter();
                const double dx = scale[0] - anchor[0];
                const double dy = scale[1] - anchor[1];
                const double dz = scale[2] - anchor[2];
                options.referenceGaugeAnchorCameraIndex = 0;
                options.referenceGaugeScaleCameraIndex = scale_camera_index;
                options.referenceGaugeBaseline = std::sqrt(dx * dx + dy * dy + dz * dz);
            }

            const CameraModelRunInfo cameraModelInfo = configureCameraModel(settings, cameras, tracks, &options);

            const auto started = std::chrono::steady_clock::now();
            const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);
            const double apiWallSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

            std::cout << name << ",repetition=" << repetition << ",cold_start=" << (repetition == 1 ? "true" : "false")
                      << ",requested=" << xjw::BundleAdjust::backendName(result.requestedBackend)
                      << ",used=" << xjw::BundleAdjust::backendName(result.usedBackend)
                      << ",status=" << xjw::BundleAdjust::solveStatusName(result.solveStatus)
                      << ",gpu=" << (result.usedGpu ? "true" : "false")
                      << ",fallback=" << (result.backendFallback ? "true" : "false")
                      << ",observations=" << result.observationCount << ",tracks=" << result.totalTracks
                      << ",optimized=" << result.optimizedTracks << ",valid_ratio=" << result.validTrackRatio
                      << ",rms_before=" << result.meanRmsBefore << ",rms_after=" << result.meanRmsAfter
                      << ",quality_rejected=" << (result.qualityGateRejected ? "true" : "false")
                      << ",quality_message=" << sanitizeField(result.qualityGateMessage)
                      << ",plamatrix_initial_cost=" << result.plaMatrixInitialCost
                      << ",plamatrix_final_cost=" << result.plaMatrixFinalCost
                      << ",plamatrix_accepted_steps=" << result.plaMatrixAcceptedSteps
                      << ",plamatrix_rejected_steps=" << result.plaMatrixRejectedSteps
                      << ",plamatrix_linearizations=" << result.plaMatrixLinearizations
                      << ",plamatrix_objective_evaluations=" << result.plaMatrixObjectiveEvaluations
                      << ",plamatrix_rejected_initial_tracks=" << result.plaMatrixRejectedInitialTracks
                      << ",plamatrix_reference_online_schur_used="
                      << (result.plaMatrixReferenceOnlineSchurUsed ? "true" : "false")
                      << ",plamatrix_linear_solver=" << sanitizeField(result.plaMatrixLinearSolverName)
                      << ",plamatrix_preconditioner=" << sanitizeField(result.plaMatrixPreconditionerName)
                      << ",plamatrix_device=" << sanitizeField(result.plaMatrixDeviceName)
                      << ",plamatrix_linear_iterations=" << result.plaMatrixLinearIterations
                      << ",plamatrix_dense_fallbacks=" << result.plaMatrixDenseFallbacks
                      << ",plamatrix_dense_fallback_message=" << sanitizeField(result.plaMatrixDenseFallbackMessage)
                      << ",plamatrix_assembly_seconds=" << result.plaMatrixAssemblySeconds
                      << ",plamatrix_objective_seconds=" << result.plaMatrixObjectiveSeconds
                      << ",plamatrix_trial_state_seconds=" << result.plaMatrixTrialStateSeconds
                      << ",plamatrix_linear_tolerance_policy=strict_reference"
                      << ",plamatrix_linear_tolerance_min=" << result.plaMatrixLinearToleranceMinimum
                      << ",plamatrix_linear_tolerance_max=" << result.plaMatrixLinearToleranceMaximum
                      << ",plamatrix_schur_pattern_builds=" << result.plaMatrixSchurPatternBuilds
                      << ",plamatrix_schur_pattern_reuses=" << result.plaMatrixSchurPatternReuses
                      << ",plamatrix_schur_assembly_on_device=" << (result.plaMatrixSchurAssemblyOnDevice ? 1 : 0)
                      << ",plamatrix_mixed_precision_used=" << (result.plaMatrixMixedPrecisionUsed ? 1 : 0)
                      << ",plamatrix_small_block_inverse_seconds=" << result.plaMatrixSmallBlockInverseSeconds
                      << ",plamatrix_schur_accumulation_seconds=" << result.plaMatrixSchurAccumulationSeconds
                      << ",plamatrix_csr_conversion_seconds=" << result.plaMatrixCsrConversionSeconds
                      << ",plamatrix_schur_assembly_seconds=" << result.plaMatrixSchurAssemblySeconds
                      << ",plamatrix_cholesky_factorization_seconds=" << result.plaMatrixCholeskyFactorizationSeconds
                      << ",plamatrix_triangular_solve_seconds=" << result.plaMatrixTriangularSolveSeconds
                      << ",plamatrix_symbolic_analysis_seconds=" << result.plaMatrixSymbolicAnalysisSeconds
                      << ",plamatrix_symbolic_analysis_reuses=" << result.plaMatrixSymbolicAnalysisReuses
                      << ",plamatrix_residual_check_seconds=" << result.plaMatrixResidualCheckSeconds
                      << ",plamatrix_linear_solve_seconds=" << result.plaMatrixLinearSolveSeconds
                      << ",plamatrix_back_substitution_seconds=" << result.plaMatrixBackSubstitutionSeconds
                      << ",setup_seconds=" << result.setupSeconds << ",solve_seconds=" << result.solveSeconds
                      << ",postprocess_seconds=" << result.postprocessSeconds
                      << ",total_seconds=" << result.totalSeconds << ",api_wall_seconds=" << apiWallSeconds
                      << ",seconds=" << apiWallSeconds << ",load_seconds=" << settings.loadSeconds
                      << ",backend_reason=" << sanitizeField(result.backendSelectionReason)
                      << ",backend_message=" << sanitizeField(result.backendMessage);
            appendCameraModelMetrics(std::cout, settings, cameraModelInfo, result, apiWallSeconds);
            std::cout << "\n" << std::flush;
        }
    }
    int parseInteger(const std::string& value, const std::string& option, int minimum)
    {
        char* end = nullptr;
        const long parsed = std::strtol(value.c_str(), &end, 10);
        if (!end || *end != '\0' || parsed < minimum || parsed > std::numeric_limits<int>::max())
        {
            throw std::runtime_error(option + " 需要 >= " + std::to_string(minimum) + " 的整数");
        }
        return static_cast<int>(parsed);
    }
    void printUsage(const char* program)
    {
        std::cout << "用法:\n"
                  << "  " << program
                  << " [camera_count track_count views_per_track iterations threads refine_pose backends"
                     " max_dense_schur_cameras reference_online_schur]\n"
                  << "  " << program << " --dataset-json PATH --camera-list PATH [选项]\n\n"
                  << "真实数据选项:\n"
                  << "  --backend NAME[,NAME...]              默认 plamatrix_cpu\n"
                  << "  --iterations N                        默认 8\n"
                  << "  --threads N                           默认 32\n"
                  << "  --repetitions N                       默认 1；第 1 轮标记为冷启动\n"
                  << "  --refine-pose                         联合优化相机位姿\n"
                  << "  --reference-online-schur              使用参考在线点 Schur 装配与回代\n"
                  << "  --camera-model fixed|full|adaptive    默认 fixed；控制共享内参自标定模型\n"
                  << "  --max-dense-schur-cameras N           默认 200\n";
    }

    RealDatasetOptions parseRealOptions(int argc, char** argv)
    {
        RealDatasetOptions options;
        options.settings.isolatedBackend = true;
        for (int i = 1; i < argc; ++i)
        {
            const std::string option = argv[i];
            auto value = [&]() -> std::string
            {
                if (++i >= argc)
                {
                    throw std::runtime_error(option + " 缺少参数值");
                }
                return argv[i];
            };
            if (option == "--dataset-json")
            {
                options.datasetJson = xjw::common::io::toFilesystemPath(value());
            }
            else if (option == "--camera-list")
            {
                options.cameraList = xjw::common::io::toFilesystemPath(value());
            }
            else if (option == "--backend")
            {
                options.backends = value();
            }
            else if (option == "--iterations")
            {
                options.settings.iterations = parseInteger(value(), option, 1);
            }
            else if (option == "--threads")
            {
                options.settings.threads = parseInteger(value(), option, 1);
            }
            else if (option == "--repetitions")
            {
                options.settings.repetitions = parseInteger(value(), option, 1);
            }
            else if (option == "--camera-model")
            {
                options.settings.cameraModel = value();
                if (options.settings.cameraModel != "fixed" && options.settings.cameraModel != "full" &&
                    options.settings.cameraModel != "adaptive")
                {
                    throw std::runtime_error("--camera-model 必须是 fixed、full 或 adaptive");
                }
            }
            else if (option == "--max-dense-schur-cameras")
            {
                options.settings.maxDenseSchurCameras = parseInteger(value(), option, 0);
            }
            else if (option == "--refine-pose")
            {
                options.settings.refinePose = true;
            }
            else if (option == "--reference-online-schur")
            {
                options.settings.referenceOnlineSchur = true;
            }
            else if (option != "--help" && option != "-h")
            {
                throw std::runtime_error("未知选项: " + option);
            }
        }
        if (options.datasetJson.empty() || options.cameraList.empty())
        {
            throw std::runtime_error("真实数据模式必须同时提供 --dataset-json 和 --camera-list");
        }
        return options;
    }

    void runRequestedBackends(const std::string& raw,
                              const xjw::ba_benchmark::BenchmarkDataset& dataset,
                              const BenchmarkSettings& settings)
    {
        const std::vector<std::string> names = splitBackends(raw);
        if (names.empty())
        {
            throw std::runtime_error("backend 列表为空");
        }
        for (const std::string& name : names)
        {
            xjw::BABackend backend;
            if (!parseBackend(name, &backend))
            {
                throw std::runtime_error("未知 backend: " + name);
            }
            runCase(name, backend, dataset, settings);
        }
    }

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::cout << std::setprecision(10);
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
        {
            printUsage(argv[0]);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]).starts_with("--"))
        {
            RealDatasetOptions options = parseRealOptions(argc, argv);
            const auto started = std::chrono::steady_clock::now();
            const xjw::ba_benchmark::BenchmarkDataset dataset =
                xjw::ba_benchmark::loadRealDataset(options.datasetJson, options.cameraList);
            options.settings.loadSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << "dataset,mode=real,cameras=" << dataset.cameras.size() << ",tracks=" << dataset.tracks.size()
                      << ",observations=" << dataset.observations << ",iterations=" << options.settings.iterations
                      << ",threads=" << options.settings.threads
                      << ",refine_pose=" << (options.settings.refinePose ? "true" : "false")
                      << ",reference_online_schur=" << (options.settings.referenceOnlineSchur ? "true" : "false")
                      << ",camera_model=" << options.settings.cameraModel
                      << ",max_dense_schur_cameras=" << options.settings.maxDenseSchurCameras
                      << ",repetitions=" << options.settings.repetitions
                      << ",load_seconds=" << options.settings.loadSeconds << "\n";
            runRequestedBackends(options.backends, dataset, options.settings);
            return 0;
        }

        const int cameraCount = argc > 1 ? std::max(2, std::atoi(argv[1])) : 80;
        const int trackCount = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3000;
        const int viewsPerTrack = argc > 3 ? std::max(2, std::atoi(argv[3])) : 8;
        BenchmarkSettings settings;
        settings.iterations = argc > 4 ? std::max(1, std::atoi(argv[4])) : 8;
        settings.threads = argc > 5 ? std::max(1, std::atoi(argv[5])) : 32;
        settings.refinePose = argc > 6 ? std::atoi(argv[6]) != 0 : false;
        const std::string backends = argc > 7 ? argv[7] : "plamatrix_cpu,plamatrix_cuda,plamatrix_opencl,auto";
        settings.maxDenseSchurCameras = argc > 8 ? std::max(0, std::atoi(argv[8])) : settings.maxDenseSchurCameras;
        settings.referenceOnlineSchur = argc > 9 ? std::atoi(argv[9]) != 0 : false;

        const auto started = std::chrono::steady_clock::now();
        xjw::ba_benchmark::BenchmarkDataset dataset;
        dataset.cameras = makeCameras(cameraCount);
        dataset.tracks = makeTracks(dataset.cameras, trackCount, viewsPerTrack);
        for (const xjw::BATrack& track : dataset.tracks)
        {
            dataset.observations += track.observations.size();
        }
        settings.loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "dataset,mode=synthetic,cameras=" << dataset.cameras.size() << ",tracks=" << dataset.tracks.size()
                  << ",observations=" << dataset.observations << ",views_per_track=" << viewsPerTrack
                  << ",iterations=" << settings.iterations << ",threads=" << settings.threads
                  << ",refine_pose=" << (settings.refinePose ? "true" : "false")
                  << ",reference_online_schur=" << (settings.referenceOnlineSchur ? "true" : "false")
                  << ",camera_model=" << settings.cameraModel
                  << ",max_dense_schur_cameras=" << settings.maxDenseSchurCameras
                  << ",load_seconds=" << settings.loadSeconds << "\n";
        runRequestedBackends(backends, dataset, settings);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ba_backend_benchmark: " << error.what() << "\n";
        return 2;
    }
}

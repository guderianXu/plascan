#include "cli_common.h"
#include "CliConsole.h"

#include "IsisControlNetworkPvl.h"
#include "PlanetaryLaserJson.h"
#include "PlanetaryLineScanBaCliOutput.h"
#include "PlanetaryLineScanBundleAdjust.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QString>
#include <QStringList>

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace
{

xjw::lidar::PlanetaryLaserLineScanTimeMode parseTimeMode(const std::string &value)
{
    if (value == "shot_et")
    {
        return xjw::lidar::PlanetaryLaserLineScanTimeMode::ShotEphemerisTime;
    }
    if (value == "isis_line")
    {
        return xjw::lidar::PlanetaryLaserLineScanTimeMode::IsisSimultaneousMeasureLine;
    }
    cli::fatal("--laser-time-mode must be shot_et or isis_line");
    return xjw::lidar::PlanetaryLaserLineScanTimeMode::ShotEphemerisTime;
}

xjw::lidar::PlanetaryLaserDataset loadLaserDataset(const std::string &path)
{
    xjw::lidar::PlanetaryLaserIsisContext context;
    context.reference.targetName = "MOON";
    context.reference.bodyFixedFrame = "MOON_ME";
    context.reference.laserFrame = "LRO_LOLA";
    context.reference.timeSystem = xjw::lidar::PlanetaryLaserTimeSystem::TdbEtSeconds;
    context.reference.latitudeType = "planetocentric";
    context.reference.longitudeDirection = "positive_east";
    context.sensorModel = xjw::lidar::PlanetaryLaserSensorModel::LineScan;
    context.rangeType = xjw::lidar::PlanetaryLaserRangeType::OneWay;
    context.leverArmSensorMeters = std::array<double, 3>{{0.0, 0.0, 0.0}};

    xjw::lidar::PlanetaryLaserJsonParseOptions parseOptions;
    parseOptions.isisContext = context;
    xjw::lidar::PlanetaryLaserDataset dataset;
    std::string error;
    if (!xjw::lidar::loadPlanetaryLaserJsonFile(path, parseOptions, &dataset, &error))
    {
        cli::fatal("failed to load ISIS LidarData JSON: " + error, cli::EXIT_IO_ERR);
    }
    return dataset;
}

QString firstExistingArtifact(const QDir &directory)
{
    const QStringList names{
        QStringLiteral("no_laser_result.json"),
        QStringLiteral("no_laser_sparse_moon_me.ply"),
        QStringLiteral("no_laser_sparse_local.ply"),
        QStringLiteral("no_laser_laser_residuals.csv"),
        QStringLiteral("with_laser_result.json"),
        QStringLiteral("with_laser_sparse_moon_me.ply"),
        QStringLiteral("with_laser_sparse_local.ply"),
        QStringLiteral("with_laser_laser_residuals.csv"),
        QStringLiteral("comparison.json"),
    };
    for (const QString &name : names)
    {
        if (QFileInfo::exists(directory.filePath(name)))
        {
            return name;
        }
    }
    return {};
}

void writeComparison(const QString &outputDirectory,
                     const QString &timeMode,
                     const xjw::lidar::PlanetaryLineScanBaResult &withoutLaser,
                     const xjw::lidar::PlanetaryLineScanBaResult &withLaser)
{
    QJsonObject comparison{
        {QStringLiteral("schema"), QStringLiteral("plascan.planetary_linescan_ba_comparison")},
        {QStringLiteral("version"), 1},
        {QStringLiteral("coordinate_frame"), QStringLiteral("MOON_ME")},
        {QStringLiteral("laser_time_mode"), timeMode},
        {QStringLiteral("range_residual_sign"), QStringLiteral("computed_minus_observed")},
        {QStringLiteral("model_level"), QStringLiteral("P0")},
        {QStringLiteral("range_origin_model"),
         QStringLiteral("image_sensor_center_zero_lever_arm")},
        {QStringLiteral("camera_bias_model"),
         QStringLiteral("per_image_rigid_body_fixed_6dof")},
        {QStringLiteral("nominal_trajectory_interpolation"),
         QStringLiteral("raw_isd_hermite_position_slerp_attitude")},
        {QStringLiteral("input_isd_interpolation_method"), QStringLiteral("lagrange")},
        {QStringLiteral("approximate_usgscsm_interpolation_accepted"), true},
        {QStringLiteral("without_laser"),
         xjw::cli::planetaryLineScanBaResultToJson(withoutLaser)},
        {QStringLiteral("with_laser"),
         xjw::cli::planetaryLineScanBaResultToJson(withLaser)},
        {QStringLiteral("range_rms_improvement_m"),
         withoutLaser.refinedLaserRangeRmsMeters - withLaser.refinedLaserRangeRmsMeters},
        {QStringLiteral("image_rms_change_px"),
         withLaser.refinedImageRmsPixels - withoutLaser.refinedImageRmsPixels},
    };
    QSaveFile file(QDir(outputDirectory).filePath(QStringLiteral("comparison.json")));
    if (!file.open(QIODevice::WriteOnly) ||
        file.write(QJsonDocument(comparison).toJson(QJsonDocument::Indented)) < 0 ||
        !file.commit())
    {
        cli::fatal("failed to write comparison.json", cli::EXIT_IO_ERR);
    }
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qtApplication(argc, argv);
    Q_UNUSED(qtApplication);

    CLI::App app{
        "PlaScan P0 planetary pushbroom sparse BA with optional ISIS/LOLA laser ranges"};
    std::vector<std::string> isdPaths;
    std::vector<std::string> serialNumbers;
    std::string controlNetworkPath;
    std::string laserDataPath;
    std::string outputDirectory;
    std::string laserTimeMode = "shot_et";
    int maximumIterations = 50;
    int threads = 0;
    double positionSigmaMeters = 1000.0;
    double angleSigmaDegrees = 2.0;
    double imageSigmaPixels = 1.0;
    double imageHuberPixels = 3.0;
    double rangeWeight = 1.0;
    double rangeHuberSigma = 3.0;
    bool assumeZeroLaserLeverArm = false;
    bool acceptApproximateUsgsCsmInterpolation = false;

    app.add_option("--isd", isdPaths,
                   "USGSCSM line-scan ISD path; repeat in camera order")
        ->required();
    app.add_option("--serial", serialNumbers,
                   "ISIS camera serial matching each --isd; repeat in the same order")
        ->required();
    app.add_option("--control-network", controlNetworkPath,
                   "ISIS jigsaw textual control network PVL")
        ->required();
    app.add_option("--laser-data", laserDataPath,
                   "ISIS LidarData JSON; when supplied, run no-laser/with-laser A/B");
    app.add_flag("--assume-zero-laser-lever-arm", assumeZeroLaserLeverArm,
                 "explicitly accept the P0 image-sensor-center range origin");
    app.add_flag("--accept-approximate-usgscsm-interpolation",
                 acceptApproximateUsgsCsmInterpolation,
                 "accept raw ISD Hermite/SLERP instead of full ALE/USGSCSM resampled Lagrange");
    app.add_option("-o,--output-dir", outputDirectory, "output directory")->required();
    app.add_option("--laser-time-mode", laserTimeMode,
                   "shot_et (direct observation timestamp, default) or isis_line (ISIS regression compatibility)");
    app.add_option("--max-iterations", maximumIterations, "Ceres maximum iterations");
    app.add_option("--threads", threads, "Ceres threads; 0 uses hardware concurrency");
    app.add_option("--camera-position-sigma-m", positionSigmaMeters,
                   "per-image body-fixed translation prior sigma in metres");
    app.add_option("--camera-angle-sigma-deg", angleSigmaDegrees,
                   "per-image small-angle prior sigma in degrees");
    app.add_option("--image-sigma-px", imageSigmaPixels, "control measure sigma in pixels");
    app.add_option("--image-huber-px", imageHuberPixels, "image Huber threshold in pixels");
    app.add_option("--laser-range-weight", rangeWeight, "laser range residual weight");
    app.add_option("--laser-huber-sigma", rangeHuberSigma,
                   "laser Huber threshold in normalized sigma units");
    CLI11_PARSE(app, argc, argv);

    if (isdPaths.size() != serialNumbers.size() || isdPaths.size() < 2)
    {
        cli::fatal("--isd and --serial counts must match and contain at least two cameras");
    }
    if (maximumIterations <= 0 || threads < 0 || !(positionSigmaMeters > 0.0) ||
        !(angleSigmaDegrees > 0.0) || !(imageSigmaPixels > 0.0) ||
        imageHuberPixels < 0.0 || !(rangeWeight > 0.0) || rangeHuberSigma < 0.0)
    {
        cli::fatal("invalid line-scan BA iteration, sigma, Huber, thread, or weight option");
    }
    if (!laserDataPath.empty() && !assumeZeroLaserLeverArm)
    {
        cli::fatal("--laser-data requires --assume-zero-laser-lever-arm in the P0 solver");
    }
    if (!acceptApproximateUsgsCsmInterpolation)
    {
        cli::fatal("P0 line-scan BA requires --accept-approximate-usgscsm-interpolation; "
                   "the nominal pose is not an exact ALE/USGSCSM Lagrange evaluation");
    }

    std::vector<xjw::lidar::PlanetaryLineScanBaCamera> cameras;
    cameras.reserve(isdPaths.size());
    for (std::size_t index = 0; index < isdPaths.size(); ++index)
    {
        xjw::lidar::PlanetaryLineScanBaCamera camera;
        camera.serialNumber = serialNumbers[index];
        std::string error;
        if (!camera.model.loadFromIsd(isdPaths[index], &error))
        {
            cli::fatal("failed to load line-scan ISD " + isdPaths[index] + ": " + error,
                       cli::EXIT_IO_ERR);
        }
        cameras.push_back(std::move(camera));
    }

    xjw::lidar::IsisControlNetwork controlNetwork;
    std::string error;
    if (!xjw::lidar::loadIsisControlNetworkPvlFile(
            controlNetworkPath, &controlNetwork, &error))
    {
        cli::fatal("failed to load ISIS control network: " + error, cli::EXIT_IO_ERR);
    }

    xjw::lidar::PlanetaryLaserDataset laserDataset;
    const xjw::lidar::PlanetaryLaserDataset *laserDatasetPointer = nullptr;
    if (!laserDataPath.empty())
    {
        laserDataset = loadLaserDataset(laserDataPath);
        laserDatasetPointer = &laserDataset;
    }

    xjw::cli::printUtf8(
        stdout,
        QStringLiteral("P0 模型: 每景 6DoF 刚性偏差；测距起点采用影像传感器中心，激光杆臂为零；"
                       "名义姿轨采用已显式接受的原始 ISD Hermite/SLERP 近似。"));

    xjw::lidar::PlanetaryLineScanBaOptions options;
    options.laserTimeMode = parseTimeMode(laserTimeMode);
    options.maximumIterations = maximumIterations;
    options.threadCount = threads;
    options.cameraPositionSigmaMeters = positionSigmaMeters;
    options.cameraAngleSigmaDegrees = angleSigmaDegrees;
    options.imageSigmaPixels = imageSigmaPixels;
    options.imageHuberDeltaPixels = imageHuberPixels;
    options.laserRangeWeight = rangeWeight;
    options.laserRangeHuberDeltaSigma = rangeHuberSigma;

    QDir directory(QString::fromStdString(outputDirectory));
    if (!directory.mkpath(QStringLiteral(".")))
    {
        cli::fatal("failed to create output directory", cli::EXIT_IO_ERR);
    }
    const QString existingArtifact = firstExistingArtifact(directory);
    if (!existingArtifact.isEmpty())
    {
        cli::fatal(
            QStringLiteral("output directory already contains %1; choose a fresh directory "
                           "to avoid mixing line-scan BA runs")
                .arg(existingArtifact)
                .toStdString(),
            cli::EXIT_IO_ERR);
    }
    QString artifactError;

    xjw::lidar::PlanetaryLineScanBaResult withoutLaser;
    options.enableLaserRangeConstraints = false;
    if (!xjw::lidar::runPlanetaryLineScanBundleAdjust(
            cameras, controlNetwork, laserDatasetPointer, options, &withoutLaser, &error))
    {
        cli::fatal("line-scan BA without laser failed: " + error, cli::EXIT_ALGO_ERR);
    }
    xjw::cli::printUtf8(
        stdout,
        QStringLiteral("无激光: image RMS %1 -> %2 px, range RMS %3 -> %4 m")
            .arg(withoutLaser.initialImageRmsPixels, 0, 'g', 10)
            .arg(withoutLaser.refinedImageRmsPixels, 0, 'g', 10)
            .arg(withoutLaser.initialLaserRangeRmsMeters, 0, 'g', 10)
            .arg(withoutLaser.refinedLaserRangeRmsMeters, 0, 'g', 10));
    if (!withoutLaser.converged)
    {
        xjw::cli::printUtf8(
            stderr,
            QStringLiteral("警告: 无激光解可用但未收敛 (%1)")
                .arg(QString::fromStdString(withoutLaser.terminationType)));
    }

    if (!laserDatasetPointer)
    {
        if (!xjw::cli::writePlanetaryLineScanBaArtifacts(
                directory.absolutePath(), QStringLiteral("no_laser"),
                withoutLaser, &artifactError))
        {
            cli::fatal(artifactError.toStdString(), cli::EXIT_IO_ERR);
        }
        return cli::EXIT_OK;
    }

    xjw::lidar::PlanetaryLineScanBaResult withLaser;
    options.enableLaserRangeConstraints = true;
    if (!xjw::lidar::runPlanetaryLineScanBundleAdjust(
            cameras, controlNetwork, laserDatasetPointer, options, &withLaser, &error))
    {
        cli::fatal("line-scan BA with laser failed: " + error, cli::EXIT_ALGO_ERR);
    }
    // Publish the A/B artifacts only after both solves succeed. Combined with
    // the fresh-output check above, a solver failure cannot leave mixed runs.
    if (!xjw::cli::writePlanetaryLineScanBaArtifacts(
            directory.absolutePath(), QStringLiteral("no_laser"),
            withoutLaser, &artifactError))
    {
        cli::fatal(artifactError.toStdString(), cli::EXIT_IO_ERR);
    }
    if (!xjw::cli::writePlanetaryLineScanBaArtifacts(
            directory.absolutePath(), QStringLiteral("with_laser"), withLaser, &artifactError))
    {
        cli::fatal(artifactError.toStdString(), cli::EXIT_IO_ERR);
    }
    writeComparison(directory.absolutePath(),
                    QString::fromLatin1(xjw::lidar::planetaryLaserLineScanTimeModeName(
                        options.laserTimeMode)),
                    withoutLaser, withLaser);

    xjw::cli::printUtf8(
        stdout,
        QStringLiteral("有激光: image RMS %1 -> %2 px, range RMS %3 -> %4 m，改善 %5 m")
            .arg(withLaser.initialImageRmsPixels, 0, 'g', 10)
            .arg(withLaser.refinedImageRmsPixels, 0, 'g', 10)
            .arg(withLaser.initialLaserRangeRmsMeters, 0, 'g', 10)
            .arg(withLaser.refinedLaserRangeRmsMeters, 0, 'g', 10)
            .arg(withoutLaser.refinedLaserRangeRmsMeters -
                     withLaser.refinedLaserRangeRmsMeters,
                 0, 'g', 10));
    if (!withLaser.converged)
    {
        xjw::cli::printUtf8(
            stderr,
            QStringLiteral("警告: 有激光解可用但未收敛 (%1)")
                .arg(QString::fromStdString(withLaser.terminationType)));
    }
    xjw::cli::printUtf8(stdout,
                        QStringLiteral("结果目录: %1").arg(directory.absolutePath()));
    return cli::EXIT_OK;
}

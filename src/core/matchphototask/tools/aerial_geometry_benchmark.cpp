#include "Camera.h"
#include "FeatureData.h"
#include "FeatureFileIO.h"
#include "FeatureOutput.h"
#include "Intersection.h"
#include "MatchGeometryFilter.h"
#include "pose/PnpSolver.h"
#include "match.h"

#include <QCoreApplication>
#include <QDir>
#include <QTemporaryDir>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct BenchmarkOptions
{
    int keypoints = 8000;
    int descriptorDimension = 128;
    int pairCount = 120;
    int geometryMatches = 2000;
    int triangulationPoints = 4000;
    int pnpImages = 16;
};

int positiveArgument(char **argv, int argc, int index, int fallback)
{
    if (index >= argc)
    {
        return fallback;
    }
    return std::max(1, std::atoi(argv[index]));
}

BenchmarkOptions parseOptions(int argc, char **argv)
{
    BenchmarkOptions options;
    options.keypoints = positiveArgument(argv, argc, 1, options.keypoints);
    options.descriptorDimension =
        positiveArgument(argv, argc, 2, options.descriptorDimension);
    options.pairCount = positiveArgument(argv, argc, 3, options.pairCount);
    options.geometryMatches =
        positiveArgument(argv, argc, 4, options.geometryMatches);
    options.triangulationPoints =
        positiveArgument(argv, argc, 5, options.triangulationPoints);
    options.pnpImages = positiveArgument(argv, argc, 6, options.pnpImages);
    return options;
}

double elapsedMilliseconds(
    const std::chrono::steady_clock::time_point &start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

xjw::Camera makeCamera(const std::array<double, 3> &center)
{
    xjw::Camera camera;
    camera.setIntrinsics(1200.0, 1200.0, 640.0, 480.0);
    camera.setPose(
        {{1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0}},
        center);
    return camera;
}

std::array<double, 3> syntheticWorldPoint(int index)
{
    const double x = (static_cast<double>(index % 50) - 24.5) * 0.04;
    const double y =
        (static_cast<double>((index / 50) % 40) - 19.5) * 0.04;
    const double z = 7.0 + static_cast<double>(index % 17) * 0.03;
    return {{x, y, z}};
}

std::array<double, 2> project(
    const xjw::Camera &camera,
    const std::array<double, 3> &point)
{
    const double world[3] = {point[0], point[1], point[2]};
    double pixel[2] = {0.0, 0.0};
    if (!camera.projectWorldPoint(world, pixel))
    {
        return {{0.0, 0.0}};
    }
    return {{pixel[0], pixel[1]}};
}

FeatureOutput makeFeatureOutput(
    int keypointCount,
    int descriptorDimension)
{
    FeatureOutput output;
    output.imageWidth = 1280;
    output.imageHeight = 960;
    output.keypoints.reserve(static_cast<std::size_t>(keypointCount));
    output.scores.reserve(static_cast<std::size_t>(keypointCount));
    for (int index = 0; index < keypointCount; ++index)
    {
        const float x = static_cast<float>((index * 37) % output.imageWidth);
        const float y = static_cast<float>((index * 53) % output.imageHeight);
        const float score = 1.0f - 0.5f *
            static_cast<float>(index % 100) / 100.0f;
        output.keypoints.emplace_back(
            cv::Point2f(x, y),
            4.0f,
            static_cast<float>(index % 360),
            score);
        output.scores.push_back(score);
    }
    output.descriptors = torch::zeros(
        {keypointCount, descriptorDimension},
        torch::kFloat32);
    return output;
}

void benchmarkFeatureReads(
    const BenchmarkOptions &options,
    const QString &featurePath)
{
    std::size_t checksum = 0;
    auto start = std::chrono::steady_clock::now();
    for (int pair = 0; pair < options.pairCount; ++pair)
    {
        for (int side = 0; side < 2; ++side)
        {
            QString imageName;
            xjw::feature_extractors::FeatureData feature;
            if (!FeatureFileIO::readData(featurePath, imageName, feature))
            {
                throw std::runtime_error("full feature read failed");
            }
            checksum += static_cast<std::size_t>(
                feature.size() + feature.descriptorDim());
        }
    }
    const double fullReadMs = elapsedMilliseconds(start);

    start = std::chrono::steady_clock::now();
    for (int pair = 0; pair < options.pairCount; ++pair)
    {
        for (int side = 0; side < 2; ++side)
        {
            QString imageName;
            xjw::feature_extractors::FeatureData feature;
            if (!FeatureFileIO::readGeometryData(
                    featurePath, imageName, feature))
            {
                throw std::runtime_error("geometry feature read failed");
            }
            checksum += static_cast<std::size_t>(feature.size());
        }
    }
    const double geometryReadMs = elapsedMilliseconds(start);

    std::cout << "feature_read"
              << ",keypoints=" << options.keypoints
              << ",descriptor_dim=" << options.descriptorDimension
              << ",pair_count=" << options.pairCount
              << ",full_ms=" << fullReadMs
              << ",geometry_ms=" << geometryReadMs
              << ",speedup=" << (geometryReadMs > 0.0
                    ? fullReadMs / geometryReadMs
                    : 0.0)
              << ",checksum=" << checksum
              << '\n';
}

void benchmarkGeometry(
    const BenchmarkOptions &options,
    const xjw::Camera &leftCamera,
    const xjw::Camera &rightCamera)
{
    xjw::feature_extractors::FeatureData left;
    xjw::feature_extractors::FeatureData right;
    std::vector<cv::DMatch> cvMatches;
    left.keypoints.reserve(static_cast<std::size_t>(options.geometryMatches));
    right.keypoints.reserve(static_cast<std::size_t>(options.geometryMatches));
    cvMatches.reserve(static_cast<std::size_t>(options.geometryMatches));
    for (int index = 0; index < options.geometryMatches; ++index)
    {
        const std::array<double, 3> world = syntheticWorldPoint(index);
        const std::array<double, 2> leftPixel = project(leftCamera, world);
        std::array<double, 2> rightPixel = project(rightCamera, world);
        if (index % 10 == 0)
        {
            rightPixel[0] += 80.0 + static_cast<double>(index % 7);
            rightPixel[1] -= 45.0;
        }
        left.keypoints.emplace_back(
            cv::Point2f(
                static_cast<float>(leftPixel[0]),
                static_cast<float>(leftPixel[1])),
            1.0f);
        right.keypoints.emplace_back(
            cv::Point2f(
                static_cast<float>(rightPixel[0]),
                static_cast<float>(rightPixel[1])),
            1.0f);
        cvMatches.emplace_back(index, index, 0.0f);
    }

    const xjw::feature_match::MatchResult matches =
        xjw::feature_match::MatchResult::fromCvMatches(
            cvMatches,
            options.geometryMatches,
            options.geometryMatches,
            "benchmark");
    xjw::feature_match::OutlierFilterConfig filter;
    filter.method =
        xjw::feature_match::OutlierMethod::FundamentalUsacMagsac;
    filter.reprojThreshold = 1.5;
    filter.minInliers = 20;
    filter.maxIters = 10000;
    filter.randomSeed = 0;

    std::size_t inlierChecksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int pair = 0; pair < options.pairCount; ++pair)
    {
        int inlierCount = 0;
        const auto result = xjw::feature_match::MatchGeometryFilter::filter(
            matches,
            left.keypoints,
            right.keypoints,
            filter,
            &inlierCount);
        inlierChecksum += static_cast<std::size_t>(
            inlierCount + result.numMatches);
    }

    std::cout << "usac_magsac"
              << ",matches_per_pair=" << options.geometryMatches
              << ",pair_count=" << options.pairCount
              << ",total_ms=" << elapsedMilliseconds(start)
              << ",checksum=" << inlierChecksum
              << '\n';
}

void benchmarkPnp(
    const BenchmarkOptions &options,
    const xjw::Camera &camera)
{
    std::vector<std::array<double, 3>> worldPoints;
    std::vector<std::array<double, 2>> imagePoints;
    worldPoints.reserve(static_cast<std::size_t>(options.geometryMatches));
    imagePoints.reserve(static_cast<std::size_t>(options.geometryMatches));
    for (int index = 0; index < options.geometryMatches; ++index)
    {
        const std::array<double, 3> world = syntheticWorldPoint(index);
        std::array<double, 2> pixel = project(camera, world);
        if (index % 20 == 0)
        {
            pixel[0] += 100.0;
        }
        worldPoints.push_back(world);
        imagePoints.push_back(pixel);
    }

    xjw::PnpOptions pnpOptions;
    pnpOptions.maxIterations = 1000;
    pnpOptions.minNumInliers = 10;
    pnpOptions.minInlierRatio = 0.25;
    pnpOptions.ransacSeed = 0;

    int inlierChecksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int image = 0; image < options.pnpImages; ++image)
    {
        const xjw::PnpResult result = xjw::PnpSolver::solveWithCamera(
            worldPoints,
            imagePoints,
            camera,
            pnpOptions);
        inlierChecksum += result.numInliers;
    }

    std::cout << "pnp_ransac"
              << ",correspondences=" << options.geometryMatches
              << ",image_count=" << options.pnpImages
              << ",total_ms=" << elapsedMilliseconds(start)
              << ",checksum=" << inlierChecksum
              << '\n';
}

void benchmarkTriangulation(
    const BenchmarkOptions &options,
    const xjw::Camera &leftCamera,
    const xjw::Camera &rightCamera)
{
    std::vector<std::pair<double, double>> leftPixels;
    std::vector<std::pair<double, double>> rightPixels;
    leftPixels.reserve(
        static_cast<std::size_t>(options.triangulationPoints));
    rightPixels.reserve(
        static_cast<std::size_t>(options.triangulationPoints));
    for (int index = 0; index < options.triangulationPoints; ++index)
    {
        const std::array<double, 3> world = syntheticWorldPoint(index);
        const std::array<double, 2> left = project(leftCamera, world);
        const std::array<double, 2> right = project(rightCamera, world);
        leftPixels.emplace_back(left[0], left[1]);
        rightPixels.emplace_back(right[0], right[1]);
    }

    int serialValid = 0;
    auto start = std::chrono::steady_clock::now();
    for (int index = 0; index < options.triangulationPoints; ++index)
    {
        const xjw::Intersection::Result result =
            xjw::Intersection::intersectPair(
                leftCamera,
                leftPixels[static_cast<std::size_t>(index)].first,
                leftPixels[static_cast<std::size_t>(index)].second,
                rightCamera,
                rightPixels[static_cast<std::size_t>(index)].first,
                rightPixels[static_cast<std::size_t>(index)].second);
        serialValid += result.valid ? 1 : 0;
    }
    const double serialMs = elapsedMilliseconds(start);

    start = std::chrono::steady_clock::now();
    const std::vector<xjw::Intersection::Result> parallelResults =
        xjw::Intersection::intersectBatch(
            leftCamera,
            leftPixels,
            rightCamera,
            rightPixels,
            0);
    const double parallelMs = elapsedMilliseconds(start);
    const int parallelValid = static_cast<int>(std::count_if(
        parallelResults.begin(),
        parallelResults.end(),
        [](const xjw::Intersection::Result &result)
        {
            return result.valid;
        }));

    std::cout << "triangulation"
              << ",point_count=" << options.triangulationPoints
              << ",serial_ms=" << serialMs
              << ",parallel_ms=" << parallelMs
              << ",speedup=" << (parallelMs > 0.0
                    ? serialMs / parallelMs
                    : 0.0)
              << ",serial_valid=" << serialValid
              << ",parallel_valid=" << parallelValid
              << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication application(argc, argv);
    const BenchmarkOptions options = parseOptions(argc, argv);
    QTemporaryDir temporaryDirectory;
    if (!temporaryDirectory.isValid())
    {
        std::cerr << "cannot create temporary benchmark directory\n";
        return 2;
    }

    const QString featurePath =
        QDir(temporaryDirectory.path()).filePath(
            QStringLiteral("benchmark.sift"));
    const FeatureOutput featureOutput = makeFeatureOutput(
        options.keypoints,
        options.descriptorDimension);
    if (!FeatureFileIO::write(
            featurePath,
            QStringLiteral("benchmark.png"),
            featureOutput,
            "sift"))
    {
        std::cerr << "cannot write temporary feature file\n";
        return 3;
    }

    const xjw::Camera leftCamera = makeCamera({{-0.35, 0.0, 0.0}});
    const xjw::Camera rightCamera = makeCamera({{0.35, 0.0, 0.0}});
    try
    {
        benchmarkFeatureReads(options, featurePath);
        benchmarkGeometry(options, leftCamera, rightCamera);
        benchmarkPnp(options, rightCamera);
        benchmarkTriangulation(options, leftCamera, rightCamera);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 4;
    }

    return 0;
}

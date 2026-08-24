#pragma once

#include <QJsonObject>
#include <QString>

#include <functional>
#include <atomic>

namespace xjw
{

    struct RpcStereoDemOptions
    {
        int maximumFeatures = 20000;
        double descriptorRatio = 0.75;
        double fundamentalRansacThresholdPixels = 1.5;
        double maximumReprojectionErrorPixels = 1.5;
        int minimumAcceptedPoints = 80;
        double gridResolutionMeters = 2.0;
        int holeFillIterations = 30;
    };

    class RpcStereoDemGenerator
    {
    public:
        using ProgressCallback = std::function<void(const QString&, int)>;

        static bool generate(const QString& leftImagePath,
                             const QString& rightImagePath,
                             const QString& outputDirectory,
                             const RpcStereoDemOptions& options,
                             QJsonObject* result,
                             QString* errorMessage = nullptr,
                             const ProgressCallback& progress = {},
                             const std::atomic_bool* cancelFlag = nullptr);
    };

} // namespace xjw

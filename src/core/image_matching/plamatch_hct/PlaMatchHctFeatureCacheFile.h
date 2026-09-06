#pragma once

#include "../FeatureSet.h"

#include <QString>

#include <memory>

namespace xjw::image_matching
{

    class PlaMatchHctFeatureCacheFile
    {
    public:
        static QString filePathForImage(const QString& directory, const QString& imagePath);

        static bool write(const QString& filePath,
                          const QString& imagePath,
                          const QString& producerSignature,
                          const FeatureSet& features,
                          QString* errorMessage = nullptr);

        static std::shared_ptr<FeatureSet> read(const QString& filePath,
                                                const QString& imagePath,
                                                const QString& producerSignature,
                                                QString* missReason = nullptr);
    };

} // namespace xjw::image_matching

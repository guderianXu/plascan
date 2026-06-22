#include "LayerFeatureLoader.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

// 避免 Qt 宏污染 LibTorch/ATen 头文件。
#ifdef slots
  #undef slots
  #define NEED_RESTORE_SLOTS
#endif
#ifdef signals
  #undef signals
  #define NEED_RESTORE_SIGNALS
#endif
#ifdef emit
  #undef emit
  #define NEED_RESTORE_EMIT
#endif

#include "FeatureOutput.h"
#include "FeatureFileIO.h"

#ifdef NEED_RESTORE_SLOTS
  #define slots Q_SLOTS
  #undef NEED_RESTORE_SLOTS
#endif
#ifdef NEED_RESTORE_SIGNALS
  #define signals Q_SIGNALS
  #undef NEED_RESTORE_SIGNALS
#endif
#ifdef NEED_RESTORE_EMIT
  #define emit Q_EMIT
  #undef NEED_RESTORE_EMIT
#endif

#include "ProjectIO.h"

#include <QDebug>

namespace xjw::gui::views
{

std::vector<cv::KeyPoint> loadFeatureKeypointsFromFile(const QString &featurePath)
{
    if (featurePath.isEmpty())
    {
        return {};
    }

    QString imageName;
    FeatureOutput output;
    if (!FeatureFileIO::read(featurePath, imageName, output))
    {
        qWarning() << "Failed to read feature file:" << featurePath;
        return {};
    }

    for (size_t i = 0; i < output.keypoints.size() && i < output.scores.size(); ++i)
    {
        output.keypoints[i].response = output.scores[i];
    }

    return std::move(output.keypoints);
}

std::vector<cv::KeyPoint> loadFeatureKeypointsForImage(const QString &plascanPath,
                                                       const QString &imagePath)
{
    const QString featurePath = ProjectIO::findFeatureForImage(plascanPath, imagePath);
    if (featurePath.isEmpty())
    {
        qDebug() << "LayerRenderer: No feature file found for" << imagePath;
        return {};
    }

    return loadFeatureKeypointsFromFile(featurePath);
}

} // namespace xjw::gui::views

#ifdef _MSC_VER
#pragma warning(pop)
#endif

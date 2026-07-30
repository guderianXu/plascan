#include "DenseSparseCloudPreparation.h"

#include "SparseCloudPreprocessor.h"
#include "io/PathIO.h"

#include <QFile>

namespace xjw::gui::project
{

xjw::mvs::SparseCloud prepareDenseSparseCloud(
    const QString &sparseCloudPath,
    const std::vector<xjw::mvs::CameraView> &views)
{
    xjw::mvs::SparseCloud sparseCloud;
    if (sparseCloudPath.trimmed().isEmpty() || !QFile::exists(sparseCloudPath))
    {
        return sparseCloud;
    }

    // 预处理与 PatchMatch GPU 初始化分离，避免交互工作流中出现 CPU/GPU 资源竞争。
    xjw::mvs::SparseCloudPreprocessor preprocessor(plapoint::ProcessingDevice::CPU);
    xjw::mvs::PreprocessResult result;
    std::string errorMessage;
    if (preprocessor.run(xjw::common::io::toUtf8Path(sparseCloudPath),
                         views,
                         result,
                         &errorMessage))
    {
        sparseCloud = std::move(result.cloud);
    }
    return sparseCloud;
}

} // namespace xjw::gui::project

#include "PointCloudInputPreparation.h"

#include "SparseCloudPreprocessor.h"
#include "io/PathIO.h"

#include <QFileInfo>

#include <plapoint/filters/preprocessing.h>

namespace xjw::core::project
{

PointCloudInputPreparationResult preparePointCloudInput(
    const QString &sparseCloudPath,
    const std::vector<xjw::mvs::CameraView> &views,
    plapoint::ProcessingDevice processingDevice)
{
    PointCloudInputPreparationResult prepared;
    if (sparseCloudPath.trimmed().isEmpty() || !QFileInfo::exists(sparseCloudPath))
    {
        prepared.errorMessage = QStringLiteral("正式空三稀疏点云不存在：%1")
                                    .arg(sparseCloudPath);
        return prepared;
    }

    xjw::mvs::SparseCloudPreprocessor preprocessor(processingDevice);
    xjw::mvs::PreprocessResult result;
    std::string error_message;
    if (!preprocessor.run(xjw::common::io::toUtf8Path(sparseCloudPath),
                          views,
                          result,
                          &error_message))
    {
        prepared.errorMessage = error_message.empty()
            ? QStringLiteral("稀疏点云预处理失败")
            : QString::fromUtf8(error_message.c_str());
        return prepared;
    }
    if (result.cloud.points.empty())
    {
        prepared.errorMessage = QStringLiteral("稀疏点云预处理后没有可用于 MVS 的点");
        return prepared;
    }

    prepared.ok = true;
    prepared.cloud = std::move(result.cloud);
    return prepared;
}

} // namespace xjw::core::project

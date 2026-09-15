#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    bool MvsPipelineService::ensurePreparedRasterArtifact(int frameIndex,
                                                          MvsPreparedRasterArtifact* artifact,
                                                          QString* errorMessage)
    {
        if (!artifact || frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS prepared raster 帧参数无效");
            }
            return false;
        }
        if (_workspaceManifestPath.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS workspace 路径为空，无法持久化相机匹配的工作栅格");
            }
            return false;
        }

        std::lock_guard<std::mutex> lock(_preparedRasterArtifactsMutex);
        if (_preparedRasterArtifacts.size() != _views.size())
        {
            _preparedRasterArtifacts.assign(_views.size(), MvsPreparedRasterArtifact{});
        }
        MvsPreparedRasterArtifact& cached = _preparedRasterArtifacts[static_cast<std::size_t>(frameIndex)];
        if (!cached.imagePath.empty() && !cached.validMaskPath.empty() && cached.camera.isValid() &&
            QFileInfo::exists(xjw::common::io::fromUtf8Path(cached.imagePath)) &&
            QFileInfo::exists(xjw::common::io::fromUtf8Path(cached.validMaskPath)))
        {
            *artifact = cached;
            if (errorMessage)
            {
                errorMessage->clear();
            }
            return true;
        }

        const CameraView& view = _views[static_cast<std::size_t>(frameIndex)];
        if (!view.preparedImagePath.empty() && !view.preparedValidMaskPath.empty() && view.camera.isValid())
        {
            const cv::Mat prepared_valid_mask =
                xjw::common::io::readImage(view.preparedValidMaskPath, cv::IMREAD_GRAYSCALE);
            MvsPreparedRasterArtifact saved;
            std::string save_error;
            if (prepared_valid_mask.empty() ||
                !saveMvsPreparedRasterArtifact(
                    mvsRasterPath(view),
                    view.camera,
                    prepared_valid_mask,
                    xjw::common::io::toUtf8Path(QFileInfo(_workspaceManifestPath).absolutePath()),
                    frameIndex,
                    &saved,
                    &save_error))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("复用第 %1 帧已有 MVS prepared raster 失败：%2")
                                        .arg(frameIndex)
                                        .arg(QString::fromStdString(save_error));
                }
                return false;
            }
            cached = saved;
            *artifact = saved;
            if (errorMessage)
            {
                errorMessage->clear();
            }
            return true;
        }

        if (!_imageCache)
        {
            // The recovered scene owns image decoding and does not initialize the
            // legacy per-frame provider. Persist the same source mask passed into
            // recovered CUDA so replay cannot silently widen the processed domain.
            RecoveredSourceMask recovered_source_mask;
            std::string recovered_mask_error;
            if (!prepareRecoveredSourceMask(view, &recovered_source_mask, &recovered_mask_error))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("准备第 %1 帧 recovered 蒙版失败：%2")
                                        .arg(frameIndex)
                                        .arg(QString::fromStdString(recovered_mask_error));
                }
                return false;
            }
            cv::Mat recovered_valid_mask;
            if (!recovered_source_mask.bytes.empty())
            {
                recovered_valid_mask =
                    cv::Mat(view.imageHeight, view.imageWidth, CV_8U, recovered_source_mask.bytes.data()).clone();
            }
            MvsPreparedRasterArtifact saved;
            std::string save_error;
            if (!saveMvsPreparedRasterArtifact(
                    mvsRasterPath(view),
                    view.camera,
                    recovered_valid_mask,
                    xjw::common::io::toUtf8Path(QFileInfo(_workspaceManifestPath).absolutePath()),
                    frameIndex,
                    &saved,
                    &save_error))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("保存第 %1 帧 recovered prepared raster 失败：%2")
                                        .arg(frameIndex)
                                        .arg(QString::fromStdString(save_error));
                }
                return false;
            }
            cached = saved;
            *artifact = saved;
            if (errorMessage)
            {
                errorMessage->clear();
            }
            return true;
        }

        std::string load_error;
        MvsImageCache::ImageLease image_lease = acquireImageFrame(frameIndex, &load_error);
        if (!image_lease)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法获取第 %1 帧以保存 prepared raster：%2")
                                    .arg(frameIndex)
                                    .arg(QString::fromStdString(load_error));
            }
            return false;
        }

        MvsPreparedRasterArtifact saved;
        std::string save_error;
        if (!saveMvsPreparedRasterArtifact(
                mvsRasterPath(view),
                view.camera,
                image_lease->validMask,
                xjw::common::io::toUtf8Path(QFileInfo(_workspaceManifestPath).absolutePath()),
                frameIndex,
                &saved,
                &save_error))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("保存第 %1 帧 MVS prepared raster 失败：%2")
                                    .arg(frameIndex)
                                    .arg(QString::fromStdString(save_error));
            }
            return false;
        }

        cached = saved;
        *artifact = saved;
        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
    }
} // namespace xjw::mvs

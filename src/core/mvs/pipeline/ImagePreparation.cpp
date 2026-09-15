#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    // =============================================================================
    // 预加载所有图像到灰度缓存，避免逐帧重复从磁盘读取
    // =============================================================================
    bool MvsPipelineService::probeImageMetadata(QString* errorMessage)
    {
        for (int index = 0; index < static_cast<int>(_views.size()); ++index)
        {
            if (_cancelled.load(std::memory_order_relaxed))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("影像头部探测已取消");
                }
                return false;
            }

            MvsImageMetadata metadata;
            std::string probeError;
            const std::string& raster_path = mvsRasterPath(_views[static_cast<std::size_t>(index)]);
            if (!probeMvsImageMetadata(raster_path, &metadata, &probeError))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法读取影像 %1 的头部尺寸：%2")
                                        .arg(index)
                                        .arg(QString::fromStdString(probeError));
                }
                return false;
            }

            CameraView& view = _views[static_cast<std::size_t>(index)];
            if ((view.imageWidth > 0 && view.imageWidth != metadata.width) ||
                (view.imageHeight > 0 && view.imageHeight != metadata.height))
            {
                LOG_WARN(QStringLiteral("[MVS][内存规划] 视图 %1 声明尺寸 %2x%3 与影像头 %4x%5 不一致，"
                                        "以内嵌元数据为准")
                             .arg(index)
                             .arg(view.imageWidth)
                             .arg(view.imageHeight)
                             .arg(metadata.width)
                             .arg(metadata.height));
            }
            view.imageWidth = metadata.width;
            view.imageHeight = metadata.height;
        }

        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
    }

    bool MvsPipelineService::loadMvsImageFrame(int frameIndex,
                                               const std::atomic_bool* cancelFlag,
                                               MvsImageFrame* frame,
                                               std::string* errorMessage)
    {
        if (!frame || frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()))
        {
            if (errorMessage)
            {
                *errorMessage = "MVS image loader received an invalid frame";
            }
            return false;
        }

        const auto cancelled = [cancelFlag]() { return cancelFlag && cancelFlag->load(std::memory_order_relaxed); };
        const auto stopIfCancelled = [&](const char* message)
        {
            if (!cancelled())
            {
                return false;
            }
            if (errorMessage)
            {
                *errorMessage = message;
            }
            return true;
        };
        if (stopIfCancelled("MVS image load cancelled"))
        {
            return false;
        }

        const CameraView& view = _views[static_cast<std::size_t>(frameIndex)];
        frame->gray = xjw::common::io::readImage(mvsRasterPath(view), cv::IMREAD_GRAYSCALE);
        if (frame->gray.empty())
        {
            if (errorMessage)
            {
                *errorMessage = "无法解码影像: " + mvsRasterPath(view);
            }
            return false;
        }
        if (stopIfCancelled("MVS image load cancelled"))
        {
            return false;
        }

        if (!view.preparedValidMaskPath.empty())
        {
            cv::Mat prepared_valid_mask = xjw::common::io::readImage(view.preparedValidMaskPath, cv::IMREAD_GRAYSCALE);
            if (stopIfCancelled("MVS prepared valid mask load cancelled"))
            {
                return false;
            }
            if (prepared_valid_mask.empty())
            {
                if (errorMessage)
                {
                    *errorMessage = "无法读取 MVS prepared valid mask: " + view.preparedValidMaskPath;
                }
                return false;
            }
            if (prepared_valid_mask.size() != frame->gray.size())
            {
                if (errorMessage)
                {
                    *errorMessage = "MVS prepared valid mask 与 prepared raster 尺寸不一致";
                }
                return false;
            }
            cv::threshold(prepared_valid_mask, frame->validMask, 0.0, 255.0, cv::THRESH_BINARY);
            const std::string& prepared_mask_source = view.preparedValidMaskSource;
            frame->validMaskSource = prepared_mask_source == "project"
                                         ? "project"
                                         : (prepared_mask_source == "content" ? "content" : "technical");
        }
        else if (!view.validRegionMaskPath.empty())
        {
            const cv::Mat projectMask = xjw::common::io::readImage(view.validRegionMaskPath, cv::IMREAD_GRAYSCALE);
            if (stopIfCancelled("MVS project mask load cancelled"))
            {
                return false;
            }
            if (!projectMask.empty())
            {
                frame->validMask = projectMaskToValidMask(projectMask, frame->gray.size());
                if (stopIfCancelled("MVS project mask preparation cancelled"))
                {
                    return false;
                }
                bool contentRefined = false;
                float retainedRatio = 1.0f;
                if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
                {
                    frame->validMask =
                        refineOrbitalProjectValidMask(frame->gray, frame->validMask, &contentRefined, &retainedRatio);
                }
                if (stopIfCancelled("MVS project mask refinement cancelled"))
                {
                    return false;
                }
                frame->validMaskSource = "project";
                if (contentRefined)
                {
                    LOG_DEBUG(QStringLiteral("[MVS][图像缓存] 视图 %1 暗背景蒙版细化后保留 %2%")
                                  .arg(frameIndex)
                                  .arg(100.0 * static_cast<double>(retainedRatio), 0, 'f', 1));
                }
            }
            else
            {
                LOG_WARN(QStringLiteral("[MVS][图像缓存] 视图 %1 项目蒙版读取失败，"
                                        "回退内容区域检测: %2")
                             .arg(frameIndex)
                             .arg(QString::fromStdString(view.validRegionMaskPath)));
            }
        }

        if (frame->validMaskSource.empty())
        {
            float coverage = 0.0f;
            double otsuThreshold = 0.0;
            int adaptiveThreshold = 0;
            frame->validMask = buildContentMask(frame->gray, &coverage, &otsuThreshold, &adaptiveThreshold);
            frame->validMaskSource = "content";
            if (stopIfCancelled("MVS content mask preparation cancelled"))
            {
                return false;
            }
            LOG_DEBUG("[MVS][图像缓存] 视图 %d 内容掩码 coverage=%.1f%% "
                      "Otsu=%.0f adaptive_threshold=%d",
                      frameIndex,
                      coverage * 100.0f,
                      otsuThreshold,
                      adaptiveThreshold);
        }

        frame->gray = normalizeMvsPhotometry(frame->gray, _effectiveSceneProfile);
        if (stopIfCancelled("MVS image preprocessing cancelled"))
        {
            return false;
        }

        std::string preparationError;
        cv::Mat prepared_valid_mask;
        if (!prepareMvsImageAndMask(frame->gray,
                                    frame->validMask,
                                    view.camera,
                                    &frame->preparedGray,
                                    &prepared_valid_mask,
                                    &frame->preparedCamera,
                                    &preparationError))
        {
            if (errorMessage)
            {
                *errorMessage = "影像 MVS 预处理失败: " + preparationError;
            }
            return false;
        }
        frame->validMask = std::move(prepared_valid_mask);
        if (stopIfCancelled("MVS image preparation cancelled"))
        {
            frame->preparedGray.release();
            return false;
        }

        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
    }

    bool MvsPipelineService::initializeImageProvider(const MvsPipelineMemoryPolicyDecision& decision,
                                                     QString* errorMessage)
    {
        if (decision.imageStrategy == MvsImageCacheStrategy::Insufficient || decision.imageCacheCapacity == 0)
        {
            if (errorMessage)
            {
                const std::string_view strategy = mvsImageCacheStrategyName(decision.imageStrategy);
                *errorMessage =
                    QStringLiteral("MVS 内存不足: required=%1 GiB available=%2 GiB strategy=%3。"
                                   "visibility=%4 GiB (bitset=%5 GiB visible_index=%6 GiB "
                                   "pairs=%7 GiB nominated=%8 GiB adjacency=%9 GiB "
                                   "pair_bound=%10 saturated=%11)。"
                                   "请降低 gpuFrameWorkerCount/cpuFrameWorkerCount、numSourceViews、"
                                   "输入分辨率或质量档位，或减少输入视图/稀疏点规模后重试")
                        .arg(bytesToGiB(decision.requiredBytes), 0, 'f', 2)
                        .arg(bytesToGiB(decision.availableBytes), 0, 'f', 2)
                        .arg(QString::fromLatin1(strategy.data(), static_cast<int>(strategy.size())))
                        .arg(bytesToGiB(decision.estimate.visibility.totalBytes), 0, 'f', 2)
                        .arg(bytesToGiB(decision.estimate.visibility.visibilityBitsetBytes), 0, 'f', 2)
                        .arg(bytesToGiB(decision.estimate.visibility.visibleIndexBytes), 0, 'f', 2)
                        .arg(bytesToGiB(decision.estimate.visibility.pairBytes), 0, 'f', 2)
                        .arg(bytesToGiB(decision.estimate.visibility.nominatedPeerBytes), 0, 'f', 2)
                        .arg(bytesToGiB(decision.estimate.visibility.adjacencyBytes), 0, 'f', 2)
                        .arg(decision.estimate.visibility.candidatePairUpperBound)
                        .arg(decision.estimate.visibility.saturated ? QStringLiteral("true") : QStringLiteral("false"));
            }
            return false;
        }

        _pipelineMemoryDecision = decision;
        _imageCache = std::make_unique<MvsImageCache>(
            _views.size(),
            decision.imageCacheCapacity,
            [this](int frameIndex, const std::atomic_bool* cancelFlag, MvsImageFrame* frame, std::string* loaderError)
            { return loadMvsImageFrame(frameIndex, cancelFlag, frame, loaderError); });
        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
    }

    bool MvsPipelineService::preloadImages(QString* errorMessage)
    {
        if (!_imageCache)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS 图像 provider 尚未初始化");
            }
            return false;
        }
        if (_pipelineMemoryDecision.imageStrategy != MvsImageCacheStrategy::Eager)
        {
            return true;
        }

        const auto preloadStart = Clock::now();
        std::string preloadError;
        const int workerCount =
            preloadImagesWorkerCount(static_cast<int>(_views.size()), resolvedTotalCpuThreadBudget(_config));
        if (!_imageCache->preloadAll(workerCount, &_cancelled, &preloadError))
        {
            if (errorMessage)
            {
                *errorMessage =
                    QStringLiteral("MVS 影像 eager 预加载失败：%1").arg(QString::fromStdString(preloadError));
            }
            return false;
        }

        LOG_INFO(QStringLiteral("[MVS][图像缓存] eager 预加载完成: resident=%1/%2 "
                                "bytes=%3 GiB elapsed=%4 ms")
                     .arg(_imageCache->residentCount())
                     .arg(_imageCache->frameCount())
                     .arg(bytesToGiB(_imageCache->residentBytes()), 0, 'f', 2)
                     .arg(elapsedMs(preloadStart, Clock::now()), 0, 'f', 1));
        return true;
    }

    MvsImageCache::ImageLease MvsPipelineService::acquireImageFrame(int frameIndex, std::string* errorMessage)
    {
        if (!_imageCache)
        {
            if (errorMessage)
            {
                *errorMessage = "MVS image provider is not initialized";
            }
            return {};
        }
        return _imageCache->acquire(frameIndex, &_cancelled, errorMessage);
    }
} // namespace xjw::mvs

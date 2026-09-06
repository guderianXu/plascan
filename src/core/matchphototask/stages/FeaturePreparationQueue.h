#pragma once

#include <QString>

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace xjw
{
    namespace image_matching
    {
        struct FeatureSet;
    }

    namespace matchphotos
    {

        struct FeaturePreparationRequest
        {
            int index = -1;
            QString imagePath;
            QString featurePath;
        };

        // CPU 预取阶段的产物。inputImage 的通道数由算法能力决定：SIFT 使用灰度，
        // LoMa-R 使用 BGR；队列仍只保留一份缩放后的输入，避免重复驻留原图。
        struct PreparedFeatureImage
        {
            int index = -1;
            QString imagePath;
            QString featurePath;
            QString maskPath;
            QString errorMessage;

            bool reused = false;
            int effectiveKeypointLimit = 0;
            int originalWidth = 0;
            int originalHeight = 0;
            double resizeScale = 1.0;
            double coordinateScale = 1.0;
            double coordinateOffsetX = 0.0;
            double coordinateOffsetY = 0.0;

            cv::Mat inputImage;
            cv::Mat mask;
            std::shared_ptr<image_matching::FeatureSet> cachedFeatures;
            QString cacheMissReason;

            std::int64_t imageReadMs = 0;
            std::int64_t imageResizeMs = 0;
            std::int64_t maskReadMs = 0;
            std::int64_t preparationMs = 0;
        };

        using FeaturePrepareFunction = std::function<PreparedFeatureImage(const FeaturePreparationRequest&)>;

        // 多生产者、有界、保序队列。多个 worker 并行完成 CPU 解码、缩放和蒙版准备；
        // 多个提取流水线 worker 可安全消费，但取图顺序仍与请求顺序一致。CUDA SIFT
        // 后端负责保护第三方设备全局状态，CPU 后处理可与下一张影像的 GPU 段重叠。
        class FeaturePreparationQueue
        {
        public:
            FeaturePreparationQueue(std::vector<FeaturePreparationRequest> requests,
                                    int capacity,
                                    std::atomic_bool* cancelFlag,
                                    FeaturePrepareFunction prepareFunction,
                                    int workerCount = 1);
            ~FeaturePreparationQueue();

            FeaturePreparationQueue(const FeaturePreparationQueue&) = delete;
            FeaturePreparationQueue& operator=(const FeaturePreparationQueue&) = delete;

            bool take(PreparedFeatureImage* prepared);
            void stop();

            int capacity() const;
            int workerCount() const;
            int peakBufferedCount() const;

        private:
            bool cancellationRequested() const;
            void produce();

            std::vector<FeaturePreparationRequest> _requests;
            int _capacity = 1;
            int _workerCount = 1;
            std::atomic_bool* _cancelFlag = nullptr;
            FeaturePrepareFunction _prepareFunction;

            mutable std::mutex _mutex;
            std::condition_variable _notEmpty;
            std::condition_variable _notFull;
            std::map<std::size_t, PreparedFeatureImage> _preparedByPosition;
            std::exception_ptr _producerFailure;
            std::size_t _nextRequestPosition = 0;
            std::size_t _nextTakePosition = 0;
            int _activeProducerCount = 0;
            bool _stopRequested = false;
            bool _finished = false;
            int _peakBufferedCount = 0;
            std::vector<std::thread> _producers;
        };

    } // namespace matchphotos
} // namespace xjw

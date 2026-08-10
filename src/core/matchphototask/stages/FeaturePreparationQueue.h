#pragma once

#include <QString>

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace xjw
{
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
    int existingKeypointCount = 0;
    int effectiveKeypointLimit = 0;
    int originalWidth = 0;
    int originalHeight = 0;
    double resizeScale = 1.0;

    cv::Mat inputImage;
    cv::Mat mask;

    std::int64_t imageReadMs = 0;
    std::int64_t imageResizeMs = 0;
    std::int64_t maskReadMs = 0;
    std::int64_t preparationMs = 0;
};

using FeaturePrepareFunction =
    std::function<PreparedFeatureImage(const FeaturePreparationRequest &)>;

// 单生产者、有界、保序队列。它只重叠 CPU 图像准备与当前影像的特征提取，
// 不会在多个线程中同时调用 CUDA SIFT。
class FeaturePreparationQueue
{
public:
    FeaturePreparationQueue(std::vector<FeaturePreparationRequest> requests,
                            int capacity,
                            std::atomic_bool *cancelFlag,
                            FeaturePrepareFunction prepareFunction);
    ~FeaturePreparationQueue();

    FeaturePreparationQueue(const FeaturePreparationQueue &) = delete;
    FeaturePreparationQueue &operator=(const FeaturePreparationQueue &) = delete;

    bool take(PreparedFeatureImage *prepared);
    void stop();

    int capacity() const;
    int peakBufferedCount() const;

private:
    bool cancellationRequested() const;
    void produce();

    std::vector<FeaturePreparationRequest> _requests;
    int _capacity = 1;
    std::atomic_bool *_cancelFlag = nullptr;
    FeaturePrepareFunction _prepareFunction;

    mutable std::mutex _mutex;
    std::condition_variable _notEmpty;
    std::condition_variable _notFull;
    std::deque<PreparedFeatureImage> _buffer;
    std::exception_ptr _producerFailure;
    bool _stopRequested = false;
    bool _finished = false;
    int _peakBufferedCount = 0;
    // 必须最后构造：线程入口会立即访问上面的全部同步状态。
    std::thread _producer;
};

} // namespace matchphotos
} // namespace xjw

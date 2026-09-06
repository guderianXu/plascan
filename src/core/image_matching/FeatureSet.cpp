#include "FeatureSet.h"

namespace xjw::image_matching
{

bool FeatureSet::empty() const
{
    return keypoints.empty();
}

int FeatureSet::size() const
{
    return static_cast<int>(keypoints.size());
}

int FeatureSet::descriptorDimension() const
{
    return descriptors.empty() ? 0 : descriptors.cols;
}

std::size_t FeatureSet::approximateBytes() const
{
    const std::size_t descriptorBytes = descriptors.empty()
        ? 0
        : descriptors.total() * descriptors.elemSize();
    const std::size_t payloadBytes = payload ? payload->approximateBytes() : 0;
    return keypoints.size() * sizeof(cv::KeyPoint) +
        scores.size() * sizeof(float) + descriptorBytes + payloadBytes;
}

bool FeatureSet::isConsistent() const
{
    if (keypoints.empty())
    {
        return descriptors.empty() && scores.empty();
    }
    const bool supportedDescriptorType =
        descriptors.type() == CV_32F || descriptors.type() == CV_8U;
    const bool baseConsistent = supportedDescriptorType &&
        descriptors.rows == size() &&
        scores.size() == keypoints.size() &&
        imageWidth > 0 && imageHeight > 0;
    return baseConsistent && (!payload || payload->isConsistent(keypoints.size()));
}

} // namespace xjw::image_matching

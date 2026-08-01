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
    return keypoints.size() * sizeof(cv::KeyPoint) +
        scores.size() * sizeof(float) + descriptorBytes;
}

bool FeatureSet::isConsistent() const
{
    if (keypoints.empty())
    {
        return descriptors.empty() && scores.empty();
    }
    return descriptors.type() == CV_32F &&
        descriptors.rows == size() &&
        scores.size() == keypoints.size() &&
        imageWidth > 0 && imageHeight > 0;
}

} // namespace xjw::image_matching

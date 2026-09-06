#include "ImageMatchingAlgorithm.h"

#include <stdexcept>

namespace xjw::image_matching
{

    FeatureSet IImageMatchingAlgorithm::extract(const ImageFeatureInput&) const
    {
        throw std::logic_error("the image matching algorithm does not expose reusable features");
    }

    MatchResult IImageMatchingAlgorithm::matchFeatures(const FeatureSet&, const FeatureSet&)
    {
        throw std::logic_error("the image matching algorithm does not accept reusable features");
    }

    std::vector<MatchResult> IImageMatchingAlgorithm::matchFeatureBatch(std::span<const FeaturePairInput>,
                                                                        const std::function<bool()>&,
                                                                        const BatchMatchProgressCallback&)
    {
        throw std::logic_error("the image matching algorithm does not support batch feature matching");
    }

    MatchResult IImageMatchingAlgorithm::matchImages(const cv::Mat&, const cv::Mat&)
    {
        throw std::logic_error("the image matching algorithm is not an end-to-end image matcher");
    }

} // namespace xjw::image_matching

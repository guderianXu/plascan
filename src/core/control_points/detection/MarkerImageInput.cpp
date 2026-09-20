#include "MarkerDetector.h"

#include <stdexcept>

namespace xjw::control_points
{
    void validateMarkerImages(const cv::Mat& image, const cv::Mat& mask)
    {
        if (image.dims != 2 || image.type() != CV_8UC1)
        {
            throw std::invalid_argument("Marker detection requires a two-dimensional CV_8UC1 grayscale image");
        }
        if (!mask.empty() && (mask.dims != 2 || mask.type() != CV_8UC1 || mask.size() != image.size()))
        {
            throw std::invalid_argument("Marker mask must be CV_8UC1 and match the source image dimensions");
        }
    }
} // namespace xjw::control_points

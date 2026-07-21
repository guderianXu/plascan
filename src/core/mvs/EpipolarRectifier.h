#pragma once

#include "MvsTypes.h"
#include <opencv2/core.hpp>
#include <string>

namespace xjw
{
namespace mvs
{

class EpipolarRectifier
{
public:
    struct RectifiedPair
    {
        cv::Mat rectLeft;
        cv::Mat rectRight;
        Camera rectCamLeft;
        Camera rectCamRight;
        cv::Mat H1;
        cv::Mat H2;
        cv::Mat H1inv;
        cv::Mat H2inv;
        int origW = 0;
        int origH = 0;
        bool refIsRight = false;
        bool transposed = false;
    };

    static bool rectify(
        const cv::Mat &imgLeft,
        const cv::Mat &imgRight,
        const Camera &camLeft,
        const Camera &camRight,
        RectifiedPair &result,
        std::string *errorMsg = nullptr);

    static cv::Mat unrectifyDepth(
        const cv::Mat &rectifiedDepth,
        const RectifiedPair &pair,
        int origW, int origH);

    static cv::Mat unrectifyNearest(
        const cv::Mat &rectifiedArtifact,
        const RectifiedPair &pair,
        int origW,
        int origH);
};

} // namespace mvs
} // namespace xjw

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
        FramePinholeCamera rectCamLeft;
        FramePinholeCamera rectCamRight;
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
        const FramePinholeCamera &camLeft,
        const FramePinholeCamera &camRight,
        RectifiedPair &result,
        std::string *errorMsg = nullptr);

    static cv::Mat unrectifyDepth(
        const cv::Mat &rectifiedDepth,
        const RectifiedPair &pair,
        const FramePinholeCamera &originalReferenceCamera,
        int origW, int origH);

    /// Converts an axial positive-depth interval from the original reference
    /// camera into a conservative interval for the rectified reference camera.
    static bool rectifiedDepthRange(
        const FramePinholeCamera &originalReferenceCamera,
        const FramePinholeCamera &rectifiedReferenceCamera,
        int originalWidth,
        int originalHeight,
        float originalNear,
        float originalFar,
        float &rectifiedNear,
        float &rectifiedFar);

    static cv::Mat unrectifyNearest(
        const cv::Mat &rectifiedArtifact,
        const RectifiedPair &pair,
        int origW,
        int origH);
};

} // namespace mvs
} // namespace xjw

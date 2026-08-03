#include "PointCloudDomInternal.h"

#include <utility>

namespace xjw::point_cloud_dom_internal
{

void fillSmallGaps(cv::Mat *image, cv::Mat *mask, int iterations)
{
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        cv::Mat nextImage = image->clone();
        cv::Mat nextMask = mask->clone();
        int additions = 0;
        for (int row = 0; row < image->rows; ++row)
        {
            for (int col = 0; col < image->cols; ++col)
            {
                if (mask->at<uchar>(row, col) != 0)
                {
                    continue;
                }
                cv::Vec3d colorSum(0.0, 0.0, 0.0);
                int neighborCount = 0;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const int sourceRow = row + dy;
                        const int sourceCol = col + dx;
                        if ((dx == 0 && dy == 0) || sourceRow < 0 || sourceCol < 0
                            || sourceRow >= image->rows || sourceCol >= image->cols
                            || mask->at<uchar>(sourceRow, sourceCol) == 0)
                        {
                            continue;
                        }
                        colorSum += image->at<cv::Vec3b>(sourceRow, sourceCol);
                        ++neighborCount;
                    }
                }
                if (neighborCount >= 3)
                {
                    nextImage.at<cv::Vec3b>(row, col) = colorSum / neighborCount;
                    nextMask.at<uchar>(row, col) = 255;
                    ++additions;
                }
            }
        }
        *image = std::move(nextImage);
        *mask = std::move(nextMask);
        if (additions == 0)
        {
            break;
        }
    }
}

} // namespace xjw::point_cloud_dom_internal

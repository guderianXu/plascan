#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    cv::Mat MvsPipelineService::buildContentMask(const cv::Mat& gray,
                                                 float* coverage,
                                                 double* otsuThreshold,
                                                 int* adaptiveThreshold)
    {
        if (gray.empty())
        {
            if (coverage)
            {
                *coverage = 0.f;
            }
            return cv::Mat();
        }

        cv::Mat blurOrig;
        cv::GaussianBlur(gray, blurOrig, cv::Size(15, 15), 0);

        cv::Mat otsuBin;
        const double otsuThresh = cv::threshold(blurOrig, otsuBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        const int adaptiveThresh = std::max(8, static_cast<int>(otsuThresh * 0.3));

        cv::Mat mask = (blurOrig > adaptiveThresh);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        const int totalPx = gray.rows * gray.cols;
        const float maskCoverage =
            totalPx > 0 ? static_cast<float>(cv::countNonZero(mask)) / static_cast<float>(totalPx) : 0.f;

        if (coverage)
        {
            *coverage = maskCoverage;
        }
        if (otsuThreshold)
        {
            *otsuThreshold = otsuThresh;
        }
        if (adaptiveThreshold)
        {
            *adaptiveThreshold = adaptiveThresh;
        }

        if (maskCoverage >= kSkipContentMaskCoverage)
        {
            return cv::Mat();
        }

        return mask;
    }

    cv::Mat MvsPipelineService::projectMaskToValidMask(const cv::Mat& projectMask, cv::Size targetSize)
    {
        if (projectMask.empty() || targetSize.width <= 0 || targetSize.height <= 0)
        {
            return cv::Mat();
        }

        cv::Mat gray_mask;
        if (projectMask.channels() == 1)
        {
            gray_mask = projectMask;
        }
        else
        {
            cv::cvtColor(
                projectMask, gray_mask, projectMask.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
        }
        if (gray_mask.depth() != CV_8U)
        {
            gray_mask.convertTo(gray_mask, CV_8U);
        }
        if (gray_mask.size() != targetSize)
        {
            cv::resize(gray_mask, gray_mask, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
        }

        cv::Mat valid_mask;
        cv::compare(gray_mask, 0, valid_mask, cv::CMP_EQ);
        return valid_mask;
    }

    cv::Mat MvsPipelineService::refineOrbitalProjectValidMask(const cv::Mat& gray,
                                                              const cv::Mat& projectValidMask,
                                                              bool* refined,
                                                              float* retainedRatio)
    {
        if (refined)
        {
            *refined = false;
        }
        if (retainedRatio)
        {
            *retainedRatio = 1.0f;
        }
        if (gray.empty() || projectValidMask.empty() || gray.type() != CV_8UC1 || projectValidMask.type() != CV_8UC1 ||
            gray.size() != projectValidMask.size())
        {
            return projectValidMask.clone();
        }

        const int valid_pixels = cv::countNonZero(projectValidMask);
        if (valid_pixels <= 0)
        {
            return projectValidMask.clone();
        }

        cv::Mat excluded_mask;
        cv::compare(projectValidMask, 0, excluded_mask, cv::CMP_EQ);
        const int excluded_pixels = cv::countNonZero(excluded_mask);
        const double excluded_mean = excluded_pixels > 0 ? cv::mean(gray, excluded_mask)[0] : 255.0;
        if (excluded_mean > 45.0)
        {
            return projectValidMask.clone();
        }

        cv::Mat content_mask = buildContentMask(gray);
        if (content_mask.empty())
        {
            return projectValidMask.clone();
        }

        cv::Mat protected_interior;
        const cv::Mat boundary_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
        cv::erode(projectValidMask, protected_interior, boundary_kernel);

        cv::Mat content_excluded;
        cv::compare(content_mask, 0, content_excluded, cv::CMP_EQ);
        cv::Mat removable;
        cv::bitwise_and(protected_interior, content_excluded, removable);
        if (cv::countNonZero(removable) == 0)
        {
            return projectValidMask.clone();
        }

        cv::Mat candidate = projectValidMask.clone();
        candidate.setTo(0, removable);
        const int retained_pixels = cv::countNonZero(candidate);
        const float retained_ratio = static_cast<float>(retained_pixels) / static_cast<float>(valid_pixels);
        if (retainedRatio)
        {
            *retainedRatio = retained_ratio;
        }
        if (retained_ratio < 0.75f)
        {
            return projectValidMask.clone();
        }

        if (refined)
        {
            *refined = retained_pixels < valid_pixels;
        }
        return candidate;
    }
} // namespace xjw::mvs

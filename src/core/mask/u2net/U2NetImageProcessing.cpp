#include "U2NetImageProcessing.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace xjw::mask
{
    namespace
    {

        cv::Mat toBgr8(const cv::Mat& image)
        {
            if (image.empty())
            {
                return {};
            }

            cv::Mat image8;
            if (image.depth() == CV_8U)
            {
                image8 = image;
            }
            else
            {
                cv::normalize(image, image8, 0, 255, cv::NORM_MINMAX, CV_8U);
            }

            cv::Mat bgr;
            if (image8.channels() == 1)
            {
                cv::cvtColor(image8, bgr, cv::COLOR_GRAY2BGR);
            }
            else if (image8.channels() == 3)
            {
                bgr = image8.clone();
            }
            else if (image8.channels() == 4)
            {
                cv::cvtColor(image8, bgr, cv::COLOR_BGRA2BGR);
            }
            else
            {
                throw std::runtime_error("U2Net only supports 1, 3, or 4 channel images.");
            }
            return bgr;
        }

        cv::Mat firstOutputPlane(const cv::Mat& output)
        {
            if (output.empty())
            {
                return {};
            }
            if (output.type() != CV_32F)
            {
                throw std::runtime_error("U2Net inference output must use float32 tensors.");
            }
            if (output.dims == 4)
            {
                return cv::Mat(output.size[2], output.size[3], CV_32F, const_cast<float*>(output.ptr<float>())).clone();
            }
            if (output.dims == 3)
            {
                return cv::Mat(output.size[1], output.size[2], CV_32F, const_cast<float*>(output.ptr<float>())).clone();
            }
            if (output.dims == 2)
            {
                return output.clone();
            }
            return {};
        }

        cv::Mat filterForeground(const cv::Mat& foreground,
                                 int morphologyRadius,
                                 int minComponentArea,
                                 bool keepLargestComponent)
        {
            cv::Mat clean = foreground.clone();
            if (morphologyRadius > 0)
            {
                const int radius = std::clamp(morphologyRadius, 1, 64);
                const cv::Mat kernel =
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
                cv::morphologyEx(clean, clean, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(clean, clean, cv::MORPH_CLOSE, kernel);
            }

            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            const int components = cv::connectedComponentsWithStats(clean, labels, stats, centroids, 8, CV_32S);
            if (components <= 1)
            {
                return clean;
            }

            const int minimumArea = std::max(1, minComponentArea);
            int largestLabel = -1;
            int largestArea = 0;
            for (int label = 1; label < components; ++label)
            {
                const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                if (area > largestArea)
                {
                    largestArea = area;
                    largestLabel = label;
                }
            }

            cv::Mat filtered = cv::Mat::zeros(clean.size(), CV_8UC1);
            for (int label = 1; label < components; ++label)
            {
                const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                const bool keep = keepLargestComponent ? label == largestLabel : area >= minimumArea;
                if (keep && area >= minimumArea)
                {
                    filtered.setTo(255, labels == label);
                }
            }
            return filtered;
        }

    } // namespace

    cv::Mat makeU2NetBlob(const cv::Mat& image, int inputSize)
    {
        const cv::Mat bgr = toBgr8(image);
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(inputSize, inputSize), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat normalized;
        resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(normalized, channels);
        const double mean[3] = {0.485, 0.456, 0.406};
        const double standardDeviation[3] = {0.229, 0.224, 0.225};
        for (int index = 0; index < 3; ++index)
        {
            channels[index] = (channels[index] - mean[index]) / standardDeviation[index];
        }
        cv::merge(channels, normalized);

        return cv::dnn::blobFromImage(normalized, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
    }

    cv::Mat u2netProbabilityFromOutput(const cv::Mat& output)
    {
        const cv::Mat scores = firstOutputPlane(output);
        if (scores.empty())
        {
            throw std::runtime_error("U2Net inference output is empty or has an unsupported shape.");
        }

        cv::Mat probability;
        scores.convertTo(probability, CV_32F);
        double minimum = 0.0;
        double maximum = 0.0;
        cv::minMaxLoc(probability, &minimum, &maximum);
        if (maximum - minimum > 1e-6)
        {
            probability = (probability - minimum) / (maximum - minimum);
        }
        else
        {
            probability.setTo(0.0f);
        }
        return probability;
    }

    cv::Mat makeU2NetMask(const cv::Mat& probability,
                          const cv::Size& outputSize,
                          float foregroundThreshold,
                          int morphologyRadius,
                          int minComponentArea,
                          bool keepLargestComponent)
    {
        cv::Mat fullResolution;
        cv::resize(probability, fullResolution, outputSize, 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat foregroundFloat;
        cv::threshold(
            fullResolution, foregroundFloat, std::clamp(foregroundThreshold, 0.01f, 0.99f), 255.0, cv::THRESH_BINARY);
        cv::Mat foreground;
        foregroundFloat.convertTo(foreground, CV_8U);
        foreground = filterForeground(foreground, morphologyRadius, minComponentArea, keepLargestComponent);

        cv::Mat mask;
        cv::bitwise_not(foreground, mask);
        return mask;
    }

} // namespace xjw::mask

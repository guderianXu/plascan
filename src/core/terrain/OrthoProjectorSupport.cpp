#include "OrthoProjectorInternal.h"

#include "io/PathIO.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xjw::ortho_internal
{

namespace
{

bool isCancelled(const std::atomic_bool *cancelFlag)
{
    return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
}

bool finiteCamera(const Camera &camera)
{
    if (!camera.isValid()
        || !std::isfinite(camera.focalX())
        || !std::isfinite(camera.focalY())
        || !std::isfinite(camera.principalX())
        || !std::isfinite(camera.principalY())
        || camera.focalX() <= 0.0
        || camera.focalY() <= 0.0)
    {
        return false;
    }
    for (double value : camera.cameraCenter())
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    const auto rotation = camera.cameraToWorldRotation();
    for (double value : rotation)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    const double determinant =
        rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
        - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
        + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
    return std::isfinite(determinant) && std::abs(determinant) > 1e-6;
}

double medianValue(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double median = values[middle];
    if (values.size() % 2 == 0)
    {
        const auto lower = std::max_element(values.begin(), values.begin() + middle);
        median = 0.5 * (median + *lower);
    }
    return median;
}

double imageLuma(const cv::Mat &image, const cv::Mat &exclusionMask)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    if (exclusionMask.empty())
    {
        return cv::mean(gray)[0];
    }
    cv::Mat valid;
    cv::compare(exclusionMask, 0, valid, cv::CMP_EQ);
    return cv::countNonZero(valid) > 0 ? cv::mean(gray, valid)[0] : cv::mean(gray)[0];
}

double imageSharpness(const cv::Mat &image, const cv::Mat &exclusionMask)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F, 3);
    cv::Scalar mean;
    cv::Scalar deviation;
    if (exclusionMask.empty())
    {
        cv::meanStdDev(laplacian, mean, deviation);
    }
    else
    {
        cv::Mat valid;
        cv::compare(exclusionMask, 0, valid, cv::CMP_EQ);
        cv::meanStdDev(laplacian, mean, deviation, valid);
    }
    return std::max(1e-6, deviation[0] * deviation[0]);
}

cv::Vec3f sampleBilinear(const cv::Mat &image, double u, double v)
{
    const int x0 = std::clamp(static_cast<int>(std::floor(u)), 0, image.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(v)), 0, image.rows - 1);
    const int x1 = std::min(x0 + 1, image.cols - 1);
    const int y1 = std::min(y0 + 1, image.rows - 1);
    const float fx = static_cast<float>(u - static_cast<double>(x0));
    const float fy = static_cast<float>(v - static_cast<double>(y0));
    const cv::Vec3f c00 = image.at<cv::Vec3b>(y0, x0);
    const cv::Vec3f c10 = image.at<cv::Vec3b>(y0, x1);
    const cv::Vec3f c01 = image.at<cv::Vec3b>(y1, x0);
    const cv::Vec3f c11 = image.at<cv::Vec3b>(y1, x1);
    return (1.0f - fx) * (1.0f - fy) * c00
        + fx * (1.0f - fy) * c10
        + (1.0f - fx) * fy * c01
        + fx * fy * c11;
}

bool componentTouchesBoundary(const cv::Mat &labels, int label)
{
    for (int col = 0; col < labels.cols; ++col)
    {
        if (labels.at<int>(0, col) == label
            || labels.at<int>(labels.rows - 1, col) == label)
        {
            return true;
        }
    }
    for (int row = 0; row < labels.rows; ++row)
    {
        if (labels.at<int>(row, 0) == label
            || labels.at<int>(row, labels.cols - 1) == label)
        {
            return true;
        }
    }
    return false;
}

bool componentTouchesInvalidSurface(const cv::Mat &labels,
                                    int label,
                                    const cv::Mat &surfaceMask,
                                    const cv::Rect &bounds)
{
    for (int row = bounds.y; row < bounds.y + bounds.height; ++row)
    {
        for (int col = bounds.x; col < bounds.x + bounds.width; ++col)
        {
            if (labels.at<int>(row, col) != label)
            {
                continue;
            }
            for (int deltaRow = -1; deltaRow <= 1; ++deltaRow)
            {
                for (int deltaCol = -1; deltaCol <= 1; ++deltaCol)
                {
                    const int neighborRow = row + deltaRow;
                    const int neighborCol = col + deltaCol;
                    if (neighborRow < 0 || neighborCol < 0
                        || neighborRow >= surfaceMask.rows
                        || neighborCol >= surfaceMask.cols
                        || surfaceMask.at<uchar>(neighborRow, neighborCol) == 0)
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

} // namespace

bool loadFrames(const std::vector<OrthoImageInput> &inputs,
                const OrthoGenerationOptions &options,
                std::vector<LoadedFrame> *frames,
                const std::atomic_bool *cancelFlag,
                QString *errorMsg)
{
    frames->clear();
    std::vector<double> lumas;
    std::vector<double> sharpness;
    for (const OrthoImageInput &input : inputs)
    {
        if (isCancelled(cancelFlag))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射影像生成已取消");
            }
            return false;
        }
        if (!finiteCamera(input.camera))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("影像相机参数无效（内参或位姿）: %1")
                                .arg(input.imagePath);
            }
            return false;
        }
        cv::Mat image = xjw::common::io::readImage(input.imagePath, cv::IMREAD_COLOR);
        if (image.empty())
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射输入影像读取失败: %1")
                                .arg(input.imagePath);
            }
            return false;
        }

        LoadedFrame frame;
        frame.input = input;
        frame.imageBgr = std::move(image);
        if (options.useProjectMasks && !input.exclusionMaskPath.isEmpty())
        {
            frame.exclusionMask =
                xjw::common::io::readImage(input.exclusionMaskPath, cv::IMREAD_GRAYSCALE);
            if (frame.exclusionMask.empty())
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("项目蒙版读取失败: %1")
                                    .arg(input.exclusionMaskPath);
                }
                return false;
            }
            if (frame.exclusionMask.size() != frame.imageBgr.size())
            {
                cv::resize(frame.exclusionMask,
                           frame.exclusionMask,
                           frame.imageBgr.size(),
                           0.0,
                           0.0,
                           cv::INTER_NEAREST);
            }
        }
        lumas.push_back(imageLuma(frame.imageBgr, frame.exclusionMask));
        sharpness.push_back(imageSharpness(frame.imageBgr, frame.exclusionMask));
        frames->push_back(std::move(frame));
    }

    if (frames->empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("未找到可用于正射投影的有效影像和相机参数");
        }
        return false;
    }

    const double target_luma = medianValue(lumas);
    const double median_sharpness = std::max(1e-6, medianValue(sharpness));
    for (std::size_t index = 0; index < frames->size(); ++index)
    {
        if (options.colorCorrection)
        {
            (*frames)[index].gain =
                std::clamp(target_luma / std::max(1e-6, lumas[index]), 0.7, 1.3);
        }
        if (options.sharpnessWeighting)
        {
            (*frames)[index].sharpnessWeight =
                std::clamp(sharpness[index] / median_sharpness, 0.5, 2.0);
        }
    }
    return true;
}

bool sampleFrame(const LoadedFrame &frame,
                 const double world[3],
                 ColorCandidate *candidate)
{
    double pixel[2]{0.0, 0.0};
    if (!candidate || !frame.input.camera.projectWorldPoint(world, pixel))
    {
        return false;
    }
    const double u = pixel[0];
    const double v = pixel[1];
    if (u < 0.0 || v < 0.0
        || u >= static_cast<double>(frame.imageBgr.cols - 1)
        || v >= static_cast<double>(frame.imageBgr.rows - 1))
    {
        return false;
    }

    if (!frame.exclusionMask.empty())
    {
        const int mask_x = std::clamp(
            static_cast<int>(std::lround(u)), 0, frame.exclusionMask.cols - 1);
        const int mask_y = std::clamp(
            static_cast<int>(std::lround(v)), 0, frame.exclusionMask.rows - 1);
        if (frame.exclusionMask.at<uchar>(mask_y, mask_x) != 0)
        {
            return false;
        }
    }

    const double du = (u - frame.input.camera.principalX())
        / std::max(1.0, frame.input.camera.focalX());
    const double dv = (v - frame.input.camera.principalY())
        / std::max(1.0, frame.input.camera.focalY());
    const double view_weight = 1.0 / (1.0 + du * du + dv * dv);
    const double edge_distance = std::min(
        std::min(u, static_cast<double>(frame.imageBgr.cols - 1) - u),
        std::min(v, static_cast<double>(frame.imageBgr.rows - 1) - v));
    const double edge_weight = std::clamp(edge_distance / 20.0, 0.05, 1.0);
    candidate->color =
        sampleBilinear(frame.imageBgr, u, v) * static_cast<float>(frame.gain);
    candidate->weight = view_weight * edge_weight * frame.sharpnessWeight;
    return candidate->weight > 0.0;
}

void filterGhostCandidates(std::vector<ColorCandidate> *candidates)
{
    if (!candidates || candidates->size() < 3)
    {
        return;
    }

    cv::Vec3d median_color;
    for (int channel = 0; channel < 3; ++channel)
    {
        std::vector<double> values;
        values.reserve(candidates->size());
        for (const ColorCandidate &candidate : *candidates)
        {
            values.push_back(candidate.color[channel]);
        }
        median_color[channel] = medianValue(std::move(values));
    }

    std::vector<double> residuals;
    residuals.reserve(candidates->size());
    for (const ColorCandidate &candidate : *candidates)
    {
        const cv::Vec3d difference =
            cv::Vec3d(candidate.color) - median_color;
        residuals.push_back(cv::norm(difference));
    }
    const double median_residual = medianValue(residuals);
    std::vector<double> deviations;
    deviations.reserve(residuals.size());
    for (double residual : residuals)
    {
        deviations.push_back(std::abs(residual - median_residual));
    }
    const double threshold = std::max(
        24.0, median_residual + 3.5 * 1.4826 * medianValue(std::move(deviations)));

    std::vector<ColorCandidate> filtered;
    filtered.reserve(candidates->size());
    for (std::size_t index = 0; index < candidates->size(); ++index)
    {
        if (residuals[index] <= threshold)
        {
            filtered.push_back((*candidates)[index]);
        }
    }
    if (!filtered.empty())
    {
        *candidates = std::move(filtered);
    }
}

qint64 fillSmallInteriorHoles(cv::Mat *imageBgr,
                              const cv::Mat &surfaceMask,
                              const cv::Mat &coverageMask,
                              const OrthoGenerationOptions &options,
                              cv::Mat *filledMask,
                              const std::atomic_bool *cancelFlag)
{
    *filledMask = cv::Mat(surfaceMask.size(), CV_8UC1, cv::Scalar(0));
    if (!options.fillHoles)
    {
        return 0;
    }

    cv::Mat holes;
    cv::bitwise_and(surfaceMask, coverageMask == 0, holes);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(
        holes, labels, stats, centroids, 8, CV_32S);
    for (int label = 1; label < count; ++label)
    {
        if (isCancelled(cancelFlag))
        {
            return -1;
        }
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const cv::Rect componentBounds(
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT));
        if (area <= options.holeFillMaxArea
            && !componentTouchesBoundary(labels, label)
            && !componentTouchesInvalidSurface(
                labels, label, surfaceMask, componentBounds))
        {
            filledMask->setTo(255, labels == label);
        }
    }

    cv::Mat valid = coverageMask.clone();
    const int radius = std::max(1, static_cast<int>(std::ceil(options.holeFillRadius)));
    qint64 total_filled = 0;
    const int maximum_passes = std::min(
        32,
        static_cast<int>(std::ceil(std::sqrt(
            static_cast<double>(options.holeFillMaxArea)))) + radius);
    for (int pass = 0; pass < maximum_passes; ++pass)
    {
        if (isCancelled(cancelFlag))
        {
            return -1;
        }
        cv::Mat next_image = imageBgr->clone();
        cv::Mat next_valid = valid.clone();
        int pass_filled = 0;
        for (int row = 0; row < imageBgr->rows; ++row)
        {
            if (isCancelled(cancelFlag))
            {
                return -1;
            }
            for (int col = 0; col < imageBgr->cols; ++col)
            {
                if (filledMask->at<uchar>(row, col) == 0
                    || valid.at<uchar>(row, col) != 0)
                {
                    continue;
                }
                cv::Vec3d sum(0.0, 0.0, 0.0);
                int neighbors = 0;
                for (int dr = -radius; dr <= radius; ++dr)
                {
                    for (int dc = -radius; dc <= radius; ++dc)
                    {
                        const int source_row = row + dr;
                        const int source_col = col + dc;
                        if (source_row < 0 || source_col < 0
                            || source_row >= valid.rows || source_col >= valid.cols
                            || valid.at<uchar>(source_row, source_col) == 0)
                        {
                            continue;
                        }
                        sum += imageBgr->at<cv::Vec3b>(source_row, source_col);
                        ++neighbors;
                    }
                }
                if (neighbors >= 3)
                {
                    next_image.at<cv::Vec3b>(row, col) =
                        cv::Vec3b(sum / static_cast<double>(neighbors));
                    next_valid.at<uchar>(row, col) = 255;
                    ++pass_filled;
                }
            }
        }
        *imageBgr = std::move(next_image);
        valid = std::move(next_valid);
        total_filled += pass_filled;
        if (pass_filled == 0)
        {
            break;
        }
    }
    cv::bitwise_and(*filledMask, valid, *filledMask);
    return total_filled;
}

} // namespace xjw::ortho_internal

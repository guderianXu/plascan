#include "TextureMappingV4Internal.h"

#include <metashape_texture/recovered_kernels.hpp>

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace xjw::mesh::texture_v4
{
    namespace
    {
        cv::Mat removeSmallWinnerComponents(const cv::Mat& winner)
        {
            if (winner.type() != CV_8UC1 || winner.empty())
            {
                return winner;
            }

            metashape_texture::Image<std::uint8_t> recovered_winner(
                winner.cols, winner.rows);
            for (int row = 0; row < winner.rows; ++row)
            {
                for (int column = 0; column < winner.cols; ++column)
                {
                    recovered_winner(column, row) =
                        winner.at<std::uint8_t>(row, column);
                }
            }
            const metashape_texture::Image<std::uint8_t> filtered =
                metashape_texture::recovered::remove_small_components_8(
                    recovered_winner, 2U);
            cv::Mat result(winner.size(), CV_8UC1);
            for (int row = 0; row < result.rows; ++row)
            {
                for (int column = 0; column < result.cols; ++column)
                {
                    result.at<std::uint8_t>(row, column) = filtered(column, row);
                }
            }
            return result;
        }

        cv::Mat winnerMask(const PipelineData& data, int view_index)
        {
            const PreparedView& view = data.views[view_index];
            cv::Mat winner(view.colorBgr.size(), CV_8UC1, cv::Scalar(0));
            if (!view.finalMeshFaceIds.empty())
            {
                for (int row = 0; row < winner.rows; ++row)
                {
                    for (int column = 0; column < winner.cols; ++column)
                    {
                        const int face = view.finalMeshFaceIds.at<int>(row, column);
                        if (face >= 0 && face < data.assignments.size() &&
                            data.assignments[face].primaryView == view_index)
                        {
                            winner.at<std::uint8_t>(row, column) = 255;
                        }
                    }
                }
                return removeSmallWinnerComponents(winner);
            }
            // The diagnostic visibility-off mode still needs spatial winner
            // seeds. Normal workflow uses the final mesh z-buffer above.
            for (const TextureChart& chart : data.charts)
            {
                if (chart.primaryView != view_index)
                {
                    continue;
                }
                for (const int face : chart.faces)
                {
                    std::array<cv::Point, 3> triangle;
                    bool valid = true;
                    for (int corner = 0; corner < 3; ++corner)
                    {
                        double pixel[2]{};
                        double depth = 0.0;
                        valid = valid && view.colorCamera.projectWorldPointWithDepth(
                                             data.geometry[face].vertices[corner].data(), pixel, depth);
                        triangle[corner] = cv::Point(cvRound(pixel[0]), cvRound(pixel[1]));
                    }
                    if (valid)
                    {
                        cv::fillConvexPoly(winner, triangle.data(), 3, cv::Scalar(255));
                    }
                }
            }
            return removeSmallWinnerComponents(winner);
        }
    } // namespace

    bool prepareTextureSourcePyramids(const TextureMappingConfig& config,
                                      PipelineData* data,
                                      TextureMappingResult* result,
                                      std::string* errorMsg)
    {
        if (config.blendMode != TextureBlendMode::Natural)
        {
            return true;
        }
        QVector<bool> used(data->views.size(), false);
        for (const FaceAssignment& assignment : data->assignments)
        {
            for (const FaceCandidate& candidate : assignment.candidates)
            {
                used[candidate.viewIndex] = true;
            }
        }
        double source_bytes = 0.0;
        double largest_pixels = 0.0;
        for (int index = 0; index < data->views.size(); ++index)
        {
            const PreparedView& view = data->views[index];
            const double pixels = static_cast<double>(view.colorBgr.total());
            // Prepared color, gray, focus, support-distance, face IDs, plus
            // stored weights and four coarse RGB levels. The level-zero float
            // color is transient, not retained for every camera.
            source_bytes += pixels * (used[index] ? 23.0 : 16.0);
            if (used[index])
            {
                largest_pixels = std::max(largest_pixels, pixels);
            }
        }
        result->peakMemoryEstimateMiB += (source_bytes + largest_pixels * 40.0) / (1024.0 * 1024.0);
        if (result->peakMemoryEstimateMiB > 3072.0)
        {
            if (errorMsg)
            {
                *errorMsg = "纹理图集和源影像金字塔的估算工作内存超过 3 GiB，请降低纹理大小或提高影像降采样倍数";
            }
            return false;
        }
        for (int index = 0; index < data->views.size(); ++index)
        {
            if (config.isCancelled && config.isCancelled())
            {
                result->cancelled = true;
                if (errorMsg)
                {
                    *errorMsg = "纹理映射已取消";
                }
                return false;
            }
            if (!used[index])
            {
                continue;
            }
            if (config.progressFn)
            {
                config.progressFn("正在构建源影像多频段金字塔 " + std::to_string(index + 1) + "/" +
                                      std::to_string(data->views.size()) + "...",
                                  65);
            }
            PreparedView& view = data->views[index];
            cv::Mat support = view.supportDistance > 0.0f;
            if (!view.finalMeshFaceIds.empty())
            {
                support.setTo(0, view.finalMeshFaceIds < 0);
            }
            const cv::Mat winner = winnerMask(*data, index);
            view.blendPyramid =
                buildTextureSourcePyramid(view.colorBgr, support, winner, view.exposureGain, config.isCancelled);
            if (view.blendPyramid.weight[0].empty())
            {
                result->cancelled = true;
                if (errorMsg)
                {
                    *errorMsg = "纹理映射已取消";
                }
                return false;
            }
        }
        return true;
    }
} // namespace xjw::mesh::texture_v4

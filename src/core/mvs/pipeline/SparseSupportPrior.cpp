#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    cv::Mat MvsPipelineService::buildSparseSeedDepthFromProjectedSamples(
        int refIdx, int W, int H, const std::vector<ProjectedSparseDepthSample>& samples, int seedRadius)
    {
        (void)refIdx;
        if (samples.empty() || W <= 0 || H <= 0)
        {
            return cv::Mat();
        }

        const int radius = std::clamp(seedRadius, 0, 8);
        cv::Mat hint(H, W, CV_32F, cv::Scalar(0.f));

        for (const ProjectedSparseDepthSample& sample : samples)
        {
            const int iu = static_cast<int>(std::round(sample.uNorm * static_cast<float>(W)));
            const int iv = static_cast<int>(std::round(sample.vNorm * static_cast<float>(H)));
            if (iu < 0 || iu >= W || iv < 0 || iv >= H || sample.depth <= 0.0f)
            {
                continue;
            }

            for (int dv = -radius; dv <= radius; ++dv)
            {
                for (int du = -radius; du <= radius; ++du)
                {
                    int nu = iu + du, nv = iv + dv;
                    if (nu < 0 || nu >= W || nv < 0 || nv >= H)
                    {
                        continue;
                    }
                    float& h = hint.at<float>(nv, nu);
                    if (h == 0.f || sample.depth < h)
                    {
                        h = sample.depth;
                    }
                }
            }
        }

        return cv::countNonZero(hint > 0) > 0 ? hint : cv::Mat();
    }

    // =============================================================================
    cv::Mat MvsPipelineService::buildSparseSupportMask(const std::vector<CameraView>& views,
                                                       const SparseCloud& sparse,
                                                       int refIdx,
                                                       int W,
                                                       int H,
                                                       const std::vector<int>& sourceIndices)
    {
        if (W <= 0 || H <= 0 || refIdx < 0 || refIdx >= static_cast<int>(views.size()) || sparse.points.size() < 20)
        {
            return cv::Mat();
        }

        const FramePinholeCamera cam = mvsPinholeCamera(views[refIdx].camera);
        if (!cam.isValid())
        {
            return cv::Mat();
        }

        const int minSourceViews = sourceIndices.empty() ? 0 : 1;
        std::vector<size_t> visiblePointIndices =
            collectMvsVisibleSparsePointIndices(views, sparse, refIdx, sourceIndices, minSourceViews);
        if (visiblePointIndices.size() < 20 && minSourceViews > 0)
        {
            visiblePointIndices = collectMvsVisibleSparsePointIndices(views, sparse, refIdx, {}, 0);
        }
        if (visiblePointIndices.size() < 20)
        {
            return cv::Mat();
        }

        const std::vector<ProjectedSparseDepthSample> samples =
            collectProjectedSparseDepthSamples(sparse, cam, W, H, visiblePointIndices);
        return buildSparseSupportMaskFromProjectedSamples(refIdx, W, H, samples);
    }

    cv::Mat MvsPipelineService::buildSparseSupportMaskFromProjectedSamples(
        int refIdx, int W, int H, const std::vector<ProjectedSparseDepthSample>& samples)
    {
        if (W <= 0 || H <= 0 || samples.size() < 20)
        {
            return cv::Mat();
        }

        std::vector<cv::Point> seed_points;
        seed_points.reserve(samples.size());
        int projectedSeeds = 0;
        for (const ProjectedSparseDepthSample& sample : samples)
        {
            const int iu = static_cast<int>(std::round(sample.uNorm * static_cast<float>(W)));
            const int iv = static_cast<int>(std::round(sample.vNorm * static_cast<float>(H)));
            if (iu < 0 || iu >= W || iv < 0 || iv >= H || sample.depth <= 0.0f)
            {
                continue;
            }

            seed_points.emplace_back(iu, iv);
            ++projectedSeeds;
        }

        std::sort(seed_points.begin(),
                  seed_points.end(),
                  [](const cv::Point& left, const cv::Point& right)
                  { return left.y < right.y || (left.y == right.y && left.x < right.x); });
        seed_points.erase(std::unique(seed_points.begin(), seed_points.end()), seed_points.end());
        const int seedPixels = static_cast<int>(seed_points.size());
        if (seedPixels < 10)
        {
            return cv::Mat();
        }

        const int maxDim = std::max(W, H);
        const int radius = maxDim < 512
                               ? std::clamp(maxDim / 8, 8, 48)
                               : (maxDim < 1200 ? std::clamp(maxDim / 16, 32, 64) : std::clamp(maxDim / 48, 48, 128));
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
        std::vector<std::pair<int, int>> kernel_spans(static_cast<std::size_t>(kernel.rows), {0, -1});
        for (int kernel_row = 0; kernel_row < kernel.rows; ++kernel_row)
        {
            const uint8_t* row = kernel.ptr<uint8_t>(kernel_row);
            int first = 0;
            while (first < kernel.cols && row[first] == 0)
            {
                ++first;
            }
            int last = kernel.cols - 1;
            while (last >= first && row[last] == 0)
            {
                --last;
            }
            kernel_spans[static_cast<std::size_t>(kernel_row)] = {first, last};
        }

        cv::Mat support(H, W, CV_8U, cv::Scalar(0));
        for (const cv::Point& seed_point : seed_points)
        {
            // Binary dilation of point seeds is exactly the union of translated
            // structuring elements. Stamp the ellipse as contiguous row spans so
            // the mask stays bit-identical to cv::dilate without scanning every
            // full-resolution pixel with a 257x257 kernel between GPU frames.
            for (int kernel_row = 0; kernel_row < kernel.rows; ++kernel_row)
            {
                const int support_row = seed_point.y + kernel_row - radius;
                if (support_row < 0 || support_row >= H)
                {
                    continue;
                }
                const auto [span_first, span_last] = kernel_spans[static_cast<std::size_t>(kernel_row)];
                const int first = std::max(0, seed_point.x + span_first - radius);
                const int last = std::min(W - 1, seed_point.x + span_last - radius);
                if (first <= last)
                {
                    uint8_t* row = support.ptr<uint8_t>(support_row);
                    std::fill(row + first, row + last + 1, static_cast<uint8_t>(255));
                }
            }
        }

        const int supportPixels = cv::countNonZero(support);
        const float coverage = static_cast<float>(supportPixels) / static_cast<float>(W * H);
        if (coverage < 0.03f || coverage > 0.95f)
        {
            return cv::Mat();
        }

        LOG_DEBUG("[MVS][帧 %d][稀疏支撑] seeds=%d/%d radius=%d coverage=%d/%d (%.1f%%)",
                  refIdx,
                  seedPixels,
                  projectedSeeds,
                  radius,
                  supportPixels,
                  W * H,
                  coverage * 100.0f);
        return support;
    }

    cv::Mat MvsPipelineService::buildSparseSupportMaskFromVisiblePoints(
        int refIdx, int W, int H, const std::vector<size_t>& visiblePointIndices) const
    {
        if (refIdx < 0 || refIdx >= static_cast<int>(_views.size()))
        {
            return cv::Mat();
        }

        return buildSparseSupportMaskForCamera(
            refIdx, mvsPinholeCamera(_views[refIdx].camera), W, H, visiblePointIndices);
    }

    cv::Mat MvsPipelineService::buildSparseSupportMaskForCamera(int refIdx,
                                                                const FramePinholeCamera& camera,
                                                                int W,
                                                                int H,
                                                                const std::vector<size_t>& visiblePointIndices) const
    {
        if (W <= 0 || H <= 0 || refIdx < 0 || refIdx >= static_cast<int>(_views.size()) || _sparse.points.size() < 20)
        {
            return cv::Mat();
        }

        if (!camera.isValid() || visiblePointIndices.size() < 20)
        {
            return cv::Mat();
        }

        const std::vector<ProjectedSparseDepthSample> samples =
            collectProjectedSparseDepthSamples(_sparse, camera, W, H, visiblePointIndices);
        return buildSparseSupportMaskFromProjectedSamples(refIdx, W, H, samples);
    }
} // namespace xjw::mvs

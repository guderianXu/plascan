#include "TextureOverlapExposure.h"

#include "TextureAtlasSampling.h"
#include "TextureMappingV4Internal.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::mesh::texture_v4
{
namespace
{

float srgbChannelToLinear(float value)
{
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    const float normalized = std::clamp(value / 255.0f, 0.0f, 1.0f);
    if (normalized <= 0.04045f)
    {
        return normalized / 12.92f;
    }
    return std::pow((normalized + 0.055f) / 1.055f, 2.4f);
}

float linearChannelToSrgb8(float value)
{
    const float linear = std::clamp(value, 0.0f, 1.0f);
    const float encoded = linear <= 0.0031308f
        ? linear * 12.92f
        : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    return encoded * 255.0f;
}

double linearLuminance(const cv::Vec3f &linear_bgr)
{
    return 0.0722 * linear_bgr[0] +
        0.7152 * linear_bgr[1] +
        0.2126 * linear_bgr[2];
}

void copySolveDiagnostics(const ExposureSolveResult &solve_result,
                          TextureMappingResult *result)
{
    result->exposureCorrectionObservationCount =
        solve_result.observationCount;
    result->exposureCorrectionCandidatePairCount =
        solve_result.candidatePairCount;
    result->exposureCorrectionAcceptedPairCount =
        solve_result.acceptedPairCount;
    result->exposureCorrectionRejectedInsufficientPairCount =
        solve_result.rejectedInsufficientPairCount;
    result->exposureCorrectionRejectedHighMadPairCount =
        solve_result.rejectedHighMadPairCount;
    result->exposureCorrectionMaximumAcceptedLogMad =
        solve_result.maximumAcceptedLogMad;
    result->exposureCorrectionGraphConnected =
        solve_result.graphConnected;
    result->exposureCorrectionApplied = solve_result.applied;
    result->exposureCorrectionStatus = solve_result.status;
    if (!solve_result.gains.empty())
    {
        const auto [minimum, maximum] = std::minmax_element(
            solve_result.gains.begin(), solve_result.gains.end());
        result->exposureCorrectionMinimumGain = *minimum;
        result->exposureCorrectionMaximumGain = *maximum;
    }
}

} // namespace

cv::Vec3f srgb8ToLinear(const cv::Vec3f &srgbColor)
{
    return cv::Vec3f(
        srgbChannelToLinear(srgbColor[0]),
        srgbChannelToLinear(srgbColor[1]),
        srgbChannelToLinear(srgbColor[2]));
}

cv::Vec3f applyLinearSrgbExposureGain(const cv::Vec3f &srgbColor,
                                      float gain)
{
    if (std::fabs(gain - 1.0f) <= 1.0e-6f)
    {
        return srgbColor;
    }
    const cv::Vec3f linear = srgb8ToLinear(srgbColor) *
        std::max(gain, 0.0f);
    return cv::Vec3f(
        linearChannelToSrgb8(linear[0]),
        linearChannelToSrgb8(linear[1]),
        linearChannelToSrgb8(linear[2]));
}

bool estimateOverlapExposureGains(const TextureMappingConfig &config,
                                  PipelineData *data,
                                  TextureMappingResult *result,
                                  std::string *errorMsg)
{
    if (!data || !result)
    {
        if (errorMsg)
        {
            *errorMsg = "纹理曝光校正输出参数为空";
        }
        return false;
    }
    if (!config.enableColorCorrection)
    {
        result->exposureCorrectionStatus = "disabled";
        return true;
    }
    if (config.progressFn)
    {
        config.progressFn("正在估计共同可见区域曝光...", 19);
    }

    constexpr int maximum_sample_points = 16384;
    const int face_count = data->geometry.size();
    const int sample_stride = std::max(
        1,
        (face_count + maximum_sample_points - 1) / maximum_sample_points);
    std::vector<ExposureObservation> observations;
    observations.reserve(static_cast<std::size_t>(
        std::min(face_count, maximum_sample_points) * data->views.size()));
    FaceCandidate candidate;
    candidate.score = 1.0f;
    candidate.strict = true;
    candidate.finalMeshVisibilityRequired =
        config.enableFinalMeshVisibility;
    for (int face_index = 0;
         face_index < face_count;
         face_index += sample_stride)
    {
        if (((face_index / sample_stride) % 128 == 0) &&
            config.isCancelled && config.isCancelled())
        {
            result->cancelled = true;
            if (errorMsg)
            {
                *errorMsg = "纹理映射已取消";
            }
            return false;
        }
        const std::array<double, 3> &world =
            data->geometry[face_index].centroid;
        for (int view_index = 0; view_index < data->views.size(); ++view_index)
        {
            candidate.viewIndex = view_index;
            WeightedColor sample;
            if (sampleTextureView(
                    data->views[view_index],
                    world,
                    candidate,
                    config,
                    data->medianEdgeLength,
                    config.padding,
                    &sample,
                    face_index) != TextureSampleStatus::Sampled)
            {
                continue;
            }
            const cv::Vec3f linear_bgr = srgb8ToLinear(sample.color);
            const double luminance = linearLuminance(linear_bgr);
            const float maximum_channel = std::max(
                {linear_bgr[0], linear_bgr[1], linear_bgr[2]});
            if (!std::isfinite(luminance) || luminance < 0.01 ||
                luminance > 0.95 || maximum_channel >= 0.98f)
            {
                continue;
            }
            observations.push_back({
                view_index,
                static_cast<std::uint64_t>(face_index),
                luminance});
        }
    }

    const ExposureSolveResult solve_result = solveRobustOverlapExposure(
        data->views.size(), std::move(observations));
    for (int view_index = 0; view_index < data->views.size(); ++view_index)
    {
        data->views[view_index].exposureGain =
            solve_result.gains[static_cast<std::size_t>(view_index)];
    }
    copySolveDiagnostics(solve_result, result);
    return true;
}

} // namespace xjw::mesh::texture_v4

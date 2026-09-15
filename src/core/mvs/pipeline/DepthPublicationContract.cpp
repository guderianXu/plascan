#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;
    // namespace detail

    void detail::applySourceAngleCapShortfallSafety(DepthFrameResult& result)
    {
        if (!result.sourceAngleCapEnabled || result.sourceViewShortfall <= 0 ||
            result.qualityDecision.acceptance == DepthFrameAcceptance::Rejected)
        {
            return;
        }

        if (result.qualityDecision.acceptance == DepthFrameAcceptance::Accepted)
        {
            result.qualityDecision.acceptance = DepthFrameAcceptance::ValidationOnly;
        }

        constexpr const char* kReason = "source_angle_cap_source_shortfall";
        if (std::find(result.qualityDecision.reasons.cbegin(), result.qualityDecision.reasons.cend(), kReason) ==
            result.qualityDecision.reasons.cend())
        {
            result.qualityDecision.reasons.emplace_back(kReason);
        }
    }

    bool detail::hasCompletedConsistencyPublication(const DepthFrameResult& result)
    {
        return result.depthCompleteness.preConsistencyValidCount >= 0 &&
               result.depthCompleteness.postConsistencyValidCount >= 0 &&
               result.depthCompleteness.publishedPostConsistencyValidCount >= 0;
    }

    bool detail::expectsConsistencyPublication(const DepthFrameResult& result, int view_count)
    {
        return view_count >= 2 && result.success && result.initialQualityAcceptanceAvailable &&
               result.initialQualityAcceptance != DepthFrameAcceptance::Rejected;
    }

    std::vector<DepthConsistencyFrameSourcePlan>
    freezeDepthConsistencySourcePlans(const std::vector<DepthFrameResult>& frames,
                                      int view_count,
                                      MvsSceneProfile scene_profile,
                                      int requested_repair_source_count)
    {
        if (view_count <= 0)
        {
            return {};
        }

        std::vector<bool> source_eligibility(static_cast<std::size_t>(view_count), false);
        const int available_frame_count = std::min(view_count, static_cast<int>(frames.size()));
        for (int frame_index = 0; frame_index < available_frame_count; ++frame_index)
        {
            source_eligibility[static_cast<std::size_t>(frame_index)] =
                frames[static_cast<std::size_t>(frame_index)].eligibleAsConsistencySource();
        }

        std::vector<DepthConsistencyFrameSourcePlan> plans(static_cast<std::size_t>(view_count));
        for (int reference_index = 0; reference_index < view_count; ++reference_index)
        {
            DepthConsistencyFrameSourcePlan& plan = plans[static_cast<std::size_t>(reference_index)];
            plan.consistencySourceIndices = consistencySourceIndicesForFrame(frames, reference_index, view_count);
            if (scene_profile == MvsSceneProfile::OrbitalObject)
            {
                plan.geometrySourceViewIndices = planMvsRepairSourceViews(
                    plan.consistencySourceIndices, source_eligibility, reference_index, requested_repair_source_count);
                continue;
            }

            for (const int source_index : plan.consistencySourceIndices)
            {
                if (plan.geometrySourceViewIndices.size() >= 16 || source_index < 0 || source_index >= view_count ||
                    !source_eligibility[static_cast<std::size_t>(source_index)])
                {
                    continue;
                }
                plan.geometrySourceViewIndices.push_back(source_index);
            }
        }
        return plans;
    }

    void
    detail::runDepthMapBackgroundTaskWithExceptionBoundary(const std::function<void()>& task,
                                                           const std::function<void(const QString&)>& failureHandler)
    {
        try
        {
            task();
        }
        catch (const std::exception& error)
        {
            failureHandler(QStringLiteral("MVS 后台任务异常终止：%1").arg(QString::fromUtf8(error.what())));
        }
        catch (...)
        {
            failureHandler(QStringLiteral("MVS 后台任务异常终止：未知异常"));
        }
    }
} // namespace xjw::mvs

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh::texture_v4
{

    using TextureLabelCost = std::int64_t;

    inline constexpr TextureLabelCost kForbiddenTextureLabelCost = static_cast<TextureLabelCost>(1) << 60;

    struct TextureLabelEdge
    {
        int first = -1;
        int second = -1;
        TextureLabelCost cost = 0;
    };

    struct TextureLabelOptimizationProblem
    {
        int nodeCount = 0;
        int labelCount = 0;
        int maximumPasses = 1;
        std::vector<TextureLabelEdge> edges;
        std::vector<int> initialLabels;
        std::function<TextureLabelCost(int, int)> unaryCost;
        std::function<bool()> isCancelled;
        std::function<void(int, int)> progressFn;
    };

    struct TextureLabelOptimizationResult
    {
        bool solved = false;
        bool cancelled = false;
        std::string error;
        std::vector<int> labels;
        int changedNodeCount = 0;
        int completedExpansionCount = 0;
        TextureLabelCost energy = 0;
    };

    /**
     * @brief 用 Potts 接缝代价执行确定性的多标签 alpha-expansion 图割。
     */
    TextureLabelOptimizationResult optimizeTextureLabels(const TextureLabelOptimizationProblem& problem);

} // namespace xjw::mesh::texture_v4

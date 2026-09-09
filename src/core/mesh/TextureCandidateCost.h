#pragma once

#include <cstdint>
#include <vector>

namespace xjw::mesh::texture_v4
{

    inline constexpr float kMissingTextureQuality = -128.0f;

    struct TextureCandidateQuality
    {
        float sharpness = kMissingTextureQuality;
        float photometricConsistency = kMissingTextureQuality;
        float resolution = kMissingTextureQuality;
        float nadirness = kMissingTextureQuality;
    };

    struct TextureCandidateCostFlags
    {
        bool sharpness = false;
        bool photometricConsistency = true;
        bool resolution = true;
        bool nadirness = true;
    };

    /**
     * @brief 将同一三角面上的相机质量转换为可比较的一元代价。
     *
     * 各质量项只在当前面的候选相机之间做相对归一化，返回值越小越好。
     * 该公式来自参考实现中已恢复的候选相机代价模型。
     */
    std::vector<std::int32_t> buildTextureCandidateUnaryCosts(const std::vector<TextureCandidateQuality>& qualities,
                                                              float face_weight,
                                                              const TextureCandidateCostFlags& flags);

} // namespace xjw::mesh::texture_v4

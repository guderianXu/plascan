#include "TextureCandidateCost.h"

#include <metashape_texture/candidate_unary.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace xjw::mesh::texture_v4
{

    std::vector<std::int32_t> buildTextureCandidateUnaryCosts(const std::vector<TextureCandidateQuality>& qualities,
                                                              float face_weight,
                                                              const TextureCandidateCostFlags& flags)
    {
        if (qualities.empty())
        {
            return {};
        }
        if (!std::isfinite(face_weight))
        {
            throw std::invalid_argument("texture face weight must be finite");
        }

        std::vector<metashape_texture::recovered::FaceCameraQuality> recovered_qualities;
        recovered_qualities.reserve(qualities.size());
        for (const TextureCandidateQuality& quality : qualities)
        {
            const auto valid = [](bool enabled, float value)
            { return !enabled || (std::isfinite(value) && value != kMissingTextureQuality); };
            if (!valid(flags.sharpness, quality.sharpness) ||
                !valid(flags.photometricConsistency, quality.photometricConsistency) ||
                !valid(flags.resolution, quality.resolution) || !valid(flags.nadirness, quality.nadirness))
            {
                throw std::invalid_argument("enabled texture candidate quality is missing or non-finite");
            }
            recovered_qualities.push_back(
                {quality.sharpness, quality.photometricConsistency, quality.resolution, quality.nadirness});
        }

        // Preserve PlaScan's input validation boundary, then use the recovered
        // Record20 binary32 unary conversion rather than a parallel approximation.
        return metashape_texture::recovered::build_face_camera_unary(
            recovered_qualities,
            face_weight,
            {flags.sharpness, flags.photometricConsistency, flags.nadirness, flags.resolution});
    }

} // namespace xjw::mesh::texture_v4

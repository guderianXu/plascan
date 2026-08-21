#pragma once

#include <algorithm>

#include <QString>

namespace xjw::mvs
{

enum class DepthFrameRole
{
    Excluded,
    CoverageAuxiliary,
    Primary
};

enum class DepthSceneProfile
{
    Unknown,
    AerialTerrain,
    OrbitalObject,
    Custom
};

inline constexpr float kCoverageAuxiliaryWeightMultiplier = 0.35f;

inline QString canonicalDepthSceneProfile(const QString &profile)
{
    const QString normalized = profile.trimmed().toLower();
    return normalized == QStringLiteral("general")
        ? QStringLiteral("custom")
        : normalized;
}

inline DepthSceneProfile classifyDepthSceneProfile(const QString &profile)
{
    const QString canonical = canonicalDepthSceneProfile(profile);
    if (canonical == QStringLiteral("aerial_terrain"))
    {
        return DepthSceneProfile::AerialTerrain;
    }
    if (canonical == QStringLiteral("orbital_object"))
    {
        return DepthSceneProfile::OrbitalObject;
    }
    if (canonical == QStringLiteral("custom"))
    {
        return DepthSceneProfile::Custom;
    }
    return DepthSceneProfile::Unknown;
}

inline bool isKnownDepthSceneProfile(const QString &profile)
{
    return classifyDepthSceneProfile(profile) != DepthSceneProfile::Unknown;
}

inline bool isOrbitalDepthSceneProfile(const QString &profile)
{
    return classifyDepthSceneProfile(profile) ==
        DepthSceneProfile::OrbitalObject;
}

/// Extends a persisted scene-profile batch while enforcing one canonical,
/// known profile. Callers should only submit frames that hold a usable role.
inline bool extendCanonicalDepthSceneProfileBatch(
    const QString &profile,
    QString *canonical_batch_profile)
{
    if (canonical_batch_profile == nullptr ||
        !isKnownDepthSceneProfile(profile))
    {
        return false;
    }

    const QString canonical = canonicalDepthSceneProfile(profile);
    if (canonical_batch_profile->isEmpty())
    {
        *canonical_batch_profile = canonical;
        return true;
    }
    return *canonical_batch_profile == canonical;
}

/// Maps persisted quality metadata to the only roles that downstream geometry
/// consumers may use. Missing or contradictory metadata never grants a role.
inline DepthFrameRole qualifyDepthFrameRole(
    const QString &acceptance,
    bool fusion_eligibility_known,
    bool fusion_eligible,
    const QString &status) noexcept
{
    const QString normalized_status = status.trimmed().toLower();
    if (normalized_status != QStringLiteral("completed"))
    {
        return DepthFrameRole::Excluded;
    }

    if (!fusion_eligibility_known)
    {
        return DepthFrameRole::Excluded;
    }

    const QString normalized_acceptance = acceptance.trimmed().toLower();
    if (normalized_acceptance == QStringLiteral("accepted") &&
        fusion_eligible)
    {
        return DepthFrameRole::Primary;
    }
    if (normalized_acceptance == QStringLiteral("validation_only"))
    {
        return DepthFrameRole::CoverageAuxiliary;
    }
    return DepthFrameRole::Excluded;
}

inline bool isPrimaryFusionFrame(DepthFrameRole role) noexcept
{
    return role == DepthFrameRole::Primary;
}

inline int minimumOrbitalPrimaryDepthFrameCount(
    int discovered_frame_count) noexcept
{
    const int normalized_frame_count = std::max(0, discovered_frame_count);
    const int quarter_coverage = normalized_frame_count / 4 +
        (normalized_frame_count % 4 == 0 ? 0 : 1);
    return std::max(3, quarter_coverage);
}

} // namespace xjw::mvs

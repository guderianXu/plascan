#include "PairSelectionPolicy.h"

namespace xjw
{
namespace matchphotos
{

PairSelectionPolicy makePairSelectionPolicy(PairSelectionPreset preset)
{
    PairSelectionPolicy policy;
    switch (preset)
    {
    case PairSelectionPreset::Fast:
        policy.exhaustiveMaxImages = 12;
        policy.sequenceWindow = 3;
        policy.maxPairs = 0;
        break;
    case PairSelectionPreset::HighAccuracy:
        policy.exhaustiveMaxImages = 30;
        policy.sequenceWindow = 8;
        policy.maxPairs = 0;
        break;
    case PairSelectionPreset::CpuCompatible:
        policy.exhaustiveMaxImages = 16;
        policy.sequenceWindow = 4;
        policy.maxPairs = 0;
        break;
    case PairSelectionPreset::DifficultTexture:
        policy.exhaustiveMaxImages = 24;
        policy.sequenceWindow = 6;
        policy.maxPairs = 0;
        break;
    case PairSelectionPreset::Auto:
        break;
    }
    return policy;
}

} // namespace matchphotos
} // namespace xjw

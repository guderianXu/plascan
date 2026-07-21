#pragma once

#include "IncrementalSfm.h"

namespace xjw
{

class InitialPairInitializer
{
  public:
    explicit InitialPairInitializer(IncrementalSfm &owner);

    std::vector<std::pair<ImageId, ImageId>> selectCandidates(int maxCandidates) const;
    bool initialize(ImageId id1, ImageId id2);
    void resetTrial(const SfmReconstruction &baseReconstruction);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw

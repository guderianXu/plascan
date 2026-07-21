#pragma once

#include "IncrementalSfm.h"

namespace xjw
{

class SfmBundleAdjustCoordinator
{
  public:
    explicit SfmBundleAdjustCoordinator(IncrementalSfm &owner);

    void run(bool localOnly = false, const std::vector<ImageId> &anchorIds = {});
    void iterative();
    int filterNegativeDepthPoints();

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw

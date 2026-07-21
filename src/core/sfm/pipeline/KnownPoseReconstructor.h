#pragma once

#include "IncrementalSfm.h"

namespace xjw
{

class KnownPoseReconstructor
{
  public:
    explicit KnownPoseReconstructor(IncrementalSfm &owner);

    IncrementalSfmResult run(SfmProgressCallback progressCb);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw

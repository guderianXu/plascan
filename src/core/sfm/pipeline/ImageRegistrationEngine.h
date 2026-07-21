#pragma once

#include "IncrementalSfm.h"

namespace xjw
{

class ImageRegistrationEngine
{
  public:
    explicit ImageRegistrationEngine(IncrementalSfm &owner);

    IncrementalSfmResult run(int totalImages, SfmProgressCallback progressCb);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw

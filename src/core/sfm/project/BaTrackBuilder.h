#pragma once

#include "project/BaInputBuilder.h"
#include "project/ProjectMatchInputReader.h"

namespace xjw::core::project
{

void appendBaTracks(const ProjectMatchInput &input, BaInputBuildResult *result);

} // namespace xjw::core::project

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace xjw::mvs::detail
{

/**
 * @brief Validate the complete binary PLY artifact emitted by dense-cloud refinement.
 *
 * Refinement outputs contain one scalar-only vertex element and no trailing elements.
 * Validation therefore checks the format, vertex schema, expected point count and exact
 * byte length before a staged file is allowed to replace an existing result.
 */
bool validateDenseCloudPlyArtifact(const std::filesystem::path &path,
                                   std::size_t expectedVertexCount,
                                   std::string *errorMessage = nullptr);

} // namespace xjw::mvs::detail

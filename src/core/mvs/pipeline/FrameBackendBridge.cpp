#include "MvsPipelineInternals.h"
#include "AdaptivePatchMatchBackend.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    namespace detail
    {

        bool estimateAdaptivePatchMatchLevel(int reference_index,
                                             const PatchMatchBackendRequest& request,
                                             DepthLevelResult& result,
                                             std::string* error_message)
        {
            AdaptivePatchMatchBackend backend(reference_index);
            return backend.estimate(request, result, error_message);
        }

    } // namespace detail
} // namespace xjw::mvs

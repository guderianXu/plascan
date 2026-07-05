#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

namespace xjw
{
namespace matchphotos
{

// 类 Metashape “匹配照片”命令的高层编排边界。
// 底层模块保持独立：overlap 生成候选影像对，feature_extractors 生成特征，
// feature_match 生成两两匹配，sfm 将验证后的匹配合并成多视图轨迹。
class MatchPhotosTask
{
public:
    explicit MatchPhotosTask(const MatchPhotosOptions &options = {});

    MatchPhotosResult run(const MatchPhotosContext &context) const;
    const MatchPhotosOptions &options() const;

private:
    MatchPhotosOptions _options;
};

} // namespace matchphotos
} // namespace xjw

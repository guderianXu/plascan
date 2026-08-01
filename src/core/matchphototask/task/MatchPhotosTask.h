#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

namespace xjw
{
namespace matchphotos
{

// 类 Metashape “匹配照片”命令的高层编排边界。
// 底层模块保持独立：overlap 生成候选影像对，image_matching 在任务内存中
// 完成 CUDA SIFT + LightGlue 并持久化逐影像 `.pimatch`，sfm 消费多视连接点轨迹。
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

# SuperGlue C++ 快速入门

`superglue/` 目录现在只保留 SuperGlue TorchScript 推理相关代码。匹配文件 I/O、CSV/COLMAP 导出、可视化和几何粗差剔除已经放到 `src/core/feature_match/` 公共模块，供 SuperGlue、LightGlue、传统匹配器等共用。

## 配置

```cpp
struct SuperGlueConfig
{
    std::string model_path;
    float match_threshold;
    int batch_size;
    int sinkhorn_iterations;
    int max_keypoints;
    bool use_cuda;
    int cuda_device_id;
    int num_threads;
    bool verbose;
};
```

## 基本使用

```cpp
#include "SuperGlueMatcher.h"

SuperGlueConfig config;
config.model_path = "superglue_outdoor_cpu.pt";
config.match_threshold = 0.2f;
config.use_cuda = false;
config.verbose = true;

superglue::SuperGlueMatcher matcher(config);
superglue::MatchResult result = matcher.match(img0, img1, kpts0, kpts1);
std::cout << "matches: " << result.numMatches << std::endl;
```

## 公共后处理

需要写 `.match`、导出 CSV/COLMAP 或做几何剔除时，不要在 `superglue/` 下新增实现，使用公共模块：

- `MatchFileIO.h/cpp`
- `MatchExportIO.h/cpp`
- `MatchGeometryFilter.h/cpp`
- `MatchVisualization.h/cpp`

示例：

```cpp
#include "MatchFileIO.h"
#include "MatchGeometryFilter.h"

xjw::feature_match::OutlierFilterConfig filterConfig;
auto filtered = xjw::feature_match::MatchGeometryFilter::filter(
    result,
    keypoints0,
    keypoints1,
    filterConfig);

xjw::feature_match::writeIndexedMatchFile(
    outputPath,
    image0Name,
    image1Name,
    filtered);
```

## 模型导出

```bash
python export_torchscript.py --weights outdoor --device cpu
python export_torchscript.py --weights outdoor --device cuda
```

输出文件示例：

- `superglue_outdoor_cpu.pt`
- `superglue_outdoor_cuda.pt`
- `superglue_indoor_cpu.pt`
- `superglue_indoor_cuda.pt`

# PlaMatch-HCT 核心源码边界

本目录保存用户自有、已经独立验证可用的 PlaMatch-HCT 算法核心，来源为：

```text
/home/guderian/code/metashape_code/对齐照片
```

接入范围仅包含特征与匹配所需的 `features`、`matching`、`gpu` CPU/CUDA/OpenCL 实现及其头文件。参考工程中的
几何验证、重建、相似变换、命令行解析、特征文件缓存和图像编解码没有迁入；这些职责分别复用 PlaScan 的
OpenCV 输入桥、`GeometryVerifyStage`、`TrackBuildStage` 和 `.pimatch` 存储。

为了让算法直接消费 PlaScan 已解码的 `cv::Mat`，`features.hpp/.cpp` 增加了内存 `Image` 重载；
`matching.hpp/.cpp` 增加了接受显式候选图、预建 coarse 索引和零拷贝特征指针 batch 的入口。其余算法常量、检测、MLDB、HCTree、
ratio、唯一性、方向行合并、局部一致性和森林削减逻辑保持在隔离边界内。大型源文件是为了保留已验证实现，
不按 PlaScan 普通业务文件的 400 行建议机械拆分。

PlaScan 适配代码位于上一级目录，修改通用接口或任务流程时应优先改适配层。若更新本目录算法源码，应同时
递增 `kPlaMatchHctAlgorithmVersion` 或 feature schema，并重新执行算法单测、双图真实数据和多图 coarse
预选验证。

# PlaScan recovered depth core

这里是 PlaScan MVS 模块自身的多视角深度实现，不是第三方库。算法来源于本机
`生成模型` 的 recovered production 路径，生产入口为
`metmodel::run_recovered_patchmatch_d4_scene_cuda`。

`neighbor_selection.cpp` 直接使用参考 tracked-scene 选邻：先以 float3 点记录和定向 region 的
包含边界筛选共同 track，再执行角度分带、条件数打分和共同观测排序。无 track 时明确失败；
不调用 PlaScan 旧选源器，也不引入参考工程的 legacy baseline fallback。
当前 recovered production 接受每个参考相机 1..16 个有效邻居：N=1 已有 strict-CUDA 双目捕获，
N=6..16 已由 South target 数值回归覆盖；N=2..5 复用同一 selector、packed-mask、CUDA cost 与
三层 voting 结构，但尚未逐计数声明 target 数值闭合。零邻居和超过 16 个邻居仍明确失败。

CUDA 内核由 `recovered_cuda_source.cu` 随 `mvs` target 直接编译。目录不包含、
安装或运行时加载 `kernels/*.ptx`；参考工程里保留的 PTX oracle 只用于其差分验证，
不属于 PlaScan 的生产资源。
源码 filter kernel 使用显式 `work_items` 保护最后一个不足 128 线程的 block，因此 d4/d8/d16
活动像素数不必是 128 的倍数；完整 block 保持参考运算路径，PTX oracle 仍保留原整块限制。

`metalign` 兼容层实现深度和 OOC 所需的相机、图像与数学接口，包括目标 Golub-Reinsch SVD。
JPEG 使用 libjpeg-turbo 3.1.2 的精确 IDCT/色彩转换，其余格式使用 OpenCV；灰度转换保持
`0.299 R + 0.587 G + 0.114 B` 后截断的参考公式。
已同步参考 CUDA 的随机深度、粗层范围和重投影成本中的显式 FFMA/RN 运算顺序。
输入、编译器和参考版本变化仍需重新比较；采用相同源码不等于已证明与 Metashape 输出一致。
公开 d4 工件的相机按参考 `f/cx/cy ÷ 4` 导出，清零畸变和焦距仿射差；不用 OpenCV 半像素缩放。
这修正了旧元数据中的 0.375 像素主点偏移，不改变已计算的深度平面和内部 OOC 输入。

`fusion.cpp`、`octree_prepare.cpp`、`ooc_pyramid_io.cpp` 与 `model_pipeline.cpp` 实现内部 OOC
金字塔、Morton 树、CUDA 直方图及隐式场求解；源码与算法常量保持集中以便参考核对。
`RecoveredModelInput` 将三层 voting-after-components 深度保存到工作区，模型消费者详见
`src/core/mesh/recovered_model/README.md`。此前占位的 `depth_link_compat.cpp` 已由完整 fusion 实现替代。

每次 scene-wide 运行都在工作区内创建带 UUID 的 `.recovered_patchmatch_store-*` 一次性目录，
并由作用域清理覆盖成功、失败和取消路径。旧版本遗留的固定 `.recovered_patchmatch_store` 不再阻塞
新任务，也不会由新任务擅自删除。`recovered_model_input` 先在同级 staging 中完整写入并校验，
强制重算时再以备份/重命名事务替换旧版本；新输入未完成时继续保留上一份可用输入。

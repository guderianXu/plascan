#pragma once

// Qt 的关键字宏会与 LibTorch 头文件中的标识符冲突。所有同时包含 Qt 和
// LibTorch 的模块在引入 Torch 前统一包含此头文件。
#ifdef slots
#undef slots
#endif

#ifdef signals
#undef signals
#endif

#ifdef emit
#undef emit
#endif

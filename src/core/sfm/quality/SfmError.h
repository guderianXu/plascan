#pragma once

/**
 * @file SfmError.h
 * @brief SfM 核心层的稳定错误分类。
 *
 * code 用于程序分支，diagnostic 用于日志/GUI。调用方不得通过解析 diagnostic 文本
 * 判断错误类型；新增失败类别时应优先扩展枚举。
 */

#include <string>

namespace xjw
{

enum class SfmErrorCode
{
    None, ///< 无错误。
    InvalidInput, ///< 图像、关键点、匹配或参数契约非法。
    CameraUnavailable, ///< 已知相机缺失或不可解析。
    InitializationFailed, ///< 没有初始对通过双视几何门控。
    RegistrationFailed, ///< 增量 PnP 无法继续注册影像。
    OptimizationFailed ///< BA/重三角化后模型不可用。
};

struct SfmError
{
    SfmErrorCode code = SfmErrorCode::None;
    std::string diagnostic; ///< 包含阶段、计数和阈值的可读诊断。

    explicit operator bool() const
    {
        return code != SfmErrorCode::None;
    }
};

} // namespace xjw

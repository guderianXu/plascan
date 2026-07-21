// =============================================================================
// 文件: cli_common.h
// 功能: CLI 工具公共基础设施 (基于 CLI11, 零 Qt 依赖)
// =============================================================================
#pragma once

#include "CLI11.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

namespace cli
{

enum ExitCode
{
    EXIT_OK       = 0,
    EXIT_ARG_ERR  = 1,
    EXIT_IO_ERR   = 2,
    EXIT_ALGO_ERR = 3,
};

// 统一错误输出
inline void fatal(const std::string &msg, int code = EXIT_ARG_ERR)
{
    fprintf(stderr, "错误: %s\n", msg.c_str());
    std::exit(code);
}

} // namespace cli

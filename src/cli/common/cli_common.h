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

// Keep the user-facing shell of every executable consistent. Domain-specific
// options remain close to their command, while help, version and exit-code
// guidance are defined once here.
inline void configureApp(CLI::App &app)
{
    app.set_help_flag("-h,--help", "显示帮助信息并退出");
#ifdef PLASCAN_VERSION
    app.set_version_flag("--version",
                         std::string("PlaScan ") + PLASCAN_VERSION,
                         "显示版本信息并退出");
#endif
    app.footer("退出码：0 成功；1 参数错误；2 输入/输出错误；3 算法执行错误。");
    app.get_formatter()->column_width(36);
    app.get_formatter()->right_column_width(100);
}

// 统一错误输出
inline void fatal(const std::string &msg, int code = EXIT_ARG_ERR)
{
    fprintf(stderr, "错误: %s\n", msg.c_str());
    std::exit(code);
}

} // namespace cli

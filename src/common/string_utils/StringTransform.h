#pragma once

#include <string>
#include <string_view>

namespace xjw::common::string_utils
{

/**
 * @brief 返回文本副本，并仅将 ASCII 范围 A-Z 转换为 a-z。
 * @param text 待转换的文本；非 ASCII 字节保持不变。
 * @return 独立持有内容的转换结果。
 */
std::string asciiLowerCopy(std::string_view text);

/**
 * @brief 移除文本两端的 ASCII 空白字符。
 * @param text 待裁剪的文本；ASCII 空格、\t、\n、\r、\f、\v 会被识别为空白。
 * @return 独立持有内容的裁剪结果，内部字符和非 ASCII 字节保持不变。
 */
std::string trimAsciiWhitespace(std::string_view text);

/**
 * @brief 判断文本是否以给定后缀结尾，比较时仅忽略 ASCII A-Z/a-z 的大小写。
 * @param text 待检查的文本。
 * @param suffix 待匹配的后缀；空后缀总是匹配。
 * @return 后缀按 ASCII 大小写不敏感规则匹配时返回 true；非 ASCII 字节按原值精确比较。
 */
bool endsWithAsciiIgnoreCase(std::string_view text, std::string_view suffix);

} // namespace xjw::common::string_utils

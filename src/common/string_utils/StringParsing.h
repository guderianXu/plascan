#pragma once

#include <string>
#include <vector>

namespace xjw::common::string_utils
{

/**
 * @brief 按数值子串从文本中尽力提取所有浮点数。
 * @param text 待扫描的文本。
 * @param out 输出解析出的所有 double 值；调用开始时会清空。
 * @return 至少解析到一个数值时返回 true。
 * @note 采用数值子串扫描，因此 k1/p1 等标识符中的数字也可能被提取。
 * @note 转换时发生 out_of_range 的匹配项会被忽略，其余有效匹配仍会保留。
 */
bool extractDoublesFromText(const std::string &text, std::vector<double> &out);

} // namespace xjw::common::string_utils

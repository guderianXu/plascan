#ifndef SUPERGLUEMATCH_IO_H
#define SUPERGLUEMATCH_IO_H

#include "../match.h"

#include <QString>
#include <QFile>
#include <QDataStream>
#include <vector>
#include <string>

/**
 * @brief SuperGlueMatchIO - SuperGlue匹配结果二进制IO
 * 
 * 功能：将SuperGlue的匹配结果序列化为二进制文件，便于持久化存储和快速读取
 * 
 * 文件格式（小端序）：
 *  - magic: 4 字节 ASCII 'SGMT' (SuperGlue Match)
 *  - version: quint32 (当前为 1)
 *  - image0_name_len: quint32
 *  - image0_name: UTF-8 bytes
 *  - image1_name_len: quint32
 *  - image1_name: UTF-8 bytes
 *  - num_matches: qint32 (有效匹配数)
 *  - num_keypoints0: qint32 (图像0的关键点总数)
 *  - num_keypoints1: qint32 (图像1的关键点总数)
 *  
 *  对于每个图像0的关键点 i (共 num_keypoints0 个):
 *    - match_idx: qint32 (匹配到图像1的索引，-1表示无匹配)
 *    - match_score: float (匹配置信度)
 *  
 *  对于每个图像1的关键点 j (共 num_keypoints1 个):
 *    - match_idx: qint32 (匹配到图像0的索引，-1表示无匹配)
 *    - match_score: float (匹配置信度)
 * 
 * 注意：
 *  - 本文件只存储匹配关系，不存储关键点坐标和描述符（它们由 FeatureFileIO 保存）
 *  - 读取时需先加载两张影像的特征点文件，再加载匹配文件
 *  - 兼容 MatchViewerDialog 的 .match 文件格式要求
 */
class SuperGlueMatchIO 
{
public:
    /**
     * @brief 写入匹配结果到文件
     * @param path 输出文件路径（.sgmatch 扩展名）
     * @param image0_name 图像0名称
     * @param image1_name 图像1名称
     * @param result SuperGlue匹配结果
     * @return 成功返回true，失败返回false
     */
    static bool write(const QString& path, 
                     const QString& image0_name,
                     const QString& image1_name,
                     const xjw::feature_match::MatchResult& result);

    /**
     * @brief 从文件读取匹配结果
     * @param path 输入文件路径
     * @param image0_name [out] 图像0名称
     * @param image1_name [out] 图像1名称
     * @param result [out] 匹配结果
     * @return 成功返回true，失败返回false
     */
    static bool read(const QString& path,
                    QString& image0_name,
                    QString& image1_name,
                    xjw::feature_match::MatchResult& result);
    
    /**
     * @brief 将匹配结果导出为CSV格式（便于调试和兼容性）
     * @param path 输出CSV文件路径
     * @param result 匹配结果
     * @return 成功返回true，失败返回false
     */
    static bool exportToCSV(const QString& path,
                           const xjw::feature_match::MatchResult& result);
    
    /**
     * @brief 将匹配结果导出为COLMAP格式的matches.txt
     * @param path 输出文件路径
     * @param image0_name 图像0名称
     * @param image1_name 图像1名称
     * @param result 匹配结果
     * @return 成功返回true，失败返回false
     */
    static bool exportToCOLMAP(const QString& path,
                              const QString& image0_name,
                              const QString& image1_name,
                              const xjw::feature_match::MatchResult& result);
};

#endif // SUPERGLUEMATCH_IO_H

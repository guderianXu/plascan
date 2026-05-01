// =============================================================================
// 文件: SuperGlueMatchIO.cpp
// 说明:
//   实现 SuperGlue 匹配结果的二进制读写和多种格式导出。
//   文件头魔数字为 SGMT，小端序，与 QDataStream::Qt_5_15 版本对齐。
// =============================================================================
// 防止 Qt 的 slots/signals 宏与 LibTorch 冲突
#define QT_NO_KEYWORDS

#include "SuperGlueMatchIO.h"
#include "SuperGlueMatcher.h"
#include "log/Logger.h"
#include <QFile>
#include <QDataStream>
#include <QTextStream>

bool SuperGlueMatchIO::write(const QString& path,
                             const QString& image0_name,
                             const QString& image1_name,
                             const xjw::feature_match::MatchResult& result)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        LOG_WARN("无法打开文件用于写入: %s", qUtf8Printable(path));
        return false;
    }
    
    QDataStream out(&file);
    out.setVersion(QDataStream::Qt_5_15); // 与读取时版本对齐
    
    // --- 文件头 ---
    out.writeRawData("SGMT", 4); // SuperGlue Match Table
    out << quint32(1);           // 文件格式版本号
    
    // --- 影像名 ---
    QByteArray img0_bytes = image0_name.toUtf8(); // 统一 UTF-8 编码
    QByteArray img1_bytes = image1_name.toUtf8();
    out << quint32(img0_bytes.size());
    out.writeRawData(img0_bytes.data(), img0_bytes.size());
    out << quint32(img1_bytes.size());
    out.writeRawData(img1_bytes.data(), img1_bytes.size());
    
    // --- 匹配统计信息 ---
    out << qint32(result.numMatches);           // 有效匹配对数
    out << qint32(result.matches0.size());      // 图像0 关键点数
    out << qint32(result.matches1.size());      // 图像1 关键点数
    
    // --- 图像0 匹配详情 (match_idx, score) x num_keypoints0 ---
    for (size_t i = 0; i < result.matches0.size(); ++i) {
        out << qint32(result.matches0[i]);          // -1 表示无匹配
        out << float(result.matchingScores0[i]);    // 置信度分数 [0,1]
    }
    
    // --- 图像1 匹配详情 (match_idx, score) x num_keypoints1 ---
    for (size_t i = 0; i < result.matches1.size(); ++i) {
        out << qint32(result.matches1[i]);
        out << float(result.matchingScores1[i]);
    }
    
    file.close();
    return true;
}

bool SuperGlueMatchIO::read(const QString& path,
                           QString& image0_name,
                           QString& image1_name,
                           xjw::feature_match::MatchResult& result)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        LOG_WARN("无法打开文件用于读取: %s", qUtf8Printable(path));
        return false;
    }
    
    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);
    
    // --- 验证文件头 ---
    char magic[4];
    if (in.readRawData(magic, 4) != 4 || strncmp(magic, "SGMT", 4) != 0) {
        LOG_WARN("无效的文件格式（magic错误）: %s", qUtf8Printable(path));
        return false;
    }
    
    // --- 读取版本号 ---
    quint32 version;
    in >> version;
    if (version != 1) {
        LOG_WARN("不支持的文件版本: %u", version);
        return false;
    }
    
    // --- 读取影像名 ---
    quint32 img0_len, img1_len;
    in >> img0_len;
    QByteArray img0_bytes(img0_len, 0);
    in.readRawData(img0_bytes.data(), img0_len);
    image0_name = QString::fromUtf8(img0_bytes);
    
    in >> img1_len;
    QByteArray img1_bytes(img1_len, 0);
    in.readRawData(img1_bytes.data(), img1_len);
    image1_name = QString::fromUtf8(img1_bytes);
    
    // --- 读取匹配统计信息 ---
    qint32 num_matches, num_kp0, num_kp1;
    in >> num_matches >> num_kp0 >> num_kp1;
    
    result.numMatches = num_matches;
    result.matches0.resize(num_kp0);
    result.matchingScores0.resize(num_kp0);
    result.matches1.resize(num_kp1);
    result.matchingScores1.resize(num_kp1);
    
    // --- 读取图像0 匹配详情 ---
    for (int i = 0; i < num_kp0; ++i) {
        qint32 match_idx;
        float score;
        in >> match_idx >> score;
        result.matches0[i] = match_idx;
        result.matchingScores0[i] = score;
    }
    
    // --- 读取图像1 匹配详情 ---
    for (int i = 0; i < num_kp1; ++i) {
        qint32 match_idx;
        float score;
        in >> match_idx >> score;
        result.matches1[i] = match_idx;
        result.matchingScores1[i] = score;
    }
    
    // --- 重建 cv::DMatch 列表 ---
    // 匹配文件中只存储索引，读回时重新构建 DMatch 以支持 OpenCV 匹配显示
    result.cvMatches.clear();
    for (int i = 0; i < num_kp0; ++i) {
        if (result.matches0[i] >= 0) {
            cv::DMatch m;
            m.queryIdx = i;
            m.trainIdx = result.matches0[i];
            m.distance = 1.0f - result.matchingScores0[i]; // 转换为距离
            result.cvMatches.push_back(m);
        }
    }
    
    file.close();
    return true;
}

bool SuperGlueMatchIO::exportToCSV(const QString& path,
                                  const xjw::feature_match::MatchResult& result)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        LOG_WARN("无法打开CSV文件用于写入: %s", qUtf8Printable(path));
        return false;
    }
    
    QTextStream out(&file);
    out << "idx0,idx1,score\n";
    
    for (size_t i = 0; i < result.matches0.size(); ++i) {
        if (result.matches0[i] >= 0) {
            out << i << "," 
                << result.matches0[i] << "," 
                << result.matchingScores0[i] << "\n";
        }
    }
    
    file.close();
    return true;
}

bool SuperGlueMatchIO::exportToCOLMAP(const QString& path,
                                     const QString& image0_name,
                                     const QString& image1_name,
                                     const xjw::feature_match::MatchResult& result)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        LOG_WARN("无法打开COLMAP文件用于写入: %s", qUtf8Printable(path));
        return false;
    }
    
    QTextStream out(&file);
    out << "# SuperGlue matches export\n";
    out << "# image0: " << image0_name << "\n";
    out << "# image1: " << image1_name << "\n";
    out << "# num_matches: " << result.numMatches << "\n";
    out << "\n";
    
    // COLMAP matches.txt format: image0_idx image1_idx [matches as pairs]
    out << image0_name << " " << image1_name << "\n";
    
    for (const auto& m : result.cvMatches) {
        out << m.queryIdx << " " << m.trainIdx << " ";
    }
    out << "\n";
    
    file.close();
    return true;
}

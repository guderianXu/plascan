#include "DepthMatStorage.h"

#include <array>
#include <cstring>
#include <utility>

#include <QFile>
#include <opencv2/core.hpp>

namespace xjw::core::project
{
    namespace
    {

        constexpr std::array<char, 16> kFastDepthMatMagic{
            'P', 'L', 'A', 'S', 'D', 'E', 'P', 'T', 'H', 'M', 'A', 'T', '0', '1', '\0', '\0'};

        struct FastDepthMatHeader
        {
            char magic[16] = {};
            qint32 rows = 0;
            qint32 cols = 0;
            qint32 type = 0;
            quint32 reserved = 0; // Former ABI padding; readers ignore legacy non-zero bytes.
            quint64 dataBytes = 0;
        };

        static_assert(sizeof(FastDepthMatHeader) == 40,
                      "Fast depth matrix header layout must remain backward compatible");

        xjw::common::OperationResult loadFastDepthMatStorage(const QString& path, cv::Mat* matrix)
        {
            if (!matrix)
            {
                return {false, QStringLiteral("内部错误：矩阵输出参数无效")};
            }

            QFile file(path);
            if (!file.open(QIODevice::ReadOnly))
            {
                return {false, QStringLiteral("无法读取二进制深度文件：%1").arg(path)};
            }

            FastDepthMatHeader header;
            if (file.read(reinterpret_cast<char*>(&header), sizeof(header)) != static_cast<qint64>(sizeof(header)))
            {
                return {false, QStringLiteral("二进制深度文件头不完整：%1").arg(path)};
            }

            if (std::memcmp(header.magic, kFastDepthMatMagic.data(), kFastDepthMatMagic.size()) != 0)
            {
                return {false, QStringLiteral("二进制深度文件标识无效：%1").arg(path)};
            }
            if (header.rows <= 0 || header.cols <= 0 || header.dataBytes == 0)
            {
                return {false, QStringLiteral("二进制深度文件尺寸无效：%1").arg(path)};
            }

            const size_t elemSize = CV_ELEM_SIZE(header.type);
            if (elemSize == 0)
            {
                return {false, QStringLiteral("二进制深度文件类型无效：%1").arg(path)};
            }

            const quint64 expectedBytes =
                static_cast<quint64>(header.rows) * static_cast<quint64>(header.cols) * static_cast<quint64>(elemSize);
            if (header.dataBytes != expectedBytes)
            {
                return {false, QStringLiteral("二进制深度文件大小不匹配：%1").arg(path)};
            }

            cv::Mat loaded(header.rows, header.cols, header.type);
            if (file.read(reinterpret_cast<char*>(loaded.data), static_cast<qint64>(header.dataBytes)) !=
                static_cast<qint64>(header.dataBytes))
            {
                return {false, QStringLiteral("二进制深度文件数据不完整：%1").arg(path)};
            }

            *matrix = std::move(loaded);
            return {true, QString()};
        }

    } // namespace

    xjw::common::OperationResult loadDepthMatStorage(const QString& path, cv::Mat* matrix)
    {
        if (!path.endsWith(QStringLiteral(".bin"), Qt::CaseInsensitive))
        {
            return {false, QStringLiteral("不支持的深度矩阵格式：%1").arg(path)};
        }
        return loadFastDepthMatStorage(path, matrix);
    }

    xjw::common::OperationResult writeDepthMatStorage(const QString& path, const cv::Mat& matrix)
    {
        if (matrix.empty())
        {
            return {false, QStringLiteral("矩阵为空，无法写入二进制深度文件：%1").arg(path)};
        }

        QFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        {
            return {false, QStringLiteral("无法写入二进制深度文件：%1").arg(path)};
        }

        const cv::Mat contiguous = matrix.isContinuous() ? matrix : matrix.clone();
        FastDepthMatHeader header;
        std::memcpy(header.magic, kFastDepthMatMagic.data(), kFastDepthMatMagic.size());
        header.rows = contiguous.rows;
        header.cols = contiguous.cols;
        header.type = contiguous.type();
        header.dataBytes = static_cast<quint64>(contiguous.total() * contiguous.elemSize());

        if (file.write(reinterpret_cast<const char*>(&header), sizeof(header)) != static_cast<qint64>(sizeof(header)) ||
            file.write(reinterpret_cast<const char*>(contiguous.data), static_cast<qint64>(header.dataBytes)) !=
                static_cast<qint64>(header.dataBytes))
        {
            return {false, QStringLiteral("写入二进制深度文件失败：%1").arg(path)};
        }

        return {true, QString()};
    }

} // namespace xjw::core::project

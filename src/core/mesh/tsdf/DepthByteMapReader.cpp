#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    bool loadByteMap(const QString& path, const cv::Size& size, cv::Mat* map, QString* reason, bool allow_resize)
    {
        if (!map)
        {
            return false;
        }
        *map = xjw::common::io::readImage(xjw::common::io::toUtf8Path(path), cv::IMREAD_GRAYSCALE);
        if (map->empty())
        {
            if (reason)
            {
                *reason = QStringLiteral("byte map cannot be read");
            }
            return false;
        }
        if (map->type() == CV_8UC1 && map->size() != size && allow_resize)
        {
            cv::resize(*map, *map, size, 0.0, 0.0, cv::INTER_NEAREST);
        }
        if (map->type() != CV_8UC1 || map->size() != size)
        {
            if (reason)
            {
                *reason = QStringLiteral("expected CV_8UC1 %1x%2, got type=%3 %4x%5")
                              .arg(size.width)
                              .arg(size.height)
                              .arg(map->type())
                              .arg(map->cols)
                              .arg(map->rows);
            }
            return false;
        }
        return true;
    }
} // namespace xjw::mesh::tsdf_detail

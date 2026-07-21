#pragma once

#include <QVector>

namespace xjw::aerial_triangulation
{

struct AdaptiveFocalCandidate
{
    double focalScale = 1.0;
    bool success = false;
    int registeredImages = 0;
    int points3D = 0;
    double meanReprojectionError = 0.0;
};

class AdaptiveFocalSearch
{
public:
    static int selectBestCandidate(const QVector<AdaptiveFocalCandidate> &candidates,
                                   int totalImages);
};

} // namespace xjw::aerial_triangulation

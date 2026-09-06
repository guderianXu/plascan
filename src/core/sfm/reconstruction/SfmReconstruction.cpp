#include "SfmReconstruction.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace xjw
{

    // ============================================================
    // 图像管理
    // ============================================================

    void SfmReconstruction::addImage(const ImageData& data)
    {
        imageDataMap[data.id] = data;
    }

    /**
     * @brief 将新 `ImageData` 添加到重建容器中。
     *
     * 仅做简单插入；若 id 冲突则覆盖旧数据。
     */

    ImageData& SfmReconstruction::image(ImageId id)
    {
        auto it = imageDataMap.find(id);
        if (it == imageDataMap.end())
        {
            throw std::out_of_range("SfmReconstruction::image: invalid imageId");
        }
        return it->second;
    }

    const ImageData& SfmReconstruction::image(ImageId id) const
    {
        auto it = imageDataMap.find(id);
        if (it == imageDataMap.end())
        {
            throw std::out_of_range("SfmReconstruction::image: invalid imageId");
        }
        return it->second;
    }

    bool SfmReconstruction::hasImage(ImageId id) const
    {
        return imageDataMap.count(id) > 0;
    }

    bool SfmReconstruction::isRegistered(ImageId id) const
    {
        auto it = imageDataMap.find(id);
        return (it != imageDataMap.end()) && it->second.registered;
    }

    std::vector<ImageId> SfmReconstruction::allImageIds() const
    {
        std::vector<ImageId> ids;
        ids.reserve(imageDataMap.size());
        for (auto& [id, imageData] : imageDataMap)
        {
            (void)imageData;
            ids.push_back(id);
        }
        return ids;
    }

    std::vector<ImageId> SfmReconstruction::registeredImageIds() const
    {
        std::vector<ImageId> ids;
        for (auto& [id, imageData] : imageDataMap)
        {
            if (imageData.registered)
            {
                ids.push_back(id);
            }
        }
        return ids;
    }

    size_t SfmReconstruction::numRegisteredImages() const
    {
        size_t count = 0;
        for (auto& [imageId, imageData] : imageDataMap)
        {
            (void)imageId;
            if (imageData.registered)
            {
                ++count;
            }
        }
        return count;
    }

    void SfmReconstruction::registerImage(ImageId imageId, const FramePinholeCamera& camera)
    {
        auto it = imageDataMap.find(imageId);
        if (it == imageDataMap.end())
        {
            return;
        }
        it->second.registered = true;
        cameraMap[imageId] = camera;
    }

    void SfmReconstruction::deregisterImage(ImageId imageId)
    {
        auto it = imageDataMap.find(imageId);
        if (it != imageDataMap.end())
        {
            it->second.registered = false;
        }
        cameraMap.erase(imageId);
    }

    // ============================================================
    // 相机管理
    // ============================================================

    FramePinholeCamera& SfmReconstruction::camera(ImageId imageId)
    {
        auto it = cameraMap.find(imageId);
        if (it == cameraMap.end())
        {
            throw std::out_of_range("SfmReconstruction::camera: no camera for imageId");
        }
        return it->second;
    }

    const FramePinholeCamera& SfmReconstruction::camera(ImageId imageId) const
    {
        auto it = cameraMap.find(imageId);
        if (it == cameraMap.end())
        {
            throw std::out_of_range("SfmReconstruction::camera: no camera for imageId");
        }
        return it->second;
    }

    bool SfmReconstruction::hasCamera(ImageId imageId) const
    {
        return cameraMap.count(imageId) > 0;
    }

    // ============================================================
    // 三维点管理
    // ============================================================

    Point3DId SfmReconstruction::addPoint3D(const ScenePoint3D& point)
    {
        Point3DId id = (point.id != kInvalidPoint3DId) ? point.id : nextPoint3DId++;
        ScenePoint3D storedPoint = point;
        storedPoint.id = id;
        inactivePoint3DMap.erase(id);
        point3DMap[id] = std::move(storedPoint);
        // 保证下一次自动分配不会冲突
        if (id >= nextPoint3DId)
        {
            nextPoint3DId = id + 1;
        }
        return id;
    }

    Point3DId SfmReconstruction::addPoint3DWithTrack(const std::array<double, 3>& xyz, const Track& track)
    {
        ScenePoint3D pt;
        pt.xyz = xyz;
        pt.track = track;
        Point3DId id = addPoint3D(pt);

        // 同步更新每个观测图像的 point3DIds
        for (auto& trackElement : track.elements)
        {
            auto imageIt = imageDataMap.find(trackElement.imageId);
            if (imageIt == imageDataMap.end())
            {
                continue;
            }
            auto& point3DIds = imageIt->second.point3DIds;
            if (trackElement.featureIdx < point3DIds.size())
            {
                point3DIds[trackElement.featureIdx] = id;
            }
        }

        return id;
    }

    ScenePoint3D& SfmReconstruction::point3D(Point3DId id)
    {
        auto it = point3DMap.find(id);
        if (it == point3DMap.end())
        {
            throw std::out_of_range("SfmReconstruction::point3D: invalid point3DId");
        }
        return it->second;
    }

    const ScenePoint3D& SfmReconstruction::point3D(Point3DId id) const
    {
        auto it = point3DMap.find(id);
        if (it == point3DMap.end())
        {
            throw std::out_of_range("SfmReconstruction::point3D: invalid point3DId");
        }
        return it->second;
    }

    bool SfmReconstruction::hasPoint3D(Point3DId id) const
    {
        return point3DMap.count(id) > 0;
    }

    bool SfmReconstruction::hasInactivePoint3D(Point3DId id) const
    {
        return inactivePoint3DMap.count(id) > 0;
    }

    bool SfmReconstruction::deactivatePoint3D(Point3DId id)
    {
        auto node = point3DMap.extract(id);
        if (node.empty())
        {
            return false;
        }
        inactivePoint3DMap.erase(id);
        inactivePoint3DMap.insert(std::move(node));
        return true;
    }

    bool SfmReconstruction::restorePoint3DWithTrack(Point3DId id,
                                                    const std::array<double, 3>& xyz,
                                                    const Track& track,
                                                    double error)
    {
        auto node = inactivePoint3DMap.extract(id);
        if (node.empty())
        {
            return false;
        }

        ScenePoint3D& point = node.mapped();
        point.xyz = xyz;
        point.track = track;
        point.error = error;
        point3DMap.erase(id);
        point3DMap.insert(std::move(node));

        for (const TrackElement& trackElement : track.elements)
        {
            auto imageIt = imageDataMap.find(trackElement.imageId);
            if (imageIt == imageDataMap.end() || trackElement.featureIdx >= imageIt->second.point3DIds.size())
            {
                continue;
            }
            imageIt->second.point3DIds[trackElement.featureIdx] = id;
        }
        return true;
    }

    void SfmReconstruction::deletePoint3D(Point3DId id)
    {
        auto it = point3DMap.find(id);
        if (it == point3DMap.end())
        {
            auto inactiveIt = inactivePoint3DMap.find(id);
            if (inactiveIt == inactivePoint3DMap.end())
            {
                return;
            }
            for (const TrackElement& trackElement : inactiveIt->second.track.elements)
            {
                auto imageIt = imageDataMap.find(trackElement.imageId);
                if (imageIt == imageDataMap.end() || trackElement.featureIdx >= imageIt->second.point3DIds.size())
                {
                    continue;
                }
                Point3DId& pointId = imageIt->second.point3DIds[trackElement.featureIdx];
                if (pointId == id)
                {
                    pointId = kInvalidPoint3DId;
                }
            }
            inactivePoint3DMap.erase(inactiveIt);
            return;
        }

        // 清理关联观测
        for (auto& trackElement : it->second.track.elements)
        {
            auto imageIt = imageDataMap.find(trackElement.imageId);
            if (imageIt == imageDataMap.end())
            {
                continue;
            }
            auto& point3DIds = imageIt->second.point3DIds;
            if (trackElement.featureIdx < point3DIds.size() && point3DIds[trackElement.featureIdx] == id)
            {
                point3DIds[trackElement.featureIdx] = kInvalidPoint3DId;
            }
        }

        point3DMap.erase(it);
    }

    void SfmReconstruction::clearPoints3D()
    {
        point3DMap.clear();
        inactivePoint3DMap.clear();
        nextPoint3DId = 0;
        for (auto& [imageId, image] : imageDataMap)
        {
            (void)imageId;
            std::fill(image.point3DIds.begin(), image.point3DIds.end(), kInvalidPoint3DId);
        }
    }

    bool SfmReconstruction::removeObservation(Point3DId id, ImageId imageId, FeatureIdx featureIdx)
    {
        auto pointIt = point3DMap.find(id);
        if (pointIt == point3DMap.end())
        {
            return false;
        }

        auto& elements = pointIt->second.track.elements;
        const auto elementIt = std::find_if(elements.begin(),
                                            elements.end(),
                                            [imageId, featureIdx](const TrackElement& element)
                                            { return element.imageId == imageId && element.featureIdx == featureIdx; });
        if (elementIt == elements.end())
        {
            return false;
        }

        auto imageIt = imageDataMap.find(imageId);
        if (imageIt != imageDataMap.end() && featureIdx < imageIt->second.point3DIds.size() &&
            imageIt->second.point3DIds[featureIdx] == id)
        {
            imageIt->second.point3DIds[featureIdx] = kInvalidPoint3DId;
        }
        elements.erase(elementIt);
        return true;
    }

    std::vector<Point3DId> SfmReconstruction::allPoint3DIds() const
    {
        std::vector<Point3DId> ids;
        ids.reserve(point3DMap.size());
        for (auto& [id, point] : point3DMap)
        {
            (void)point;
            ids.push_back(id);
        }
        return ids;
    }

    // ============================================================
    // 统计
    // ============================================================

    double SfmReconstruction::meanReprojError() const
    {
        if (point3DMap.empty())
        {
            return 0.0;
        }

        double sum = 0.0;
        for (auto& [pointId, point] : point3DMap)
        {
            (void)pointId;
            sum += point.error;
        }
        return sum / static_cast<double>(point3DMap.size());
    }

    std::string SfmReconstruction::summary() const
    {
        std::ostringstream oss;
        oss << "SfmReconstruction: " << numRegisteredImages() << "/" << numImages() << " images registered, "
            << numPoints3D() << " 3D points, " << "mean reproj error = " << meanReprojError() << " px";
        return oss.str();
    }

} // namespace xjw

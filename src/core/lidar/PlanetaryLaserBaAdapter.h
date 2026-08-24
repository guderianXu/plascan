#pragma once

#include "BundleAdjustSolver.h"
#include "PlanetaryLaserShot.h"

#include <string>
#include <vector>

namespace xjw
{
namespace lidar
{

/**
 * @brief 行星激光 shot 到静态 frame-camera BA 的适配选项。
 *
 * 当前适配器不执行坐标转换。cameraCoordinateFrame 必须与数据集
 * reference.bodyFixedFrame 完全一致，调用方需先把相机、普通 tracks 和激光落点
 * 放入同一个求解坐标系。imageAliasesByCameraIndex 与 BA cameras 同序，每台相机
 * 可以提供绝对路径、文件名、工程 ID 或 ISIS serial number 等多个稳定别名。
 */
struct PlanetaryLaserBaAdapterOptions
{
    std::vector<std::vector<std::string>> imageAliasesByCameraIndex;
    std::string cameraCoordinateFrame;
    std::string cameraSensorFrame;
    bool confirmUnknownSensorModelIsFrame = false;
    bool confirmUnknownRangeTypeIsOneWay = false;
    bool allowUnmappedShots = false;
    bool allowUnmappedMeasuredImages = false;
};

struct PlanetaryLaserBaAdapterSummary
{
    int totalShots = 0;
    int acceptedShots = 0;
    int fixedPointShots = 0;
    int constrainedPointShots = 0;
    int freePointShots = 0;
    int skippedUnmappedShots = 0;
    int measuredImageObservations = 0;
    int ignoredProjectedMeasures = 0;
    int ignoredUnmappedMeasuredImages = 0;
};

/**
 * @brief 构建独立激光测距约束，不修改普通摄影测量 BATrack。
 *
 * 仅支持静态 frame camera 和已经换算为单程距离的观测。Line-scan、往返距离、
 * 坐标系不一致、同时影像映射歧义都会被明确拒绝。ISIS projected/virtual measure
 * 永远不会进入 measuredImageObservations。
 */
bool buildPlanetaryLaserRangeConstraints(
    const PlanetaryLaserDataset &dataset,
    const PlanetaryLaserBaAdapterOptions &options,
    std::vector<BALaserRangeConstraint> *constraints,
    PlanetaryLaserBaAdapterSummary *summary = nullptr,
    std::string *errorMessage = nullptr);

} // namespace lidar
} // namespace xjw

#pragma once

namespace xjw
{

class IncrementalSfm;

struct HierarchicalBaRunSummary
{
    bool attempted = false;
    int plannedBlocks = 0;
    int appliedBlocks = 0;
    int updatedCameras = 0;
    int updatedPoints = 0;
    double totalSeconds = 0.0;

    bool applied() const
    {
        return appliedBlocks > 0;
    }
};

/**
 * @brief 大型重建的重叠块 BA 调度器。
 *
 * 块内相机和点并行优化；重叠相机固定在进入本轮时的共同坐标系中，核心相机
 * 只由唯一所属块写回。最终全局 BA 再统一共享内参与块间接缝。
 */
class HierarchicalBundleAdjuster
{
  public:
    explicit HierarchicalBundleAdjuster(IncrementalSfm &owner);

    HierarchicalBaRunSummary run();

    static bool shouldRun(bool enabled,
                          int registeredImageCount,
                          int minimumImageCount,
                          bool refineCameraPose);

    static int resolveWorkerCount(int blockCount,
                                  int totalThreadCount,
                                  int configuredMaximum,
                                  bool concurrentBackendAvailable);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw

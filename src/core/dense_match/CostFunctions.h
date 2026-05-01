// =============================================================================
// 文件: CostFunctions.h
// 功能: 密集匹配代价函数声明（CPU + CUDA）
// =============================================================================
#pragma once

#include "DenseMatchTypes.h"
#include <opencv2/core.hpp>
#include <vector>

namespace xjw::dense_match
{

using CostVolume = std::vector<cv::Mat>;

float computeCost(const uchar *left, const uchar *right,
                  int x, int y, int d, int kernelW, int kernelH,
                  int imgW, int imgH, CostFunction func);

CostVolume computeCostVolume(const cv::Mat &left, const cv::Mat &right,
                             int minDisp, int maxDisp,
                             int kernelW, int kernelH,
                             CostFunction func, int numThreads = 1);

#ifdef DM_ENABLE_CUDA
CostVolume computeCostVolumeCUDA(const cv::Mat &left, const cv::Mat &right,
                                 int minDisp, int maxDisp,
                                 int kernelW, int kernelH,
                                 CostFunction func, int cudaDevice = 0);
#endif

} // namespace xjw::dense_match

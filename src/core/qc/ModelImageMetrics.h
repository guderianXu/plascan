#pragma once

#include "ModelImageQualityTypes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace xjw::qc
{

ModelViewQuality evaluateModelViewMasks(const cv::Mat &referenceMask,
                                        const cv::Mat &renderedMask);

void evaluateModelViewAppearance(const cv::Mat &sourceBgr,
                                 const cv::Mat &renderedBgr,
                                 const cv::Mat &validMask,
                                 ModelViewQuality *quality);

void evaluateModelViewStructure(const cv::Mat &sourceBgr,
                                const cv::Mat &renderedBgr,
                                const cv::Mat &validMask,
                                ModelViewQuality *quality);

cv::Mat buildDinoForegroundMask(const cv::Mat &sourceBgr);

} // namespace xjw::qc

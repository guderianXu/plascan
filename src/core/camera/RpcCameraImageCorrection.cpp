#include "RpcCameraModel.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace xjw
{
    namespace
    {

        bool finite(double value)
        {
            return std::isfinite(value);
        }

    } // namespace

    bool RpcCameraModel::setImageCorrection(const ImageCorrection& correction, std::string* errorMessage)
    {
        const std::array<double, 6> values{{correction.sampleOffsetPixels,
                                            correction.sampleSamplePixels,
                                            correction.sampleLinePixels,
                                            correction.lineOffsetPixels,
                                            correction.lineSamplePixels,
                                            correction.lineLinePixels}};
        if (!std::all_of(values.begin(), values.end(), finite))
        {
            if (errorMessage)
            {
                *errorMessage = "RPC image-correction coefficients must be finite";
            }
            return false;
        }
        _imageCorrection = correction;
        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
    }

    bool RpcCameraModel::applyImageCorrection(const CameraImageCoordinate& uncorrected,
                                              CameraImageCoordinate* corrected) const
    {
        if (!corrected || !finite(uncorrected.sample) || !finite(uncorrected.line))
        {
            return false;
        }
        const double normalized_sample = (uncorrected.sample - _parameters.sampleOffset) / _parameters.sampleScale;
        const double normalized_line = (uncorrected.line - _parameters.lineOffset) / _parameters.lineScale;
        corrected->sample = uncorrected.sample + _imageCorrection.sampleOffsetPixels +
                            _imageCorrection.sampleSamplePixels * normalized_sample +
                            _imageCorrection.sampleLinePixels * normalized_line;
        corrected->line = uncorrected.line + _imageCorrection.lineOffsetPixels +
                          _imageCorrection.lineSamplePixels * normalized_sample +
                          _imageCorrection.lineLinePixels * normalized_line;
        return finite(corrected->sample) && finite(corrected->line);
    }

} // namespace xjw

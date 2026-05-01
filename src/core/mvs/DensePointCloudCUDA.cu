// =============================================================================
// 文件: DensePointCloudCUDA.cu
// 模块: MVS - GPU 稠密点云生成
// =============================================================================

#include "DensePointCloudCUDA.h"
#include "DenseCloudBuilder.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>

namespace xjw {
namespace mvs {

// =============================================================================
// Device Kernel
// =============================================================================

struct CamParams 
{
    float fx, cx, fy, cy;
    float R_cw[9], T[3]; // R_cw * X_world + T = X_cam
};

/// 反投影 kernel
/// valid[i] = 1 表示该像素输出有效点
/// 输出：outXYZ[i*3], outRGB[i*3] (可为空时灰度)
__global__ void kernelUnproject(
    const float   *depth,    // H*W float
    const uint8_t *mask,     // H*W uint8, can be nullptr (全有效)
    const uint8_t *color,    // H*W*ch uint8, can be nullptr
    int W, int H, int ch,    // ch=0/1/3
    float minDepth, float maxDepth,
    CamParams cam,
    float   *outXYZ,         // H*W*3 float（稀疏填充，由 valid 控制）
    uint8_t *outRGB,         // H*W*3 uint8
    int32_t *valid)          // H*W int32
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    int idx = v * W + u;
    valid[idx] = 0;

    if (mask != nullptr && mask[idx] == 0) return;

    float d = depth[idx];
    if (d < minDepth || d > maxDepth) return;

    // 反归一化像素 → 相机坐标
    float xn = (u - cam.cx) / cam.fx;
    float yn = (v - cam.cy) / cam.fy;
    // d 是沿 z 轴的深度（COLMAP 约定）
    float Xc = xn * d, Yc = yn * d, Zc = d;

    // 相机坐标 → 世界坐标：X_world = R_cw^T * (X_cam - T)
    // R_wc = R_cw^T
    float Xt = Xc - cam.T[0], Yt = Yc - cam.T[1], Zt = Zc - cam.T[2];
    float Xw = cam.R_cw[0]*Xt + cam.R_cw[3]*Yt + cam.R_cw[6]*Zt;
    float Yw = cam.R_cw[1]*Xt + cam.R_cw[4]*Yt + cam.R_cw[7]*Zt;
    float Zw = cam.R_cw[2]*Xt + cam.R_cw[5]*Yt + cam.R_cw[8]*Zt;

    outXYZ[idx*3+0] = Xw;
    outXYZ[idx*3+1] = Yw;
    outXYZ[idx*3+2] = Zw;

    // 颜色
    if (ch == 3 && color != nullptr) {
        outRGB[idx*3+0] = color[idx*3+2]; // B→R
        outRGB[idx*3+1] = color[idx*3+1];
        outRGB[idx*3+2] = color[idx*3+0]; // R→B
    } else if (ch == 1 && color != nullptr) {
        uint8_t g = color[idx];
        outRGB[idx*3+0] = outRGB[idx*3+1] = outRGB[idx*3+2] = g;
    } else {
        outRGB[idx*3+0] = outRGB[idx*3+1] = outRGB[idx*3+2] = 128;
    }

    valid[idx] = 1;
}

// =============================================================================
std::vector<DensePoint> DensePointCloudCUDA::unprojectGPU(
    const cv::Mat                 &depth,
    const cv::Mat                 &mask,
    const PositiveDepthCameraModel &cam,
    const cv::Mat                 &colorImg,
    float                          minDepth,
    float                          maxDepth,
    std::string                   *errorMsg,
    const DenseCloudOptions       *options)
{
    // CUDA 可用性检查
    int devCnt = 0;
    cudaGetDeviceCount(&devCnt);
    if (devCnt == 0) {
        // 回退 CPU
        cv::Mat validMask = mask.empty() ? cv::Mat() : mask;
        DenseCloudOptions opt; opt.minDepth = minDepth; opt.maxDepth = maxDepth;
        return DenseCloudBuilder::unproject(depth, validMask, cam, colorImg, opt);
    }

#define GPU_CHECK(x) do { \
    cudaError_t _e=(x); \
    if(_e!=cudaSuccess){ \
        if(errorMsg) *errorMsg=std::string("CUDA: ")+cudaGetErrorString(_e); \
        goto cleanup_gpu; \
    }} while(0)

    const int W = depth.cols, H = depth.rows;
    const int N = W * H;
    const int ch = colorImg.empty() ? 0 : colorImg.channels();

    // host → device
    float   *d_depth = nullptr;
    uint8_t *d_mask  = nullptr;
    uint8_t *d_color = nullptr;
    float   *d_xyz   = nullptr;
    uint8_t *d_rgb   = nullptr;
    int32_t *d_valid = nullptr;

    std::vector<DensePoint> result;

    GPU_CHECK(cudaMalloc(&d_depth, N * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_xyz,   N * 3 * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_rgb,   N * 3 * sizeof(uint8_t)));
    GPU_CHECK(cudaMalloc(&d_valid, N * sizeof(int32_t)));

    {
        cv::Mat depthF;
        if (depth.type() == CV_32F) depthF = depth;
        else depth.convertTo(depthF, CV_32F);
        GPU_CHECK(cudaMemcpy(d_depth, depthF.ptr<float>(), N*sizeof(float), cudaMemcpyHostToDevice));
    }

    if (!mask.empty()) {
        GPU_CHECK(cudaMalloc(&d_mask, N * sizeof(uint8_t)));
        GPU_CHECK(cudaMemcpy(d_mask, mask.ptr<uint8_t>(), N*sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    if (!colorImg.empty()) {
        cv::Mat contColor;
        if (colorImg.isContinuous()) contColor = colorImg;
        else colorImg.copyTo(contColor);
        GPU_CHECK(cudaMalloc(&d_color, N * ch * sizeof(uint8_t)));
        GPU_CHECK(cudaMemcpy(d_color, contColor.data, N*ch*sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    {
        CamParams cparam;
        cparam.fx = cam.fx; cparam.cx = cam.cx;
        cparam.fy = cam.fy; cparam.cy = cam.cy;
        memcpy(cparam.R_cw, cam.R_cw, 9*sizeof(float));
        memcpy(cparam.T,    cam.T,    3*sizeof(float));

        const int bW = (options && options->cudaBlockW > 0) ? options->cudaBlockW : 16;
        const int bH = (options && options->cudaBlockH > 0) ? options->cudaBlockH : 16;
        dim3 block(bW, bH);
        dim3 grid((W + bW - 1) / bW, (H + bH - 1) / bH);
        kernelUnproject<<<grid, block>>>(
            d_depth, d_mask, d_color,
            W, H, ch, minDepth, maxDepth, cparam,
            d_xyz, d_rgb, d_valid);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
    }

    {
        std::vector<float>   hXYZ(N*3);
        std::vector<uint8_t> hRGB(N*3);
        std::vector<int32_t> hValid(N);
        GPU_CHECK(cudaMemcpy(hXYZ.data(),   d_xyz,   N*3*sizeof(float),   cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(hRGB.data(),   d_rgb,   N*3*sizeof(uint8_t), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(hValid.data(), d_valid, N*sizeof(int32_t),   cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i) {
            if (!hValid[i]) continue;
            DensePoint pt;
            pt.x = hXYZ[i*3]; pt.y = hXYZ[i*3+1]; pt.z = hXYZ[i*3+2];
            pt.r = hRGB[i*3]; pt.g = hRGB[i*3+1]; pt.b = hRGB[i*3+2];
            result.push_back(pt);
        }
    }

cleanup_gpu:
    if (d_depth) cudaFree(d_depth);
    if (d_mask)  cudaFree(d_mask);
    if (d_color) cudaFree(d_color);
    if (d_xyz)   cudaFree(d_xyz);
    if (d_rgb)   cudaFree(d_rgb);
    if (d_valid) cudaFree(d_valid);

    return result;

#undef GPU_CHECK
}

} // namespace mvs
} // namespace xjw

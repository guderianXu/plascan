// ============================================================
// 文件：tsai_projector.cpp
// 目标：placamera_tsai_projector（手工诊断程序，不由 CTest 注册）。
//
// 用法：placamera_tsai_projector <camera.tsai> <world_points.txt|->
// 点文件或标准输入每行提供 X Y Z，输入 END 结束。程序打印解析外参、
// 旋转正交性/行列式、逐点投影结果，并沿解析出的光轴构造一个自检点。
// ============================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <placamera/tsai.h>
int main(int argc, char** argv)
{
    // 两个位置参数缺一不可：相机文件和点源；程序名占 argv[0]。
    if (argc < 3)
    {
        std::cerr << "用法: tsai_projector <camera.tsai> <world_points.txt 或 '-' 表示从标准输入读取>\n";
        std::cerr << "世界点格式: 每行 X Y Z（以空格分隔）\n";
        return 1;
    }
    std::string camfile = argv[1];
    std::string ptsfile = argv[2];

    const placamera::FrameId world_frame("camera-test-world");
    const auto parsed =
        placamera::loadTsaiFramePinhole(camfile, placamera::CameraDefinitionId("camera-test-definition"), world_frame);
    if (!parsed)
    {
        std::cerr << "无法加载相机文件: " << camfile << " (" << parsed.message() << ")\n";
        return 2;
    }
    const auto bound = placamera::bindCentralCamera(parsed.value(),
                                                    {placamera::CameraInstanceId("camera-test-instance"),
                                                     placamera::ImageId("camera-test-image"),
                                                     {1, 1}}); // Tsai 文件不含影像尺寸；诊断程序不要求投影位于影像内。
    if (!bound)
    {
        std::cerr << "无法绑定相机: " << bound.message() << "\n";
        return 2;
    }
    const placamera::CentralCameraModel& camera = *bound.value();

    // PlaCamera 位姿暴露行优先 R_cw。R*R^T 和 det(R) 可快速发现转置、
    // 非正交或反射矩阵问题，而不依赖具体世界点。
    const auto& R = camera.pose().cameraToWorldRotation;
    const auto& C = camera.pose().center;
    std::cout << "解析到的外参 C = [" << C[0] << ", " << C[1] << ", " << C[2] << "]\n";
    std::cout << "解析到的 R (row-major):\n";
    for (int r = 0; r < 3; ++r)
    {
        std::cout << "  ";
        for (int c = 0; c < 3; ++c)
        {
            std::cout << R[r * 3 + c] << (c < 2 ? ", " : "\n");
        }
    }
    double max_err = 0.0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            double s = 0.0;
            for (int k = 0; k < 3; k++)
            {
                s += R[i * 3 + k] * R[j * 3 + k];
            }
            double want = (i == j) ? 1.0 : 0.0;
            max_err = std::max(max_err, std::abs(s - want));
        }
    }
    std::cout << "R*R^T 与 I 的最大偏差: " << max_err << "\n";
    double det =
        R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
    std::cout << "R 的行列式 det(R) = " << det << "\n";

    std::istream* pin = &std::cin;
    std::ifstream ifs;
    if (ptsfile != "-")
    {
        ifs.open(ptsfile);
        if (!ifs)
        {
            std::cerr << "无法打开点文件: " << ptsfile << "\n";
            return 3;
        }
        pin = &ifs;
    }

    std::string line;
    // 输出头
    std::cout << "# tsai_projector 输出: " << camfile << "\n";
    std::cout << "# 输入: X Y Z    输出: u v（像素）\n";
    while (std::getline(*pin, line) && line != "END")
    {
        // END 允许交互式 stdin 明确结束；文件模式也可省略并依赖 EOF。
        if (line.empty())
        {
            continue;
        }
        std::istringstream iss(line);
        double X, Y, Z;
        if (!(iss >> X >> Y >> Z))
        {
            continue;
        }
        const auto projected = camera.groundToImage({world_frame, {X, Y, Z}});
        if (!projected.ok())
        {
            std::cout << X << " " << Y << " " << Z << " -> " << "NaN NaN\n";
        }
        else
        {
            std::cout << X << " " << Y << " " << Z << " -> " << projected.value().image.sample << " "
                      << projected.value().image.line << "\n";
        }
    }

    // 自检：R_cw 的第三列是相机 +Z 轴在世界系中的方向；按物理前向符号
    // 从 C 前进 10 个世界单位，反向深度相机也应能成功投影。
    double d = 10.0;
    // 相机 z 轴在世界坐标：R_cw * e3。
    double cam_z_world[3] = {R[2], R[5], R[8]};
    std::cout << "相机 z 轴在世界坐标方向 (cam_z_world) = [" << cam_z_world[0] << ", " << cam_z_world[1] << ", "
              << cam_z_world[2] << "]\n";
    const double forward_sign = camera.pinholeDefinition().depthAxisFlipped() ? -1.0 : 1.0;
    double test_world[3] = {C[0] + forward_sign * cam_z_world[0] * d,
                            C[1] + forward_sign * cam_z_world[1] * d,
                            C[2] + forward_sign * cam_z_world[2] * d};
    // R 为 camera-to-world，因此相机坐标为 Xc = R^T*(Xw-C)。
    double x = test_world[0] - C[0];
    double y = test_world[1] - C[1];
    double z = test_world[2] - C[2];
    double cam_coords[3];
    cam_coords[0] = R[0] * x + R[3] * y + R[6] * z;
    cam_coords[1] = R[1] * x + R[4] * y + R[7] * z;
    cam_coords[2] = R[2] * x + R[5] * y + R[8] * z;
    std::cerr << "自检: 测试点在相机坐标系下 = [" << cam_coords[0] << ", " << cam_coords[1] << ", " << cam_coords[2]
              << "]\n";
    const auto projected_test = camera.groundToImage({world_frame, {test_world[0], test_world[1], test_world[2]}});
    if (projected_test.ok())
    {
        std::cerr << "投影像素: u=" << projected_test.value().image.sample << " v=" << projected_test.value().image.line
                  << "\n";
    }
    else
    {
        std::cerr << "投影失败（点可能在相机后方或投影不可用）。\n";
    }

    return 0;
}

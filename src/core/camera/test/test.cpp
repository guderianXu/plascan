// ============================================================
// 文件：test.cpp
// 目标：camera_test（手工诊断程序，不由 gtest_discover_tests 注册）。
//
// 用法：camera_test <camera.tsai> <world_points.txt|->
// 点文件或标准输入每行提供 X Y Z，输入 END 结束。程序打印解析外参、
// 旋转正交性/行列式、逐点投影结果，并沿解析出的光轴构造一个自检点。
// ============================================================

#include "FramePinholeCamera.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace xjw;

// 一份可复制到临时 .tsai 文件的诊断样例；main 当前始终读取用户给定路径，
// 保留该文本便于调试器中快速检查解析字段和畸变参数。
const char* default_cam_text = R"CAM(
VERSION_4
PINHOLE
fu = 120.5
fv = 120.5
cu = 6.797699999999999
cv = 6.3634999999999993
u_direction = 1 0 0
v_direction = 0 1 0
w_direction = 0 0 1
C = 5878.0357304559748 4651.4069330884004 -697.94428393487954
R = -0.62169615905413556 -0.073272676505342638 -0.77982369846977706 0.7832535785538598 -0.054627173991091627 -0.61929775031427559 0.0027780388605792572 -0.99581473514880958 0.091352590333272476
pitch = 0.012999999999999999
TSAI
k1 = 7.5670432940895918e-07
k2 = 1.9558445497448025
p1 = 4.4220651700371528e-11
p2 = -2.7052458084484302e-11
k3 = 0.2125760596266677
)CAM";
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

    xjw::FramePinholeCamera cam;
    if (!cam.loadFromFile(camfile))
    {
        std::cerr << "无法加载相机文件: " << camfile << "\n";
        return 2;
    }

    // FramePinholeCamera 暴露的是行优先 R_cw。R*R^T 和 det(R) 可快速发现转置、
    // 非正交或反射矩阵问题，而不依赖具体世界点。
    auto R = cam.cameraToWorldRotation();
    auto C = cam.cameraCenter();
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
    double det = R[0]*(R[4]*R[8]-R[5]*R[7]) - R[1]*(R[3]*R[8]-R[5]*R[6]) + R[2]*(R[3]*R[7]-R[4]*R[6]);
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
        double uv[2];
        const double world[3] = {X,Y,Z};
        bool ok = cam.projectWorldPoint(world, uv);
        if (!ok)
        {
            std::cout << X << " " << Y << " " << Z << " -> " << "NaN NaN\n";
        }
        else
        {
            std::cout << X << " " << Y << " " << Z << " -> " << uv[0] << " " << uv[1] << "\n";
        }
    }

    // 自检：R_cw 的第三列是相机 +Z 轴在世界系中的方向；从 C 沿该轴
    // 前进 10 个世界单位，常规相机应得到约 [0,0,10] 的相机坐标。
    double d = 10.0;
    // 相机 z 轴在世界坐标（cam->world = R^T * e3）
    double cam_z_world[3] = { R[2], R[5], R[8] };
    std::cout << "相机 z 轴在世界坐标方向 (cam_z_world) = [" << cam_z_world[0] << ", " << cam_z_world[1] << ", " << cam_z_world[2] << "]\n";
    double test_world[3] = { C[0] + cam_z_world[0]*d, C[1] + cam_z_world[1]*d, C[2] + cam_z_world[2]*d };
    // 计算该点在相机坐标系下的坐标：Xc = R*(Xw-C)
    double x = test_world[0] - C[0];
    double y = test_world[1] - C[1];
    double z = test_world[2] - C[2];
    double cam_coords[3];
    cam_coords[0] = R[0]*x + R[1]*y + R[2]*z;
    cam_coords[1] = R[3]*x + R[4]*y + R[5]*z;
    cam_coords[2] = R[6]*x + R[7]*y + R[8]*z;
    std::cerr << "自检: 测试点在相机坐标系下 = [" << cam_coords[0] << ", " << cam_coords[1] << ", " << cam_coords[2] << "]\n";
    double uvtest[2];
    bool oktest = cam.projectWorldPoint(test_world, uvtest);
    if (oktest)
    {
        std::cerr << "投影像素: u=" << uvtest[0] << " v=" << uvtest[1] << "\n";
    }
    else
    {
        std::cerr << "投影失败（点可能在相机后方或投影不可用）。\n";
    }

    return 0;
}

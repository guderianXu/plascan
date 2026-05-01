# PlaScan — 离线部署镜像
# 基础镜像：CUDA 13.1 runtime + Ubuntu 22.04
# 驱动要求：宿主机 NVIDIA 驱动 >= 525（推荐 580.x）
FROM nvidia/cuda:13.1.1-runtime-ubuntu22.04

LABEL maintainer="PlaScan"
LABEL description="PlaScan planetary photogrammetry pipeline"

# ── 系统基础包（最小集，仅运行时需要）──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libdbus-1-3 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libfontconfig1 \
        libfreetype6 \
        libx11-6 \
        libx11-xcb1 \
        libxext6 \
        libxrender1 \
        libxi6 \
        libxrandr2 \
        libxfixes3 \
        libxcursor1 \
        libxtst6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── 复制 conda 环境（构建时由 build.sh 打包为 plascan-env.tar.gz）──────────
COPY plascan-env.tar.gz /tmp/
RUN mkdir -p /opt/conda/envs/plascan \
    && tar -xzf /tmp/plascan-env.tar.gz -C /opt/conda/envs/plascan \
    && rm /tmp/plascan-env.tar.gz

# ── 复制应用程序 ──────────────────────────────────────────────────────────────
COPY app/ /opt/plascan/

# ── 环境变量（基础，RPATH 覆盖在 entrypoint.sh 里完成）────────────────────
ENV CONDA_PREFIX=/opt/conda/envs/plascan
ENV PATH=/opt/conda/envs/plascan/bin:$PATH
ENV QT_QPA_PLATFORM=xcb
ENV PLASCAN_MODEL_DIR=/opt/plascan/models

# ── 工作目录 ──────────────────────────────────────────────────────────────────
WORKDIR /opt/plascan

# ── 启动脚本 ──────────────────────────────────────────────────────────────────
COPY entrypoint.sh /opt/plascan/entrypoint.sh
RUN chmod +x /opt/plascan/entrypoint.sh

ENTRYPOINT ["/opt/plascan/entrypoint.sh"]

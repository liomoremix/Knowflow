FROM python:3.12-slim

# CPU 版 torch 为默认（任何机器都能跑）；GPU 构建见 README「使用 GPU」
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_SPEC=torch==2.14.0+cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先装 torch 再装其余依赖：requirements.txt 里的 torch==2.14.0 会被 +cpu 本地版本号满足，
# 从而不会把 CPU 版覆盖成 5GB 的 CUDA 版
COPY requirements.txt .
RUN pip install --index-url ${PIP_INDEX_URL} --extra-index-url ${TORCH_INDEX} ${TORCH_SPEC} \
    && pip install --index-url ${PIP_INDEX_URL} -r requirements.txt

COPY . .

# 两个服务（import 8000 / query 8001）共用此镜像，command 由 compose 指定

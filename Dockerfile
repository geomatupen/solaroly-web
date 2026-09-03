FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DIRP_SDK_PATH=/app/third_party/utility/bin/linux/release_x64/libdirp.so \
    LD_LIBRARY_PATH=/app/third_party/utility/bin/linux/release_x64 \
    PVRT_ENABLE_DETECTRON=1 \
    PVRT_ENABLE_YOLO=1 \
    PVRT_ENABLE_THERMAL=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Detectron2 imports PyTorch while its wheel is built, so install the selected
# PyTorch runtime first. The default index produces a portable CPU image.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        --timeout 120 \
        --resume-retries 20 \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url "${TORCH_INDEX_URL}"
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-build-isolation -r requirements.txt \
    && python -c "import detectron2, lightglue, ultralytics; print('ML integrations ready')"

COPY . .

# Runtime output belongs in a volume. Running as an unprivileged user also
# keeps uploaded data and generated artifacts from being owned by root.
RUN mkdir -p /app/backend/projects \
    && useradd --create-home --uid 10001 solaroly \
    && chown -R solaroly:solaroly /app

USER solaroly

EXPOSE 8001
VOLUME ["/app/backend/projects"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/health', timeout=4).read()"

CMD ["uvicorn", "backend.pvrt.web.app:app", "--host", "0.0.0.0", "--port", "8001"]

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

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

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install 'git+https://github.com/facebookresearch/detectron2.git' \
    && pip install ultralytics==8.0.* \
    && python -c "from lightglue import LightGlue; LightGlue(features='sift')"

COPY . .

ENV DIRP_SDK_PATH="/app/third_party/utility/bin/linux/release_x64/libdirp.so"
ENV LD_LIBRARY_PATH="/app/third_party/utility/bin/linux/release_x64"
ENV PVRT_ENABLE_DETECTRON=1 \
    PVRT_ENABLE_YOLO=1 \
    PVRT_ENABLE_COLMAP=0

EXPOSE 8001

CMD ["uvicorn", "backend.pvrt.web.app:app", "--host", "0.0.0.0", "--port", "8001"]

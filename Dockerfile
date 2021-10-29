FROM nvcr.io/nvidia/tensorrt:21.06-py3
RUN chmod 1777 /tmp
RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends --no-install-suggests \
    ninja-build \
    python3.8 python3-pip python3.8-dev \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-libav \
    cmake pkg-config unzip yasm git checkinstall \
    libpng-dev \
    libatlas-base-dev gfortran \
    libtbb-dev \
    libunistring-dev \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    wget \
    yasm \
    zlib1g-dev\
    libunistring-dev libaom-dev\
    libgirepository1.0-dev gcc libcairo2-dev pkg-config gir1.2-gtk-3.0 \
    gir1.2-gst-rtsp-server-1.0 libx264-dev libx265-dev libnuma-dev
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
WORKDIR /app
COPY requirements.txt .

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY yolov5 /app/yolov5
COPY weights /app/weights
COPY config /app/config

ENV PYTHONPATH=/app/dz:/app/yolov5
COPY dz /app/dz

RUN pip install jupyterlab
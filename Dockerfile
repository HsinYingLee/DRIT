FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    wget \
    vim \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-pydot \
    python3-setuptools \
    python3-scipy \
    ffmpeg

RUN pip3 install --no-cache-dir \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado \
    Pillow \
    cython \
    opencv-python

## setup path
ENV PATH=/usr/local/cuda-9.0/bin:$PATH \
   LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
   LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
   LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/ \
   LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib

## install pytorch
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision 

## other packages
RUN pip3 install tensorboardX==1.2
RUN pip3 install tensorflow

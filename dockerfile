FROM  pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

EXPOSE 80

RUN pip install numba opencv-python torchpack 

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN  apt-get update -y 

RUN apt install apt-utils -y

RUN apt-get install git -y

RUN apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y;


RUN apt-get install libsparsehash-dev

RUN pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

RUN apt-get install xvfb -y && apt-get install libxrender1 wget -y &&pip install open3d 


## Required for mpi4py
RUN apt-get install sox ffmpeg libcairo2 libcairo2-dev -y

RUN apt-get install texlive-full -y

RUN apt install libgirepository1.0-dev -y

RUN pip3 install manimlib

# pip3 install manimce # or pip install manimce
RUN apt-get install libopenmpi-dev -y

## Required for mpi4py

RUN pip3 install mpi4py

RUN pip install vtk==8.1.2
RUN pip install mayavi


WORKDIR /app/spvnas
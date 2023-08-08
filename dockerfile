FROM  pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# https://github.com/Gymat/SurfelNeRF/tree/unofficial

ARG DEBIAN_FRONTEND=noninteractive

# ENV FORCE_CUDA="1"
ENV FORCE_CUDA=1

ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

EXPOSE 80

RUN pip install numba opencv-python torchpack 

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN python - V 

RUN  apt-get update -y 

RUN apt install apt-utils -y

RUN apt-get install git -y

RUN apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y;


RUN apt-get install libsparsehash-dev

# RUN pip uninstall torchsparse
RUN python - V 

# RUN pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

RUN apt-get install xvfb -y && apt-get install libxrender1 wget -y &&pip install open3d 

# Conda Environment 
ARG conda_env=new_conda_env


RUN  conda create -n $conda_env  python=3.8 
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH


RUN echo "source activate $conda_env" > ~/.bashrc && conda install -c pytorch pytorch=1.6.0 torchvision cudatoolkit=10.2\
    && conda install -c fvcore -c iopath -c conda-forge fvcore iopath \
    && conda install -c pytorch3d pytorch3d


## ADDITIONAL LIBRARIES
RUN apt-get install sox ffmpeg libcairo2 libcairo2-dev -y
RUN apt-get install texlive-full -y
RUN apt install libgirepository1.0-dev -y
RUN pip3 install manimlib


RUN pip install matplotlib scikit-image lpips  
RUN pip install  opencv-python 
RUN pip install  inplace_abn 
RUN pip install  open3d
RUN pip install  Pillow
RUN pip install  tqdm


COPY ./SurfelNeRF-unofficial/ /app/
WORKDIR /app/SurfelNeRF-unofficial/

RUN echo "sh build.sh" > ~/.bashrc 
WORKDIR /app/

RUN rm -rf /app/SurfelNeRF-unofficial
WORKDIR /app/SurfelNeRF-unofficial/

# ENTRYPOINT [ "source activate new_conda_env " ] 
RUN echo "source activate $conda_env" > ~/.bashrc

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "new_conda_env"]

# Use these commands to run the container: 

# docker build -t nerf_image:ready . 

# docker run --gpus all -it  --rm --volume=$PWD/SurfelNeRF-unofficial:/app/SurfelNeRF-unofficial --volume=$PWD/preprocessed_data:/app/preprocessed_data --volume=$PWD/export_Scannet_test:/app/export_Scannet_test --name nerf_container nerf_image:ready

# docker start -ai  nerf_container 

FROM  nvidia/cuda:11.7.1-devel-ubuntu22.04
## THE SCRIPT IS NOT COMPITABLE WITH  AMPERE and ADA ARCHITECTURE 
## due to CUDA 10.2 based libraries.

## The script used in the docker environment is acquired from: 
## https://github.com/Gymat/SurfelNeRF/tree/unofficial
## The developers and the authors of the paper are the creators of the main algorithm.
## This script provides docker containerization solution to have a clear env. 

ARG DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1

ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing" 
# ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Ampere" # AMPERE is not compatible


# ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
## AMPERE is excluded due to lacking of support for CUDA 10.2

## IMPORTANT: ALTERNATIVE: UPDATE THE TORCH_CUDA_ARCH_LIST according to used GPUs
## i.e. Nvidia GTX 970  ==> 5.2
## i.e. Nvidia T4 16 GB ==> 7.5
## i.e. Nvidia K80 ==> 3.7
# ARG TORCH_CUDA_ARCH_LIST= "3.7 5.2 7.5"
#T4 7.5

ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"










# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

## Install the base libs of Ubuntu Env
RUN apt-get update -y && apt install apt-utils -y && apt-get install git -y

RUN apt-get update -y &&  apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y

RUN apt-get install nano openssh-server -y
## INSTALL CONDA STARTS

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version
## INSTALL CONDA END

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:$PATH"

# Conda Environment 
ARG conda_env=pt_3d_env

RUN conda create -n $conda_env  python=3.8 
RUN echo "source activate pt_3d_env" > ~/.bashrc
SHELL ["conda", "run", "-n", "pt_3d_env", "/bin/bash", "-c"]

RUN conda install -c pytorch pytorch=1.6.0 torchvision cudatoolkit=10.2 -y\
    && conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y\
    && conda install -c bottler nvidiacub -y \
    && conda install -c pytorch3d pytorch3d -y 

# WE HAD A PROBLEM HERE !!  ==> Be Careful about TORCH_CUDA_ARCH_LIST
RUN pip install inplace_abn --ignore-installed PyYAML 
### in case of any problem try building it from 
## Alternative solutions for Ampere architecture are suggested but they did not work properly.
## Conda Pytorch xx Cuda11.x ==> No Cuda Kernel Problem 

### IMPORTANT: lpips finds a proper version and gets install without breaking anything in conteiner but it breaks dependencies in dockerfile
RUN pip install lpips==0.1.3 
### lpips breaks the pytorch dependency ==> Use a certain version ==> 0.1.3 works fine

RUN pip install matplotlib scikit-image  opencv-python  open3d Pillow tqdm  --ignore-installed PyYAML
## Ignore the PyYAML warning! & Continue
RUN pip install gdown


## Copy The repo directly we need to be sure the version we install is static
## & we met the dependency requirements 

COPY ./SurfelNeRF-unofficial/ /app/SurfelNeRF-unofficial/
WORKDIR /app/SurfelNeRF-unofficial/

RUN ls && chmod +x build.sh &&  sh build.sh  || true
## the second part of the installation is broken 
## the provided build comamnd will return an error while installing torch-batch-svd
## We will install torch-batch-svd using the installation provided on the github page
## Use true flag to skip this line when it returns and error

## Warning: Reminder of an Solved Error:
## The build command return error when installing frnn if torch - cuda dependency is broken
## lpips was the source of the problem. It broke cuda-toolkit-10.2.0 and installed 11.7
## this problem caused mismatch in dependencies and libraries.

## The error will be covered in the next command
## Alternative installation of the torch-batch-svd
RUN mkdir /globLib/

RUN  cd /globLib/  \
    && rm -rf torch-batch-svd/ \
    && git clone https://github.com/KinglittleQ/torch-batch-svd \
    && cd torch-batch-svd/ \
    && python setup.py install 

## This step is extremely slow but it is required. 
## The main installation in the script does not work.
RUN cd /globLib/ \
    && git clone --recursive https://github.com/lxxue/FRNN.git \
    && cd FRNN/external/prefix_sum \ 
    && pip install . \ 
    && cd ../../ \
    && pip install -e .

RUN pip install trimesh --ignore-installed PyYAML
RUN pip install scikit-image lpips matplotlib tensorboard

WORKDIR /app/


RUN pip install gdown
RUN pip install pypng
RUN pip install scikit-image==0.16.2
RUN apt-get update && apt-get install unzip -y
RUN chmod +x /app/SurfelNeRF-unofficial/download_data.sh  
# RUN /app/SurfelNeRF-unofficial/download_data.sh 

## download data to be used in the proces


# RUN rm -rf /app/SurfelNeRF-unofficial
WORKDIR /app/SurfelNeRF-unofficial/





# ## SSH --- Setting --- Starts
# # Update the system
# RUN apt-get update && apt-get upgrade -y

# # Install OpenSSH Server
# RUN apt-get install -y openssh-server

# # Set up configuration for SSH
# RUN mkdir /var/run/sshd
# RUN echo 'root:nerfpass' | chpasswd
# RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# # SSH login fix. Otherwise, user is kicked off after login
# RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# ENV NOTVISIBLE "in users profile"
# RUN echo "export VISIBLE=now" >> /etc/profile

# # Expose the SSH port
# EXPOSE 80


# # Run SSH
# CMD ["/usr/sbin/sshd", "-D"]
# ## SSH --- Setting --- Ends 


EXPOSE 80




# Use these commands to run the container: 

# run the commands in docker_folder directory

# docker build -t nerf_image:cu117a . 

# consider removing --rm flag to keep container active after stopping it. Otherwise the modifications will be removed. 


# sudo docker run --gpus all -it --rm  --ipc=host \
# --volume=$PWD/../preprocessed_data:/app/SurfelNeRF-unofficial/datasets/preprocessed_data \
# --volume=$PWD/../export_Scannet_test:/app/SurfelNeRF-unofficial/datasets/export_Scannet_test \
# --volume=$PWD/../pretrained__checkpoints.pth:/app/SurfelNeRF-unofficial/checkpoints/pretrained__checkpoints.pth \
# --name nerf_container nerf_image:cu117b


# --volume=$PWD/../SurfelNeRF-unofficial:/app/ \


# sudo docker run --gpus all -it --rm  --ipc=host \
# --volume=$PWD/../preprocessed_data:/app/SurfelNeRF-unofficial/datasets/preprocessed_data \
# --volume=$PWD/../preprocessed_data:/app/SurfelNeRF-unofficial/datasets/preprocessed_data \
# --volume=$PWD/../export_Scannet_test:/app/SurfelNeRF-unofficial/datasets/export_Scannet_test \
# --volume=$PWD/../pretrained__checkpoints.pth:/app/SurfelNeRF-unofficial/checkpoints/pretrained__checkpoints.pth \
# --name nerf_container nerf_image:cu117b

# docker start -ai  nerf_container 
# docker stop nerf_container

# NOTE: IF inplace_abn can not be installed using pip consider using the followings:

# git clone https://github.com/mapillary/inplace_abn
# cd inplace_abn
# python setup.py install
# cd scripts
# pip install -r requirements.txt

# sudo docker run --gpus all -it --rm -p 2200:22 --ipc=host --volume=$PWD/SurfelNeRF-unofficial:/app/ --volume=$PWD/preprocessed_data:/app/SurfelNeRF-unofficial/datasets/preprocessed_data --volume=$PWD/export_Scannet_test:/app/SurfelNeRF-unofficial/datasets/export_Scannet_test --name nerf_container nerf_image:cu117a


#  docker image tag nerf_image:without_data hoareyou/test_docker:without_data
# sudo docker push hoareyou/test_docker:without_data

# docker image tag nerf_image:with_data_2 hoareyou/test_docker:with_data_2
# sudo docker push hoareyou/test_docker:with_data_2

# TEST: sh dev_scripts/pretrained_test/scene0521_00/NoPre_0begin_test.sh



## ssh into docker environment 
# apt-get install nano openssh-server
# passwd root 

# nano /etc/ssh/sshd_config
# PermitRootLogin Yes

# service ssh start
# ssh root@localhost -p 2200


# nerf_image:with_data  

# sudo docker run --gpus all -it --rm  --ipc=host \
# -p 8800 --name nerf_container nerf_image:with_data  

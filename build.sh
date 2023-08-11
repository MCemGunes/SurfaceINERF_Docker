# USed Commands 

export TORCH_CUDA_ARCH_LIST=5.2 

conda_env=”pt_3d_env”
conda create -n $conda_env  python=3.8 -y
source activate $conda_env

export TORCH_CUDA_ARCH_LIST=5.2 

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y

 conda install -c pytorch3d pytorch3d -y


pip install matplotlib scikit-image lpips opencv-python inplace_abn open3d Pillow tqdm


pip install matplotlib scikit-image lpips  
pip install  opencv-python 
pip install  open3d
pip install  Pillow
pip install  tqdm
pip install  inplace_abn  #BE SURE TORCH CUDA_ARCH_LIST is right! 

pip install trimesh
pip install tensorboard

sh build.sh

cd ..
rm -rf torch-batch-svd/
git clone https://github.com/KinglittleQ/torch-batch-svd
cd torch-batch-svd/
python setup.py install
cd ../SurfelNeRF-unofficial

sh dev_scripts/pretrained_test/scene0521_00/NoPre_0begin_test.sh

pip install scikit-image lpips inplace_abn matplotlib

sh dev_scripts/pretrained_test/scene0521_00/NoPre_0begin_test.sh
